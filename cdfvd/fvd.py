import os
import scipy
import torch
import numpy as np
from .utils.data_utils import get_dataloader, VID_EXTENSIONS
from .utils.metric_utils import seed_everything, FeatureStats

import numpy as np
import torch
import requests

from tqdm import tqdm
from einops import rearrange

from .third_party.VideoMAEv2.utils import load_videomae_model, preprocess_videomae
from .third_party.i3d.utils import load_i3d_model, preprocess_i3d

from typing import List, Optional, Union
import numpy.typing as npt

def get_videomae_features(stats, model, videos, batchsize=16, device='cuda', model_dtype=torch.float32):
    vid_length = videos.shape[0]
    for i in range(0, videos.shape[0], batchsize):
        batch = videos[i:min(vid_length, i + batchsize)]
        input_data = preprocess_videomae(batch)  # torch.Size([B, 3, T, H, W])
        input_data = input_data.to(device=device, dtype=model_dtype)
        with torch.no_grad():
            features = model.forward_features(input_data)
            stats.append_torch(features, num_gpus=1, rank=0)
    return stats


def get_i3d_logits(stats, i3d, videos, batchsize=16, device='cuda', model_dtype=torch.float32):
    vid_length = videos.shape[0]
    for i in range(0, vid_length, batchsize):
        batch = videos[i:min(vid_length, i + batchsize)]
        input_data = preprocess_i3d(batch)
        input_data = input_data.to(device=device, dtype=model_dtype)
        with torch.no_grad():
            features = i3d(input_data)
            stats.append_torch(features, num_gpus=1, rank=0)
    return stats


class cdfvd(object):
    '''This class loads a pretrained model (I3D or VideoMAE) and contains functions to compute the FVD score between real and fake videos.

    Args:
        model: Name of the model to use, either `videomae` or `i3d`.
        n_real: Number of real videos to use for computing the FVD score, if `'full'`, all the videos in the dataset will be used.
        n_fake: Number of fake videos to use for computing the FVD score, if `'full'`, all the videos in the dataset will be used.
        ckpt_path: Path to save the model checkpoint.
        seed: Random seed.
        compute_feats: Whether to compute all features or just mean and covariance.
        device: Device to use for computing the features.
        half_precision: Whether to use half precision for the model.
    '''
    def __init__(self, model: str = 'i3d', n_real: str = 'full', n_fake: int = 2048, ckpt_path: Optional[str] = None,
                 seed: int = 42, compute_feats: bool = False, device: str = 'cuda', half_precision: bool = False,
                 *args, **kwargs):
        self.model_name = model
        self.ckpt_path = ckpt_path
        self.seed = seed
        self.device = device
        self.n_real = n_real
        self.n_fake = n_fake
        self.real_stats = FeatureStats(max_items=None if n_real == 'full' else n_real, capture_mean_cov=True, capture_all=compute_feats)
        self.fake_stats = FeatureStats(max_items=None if n_fake == 'full' else n_fake, capture_mean_cov=True, capture_all=compute_feats)
        self.model_dtype = (
            torch.float16 if half_precision else torch.float32
        )
        assert self.model_name in ['videomae', 'i3d']
        print('Loading %s model ...' % self.model_name)
        if self.model_name == 'videomae':
            self.model = load_videomae_model(torch.device(device), ckpt_path).eval().to(dtype=self.model_dtype)
            self.feature_fn = get_videomae_features
        else:
            self.model = load_i3d_model(torch.device(device), ckpt_path).eval().to(dtype=self.model_dtype)
            self.feature_fn = get_i3d_logits

    def compute_fvd_from_stats(self, fake_stats: Optional[FeatureStats] = None, real_stats: Optional[FeatureStats] = None) -> float:
        '''This function computes the FVD score between real and fake videos using precomputed features.
        If the stats are not provided, it uses the stats stored in the object.
        
        Args:
            fake_stats: `FeatureStats` object containing the features of the fake videos.
            real_stats: `FeatureStats` object containing the features of the real videos.
        
        Returns:
            FVD score between the real and fake videos.
        '''
        fake_stats = self.fake_stats if fake_stats is None else fake_stats
        real_stats = self.real_stats if real_stats is None else real_stats
        mu_fake, sigma_fake = fake_stats.get_mean_cov()
        mu_real, sigma_real = real_stats.get_mean_cov()
        m = np.square(mu_real - mu_fake).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_real, sigma_fake), disp=False)
        return np.real(m + np.trace(sigma_fake + sigma_real - s * 2))
    
    def compute_fvd(self, real_videos: npt.NDArray[np.uint8], fake_videos: npt.NDArray[np.uint8]) -> float:
        '''
        This function computes the FVD score between real and fake videos in the form of numpy arrays.

        Args:
            real_videos: A numpy array of videos with shape `(B, T, H, W, C)`, values in the range `[0, 255]`
            fake_videos: A numpy array of videos with shape `(B, T, H, W, C)`, values in the range `[0, 255]`
        
        Returns:
            FVD score between the real and fake videos.
        '''
        self.real_stats = self.feature_fn(self.real_stats, self.model, real_videos, device=self.device, model_dtype=self.model_dtype)
        self.fake_stats = self.feature_fn(self.fake_stats, self.model, fake_videos, device=self.device, model_dtype=self.model_dtype)
        return self.compute_fvd_from_stats(
            self.fake_stats, self.real_stats)

    def compute_real_stats(self, loader: Union[torch.utils.data.DataLoader, List, None] = None) -> FeatureStats:
        '''
        This function computes the real features from a dataset.

        Args:
            loader: real videos, either in the type of dataloader or list of numpy arrays.

        Returns:
            FeatureStats object containing the features of the real videos.
        '''
        seed_everything(self.seed)
        if loader is None:
            assert self.real_stats.max_items is not None
            return

        while self.real_stats.max_items is None or self.real_stats.num_items < self.real_stats.max_items:
            for batch in tqdm(loader):
                real_videos = rearrange(batch['video']*255, 'b c t h w -> b t h w c').byte().data.numpy()
                self.real_stats = self.feature_fn(self.real_stats, self.model, real_videos, device=self.device, model_dtype=self.model_dtype)
                if self.real_stats.max_items is not None and self.real_stats.num_items >= self.real_stats.max_items:
                    break
            if self.real_stats.max_items is None:
                break

        return self.real_stats
    
    def compute_fake_stats(self, loader: Union[torch.utils.data.DataLoader, List, None] = None) -> FeatureStats:
        '''
        This function computes the fake features from a dataset.
        
        Args:
            loader: fake videos, either in the type of dataloader or list of numpy arrays.
        
        Returns:
            FeatureStats object containing the features of the fake videos.
        '''
        seed_everything(self.seed)
        while self.fake_stats.max_items is None or self.fake_stats.num_items < self.fake_stats.max_items:
            for batch in tqdm(loader):
                fake_videos = rearrange(batch['video']*255, 'b c t h w -> b t h w c').byte().data.numpy()
                self.fake_stats = self.feature_fn(self.fake_stats, self.model, fake_videos, device=self.device, model_dtype=self.model_dtype)
                if self.fake_stats.max_items is not None and self.fake_stats.num_items >= self.fake_stats.max_items:
                    break
            if self.fake_stats.max_items is None:
                break

        return self.fake_stats


    def add_real_stats(self, real_videos: npt.NDArray[np.uint8]):
        '''
        This function adds features of real videos to the real_stats object.

        Args:
            real_videos: A numpy array of videos with shape `(B, T, H, W, C)`, values in the range `[0, 255]`.
        '''
        self.real_stats = self.feature_fn(self.real_stats, self.model, real_videos, device=self.device, model_dtype=self.model_dtype)

    def add_fake_stats(self, fake_videos: npt.NDArray[np.uint8]):
        '''
        This function adds features of fake videos to the fake_stats object.
        
        Args:
            fake_videos: A numpy array of videos with shape `(B, T, H, W, C)`, values in the range `[0, 255]`.
        '''
        self.fake_stats = self.feature_fn(self.fake_stats, self.model, fake_videos, device=self.device, model_dtype=self.model_dtype)
    
    def empty_real_stats(self):
        '''
        This function empties the real_stats object.
        '''
        self.real_stats = FeatureStats(max_items=self.real_stats.max_items, capture_mean_cov=True)

    def empty_fake_stats(self):
        '''
        This function empties the real_stats object.
        '''
        self.fake_stats = FeatureStats(max_items=self.fake_stats.max_items, capture_mean_cov=True)
    
    def save_real_stats(self, path: str):
        '''
        This function saves the real_stats object to a file.

        Args:
            path: Path to save the real_stats object.
        '''
        self.real_stats.save(path)
        print('Real stats saved to %s' % path)
    
    def load_real_stats(self, path: str):
        '''
        This function loads the real_stats object from a file.

        Args:
            path: Path to load the real_stats object.
        '''
        self.real_stats = self.real_stats.load(path)
        print('Real stats loaded from %s' % path)

    def load_videos(self, video_info: str, resolution: int = 256, sequence_length: int = 16, sample_every_n_frames: int = 1,
                    data_type: str = 'video_numpy', num_workers: int = 4, batch_size: int = 16) -> Union[torch.utils.data.DataLoader, List, None]:
        '''
        This function loads videos from a way specified by `data_type`. 
        `video_numpy` loads videos from a file containing a numpy array with the shape `(B, T, H, W, C)`.
        `video_folder` loads videos from a folder containing video files.
        `image_folder` loads videos from a folder containing image files.
        `stats_pkl` indicates that `video_info` of a dataset name for pre-computed features. Currently supports `ucf101`, `kinetics`, `sky`, `ffs`, and `taichi`.

        Args:
            video_info: Path to the video file or folder.
            resolution: Resolution of the video.
            sequence_length: Length of the video sequence.
            sample_every_n_frames: Number of frames to skip.
            data_type: Type of the video data, either `video_numpy`, `video_folder`, `image_folder`, or `stats_pkl`.
            num_workers: Number of workers for the dataloader.
            batch_size: Batch size for the dataloader.
        
        Returns:
            Dataloader or list of numpy arrays containing the videos.
        '''
        if data_type=='video_numpy' or video_info.endswith('.npy'):
            video_array = np.load(video_info)
            video_loader = [{'video':  rearrange(torch.from_numpy(video_array[i:i+batch_size])/255., 'b t h w c -> b c t h w')} for i in range(0, video_array.shape[0], batch_size)]
        elif data_type=='video_folder':
            print('Loading from video files ...')
            video_loader = get_dataloader(video_info, image_folder=False,
                                    resolution=resolution, sequence_length=sequence_length,
                                    sample_every_n_frames=sample_every_n_frames,
                                    batch_size=batch_size, num_workers=num_workers)
        elif data_type=='image_folder':
            print('Loading from frame files ...')
            video_loader = get_dataloader(video_info, image_folder=True,
                                    resolution=resolution, sequence_length=sequence_length,
                                    sample_every_n_frames=sample_every_n_frames,
                                    batch_size=batch_size, num_workers=num_workers)
        elif data_type=='stats_pkl':
            video_loader = None
            cache_name = '%s_%s_%s_res%d_len%d_skip%d_seed%d.pkl' % (self.model_name.lower(), video_info, self.n_real, resolution, sequence_length, sample_every_n_frames, 0)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            ckpt_path = os.path.join(current_dir, 'fvd_stats_cache', cache_name)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)

            if not os.path.exists(ckpt_path):
                # download the ckpt to the path
                ckpt_url = 'https://content-debiased-fvd.github.io/files/%s' % cache_name
                response = requests.get(ckpt_url, stream=True, allow_redirects=True)
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024

                with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                    with open(ckpt_path, "wb") as fw:
                        for data in response.iter_content(block_size):
                            progress_bar.update(len(data))
                            fw.write(data)

            self.real_stats = self.real_stats.load(ckpt_path)

        else:
            raise ValueError('Invalid real_video path')
        return video_loader

    def offload_model_to_cpu(self):
        '''
        This function offloads the model to the CPU to release the memory.
        '''
        self.model = self.model.cpu()
        torch.cuda.empty_cache()
