import os
import glob
import scipy
import torch
import numpy as np
from utils.data_utils import get_dataloader, VID_EXTENSIONS
from utils.metric_utils import seed_everything, FeatureStats

import numpy as np
import torch

from tqdm import tqdm
from einops import rearrange

from third_party.VideoMAEv2.utils import load_videomae_model, preprocess_videomae
from third_party.i3d.utils import load_i3d_model, preprocess_i3d


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


class cd_fvd(object):
    '''
    This class is used to compute the FVD score between real and fake videos.
    model: str, name of the model to use, either 'videomae' or 'i3d'
    n_real: int, number of real videos to use for computing the FVD score, if 'full', all real videos are used
    n_fake: int, number of fake videos to use for computing the FVD score
    ckpt_path: str, path to the model checkpoint
    seed: int, random seed
    compute_feats: bool, whether to compute all features or just mean and covariance
    device: str, device to use for computing the features
    half_precision: bool, whether to use half precision for the model
    '''
    def __init__(self, model, n_real='full', n_fake=2048, ckpt_path=None, seed=42, compute_feats=False, device='cuda', half_precision=False, *args, **kwargs):
        self.model_name = model
        self.ckpt_path = ckpt_path
        self.seed = seed
        self.device = device
        self.real_stats = FeatureStats(max_items=None if n_real == 'full' else n_real, capture_mean_cov=True, capture_all=compute_feats)
        self.fake_stats = FeatureStats(max_items=n_fake, capture_mean_cov=True, capture_all=compute_feats)
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

    def compute_fvd_from_stats(self, fake_stats=None, real_stats=None):
        fake_stats = self.fake_stats if fake_stats is None else fake_stats
        real_stats = self.real_stats if real_stats is None else real_stats
        mu_fake, sigma_fake = fake_stats.get_mean_cov()
        mu_real, sigma_real = real_stats.get_mean_cov()
        m = np.square(mu_real - mu_fake).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_real, sigma_fake), disp=False) # pylint: disable=no-member
        fvd = np.real(m + np.trace(sigma_fake + sigma_real - s * 2))
        return fvd
    
    def compute_fvd(self, real_videos, fake_videos):
        '''
        This function computes the FVD score between real and fake videos in the form of numpy arrays.
        real_videos: np.array of shape (B, T, H, W, C)
        fake_videos: np.array of shape (B, T, H, W, C)
        '''
        self.real_stats = self.feature_fn(self.real_stats, self.model, real_videos, device=self.device, model_dtype=self.model_dtype)
        self.fake_stats = self.feature_fn(self.fake_stats, self.model, fake_videos, device=self.device, model_dtype=self.model_dtype)
        return self.compute_fvd_w_precomputed_stats(
            self.fake_stats, self.real_stats)

    def compute_real_stats(self, loader):
        '''
        This function computes the real features from a dataset.
        loader: torch.utils.data.DataLoader, dataloader for real videos
        '''
        seed_everything(self.seed)
        while self.real_stats.max_items is None or self.real_stats.num_items < self.real_stats.max_items:
            for batch in tqdm(loader):
                real_videos = rearrange(batch['video']*255, 'b c t h w -> b t h w c').byte().data.numpy()
                self.real_stats = self.feature_fn(self.real_stats, self.model, real_videos, device=self.device, model_dtype=self.model_dtype)
                if self.real_stats.max_items is not None and self.real_stats.num_items >= self.real_stats.max_items:
                    break
            if self.real_stats.max_items is None:
                break
    
    def compute_fake_stats(self, loader):
        '''
        This function computes the fake features from a dataset.
        loader: torch.utils.data.DataLoader, dataloader for fake videos
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

    def add_real_stats(self, real_videos):
        '''
        This function adds features of real videos to the real_stats object.
        real_videos: np.array of shape (B, T, H, W, C)
        '''
        self.real_stats = self.feature_fn(self.real_stats, self.model, real_videos, device=self.device, model_dtype=self.model_dtype)

    def add_fake_stats(self, fake_videos):
        '''
        This function adds features of fake videos to the fake_stats object.
        real_videos: np.array of shape (B, T, H, W, C)
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
    
    def save_real_stats(self, path):
        '''
        This function saves the real_stats object to a file.
        path: str, path to save the real_stats object
        '''
        self.real_stats.save(path)
    
    def load_real_stats(self, path):
        '''
        This function loads the real_stats object from a file.
        path: str, path to load the real_stats object
        '''
        self.real_stats = self.real_stats.load(path)

    def load_videos(self, video_info, resolution=128, sequence_length=16, sample_every_n_frames=1, data_type='video_folder', batch_size=16, num_workers=8):    
        '''
        This function loads videos from a file or a folder.
        video_info: str, path to the video file or folder
        resolution: int, resolution of the video
        sequence_length: int, length of the video sequence
        sample_every_n_frames: int, number of frames to skip
        data_type: str, type of the video data, either 'video_numpy', 'video_folder', 'image_folder', or 'stats_pkl'
        num_workers: int, number of workers for the dataloader
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
            self.real_stats = self.real_stats.load(video_info)

        else:
            raise ValueError('Invalid real_video path')
        return video_loader

    def offload_model_to_cpu(self):
        '''
        This function offloads the model to the CPU to release the memory.
        '''
        self.model = self.model.cpu()
        torch.cuda.empty_cache()
