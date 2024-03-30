# Content-Debiased FVD

### [Project Page]() | [Paper]()

FVD is observed to favor the quality of individual frames over realistic motions. We verify this with quantitative measurement. We show that the bias can be attributed to the features extracted from a video classifier trained on the content-biased dataset in a supervised way and using features from large-scale unsupervised model can mitigate the bias.


## Quickstart
We provide a simple interface to compute FVD scores between two sets of videos that can be adapted to different scenarios. The following code snippet demonstrates how to compute FVD scores between two folders of videos.
```
from fvd import cd_fvd
fvd = cd_fvd('videomae', ckpt_path=None)
fvd.compute_real_stats(fvd.load_videos(real_video_folder))
fvd.compute_fake_stats(fvd.load_videos(fake_video_folder))
fvd = fvd.compute_fvd_from_stats()
```

## Functions for flexible usage
```
fvd = cd_fvd(model, n_real='full', n_fake=2048, ckpt_path=None, seed=0, compute_feats=False, device='cuda')
```

Load the models for computing FVDs.

**Parameters:**
* **model** - str, name of the model to use, either 'videomae' or 'i3d'
* **n_real** - int, number of real videos to use, if 'full', all real videos in the dataset are used
* **n_fake** - int, number of fake videos to use, if 'full', all real videos in the dataset are used
* **ckpt_path** - str, path to download the model checkpoint
* **seed** - int, random seed
* **compute_feats** - bool, whether to keep all features or just mean and covariance
* **device** - str, device to use for computing the features

**Note:** if `n_fake` (or `n_real`) is greater than the number of videos in the folder, then videos from the folder will be resampled to match `n_fake` (or `n_real`).

```
video_loader = fvd.load_videos(video_info, resolution=128, sequence_length=16, sample_every_n_frames=1, data_type='video_folder', num_workers=8)
```
Prepare the dataloader of different types of video sets specified by `data_type`. `video_folder` and `image_folder` are for loading videos and images from folders, `video_numpy` is for loading videos from numpy arrays with shape `[B, T, H, W, 3]`, and `stats_pkl` is for loading the statistics from a pickle file.

**Parameters:**
* **video_info** - str, path to the video file or folder
* **resolution** - int, resolution of the video
* **sequence_length** - int, length of the video sequence
* **sample_every_n_frames** - int, number of frames to skip
* **data_type** - str, type of the video data, either `video_folder`, `image_folder`, `video_numpy`, or `stats_pkl`
* **num_workers** - int, number of workers for the dataloader

```
fvd.compute_real_stats(video_loader)
```
Compute the statistics for each video set and the FVD score.

**Parameters:**
* **video_loader** - dataloader, dataloader of the video set

```
fvd.compute_fvd_from_stats(fake_stats=None, real_stats=None)
```
Compute the FVD score from the statistics of the real and fake videos.

**Parameters:**
* **fake_stats** - FeatureStats, statistics of the fake videos. Default is `None`, in which case the `fvd.fake_stats` are used.
* **real_stats** - FeatureStats, statistics of the real videos. Default is `None`, in which case the `fvd.real_stats` are used.

```
fvd.load_real_stats(filename)
```
Load the saved FVD statistics.

**Parameters:**
* **filename** - str, path to load the statistics


```
filename = 'stats_cache/real_stats.pkl'
fvd.save_real_stats(filename)
```
Save FVD statistics of the real videos for later use.

**Parameters:**
* **filename** - str, path to save the statistics


```
fvd.add_fake_stats(fake_videos)
```
This is useful to compute FVD when generating videos without saving it locally, e.g., during the validation steps. The function compute the statistics of the given videos.

**Parameters:**
* **fake_videos** - Numpy array, shape (B, T, H, W, C)

```
fvd.empty_fake_stats()
```
Reset the fake video statistics for the next validation step

```
fvd.offload_model_to_cpu()
```
Offload the model to CPU to release GPU memory

## Precomputed Datasets
We provide precomputed statistics for the following datasets: