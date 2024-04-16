# Content-Debiased FVD

### [Project Page](https://content-debiased-fvd.github.io/) | [Documentation](https://content-debiased-fvd.github.io/documentation) | [Paper]()

FVD is observed to favor the quality of individual frames over realistic motions. We verify this with quantitative measurement. We show that the bias can be attributed to the features extracted from a video classifier trained on the content-biased dataset in a supervised way and using features from large-scale unsupervised model can mitigate the bias. This repo contains code tookit for easily computing FVDs with different models. Please refer to out project page and paper for more detailed analysis. 


## Quickstart
We provide a simple interface to compute FVD scores between two sets of videos that can be adapted to different scenarios. The following code snippet demonstrates how to compute FVD scores between two folders of videos.
```
from fvd import cd_fvd
fvd = cd_fvd('videomae', ckpt_path=None)
fvd.compute_real_stats(fvd.load_videos(real_video_folder))
fvd.compute_fake_stats(fvd.load_videos(fake_video_folder))
fvd = fvd.compute_fvd_from_stats()
```
Please refer to the [documentation](https://content-debiased-fvd.github.io/documentation) for more detailed instructions and usages.

## Precomputed Datasets
We provide precomputed statistics for the following datasets:

| Dataset             |  Video Length  | Resolution | Reference Split          | # Reference Videos | Model | Skip Frame # | Seed |
| :-:              | :---:     | :-:        | :-:            |  :-:          | :-: |  :-:          | :-: |
| [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) | 16, 128     | 128, 256         | `train`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
| [Sky](https://github.com/weixiong-ur/mdgan) | 16, 128     | 128, 256         | `train`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
| [Taichi](https://github.com/AliaksandrSiarohin/first-order-model/blob/master/data/taichi-loading/README.md) | 16, 128     | 128, 256         | `train`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
| [Kinetics](https://github.com/cvdfoundation/kinetics-dataset) | 16, 128     | 128, 256         | `train`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
| [FFS](https://github.com/ondyari/FaceForensics) | 16, 128     | 128, 256         | `train`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
