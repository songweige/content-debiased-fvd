# Content-Debiased FVD for Evaluating Video Generation Models

### [Project Page](https://content-debiased-fvd.github.io/) | [Documentation](https://content-debiased-fvd.github.io/documentation) | [Paper](https://arxiv.org/abs/2404.12391)

FVD is observed to favor the quality of individual frames over realistic motions. We verify this with quantitative measurement. We show that the bias can be attributed to the features extracted from a supervised video classifier trained on the content-biased dataset and using features from large-scale unsupervised models can mitigate the bias. This repo contains code tookit for easily computing FVDs with different pre-trained models. Please refer to out project page or paper for more details about the analysis. 

***On the Content Bias in Fréchet Video Distance*** <br>
[Songwei Ge](https://songweige.github.io/), [Aniruddha Mahapatra](https://anime26398.github.io/), [Gaurav Parmar](https://gauravparmar.com/), [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/), [Jia-Bin Huang](https://jbhuang0604.github.io/)<br>
UMD, CMU, Adobe<br>
CVPR 2024

## Quickstart
We provide a simple interface to compute FVD scores between two sets of videos that can be adapted to different scenarios. You could install the library through `pip`:
```
pip install cd-fvd
```

You may choose to download some example UCF-101 videos to test the code via:
```
bash cdfvd/download_example_videos.sh
```

The following code snippet demonstrates how to compute FVD scores between a folder of videos and precomputed statistics.
```
from cdfvd import fvd
evaluator = fvd.cdfvd('videomae', ckpt_path=None)
evaluator.load_videos('ucf101', data_type='stats_pkl', resolution=128, sequence_length=16)
evaluator.compute_fake_stats(evaluator.load_videos('./example_videos/'))
score = evaluator.compute_fvd_from_stats()
```
Please refer to the [documentation](https://content-debiased-fvd.github.io/documentation) for more detailed instructions on the usage.

<b>Note:</b> By default `n_fake=2048`. If `n_fake` is greater than number of videos in `path/to/fakevideos/` folder, then same videos will be resampled `n_fake` times. If this is not the desired effect, please use custom value of `n_fake` of set `n_fake='full'` to use all videos in `path/to/fakevideos/` without repetition.

## Precomputed Datasets
We provide precomputed statistics for the following datasets. 

| Dataset             |  Video Length  | Resolution | Reference Split          | # Reference Videos | Model | Skip Frame # | Seed |
| :-:              | :---:     | :-:        | :-:            |  :-:          | :-: |  :-:          | :-: |
| [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) | 16, 128     | 128, 256         | `train+test`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
| [Sky](https://github.com/weixiong-ur/mdgan) | 16, 128     | 128, 256         | `train`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
| [Taichi](https://github.com/AliaksandrSiarohin/first-order-model/blob/master/data/taichi-loading/README.md) | 16, 128     | 128, 256         | `train`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
| [Kinetics](https://github.com/cvdfoundation/kinetics-dataset) | 16     | 128, 256         | `train`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
| [Kinetics](https://github.com/cvdfoundation/kinetics-dataset) | 128     | 128, 256         | `train`        |  2048       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |
| [FFS](https://github.com/ondyari/FaceForensics) | 16, 128     | 128, 256         | `train`        |  2048, full       |`I3D`, `VideoMAE-v2-SSv2`| 1 | 0 |

## Citation

``` bibtex
@inproceedings{ge2024content,
      title={On the Content Bias in Fréchet Video Distance},
      author={Ge, Songwei and Mahapatra, Aniruddha and Parmar, Gaurav and Zhu, Jun-Yan and Huang, Jia-Bin},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      year={2024}
}
```

## Acknowledgement

We thank Angjoo Kanazawa, Aleksander Holynski, Devi Parikh, and Yogesh Balaji for their early feedback and discussion. We thank Or Patashnik, Richard Zhang, and Hadi Alzayer for their helpful comments and paper proofreading. We thank Ivan Skorokhodov for his help with reproducing the StyleGAN-v ablation experiments. Part of the evaluation code is built on [StyleGAN-v](https://github.com/universome/stylegan-v).

## Licenses

All material in this repository is made available under the [MIT License](https://github.com/songweige/content-debiased-fvd/LICENSE). 

[metric_utils.py](https://github.com/songweige/content-debiased-fvd/utils/metric_utils.py) is adapted from the stylegan-v [metric_utils.py](https://github.com/universome/stylegan-v/blob/master/src/metrics/metric_utils.py), which was built on top of [StyleGAN2-ADA](https://github.com/nvlabs/stylegan2-ada) and restricted by the [NVidia Source Code license](https://nvlabs.github.io/stylegan2-ada-pytorch/license.html) .

VideoMAE-v2 checkpoint is [publicly available](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/MODEL_ZOO.). Please consider filling this [questionaire](https://docs.google.com/forms/d/e/1FAIpQLSd1SjKMtD8piL9uxGEUwicerxd46bs12QojQt92rzalnoI3JA/viewform) to help improve the future works.
