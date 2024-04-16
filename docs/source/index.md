# Content-Debiased FVD

|mypy| |nbsp| |pyright| |nbsp| |typescript| |nbsp| |versions|

**cd-fvd** is a library for computing the Fréchet Video Distance (FVD) in Python.

Features include:

- Computing FVDs between two sets of videos with I3D or VideoMAE models.
- Computing and saving features of videos in different formats.
- Precomputed statistics for existing datasets.

## Get Started

```bash
# Clone the repository.
git clone https://github.com/songweige/content-debiased-fvd.git

# Install the package.
# You can also install via pip: `pip install cd-fvd`.
cd ./content-debiased-fvd
pip install -e .

# Compute the FVD on two sets of videos.
from fvd import cd_fvd
fvd = cd_fvd('videomae', ckpt_path=None)
fvd.compute_real_stats(fvd.load_videos(real_video_folder))
fvd.compute_fake_stats(fvd.load_videos(fake_video_folder))
fvd = fvd.compute_fvd_from_stats()
```

<!-- prettier-ignore-start -->

.. toctree::
   :caption: Core Usage
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./fvd.md


.. toctree::
   :caption: Utils
   :hidden:
   :maxdepth: 1
   :titlesonly:

   ./data_utils.md
   ./metric_utils.md


.. |build| image:: https://github.com/nerfstudio-project/viser/workflows/build/badge.svg
   :alt: Build status icon
   :target: https://github.com/nerfstudio-project/viser
.. |mypy| image:: https://github.com/nerfstudio-project/viser/workflows/mypy/badge.svg?branch=main
   :alt: Mypy status icon
   :target: https://github.com/nerfstudio-project/viser
.. |pyright| image:: https://github.com/nerfstudio-project/viser/workflows/pyright/badge.svg?branch=main
   :alt: Mypy status icon
   :target: https://github.com/nerfstudio-project/viser
.. |typescript| image:: https://github.com/nerfstudio-project/viser/workflows/typescript-compile/badge.svg
   :alt: TypeScript status icon
   :target: https://github.com/nerfstudio-project/viser
.. |versions| image:: https://img.shields.io/pypi/pyversions/viser
   :alt: Version icon
   :target: https://pypi.org/project/viser/
.. |nbsp| unicode:: 0xA0
   :trim:

<!-- prettier-ignore-end -->
