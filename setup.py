import setuptools

if __name__=="__main__":

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setuptools.setup(
        name='cd-fvd',
        version="0.1.0",
        author="Songwei Ge",
        author_email="songweig@cs.umd.edu",
        description="FVD calculation in PyTorch with I3D or VideoMAE models",
        long_description=long_description,
        long_description_content_type="text/markdown",
        install_requires=[
            "torch",
            "torchvision",
            "numpy>=1.14.3",
            "scipy>=1.0.1",
            "tqdm>=4.28.1",
            "pillow>=8.1",
            "requests",
            "einops"
        ],
        url="https://github.com/songweige/content-debiased-fvd",
        packages=['cdfvd'],
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
        ],
    )
