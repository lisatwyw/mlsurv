import setuptools

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name="mlsurv",
    version="0.0.01",
    author="Lisa Tang",
    author_email="lisat@sfu.ca",
    description="Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lisatwyw/mlsurv",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'mlsurv = mlsurv.__main__:main'
        ]
    },
    install_requires=[
        'pydicom',
        'numpy',
        'torch',
        'scipy',
        'SimpleITK',
        'tqdm',
        'scikit-image',
        'fill_voids'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
)
