import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reqpy-M",
    version="0.3.0",
    author="Luis A. Montejo",
    author_email="luis.montejo@upr.edu",
    description="A Python module for spectral matching of earthquake records (Single, RotDnn, PSD/FAS)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LuisMontejo/REQPY",
    project_urls={
        "Bug Tracker": "https://github.com/LuisMontejo/REQPY/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Intended Audience :: Science/Research",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy>=1.6.0",
        "matplotlib",
        "numba",
    ],
    extras_require={
        "smoothing": ["pykooh"],  # Optional dependency for faster Konno-Ohmachi smoothing
    },
)