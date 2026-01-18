#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="nu1lm",
    version="0.1.0",
    description="Nu1lm - Microplastics Expert AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/nu1lm",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "accelerate>=0.28.0",
        "huggingface_hub>=0.22.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
        "sentence-transformers>=2.5.0",
    ],
    extras_require={
        "training": [
            "peft>=0.10.0",
            "bitsandbytes>=0.43.0",
            "datasets>=2.18.0",
            "trl>=0.8.0",
        ],
        "spectral": [
            "spc",
            "renishawWiRE",
        ],
    },
    entry_points={
        "console_scripts": [
            "nu1lm=nu1lm_microplastics:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
