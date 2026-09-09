"""Setup script for the snn_interpreter package.

This project trains spiking neural networks (SNNs) on MNIST using
snnTorch and PyTorch. The distribution ships an importable
``snn_interpreter`` package plus a top-level ``main`` entry point.
"""

from pathlib import Path

from setuptools import find_packages, setup


# Read the long description from a README if one exists, otherwise
# fall back to this module's docstring.
def _read_long_description() -> str:
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return __doc__ or ""


setup(
    name="snn-interpreter",
    version="0.1.0",
    description=(
        "Spiking neural network trainer/experiments for MNIST built "
        "with snnTorch and PyTorch."
    ),
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    license="MIT",
    python_requires=">=3.8",
    packages=find_packages(exclude=("tests", "tests.*")),
    py_modules=["main"],
    install_requires=[
        "torch>=1.13",
        "torchvision>=0.14",
        "snntorch>=0.8",
        "matplotlib>=3.5",
        "Pillow>=9.0",
        "numpy>=1.21",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.0.280",
        ],
    },
    entry_points={
        "console_scripts": [
            "snn-interpreter=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
