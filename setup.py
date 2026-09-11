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
    version="0.2.0",
    description=(
        "Spiking neural network trainer/experiments for MNIST built "
        "with snnTorch and PyTorch."
    ),
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    author="w4ffl35",
    author_email="contact@capsizegames.com",
    url="https://github.com/w4ffl35/snn_interpreter",
    license="BSD-3-Clause",
    license_files=["LICENSE"],
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={"snn_interpreter": ["py.typed"]},
    py_modules=["main", "main_encodings"],
    install_requires=[
        "torch>=2.5",
        "torchvision>=0.20",
        "snntorch>=1.0",
        "matplotlib>=3.8",
        "Pillow>=10.0",
        "numpy>=1.26",
        "psutil>=5.9",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "ruff>=0.0.280",
        ],
        "web": [
            "fastapi>=0.110",
            "uvicorn[standard]>=0.27",
            "websockets>=12.0",
            "pydantic>=2.5",
        ],
        "nir": [
            "nir>=1.0",
            "nirtorch>=1.0",
        ],
        "events": [
            "tonic>=1.4",
        ],
        "onnx": [
            "onnx>=1.14",
            "onnxruntime>=1.16",
        ],
        "hub": [
            "huggingface_hub>=0.20",
        ],
        "tracking": [
            "tensorboard>=2.0",
        ],
        "tracking-wandb": [
            "wandb",
        ],
        "docs": [
            "mkdocs-material>=9.0",
        ],
        "norse": [
            "norse",
        ],
        "lava": [
            "lava-nc",
        ],
    },
    entry_points={
        "console_scripts": [
            "snn-interpreter=main:main",
            "snn-interpreter-encodings=main_encodings:main",
            "snn-verify=snn_interpreter.cli.verify:main",
            "snn-records=snn_interpreter.cli.records_cli:main",
            "snn-targets=snn_interpreter.cli.target_cli:main",
            "snn-hub=snn_interpreter.hub.cli:main",
            "snn-energy=snn_interpreter.energy.cli:main",
            "snn-benchmark=snn_interpreter.benchmark.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
