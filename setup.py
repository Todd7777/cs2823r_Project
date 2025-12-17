"""Setup script for CIB-Med-1."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    requirements = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "Pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
    ]

setup(
    name="cib-med",
    version="1.0.0",
    author="Todd",
    author_email="",
    description="CIB-Med-1: A Benchmark for Reliable, Monotonic, and Clinically Faithful Editing of Chest Radiographs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/CIB-Med-1",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "mypy>=1.3.0",
        ],
        "docs": [
            "sphinx>=6.2.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cib-med=cib_med.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cib_med": ["configs/*.yaml"],
    },
)
