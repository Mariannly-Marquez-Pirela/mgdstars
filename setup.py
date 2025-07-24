from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mgdstars",
    version="0.1.0",
    author="Mariannly Marquez",
    author_email="mariannly.marquez@yachaytech.edu.ec",
    description="A Python package for Minimal Geometrical Deformation (MGD) compact star calculations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mariannly/mgdstars",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0", "black", "flake8", "mypy"],
        "docs": ["sphinx", "sphinx-rtd-theme"],
        "viz": ["matplotlib>=3.0", "seaborn"],
    },
    entry_points={
        "console_scripts": [
            "mgdstars-cli=mgdstars.cli:main",
        ],
    },
)