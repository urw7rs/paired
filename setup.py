from pathlib import Path

from setuptools import find_packages, setup


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="paired",
    version="0.0.0",
    description="Music to Dance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chanhyuk Jung",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "numpy",
        "gdown",
        "librosa",
        "soundfile",
        "matplotlib",
        "jsonargparse",
        "smplx",
        "lightning==2.1",
        "einops",
    ],
    extras_require={
        "test": ["pytest", "pytest-xdist"],
        "dev": ["black", "ruff", "bumpver"],
    },
)
