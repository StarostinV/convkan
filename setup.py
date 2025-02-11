from pathlib import Path

from setuptools import find_packages, setup


def read_file(filename: str) -> str:
    with open(Path(__file__).parent / filename, mode="r", encoding="utf-8") as f:
        return f.read()


PACKAGE_NAME = "convkan"

long_description = read_file("README.md")
long_description_content_type = "text/markdown"

python_requires = ">=3.8"

setup(
    name=PACKAGE_NAME,
    packages=find_packages(exclude=("tests",)),
    version="0.0.1.3",
    author="Vladimir Starostin",
    author_email="vladimir.starostin@uni-tuebingen.de",
    license="MIT",
    description="Convolutional KAN layer",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    python_requires=python_requires,
    install_requires=["torch>=2.3.0", "torchvision>=0.18.0", "tqdm>=4.66.2"],
)
