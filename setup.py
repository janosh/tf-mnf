# Install this package via `pip install -e .`.
# Remember to activate correct target env first.

from setuptools import find_namespace_packages, setup

setup(
    name="TF-MNF",
    version="0.1.0",
    author="Janosh Riebesell",
    author_email="janosh.riebesell@gmail.com",
    packages=find_namespace_packages(include=["tf_mnf*"]),
    url="https://github.com/janosh/tf-mnf",
    description="TensorFlow 2.0 implementation of Multiplicative Normalizing Flows",
)
