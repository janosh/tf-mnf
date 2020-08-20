# Install this package via `pip install .`. Don't use the `-e` option
# (development mode) as that will install to the wrong place. See
# https://github.com/conda/conda/issues/5861.

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
