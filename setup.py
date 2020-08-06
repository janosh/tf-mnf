from setuptools import find_namespace_packages, setup

setup(
    name="MNF-BNN",
    version="0.1.0",
    author="Janosh Riebesell",
    author_email="janosh.riebesell@gmail.com",
    packages=find_namespace_packages(include=["mnf_bnn*"]),
    url="https://github.com/janosh/mnf-bnn",
    description="Multiplicative Normalizing Flow Implementation in TensorFlow 2.0.",
)
