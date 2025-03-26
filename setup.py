from setuptools import setup, find_packages

setup(
    name="fluon",
    version="0.1.0",
    description="A physics library for fluid simulation using PyTorch",
    author="Michael Adewole",
    packages=find_packages(),
    install_requires=["torch"],
)
