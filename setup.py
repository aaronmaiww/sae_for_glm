# In ~/Downloads/sae_for_glm/setup.py
from setuptools import setup, find_packages

setup(
    name="sae_for_glm",
    version="0.1.0",
    packages=find_packages(where='.'),
    package_dir={'': '.'}
)