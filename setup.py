from setuptools import setup, find_packages

setup(
    name="neuro-quest",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "faiss-cpu",
        "sentence-transformers",
        "numpy",
        "pytest"
    ],
) 