from setuptools import setup, find_packages

setup(
    name="music-ml",
    version="0.1.0",
    description="Music machine learning project",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "python-dotenv",
        "rich",
        "panns_inference",
    ],
)
