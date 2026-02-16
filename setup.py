from setuptools import setup, find_packages

setup(
    name="pedcxr-sex",
    version="0.1.0",
    description="Pediatric CXR sex classification with optional preprocessing and thoracic cropping",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.23",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "tqdm>=4.66",
        "pillow>=10.0",
        "opencv-python>=4.8",
        "torch>=2.1",
        "torchvision>=0.16",
        "timm>=0.9",
        "matplotlib>=3.7",
    ],
    python_requires=">=3.9",
)
