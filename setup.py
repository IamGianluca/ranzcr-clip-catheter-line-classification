from setuptools import setup
from setuptools import find_packages


setup(
    name="ml",
    version="0.1.0",
    description="Helper functions for RANZCR CLiP Kaggle competition",
    author="Gianluca Rossi",
    author_email="gr.gianlucarossi@gmail.com",
    license="MIT",
    # packages=find_packages(),
    install_requires=[],
    packages=["ml"],
    package_dir={"ml": "src/ml"},
)
