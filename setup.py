from setuptools import find_packages, setup

setup(
    name="ml",
    version="0.1.0",
    description="A toolkit to do well at ML challenges when you already have a full-time job ― a.k.a. little time to invest.",
    author="Gianluca Rossi",
    author_email="gr.gianlucarossi@gmail.com",
    license="MIT",
    install_requires=[],  # TODO: fix this before publishing
    packages=["ml"],
    package_dir={"ml": "src/ml"},
)
