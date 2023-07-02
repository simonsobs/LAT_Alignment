from setuptools import setup

setup(
    name="lat_alignment",
    version="1.0.0",
    packages=["lat_alignment"],
    install_requires=["scipy", "numpy", "matplotlib", "pyyaml"],
    entry_points={"console_scripts": ["lat_alignment=lat_alignment.alignment:main"]},
)
