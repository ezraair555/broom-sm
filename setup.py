from setuptools import setup, find_packages

setup(
    name="broom-sm",
    version="0.1.0",
    description="Tidy statistical modeling helpers blending infer, DABEST, and Pingouin",
    author="EzraAir",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0",
        "numpy>=2.0",
        "scipy>=1.9",
        "matplotlib>=3.7",
        "pingouin>=0.5",
        "dabest>=2025.10",
        "statsmodels>=0.14",
    ],
)
