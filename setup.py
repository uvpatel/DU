from setuptools import setup, find_packages

setup(
    name="DU",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn"
    ],
    author="Urvil Patel",
    description="A custom data science utility package",
    python_requires=">=3.8",
)