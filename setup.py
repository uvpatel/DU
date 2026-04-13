from setuptools import setup, find_packages

setup(
    name="du-data-understanding",
    version="0.1.0",
    packages=find_packages(include=["du", "du.*"]),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "streamlit",
        "tensorflow",
    ],
    author="Urvil Patel",
    description="Production-ready Data Understanding library",
    python_requires=">=3.9",
    entry_points={"console_scripts": ["du=du.cli.main:main"]},
)