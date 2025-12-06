from setuptools import setup, find_packages

setup(
    name="python-gpu-statistical-analysis",
    version="1.0.0",
    description="GPU-accelerated statistical analysis with CPU fallback",
    author="GuilhermeP96",
    author_email="",
    url="https://github.com/GuilhermeP96/python-gpu-statistical-analysis",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.26.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda12x>=13.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-benchmark>=4.0.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
)
