from setuptools import setup, find_packages

setup(
    name="wwtest",
    version="1.0.1",
    author="Jonquil Z. Liao",
    author_email="zliao42@wisc.edu",
    description="Wilcoxon--Wigner Test for Matrix Homogeneity",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/JonquilLiao/wwtest",
    packages=find_packages(),
    install_requires=["numpy", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
