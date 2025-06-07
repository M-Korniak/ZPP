import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="generator",
    version="0.0.1",
    author="Zuz Kiczak, Michał Korniak, Adam Jaskuła",
    author_email="mk448287@students.mimuw.edu.pl",
    description="A package for generating spatio-temporal cellular data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MichalMaszkowski/ZPP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
