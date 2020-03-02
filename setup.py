import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="little-mallet-wrapper-YOUR-USERNAME-HERE", # Replace with your own username
    version="0.0.1",
    author="Maria Antoniak",
    author_email="maa343@cornell.edu",
    description="A small wrapper for the topic modeling functions of MALLET",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maria-antoniak/little-mallet-wrapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)