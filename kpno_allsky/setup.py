import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kpno-allsky",
    version="0.1.0",
    author="Dylan Green",
    author_email="dylanag@uci.edu",
    description="These are some scripts based on all-sky images from the Kitt Peak National Observatory.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dylanagreen/kpno-allsky",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)