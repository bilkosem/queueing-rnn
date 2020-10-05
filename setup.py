import setuptools
with open("README.md", "r",encoding="utf8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="queueing_rnn",
    version="1.0.4",
    author="bilkosem",
    author_email="bilkos92@gmail.com",
    description="Queueing Recurrent Neural Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bilkosem/queueing_rnn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)