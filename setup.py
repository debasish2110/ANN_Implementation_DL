from os import setup


with open("README.MD", "r", encoding="utf-8") as fh:
    long_description = fh.read()
setup(
    name="src",
    version="0.0.1",
    author="debasish2110",
    author_email="debasishdash98@gmail.com",
    description="A small package for ANN Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/debasish2110/ANN_Implementation_DL.git",
    packages=["src"],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "pandas",
        "sklearn",
        "matplotlib",
        "seaborn",
        "tensorflow"
        ]
)