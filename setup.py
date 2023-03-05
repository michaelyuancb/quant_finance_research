import setuptools

setuptools.setup(
    name="quant_finance_research",
    version="1.0",
    author="Michael Yuan",
    author_email="michael.yuan.cb@whu.edu.com",
    description="quant finance research library",
    long_description="quant finance research library, especially for machine learning methods",
    long_description_content_type="quant_finance_research",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: LGPL-2.1 license",
        "Operating System :: OS Independent",
    ],
)