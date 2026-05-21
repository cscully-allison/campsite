from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Get the version in a safe way
version = {}
with open("./campsite/version.py") as fp:
    exec(fp.read(), version)

setup(
    name='campsite',
    version=version["__version__"],
    author='Connor Scully-Allison',
    author_email='cscullyallison@sci.utah.edu',
    description='Campsite. An AI-powered analysis assistant for Jupyter notebooks.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'anywidget',
        'traitlets',
        'flask>=2.0',
        'flask-cors>=3.0',
        'langgraph>=0.0.20',
        'langchain-openai>=0.0.5',
        'langchain-anthropic>=0.3',
        'lark>=1.1',
        'pydantic>=2.0',
        'requests>=2.25',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    include_package_data=True,
    package_data={
        'campsite': ['static/*.js'],
    },
)
