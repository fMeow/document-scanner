import re
import setuptools


def find_version(fname):
    """Attempts to find the version number in the file names fname.
    Raises RuntimeError if not found.
    """
    version = ''
    with open(fname, 'r') as fp:
        reg = re.compile(r'__version__ = [\'"]([^\'"]*)[\'"]')
        for line in fp:
            m = reg.match(line)
            if m:
                version = m.group(1)
                break
    if not version:
        raise RuntimeError('Cannot find version information')
    return version


__version__ = find_version('doc_scanner/__init__.py')

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="doc_scanner",
    version=__version__,
    author="Guoli Lyu",
    author_email="guoli-lyu@outlook.com",
    description="A document scanner based on openCV3 and scikit-image",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Guoli-Lyu/document-scanner",
    packages=setuptools.find_packages(),
    classifiers=(
        'Development Status :: 4 - Beta',
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ),
    test_suite='tests',
    project_urls={
        'Bug Reports': 'https://github.com/Guoli-Lyu/document-scanner/issues',
    },
    install_requires=[
        'numpy',
        'scikit-image',
        'opencv-python',
        'pandas',
    ],
)
