import sys
from setuptools import setup, find_packages

MY_VERSION = sys.version_info
if not (MY_VERSION.major == 3 and MY_VERSION.minor == 6):
    sys.exit('Only Python 3.6 is currently supported.')

PACKAGE_NAME = 'ggx'

setup(
    name=PACKAGE_NAME,
    version='0.1.0',
    packages=find_packages(include=[PACKAGE_NAME]),
    install_requires=[
        'pdfminer.six==20201018',
        'selenium==3.141.0',
        'bs4==0.0.1',
        'openpyxl==3.0.5',
        'sklearn==0.0',
        'pandas==1.1.5',
        'scipy==1.5.4',
        'spacy==2.0',
        'Pillow==8.0.1',
        'nltk==3.5',
        'dicttoxml==1.7.4',
        'greek-stemmer==0.1.1',
        'trie-search==0.3.0',
    ]
)
