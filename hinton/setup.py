
from setuptools import setup, find_packages

setup(
    name = 'hinton',
    version = '0.0.1',
    url = 'https://github.com/sai-prasanna/suggestion-mining-semeval19.git',
    author = 'Sai Prasanna',
    author_email = 'sai.r.prasanna@gmail.com',
    description = 'Package for semeval 19 task 9 submission',
    packages = find_packages(),    
    install_requires = ['allennlp >= 0.8.1'],
)