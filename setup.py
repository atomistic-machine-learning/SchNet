import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='SchNet',
    version='0.1',
    author='Kristof T. Schütt',
    email='kristof.schuett@tu-berlin.de',
    url='https://github.com/atomistic-machine-learning/SchNet',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    license='MIT',
    description='SchNet - a deep learning architecture for quantum chemistry',
    long_description=read('README.md')
)
