# setup.py
from setuptools import setup, find_packages

setup(
    name='ARA validation',
    version='0.1.0',
    author='Laurenz Roither',
    author_email='laurenz.roither@gmail.com',
    description='A Python package for high-resolution climate data analysis and validation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/laroith/ARA_analysis',
    packages=find_packages(where='scripts'),
    package_dir={'': 'scripts'},
    install_requires=[
        'numpy',
        'xarray',
        'matplotlib',
        'scipy',
        'dask'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
