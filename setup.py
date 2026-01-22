"""
Setup configuration for pySEAFOM package.
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name='pySEAFOM',
    version='0.1.8',
    author='SEAFOM Fiber Optic Monitoring Group',
    author_email='peyman.moradi@bakerhughes.com',  # TODO: Update this
    description='SEAFOM DAS Performance Testing Suite',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SEAFOM-Fiber-Optic-Monitoring-Group/pySEAFOM',
    project_urls={
        'Bug Tracker': 'https://github.com/SEAFOM-Fiber-Optic-Monitoring-Group/pySEAFOM/issues',
        'Documentation': 'https://github.com/SEAFOM-Fiber-Optic-Monitoring-Group/pySEAFOM',
        'Source Code': 'https://github.com/SEAFOM-Fiber-Optic-Monitoring-Group/pySEAFOM',
    },
    packages=find_packages(where='source'),
    package_dir={'': 'source'},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=3.0',
            'black>=22.0',
            'flake8>=4.0',
            'jupyter>=1.0',
        ],
    },
    keywords='DAS, distributed acoustic sensing, self-noise, fiber optic, seismic',
    include_package_data=True,
)
