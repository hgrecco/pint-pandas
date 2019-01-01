#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

try:
    reload(sys).setdefaultencoding("UTF-8")
except:
    pass

try:
    from setuptools import setup, find_packages
except ImportError:
    print('Please install or upgrade setuptools or pip to continue')
    sys.exit(1)

import codecs


def read(filename):
    return codecs.open(filename, encoding='utf-8').read()


long_description = '\n\n'.join([read('README.md'),
                                # read('AUTHORS'),
                                # read('CHANGES')]
								)

__doc__ = long_description

setup(
    name='Pint-Pandas',
    version='0.1.dev0',
    description='Pandas interface for Pint module',
    long_description=long_description,
    keywords='physical quantities unit conversion science',
    author='Hernan E. Grecco',
    author_email='hernan.grecco@gmail.com',
    url='https://github.com/hgrecco/pint-pandas',
    test_suite='pintpandas.testsuite',
    packages=find_packages(),
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3.6',
    ],
    )
