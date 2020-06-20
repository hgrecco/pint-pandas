import versioneer
import sys
from setuptools.command.test import test as TestCommand

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
                                # read('CHANGES')
								])

__doc__ = long_description

install_requirements = [
    "pint>=0.10.1",
    "pandas>=0.24.0",
]

doc_requirements = ["sphinx>2.1", "sphinx_rtd_theme"]

dev_requirements = (
    doc_requirements
)

extra_requirements = {
    "test": ["pytest", "pytest-cov", "codecov", "coveralls", "nbval"],
    "docs": doc_requirements,
    "dev": dev_requirements,
}

class PintPandas(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest

        pytest.main(self.test_args)


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"test": PintPandas})

setup(
    name='pint-pandas',
    version=versioneer.get_version(),
    description='Pandas interface for Pint',
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
    install_requires=install_requirements,
    extras_require=extra_requirements,
    cmdclass=cmdclass,
)
