from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name="c_utils",
    ext_modules=cythonize(
        "c_utils.pyx", compiler_directives={"language_level": "3"}, annotate=True
    ),
    include_dirs=[numpy.get_include()],
)