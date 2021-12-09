from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import os

def readme():
      with open('README.md') as f:
            return f.read()

with open("requirements.txt", "r") as fh:
   requirements = fh.readlines()

fibertree_sources = ['fibertree/__init__.py']
fibertree_core = []

for py_core_f in os.listdir("./fibertree/core"):
      if py_core_f.endswith(".py"):
            print(py_core_f)
            fibertree_core.append('fibertree/core/'+py_core_f)


extensions=[Extension('fibertree', fibertree_sources),
            Extension('fibertree.core', fibertree_core)
           ]
#            Extension('fibertree.graphics',fibertree_graphics),
#            Extension('fibertree.notebook',fibertree_notebook),
#            Extension('fibertree.codec', fibertree_codec),
#            Extension('fibertree.codec.formats', fibertree_formats)
#           ]

setup(name='fiber-tree',
      version='0.1',
      description='Fibertree style tensor simulator',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.0',
        'Topic :: Scientific/Engineering',
      ],
      keywords='tensors',
      url='https://github.com/FPSG-UIUC/fibertree/',
      author='Joel S. Emer',
      author_email='jsemer@mit.edu',
      license='MIT',
      packages=['fibertree',
                'fibertree.core',
                'fibertree.graphics',
                'fibertree.notebook',
                'fibertree.codec',
                'fibertree.codec.formats'],
      ext_modules=cythonize(extensions),
      install_requires=[req for req in requirements if req[:2] != "# "],
      include_package_data=True,
      zip_safe=False,
)
