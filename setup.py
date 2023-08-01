from setuptools import setup, find_packages
from setuptools.extension import Extension
#from Cython.Build import cythonize
import os

try:
    from Cython.Build import cythonize
except ImportError:
    # create closure for deferred import
    def cythonize (*args, ** kwargs ):
        from Cython.Build import cythonize
        return cythonize(*args, ** kwargs)

def readme():
      with open('README.md') as f:
            return f.read()

with open("requirements.txt", "r") as fh:
   requirements = fh.readlines()

fibertree_sources = ['fibertree/__init__.py']
fibertree_core = []
#fibertree_core_src = []
fibertree_core_pkg = []
fibertree_model = []
fibertree_model_pkg = []

extensions = []

# False=Python, True=Cython
if True:
    for py_core_f in os.listdir("./fibertree/core"):
          if py_core_f.endswith(".py"):
                print("    Processing: ", py_core_f)
                fibertree_core.append('fibertree/core/'+py_core_f)

                py_core = os.path.splitext(py_core_f)[0]
                fibertree_core_pkg.append('fibertree.core.'+py_core)

                extensions.append(Extension('fibertree.core.'+py_core,
                                            sources=['fibertree/core/'+py_core_f]
                                           )
                                 )

    for py_model_f in os.listdir("./fibertree/model"):
          if py_model_f.endswith(".py"):
                print("    Processing: ", py_model_f)
                fibertree_model.append('fibertree/model/'+py_model_f)

                py_model = os.path.splitext(py_model_f)[0]
                fibertree_model_pkg.append('fibertree.model.'+py_model)

                extensions.append(Extension('fibertree.model.'+py_model,
                                            sources=['fibertree/model/'+py_model_f]
                                           )
                                 )

#extensions=[#Extension('fibertree', sources=fibertree_sources),
#            Extension('fibertree.core', sources=fibertree_core)
#           ]
#            Extension('fibertree.graphics',fibertree_graphics),
#            Extension('fibertree.notebook',fibertree_notebook),
#            Extension('fibertree.codec', fibertree_codec),
#            Extension('fibertree.codec.formats', fibertree_formats)
#           ]

setup(name='fibertree',
      version='0.2',
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
      ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
      packages=['fibertree',
                'fibertree.core',
                'fibertree.graphics',
                'fibertree.model',
                'fibertree.notebook',
                'fibertree.codec',
                'fibertree.codec.formats'],

      package_data={'fibertree': ["fonts/*.ttf"]},

      install_requires=[req for req in requirements if req[:2] != "# "],
      include_package_data=True,
      zip_safe=False,)
