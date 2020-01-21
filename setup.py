from setuptools import setup

def readme():
      with open('README.md') as f:
            return f.read()

setup(name='fiber-tree',
      version='0.1',
      description='Fiber-tree style tensor simulator',
      long_description=readme(),
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.0',
        'Topic :: Scientific/Engineering',
      ],
      keywords='tensors',
      url='https://github.mit.edu/symphony/fiber-tree',
      author='Joel S. Emer',
      author_email='jsemer@mit.edu',
      license='MIT',
      packages=['fibertree'],
      install_requires=['opencv-python'],
      include_package_data=True,
      zip_safe=False)
