from setuptools import setup

def readme():
      with open('README.md') as f:
            return f.read()

with open("requirements.txt", "r") as fh:
   requirements = fh.readlines()


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
      url='https://github.com/Fibertree-Project/fibertree',
      author='Joel S. Emer',
      author_email='jsemer@mit.edu',
      license='MIT',
      packages=['fibertree',
                'fibertree.core',
                'fibertree.graphics',
                'fibertree.notebook',
                'fibertree.codec',
                'fibertree.codec.formats'],
      install_requires=[req for req in requirements if req[:2] != "# "],
      include_package_data=True,
      zip_safe=False)
