from setuptools import setup, find_packages

setup(
    name='hpbu',
    description='Hierarchical Predictive Belief Update',
    author='Sebastian Kahl',
    author_email='sebkahl@gmail.com',
    url='https://www.glialfire.net',
    version='0.3',
    packages=find_packages(),
    py_modules=['hpbu'],
    include_package_data=True,
    license='http://www.apache.org/licenses/LICENSE-2.0',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'simplejson',
        'fastdtw'
    ],
)