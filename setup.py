from setuptools import setup, find_packages

setup(
    name='nirmapper',
    version='1.1.9',
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={'': ['license.txt']},
    include_package_data=True,
    url='https://github.com/fechbmaster/3DNIRmapper',
    license='Apache-2.0',
    author='fechbmaster',
    author_email='bernd.fecht1@hs-augsburg.de',
    description='A 3D Mapper to map NIR images to a 3d tooth model.',
    dependency_links=['https://github.com/pywavefront/PyWavefront', 'https://github.com/pycollada/pycollada'],
    python_requires='>2.7',
    install_requires=['numpy', 'Click', 'PyWavefront', 'pycollada'],
    scripts=['bin/nirmapper']
)
