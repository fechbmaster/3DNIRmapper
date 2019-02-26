from setuptools import setup, find_packages

setup(
    name='nirmapper',
    version='1.0.0',
    packages=find_packages(exclude='tests'),
    url='https://github.com/fechbmaster/3DNIRmapper',
    license='Apache-2.0',
    author='fechbmaster',
    author_email='bernd.fecht1@hs-augsburg.de',
    description='A 3D Mapper to map NIR images to a 3d tooth model.',
    dependency_links=['https://github.com/pywavefront/PyWavefront', 'https://github.com/pycollada/pycollada'],
    install_requires=['numpy', 'Click'],
    scripts=['bin/nirmapper']
)
