from setuptools import setup

setup(
    name='nirmapper',
    version='0.1.0',
    packages=['nirmapper', 'nirmapper.model', 'nirmapper.renderer', 'nirmapper.tests', 'nirmapper.resources',
              'nirmapper.resources.images'],
    url='https://github.com/fechbmaster/3DNIRmapper',
    license='Apache-2.0',
    author='fechbmaster',
    author_email='bernd.fecht1@hs-augsburg.de',
    description='A 3D Mapper to map NIR images to a 3d tooth model.',
    dependency_links=['https://github.com/pywavefront/PyWavefront', 'https://github.com/pycollada/pycollada'],
    install_requires=['numpy']
)
