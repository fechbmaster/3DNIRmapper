from setuptools import setup, find_packages

setup(
    name='nirmapper',
    version='1.1.10',
    packages=find_packages(),
    package_data={'': ['license.txt']},
    include_package_data=True,
    url='https://github.com/fechbmaster/3DNIRmapper',
    description='Python library to map textures on a Wavefront .obj files.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='Apache-2.0',
    author='fechbmaster',
    author_email='bernd.fecht1@hs-augsburg.de',
    dependency_links=['https://github.com/pywavefront/PyWavefront', 'https://github.com/pycollada/pycollada'],
    python_requires='>2.7',
    install_requires=['numpy', 'Click', 'PyWavefront', 'pycollada'],
    scripts=['bin/nirmapper']
)
