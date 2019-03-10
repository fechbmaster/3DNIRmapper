[![PyPI version](https://badge.fury.io/py/nirmapper.svg)](https://badge.fury.io/py/nirmapper)


# 3DNIRmapper

3DNIRMapper maps multiple textures with given camera parameters on a 3d-model using a z-buffer approach. It was originaly developed to map nearinfrared pictures on a 3d tooth model and was developed in proceedings of my master thesis.

For example with the given camera parameters it can map these two images to a 3d-modell...

<div align="center">

[![preview](https://github.com/fechbmaster/3DNIRmapper/blob/master/nirmapper/resources/images/sample1.jpg)](#readme)
[![preview](https://github.com/fechbmaster/3DNIRmapper/blob/master/nirmapper/resources/images/sample2.jpg)](#readme)

</div>

...by projecting them to the image area and resolving the overlapping parts as shown here:

<div align="center">

[![preview](https://github.com/fechbmaster/3DNIRmapper/blob/master/nirmapper/resources/images/overlap.png)](#readme)


</div>

... and finally creates the combined textured 3d-modell:

<div align="center">

[![preview](https://github.com/fechbmaster/3DNIRmapper/blob/master/nirmapper/resources/images/result.png)](#readme)

</div>

The program is able to import wavefront objects and exports them to textured Collada files.

The package is on [pypi](https://pypi.org/project/nirmapper/)
or can be cloned on [github](https://github.com/fechbmaster/3DNIRmapper).

```
pip install nirmapper
```

## CLI Usage

The program comes with a cli, developed with Click. It contains two commands. 

### map

The first command maps textures to a 3d-model:
```
Usage: nirmapper map [OPTIONS] NAME MODEL_SRC TEXTURE_SRC DST

Options:
  --zfactor FLOAT        The z factor defines how big the z-buffer should be
                         for the visibility analysis. If results are bad put
                         this up to 2 or 3. Be careful with values below zero
                         because zfactor is multiplied with resolution of
                         camera and must match aspect ratio of resolution.
  --thread / --unthread
  --help                 Show this message and exit.
```
where
* NAME is the name of the model.
* MODEL_SRC is the path to the model.
* TEXTURE_SRC ist the path of the textures to map.
* DST is the destination path.

The camera parameters must be provided for every picture to map in a XML-file in the TEXTURE_SRC that looks like this:

```
<?xml version="1.0"?>
<data>
    <focal-length>35</focal-length>
    <resolution>
        <width>1280</width>
        <height>1024</height>
    </resolution>
    <sensor>
        <width>32</width>
        <height>25.6</height>
    </sensor>
    <location>
        <x>-1.2196</x>
        <y>1.2096</y>
        <z>9.8</z>
    </location>
    <!-- <rotation type="EULER">
         <x>-8</x>
         <y>20.2</y>
         <z>85.2</z>
     </rotation>-->
    <rotation type="QUAT">
        <w>0.715</w>
        <x>-0.169</x>
        <y>0.082</y>
        <z>0.674</z>
    </rotation>
</data>
```
It can contain either euler or quaternion rotation although quaternions are highly recommended. For every texture to map there must be an .xml file with the same file name providing those parameters. An example can be found in nirmapper/resources/xmlExample/.

### example

The second cli call creates a cube, tooth or elefant example:

```
Usage: nirmapper example [OPTIONS] DST

Options:
  --type [cube|tooth|elephant]
  --help                        Show this message and exit.

```
where:
* DST is the destination path.

## Code Usage

To use the implemented modules for own developments or improvement just include them to your project:

```python
import nirmapper
```

For example if you want to use on of the examples:

```python
from nirmapper.examples import generate_cube_example

dst = '/tmp/Cube'

generate_cube_example(dst)
```

## Tests

All tests can be found in the `tests` directory. To run the tests:

```bash
# Install pywavefront in develop mode
python setup.py develop

# Install required packages for running tests
pip install -r test-requirements.txt

# Run all tests
pytest

# Optionally specific tests modules can be runned sepeartely
pytest tests/test_parser.py
```

License
-------

3DNIRMapper is [Apache-licensed](https://github.com/fechbmaster/3DNirmapper/LICENSE)

