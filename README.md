[![PyPI version](https://badge.fury.io/py/nirmapper.svg)](https://badge.fury.io/py/nirmapper)

<div align="center" width="200">

[![preview](https://github.com/fechbmaster/3DNIRmapper/blob/master/nirmapper/resources/images/result.png)](#readme)

</div>

# 3DNIRmapper

3DNirMapper maps multiple textures with given camera parameters on a 3d-model using a z-buffer approach. It was originaly developed to map nearinfrared pictures to a 3d tooth model and was developed in proceedings of my master thesis.

The program is able to import wavefront objects and export them to textured Collada files.

The package is on [pypi](https://pypi.org/project/nirmapper/)
or can be cloned on [github](https://github.com/fechbmaster/3DNIRmapper).

```
pip install nirmapper
```

## Usage

The program comes with a cli, developed with Click. It contains two commands:
```
Usage: nirmapper mapp [OPTIONS] NAME MODEL_SRC TEXTURE_SRC DST

Options:
  --zfactor FLOAT        The z factor defines how big the z-buffer should be
                         for the visibility analysis. If results are bad put
                         this up to 2 or 3. Be careful with values below zero
                         because zfactor is multiplied with resolution of
                         camera and must match aspect ratio of resolution.
  --thread / --unthread
  --help    
```


