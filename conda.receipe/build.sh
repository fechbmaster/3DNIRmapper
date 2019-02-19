#!/bin/bash

$PYTHON -m pip install pycollada --no-deps -vv
$PYTHON -m pip install PyWavefront --no-deps -vv
$PYTHON setup.py install

