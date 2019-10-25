#!/bin/bash

set -e

isort ./tf_parser/*.py
yapf --recursive --in-place ./tf_parser/*.py
flake8 ./tf_parser/*.py

