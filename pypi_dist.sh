#!/usr/bin/env bash

pandoc --from=markdown --to=rst --output=README.rst README.md

python3 setup.py build
python3 setup.py sdist
python3 setup.py bdist_wheel

twine upload dist/* --skip-existing
