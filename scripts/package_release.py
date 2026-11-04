#!/usr/bin/env python3
"""Package the published library into GitHub release assets.

Outputs (dist/, ignored by git):
  nn_model_library.xml                    all templates, open directly in draw.io
  nn_components.xml                       building blocks
  NN-Models-drawio-library-<ver>.zip      libraries + per-category libraries + .drawio files
  NN-Models-drawio-library-<ver>-previews.zip   SVG and PNG previews

Usage: python3 scripts/package_release.py v1.0.0
"""

import os
import shutil
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, 'dist')
NAME = 'NN-Models-drawio-library'


def add_tree(zf, rel_dir, prefix):
    base = os.path.join(ROOT, rel_dir)
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isfile(path):
            zf.write(path, f'{prefix}/{rel_dir}/{name}')


def main():
    if len(sys.argv) != 2 or not sys.argv[1].startswith('v'):
        sys.exit('usage: package_release.py vX.Y.Z')
    ver = sys.argv[1]
    prefix = f'{NAME}-{ver}'

    shutil.rmtree(DIST, ignore_errors=True)
    os.makedirs(DIST)

    for f in ('nn_model_library.xml', 'nn_components.xml'):
        shutil.copy2(os.path.join(ROOT, f), DIST)

    main_zip = os.path.join(DIST, f'{prefix}.zip')
    with zipfile.ZipFile(main_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in ('nn_model_library.xml', 'nn_components.xml', 'README.md', 'LICENSE'):
            zf.write(os.path.join(ROOT, f), f'{prefix}/{f}')
        add_tree(zf, 'libraries', prefix)
        add_tree(zf, 'exports/drawio', prefix)

    prev_zip = os.path.join(DIST, f'{prefix}-previews.zip')
    with zipfile.ZipFile(prev_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        add_tree(zf, 'exports/svg', prefix)
        add_tree(zf, 'exports/png', prefix)

    for name in sorted(os.listdir(DIST)):
        size = os.path.getsize(os.path.join(DIST, name))
        print(f'{size / 1e6:8.2f} MB  dist/{name}')


if __name__ == '__main__':
    main()
