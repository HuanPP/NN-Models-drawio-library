#!/usr/bin/env python3
"""Build, check and render templates for review.

Usage:
  python3 scripts/render.py --module diffusion            # all models in models/diffusion.py
  python3 scripts/render.py --module diffusion --title "DiT Block"
  python3 scripts/render.py --all

Output goes to /tmp/nnlib/review/<module>/ : <slug>.drawio, <slug>.png, <slug>.svg
and a report is printed. Exit code 1 if any model has ERRORs.
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nnlib.core import REGISTRY, build, drawio_file  # noqa: E402
from nnlib.qa import check  # noqa: E402
from nnlib.export import export_dir, verify_render  # noqa: E402
import models  # noqa: E402


def slug(title):
    s = ''.join(c.lower() if c.isalnum() else '-' for c in title)
    while '--' in s:
        s = s.replace('--', '-')
    return s.strip('-')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', help='module name under scripts/models')
    ap.add_argument('--title', action='append', help='only these titles (repeatable)')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--out', default='/tmp/nnlib/review')
    ap.add_argument('--no-render', action='store_true', help='static checks only')
    args = ap.parse_args()

    models.load(None if args.all else args.module)
    entries = [e for e in REGISTRY
               if args.all or e['module'].split('.')[-1] == args.module]
    if args.title:
        entries = [e for e in entries if e['title'] in args.title]
    if not entries:
        print('no matching models')
        return 2

    tag = 'all' if args.all else args.module
    out_dir = os.path.join(args.out, tag)
    src_dir = os.path.join(out_dir, '_src')
    shutil.rmtree(src_dir, ignore_errors=True)
    os.makedirs(src_dir, exist_ok=True)

    built = []
    n_err = 0
    for e in entries:
        try:
            d = build(e, normalize=False)
        except Exception as ex:  # noqa: BLE001
            print(f'[BUILD FAIL] {e["title"]}: {ex!r}')
            n_err += 1
            continue
        issues = check(d)          # coordinates in messages = coordinates in the model code
        d.normalize()
        s = slug(e['title'])
        with open(os.path.join(src_dir, s + '.drawio'), 'w', encoding='utf-8') as f:
            f.write(drawio_file(d.to_xml(), e['title']))
        built.append((e, d, s, issues))

    if not args.no_render and built:
        export_dir(src_dir, out_dir, 'png', scale=2, border=10)
        export_dir(src_dir, out_dir, 'svg')

    for e, d, s, issues in built:
        if not args.no_render:
            svg = os.path.join(out_dir, s + '.svg')
            if os.path.exists(svg):
                with open(svg, encoding='utf-8') as f:
                    issues += [('ERROR', p) for p in verify_render(f.read(), d)]
            else:
                issues.append(('ERROR', 'render failed: no SVG produced'))
        errs = [m for lv, m in issues if lv == 'ERROR']
        warns = [m for lv, m in issues if lv == 'WARN']
        n_err += bool(errs)
        w, h = d.size()
        status = 'FAIL' if errs else ('WARN' if warns else 'OK')
        print(f'[{status}] {e["title"]}  ({w}x{h})  -> {os.path.join(out_dir, s)}.png')
        for m in errs:
            print(f'    ERROR {m}')
        for m in warns:
            print(f'    WARN  {m}')
    print(f'\n{len(built)} built, {n_err} with errors')
    return 1 if n_err else 0


if __name__ == '__main__':
    sys.exit(main())
