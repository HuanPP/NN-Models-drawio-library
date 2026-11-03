#!/usr/bin/env python3
"""Assemble the published library from the legacy templates and scripts/models/*.

Outputs (repository root):
  nn_model_library.xml          all architecture templates
  nn_components.xml             small building blocks (category 'Components')
  libraries/<category>.xml      one library per category
  exports/{drawio,png,svg}/     standalone file + previews for every template

README.md is written by hand and is never touched by this script. Existing
legacy exports (exports/*/NN-<slug>.*) keep their names because the README
links to them.

Usage: python3 scripts/build.py [--skip-render] [--allow-errors]
"""

import argparse
import html
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from nnlib.core import REGISTRY, build, drawio_file, library_entry  # noqa: E402
from nnlib.qa import check  # noqa: E402
from nnlib.export import export_dir  # noqa: E402
import models  # noqa: E402

CATEGORIES = [
    ('CNN Architectures', 'cnn'),
    ('Transformers', 'transformers'),
    ('Vision Transformers', 'vision-transformers'),
    ('Large Language Models', 'llm'),
    ('Sequence Models', 'sequence'),
    ('Object Detection', 'detection'),
    ('Segmentation', 'segmentation'),
    ('Generative Models', 'generative'),
    ('Diffusion Models', 'diffusion'),
    ('Multimodal', 'multimodal'),
    ('Representation Learning', 'representation-learning'),
    ('Graph Neural Networks', 'gnn'),
    ('Reinforcement Learning', 'rl'),
    ('3D Vision & Neural Rendering', '3d-vision'),
    ('Scientific ML', 'scientific-ml'),
    ('Recommender Systems', 'recsys'),
    ('Training & Systems', 'training-systems'),
]
COMPONENTS = 'Components'
CAT_SLUG = dict(CATEGORIES)


def slug(title):
    title = re.sub(r'\+$', ' plus', title.replace('++', 'pp'))   # UNet++ / DeepLabv3+
    s = ''.join(c.lower() if c.isalnum() else '-' for c in title)
    while '--' in s:
        s = s.replace('--', '-')
    return s.strip('-')


def library_xml(entries):
    payload = [{'xml': e['xml'], 'w': e['w'], 'h': e['h'], 'title': e['title']} for e in entries]
    return '<mxlibrary>' + json.dumps(payload, ensure_ascii=False, indent=2) + '</mxlibrary>\n'


def collect(allow_errors):
    with open(os.path.join(HERE, 'legacy', 'legacy_library.json'), encoding='utf-8') as f:
        legacy = json.load(f)
    for e in legacy:
        e['legacy'] = True
    models.load()
    new, failed = [], []
    for r in REGISTRY:
        d = build(r, normalize=False)
        errs = [m for lv, m in check(d) if lv == 'ERROR']
        if errs:
            failed.append((r['title'], errs))
            if not allow_errors:
                continue
        d.normalize()
        e = library_entry(r['title'], d)
        e['category'] = r['category']
        e['legacy'] = False
        new.append(e)
    titles = [e['title'] for e in legacy + new]
    dup = {t for t in titles if titles.count(t) > 1}
    if dup:
        raise SystemExit(f'duplicate titles: {sorted(dup)}')
    unknown = {e['category'] for e in new} - set(CAT_SLUG) - {COMPONENTS}
    if unknown:
        raise SystemExit(f'unknown categories: {sorted(unknown)}')
    return legacy, new, failed


def ordered(entries):
    rank = {c: i for i, (c, _) in enumerate(CATEGORIES)}
    # legacy first inside a category (keeps the familiar templates on top), stable otherwise
    return sorted(entries, key=lambda e: (rank[e['category']], not e['legacy']))


def write_exports(entries, skip_render):
    """Write exports/{drawio,png,svg}/<stem>.* for every template.

    Legacy templates that already have a numbered export (01-lenet-5.*) keep
    it untouched; everything else is (re)written under its slug. Files that no
    longer belong to any template are removed.
    """
    exp = os.path.join(ROOT, 'exports')
    for sub in ('drawio', 'png', 'svg'):
        os.makedirs(os.path.join(exp, sub), exist_ok=True)
    numbered = {}
    for f in os.listdir(os.path.join(exp, 'drawio')):
        m = re.match(r'^\d+-(.+)\.drawio$', f)
        if m:
            numbered[m.group(1)] = f[:-len('.drawio')]
    todo, stems = [], set()
    for e in entries:
        if e['legacy'] and e['slug'] in numbered:
            e['stem'] = numbered[e['slug']]
        else:
            e['stem'] = e['slug']
            todo.append(e)
        stems.add(e['stem'])
    for sub in ('drawio', 'png', 'svg'):
        for f in os.listdir(os.path.join(exp, sub)):
            if os.path.splitext(f)[0] not in stems:
                os.remove(os.path.join(exp, sub, f))
    stale = []
    for e in todo:
        path = os.path.join(exp, 'drawio', e['stem'] + '.drawio')
        content = drawio_file(html.unescape(e['xml']), e['title'])
        old = open(path, encoding='utf-8').read() if os.path.exists(path) else None
        if old != content:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
        previews = [os.path.join(exp, s, e['stem'] + '.' + s) for s in ('png', 'svg')]
        if old != content or not all(os.path.exists(p) for p in previews):
            stale.append(e)
    print(f'{len(stale)} of {len(todo)} generated templates need rendering', flush=True)
    if skip_render:
        return
    # draw.io's folder export is slow (about 2 s per file and format); small chunks
    # keep progress visible and let an interrupted build resume where it stopped
    step = 10
    for i in range(0, len(stale), step):
        chunk = stale[i:i + step]
        tmp = tempfile.mkdtemp(prefix='nnlib-build-')
        try:
            for e in chunk:
                shutil.copy(os.path.join(exp, 'drawio', e['stem'] + '.drawio'), tmp)
            export_dir(tmp, os.path.join(exp, 'png'), 'png', scale=2, border=10)
            export_dir(tmp, os.path.join(exp, 'svg'), 'svg', border=10)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        print(f'  rendered {min(i + step, len(stale))}/{len(stale)}', flush=True)
    missing = [e['stem'] for e in todo
               if not os.path.exists(os.path.join(exp, 'png', e['stem'] + '.png'))
               or not os.path.exists(os.path.join(exp, 'svg', e['stem'] + '.svg'))]
    if missing:
        raise SystemExit(f'render failed for: {missing}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-render', action='store_true')
    ap.add_argument('--allow-errors', action='store_true')
    args = ap.parse_args()

    legacy, new, failed = collect(args.allow_errors)
    for t, errs in failed:
        print(f'[QA ERROR] {t}: {len(errs)} error(s), first: {errs[0]}')
    if failed and not args.allow_errors:
        raise SystemExit('fix QA errors first (or pass --allow-errors)')

    comps = [e for e in new if e['category'] == COMPONENTS]
    arch = ordered(legacy + [e for e in new if e['category'] != COMPONENTS])
    for e in arch + comps:
        e['slug'] = slug(e['title'])

    with open(os.path.join(ROOT, 'nn_model_library.xml'), 'w', encoding='utf-8') as f:
        f.write(library_xml(arch))
    with open(os.path.join(ROOT, 'nn_components.xml'), 'w', encoding='utf-8') as f:
        f.write(library_xml(comps))
    libdir = os.path.join(ROOT, 'libraries')
    shutil.rmtree(libdir, ignore_errors=True)
    os.makedirs(libdir)
    for c, s in CATEGORIES:
        sub = [e for e in arch if e['category'] == c]
        if sub:
            with open(os.path.join(libdir, s + '.xml'), 'w', encoding='utf-8') as f:
                f.write(library_xml(sub))

    write_exports(arch, args.skip_render)
    print(f'{len(arch)} templates ({len(legacy)} legacy, {len(arch) - len(legacy)} new), '
          f'{len(comps)} components')


if __name__ == '__main__':
    main()
