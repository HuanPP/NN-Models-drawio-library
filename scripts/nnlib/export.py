"""Thin wrapper around the draw.io desktop CLI (headless via xvfb-run).

All draw.io invocations on the machine are serialised through a file lock so
several agents can render concurrently without starting many Electron
instances at once.
"""

import fcntl
import os
import re
import shutil
import subprocess
import tempfile

from .core import STYLES

LOCK = '/tmp/nnlib-drawio.lock'


def _drawio(args, timeout=600):
    cmd = ['xvfb-run', '-a', 'drawio'] + args + ['--no-sandbox', '--disable-gpu']
    with open(LOCK, 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            # without a session bus Electron waits ~25 s for a DBus timeout
            env = dict(os.environ, DBUS_SESSION_BUS_ADDRESS='disabled:')
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return res


def export_dir(src_dir, out_dir, fmt, scale=None, border=None):
    """Export every .drawio file in src_dir to out_dir/<name>.<fmt>."""
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='nnlib-exp-')
    try:
        args = ['-x', '-f', fmt]
        if scale:
            args += ['-s', str(scale)]
        if border is not None:
            args += ['-b', str(border)]
        if fmt == 'svg':
            args += ['--svg-theme', 'light']
        args += ['-o', tmp + '/', src_dir]
        res = _drawio(args)
        for f in os.listdir(tmp):
            shutil.move(os.path.join(tmp, f), os.path.join(out_dir, f))
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_NUM = re.compile(r'-?\d+(?:\.\d+)?')


def svg_edge_paths(svg_text):
    """Map edge cell id -> list of (x, y) points of its main path."""
    out = {}
    for m in re.finditer(r'<g data-cell-id="(e\d+)">(.*?)</g>\s*</g>', svg_text, flags=re.S):
        body = m.group(2)
        pm = re.search(r'<path d="([^"]+)"', body)
        if not pm:
            continue
        nums = [float(v) for v in _NUM.findall(pm.group(1))]
        out[m.group(1)] = list(zip(nums[0::2], nums[1::2]))
    return out


def svg_offset(svg_text, d):
    """Translation between model coordinates and SVG coordinates, from one rect."""
    for n in d.nodes:
        if n.kind not in ('node', 'container') or 'shape=' in STYLES[n.style_key][1]:
            continue
        m = re.search(r'<g data-cell-id="%s">\s*<g>\s*<rect x="([-\d.]+)" y="([-\d.]+)"' % n.id, svg_text)
        if m:
            return float(m.group(1)) - n.x, float(m.group(2)) - n.y
    return 0.0, 0.0


def _dist_to_polyline(p, pts):
    best = 1e9
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        lx, hx = sorted((x0, x1))
        ly, hy = sorted((y0, y1))
        cx = min(max(p[0], lx), hx)
        cy = min(max(p[1], ly), hy)
        best = min(best, ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5)
    return best


def verify_render(svg_text, d):
    """Compare the rendered edge routes with the generator's intent."""
    problems = []
    paths = svg_edge_paths(svg_text)
    dx, dy = svg_offset(svg_text, d)
    for e in d.edges:
        rp = paths.get(e.id)
        if not rp:
            problems.append(f'edge {e.id}: not found in rendered SVG')
            continue
        rp = [(x - dx, y - dy) for x, y in rp]
        intent = e.points()
        off = max(_dist_to_polyline(p, intent) for p in rp)
        if off > (10.0 if e.jump else 1.0):
            problems.append(f'edge {e.id}: draw.io routed it differently from the waypoints '
                            f'(max deviation {off:.0f}px)')
            continue
        for wp in intent[1:-1]:
            if min(((wp[0] - x) ** 2 + (wp[1] - y) ** 2) ** 0.5 for x, y in rp) > 14:
                problems.append(f'edge {e.id}: rendered route skips waypoint ({wp[0]:.0f},{wp[1]:.0f})')
                break
    return problems
