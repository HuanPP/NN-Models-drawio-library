"""Static layout checks for a generated Diagram.

check(d) returns a list of (level, message) with level in {'ERROR', 'WARN'}.
A template is acceptable only when it has zero ERRORs; WARNs must be looked
at in the rendered PNG and either fixed or consciously accepted.
"""

import re

from .core import STYLES, SIDE_DIR, plain_text

TOL = 0.5
MIN_NODE_GAP = 8          # px between two solid nodes
CONTAINER_PAD = 8         # px between container border and its content
TITLE_BAND = 28           # px reserved for a container title
MIN_LAST_SEG = 10         # px, room for the arrow head
MIN_PARALLEL_GAP = 6      # px, parallel edge segments closer than this look smudged
CHAR_W = 0.50             # average glyph width in em (Times New Roman)


def _font_size(style):
    m = re.search(r'fontSize=(\d+(?:\.\d+)?)', style)
    return float(m.group(1)) if m else 12.0


def _depth(style):
    if 'shape=cube' not in style:
        return 0.0
    m = re.search(r';size=(\d+)', style)
    return float(m.group(1)) if m else 10.0


def _rect(n):
    return (n.x, n.y, n.r, n.b)


def _overlap(a, b, pad=0.0):
    return a[0] < b[2] + pad and b[0] < a[2] + pad and a[1] < b[3] + pad and b[1] < a[3] + pad


def _inside(inner, outer, pad):
    return (inner[0] >= outer[0] + pad - TOL and inner[1] >= outer[1] + pad - TOL and
            inner[2] <= outer[2] - pad + TOL and inner[3] <= outer[3] - pad + TOL)


def _gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0)
    dy = max(b[1] - a[3], a[1] - b[3], 0)
    if dx > 0 and dy > 0:
        return (dx * dx + dy * dy) ** 0.5
    return max(dx, dy)


def _label(n):
    return plain_text(n.value)[0][:24] or n.id


def _segments(points):
    segs = []
    for i in range(len(points) - 1):
        (x0, y0), (x1, y1) = points[i], points[i + 1]
        if abs(x0 - x1) < TOL and abs(y0 - y1) < TOL:
            continue
        segs.append(((x0, y0), (x1, y1), i))
    return segs


def _is_h(s):
    return abs(s[0][1] - s[1][1]) < TOL


def _seg_hits_rect(s, r, shrink=1.0):
    """True if the axis-aligned segment passes through the open interior of r."""
    (x0, y0), (x1, y1) = s[0], s[1]
    rx0, ry0, rx1, ry1 = r[0] + shrink, r[1] + shrink, r[2] - shrink, r[3] - shrink
    if rx0 >= rx1 or ry0 >= ry1:
        return False
    if _is_h(s):
        lo, hi = sorted((x0, x1))
        return ry0 < y0 < ry1 and lo < rx1 and hi > rx0
    lo, hi = sorted((y0, y1))
    return rx0 < x0 < rx1 and lo < ry1 and hi > ry0


def _seg_hugs_rect(s, r, dist=3.0):
    """True if the segment runs along (within dist px of) a border of r for > 4 px."""
    (x0, y0), (x1, y1) = s[0], s[1]
    if _is_h(s):
        lo, hi = sorted((x0, x1))
        span = min(hi, r[2]) - max(lo, r[0])
        return span > 4 and (abs(y0 - r[1]) < dist or abs(y0 - r[3]) < dist)
    lo, hi = sorted((y0, y1))
    span = min(hi, r[3]) - max(lo, r[1])
    return span > 4 and (abs(x0 - r[0]) < dist or abs(x0 - r[2]) < dist)


def _text_need(n):
    style = STYLES[n.style_key][1]
    fs = _font_size(style)
    lines = plain_text(n.value)
    width = max(len(line) for line in lines) * fs * CHAR_W
    height = len(lines) * fs * 1.2
    return width, height, _depth(style)


def check(d):
    out = []

    def err(msg):
        out.append(('ERROR', msg))

    def warn(msg):
        out.append(('WARN', msg))

    solid = [n for n in d.nodes if n.kind in ('node', 'op')]
    texts = [n for n in d.nodes if n.kind == 'text']
    conts = [n for n in d.nodes if n.kind == 'container']

    # ---------------------------------------------------------------- nodes
    for i, a in enumerate(solid):
        for b in solid[i + 1:]:
            ra, rb = _rect(a), _rect(b)
            if _overlap(ra, rb):
                err(f'nodes overlap: "{_label(a)}" and "{_label(b)}"')
            elif _gap(ra, rb) < MIN_NODE_GAP:
                err(f'nodes too close ({_gap(ra, rb):.0f}px < {MIN_NODE_GAP}): "{_label(a)}" and "{_label(b)}"')
    for t in texts:
        for n in solid + [x for x in texts if x is not t]:
            if n is t:
                continue
            if _overlap(_rect(t), _rect(n), pad=-1):
                if n.kind == 'text' and id(n) < id(t):
                    continue
                err(f'text "{_label(t)}" overlaps "{_label(n)}"')

    for n in solid:
        if n.value == '':
            continue
        need_w, need_h, depth = _text_need(n)
        avail_w = n.w - depth - (2 if n.kind == 'op' else 8)
        avail_h = n.h - depth - 2
        if n.kind == 'op':
            continue
        if need_w > avail_w * 1.12:
            err(f'label too wide for box: "{_label(n)}" needs ~{need_w:.0f}px, box gives {avail_w:.0f}px')
        elif need_w > avail_w:
            warn(f'label tight: "{_label(n)}" needs ~{need_w:.0f}px, box gives {avail_w:.0f}px')
        if need_h > avail_h + 1:
            err(f'label too tall for box: "{_label(n)}" needs ~{need_h:.0f}px, box gives {avail_h:.0f}px')
    for t in texts:
        need_w, need_h, _ = _text_need(t)
        if need_w > t.w * 1.12 or need_h > t.h + 2:
            warn(f'text box small for its content: "{_label(t)}" (~{need_w:.0f}x{need_h:.0f} in {t.w:.0f}x{t.h:.0f})')

    # ----------------------------------------------------------- containers
    def title_box(c):
        fs = _font_size(STYLES[c.style_key][1])
        tw = len(plain_text(c.value)[0]) * fs * 0.56 + 18
        if c.style_key == 'container_repeat':
            return (c.r - tw, c.b - TITLE_BAND + 4, c.r, c.b)
        return (c.x, c.y, c.x + tw, c.y + TITLE_BAND)

    for c in conts:
        rc = _rect(c)
        tb = title_box(c) if c.value else None
        for n in solid + texts:
            rn = _rect(n)
            if not _overlap(rn, rc):
                continue
            if not _inside(rn, rc, CONTAINER_PAD):
                err(f'"{_label(n)}" sticks out of / touches container "{_label(c)}" (pad {CONTAINER_PAD}px)')
            elif tb and _overlap(rn, tb):
                err(f'"{_label(n)}" collides with title of container "{_label(c)}"')
        if tb and tb[2] > c.r + TOL:
            err(f'container "{_label(c)}" narrower than its title')
    for i, a in enumerate(conts):
        for b in conts[i + 1:]:
            ra, rb = _rect(a), _rect(b)
            if not _overlap(ra, rb, pad=6):
                continue
            if _inside(ra, rb, 6) or _inside(rb, ra, 6):
                inner, outer = (a, b) if _inside(ra, rb, 6) else (b, a)
                tb = title_box(outer) if outer.value else None
                if tb and _overlap(_rect(inner), tb):
                    err(f'container "{_label(inner)}" collides with title of "{_label(outer)}"')
                continue
            err(f'containers "{_label(a)}" and "{_label(b)}" partially overlap or are < 6px apart')

    # ---------------------------------------------------------------- edges
    seg_list = []   # (edge, seg)
    for e in d.edges:
        pts = e.points()
        name = f'edge {e.id} ({_label(e.src) if e.src else "free"} -> {_label(e.dst) if e.dst else "free"})'
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if abs(x0 - x1) > TOL and abs(y0 - y1) > TOL:
                err(f'{name}: diagonal segment ({x0:.0f},{y0:.0f})->({x1:.0f},{y1:.0f}); add a waypoint')
        segs = _segments(pts)
        if not segs:
            err(f'{name}: zero length')
            continue
        for n, side, pos in ((e.src, e.exit, e.exit_pos), (e.dst, e.entry, e.entry_pos)):
            if n is not None and n.kind == 'op' and abs(pos - 0.5) > 1e-6:
                err(f'{name}: off-centre port on circular operator "{_label(n)}"')
        if e.src is not None:
            first = segs[0]
            dx, dy = first[1][0] - first[0][0], first[1][1] - first[0][1]
            ux, uy = SIDE_DIR[e.exit]
            if not (dx * ux > 0 or dy * uy > 0) or (ux and abs(dy) > TOL) or (uy and abs(dx) > TOL):
                err(f'{name}: first segment does not leave "{_label(e.src)}" through its {e.exit} side')
        if e.dst is not None:
            last = segs[-1]
            dx, dy = last[1][0] - last[0][0], last[1][1] - last[0][1]
            ux, uy = SIDE_DIR[e.entry]
            if not (dx * -ux > 0 or dy * -uy > 0) or (ux and abs(dy) > TOL) or (uy and abs(dx) > TOL):
                err(f'{name}: last segment does not enter "{_label(e.dst)}" through its {e.entry} side')
        if EDGE_HAS_HEAD(e):
            last = segs[-1]
            ln = abs(last[1][0] - last[0][0]) + abs(last[1][1] - last[0][1])
            if ln < MIN_LAST_SEG:
                err(f'{name}: last segment only {ln:.0f}px, arrow head is cramped (min {MIN_LAST_SEG})')
        for s in segs:
            for n in solid + texts:
                if _seg_hits_rect(s, _rect(n)):
                    err(f'{name}: passes through "{_label(n)}"')
                elif n is not e.src and n is not e.dst and _seg_hugs_rect(s, _rect(n)):
                    err(f'{name}: runs along the border of "{_label(n)}"')
            for c in conts:
                if _seg_hugs_rect(s, _rect(c), dist=4):
                    err(f'{name}: runs along the border of container "{_label(c)}"')
                tb = title_box(c) if c.value else None
                if tb and _seg_hits_rect(s, tb, shrink=0):
                    err(f'{name}: crosses the title of container "{_label(c)}"')
            seg_list.append((e, s))
        if e.label:
            lw = max(len(s) for s in plain_text(e.label)) * 11 * CHAR_W + 6
            mx, my = _midpoint(pts)
            lbox = (mx - lw / 2, my - 8, mx + lw / 2, my + 8)
            for n in solid + texts:
                if _overlap(lbox, _rect(n)):
                    warn(f'{name}: label "{e.label}" may overlap "{_label(n)}"')

    def related(e1, e2):
        same_src = e1.src is not None and e1.src is e2.src
        same_dst = e1.dst is not None and e1.dst is e2.dst
        return same_src or same_dst

    for i, (e1, s1) in enumerate(seg_list):
        for e2, s2 in seg_list[i + 1:]:
            if e1 is e2:
                continue
            h1, h2 = _is_h(s1), _is_h(s2)
            if h1 != h2:
                hs, vs = (s1, s2) if h1 else (s2, s1)
                y = hs[0][1]
                x = vs[0][0]
                hlo, hhi = sorted((hs[0][0], hs[1][0]))
                vlo, vhi = sorted((vs[0][1], vs[1][1]))
                inner_h = hlo + TOL < x < hhi - TOL
                inner_v = vlo + TOL < y < vhi - TOL
                on_h = hlo - TOL <= x <= hhi + TOL
                on_v = vlo - TOL <= y <= vhi + TOL
                if inner_h and inner_v:
                    if e1.jump or e2.jump:
                        warn(f'edges {e1.id} and {e2.id} cross at ({x:.0f},{y:.0f}) (line hop drawn)')
                    else:
                        err(f'edges {e1.id} and {e2.id} cross at ({x:.0f},{y:.0f})')
                elif (inner_h and on_v) or (inner_v and on_h):
                    if not related(e1, e2):
                        err(f'edges {e1.id} and {e2.id} touch at ({x:.0f},{y:.0f}) forming a false junction')
            else:
                if h1:
                    c1, c2 = s1[0][1], s2[0][1]
                    a = sorted((s1[0][0], s1[1][0]))
                    b = sorted((s2[0][0], s2[1][0]))
                else:
                    c1, c2 = s1[0][0], s2[0][0]
                    a = sorted((s1[0][1], s1[1][1]))
                    b = sorted((s2[0][1], s2[1][1]))
                ov = min(a[1], b[1]) - max(a[0], b[0])
                dist = abs(c1 - c2)
                if ov > TOL and dist < TOL and not related(e1, e2):
                    err(f'edges {e1.id} and {e2.id} overlap on the same line')
                elif ov > 2 and TOL <= dist < MIN_PARALLEL_GAP:
                    err(f'edges {e1.id} and {e2.id} run parallel only {dist:.1f}px apart')
    return out


def EDGE_HAS_HEAD(e):
    return e.style_key not in ('line', 'skip_line')


def _midpoint(pts):
    total = 0.0
    lens = []
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ln = abs(x1 - x0) + abs(y1 - y0)
        lens.append(ln)
        total += ln
    half = total / 2
    for ((x0, y0), (x1, y1)), ln in zip(zip(pts, pts[1:]), lens):
        if half <= ln and ln > 0:
            t = half / ln
            return x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        half -= ln
    return pts[-1]
