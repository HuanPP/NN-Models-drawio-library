"""Core primitives for generating draw.io neural-network templates.

Every model is a function ``fn(d: Diagram) -> None`` decorated with
``@register(category, title)``. Coordinates are absolute pixels; the
diagram is translated automatically so that its bounding box starts at
(MARGIN, MARGIN).

Edges are real draw.io connectors (source/target bound to the nodes) with
fixed exit/entry ports and explicit orthogonal waypoints, so the rendered
route is fully determined by the generator and stays attached when the
user drags nodes around in draw.io.
"""

import html
import re

MARGIN = 10
FONT = 'fontFamily=Times New Roman;'

# --------------------------------------------------------------------------
# Palette (IEEE-style, draw.io standard pastel colours)
# --------------------------------------------------------------------------
PALETTE = {
    'red':    ('#F8CECC', '#B85450'),   # input
    'blue':   ('#DAE8FC', '#6C8EBF'),   # convolution / encoder
    'sky':    ('#DCEEF8', '#56A5C9'),   # deconvolution / upsampling
    'green':  ('#D5E8D4', '#82B366'),   # pooling / downsampling / decoder
    'orange': ('#FFE6CC', '#D79B00'),   # FC / MLP / FFN
    'yellow': ('#FFF2CC', '#D6B656'),   # output / positional / condition
    'purple': ('#E1D5E7', '#9673A6'),   # attention / embedding
    'gray':   ('#F5F5F5', '#999999'),   # normalisation / reshape
    'teal':   ('#D5F0EE', '#4A9C94'),   # state-space / recurrence / memory
    'pink':   ('#F9DCEB', '#B5739D'),   # gating / routing / quantisation
    'white':  ('#FFFFFF', '#666666'),   # operators, activations
    'dark':   ('#333333', '#1A1A1A'),   # masks
}

_BOX = 'rounded=1;arcSize=8;whiteSpace=wrap;html=1;fontColor=#333333;strokeWidth=1.5;' + FONT
_CUBE = ('shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.05;'
         'darkOpacity2=0.1;fontColor=#333333;strokeWidth=1.5;verticalAlign=middle;' + FONT)
_OP = 'shape=ellipse;whiteSpace=wrap;html=1;aspect=fixed;fontColor=#333333;strokeWidth=1.5;fontStyle=1;' + FONT
_CONT = ('rounded=1;arcSize=4;whiteSpace=wrap;html=1;dashed=1;dashPattern=8 4;fillColor=none;strokeWidth=2;'
         'verticalAlign=top;align=left;spacingLeft=8;spacingTop=5;fontSize=14;fontStyle=1;' + FONT)
_EDGE = ('edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#333333;strokeWidth=1.5;'
         'fontSize=11;fontColor=#333333;labelBackgroundColor=#FFFFFF;' + FONT)


def _fill(color):
    f, s = PALETTE[color]
    return f'fillColor={f};strokeColor={s};'


def _box(color, size=12, extra=''):
    return _BOX + _fill(color) + f'fontSize={size};' + extra


def _cube(color, size=13, depth=10):
    return _CUBE + _fill(color) + f'fontSize={size};size={depth};'


# kind: 'node' (regular block), 'op' (small circular operator),
# 'container' (dashed grouping frame), 'text' (free label, no border)
STYLES = {
    # ---- 3-D tensor blocks (cubes) ----
    'input':      ('node', _cube('red', 14)),
    'conv':       ('node', _cube('blue', 13)),
    'conv_sm':    ('node', _cube('blue', 11, 8)),
    'deconv':     ('node', _cube('sky', 13)),
    'deconv_sm':  ('node', _cube('sky', 11, 8)),
    'pool':       ('node', _cube('green', 13)),
    'pool_sm':    ('node', _cube('green', 11, 8)),
    'fc':         ('node', _cube('orange', 13, 8)),
    'output':     ('node', _cube('yellow', 13, 8)),
    'feature':    ('node', _cube('gray', 11, 8)),     # generic feature map / tensor
    'latent':     ('node', _cube('yellow', 12, 8)),   # latent tensor
    # ---- flat blocks ----
    'in_box':     ('node', _box('red')),              # generic input (tokens, image, x)
    'out_box':    ('node', _box('yellow')),           # generic output / prediction
    'conv_box':   ('node', _box('blue')),
    'deconv_box': ('node', _box('sky')),              # upsample / transposed conv
    'pool_box':   ('node', _box('green')),            # pooling / downsample / patch merging
    'fc_box':     ('node', _box('orange')),           # linear / MLP / FFN / experts
    'attn':       ('node', _box('purple')),           # (self-)attention / embedding
    'cross_attn': ('node', _box('red')),              # cross-attention (legacy convention)
    'norm':       ('node', _box('gray')),             # BN / LN / RMSNorm / GN
    'reshape':    ('node', _box('gray')),             # flatten / reshape / patchify / split
    'act':        ('node', _box('white')),            # ReLU / GELU / SiLU / Softmax ...
    'softmax':    ('node', _box('yellow')),
    'embed':      ('node', _box('purple')),           # token / patch / time embedding
    'pos':        ('node', _box('yellow')),           # positional encoding / condition
    'ssm':        ('node', _box('teal')),             # SSM / scan / recurrence / memory
    'rnn':        ('node', _box('teal', 14, 'arcSize=20;')),  # LSTM / GRU cell
    'gate':       ('node', _box('pink')),             # router / gating / quantiser / codebook
    'q':          ('node', _box('orange', 13, 'fontStyle=1;')),
    'k':          ('node', _box('blue', 13, 'fontStyle=1;')),
    'v':          ('node', _box('green', 13, 'fontStyle=1;')),
    'loss':       ('node', _box('red', 12, 'dashed=1;dashPattern=4 3;fillColor=#FFFFFF;')),
    'frozen':     ('node', _box('gray', 12, 'strokeColor=#666666;fontStyle=2;')),  # frozen module
    'module':     ('node', _box('blue', 13, 'arcSize=12;')),   # large generic module (encoder, LLM...)
    'module_alt': ('node', _box('green', 13, 'arcSize=12;')),  # paired module (decoder, critic...)
    'module_gen': ('node', _box('purple', 13, 'arcSize=12;')), # generator / transformer stack
    'noise':      ('node', _box('gray', 11, 'dashed=1;dashPattern=4 4;')),
    'mask':       ('node', _box('dark', 12, 'fontColor=#FFFFFF;')),
    'env':        ('node', _box('green', 13, 'arcSize=12;')),  # RL environment / external world
    'db':         ('node', 'shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=8;'
                           'fontColor=#333333;strokeWidth=1.5;fontSize=12;' + _fill('gray') + FONT),
    'trapezoid_enc': ('node', 'shape=trapezoid;whiteSpace=wrap;html=1;fixedSize=1;size=15;perimeter=trapezoidPerimeter;'
                              'direction=south;fontColor=#333333;strokeWidth=1.5;fontSize=13;' + _fill('blue') + FONT),
    'trapezoid_dec': ('node', 'shape=trapezoid;whiteSpace=wrap;html=1;fixedSize=1;size=15;perimeter=trapezoidPerimeter;'
                              'direction=north;fontColor=#333333;strokeWidth=1.5;fontSize=13;' + _fill('green') + FONT),
    # decoder drawn right-to-left (wide on the left, narrow on the right)
    'trapezoid_dec_rev': ('node', 'shape=trapezoid;whiteSpace=wrap;html=1;fixedSize=1;size=15;perimeter=trapezoidPerimeter;'
                                  'direction=south;fontColor=#333333;strokeWidth=1.5;fontSize=13;' + _fill('green') + FONT),
    'decision':   ('node', 'rhombus;whiteSpace=wrap;html=1;fontColor=#333333;strokeWidth=1.5;fontSize=11;'
                           + _fill('white') + FONT),
    # ---- operators (circles, use symbols: +  x  C  @  s  ...) ----
    'op':         ('op', _OP + _fill('white') + 'fontSize=16;'),
    'op_sm':      ('op', _OP + _fill('white') + 'fontSize=12;'),
    'dot':        ('op', 'shape=ellipse;html=1;aspect=fixed;fillColor=#333333;strokeColor=#333333;'
                         'strokeWidth=1;fontSize=0;'),
    # ---- containers ----
    'container_gray':   ('container', _CONT + 'strokeColor=#999999;fontColor=#999999;'),
    'container_blue':   ('container', _CONT + 'strokeColor=#6C8EBF;fontColor=#6C8EBF;'),
    'container_green':  ('container', _CONT + 'strokeColor=#82B366;fontColor=#82B366;'),
    'container_red':    ('container', _CONT + 'strokeColor=#B85450;fontColor=#B85450;'),
    'container_purple': ('container', _CONT + 'strokeColor=#9673A6;fontColor=#9673A6;'),
    'container_orange': ('container', _CONT + 'strokeColor=#D79B00;fontColor=#D79B00;'),
    'container_teal':   ('container', _CONT + 'strokeColor=#4A9C94;fontColor=#4A9C94;'),
    # title at bottom-right, used for "x N" repeat frames
    'container_repeat': ('container', _CONT.replace('verticalAlign=top;align=left;spacingLeft=8;spacingTop=5;',
                                                    'verticalAlign=bottom;align=right;spacingRight=8;spacingBottom=4;')
                         .replace('fontSize=14;', 'fontSize=13;') + 'strokeColor=#666666;fontColor=#666666;'),
    # ---- free text ----
    'label':      ('text', 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;'
                           'whiteSpace=wrap;fontSize=11;fontColor=#333333;' + FONT),
    'label_left': ('text', 'text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;'
                           'whiteSpace=wrap;spacingLeft=0;fontSize=11;fontColor=#333333;' + FONT),
    'dim_label':  ('text', 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;'
                           'whiteSpace=wrap;fontSize=10;fontColor=#999999;' + FONT),
    'caption':    ('text', 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;'
                           'whiteSpace=wrap;fontSize=14;fontStyle=2;fontColor=#333333;' + FONT),
    'ellipsis':   ('text', 'text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;'
                           'whiteSpace=wrap;fontSize=16;fontStyle=1;fontColor=#999999;' + FONT),
}

EDGE_STYLES = {
    'arrow':      _EDGE + 'endArrow=block;endFill=1;endSize=2;',
    'line':       _EDGE + 'endArrow=none;',                                   # no head
    'skip':       _EDGE + 'endArrow=block;endFill=1;endSize=2;dashed=1;dashPattern=8 4;',
    'skip_line':  _EDGE + 'endArrow=none;dashed=1;dashPattern=8 4;',
    'bidir':      _EDGE + 'endArrow=block;endFill=1;endSize=2;startArrow=block;startFill=1;startSize=2;',
    'loss_flow':  _EDGE + 'endArrow=block;endFill=1;endSize=2;dashed=1;dashPattern=3 3;strokeColor=#B85450;',
    'grad':       _EDGE + 'endArrow=block;endFill=1;endSize=2;dashed=1;dashPattern=3 3;strokeColor=#B85450;',
}

SIDES = {
    'top': (0.5, 0.0), 'bottom': (0.5, 1.0), 'left': (0.0, 0.5), 'right': (1.0, 0.5),
}
SIDE_DIR = {'top': (0, -1), 'bottom': (0, 1), 'left': (-1, 0), 'right': (1, 0)}

REGISTRY = []   # list of dicts: category, title, fn, module


def register(category, title):
    def deco(fn):
        REGISTRY.append({'category': category, 'title': title, 'fn': fn, 'module': fn.__module__})
        return fn
    return deco


def fmt(v):
    v = float(v)
    if v.is_integer():
        return str(int(v))
    # 4 decimals: port fractions on wide nodes need sub-pixel precision
    return ('%.4f' % v).rstrip('0').rstrip('.')


def esc(v):
    return html.escape(str(v), quote=True)


class Node:
    __slots__ = ('id', 'value', 'x', 'y', 'w', 'h', 'style_key', 'kind')

    def __init__(self, cid, value, x, y, w, h, style_key, kind):
        self.id, self.value, self.style_key, self.kind = cid, value, style_key, kind
        self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)

    # convenience geometry
    @property
    def cx(self):
        return self.x + self.w / 2

    @property
    def cy(self):
        return self.y + self.h / 2

    @property
    def r(self):
        return self.x + self.w

    @property
    def b(self):
        return self.y + self.h

    def port(self, side, pos=0.5):
        """Absolute coordinate of a port. pos is the fraction along the side
        (left->right for top/bottom, top->bottom for left/right)."""
        if side in ('top', 'bottom'):
            return (self.x + pos * self.w, self.y if side == 'top' else self.b)
        return (self.x if side == 'left' else self.r, self.y + pos * self.h)


class Edge:
    __slots__ = ('id', 'src', 'dst', 'exit', 'entry', 'exit_pos', 'entry_pos',
                 'via', 'style_key', 'label', 'p0', 'p1', 'jump')

    def points(self):
        """Full polyline: start port, waypoints, end port."""
        start = self.src.port(self.exit, self.exit_pos) if self.src else self.p0
        end = self.dst.port(self.entry, self.entry_pos) if self.dst else self.p1
        return [start] + list(self.via) + [end]


class Diagram:
    def __init__(self, title=''):
        self.title = title
        self.nodes = []
        self.edges = []
        self._vi = 0
        self._ei = 0

    # ------------------------------------------------------------------ nodes
    def node(self, value, x, y, w, h, style='fc_box'):
        if style not in STYLES:
            raise KeyError(f'unknown node style {style!r}')
        kind = STYLES[style][0]
        self._vi += 1
        n = Node(f'n{self._vi}', value, x, y, w, h, style, kind)
        self.nodes.append(n)
        return n

    def op(self, symbol, cx, cy, size=30, style='op'):
        """Circular operator centred at (cx, cy)."""
        return self.node(symbol, cx - size / 2, cy - size / 2, size, size, style)

    def container(self, title, x, y, w, h, style='container_gray'):
        if STYLES[style][0] != 'container':
            raise ValueError('container() needs a container_* style')
        return self.node(title, x, y, w, h, style)

    def text(self, value, x, y, w, h, style='label'):
        if STYLES[style][0] != 'text':
            raise ValueError('text() needs a text style')
        return self.node(value, x, y, w, h, style)

    # ------------------------------------------------------------------ edges
    def link(self, src, dst, exit=None, entry=None, via=(), style='arrow', label='',
             exit_pos=0.5, entry_pos=0.5, jump=False):
        """Connect two nodes with an orthogonal connector.

        exit / entry: 'top' | 'bottom' | 'left' | 'right'. When omitted they are
        inferred (vertical flow if dst is below/above src, else horizontal).
        via: list of (x, y) waypoints. Every consecutive pair of points
        (start port, waypoints..., end port) must share x or y.
        jump: draw a line-hop arc where this edge crosses another (use only for
        an unavoidable crossing).
        """
        if style not in EDGE_STYLES:
            raise KeyError(f'unknown edge style {style!r}')
        if exit is None or entry is None:
            ae, an = _auto_sides(src, dst)
            exit = exit or ae
            entry = entry or an
        self._ei += 1
        e = Edge()
        e.id, e.src, e.dst = f'e{self._ei}', src, dst
        e.exit, e.entry, e.exit_pos, e.entry_pos = exit, entry, exit_pos, entry_pos
        e.via = [(float(a), float(b)) for a, b in via]
        e.style_key, e.label, e.p0, e.p1, e.jump = style, label, None, None, jump
        self.edges.append(e)
        return e

    def arrow(self, points, style='arrow', label=''):
        """Free (unbound) orthogonal polyline, e.g. an incoming arrow from outside."""
        self._ei += 1
        e = Edge()
        e.id, e.src, e.dst = f'e{self._ei}', None, None
        e.exit = e.entry = None
        e.exit_pos = e.entry_pos = 0.5
        e.p0 = (float(points[0][0]), float(points[0][1]))
        e.p1 = (float(points[-1][0]), float(points[-1][1]))
        e.via = [(float(a), float(b)) for a, b in points[1:-1]]
        e.style_key, e.label, e.jump = style, label, False
        self.edges.append(e)
        return e

    # ---------------------------------------------------------------- output
    def bbox(self):
        xs, ys = [], []
        for n in self.nodes:
            xs += [n.x, n.r]
            ys += [n.y, n.b]
        for e in self.edges:
            for (px, py) in e.points():
                xs.append(px)
                ys.append(py)
        return min(xs), min(ys), max(xs), max(ys)

    def normalize(self):
        """Translate so that the bounding box starts at (MARGIN, MARGIN)."""
        x0, y0, _, _ = self.bbox()
        dx, dy = MARGIN - x0, MARGIN - y0
        if dx == 0 and dy == 0:
            return
        for n in self.nodes:
            n.x += dx
            n.y += dy
        for e in self.edges:
            e.via = [(a + dx, b + dy) for a, b in e.via]
            if e.p0:
                e.p0 = (e.p0[0] + dx, e.p0[1] + dy)
                e.p1 = (e.p1[0] + dx, e.p1[1] + dy)

    def size(self):
        _, _, x1, y1 = self.bbox()
        return int(round(x1 + MARGIN)), int(round(y1 + MARGIN))

    def to_xml(self):
        w, h = self.size()
        parts = ['<mxGraphModel><root><mxCell id="0"/><mxCell id="1" parent="0"/>',
                 f'<mxCell id="g" value="" style="group;" vertex="1" connectable="0" parent="1">'
                 f'<mxGeometry width="{w}" height="{h}" as="geometry"/></mxCell>']
        # containers first so they are painted underneath
        order = [n for n in self.nodes if n.kind == 'container'] + \
                [n for n in self.nodes if n.kind != 'container']
        for n in order:
            style = STYLES[n.style_key][1]
            extra = ' connectable="0"' if n.kind in ('container', 'text') else ''
            parts.append(
                f'<mxCell id="{n.id}" value="{esc(n.value)}" style="{esc(style)}" vertex="1"{extra} parent="g">'
                f'<mxGeometry x="{fmt(n.x)}" y="{fmt(n.y)}" width="{fmt(n.w)}" height="{fmt(n.h)}" as="geometry"/>'
                f'</mxCell>')
        for e in self.edges:
            style = EDGE_STYLES[e.style_key]
            if e.jump:
                style += 'jumpStyle=arc;jumpSize=8;'
            attrs = ''
            if e.src is not None:
                ex, ey = _unrotate(e.src, *_port_frac(e.exit, e.exit_pos))
                nx, ny = _unrotate(e.dst, *_port_frac(e.entry, e.entry_pos))
                style += (f'exitX={fmt(ex)};exitY={fmt(ey)};exitDx=0;exitDy=0;exitPerimeter=0;'
                          f'entryX={fmt(nx)};entryY={fmt(ny)};entryDx=0;entryDy=0;entryPerimeter=0;')
                attrs = f' source="{e.src.id}" target="{e.dst.id}"'
            geo = '<mxGeometry relative="1" as="geometry">'
            if e.src is None:
                geo += (f'<mxPoint x="{fmt(e.p0[0])}" y="{fmt(e.p0[1])}" as="sourcePoint"/>'
                        f'<mxPoint x="{fmt(e.p1[0])}" y="{fmt(e.p1[1])}" as="targetPoint"/>')
            if e.via:
                geo += '<Array as="points">' + ''.join(
                    f'<mxPoint x="{fmt(a)}" y="{fmt(b)}"/>' for a, b in e.via) + '</Array>'
            geo += '</mxGeometry>'
            parts.append(f'<mxCell id="{e.id}" value="{esc(e.label)}" style="{esc(style)}" edge="1"{attrs} '
                         f'parent="g">{geo}</mxCell>')
        parts.append('</root></mxGraphModel>')
        return ''.join(parts)


def _port_frac(side, pos):
    if side in ('top', 'bottom'):
        return pos, (0.0 if side == 'top' else 1.0)
    return (0.0 if side == 'left' else 1.0), pos


def _unrotate(node, fx, fy):
    """draw.io rotates connection constraints with the shape's direction
    (south = +90 deg, north = +270 deg). Pre-rotate the other way so the port
    lands where the generator intends."""
    style = STYLES[node.style_key][1]
    if 'direction=south' in style:
        return fy, 1.0 - fx
    if 'direction=north' in style:
        return 1.0 - fy, fx
    if 'direction=west' in style:
        return 1.0 - fx, 1.0 - fy
    return fx, fy


def _auto_sides(src, dst):
    if dst.y >= src.b:
        return 'bottom', 'top'
    if dst.b <= src.y:
        return 'top', 'bottom'
    if dst.x >= src.r:
        return 'right', 'left'
    return 'left', 'right'


def plain_text(value):
    """Strip HTML from a label; returns list of lines."""
    s = re.sub(r'<br\s*/?>', '\n', str(value), flags=re.I)
    s = re.sub(r'<[^>]+>', '', s)
    s = html.unescape(s)
    return s.split('\n')


def build(entry, normalize=True):
    """Run a registered model function and return the finished Diagram."""
    d = Diagram(entry['title'])
    entry['fn'](d)
    if normalize:
        d.normalize()
    return d


def library_entry(title, d):
    w, h = d.size()
    return {'xml': html.escape(d.to_xml(), quote=False), 'w': w, 'h': h, 'title': title}


def drawio_file(xml, name='Page-1'):
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<mxfile host="drawio" compressed="false">\n'
            f'  <diagram id="p1" name="{esc(name)}">\n{xml}\n  </diagram>\n</mxfile>\n')
