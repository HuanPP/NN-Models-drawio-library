"""Small drag-and-drop building blocks (published as nn_components.xml)."""

from nnlib.core import register

CAT = 'Components'

BW, BH = 120, 30          # standard flat block
BH2 = 40                  # two-line flat block


# --------------------------------------------------------------------------
# Tensor blocks (cubes)
# --------------------------------------------------------------------------
def _cube(title, label, w, h, style):
    @register(CAT, title)
    def fn(d):
        d.node(label, 0, 0, w, h, style)
    fn.__name__ = 'cube_' + title.lower().replace(' ', '_')
    return fn


_cube('Input Tensor', 'Input', 80, 80, 'input')
_cube('Conv Layer', 'Conv', 70, 80, 'conv')
_cube('Pooling Layer', 'Pool', 60, 60, 'pool')
_cube('Deconv Layer', 'Deconv', 70, 80, 'deconv')
_cube('FC Layer', 'FC', 40, 100, 'fc')
_cube('Output Tensor', 'Output', 60, 70, 'output')
_cube('Feature Map', 'Feature', 60, 70, 'feature')
_cube('Latent Tensor', 'Latent', 60, 60, 'latent')


@register(CAT, 'Feature Map Stack')
def feature_map_stack(d):
    for i in range(3):
        d.node('', i * 40, 0, 30, 80, 'feature')
    d.text('H x W x C', 0, 85, 110, 14, 'dim_label')


# --------------------------------------------------------------------------
# Flat blocks
# --------------------------------------------------------------------------
def _flat(title, label, style, w=BW, h=BH):
    @register(CAT, title)
    def fn(d):
        d.node(label, 0, 0, w, h, style)
    fn.__name__ = 'flat_' + title.lower().replace(' ', '_').replace('-', '_')
    return fn


_flat('Linear', 'Linear', 'fc_box')
_flat('Conv Block', 'Conv 3x3', 'conv_box')
_flat('Upsample', 'Upsample 2x', 'deconv_box')
_flat('Downsample', 'Downsample 2x', 'pool_box')
_flat('Self-Attention', 'Self-Attention', 'attn')
_flat('Cross-Attention', 'Cross-Attention', 'cross_attn')
_flat('Multi-Head Attention', 'Multi-Head<br>Attention', 'attn', h=BH2)
_flat('LayerNorm', 'LayerNorm', 'norm')
_flat('BatchNorm', 'BatchNorm', 'norm')
_flat('RMSNorm', 'RMSNorm', 'norm')
_flat('Activation', 'ReLU', 'act')
_flat('Softmax', 'Softmax', 'softmax')
_flat('Dropout', 'Dropout', 'act')
_flat('Embedding', 'Embedding', 'embed')
_flat('Positional Encoding', 'Positional<br>Encoding', 'pos', h=BH2)
_flat('SSM', 'SSM', 'ssm')
_flat('Recurrent Cell', 'LSTM', 'rnn', h=BH2)
_flat('Router', 'Router', 'gate')
_flat('Loss', 'Loss', 'loss')
_flat('Frozen Module', 'Backbone (frozen)', 'frozen')
_flat('Encoder', 'Encoder', 'module', w=140, h=BH2)
_flat('Decoder', 'Decoder', 'module_alt', w=140, h=BH2)
_flat('Noise', 'ε ~ N(0, I)', 'noise')
_flat('Mask', 'Mask', 'mask')
_flat('Memory Bank', 'Memory Bank', 'db', h=50)
_flat('Environment', 'Environment', 'env', w=140, h=BH2)


# --------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------
def _op(title, symbol):
    @register(CAT, title)
    def fn(d):
        d.op(symbol, 15, 15)
    fn.__name__ = 'op_' + title.lower().replace(' ', '_')
    return fn


_op('Add', '+')
_op('Multiply', '×')
_op('Concat', 'C')
_op('MatMul', '@')
_op('Sigmoid Gate', 'σ')


# --------------------------------------------------------------------------
# Composites
# --------------------------------------------------------------------------
@register(CAT, 'Q K V Projections')
def qkv_projections(d):
    pw, gap = 80, 30
    total = 3 * pw + 2 * gap
    cx = total / 2
    x = d.node('X', cx - BW / 2, 0, BW, BH, 'in_box')
    split = BH + 20
    for i, (lab, st) in enumerate([('Q', 'q'), ('K', 'k'), ('V', 'v')]):
        p = d.node(lab, i * (pw + gap), split + 25, pw, BH, st)
        if p.cx == cx:
            d.link(x, p)
        else:
            d.link(x, p, exit='bottom', entry='top', via=[(cx, split), (p.cx, split)])


@register(CAT, 'Residual Block')
def residual_block(d):
    x = d.node('x', 0, 0, BW, BH, 'in_box')
    l1 = d.node('Conv 3x3', 0, 54, BW, BH, 'conv_box')
    l2 = d.node('Conv 3x3', 0, 108, BW, BH, 'conv_box')
    add = d.op('+', BW / 2, 177)
    d.link(x, l1)
    d.link(l1, l2)
    d.link(l2, add)
    sx = BW + 25
    d.link(x, add, exit='right', entry='right', via=[(sx, x.cy), (sx, add.cy)], style='skip')
    d.arrow([(add.cx, add.b), (add.cx, add.b + 30)])


@register(CAT, 'Token Sequence')
def token_sequence(d):
    tw, gap = 36, 10
    xs = 0
    for i in (1, 2, 3, 4):
        d.node(f'x<sub>{i}</sub>', xs, 0, tw, BH, 'in_box')
        xs += tw + gap
    d.text('...', xs, 0, 30, BH, 'ellipsis')
    xs += 30 + gap
    d.node('x<sub>n</sub>', xs, 0, tw, BH, 'in_box')


@register(CAT, 'Repeat Frame')
def repeat_frame(d):
    d.container('× N', 0, 0, 160, 140, 'container_repeat')
    a = d.node('Self-Attention', 20, 20, BW, BH, 'attn')
    f = d.node('Feed-Forward', 20, 74, BW, BH, 'fc_box')
    d.link(a, f)


@register(CAT, 'Encoder-Decoder')
def encoder_decoder(d):
    enc = d.node('Encoder', 0, 0, 90, 100, 'trapezoid_enc')
    z = d.node('z', 130, 25, 40, 50, 'latent')
    dec = d.node('Decoder', 210, 0, 90, 100, 'trapezoid_dec')
    d.link(enc, z, exit='right', entry='left')
    d.link(z, dec, exit='right', entry='left')


# --------------------------------------------------------------------------
# Free arrows
# --------------------------------------------------------------------------
def _free_arrow(title, label, style):
    @register(CAT, title)
    def fn(d):
        d.arrow([(0, 0), (80, 0)], style=style)
        d.text(label, -10, 6, 100, 14, 'dim_label')
    fn.__name__ = 'arrow_' + title.lower().replace(' ', '_')
    return fn


_free_arrow('Data Flow Arrow', 'data flow', 'arrow')
_free_arrow('Skip Connection Arrow', 'skip connection', 'skip')
_free_arrow('Gradient Arrow', 'gradient', 'grad')


# --------------------------------------------------------------------------
# Legend
# --------------------------------------------------------------------------
LEGEND = [
    [('Input', 'in_box'), ('Convolution', 'conv_box'), ('Upsampling', 'deconv_box'),
     ('Pooling', 'pool_box'), ('Linear / MLP', 'fc_box'), ('Attention', 'attn')],
    [('Normalization', 'norm'), ('Activation', 'act'), ('State Space / Recurrent', 'ssm'),
     ('Gating / Routing', 'gate'), ('Output', 'out_box'), ('Loss', 'loss')],
]


@register(CAT, 'Color Legend')
def color_legend(d):
    col_x = [0, 175]
    for col, items in zip(col_x, LEGEND):
        for row, (lab, st) in enumerate(items):
            y = row * 24
            d.node('', col, y, 30, 16, st)
            d.text(lab, col + 38, y, 125, 16, 'label_left')
