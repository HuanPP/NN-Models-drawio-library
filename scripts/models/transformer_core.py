"""Core Transformer mechanisms: attention internals, masks, norms placement."""

from nnlib.core import register

CAT = 'Transformers'


# ---------------------------------------------------------------- helpers
def _grid(d, x0, y0, n, cell_style, cell=18, gap=8, labels=True):
    """Token-by-token attention grid. cell_style(i, j) returns a style key
    (row i = query, column j = key) or None for a masked cell."""
    pitch = cell + gap
    for i in range(n):
        for j in range(n):
            st = cell_style(i, j) or 'act'
            d.node('', x0 + j * pitch, y0 + i * pitch, cell, cell, st)
    if labels:
        for k in range(n):
            lab = f'x<sub>{k + 1}</sub>'
            d.text(lab, x0 + k * pitch - 2, y0 - 20, cell + 4, 16, 'label')
            d.text(lab, x0 - 26, y0 + k * pitch, 20, cell, 'label')
    return n * cell + (n - 1) * gap


def _legend(d, x, y, items, cell=16, label_w=70):
    """Horizontal legend: [(style, text), ...] starting at x."""
    for st, txt in items:
        d.node('', x, y, cell, cell, st)
        d.text(txt, x + cell + 6, y - 1, label_w, cell + 2, 'label_left')
        x += cell + 6 + label_w + 14
    return x


# ------------------------------------------------------------------ models
@register(CAT, 'Scaled Dot-Product Attention')
def scaled_dot_product_attention(d):
    d.container('Scaled Dot-Product Attention', 105, 10, 290, 400, 'container_purple')
    q = d.node('Q', 130, 50, 60, 28, 'q')
    k = d.node('K', 210, 50, 60, 28, 'k')
    v = d.node('V', 310, 50, 60, 28, 'v')
    mm1 = d.node('MatMul', 130, 102, 140, 28, 'act')
    scale = d.node('Scale (1/&radic;d<sub>k</sub>)', 130, 154, 140, 28, 'act')
    mask = d.node('Mask (opt.)', 130, 206, 140, 28, 'mask')
    sm = d.node('SoftMax', 130, 258, 140, 28, 'softmax')
    mm2 = d.node('MatMul', 130, 310, 140, 28, 'act')
    out = d.node('Attention(Q, K, V)', 130, 362, 140, 28, 'out_box')
    d.link(q, mm1, entry_pos=30 / 140)
    d.link(k, mm1, entry_pos=110 / 140)
    for a, b in [(mm1, scale), (scale, mask), (mask, sm), (sm, mm2), (mm2, out)]:
        d.link(a, b)
    d.link(v, mm2, exit='bottom', entry='right', via=[(v.cx, mm2.cy)])
    d.text('N x N', 205, 129, 40, 14, 'dim_label')
    d.text('N x d<sub>v</sub>', 205, 341, 44, 14, 'dim_label')


@register(CAT, 'Cross-Attention Block')
def cross_attention_block(d):
    d.container('Cross-Attention Block', 80, 10, 390, 555, 'container_red')
    x = d.node('Target X (decoder)<br>N<sub>q</sub> x d', 130, 50, 140, 40, 'in_box')
    y = d.node('Context Y (encoder)<br>N<sub>kv</sub> x d', 310, 50, 140, 40, 'in_box')
    q = d.node('Q', 170, 115, 60, 30, 'q')
    k = d.node('K', 310, 115, 60, 30, 'k')
    v = d.node('V', 390, 115, 60, 30, 'v')
    ca = d.node('Multi-Head<br>Cross-Attention', 130, 170, 140, 40, 'cross_attn')
    add1 = d.op('+', 200, 250)
    ln1 = d.node('LayerNorm', 150, 290, 100, 30, 'norm')
    ffn = d.node('Feed-Forward', 130, 345, 140, 30, 'fc_box')
    add2 = d.op('+', 200, 415)
    ln2 = d.node('LayerNorm', 150, 455, 100, 30, 'norm')
    out = d.node('Output<br>N<sub>q</sub> x d', 130, 510, 140, 40, 'out_box')

    d.link(x, q)
    d.link(y, k, exit_pos=30 / 140)
    d.link(y, v, exit_pos=110 / 140)
    d.link(q, ca)
    d.link(k, ca, exit='bottom', entry='right', entry_pos=0.25, via=[(k.cx, ca.port('right', 0.25)[1])])
    d.link(v, ca, exit='bottom', entry='right', entry_pos=0.75, via=[(v.cx, ca.port('right', 0.75)[1])])
    for a, b in [(ca, add1), (add1, ln1), (ln1, ffn), (ffn, add2), (add2, ln2), (ln2, out)]:
        d.link(a, b)
    d.link(x, add1, exit='left', entry='left', via=[(105, x.cy), (105, add1.cy)], style='skip')
    d.link(ln1, add2, exit='left', entry='left', via=[(105, ln1.cy), (105, add2.cy)], style='skip')


@register(CAT, 'Pre-LN vs Post-LN')
def pre_ln_vs_post_ln(d):
    # row tops (box h 30, op 30); dots sit in the two branch rows
    tops = [50, None, 120, 175, 230, None, 300, 355, 410, 465]
    dot_cy = {1: 100, 5: 280}

    def column(cx, seq, caption, formula):
        d.container('', cx - 95, 30, 215, 485, 'container_gray')
        nodes = []
        for r, item in enumerate(seq):
            if item == 'dot':
                nodes.append(d.op('', cx, dot_cy[r], size=6, style='dot'))
            elif item == '+':
                nodes.append(d.op('+', cx, tops[r] + 15))
            else:
                label, st, w = item
                nodes.append(d.node(label, cx - w / 2, tops[r], w, 30, st))
        for r in range(len(nodes) - 1):
            st = 'line' if seq[r + 1] == 'dot' else 'arrow'
            d.link(nodes[r], nodes[r + 1], exit='bottom', entry='top', style=st)
        sx = cx + 95
        dots = [r for r, it in enumerate(seq) if it == 'dot']
        adds = [r for r, it in enumerate(seq) if it == '+']
        for rd, ra in zip(dots, adds):
            a, b = nodes[rd], nodes[ra]
            d.link(a, b, exit='right', entry='right', via=[(sx, a.cy), (sx, b.cy)], style='skip')
        d.text(caption, cx - 95, 525, 215, 20, 'caption')
        d.text(formula, cx - 95, 548, 215, 18, 'label')

    inp = ('x<sub>l</sub>', 'in_box', 140)
    out = ('x<sub>l+1</sub>', 'out_box', 140)
    mha = ('Multi-Head Attention', 'attn', 140)
    ffn = ('Feed-Forward', 'fc_box', 140)
    ln = ('LayerNorm', 'norm', 100)
    column(165, [inp, 'dot', mha, '+', ln, 'dot', ffn, '+', ln, out], '(a) Post-LN',
           'x<sub>l+1</sub> = LN(x<sub>l</sub> + F(x<sub>l</sub>))')
    column(420, [inp, 'dot', ln, mha, '+', 'dot', ln, ffn, '+', out], '(b) Pre-LN',
           'x<sub>l+1</sub> = x<sub>l</sub> + F(LN(x<sub>l</sub>))')


@register(CAT, 'Attention Masks')
def attention_masks(d):
    n, prefix = 6, 3
    patterns = [
        ('(a) Bidirectional', 'encoder (BERT)', lambda i, j: 'attn'),
        ('(b) Causal', 'decoder (GPT)', lambda i, j: 'attn' if j <= i else None),
        ('(c) Prefix-LM', f'prefix = {prefix} tokens (T5, UniLM)',
         lambda i, j: 'attn' if (j < prefix or j <= i) else None),
    ]
    x0, y0 = 100, 70
    for cap, sub, fn in patterns:
        span = _grid(d, x0, y0, n, fn)
        d.text(cap, x0 - 30, y0 + span + 15, span + 60, 20, 'caption')
        d.text(sub, x0 - 30, y0 + span + 37, span + 60, 16, 'dim_label')
        x0 += span + 60
    ly = y0 + 148 + 75
    end = _legend(d, 180, ly, [('attn', 'attended'), ('act', 'masked')])
    d.text('rows: queries, columns: keys', end, ly - 1, 170, 18, 'label')


@register(CAT, 'Sparse Attention Patterns')
def sparse_attention_patterns(d):
    n = 8
    rand = {(1, 4), (1, 6), (2, 5), (3, 7), (3, 1), (4, 2), (4, 7), (5, 1), (6, 3), (7, 2), (7, 5)}

    def window(i, j):
        return 'attn' if abs(i - j) <= 1 else None

    def dilated(i, j):
        return 'attn' if abs(i - j) in (0, 2) else None

    def longformer(i, j):
        if i == 0 or j == 0:
            return 'fc_box'
        return window(i, j)

    def bigbird(i, j):
        if i == 0 or j == 0:
            return 'fc_box'
        if (i, j) in rand:
            return 'conv_box'
        return window(i, j)

    patterns = [
        ('(a) Sliding window', 'local, w = 3', window),
        ('(b) Dilated window', 'w = 3, dilation 2', dilated),
        ('(c) Longformer', 'global + sliding window', longformer),
        ('(d) BigBird', 'random + window + global', bigbird),
    ]
    cell, gap = 16, 8
    span = n * cell + (n - 1) * gap
    xs = [100, 100 + span + 70]
    ys = [70, 70 + span + 90]
    for idx, (cap, sub, fn) in enumerate(patterns):
        x0, y0 = xs[idx % 2], ys[idx // 2]
        _grid(d, x0, y0, n, fn, cell=cell, gap=gap)
        d.text(cap, x0 - 30, y0 + span + 12, span + 60, 20, 'caption')
        d.text(sub, x0 - 30, y0 + span + 34, span + 60, 16, 'dim_label')
    ly = ys[1] + span + 70
    _legend(d, 80, ly, [('attn', 'local window'), ('fc_box', 'global'),
                        ('conv_box', 'random'), ('act', 'masked')], label_w=62)


@register(CAT, 'Linear Attention')
def linear_attention(d):
    rows = [50, 105, 160, 215, 270]

    # (a) softmax attention: (Q K^T) first, N x N intermediate
    cx = 185
    d.container('', 85, 30, 265, 290, 'container_gray')
    q = d.node('Q', cx - 70, rows[0], 60, 30, 'q')
    k = d.node('K', cx + 10, rows[0], 60, 30, 'k')
    v = d.node('V', cx + 85, rows[0], 60, 30, 'v')
    mm1 = d.node('MatMul', cx - 70, rows[1], 140, 30, 'act')
    sm = d.node('SoftMax', cx - 70, rows[2], 140, 30, 'softmax')
    mm2 = d.node('MatMul', cx - 70, rows[3], 140, 30, 'act')
    out = d.node('Output', cx - 70, rows[4], 140, 30, 'out_box')
    d.link(q, mm1, entry_pos=30 / 140)
    d.link(k, mm1, entry_pos=110 / 140)
    for a, b in [(mm1, sm), (sm, mm2), (mm2, out)]:
        d.link(a, b)
    d.link(v, mm2, exit='bottom', entry='right', via=[(v.cx, mm2.cy)])
    d.text('QK<sup>T</sup>: N x N', cx + 5, rows[1] + 35, 70, 16, 'label')
    d.text('N x d', cx + 5, rows[3] + 35, 40, 16, 'dim_label')
    d.text('(a) Softmax attention', 85, 330, 265, 20, 'caption')
    d.text('softmax(QK<sup>T</sup>) V<br>O(N<sup>2</sup>d) time, O(N<sup>2</sup>) memory',
           85, 352, 265, 32, 'label')

    # (b) linear attention: (phi(K)^T V) first, d x d intermediate
    cx = 535
    d.container('', 365, 30, 265, 290, 'container_gray')
    q = d.node('Q', cx - 145, rows[0], 60, 30, 'q')
    k = d.node('K', cx - 70, rows[0], 60, 30, 'k')
    v = d.node('V', cx + 10, rows[0], 60, 30, 'v')
    fq = d.node('&phi;(&middot;)', cx - 145, rows[1], 60, 30, 'act')
    fk = d.node('&phi;(&middot;)', cx - 70, rows[1], 60, 30, 'act')
    mm1 = d.node('MatMul', cx - 70, rows[2], 140, 30, 'act')
    mm2 = d.node('MatMul', cx - 70, rows[3], 140, 30, 'act')
    out = d.node('Output', cx - 70, rows[4], 140, 30, 'out_box')
    d.link(q, fq)
    d.link(k, fk)
    d.link(fk, mm1, entry_pos=30 / 140)
    d.link(v, mm1, entry_pos=110 / 140)
    d.link(fq, mm2, exit='bottom', entry='left', via=[(fq.cx, mm2.cy)])
    d.link(mm1, mm2)
    d.link(mm2, out)
    d.text('&phi;(K)<sup>T</sup>V: d x d', cx + 5, rows[2] + 35, 80, 16, 'label')
    d.text('N x d', cx + 5, rows[3] + 35, 40, 16, 'dim_label')
    d.text('(b) Linear attention', 365, 330, 265, 20, 'caption')
    d.text('&phi;(Q) (&phi;(K)<sup>T</sup> V)<br>O(Nd<sup>2</sup>) time, O(d<sup>2</sup>) memory',
           365, 352, 265, 32, 'label')
