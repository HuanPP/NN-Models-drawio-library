"""Self-supervised and metric representation learning: contrastive, distillation and masked modelling."""

from nnlib.core import register

CAT = 'Representation Learning'


# ---------------------------------------------------------------- helpers
def _split(d, src, dsts, y):
    """Fan out from the bottom of src: shared trunk down to y, then one branch per target."""
    for t in dsts:
        d.link(src, t, exit='bottom', entry='top', via=[(src.cx, y), (t.cx, y)])


def _down_in(d, src, dst, style='arrow'):
    """Leave src downwards, turn once and enter dst from the side facing src."""
    entry = 'left' if src.cx < dst.x else 'right'
    d.link(src, dst, exit='bottom', entry=entry, via=[(src.cx, dst.cy)], style=style)


def _shared(d, a, b, label='shared'):
    d.link(a, b, exit='right', entry='left', style='line', label=label)


def _grid(d, x0, y0, n, cell, gap, style_of):
    """n x n grid of small square cells; style_of(r, c) gives the style key."""
    for r in range(n):
        for c in range(n):
            d.node('', x0 + c * (cell + gap), y0 + r * (cell + gap), cell, cell, style_of(r, c))


# ---------------------------------------------------------------- SimCLR
@register(CAT, 'SimCLR')
def simclr(d):
    L, R, W, H = 150, 350, 130, 30
    x = d.node('Input x', 200, 10, 100, H, 'in_box')
    rows = [
        ('Augment t ~ T', "Augment t' ~ T", 'reshape'),
        ('View x\u0303<sub>i</sub>', 'View x\u0303<sub>j</sub>', 'in_box'),
        ('Encoder f(·)', 'Encoder f(·)', 'module'),
        ('Representation h<sub>i</sub>', 'Representation h<sub>j</sub>', 'out_box'),
        ('Projection g(·)', 'Projection g(·)', 'fc_box'),
        ('z<sub>i</sub>', 'z<sub>j</sub>', 'embed'),
    ]
    left, right = [], []
    y = 80
    for la, ra, st in rows:
        left.append(d.node(la, L - W / 2, y, W, H, st))
        right.append(d.node(ra, R - W / 2, y, W, H, st))
        y += 55
    _split(d, x, [left[0], right[0]], 60)
    for col in (left, right):
        for a, b in zip(col, col[1:]):
            d.link(a, b)
    _shared(d, left[2], right[2])
    _shared(d, left[4], right[4])
    loss = d.node('NT-Xent loss<br>maximise agreement', 170, 420, 160, 40, 'loss')
    _down_in(d, left[-1], loss)
    _down_in(d, right[-1], loss)


# ---------------------------------------------------------------- MoCo
@register(CAT, 'MoCo')
def moco(d):
    L, M, Q, W = 110, 350, 550, 130
    x = d.node('Input x', 180, 10, 100, 30, 'in_box')
    xq = d.node('Query view x<sup>q</sup>', L - W / 2, 75, W, 30, 'in_box')
    xk = d.node('Key view x<sup>k</sup>', M - W / 2, 75, W, 30, 'in_box')
    _split(d, x, [xq, xk], 55)
    fq = d.node('Query<br>encoder f<sub>q</sub>', L - W / 2, 130, W, 40, 'module')
    fk = d.node('Momentum<br>encoder f<sub>k</sub>', M - W / 2, 130, W, 40, 'module_alt')
    d.link(xq, fq)
    d.link(xk, fk)
    d.link(fq, fk, exit='right', entry='left', style='grad', label='EMA')
    d.text('θ<sub>k</sub> ← mθ<sub>k</sub> + (1−m)θ<sub>q</sub>', 180, 160, 100, 14, 'dim_label')
    q = d.node('query q', L - W / 2, 195, W, 30, 'q')
    k = d.node('key k<sub>+</sub>', M - W / 2, 195, W, 30, 'k')
    d.link(fq, q)
    d.link(fk, k)
    queue = d.node('Queue of keys<br>k<sub>0</sub>, k<sub>1</sub>, ..., k<sub>K</sub>',
                   Q - W / 2, 185, W, 50, 'db')
    d.link(k, queue, exit='right', entry='left', label='enqueue')
    loss = d.node('InfoNCE loss<br>(contrastive)', M - 75, 275, 150, 40, 'loss')
    _down_in(d, q, loss)
    d.link(k, loss)
    _down_in(d, queue, loss)
    d.text('positive', M + 5, 243, 50, 14, 'dim_label')
    d.text('negatives', Q + 5, 250, 55, 14, 'dim_label')


# ---------------------------------------------------------------- BYOL
@register(CAT, 'BYOL')
def byol(d):
    L, R, W, H = 150, 400, 140, 30
    x = d.node('Input x', 225, 10, 100, H, 'in_box')
    v1 = d.node('View v', L - W / 2, 75, W, H, 'in_box')
    v2 = d.node("View v'", R - W / 2, 75, W, H, 'in_box')
    _split(d, x, [v1, v2], 55)
    d.container('Online', L - 100, 125, 200, 215, 'container_blue')
    d.container('Target', R - 100, 125, 200, 215, 'container_green')
    enc = d.node('Encoder f<sub>θ</sub>', L - W / 2, 160, W, H, 'module')
    proj = d.node('Projector g<sub>θ</sub>', L - W / 2, 220, W, H, 'fc_box')
    pred = d.node('Predictor q<sub>θ</sub>', L - W / 2, 280, W, H, 'fc_box')
    tenc = d.node('Encoder f<sub>ξ</sub>', R - W / 2, 160, W, H, 'module_alt')
    tproj = d.node('Projector g<sub>ξ</sub>', R - W / 2, 220, W, H, 'fc_box')
    sg = d.op('sg', R, 300, 30, 'op_sm')
    for a, b in [(v1, enc), (enc, proj), (proj, pred), (v2, tenc), (tenc, tproj), (tproj, sg)]:
        d.link(a, b)
    d.link(enc, tenc, exit='right', entry='left', style='grad', label='EMA')
    d.link(proj, tproj, exit='right', entry='left', style='grad', label='EMA')
    d.text('y<sub>θ</sub>', L + 5, 195, 25, 14, 'dim_label')
    d.text('z<sub>θ</sub>', L + 5, 255, 25, 14, 'dim_label')
    d.text("y'<sub>ξ</sub>", R + 5, 195, 25, 14, 'dim_label')
    d.text("z'<sub>ξ</sub>", R + 5, 255, 25, 14, 'dim_label')
    loss = d.node('MSE of l2-normalised<br>q<sub>θ</sub>(z<sub>θ</sub>) and sg(z\'<sub>ξ</sub>)',
                  190, 365, 170, 40, 'loss')
    _down_in(d, pred, loss)
    _down_in(d, sg, loss)
    d.text("symmetrised by swapping v and v'", 175, 415, 200, 14, 'dim_label')


# ---------------------------------------------------------------- SimSiam
@register(CAT, 'SimSiam')
def simsiam(d):
    L, R, W = 140, 360, 160
    x = d.node('Input x', 200, 10, 100, 30, 'in_box')
    v1 = d.node('View x<sub>1</sub>', L - W / 2, 75, W, 30, 'in_box')
    v2 = d.node('View x<sub>2</sub>', R - W / 2, 75, W, 30, 'in_box')
    _split(d, x, [v1, v2], 55)
    f1 = d.node('Encoder f<br>(backbone + projector)', L - W / 2, 130, W, 40, 'module')
    f2 = d.node('Encoder f<br>(backbone + projector)', R - W / 2, 130, W, 40, 'module')
    d.link(v1, f1)
    d.link(v2, f2)
    _shared(d, f1, f2)
    pred = d.node('Predictor h (MLP)', L - W / 2, 195, W, 30, 'fc_box')
    sg = d.op('sg', R, 210, 30, 'op_sm')
    d.link(f1, pred)
    d.link(f2, sg)
    d.text('z<sub>1</sub>', L + 5, 175, 25, 14, 'dim_label')
    d.text('z<sub>2</sub>', R + 5, 175, 25, 14, 'dim_label')
    d.text('p<sub>1</sub>', L + 5, 240, 25, 14, 'dim_label')
    loss = d.node('Negative cosine similarity<br>D(p<sub>1</sub>, sg(z<sub>2</sub>))', 165, 260, 170, 40, 'loss')
    _down_in(d, pred, loss)
    _down_in(d, sg, loss)
    d.text('symmetrised: L = ½ D(p<sub>1</sub>, sg(z<sub>2</sub>)) + ½ D(p<sub>2</sub>, sg(z<sub>1</sub>))',
           110, 310, 280, 14, 'dim_label')


# ---------------------------------------------------------------- MAE
@register(CAT, 'MAE')
def mae(d):
    cell, gap, n = 15, 10, 4
    visible = {(0, 1), (1, 3), (2, 0), (3, 2)}
    y0 = 95
    # input image with 75 % of the patches masked
    a = d.container('', 10, y0, 110, 110, 'container_gray')
    _grid(d, 20, y0 + 10, n, cell, gap, lambda r, c: 'in_box' if (r, c) in visible else 'norm')
    # visible patches only
    b = d.container('', 155, y0, 35, 110, 'container_gray')
    for i in range(n):
        d.node('', 165, y0 + 10 + i * (cell + gap), cell, cell, 'in_box')
    enc = d.node('Encoder<br>(ViT)', 225, y0, 80, 110, 'module')
    # encoded visible tokens + shared mask tokens, restored to full length
    g = d.container('', 340, y0, 110, 110, 'container_gray')
    _grid(d, 350, y0 + 10, n, cell, gap, lambda r, c: 'conv_box' if (r, c) in visible else 'norm')
    dec = d.node('Lightweight<br>Decoder', 475, y0 + 15, 80, 80, 'module_alt')
    f = d.container('', 580, y0, 110, 110, 'container_gray')
    _grid(d, 590, y0 + 10, n, cell, gap, lambda r, c: 'out_box')
    for s, t in [(a, b), (b, enc), (enc, g), (g, dec), (dec, f)]:
        d.link(s, t, exit='right', entry='left')
    d.text('Input image<br>75% masked', 10, 60, 110, 28, 'label')
    d.text('Visible<br>patches', 145, 60, 55, 28, 'label')
    d.text('Encoded tokens<br>+ mask tokens', 340, 60, 110, 28, 'label')
    d.text('Reconstructed<br>pixels', 580, 60, 110, 28, 'label')
    loss = d.node('MSE loss<br>(masked patches)', 570, 245, 130, 40, 'loss')
    d.link(f, loss)
    d.link(a, loss, exit='bottom', entry='left', via=[(a.cx, loss.cy)], label='target: original pixels')


# ---------------------------------------------------------------- DINO
@register(CAT, 'DINO')
def dino(d):
    L, R, W, H = 150, 390, 150, 30
    x = d.node('Input x', 220, 10, 100, H, 'in_box')
    v1 = d.node('Global + local crops', L - W / 2, 75, W, H, 'in_box')
    v2 = d.node('Global crops', R - W / 2, 75, W, H, 'in_box')
    _split(d, x, [v1, v2], 55)
    st = d.node('Student g<sub>θs</sub>', L - W / 2, 130, W, H, 'module')
    te = d.node('Teacher g<sub>θt</sub>', R - W / 2, 130, W, H, 'module_alt')
    d.link(v1, st)
    d.link(v2, te)
    d.link(st, te, exit='right', entry='left', style='grad', label='EMA')
    cen = d.node('Centering (− c)', R - W / 2, 185, W, H, 'norm')
    d.text('c: EMA of<br>batch mean', 235, 186, 70, 28, 'dim_label')
    ss = d.node('Softmax (τ<sub>s</sub>)', L - W / 2, 240, W, H, 'softmax')
    ts = d.node('Sharpen: softmax (τ<sub>t</sub>)', R - W / 2, 240, W, H, 'softmax')
    sg = d.op('sg', R, 305, 30, 'op_sm')
    for a, b in [(st, ss), (te, cen), (cen, ts), (ts, sg)]:
        d.link(a, b)
    d.text('p<sub>s</sub>', L + 5, 285, 25, 14, 'dim_label')
    d.text('p<sub>t</sub>', R + 5, 335, 25, 14, 'dim_label')
    loss = d.node('Cross-entropy<br>− p<sub>t</sub> log p<sub>s</sub>', 185, 345, 170, 40, 'loss')
    _down_in(d, ss, loss)
    _down_in(d, sg, loss)


# ---------------------------------------------------------------- Barlow Twins
@register(CAT, 'Barlow Twins')
def barlow_twins(d):
    L, R, W = 150, 350, 150
    x = d.node('Input x', 200, 10, 100, 30, 'in_box')
    ya = d.node('Distorted Y<sup>A</sup>', L - W / 2, 75, W, 30, 'in_box')
    yb = d.node('Distorted Y<sup>B</sup>', R - W / 2, 75, W, 30, 'in_box')
    _split(d, x, [ya, yb], 55)
    fa = d.node('Encoder f<sub>θ</sub><br>(ResNet + projector)', L - W / 2, 130, W, 40, 'module')
    fb = d.node('Encoder f<sub>θ</sub><br>(ResNet + projector)', R - W / 2, 130, W, 40, 'module')
    d.link(ya, fa)
    d.link(yb, fb)
    _shared(d, fa, fb)
    za = d.node('Embeddings Z<sup>A</sup>', L - W / 2, 195, W, 30, 'embed')
    zb = d.node('Embeddings Z<sup>B</sup>', R - W / 2, 195, W, 30, 'embed')
    d.link(fa, za)
    d.link(fb, zb)
    d.text('Cross-correlation C', 180, 240, 140, 16, 'label')
    c = d.container('', 195, 260, 120, 120, 'container_gray')
    _grid(d, 210, 275, 4, 15, 10, lambda r, k: 'fc_box' if r == k else 'norm')
    _down_in(d, za, c)
    _down_in(d, zb, c)
    d.text('N x D', L - 50, 290, 40, 14, 'dim_label')
    d.text('N x D', R + 10, 290, 40, 14, 'dim_label')
    loss = d.node('Loss: push C to I<br>Σ<sub>i</sub>(1−C<sub>ii</sub>)² + λ Σ<sub>i≠j</sub>C<sub>ij</sub>²',
                  165, 405, 180, 40, 'loss')
    d.link(c, loss)


# ---------------------------------------------------------------- Triplet Network
@register(CAT, 'Triplet Network')
def triplet(d):
    P, A, N, W, H = 100, 280, 460, 130, 30
    cols = []
    for cx, inp, emb in [(P, 'Positive x<sup>p</sup>', 'f(x<sup>p</sup>)'),
                         (A, 'Anchor x<sup>a</sup>', 'f(x<sup>a</sup>)'),
                         (N, 'Negative x<sup>n</sup>', 'f(x<sup>n</sup>)')]:
        i = d.node(inp, cx - W / 2, 10, W, H, 'in_box')
        f = d.node('Embedding net f', cx - W / 2, 65, W, H, 'module')
        e = d.node(emb, cx - W / 2, 120, W, H, 'embed')
        d.link(i, f)
        d.link(f, e)
        cols.append((i, f, e))
    _shared(d, cols[0][1], cols[1][1])
    _shared(d, cols[1][1], cols[2][1])
    dap = d.node('d(a, p)', 145, 185, 90, H, 'act')
    dan = d.node('d(a, n)', 325, 185, 90, H, 'act')
    ep, ea, en = cols[0][2], cols[1][2], cols[2][2]
    _down_in(d, ep, dap)
    _down_in(d, en, dan)
    d.link(ea, dap, exit='bottom', entry='right', via=[(A, dap.cy)])
    d.link(ea, dan, exit='bottom', entry='left', via=[(A, dan.cy)])
    loss = d.node('Triplet margin loss<br>max(d(a,p) − d(a,n) + α, 0)', 180, 255, 200, 40, 'loss')
    for s in (dap, dan):
        d.link(s, loss, exit='bottom', entry='top', via=[(s.cx, 235), (A, 235)])
