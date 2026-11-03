"""Semantic / instance segmentation architectures."""

from nnlib.core import register

CAT = 'Segmentation'


@register(CAT, 'DeepLabv3+')
def deeplabv3plus(d):
    d.container('Encoder', 0, 0, 640, 260, 'container_blue')
    img = d.node('Image', 20, 110, 70, 40, 'in_box')
    dcnn = d.node('DCNN<br>(atrous conv)', 120, 95, 110, 70, 'module')
    d.link(img, dcnn, exit='right', entry='left')
    labels = ['1x1 Conv', '3x3 Conv, rate 6', '3x3 Conv, rate 12', '3x3 Conv, rate 18', 'Image Pooling']
    branches = []
    for k, lab in enumerate(labels):
        b = d.node(lab, 270, 52 + 36 * k, 130, 26, 'pool_box' if k == 4 else 'conv_box')
        branches.append(b)
        d.link(dcnn, b, exit='right', entry='left', via=[(250, dcnn.cy), (250, b.cy)])
    cat = d.op('C', 445, 130)
    for b in branches:
        d.link(b, cat, exit='right', entry='left', via=[(420, b.cy), (420, cat.cy)])
    c1 = d.node('1x1 Conv', 485, 117, 90, 26, 'conv_box')
    d.link(cat, c1, exit='right', entry='left')
    d.text('ASPP', 270, 30, 130, 16, 'label')

    d.container('Decoder', 0, 280, 640, 150, 'container_green')
    low = d.node('1x1 Conv', 120, 330, 110, 26, 'conv_box')
    d.link(dcnn, low)
    d.text('low-level features', 182, 300, 100, 14, 'dim_label')
    up1 = d.node('Upsample x4', 485, 330, 90, 26, 'deconv_box')
    d.link(c1, up1)
    cat2 = d.op('C', 335, low.cy)
    d.link(low, cat2, exit='right', entry='left')
    d.link(up1, cat2, exit='left', entry='right')
    c3 = d.node('3x3 Conv', 280, 385, 110, 26, 'conv_box')
    up2 = d.node('Upsample x4', 415, 385, 100, 26, 'deconv_box')
    pred = d.node('Prediction', 540, 380, 80, 36, 'out_box')
    d.link(cat2, c3)
    d.link(c3, up2, exit='right', entry='left')
    d.link(up2, pred, exit='right', entry='left')


@register(CAT, 'PSPNet')
def pspnet(d):
    cy = 160
    img = d.node('Image', 0, cy - 20, 70, 40, 'in_box')
    cnn = d.node('CNN', 100, cy - 30, 80, 60, 'module')
    fm = d.node('Feature<br>Map', 195, cy - 25, 70, 50, 'feature')
    d.link(img, cnn, exit='right', entry='left')
    d.link(cnn, fm, exit='right', entry='left')
    d.container('Pyramid Pooling Module', 300, 40, 285, 220, 'container_purple')
    cat = d.op('C', 630, cy)
    for k, (bins, cyk) in enumerate(zip(['1x1', '2x2', '3x3', '6x6'], [85, 135, 185, 235])):
        p = d.node(f'Pool {bins}', 315, cyk - 13, 70, 26, 'pool_box')
        c = d.node('Conv 1x1', 410, cyk - 13, 70, 26, 'conv_box')
        u = d.node('Upsample', 505, cyk - 13, 70, 26, 'deconv_box')
        d.link(fm, p, exit='right', entry='left', via=[(282, cy), (282, cyk)])
        d.link(p, c, exit='right', entry='left')
        d.link(c, u, exit='right', entry='left')
        d.link(u, cat, exit='right', entry='left', via=[(605, cyk), (605, cy)])
    d.link(fm, cat, exit='top', entry='top', via=[(fm.cx, 15), (630, 15)])
    conv = d.node('Conv', 670, cy - 13, 70, 26, 'conv_box')
    pred = d.node('Prediction', 765, cy - 18, 80, 36, 'out_box')
    d.link(cat, conv, exit='right', entry='left')
    d.link(conv, pred, exit='right', entry='left')


@register(CAT, 'SAM')
def sam(d):
    img = d.node('Image', 0, 40, 80, 40, 'in_box')
    enc = d.node('Image Encoder<br>(ViT, MAE pre-trained)', 110, 35, 160, 50, 'module')
    emb = d.node('Image<br>Embedding', 300, 35, 80, 50, 'feature')
    add = d.op('+', 415, 60)
    dec = d.node('Lightweight<br>Mask Decoder', 475, 30, 130, 60, 'attn')
    m = d.node('Masks', 655, 15, 100, 30, 'out_box')
    iou = d.node('IoU Scores', 655, 75, 100, 30, 'out_box')
    for a, b in [(img, enc), (enc, emb), (emb, add), (add, dec)]:
        d.link(a, b, exit='right', entry='left')
    d.link(dec, m, exit='right', entry='left', via=[(630, dec.cy), (630, m.cy)])
    d.link(dec, iou, exit='right', entry='left', via=[(630, dec.cy), (630, iou.cy)])
    mk = d.node('Mask prompt', 370, 180, 90, 30, 'in_box')
    conv = d.node('Conv', 375, 125, 80, 26, 'conv_box')
    d.link(mk, conv)
    d.link(conv, add)
    pr = d.node('Points / Box / Text', 475, 180, 130, 30, 'in_box')
    pe = d.node('Prompt Encoder', 475, 125, 130, 26, 'embed')
    d.link(pr, pe)
    d.link(pe, dec)
    d.text('dense', 420, 98, 40, 14, 'dim_label')
    d.text('sparse tokens', 545, 98, 70, 14, 'dim_label')


@register(CAT, 'Mask2Former')
def mask2former(d):
    img = d.node('Image', 0, 0, 120, 30, 'in_box')
    bb = d.node('Backbone', 0, 55, 120, 36, 'module')
    pd = d.node('Pixel<br>Decoder', 0, 120, 120, 200, 'module_alt')
    pe = d.node('Per-pixel<br>Embeddings (1/4)', 0, 350, 120, 40, 'embed')
    d.link(img, bb)
    d.link(bb, pd)
    d.link(pd, pe)
    q = d.node('N learned queries', 250, 70, 150, 30, 'embed')
    d.container('× L / 3', 225, 115, 200, 205, 'container_repeat')
    layers = []
    for k, (scale, y) in enumerate([('1/32', 132), ('1/16', 192), ('1/8', 252)]):
        lay = d.node('Masked Attn, Self-Attn,<br>FFN', 240, y, 170, 36, 'attn')
        layers.append(lay)
        d.link(pd, lay, exit='right', entry='left', exit_pos=(lay.cy - pd.y) / pd.h)
        d.text(scale, 150, lay.cy - 17, 40, 14, 'dim_label')
    d.link(q, layers[0])
    d.link(layers[0], layers[1])
    d.link(layers[1], layers[2])
    qe = d.node('Query Embeddings', 255, 340, 140, 30, 'embed')
    d.link(layers[2], qe)
    cls = d.node('Class logits<br>N x (K + 1)', 460, 337, 130, 36, 'out_box')
    d.link(qe, cls, exit='right', entry='left')
    mlp = d.node('MLP', 265, 395, 120, 26, 'fc_box')
    d.link(qe, mlp)
    dot = d.op('@', 325, 455)
    d.link(mlp, dot)
    d.link(pe, dot, exit='bottom', entry='left', via=[(pe.cx, dot.cy)])
    mk = d.node('Masks<br>N x H/4 x W/4', 255, 495, 140, 36, 'out_box')
    d.link(dot, mk)
    d.text('multi-scale<br>features,<br>round robin', 125, 85, 95, 40, 'dim_label')


@register(CAT, 'SegFormer')
def segformer(d):
    img = d.node('Image H x W x 3', 10, 0, 170, 30, 'in_box')
    stages = []
    for k, cy in enumerate([80, 140, 200, 260]):
        s = d.node(f'Stage {k + 1}: Overlap PE +<br>Transformer Block × N<sub>{k + 1}</sub>',
                   10, cy - 20, 170, 40, 'attn')
        stages.append(s)
    d.link(img, stages[0])
    for a, b in zip(stages, stages[1:]):
        d.link(a, b)
    cat = d.op('C', 440, 170)
    for k, s in enumerate(stages):
        m = d.node('MLP', 220, s.cy - 13, 70, 26, 'fc_box')
        u = d.node(f'Up x{2 ** k}', 315, s.cy - 13, 70, 26, 'deconv_box')
        d.link(s, m, exit='right', entry='left')
        d.link(m, u, exit='right', entry='left')
        d.link(u, cat, exit='right', entry='left', via=[(410, s.cy), (410, cat.cy)])
        d.text(f'H/{4 * 2 ** k}', 183, s.cy - 17, 34, 14, 'dim_label')
    f1 = d.node('MLP (fuse)', 480, 157, 90, 26, 'fc_box')
    f2 = d.node('MLP', 600, 157, 70, 26, 'fc_box')
    out = d.node('Mask<br>H/4 x W/4 x N<sub>cls</sub>', 700, 150, 120, 40, 'out_box')
    for a, b in [(cat, f1), (f1, f2), (f2, out)]:
        d.link(a, b, exit='right', entry='left')
    d.text('Hierarchical Encoder (MiT)', 10, 295, 170, 16, 'caption')
    d.text('All-MLP Decoder', 220, 295, 280, 16, 'caption')


@register(CAT, 'Attention U-Net')
def attention_unet(d):
    xl = d.node('x<sup>l</sup> (skip)', 0, 30, 90, 30, 'in_box')
    g = d.node('g (gating)', 0, 120, 90, 30, 'in_box')
    wx = d.node('W<sub>x</sub>: 1x1 Conv', 120, 30, 100, 30, 'conv_box')
    wg = d.node('W<sub>g</sub>: 1x1 Conv', 120, 120, 100, 30, 'conv_box')
    add = d.op('+', 260, 90)
    d.link(xl, wx, exit='right', entry='left')
    d.link(g, wg, exit='right', entry='left')
    d.link(wx, add, exit='right', entry='top', via=[(260, wx.cy)])
    d.link(wg, add, exit='right', entry='bottom', via=[(260, wg.cy)])
    relu = d.node('ReLU', 290, 77, 60, 26, 'act')
    psi = d.node('ψ: 1x1 Conv', 375, 77, 85, 26, 'conv_box')
    sig = d.node('Sigmoid', 485, 77, 70, 26, 'act')
    rs = d.node('Resampler', 580, 77, 80, 26, 'reshape')
    mul = d.op('×', 700, 90)
    out = d.node('x̂<sup>l</sup>', 740, 72, 70, 36, 'out_box')
    for a, b in [(add, relu), (relu, psi), (psi, sig), (sig, rs), (rs, mul), (mul, out)]:
        d.link(a, b, exit='right', entry='left')
    d.text('α', 665, 70, 16, 14, 'label')
    d.link(xl, mul, exit='top', entry='top', via=[(xl.cx, 5), (700, 5)])
    d.text('Attention gate: inserted on every skip connection of a U-Net; g comes from the coarser decoder level',
           0, 165, 810, 14, 'dim_label')


@register(CAT, 'UNet++')
def unet_pp(d):
    H = 55                                  # half column
    W, Ht = 56, 32

    def cx(i, j):
        return 130 + H * (2 * j + i)

    def cy(i):
        return 40 + 85 * i

    X = {}
    for i in range(5):
        for j in range(5 - i):
            style = 'conv_box' if j == 0 else 'deconv_box'
            X[i, j] = d.node(f'X<sup>{i},{j}</sup>', cx(i, j) - W / 2, cy(i) - Ht / 2, W, Ht, style)
    off = 0.2 * W
    for i in range(4):
        mid = cy(i) + 42.5
        a, b = X[i, 0], X[i + 1, 0]
        d.link(a, b, exit='bottom', entry='top', entry_pos=0.3,
               via=[(a.cx, mid), (b.cx - off, mid)])
        for j in range(1, 5 - i):
            s, t = X[i + 1, j - 1], X[i, j]
            d.link(s, t, exit='top', entry='bottom', exit_pos=0.7,
                   via=[(s.cx + off, mid), (t.cx, mid)])
    for i in range(5):
        for j in range(1, 5 - i):
            d.link(X[i, j - 1], X[i, j], exit='right', entry='left', style='skip')
    inp = d.node('Input', 0, cy(0) - 15, 70, 30, 'in_box')
    d.link(inp, X[0, 0], exit='right', entry='left')
    out = d.node('Output', cx(0, 4) + 60, cy(0) - 15, 70, 30, 'out_box')
    d.link(X[0, 4], out, exit='right', entry='left')
    d.text('solid: down-sampling / up-sampling;  dashed: dense skip pathway '
           '(X<sup>i,j</sup> concatenates all X<sup>i,k</sup>, k &lt; j)', 0, 420, 700, 14, 'dim_label')
    d.text('deep supervision on X<sup>0,1</sup> ... X<sup>0,4</sup>', 0, 440, 700, 14, 'dim_label')
