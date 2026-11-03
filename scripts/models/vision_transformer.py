"""Vision Transformer family: hierarchical, distilled, pyramid and latent-bottleneck variants."""

from nnlib.core import register

CAT = 'Vision Transformers'


@register(CAT, 'Swin Transformer')
def swin_transformer(d):
    """Swin-T hierarchy (Liu et al., ICCV 2021, Fig. 3a): four stages, left to right."""
    bw, bh = 110, 36          # stage boxes
    top_y, bot_y = 35, 95     # rows inside a stage container
    cont_w, cont_h = 140, 165
    gap = 40
    dim_y = cont_h + 7

    # input column (no container): Images -> Patch Partition
    img = d.node('Images<br>H x W x 3', 0, top_y, 100, bh, 'in_box')
    part = d.node('Patch Partition', 0, bot_y, 100, bh, 'reshape')
    d.link(img, part)
    d.text('H/4 x W/4 x 48', 0, dim_y, 100, 14, 'dim_label')

    stages = [
        ('Stage 1', 'Linear Embedding', '2', 'H/4 x W/4 x C'),
        ('Stage 2', 'Patch Merging', '2', 'H/8 x W/8 x 2C'),
        ('Stage 3', 'Patch Merging', '6', 'H/16 x W/16 x 4C'),
        ('Stage 4', 'Patch Merging', '2', 'H/32 x W/32 x 8C'),
    ]
    prev = part
    x = 100 + gap
    for title, first, reps, dims in stages:
        d.container(title, x, 0, cont_w, cont_h, 'container_blue')
        bx = x + (cont_w - bw) / 2
        style = 'embed' if first.startswith('Linear') else 'pool_box'
        a = d.node(first, bx, top_y, bw, bh, style)
        b = d.node('Swin Transformer<br>Block', bx, bot_y, bw, bh, 'attn')
        d.link(a, b)
        d.text(f'× {reps}', bx, bot_y + bh + 4, bw, 16, 'label')
        d.text(dims, x, dim_y, cont_w, 14, 'dim_label')
        # step up from the previous column's bottom row into this stage's top row
        mid = x - gap / 2
        d.link(prev, a, exit='right', entry='left',
               via=[(mid, prev.cy), (mid, a.cy)])
        prev = b
        x += cont_w + gap


@register(CAT, 'DeiT')
def deit(d):
    """DeiT (Touvron et al., ICML 2021, Fig. 2): class + distillation tokens, hard distillation."""
    cL, cC, cR = 180, 330, 480          # column centres: class / patches / distillation
    cT, cLL = 625, 35                   # teacher column, label column
    bw = 110

    img = d.node('Image', cC - 50, 0, 100, 30, 'in_box')
    cls = d.node('Class<br>Token', cL - bw / 2, 55, bw, 40, 'embed')
    pe = d.node('Patch<br>Embedding', cC - 60, 55, 120, 40, 'embed')
    dst = d.node('Distillation<br>Token', cR - bw / 2, 55, bw, 40, 'embed')
    cat = d.op('C', cC, 135)
    add = d.op('+', cC, 190)
    pos = d.node('Pos. Embedding', cL - bw / 2, 175, bw, 30, 'pos')
    enc = d.node('Transformer Encoder × L<br>(Self-Attention + FFN)', 80, 230, 500, 40, 'module_gen')
    hc = d.node('Class Head', cL - bw / 2, 295, bw, 30, 'fc_box')
    hd = d.node('Distill. Head', cR - bw / 2, 295, bw, 30, 'fc_box')
    lce = d.node('Cross-Entropy<br>Loss', cL - bw / 2, 350, bw, 40, 'loss')
    ld = d.node('Distillation<br>Loss', cR - bw / 2, 350, bw, 40, 'loss')
    lab = d.node('True Label<br>y', cLL - bw / 2, 350, bw, 40, 'in_box')
    tea = d.node('Teacher CNN<br>(frozen)', cT - bw / 2, 350, bw, 40, 'frozen')

    d.link(img, pe)
    d.link(img, tea, exit='right', entry='top', via=[(cT, img.cy)])
    d.link(pe, cat)
    d.link(cls, cat, exit='bottom', entry='left', via=[(cL, cat.cy)])
    d.link(dst, cat, exit='bottom', entry='right', via=[(cR, cat.cy)])
    d.link(cat, add)
    d.link(pos, add, exit='right', entry='left')
    d.link(add, enc)
    d.link(enc, hc, exit='bottom', entry='top', exit_pos=0.2)
    d.link(enc, hd, exit='bottom', entry='top', exit_pos=0.8)
    d.link(hc, lce)
    d.link(hd, ld)
    d.link(lab, lce, exit='right', entry='left')
    d.link(tea, ld, exit='right' if False else 'left', entry='right')
    d.text('y<sub>t</sub>', 540, 346, 25, 14, 'dim_label')
    d.text('Z<sub>s</sub>', cL + 5, 330, 30, 14, 'dim_label')
    d.text('Z<sub>t</sub>', cR + 5, 330, 30, 14, 'dim_label')


@register(CAT, 'PVT Stage')
def pvt_stage(d):
    """One PVT stage (Wang et al., ICCV 2021, Fig. 3/4) with Spatial-Reduction Attention."""
    M = 200                  # main column centre
    qx, kvx = 140, 260       # Q path / spatial-reduction path
    mw = 120                 # main box width
    skip = 355

    inp = d.node('F<sub>i-1</sub>: H<sub>i-1</sub> x W<sub>i-1</sub> x C<sub>i-1</sub>',
                 M - 80, 0, 160, 30, 'in_box')
    pe = d.node('Patch Embedding', M - mw / 2, 50, mw, 30, 'embed')
    addp = d.op('+', M, 115)
    pos = d.node('Pos. Embedding', 45, 100, 110, 30, 'pos')

    d.container('× L<sub>i</sub>', 50, 145, 330, 515, 'container_repeat')
    dot1 = d.op('', M, 165, 6, 'dot')
    ln1 = d.node('LN', M - 45, 185, 90, 28, 'norm')

    d.container('SRA', 65, 223, 270, 221, 'container_purple')
    rs = d.node('Reshape (R<sub>i</sub>)', kvx - 50, 255, 100, 28, 'reshape')
    lin = d.node('Linear', kvx - 50, 303, 100, 28, 'fc_box')
    lns = d.node('LN', kvx - 50, 351, 100, 28, 'norm')
    mha = d.node('Multi-Head Attention', 80, 399, 240, 30, 'attn')

    add1 = d.op('+', M, 470)
    dot2 = d.op('', M, 497, 6, 'dot')
    ln2 = d.node('LN', M - 45, 517, 90, 28, 'norm')
    ffn = d.node('FFN', M - mw / 2, 565, mw, 28, 'fc_box')
    add2 = d.op('+', M, 628)

    rsh = d.node('Reshape', M - mw / 2, 683, mw, 28, 'reshape')
    out = d.node('F<sub>i</sub>: H<sub>i</sub> x W<sub>i</sub> x C<sub>i</sub>',
                 M - 80, 731, 160, 30, 'out_box')

    d.link(inp, pe)
    d.link(pe, addp)
    d.link(pos, addp, exit='right', entry='left')
    d.link(addp, dot1, style='line')
    d.link(dot1, ln1)
    ys = 235
    d.link(ln1, mha, exit='bottom', entry='top', entry_pos=0.25, via=[(M, ys), (qx, ys)])
    d.link(ln1, rs, exit='bottom', entry='top', via=[(M, ys), (kvx, ys)])
    d.link(rs, lin)
    d.link(lin, lns)
    d.link(lns, mha, exit='bottom', entry='top', entry_pos=0.75)
    d.link(mha, add1)
    d.link(dot1, add1, exit='right', entry='right', via=[(skip, dot1.cy), (skip, add1.cy)], style='skip')
    d.link(add1, dot2, style='line')
    d.link(dot2, ln2)
    d.link(ln2, ffn)
    d.link(ffn, add2)
    d.link(dot2, add2, exit='right', entry='right', via=[(skip, dot2.cy), (skip, add2.cy)], style='skip')
    d.link(add2, rsh)
    d.link(rsh, out)

    d.text('Q', qx + 4, 380, 14, 14, 'label')
    d.text('K, V', kvx + 4, 380, 30, 14, 'label')
    d.text('N<sub>i</sub> x C<sub>i</sub>', M + 6, 83, 50, 14, 'dim_label')
    d.text('(N<sub>i</sub>/R<sub>i</sub><sup>2</sup>) x C<sub>i</sub>', 150, 334, 105, 14, 'dim_label')


@register(CAT, 'Perceiver IO')
def perceiver_io(d):
    """Perceiver IO (Jaegle et al., ICLR 2022, Fig. 2): encode, process, decode."""
    y, h = 100, 40
    cy = y + h / 2
    inp = d.node('Input Array<br>M x C', 0, y, 90, h, 'in_box')
    enc = d.node('Cross-Attention<br>+ MLP', 125, y, 120, h, 'cross_attn')
    lat = d.node('Latent Array<br>N x D', 125, 20, 120, h, 'embed')
    d.container('× L', 280, 80, 150, 95, 'container_repeat')
    proc = d.node('Self-Attention<br>+ MLP', 295, y, 120, h, 'attn')
    dec = d.node('Cross-Attention<br>+ MLP', 465, y, 120, h, 'cross_attn')
    qry = d.node('Output Query<br>Array O x E', 465, 200, 120, h, 'pos')
    out = d.node('Output Array<br>O x E', 620, y, 90, h, 'out_box')

    d.link(inp, enc, exit='right', entry='left')
    d.link(lat, enc)
    d.link(enc, proc, exit='right', entry='left')
    d.link(proc, dec, exit='right', entry='left')
    d.link(qry, dec)
    d.link(dec, out, exit='right', entry='left')

    d.text('K, V', 92, cy - 18, 30, 14, 'label')
    d.text('Q', enc.cx + 5, 70, 14, 14, 'label')
    d.text('K, V', 432, cy - 18, 30, 14, 'label')
    d.text('Q', dec.cx + 5, 155, 14, 14, 'label')
    d.text('N x D', 247, cy + 4, 30, 14, 'dim_label')


@register(CAT, 'MaxViT Block')
def maxvit_block(d):
    """MaxViT block (Tu et al., ECCV 2022, Fig. 2): MBConv -> Block Attention -> Grid Attention."""
    bw, bh = 120, 28
    cw, top = 185, 50
    gap = 40

    def column(x0):
        """Return (box x, centre x, skip x) for a container starting at x0."""
        bx = x0 + 45
        return bx, bx + bw / 2, x0 + 20

    # ---- MBConv
    x1 = 0
    bx, cx, sk = column(x1)
    d.container('MBConv', x1, top, cw, 345, 'container_blue')
    inp = d.node('Input (H x W x C)', bx, 0, bw, 30, 'in_box')
    dot = d.op('', cx, 90, 6, 'dot')
    chain = [d.node('BN', bx + 20, 110, bw - 40, bh, 'norm'),
             d.node('Conv 1x1', bx, 158, bw, bh, 'conv_box'),
             d.node('DW Conv 3x3', bx, 206, bw, bh, 'conv_box'),
             d.node('SE', bx + 20, 254, bw - 40, bh, 'attn'),
             d.node('Conv 1x1', bx, 302, bw, bh, 'conv_box')]
    add = d.op('+', cx, 365)
    d.link(inp, dot, style='line')
    seq = [dot] + chain + [add]
    for a, b in zip(seq, seq[1:]):
        d.link(a, b)
    d.link(dot, add, exit='left', entry='left', via=[(sk, dot.cy), (sk, add.cy)], style='skip')
    prev = add

    # ---- Block / Grid attention
    x0 = x1 + cw + gap
    for name, kind, cstyle in [('Block Attention', 'Block', 'container_purple'),
                               ('Grid Attention', 'Grid', 'container_green')]:
        bx, cx, sk = column(x0)
        d.container(name, x0, top, cw, 436, cstyle)
        part = d.node(f'{kind} Partition', bx, 85, bw, bh, 'reshape')
        dt1 = d.op('', cx, 125, 6, 'dot')
        ln1 = d.node('LN', bx + 20, 142, bw - 40, bh, 'norm')
        att = d.node('Rel-Attention', bx, 190, bw, bh, 'attn')
        a1 = d.op('+', cx, 253)
        dt2 = d.op('', cx, 280, 6, 'dot')
        ln2 = d.node('LN', bx + 20, 297, bw - 40, bh, 'norm')
        ffn = d.node('FFN', bx, 345, bw, bh, 'fc_box')
        a2 = d.op('+', cx, 408)
        rev = d.node(f'{kind} Reverse', bx, 443, bw, bh, 'reshape')
        d.link(part, dt1, style='line')
        d.link(dt1, ln1)
        d.link(ln1, att)
        d.link(att, a1)
        d.link(a1, dt2, style='line')
        d.link(dt2, ln2)
        d.link(ln2, ffn)
        d.link(ffn, a2)
        d.link(a2, rev)
        d.link(dt1, a1, exit='left', entry='left', via=[(sk, dt1.cy), (sk, a1.cy)], style='skip')
        d.link(dt2, a2, exit='left', entry='left', via=[(sk, dt2.cy), (sk, a2.cy)], style='skip')
        mid = x0 - gap / 2
        d.link(prev, part, exit='right', entry='left', via=[(mid, prev.cy), (mid, part.cy)])
        prev = rev
        x0 += cw + gap

    out = d.node('Output (H x W x C)', prev.x, 506, bw, 30, 'out_box')
    d.link(prev, out)


@register(CAT, 'Patch Embedding')
def patch_embedding(d):
    """ViT patch embedding (Dosovitskiy et al., ICLR 2021): patchify, project, prepend [CLS], add positions."""
    cy = 60
    dim_y = 110
    img = d.node('Image', 0, 20, 80, 80, 'in_box')
    d.text('H x W x C', 0, dim_y, 80, 14, 'dim_label')

    s, step = 20, 30
    gx, gy = 110, 20
    cells = []
    for r in range(3):
        for c in range(3):
            cells.append(d.node(str(r * 3 + c + 1), gx + c * step, gy + r * step, s, s, 'in_box'))
    d.text('N = HW/P<sup>2</sup>', gx - 10, dim_y, 100, 14, 'dim_label')

    flat = d.node('Flatten', 220, cy - 15, 70, 30, 'reshape')
    d.text('N x P<sup>2</sup>C', 215, dim_y, 80, 14, 'dim_label')
    lin = d.node('Linear<br>Projection', 320, cy - 20, 90, 40, 'fc_box')
    d.text('N x D', 320, dim_y, 90, 14, 'dim_label')
    cat = d.op('C', 455, cy)
    d.text('(N+1) x D', 425, dim_y, 60, 14, 'dim_label')
    add = d.op('+', 565, cy)
    out = d.node('Embedded<br>Patches', 610, cy - 20, 90, 40, 'out_box')
    d.text('z<sub>0</sub>: (N+1) x D', 600, dim_y, 110, 14, 'dim_label')

    cls = d.node('[CLS]<br>Token', 410, -25, 90, 40, 'embed')
    pos = d.node('Position<br>Embedding', 520, -25, 90, 40, 'pos')
    d.text('1 x D', 410, -45, 90, 14, 'dim_label')
    d.text('(N+1) x D', 520, -45, 90, 14, 'dim_label')

    d.link(img, cells[3], exit='right', entry='left')
    d.link(cells[5], flat, exit='right', entry='left')
    d.link(flat, lin, exit='right', entry='left')
    d.link(lin, cat, exit='right', entry='left')
    d.link(cls, cat)
    d.link(cat, add, exit='right', entry='left')
    d.link(pos, add)
    d.link(add, out, exit='right', entry='left')
