"""Recurrent, convolutional and state-space sequence models."""

from nnlib.core import register

CAT = 'Sequence Models'


@register(CAT, 'LSTM Cell')
def lstm_cell(d):
    d.container('LSTM Cell', 100, -5, 485, 300, 'container_teal')
    cp = d.node('c<sub>t-1</sub>', 0, 25, 70, 30, 'in_box')
    mf = d.op('×', 200, 40)
    ad = d.op('+', 380, 40)
    ct = d.node('c<sub>t</sub>', 620, 25, 70, 30, 'out_box')
    d.link(cp, mf, exit='right', entry='left')
    d.link(mf, ad, exit='right', entry='left')
    d.link(ad, ct, exit='right', entry='left')
    th = d.node('tanh', 515, 90, 50, 26, 'act')
    d.link(ad, th, exit='right', entry='top', via=[(540, 40)])
    gates = {}
    for key, lab, x in [('f', 'σ', 200), ('i', 'σ', 300), ('g', 'tanh', 380), ('o', 'σ', 470)]:
        gates[key] = d.node(lab, x - 25, 200, 50, 26, 'fc_box')
    mi = d.op('×', 380, 130)
    mo = d.op('×', 540, 160)
    d.link(gates['f'], mf)
    d.link(gates['g'], mi)
    d.link(gates['i'], mi, exit='top', entry='left', via=[(300, 130)])
    d.link(mi, ad)
    d.link(th, mo)
    d.link(gates['o'], mo, exit='top', entry='left', via=[(470, 160)])
    ht = d.node('h<sub>t</sub>', 620, 145, 70, 30, 'out_box')
    d.link(mo, ht, exit='right', entry='left')
    c = d.op('C', 130, 270)
    hp = d.node('h<sub>t-1</sub>', 0, 255, 70, 30, 'in_box')
    xt = d.node('x<sub>t</sub>', 95, 320, 70, 30, 'in_box')
    d.link(hp, c, exit='right', entry='left')
    d.link(xt, c, exit='top', entry='bottom')
    for g in gates.values():
        d.link(c, g, exit='right', entry='bottom', via=[(g.cx, c.cy)])
    for txt, x, y in [('f<sub>t</sub>', 205, 160), ('i<sub>t</sub>', 305, 160),
                      ('c̃<sub>t</sub>', 385, 165), ('o<sub>t</sub>', 475, 175)]:
        d.text(txt, x, y, 20, 14, 'dim_label')


@register(CAT, 'GRU Cell')
def gru_cell(d):
    d.container('GRU Cell', 85, -10, 515, 323, 'container_teal')
    hp = d.node('h<sub>t-1</sub>', 0, 25, 70, 30, 'in_box')
    it = d.node('h<sub>t</sub> = (1 − z<sub>t</sub>) ⊙ h<sub>t-1</sub> + z<sub>t</sub> ⊙ h̃<sub>t</sub>',
                380, 20, 210, 40, 'act')
    ht = d.node('h<sub>t</sub>', 640, 25, 70, 30, 'out_box')
    d.link(hp, it, exit='right', entry='left')
    d.link(it, ht, exit='right', entry='left')
    r = d.node('σ', 145, 200, 50, 26, 'fc_box')
    z = d.node('σ', 535, 200, 50, 26, 'fc_box')
    hc = d.node('tanh', 440, 117, 60, 26, 'fc_box')
    mr = d.op('×', 170, 130)
    c = d.op('C', 110, 290)
    xt = d.node('x<sub>t</sub>', 75, 340, 70, 30, 'in_box')
    d.link(xt, c, exit='top', entry='bottom')
    d.link(hp, c, exit='bottom', entry='left', via=[(35, c.cy)])
    d.link(hp, mr, exit='bottom', entry='left', via=[(35, mr.cy)])
    for g in (r, hc, z):
        d.link(c, g, exit='right', entry='bottom', via=[(g.cx, c.cy)])
    d.link(r, mr)
    d.link(mr, hc, exit='right', entry='left')
    d.link(hc, it, exit='top', entry='bottom', entry_pos=(hc.cx - it.x) / it.w)
    d.link(z, it, exit='top', entry='bottom', entry_pos=(z.cx - it.x) / it.w)
    for txt, x, y in [('r<sub>t</sub>', 175, 165), ('h̃<sub>t</sub>', 475, 80), ('z<sub>t</sub>', 565, 120)]:
        d.text(txt, x, y, 20, 14, 'dim_label')


@register(CAT, 'TCN Residual Block')
def tcn_block(d):
    cx, W = 200, 160
    inp = d.node('Input (C<sub>in</sub> x T)', cx - 70, 0, 140, 30, 'in_box')
    d.container('TCN Block', 95, 45, 320, 535, 'container_blue')
    j = d.op('', cx, 82, 8, 'dot')
    seq = [j]
    for k, (lab, y, h, st) in enumerate([
            ('Dilated Causal Conv', 100, 28, 'conv_box'), ('WeightNorm', 152, 26, 'norm'),
            ('ReLU', 202, 24, 'act'), ('Dropout', 250, 24, 'act'),
            ('Dilated Causal Conv', 298, 28, 'conv_box'), ('WeightNorm', 350, 26, 'norm'),
            ('ReLU', 400, 24, 'act'), ('Dropout', 448, 24, 'act')]):
        w = W if st in ('conv_box',) else (110 if st == 'norm' else 90)
        seq.append(d.node(lab, cx - w / 2, y, w, h, st))
    add = d.op('+', cx, 510)
    r3 = d.node('ReLU', cx - 45, 545, 90, 24, 'act')
    seq += [add, r3]
    d.link(inp, j, style='line')
    for a, b in zip(seq, seq[1:]):
        d.link(a, b)
    sc = d.node('1x1 Conv<br>(if C<sub>in</sub> ≠ C<sub>out</sub>)', 300, 290, 100, 36, 'conv_box')
    d.link(j, sc, exit='right', entry='top', via=[(350, j.cy)], style='skip')
    d.link(sc, add, exit='bottom', entry='right', via=[(350, add.cy)], style='skip')
    out = d.node('Output (C<sub>out</sub> x T)', cx - 70, 600, 140, 30, 'out_box')
    d.link(r3, out)


@register(CAT, 'WaveNet')
def wavenet(d):
    inp = d.node('Input x', 120, 0, 120, 30, 'in_box')
    cc = d.node('Causal Conv', 120, 55, 120, 28, 'conv_box')
    d.link(inp, cc)
    d.container('× K layers', 55, 105, 305, 320, 'container_repeat')
    j = d.op('', 180, 130, 8, 'dot')
    d.link(cc, j, style='line')
    dil = d.node('Dilated Conv 2x1<br>(dilation 2<sup>k</sup>)', 110, 150, 140, 40, 'conv_box')
    th = d.node('tanh', 95, 225, 70, 26, 'act')
    sg = d.node('σ', 195, 225, 70, 26, 'act')
    mul = d.op('×', 180, 290)
    c11 = d.node('1x1 Conv', 130, 325, 100, 26, 'conv_box')
    add = d.op('+', 180, 385)
    d.link(j, dil)
    d.link(dil, th, exit='bottom', entry='top', via=[(180, 207), (th.cx, 207)])
    d.link(dil, sg, exit='bottom', entry='top', via=[(180, 207), (sg.cx, 207)])
    d.link(th, mul, exit='bottom', entry='left', via=[(th.cx, mul.cy)])
    d.link(sg, mul, exit='bottom', entry='right', via=[(sg.cx, mul.cy)])
    d.link(mul, c11)
    d.link(c11, add)
    d.link(j, add, exit='left', entry='left', via=[(75, j.cy), (75, add.cy)], style='skip')
    nxt = d.text('to next layer', 130, 440, 100, 14, 'dim_label')
    d.link(add, nxt)
    sm = d.op('+', 430, 130)
    d.link(c11, sm, exit='right', entry='left', via=[(330, c11.cy), (330, sm.cy)])
    d.text('skip', 295, 250, 30, 14, 'dim_label')
    chain = [sm]
    for lab, y, w, h, st in [('ReLU', 170, 80, 24, 'act'), ('1x1 Conv', 218, 100, 26, 'conv_box'),
                             ('ReLU', 268, 80, 24, 'act'), ('1x1 Conv', 316, 100, 26, 'conv_box'),
                             ('Softmax', 366, 100, 26, 'softmax'), ('Output p(x<sub>t</sub> | x<sub>&lt;t</sub>)', 415, 140, 30, 'out_box')]:
        chain.append(d.node(lab, 430 - w / 2, y, w, h, st))
    for a, b in zip(chain, chain[1:]):
        d.link(a, b)


@register(CAT, 'Conformer Block')
def conformer(d):
    cx, W = 150, 180
    seq = []
    inp = d.node('Input', cx - 60, 0, 120, 30, 'in_box')
    y = 52
    blocks = [('Feed Forward Module', 'fc_box', True), ('Multi-Head Self-Attention<br>(relative pos. enc.)', 'attn', False),
              ('Convolution Module', 'conv_box', False), ('Feed Forward Module', 'fc_box', True)]
    prev = inp
    conv_node = None
    for lab, st, half in blocks:
        j = d.op('', cx, y, 8, 'dot')
        d.link(prev, j, style='line')
        h = 38 if '<br>' in lab else 28
        b = d.node(lab, cx - W / 2, y + 18, W, h, st)
        d.link(j, b)
        last = b
        yy = y + 18 + h
        if half:
            hf = d.op('½', cx, yy + 25, 26, 'op_sm')
            d.link(b, hf)
            last = hf
            yy += 38
        a = d.op('+', cx, yy + 32)
        d.link(last, a)
        d.link(j, a, exit='left', entry='left', via=[(40, j.cy), (40, a.cy)], style='skip')
        if st == 'conv_box':
            conv_node = b
        prev = a
        y = yy + 47 + 20
    ln = d.node('LayerNorm', cx - 55, y + 5, 110, 26, 'norm')
    d.link(prev, ln)
    out = d.node('Output', cx - 60, y + 55, 120, 30, 'out_box')
    d.link(ln, out)
    # convolution module detail
    top = 230
    cont = d.container('Convolution Module', 300, top, 200, 355, 'container_blue')
    ops = [('LayerNorm', 'norm'), ('Pointwise Conv (2C)', 'conv_box'), ('GLU', 'act'),
           ('1D Depthwise Conv', 'conv_box'), ('BatchNorm', 'norm'), ('Swish', 'act'),
           ('Pointwise Conv', 'conv_box'), ('Dropout', 'act')]
    nodes = [d.node(lab, 320, top + 35 + 40 * k, 160, 24, st) for k, (lab, st) in enumerate(ops)]
    for a, b in zip(nodes, nodes[1:]):
        d.link(a, b)
    d.link(conv_node, cont, exit='right', entry='left', style='skip_line',
           entry_pos=(conv_node.cy - top) / 355)


@register(CAT, 'Mamba Block')
def mamba_block(d):
    inp = d.node('Input', 140, -30, 120, 30, 'in_box')
    d.container('Mamba Block', 30, 15, 330, 450, 'container_teal')
    j = d.op('', 200, 55, 8, 'dot')
    nm = d.node('RMSNorm', 150, 72, 100, 26, 'norm')
    d.link(inp, j, style='line')
    d.link(j, nm)
    l1 = d.node('Linear (expand)', 75, 130, 110, 26, 'fc_box')
    l2 = d.node('Linear (expand)', 235, 130, 110, 26, 'fc_box')
    d.link(nm, l1, exit='bottom', entry='top', via=[(200, 114), (l1.cx, 114)])
    d.link(nm, l2, exit='bottom', entry='top', via=[(200, 114), (l2.cx, 114)])
    cv = d.node('Conv1d', 80, 178, 100, 26, 'conv_box')
    s1 = d.node('SiLU', 90, 226, 80, 24, 'act')
    ssm = d.node('Selective SSM', 75, 272, 110, 30, 'ssm')
    s2 = d.node('SiLU', 250, 178, 80, 24, 'act')
    for a, b in [(l1, cv), (cv, s1), (s1, ssm), (l2, s2)]:
        d.link(a, b)
    mul = d.op('×', 200, 340)
    d.link(ssm, mul, exit='bottom', entry='left', via=[(ssm.cx, mul.cy)])
    d.link(s2, mul, exit='bottom', entry='right', via=[(s2.cx, mul.cy)])
    op = d.node('Linear (project)', 145, 375, 110, 26, 'fc_box')
    add = d.op('+', 200, 438)
    out = d.node('Output', 140, 485, 120, 30, 'out_box')
    d.link(mul, op)
    d.link(op, add)
    d.link(add, out)
    d.link(j, add, exit='left', entry='left', via=[(50, j.cy), (50, add.cy)], style='skip')
    d.text('gate', 295, 250, 40, 14, 'dim_label')


@register(CAT, 'Selective SSM')
def selective_ssm(d):
    xt = d.node('x<sub>t</sub>', 0, 150, 70, 30, 'in_box')
    pd = d.node('Linear → Δ<sub>t</sub><br>(softplus)', 110, 40, 120, 40, 'fc_box')
    pb = d.node('Linear → B<sub>t</sub>', 110, 152, 120, 26, 'fc_box')
    pc = d.node('Linear → C<sub>t</sub>', 110, 257, 120, 26, 'fc_box')
    for p in (pd, pb, pc):
        d.link(xt, p, exit='right', entry='left', via=[(90, xt.cy), (90, p.cy)])
    a = d.node('A (learned)', 305, 20, 100, 28, 'pos')
    disc = d.node('Discretize (ZOH)<br>Ā = exp(Δ A),  B̄ = Δ B', 270, 85, 170, 50, 'gate')
    d.link(a, disc)
    d.link(pd, disc, exit='right', entry='left', via=[(250, pd.cy), (250, 100)], entry_pos=0.3)
    d.link(pb, disc, exit='right', entry='left', via=[(250, pb.cy), (250, 120)], entry_pos=0.7)
    rec = d.node('h<sub>t</sub> = Ā h<sub>t-1</sub> + B̄ x<sub>t</sub>', 490, 85, 170, 50, 'ssm')
    d.link(disc, rec, exit='right', entry='left')
    d.link(xt, rec, exit='top', entry='top', via=[(xt.cx, 0), (rec.cx, 0)])
    st = d.node('State<br>h<sub>t-1</sub>', 710, 80, 80, 60, 'db')
    d.link(rec, st, exit='right', entry='left', exit_pos=0.3, entry_pos=(100 - 80) / 60)
    d.link(st, rec, exit='left', entry='right', exit_pos=(120 - 80) / 60, entry_pos=0.7)
    y = d.node('y<sub>t</sub> = C<sub>t</sub> h<sub>t</sub>', 490, 250, 170, 40, 'act')
    d.link(rec, y)
    d.link(pc, y, exit='right', entry='left')
    out = d.node('y<sub>t</sub>', 710, 255, 80, 30, 'out_box')
    d.link(y, out, exit='right', entry='left')
    d.text('Δ, B, C depend on the input (selection); training uses a hardware-aware parallel scan',
           0, 310, 790, 14, 'dim_label')
