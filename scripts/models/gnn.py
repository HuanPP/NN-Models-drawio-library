"""Graph neural networks."""

from nnlib.core import register

CAT = 'Graph Neural Networks'


@register(CAT, 'GCN Layer')
def gcn_layer(d):
    ah = d.node('Â = D̃<sup>-1/2</sup> (A + I) D̃<sup>-1/2</sup>', 70, 0, 160, 36, 'reshape')
    h = d.node('H<sup>(l)</sup><br>N x F', 0, 80, 80, 36, 'in_box')
    mm = d.op('@', 150, 98)
    w = d.node('Linear W<sup>(l)</sup>', 200, 85, 110, 26, 'fc_box')
    s = d.node('σ', 340, 85, 50, 26, 'act')
    out = d.node('H<sup>(l+1)</sup><br>N x F′', 420, 80, 90, 36, 'out_box')
    d.link(ah, mm)
    d.link(h, mm, exit='right', entry='left')
    for a, b in [(mm, w), (w, s), (s, out)]:
        d.link(a, b, exit='right', entry='left')
    d.text('H<sup>(l+1)</sup> = σ(Â H<sup>(l)</sup> W<sup>(l)</sup>)', 0, 140, 510, 16, 'label')


@register(CAT, 'GraphSAGE')
def graphsage(d):
    hv = d.node('h<sub>v</sub><sup>k-1</sup>', 0, 0, 120, 30, 'in_box')
    hn = d.node('h<sub>u</sub><sup>k-1</sup>, u ∈ N(v)', 0, 80, 120, 36, 'in_box')
    sm = d.node('Sample<br>neighbours', 145, 80, 100, 36, 'gate')
    ag = d.node('Aggregate<br>(mean / LSTM / pool)', 270, 80, 140, 36, 'pool_box')
    cat = d.op('C', 450, 98)
    w = d.node('W<sup>k</sup>, σ', 490, 85, 90, 26, 'fc_box')
    nm = d.node('ℓ2 norm', 605, 85, 80, 26, 'norm')
    out = d.node('h<sub>v</sub><sup>k</sup>', 710, 80, 80, 36, 'out_box')
    for a, b in [(hn, sm), (sm, ag), (ag, cat), (cat, w), (w, nm), (nm, out)]:
        d.link(a, b, exit='right', entry='left')
    d.link(hv, cat, exit='right', entry='top', via=[(cat.cx, hv.cy)])
    d.text('self', 330, 20, 40, 14, 'dim_label')


@register(CAT, 'GIN')
def gin(d):
    hv = d.node('h<sub>v</sub><sup>(k-1)</sup>', 0, 0, 110, 30, 'in_box')
    sc = d.node('(1 + ε) ·', 140, 0, 90, 30, 'act')
    hu = d.node('h<sub>u</sub>, u ∈ N(v)', 0, 60, 110, 30, 'in_box')
    sm = d.node('Σ<sub>u</sub>', 140, 60, 90, 30, 'pool_box')
    add = d.op('+', 270, 45)
    mlp = d.node('MLP', 310, 32, 90, 26, 'fc_box')
    out = d.node('h<sub>v</sub><sup>(k)</sup>', 430, 30, 100, 30, 'out_box')
    d.link(hv, sc, exit='right', entry='left')
    d.link(hu, sm, exit='right', entry='left')
    d.link(sc, add, exit='right', entry='top', via=[(add.cx, sc.cy)])
    d.link(sm, add, exit='right', entry='bottom', via=[(add.cx, sm.cy)])
    d.link(add, mlp, exit='right', entry='left')
    d.link(mlp, out, exit='right', entry='left')
    d.container('Graph-level readout', 0, 120, 530, 100, 'container_green')
    r0 = d.node('Σ<sub>v</sub> h<sub>v</sub><sup>(0)</sup>', 20, 160, 100, 30, 'pool_box')
    d.text('···', 130, 160, 40, 30, 'ellipsis')
    rk = d.node('Σ<sub>v</sub> h<sub>v</sub><sup>(K)</sup>', 180, 160, 100, 30, 'pool_box')
    cat = d.op('C', 330, 175)
    hg = d.node('Graph embedding h<sub>G</sub>', 370, 160, 140, 30, 'out_box')
    d.link(rk, cat, exit='right', entry='left')
    d.link(r0, cat, exit='bottom', entry='bottom', via=[(r0.cx, 205), (cat.cx, 205)])
    d.link(cat, hg, exit='right', entry='left')


@register(CAT, 'Message Passing')
def message_passing(d):
    v = d.op('v', 90, 110, 32)
    u1 = d.op('u<sub>1</sub>', 90, 35, 28, 'op_sm')
    u2 = d.op('u<sub>2</sub>', 20, 110, 28, 'op_sm')
    u3 = d.op('u<sub>3</sub>', 90, 185, 28, 'op_sm')
    d.link(u1, v)
    d.link(u2, v, exit='right', entry='left')
    d.link(u3, v, exit='top', entry='bottom')
    d.text('neighbourhood N(v)', 10, 210, 160, 14, 'dim_label')
    msg = d.node('Messages<br>m<sub>vu</sub> = M(h<sub>v</sub>, h<sub>u</sub>, e<sub>vu</sub>)', 160, 85, 190, 50, 'fc_box')
    agg = d.node('Aggregate<br>m<sub>v</sub> = ⊕<sub>u</sub> m<sub>vu</sub>', 380, 85, 130, 50, 'pool_box')
    upd = d.node('Update<br>h′<sub>v</sub> = U(h<sub>v</sub>, m<sub>v</sub>)', 540, 85, 150, 50, 'ssm')
    out = d.node('h′<sub>v</sub>', 720, 92, 70, 36, 'out_box')
    for a, b in [(v, msg), (msg, agg), (agg, upd), (upd, out)]:
        d.link(a, b, exit='right', entry='left')
    d.text('⊕: sum / mean / max', 380, 140, 130, 14, 'dim_label')


@register(CAT, 'Graph Classification Pipeline')
def graph_classification(d):
    g = d.node('Input graph<br>(X, A)', 0, 40, 90, 40, 'in_box')
    d.container('× L', 115, 15, 225, 90, 'container_repeat')
    gl = d.node('GNN layer<br>(message passing)', 130, 40, 110, 40, 'attn')
    ac = d.node('ReLU', 260, 47, 60, 26, 'act')
    emb = d.node('Node<br>embeddings H', 365, 40, 100, 40, 'embed')
    ro = d.node('Readout<br>(sum / mean / attn)', 490, 40, 130, 40, 'pool_box')
    mlp = d.node('MLP', 645, 47, 60, 26, 'fc_box')
    out = d.node('Graph<br>label', 730, 42, 70, 36, 'out_box')
    for a, b in [(g, gl), (gl, ac), (ac, emb), (emb, ro), (ro, mlp), (mlp, out)]:
        d.link(a, b, exit='right', entry='left')


@register(CAT, 'Graph Transformer')
def graph_transformer(d):
    h = d.node('Node features h<sub>i</sub>', 130, 0, 140, 30, 'in_box')
    pe = d.node('Laplacian PE λ<sub>i</sub>', 300, 45, 120, 30, 'pos')
    a0 = d.op('+', 200, 60)
    d.link(h, a0)
    d.link(pe, a0, exit='left', entry='right')
    j1 = d.op('', 200, 100, 8, 'dot')
    d.link(a0, j1, style='line')
    mha = d.node('Multi-Head Attention<br>(over neighbours j ∈ N(i))', 105, 120, 190, 40, 'attn')
    ef = d.node('Edge features<br>e<sub>ij</sub>', 330, 120, 110, 40, 'in_box')
    d.link(j1, mha)
    d.link(ef, mha, exit='left', entry='right')
    a1 = d.op('+', 200, 200)
    n1 = d.node('BatchNorm', 150, 235, 100, 26, 'norm')
    j2 = d.op('', 200, 285, 8, 'dot')
    ff = d.node('FFN', 130, 305, 140, 28, 'fc_box')
    a2 = d.op('+', 200, 370)
    n2 = d.node('BatchNorm', 150, 405, 100, 26, 'norm')
    out = d.node('h′<sub>i</sub>', 140, 455, 120, 30, 'out_box')
    d.link(mha, a1)
    d.link(a1, n1)
    d.link(n1, j2, style='line')
    d.link(j2, ff)
    d.link(ff, a2)
    d.link(a2, n2)
    d.link(n2, out)
    for j, a in [(j1, a1), (j2, a2)]:
        d.link(j, a, exit='left', entry='left', via=[(80, j.cy), (80, a.cy)], style='skip')


@register(CAT, 'ST-GCN')
def st_gcn(d):
    inp = d.node('Skeleton sequence<br>T x V x C', 130, 0, 140, 40, 'in_box')
    d.container('× 9 blocks', 70, 60, 290, 330, 'container_repeat')
    j = d.op('', 200, 80, 8, 'dot')
    gc = d.node('Spatial Graph Conv<br>(over joints, A)', 125, 100, 150, 40, 'attn')
    b1 = d.node('BN, ReLU', 155, 165, 90, 24, 'act')
    tc = d.node('Temporal Conv<br>(9 x 1, over time)', 125, 214, 150, 40, 'conv_box')
    b2 = d.node('BN, ReLU', 155, 279, 90, 24, 'act')
    add = d.op('+', 200, 340)
    d.link(inp, j, style='line')
    for a, b in [(j, gc), (gc, b1), (b1, tc), (tc, b2), (b2, add)]:
        d.link(a, b)
    d.link(j, add, exit='right', entry='right', via=[(320, j.cy), (320, add.cy)], style='skip')
    pool = d.node('Global Avg Pool', 140, 410, 120, 26, 'pool_box')
    fc = d.node('FC + Softmax', 140, 460, 120, 26, 'fc_box')
    out = d.node('Action class', 140, 510, 120, 30, 'out_box')
    d.link(add, pool)
    d.link(pool, fc)
    d.link(fc, out)
