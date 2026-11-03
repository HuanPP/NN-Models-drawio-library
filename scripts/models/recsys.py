"""Click-through-rate prediction and retrieval models for recommendation."""

from nnlib.core import register

CAT = 'Recommender Systems'


@register(CAT, 'Wide & Deep')
def wide_deep(d):
    sp = d.node('Sparse features (categorical, one-hot)', 40, 0, 320, 30, 'in_box')
    cr = d.node('Cross-product<br>transformation', 40, 60, 120, 36, 'gate')
    wl = d.node('Linear (wide)', 40, 210, 120, 26, 'fc_box')
    em = d.node('Dense<br>embeddings', 240, 60, 120, 36, 'embed')
    hs = [d.node(f'ReLU {n}', 240, y, 120, 26, 'fc_box') for n, y in [(1024, 120), (512, 165), (256, 210)]]
    d.link(sp, cr, exit_pos=(cr.cx - sp.x) / sp.w)
    d.link(sp, em, exit_pos=(em.cx - sp.x) / sp.w)
    d.link(cr, wl)
    d.link(em, hs[0])
    for a, b in zip(hs, hs[1:]):
        d.link(a, b)
    out = d.node('Output unit<br>σ(y<sub>wide</sub> + y<sub>deep</sub>)', 130, 280, 140, 36, 'out_box')
    d.link(wl, out, exit='bottom', entry='left', via=[(wl.cx, out.cy)])
    d.link(hs[-1], out, exit='bottom', entry='right', via=[(hs[-1].cx, out.cy)])
    d.text('wide: memorization', 0, 330, 190, 14, 'dim_label')
    d.text('deep: generalization', 210, 330, 190, 14, 'dim_label')


@register(CAT, 'DeepFM')
def deepfm(d):
    sp = d.node('Sparse feature fields', 40, 0, 320, 30, 'in_box')
    em = d.node('Shared embeddings', 130, 55, 140, 30, 'embed')
    d.link(sp, em)
    fm = d.node('FM layer<br>⟨w, x⟩ + Σ<sub>i&lt;j</sub> ⟨v<sub>i</sub>, v<sub>j</sub>⟩ x<sub>i</sub> x<sub>j</sub>',
                10, 115, 180, 50, 'gate')
    h1 = d.node('Hidden (ReLU)', 240, 115, 120, 26, 'fc_box')
    h2 = d.node('Hidden (ReLU)', 240, 160, 120, 26, 'fc_box')
    d.link(em, fm, exit='bottom', entry='top', via=[(200, 100), (fm.cx, 100)])
    d.link(em, h1, exit='bottom', entry='top', via=[(200, 100), (h1.cx, 100)])
    d.link(h1, h2)
    add = d.op('+', 200, 230)
    d.link(fm, add, exit='bottom', entry='left', via=[(fm.cx, add.cy)])
    d.link(h2, add, exit='bottom', entry='right', via=[(h2.cx, add.cy)])
    sg = d.node('σ', 170, 265, 60, 26, 'act')
    out = d.node('CTR ŷ', 140, 315, 120, 30, 'out_box')
    d.link(add, sg)
    d.link(sg, out)


@register(CAT, 'DCN')
def dcn(d):
    inp = d.node('Sparse + dense features', 120, 0, 160, 30, 'in_box')
    em = d.node('Embedding &amp; stacking x<sub>0</sub>', 110, 55, 180, 30, 'embed')
    d.link(inp, em)
    cs = [d.node(f'Cross layer {k}', 50, y, 120, 30, 'gate') for k, y in [(1, 120), (2, 175), (3, 230)]]
    ds = [d.node('FC + ReLU', 250, y, 120, 30, 'fc_box') for y in (120, 175, 230)]
    for c in cs:
        d.link(em, c, exit='left', entry='left', via=[(30, em.cy), (30, c.cy)])
    d.link(em, ds[0], exit='bottom', entry='top', via=[(200, 100), (ds[0].cx, 100)])
    for a, b in zip(cs, cs[1:]):
        d.link(a, b)
    for a, b in zip(ds, ds[1:]):
        d.link(a, b)
    cat = d.op('C', 200, 300)
    d.link(cs[-1], cat, exit='bottom', entry='left', via=[(cs[-1].cx, cat.cy)])
    d.link(ds[-1], cat, exit='bottom', entry='right', via=[(ds[-1].cx, cat.cy)])
    lg = d.node('Linear + σ', 150, 335, 100, 26, 'fc_box')
    out = d.node('CTR ŷ', 140, 385, 120, 30, 'out_box')
    d.link(cat, lg)
    d.link(lg, out)
    d.text('cross: x<sub>l+1</sub> = x<sub>0</sub> x<sub>l</sub><sup>T</sup> w<sub>l</sub> + b<sub>l</sub> + x<sub>l</sub>',
           0, 430, 400, 16, 'label')


@register(CAT, 'DLRM')
def dlrm(d):
    dn = d.node('Dense features', 0, 0, 100, 30, 'in_box')
    s1 = d.node('Sparse 1', 140, 0, 90, 30, 'in_box')
    sk = d.node('Sparse k', 290, 0, 90, 30, 'in_box')
    bm = d.node('Bottom MLP', 0, 55, 100, 30, 'fc_box')
    e1 = d.node('Emb. table 1', 140, 55, 90, 30, 'embed')
    ek = d.node('Emb. table k', 290, 55, 90, 30, 'embed')
    d.text('···', 235, 55, 50, 30, 'ellipsis')
    for a, b in [(dn, bm), (s1, e1), (sk, ek)]:
        d.link(a, b)
    it = d.node('Feature interaction: pairwise dot products,<br>concatenated with the dense vector',
                0, 125, 380, 40, 'gate')
    for n in (bm, e1, ek):
        d.link(n, it, entry_pos=(n.cx - it.x) / it.w)
    tm = d.node('Top MLP', 130, 195, 120, 30, 'fc_box')
    sg = d.node('σ', 160, 250, 60, 26, 'act')
    out = d.node('Click probability', 120, 300, 140, 30, 'out_box')
    d.link(it, tm, exit_pos=(tm.cx - it.x) / it.w)
    d.link(tm, sg)
    d.link(sg, out)


@register(CAT, 'Two-Tower')
def two_tower(d):
    uf = d.node('User features', 40, 0, 120, 30, 'in_box')
    ut = d.node('User tower<br>(MLP)', 40, 55, 120, 40, 'module')
    ue = d.node('User embedding u', 40, 120, 120, 30, 'embed')
    itf = d.node('Item features', 240, 0, 120, 30, 'in_box')
    it = d.node('Item tower<br>(MLP)', 240, 55, 120, 40, 'module_alt')
    ie = d.node('Item embedding v', 240, 120, 120, 30, 'embed')
    for a, b in [(uf, ut), (ut, ue), (itf, it), (it, ie)]:
        d.link(a, b)
    dot = d.op('·', 200, 190)
    d.link(ue, dot, exit='bottom', entry='left', via=[(ue.cx, dot.cy)])
    d.link(ie, dot, exit='bottom', entry='right', via=[(ie.cx, dot.cy)])
    sc = d.node('Score  u · v', 140, 230, 120, 30, 'out_box')
    d.link(dot, sc)
    ann = d.node('ANN index<br>(item embeddings)', 410, 105, 130, 60, 'db')
    d.link(ie, ann, exit='right', entry='left', style='skip')
    d.text('offline', 368, 117, 40, 14, 'dim_label')
    d.text('serving: nearest-neighbour search of u in the index', 0, 275, 540, 14, 'dim_label')


@register(CAT, 'DIN')
def din(d):
    d.text('user behaviour embeddings', 0, 18, 260, 14, 'dim_label')
    b1 = d.node('e<sub>1</sub>', 0, 40, 70, 30, 'embed')
    d.text('···', 85, 40, 90, 30, 'ellipsis')
    bn = d.node('e<sub>N</sub>', 190, 40, 70, 30, 'embed')
    ad = d.node('Candidate<br>ad e<sub>a</sub>', 310, 37, 80, 36, 'embed')
    a1 = d.node('Activation<br>unit', 0, 97, 70, 36, 'gate')
    an = d.node('Activation<br>unit', 190, 97, 70, 36, 'gate')
    d.link(b1, a1)
    d.link(bn, an)
    d.link(ad, an, exit='bottom', entry='right', via=[(ad.cx, an.cy)])
    d.link(ad, a1, exit='top', entry='left', via=[(ad.cx, -15), (-20, -15), (-20, a1.cy)])
    pool = d.node('Weighted sum pooling Σ<sub>i</sub> w<sub>i</sub> e<sub>i</sub>', 0, 170, 260, 30, 'pool_box')
    for a in (a1, an):
        d.link(a, pool, entry_pos=(a.cx - pool.x) / pool.w)
    d.text('w<sub>1</sub>', 40, 145, 20, 14, 'dim_label')
    d.text('w<sub>N</sub>', 230, 145, 24, 14, 'dim_label')
    pf = d.node('Profile +<br>context', 270, 165, 100, 40, 'in_box')
    cat = d.op('C', 320, 250)
    d.link(pool, cat, exit='bottom', entry='left', via=[(130, cat.cy)], exit_pos=0.5)
    d.link(pf, cat)
    d.link(ad, cat, exit='right', entry='right', via=[(420, ad.cy), (420, cat.cy)])
    mlp = d.node('MLP (PReLU / Dice)', 250, 285, 140, 26, 'fc_box')
    out = d.node('CTR ŷ', 260, 335, 120, 30, 'out_box')
    d.link(cat, mlp)
    d.link(mlp, out)
