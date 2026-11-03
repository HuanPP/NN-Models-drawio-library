"""Large language model building blocks: LLaMA-style decoder blocks, attention
variants (GQA, MLA), mixture-of-experts, SwiGLU, RoPE and full-model overviews."""

from nnlib.core import register

CAT = 'Large Language Models'


def _at(n, x):
    """Fraction along the top/bottom side of node n for the absolute x."""
    return (x - n.x) / n.w


def _dot(d, cx, cy):
    return d.op('', cx, cy, 8, 'dot')


# ----------------------------------------------------------------------------
# 1. LLaMA decoder block
# ----------------------------------------------------------------------------
@register(CAT, 'LLaMA Decoder Block')
def llama_block(d):
    cx, sx = 250, 410                      # main column, residual lane
    inp = d.node('Input h<sub>l-1</sub>', cx - 60, 10, 120, 30, 'in_box')
    d.container('× N', 90, 60, 345, 570, 'container_repeat')
    j1 = _dot(d, cx, 85)
    n1 = d.node('RMSNorm', cx - 55, 105, 110, 30, 'norm')
    wq = d.node('W<sub>Q</sub>', 115, 165, 70, 30, 'q')
    wk = d.node('W<sub>K</sub>', 215, 165, 70, 30, 'k')
    wv = d.node('W<sub>V</sub>', 315, 165, 70, 30, 'v')
    rq = d.node('RoPE', 115, 220, 70, 30, 'pos')
    rk = d.node('RoPE', 215, 220, 70, 30, 'pos')
    att = d.node('Grouped-Query Attention (causal)', 115, 275, 270, 30, 'attn')
    wo = d.node('Linear W<sub>O</sub>', cx - 55, 330, 110, 30, 'fc_box')
    a1 = d.op('+', cx, 400)
    j2 = _dot(d, cx, 440)
    n2 = d.node('RMSNorm', cx - 55, 460, 110, 30, 'norm')
    ffn = d.node('SwiGLU FFN', cx - 65, 515, 130, 30, 'fc_box')
    a2 = d.op('+', cx, 580)
    out = d.node('Output h<sub>l</sub>', cx - 60, 650, 120, 30, 'out_box')

    d.link(inp, j1, style='line')
    d.link(j1, n1)
    d.link(n1, wq, via=[(cx, 145), (wq.cx, 145)])
    d.link(n1, wk)
    d.link(n1, wv, via=[(cx, 145), (wv.cx, 145)])
    d.link(wq, rq)
    d.link(wk, rk)
    d.link(rq, att, entry_pos=_at(att, rq.cx))
    d.link(rk, att, entry_pos=_at(att, rk.cx))
    d.link(wv, att, entry_pos=_at(att, wv.cx))
    d.link(att, wo)
    d.link(wo, a1)
    d.link(a1, j2, style='line')
    d.link(j2, n2)
    d.link(n2, ffn)
    d.link(ffn, a2)
    d.link(a2, out)
    d.link(j1, a1, exit='right', entry='right', via=[(sx, j1.cy), (sx, a1.cy)], style='skip')
    d.link(j2, a2, exit='right', entry='right', via=[(sx, j2.cy), (sx, a2.cy)], style='skip')


# ----------------------------------------------------------------------------
# 2. Grouped-query attention (8 query heads, 2 KV heads)
# ----------------------------------------------------------------------------
@register(CAT, 'Grouped-Query Attention')
def gqa(d):
    cx = 345
    x = d.node('Input X', cx - 60, 10, 120, 30, 'in_box')
    heads = []
    for g in range(2):
        x0 = 60 + 330 * g
        row = []
        for i in range(4):
            row.append(d.node(f'Q<sub>{4 * g + i + 1}</sub>', x0 + 48 * i - 18, 80, 36, 30, 'q'))
        row.append(d.node(f'K<sub>{g + 1}</sub>', x0 + 48 * 4 - 18, 80, 36, 30, 'k'))
        row.append(d.node(f'V<sub>{g + 1}</sub>', x0 + 48 * 5 - 18, 80, 36, 30, 'v'))
        heads.append(row)
    groups = []
    for g in range(2):
        a = 4 * g
        gb = d.node(f'Attention, heads {a + 1}-{a + 4}<br>(share K<sub>{g + 1}</sub>, V<sub>{g + 1}</sub>)',
                    42 + 330 * g, 135, 276, 36, 'attn')
        groups.append(gb)
    cat = d.op('C', cx, 215)
    wo = d.node('Linear W<sub>O</sub>', cx - 60, 255, 120, 30, 'fc_box')
    out = d.node('Output', cx - 60, 310, 120, 30, 'out_box')

    for g in range(2):
        for h in heads[g]:
            d.link(x, h, via=[(cx, 60), (h.cx, 60)])
            d.link(h, groups[g], entry_pos=_at(groups[g], h.cx))
    d.link(groups[0], cat, exit='bottom', entry='left', via=[(groups[0].cx, cat.cy)])
    d.link(groups[1], cat, exit='bottom', entry='right', via=[(groups[1].cx, cat.cy)])
    d.link(cat, wo)
    d.link(wo, out)
    d.text('h × d<sub>h</sub>', cx + 8, 232, 50, 16, 'dim_label')


# ----------------------------------------------------------------------------
# 3. Multi-head latent attention (DeepSeek-V2)
# ----------------------------------------------------------------------------
@register(CAT, 'Multi-Head Latent Attention')
def mla(d):
    cx = 320
    cols = {'qc': 80, 'qr': 200, 'kr': 320, 'kc': 440, 'v': 560}
    h = d.node('Hidden h<sub>t</sub>', cx - 60, 10, 120, 30, 'in_box')
    wdq = d.node('W<sup>DQ</sup> (down-proj)', 65, 80, 150, 30, 'fc_box')
    wdkv = d.node('W<sup>DKV</sup> (down-proj)', 425, 80, 150, 30, 'fc_box')
    cq = d.node('Latent c<sup>Q</sup>', 65, 135, 150, 30, 'reshape')
    ckv = d.node('Latent c<sup>KV</sup> (cached)', 425, 135, 150, 30, 'ssm')
    wuq = d.node('W<sup>UQ</sup>', cols['qc'] - 45, 195, 90, 30, 'fc_box')
    wqr = d.node('W<sup>QR</sup>', cols['qr'] - 45, 195, 90, 30, 'fc_box')
    wkr = d.node('W<sup>KR</sup>', cols['kr'] - 45, 195, 90, 30, 'fc_box')
    wuk = d.node('W<sup>UK</sup>', cols['kc'] - 45, 195, 90, 30, 'fc_box')
    wuv = d.node('W<sup>UV</sup>', cols['v'] - 45, 195, 90, 30, 'fc_box')
    rq = d.node('RoPE', cols['qr'] - 45, 250, 90, 30, 'pos')
    rk = d.node('RoPE', cols['kr'] - 45, 250, 90, 30, 'pos')
    cat_q = d.op('C', 140, 315)
    cat_k = d.op('C', 380, 315)
    att = d.node('Multi-Head Attention', 50, 355, 540, 30, 'attn')
    wo = d.node('W<sup>O</sup>', cx - 55, 410, 110, 30, 'fc_box')
    out = d.node('Output u<sub>t</sub>', cx - 60, 465, 120, 30, 'out_box')

    d.link(h, wdq, via=[(cx, 60), (wdq.cx, 60)])
    d.link(h, wdkv, via=[(cx, 60), (wdkv.cx, 60)])
    d.link(h, wkr)
    d.link(wdq, cq)
    d.link(wdkv, ckv)
    d.link(cq, wuq, via=[(cq.cx, 180), (wuq.cx, 180)])
    d.link(cq, wqr, via=[(cq.cx, 180), (wqr.cx, 180)])
    d.link(ckv, wuk, via=[(ckv.cx, 180), (wuk.cx, 180)])
    d.link(ckv, wuv, via=[(ckv.cx, 180), (wuv.cx, 180)])
    d.link(wqr, rq)
    d.link(wkr, rk)
    d.link(wuq, cat_q, exit='bottom', entry='left', via=[(wuq.cx, cat_q.cy)])
    d.link(rq, cat_q, exit='bottom', entry='right', via=[(rq.cx, cat_q.cy)])
    d.link(rk, cat_k, exit='bottom', entry='left', via=[(rk.cx, cat_k.cy)])
    d.link(wuk, cat_k, exit='bottom', entry='right', via=[(wuk.cx, cat_k.cy)])
    d.link(cat_q, att, entry_pos=_at(att, cat_q.cx))
    d.link(cat_k, att, entry_pos=_at(att, cat_k.cx))
    d.link(wuv, att, entry_pos=_at(att, wuv.cx))
    d.link(att, wo)
    d.link(wo, out)

    d.text('q<sup>C</sup>', 85, 262, 30, 16, 'dim_label')
    d.text('q<sup>R</sup>', 205, 288, 30, 16, 'dim_label')
    d.text('k<sup>R</sup> (cached)', 370, 257, 65, 16, 'dim_label')
    d.text('k<sup>C</sup>', 445, 262, 30, 16, 'dim_label')
    d.text('v<sup>C</sup>', 565, 262, 30, 16, 'dim_label')
    d.text('q', 145, 334, 20, 16, 'dim_label')
    d.text('k', 385, 334, 20, 16, 'dim_label')


# ----------------------------------------------------------------------------
# 4. Mixture-of-experts layer with top-k routing and a shared expert
# ----------------------------------------------------------------------------
@register(CAT, 'MoE Layer')
def moe(d):
    cx = 290
    tok = d.node('Token h<sub>t</sub>', cx - 60, 10, 120, 30, 'in_box')
    router = d.node('Router<br>Linear + Softmax', -5, 80, 110, 36, 'gate')
    topk = d.node('Top-k (k = 2)', -5, 140, 110, 30, 'gate')
    e1 = d.node('Expert 1<br>FFN', 135, 80, 70, 36, 'noise')
    e2 = d.node('Expert 2<br>FFN', 215, 80, 70, 36, 'fc_box')
    e3 = d.node('Expert 3<br>FFN', 295, 80, 70, 36, 'fc_box')
    e4 = d.node('Expert 4<br>FFN', 375, 80, 70, 36, 'noise')
    d.text('...', 450, 85, 40, 26, 'ellipsis')
    en = d.node('Expert N<br>FFN', 495, 80, 70, 36, 'noise')
    sh = d.node('Shared<br>Expert', 600, 80, 80, 36, 'fc_box')
    s = d.op('Σ', cx, 205)
    add = d.op('+', cx, 260)
    out = d.node("Output h'<sub>t</sub>", cx - 60, 300, 120, 30, 'out_box')

    d.link(tok, router, exit='left', entry='top', via=[(router.cx, tok.cy)])
    d.link(router, topk)
    for e in (e2, e3, sh):
        d.link(tok, e, via=[(cx, 60), (e.cx, 60)])
    for e in (e2, e3):
        d.link(e, s, exit='bottom', entry='top', via=[(e.cx, 155), (cx, 155)])
    d.link(topk, s, exit='bottom', entry='left', via=[(topk.cx, s.cy)])
    d.link(s, add)
    d.link(sh, add, exit='bottom', entry='right', via=[(sh.cx, add.cy)])
    d.link(add, out)

    d.text('gate weights g<sub>2</sub>, g<sub>3</sub>', 110, 188, 120, 14, 'dim_label')
    d.text('routed experts (inactive: dashed)', 375, 122, 190, 14, 'dim_label')


# ----------------------------------------------------------------------------
# 5. SwiGLU feed-forward network
# ----------------------------------------------------------------------------
@register(CAT, 'SwiGLU FFN')
def swiglu(d):
    cx = 200
    x = d.node('Input x', cx - 60, 10, 120, 30, 'in_box')
    d.container('SwiGLU FFN', 45, 60, 310, 275, 'container_orange')
    w1 = d.node('W<sub>1</sub> (gate)', 75, 125, 110, 30, 'fc_box')
    w3 = d.node('W<sub>3</sub> (up)', 215, 125, 110, 30, 'fc_box')
    act = d.node('SiLU', 85, 180, 90, 30, 'act')
    mul = d.op('×', cx, 245)
    w2 = d.node('W<sub>2</sub> (down)', cx - 55, 285, 110, 30, 'fc_box')
    out = d.node('Output', cx - 60, 360, 120, 30, 'out_box')

    d.link(x, w1, via=[(cx, 100), (w1.cx, 100)])
    d.link(x, w3, via=[(cx, 100), (w3.cx, 100)])
    d.link(w1, act)
    d.link(act, mul, exit='bottom', entry='left', via=[(act.cx, mul.cy)])
    d.link(w3, mul, exit='bottom', entry='right', via=[(w3.cx, mul.cy)])
    d.link(mul, w2)
    d.link(w2, out)
    d.text('d → d<sub>ff</sub>', 275, 160, 50, 14, 'dim_label')
    d.text('d<sub>ff</sub> → d', 260, 293, 50, 14, 'dim_label')


# ----------------------------------------------------------------------------
# 6. Rotary position embedding
# ----------------------------------------------------------------------------
@register(CAT, 'Rotary Position Embedding')
def rope(d):
    qx, mx, kx = 120, 300, 480
    q0 = d.node('Query q<sub>m</sub><br>(position m)', qx - 75, 10, 150, 36, 'q')
    th = d.node('θ<sub>i</sub> = 10000<sup>-2i/d</sup><br>i = 0, ..., d/2 - 1', mx - 65, 10, 130, 36, 'pos')
    k0 = d.node('Key k<sub>n</sub><br>(position n)', kx - 75, 10, 150, 36, 'k')
    qs = d.node('Split into d/2 pairs<br>(x<sub>2i</sub>, x<sub>2i+1</sub>)', qx - 75, 70, 150, 36, 'reshape')
    ks = d.node('Split into d/2 pairs<br>(x<sub>2i</sub>, x<sub>2i+1</sub>)', kx - 75, 70, 150, 36, 'reshape')
    qr = d.node('Rotate pair i<br>by angle mθ<sub>i</sub>', qx - 75, 130, 150, 36, 'pos')
    cs = d.node('cos pθ<sub>i</sub>, sin pθ<sub>i</sub><br>(p = m, n)', mx - 65, 130, 130, 36, 'act')
    kr = d.node('Rotate pair i<br>by angle nθ<sub>i</sub>', kx - 75, 130, 150, 36, 'pos')
    q1 = d.node('R<sub>m</sub> q<sub>m</sub>', qx - 75, 190, 150, 30, 'q')
    k1 = d.node('R<sub>n</sub> k<sub>n</sub>', kx - 75, 190, 150, 30, 'k')
    dot = d.op('·', mx, 260)
    out = d.node('Score q<sub>m</sub><sup>T</sup> R<sub>n-m</sub> k<sub>n</sub><br>(depends only on m - n)',
                 mx - 90, 300, 180, 36, 'out_box')

    d.link(q0, qs)
    d.link(k0, ks)
    d.link(qs, qr)
    d.link(ks, kr)
    d.link(th, cs)
    d.link(cs, qr, exit='left', entry='right')
    d.link(cs, kr, exit='right', entry='left')
    d.link(qr, q1)
    d.link(kr, k1)
    d.link(q1, dot, exit='bottom', entry='left', via=[(q1.cx, dot.cy)])
    d.link(k1, dot, exit='bottom', entry='right', via=[(k1.cx, dot.cy)])
    d.link(dot, out)


# ----------------------------------------------------------------------------
# 7. Decoder-only LLM overview
# ----------------------------------------------------------------------------
@register(CAT, 'Decoder-Only LLM')
def decoder_only(d):
    cx, sx = 200, 300
    tok = d.node('Input tokens x<sub>1</sub>, ..., x<sub>T</sub>', cx - 80, 10, 160, 30, 'in_box')
    emb = d.node('Token Embedding', cx - 75, 60, 150, 30, 'embed')
    d.container('× N', 100, 105, 225, 370, 'container_repeat')
    j1 = _dot(d, cx, 125)
    n1 = d.node('RMSNorm', cx - 55, 145, 110, 30, 'norm')
    att = d.node('Masked Self-Attention', cx - 75, 195, 150, 30, 'attn')
    a1 = d.op('+', cx, 260)
    j2 = _dot(d, cx, 295)
    n2 = d.node('RMSNorm', cx - 55, 315, 110, 30, 'norm')
    ffn = d.node('FFN', cx - 75, 365, 150, 30, 'fc_box')
    a2 = d.op('+', cx, 430)
    fn = d.node('Final RMSNorm', cx - 55, 495, 110, 30, 'norm')
    head = d.node('LM Head (Linear)', cx - 75, 545, 150, 30, 'fc_box')
    sm = d.node('Softmax', cx - 55, 595, 110, 30, 'softmax')
    out = d.node('Next-token probabilities<br>p(x<sub>T+1</sub> | x<sub>1</sub>, ..., x<sub>T</sub>)',
                 cx - 90, 645, 180, 36, 'out_box')

    d.link(tok, emb)
    d.link(emb, j1, style='line')
    d.link(j1, n1)
    d.link(n1, att)
    d.link(att, a1)
    d.link(a1, j2, style='line')
    d.link(j2, n2)
    d.link(n2, ffn)
    d.link(ffn, a2)
    d.link(a2, fn)
    d.link(fn, head)
    d.link(head, sm)
    d.link(sm, out)
    d.link(j1, a1, exit='right', entry='right', via=[(sx, j1.cy), (sx, a1.cy)], style='skip')
    d.link(j2, a2, exit='right', entry='right', via=[(sx, j2.cy), (sx, a2.cy)], style='skip')
    d.text('T × |V| logits', cx + 8, 578, 70, 14, 'dim_label')


# ----------------------------------------------------------------------------
# 8. T5 encoder-decoder
# ----------------------------------------------------------------------------
@register(CAT, 'T5 Encoder-Decoder')
def t5(d):
    ex, dx = 170, 520                      # encoder / decoder column centres
    esx, dsx = 60, 630                     # residual lanes (encoder left, decoder right)
    lane = 345                             # encoder output channel between the stacks

    # encoder
    ein = d.node('Input tokens<br>(task prefix + text)', ex - 75, 10, 150, 36, 'in_box')
    eemb = d.node('Shared Embedding', ex - 65, 65, 130, 30, 'embed')
    d.container('Encoder × N', 35, 110, 245, 385, 'container_repeat')
    ej1 = _dot(d, ex, 130)
    eln1 = d.node('Layer Norm', ex - 55, 150, 110, 30, 'norm')
    esa = d.node('Self-Attention<br>+ relative position bias', ex - 85, 200, 170, 40, 'attn')
    ea1 = d.op('+', ex, 275)
    ej2 = _dot(d, ex, 310)
    eln2 = d.node('Layer Norm', ex - 55, 325, 110, 30, 'norm')
    effn = d.node('Feed-Forward (ReLU)', ex - 75, 380, 150, 30, 'fc_box')
    ea2 = d.op('+', ex, 450)
    efn = d.node('Final Layer Norm', ex - 65, 515, 130, 30, 'norm')

    # decoder
    din = d.node('Target tokens<br>(shifted right)', dx - 75, 10, 150, 36, 'in_box')
    demb = d.node('Shared Embedding', dx - 65, 65, 130, 30, 'embed')
    d.container('Decoder × N', 410, 110, 245, 550, 'container_repeat')
    dj1 = _dot(d, dx, 130)
    dln1 = d.node('Layer Norm', dx - 55, 150, 110, 30, 'norm')
    dsa = d.node('Masked Self-Attention<br>+ relative position bias', dx - 85, 200, 170, 40, 'attn')
    da1 = d.op('+', dx, 275)
    dj2 = _dot(d, dx, 310)
    dln2 = d.node('Layer Norm', dx - 55, 325, 110, 30, 'norm')
    dca = d.node('Cross-Attention<br>(K, V from encoder)', dx - 85, 375, 170, 40, 'cross_attn')
    da2 = d.op('+', dx, 450)
    dj3 = _dot(d, dx, 485)
    dln3 = d.node('Layer Norm', dx - 55, 500, 110, 30, 'norm')
    dffn = d.node('Feed-Forward (ReLU)', dx - 75, 550, 150, 30, 'fc_box')
    da3 = d.op('+', dx, 615)
    dfn = d.node('Final Layer Norm', dx - 65, 680, 130, 30, 'norm')
    head = d.node('Linear + Softmax', dx - 65, 730, 130, 30, 'fc_box')
    out = d.node('Output text tokens', dx - 75, 780, 150, 30, 'out_box')

    for a, b in [(ein, eemb), (ej1, eln1), (eln1, esa), (esa, ea1), (ej2, eln2), (eln2, effn),
                 (effn, ea2), (ea2, efn),
                 (din, demb), (dj1, dln1), (dln1, dsa), (dsa, da1), (dj2, dln2), (dln2, dca),
                 (dca, da2), (dj3, dln3), (dln3, dffn), (dffn, da3), (da3, dfn), (dfn, head),
                 (head, out)]:
        d.link(a, b)
    for a, b in [(eemb, ej1), (ea1, ej2), (demb, dj1), (da1, dj2), (da2, dj3)]:
        d.link(a, b, style='line')
    for j, a in [(ej1, ea1), (ej2, ea2)]:
        d.link(j, a, exit='left', entry='left', via=[(esx, j.cy), (esx, a.cy)], style='skip')
    for j, a in [(dj1, da1), (dj2, da2), (dj3, da3)]:
        d.link(j, a, exit='right', entry='right', via=[(dsx, j.cy), (dsx, a.cy)], style='skip')
    d.link(efn, dca, exit='right', entry='left', via=[(lane, efn.cy), (lane, dca.cy)])
    d.text('encoder output', 250, efn.cy + 5, 90, 14, 'dim_label')
