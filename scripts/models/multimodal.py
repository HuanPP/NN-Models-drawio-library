"""Multimodal (vision-language and speech) architectures."""

from nnlib.core import register

CAT = 'Multimodal'


@register(CAT, 'CLIP')
def clip(d):
    # text branch: encoder above the T row, fan-out into T_1..T_N
    txt_in = d.node('Text', 20, 80, 110, 30, 'in_box')
    txt_enc = d.node('Text Encoder<br>(Transformer)', 330, 75, 110, 40, 'module_alt')
    d.link(txt_in, txt_enc, exit='right', entry='left')
    # image branch: encoder left of the I column, fan-out into I_1..I_N
    img_in = d.node('Image', 20, 195, 110, 30, 'in_box')
    img_enc = d.node('Image Encoder<br>(ResNet / ViT)', 20, 255, 110, 40, 'module')
    d.link(img_in, img_enc)

    idx = ['1', '2', '3', 'N']
    cols = [280, 350, 420, 490]
    rows = [215, 255, 295, 335]
    for j, cx in zip(idx, cols):
        t = d.node(f'T<sub>{j}</sub>', cx - 30, 150, 60, 30, 'embed')
        d.link(txt_enc, t, exit='bottom', entry='top', via=[(txt_enc.cx, 130), (cx, 130)])
    for i, cy in zip(idx, rows):
        n = d.node(f'I<sub>{i}</sub>', 170, cy - 15, 60, 30, 'embed')
        d.link(img_enc, n, exit='right', entry='left', via=[(150, img_enc.cy), (150, cy)])

    # N x N similarity matrix, diagonal = matching (positive) pairs
    d.container('', 240, 190, 290, 170, 'container_gray')
    for a, (i, cy) in enumerate(zip(idx, rows)):
        for b, (j, cx) in enumerate(zip(idx, cols)):
            d.node(f'I<sub>{i}</sub>·T<sub>{j}</sub>', cx - 30, cy - 15, 60, 30,
                   'pos' if a == b else 'act')
    d.text('cosine<br>similarity<br>x exp(τ)', 540, 215, 70, 44, 'dim_label')
    d.text('diagonal:<br>positive<br>pairs', 540, 290, 70, 44, 'dim_label')

    loss = d.node('Symmetric Contrastive Loss<br>(CE over rows and columns)', 275, 390, 220, 40, 'loss')
    d.arrow([(385, 360), (loss.cx, loss.y)])


@register(CAT, 'BLIP-2')
def blip2(d):
    cy = 120
    img = d.node('Image', 20, cy - 20, 60, 40, 'in_box')
    enc = d.node('Image Encoder<br>(frozen)', 115, cy - 20, 100, 40, 'frozen')
    qf = d.node('Q-Former', 250, cy - 20, 90, 40, 'module_gen')
    fc = d.node('FC', 375, cy - 20, 60, 40, 'fc_box')
    llm = d.node('LLM Decoder<br>(frozen)', 470, cy - 20, 100, 40, 'frozen')
    out = d.node('Output<br>Text', 605, cy - 20, 90, 40, 'out_box')
    for a, b in [(img, enc), (enc, qf), (qf, fc), (fc, llm), (llm, out)]:
        d.link(a, b, exit='right', entry='left')
    q = d.node('Learned<br>Queries', 250, 40, 90, 36, 'embed')
    d.link(q, qf)
    prompt = d.node('Text Prompt', 470, 165, 100, 30, 'in_box')
    d.link(prompt, llm, exit='top', entry='bottom')


@register(CAT, 'Q-Former')
def qformer(d):
    L, R = 230, 430
    img = d.node('Image', 10, 100, 100, 30, 'in_box')
    enc = d.node('Image Encoder<br>(frozen)', 10, 165, 100, 40, 'frozen')
    d.link(img, enc)

    d.container('Q-Former', 125, 0, 410, 365, 'container_purple')
    q = d.node('Learned Queries', L - 70, 40, 140, 30, 'embed')
    txt = d.node('Input Text', R - 70, 40, 140, 30, 'in_box')
    d.container('× N', 145, 95, 370, 195, 'container_repeat')
    sa = d.node('Shared Self-Attention', 160, 115, 340, 30, 'attn')
    ca = d.node('Cross-Attention', L - 70, 170, 140, 30, 'cross_attn')
    ffn_q = d.node('Feed Forward', L - 70, 225, 140, 30, 'fc_box')
    ffn_t = d.node('Feed Forward', R - 70, 225, 140, 30, 'fc_box')
    out_q = d.node('Query Embeddings', L - 70, 315, 140, 30, 'out_box')
    out_t = d.node('Text Features', R - 70, 315, 140, 30, 'out_box')
    pl, pr = (L - sa.x) / sa.w, (R - sa.x) / sa.w
    d.link(q, sa, entry_pos=pl)
    d.link(txt, sa, entry_pos=pr)
    d.link(sa, ca, exit_pos=pl)
    d.link(sa, ffn_t, exit_pos=pr)
    d.link(enc, ca, exit='right', entry='left')
    d.text('(every other block)', 310, 178, 110, 14, 'dim_label')
    d.link(ca, ffn_q)
    d.link(ffn_q, out_q)
    d.link(ffn_t, out_t)

    objs = [('ITC<br>Image-Text Contrastive', '(uni-modal mask)'),
            ('ITM<br>Image-Text Matching', '(bi-directional mask)'),
            ('ITG<br>Image-Grounded Text Gen.', '(multimodal causal mask)')]
    for k, (name, mask) in enumerate(objs):
        y = 390 + 50 * k
        o = d.node(name, 250, y, 160, 36, 'loss')
        d.link(out_q, o, exit='bottom', entry='left', via=[(L, o.cy)])
        d.link(out_t, o, exit='bottom', entry='right', via=[(R, o.cy)])
        d.text(mask, 440, o.cy - 7, 125, 14, 'dim_label')


@register(CAT, 'LLaVA')
def llava(d):
    r1, r2 = 65, 145
    img = d.node('Image<br>X<sub>v</sub>', 20, r1 - 20, 80, 40, 'in_box')
    enc = d.node('Vision Encoder<br>CLIP ViT-L/14<br>(frozen)', 135, r1 - 25, 110, 50, 'frozen')
    proj = d.node('Projection<br>W', 280, r1 - 20, 80, 40, 'fc_box')
    hv = d.node('H<sub>v</sub>', 395, r1 - 20, 60, 40, 'embed')
    ins = d.node('Instruction<br>X<sub>q</sub>', 20, r2 - 20, 80, 40, 'in_box')
    emb = d.node('Word<br>Embedding', 280, r2 - 20, 80, 40, 'embed')
    hq = d.node('H<sub>q</sub>', 395, r2 - 20, 60, 40, 'embed')
    lm = d.node('Language<br>Model f<sub>φ</sub><br>(Vicuna)', 490, 40, 100, 130, 'module_gen')
    out = d.node('Language<br>Response<br>X<sub>a</sub>', 625, 80, 80, 50, 'out_box')
    for a, b in [(img, enc), (enc, proj), (proj, hv), (ins, emb), (emb, hq)]:
        d.link(a, b, exit='right', entry='left')
    d.text('Z<sub>v</sub>', 250, 42, 25, 14, 'dim_label')
    d.link(hv, lm, exit='right', entry='left', entry_pos=(r1 - lm.y) / lm.h)
    d.link(hq, lm, exit='right', entry='left', entry_pos=(r2 - lm.y) / lm.h)
    d.link(lm, out, exit='right', entry='left')


@register(CAT, 'Flamingo')
def flamingo(d):
    # vision branch (left column)
    vx, vw = 60, 140
    img = d.node('Images / Video', vx, 15, vw, 36, 'in_box')
    ve = d.node('Vision Encoder<br>(frozen)', vx, 75, vw, 36, 'frozen')
    pr = d.node('Perceiver<br>Resampler', vx, 135, vw, 36, 'attn')
    vt = d.node('Visual Tokens', vx, 195, vw, 30, 'embed')
    for a, b in [(img, ve), (ve, pr), (pr, vt)]:
        d.link(a, b)

    # language branch (main column, cx = 400)
    X, W = 330, 140
    txt = d.node('Interleaved Text<br>(with &lt;image&gt; tags)', 320, 15, 160, 36, 'in_box')
    d.container('× N', 235, 110, 335, 510, 'container_repeat')
    d.container('Gated XATTN-Dense', 245, 125, 315, 420, 'container_red')
    j1 = d.op('', 400, 170, size=8, style='dot')
    ca = d.node('Cross-Attention', X, 195, W, 30, 'cross_attn')
    m1 = d.op('×', 400, 265)
    g1 = d.node('tanh(α)', 290, 250, 70, 30, 'gate')
    a1 = d.op('+', 400, 320)
    j2 = d.op('', 400, 360, size=8, style='dot')
    ffw = d.node('Dense (FFW)', X, 385, W, 30, 'fc_box')
    m2 = d.op('×', 400, 455)
    g2 = d.node('tanh(β)', 290, 440, 70, 30, 'gate')
    a2 = d.op('+', 400, 510)
    lm = d.node('LM Block (frozen)', X, 570, W, 30, 'frozen')
    out = d.node('Next-Token Prediction', X, 645, W, 30, 'out_box')

    d.link(txt, j1, style='line')
    d.link(j1, ca)
    d.link(ca, m1)
    d.link(m1, a1)
    d.link(a1, j2, style='line')
    d.link(j2, ffw)
    d.link(ffw, m2)
    d.link(m2, a2)
    d.link(a2, lm)
    d.link(lm, out)
    d.link(g1, m1, exit='right', entry='left')
    d.link(g2, m2, exit='right', entry='left')
    d.link(j1, a1, exit='right', entry='right', via=[(535, j1.cy), (535, a1.cy)], style='skip')
    d.link(j2, a2, exit='right', entry='right', via=[(535, j2.cy), (535, a2.cy)], style='skip')
    d.link(vt, ca, exit='right', entry='left', label='K, V')


@register(CAT, 'Early vs Late Fusion')
def fusion_comparison(d):
    def panel(p, title):
        d.container(title, p, 10, 240, 335, 'container_gray')
        L, R, C = p + 60, p + 180, p + 120
        a = d.node('Modality A', L - 45, 55, 90, 30, 'in_box')
        b = d.node('Modality B', R - 45, 55, 90, 30, 'in_box')
        pred = d.node('Prediction', C - 60, 295, 120, 30, 'out_box')
        return L, R, C, a, b, pred

    def merge(src_l, src_r, dst, side_l='left', side_r='right'):
        d.link(src_l, dst, exit='bottom', entry=side_l, via=[(src_l.cx, dst.cy)])
        d.link(src_r, dst, exit='bottom', entry=side_r, via=[(src_r.cx, dst.cy)])

    # early fusion: concatenate raw inputs, one joint network
    L, R, C, a, b, pred = panel(10, 'Early Fusion')
    cat = d.op('C', C, 130)
    merge(a, b, cat)
    joint = d.node('Joint<br>Network', C - 70, 172, 140, 96, 'module_gen')
    d.link(cat, joint)
    d.link(joint, pred)

    # intermediate fusion: unimodal encoders, fuse features, joint layers
    L, R, C, a, b, pred = panel(270, 'Intermediate Fusion')
    ea = d.node('Encoder A', L - 45, 112, 90, 36, 'module')
    eb = d.node('Encoder B', R - 45, 112, 90, 36, 'module_alt')
    d.link(a, ea)
    d.link(b, eb)
    cat = d.op('C', C, 190)
    merge(ea, eb, cat)
    joint = d.node('Joint Network', C - 70, 232, 140, 36, 'module_gen')
    d.link(cat, joint)
    d.link(joint, pred)

    # late fusion: separate networks, fuse decisions
    L, R, C, a, b, pred = panel(530, 'Late Fusion')
    na = d.node('Network A', L - 45, 112, 90, 36, 'module')
    nb = d.node('Network B', R - 45, 112, 90, 36, 'module_alt')
    da = d.node('Decision A', L - 45, 175, 90, 30, 'out_box')
    db = d.node('Decision B', R - 45, 175, 90, 30, 'out_box')
    for s, t in [(a, na), (b, nb), (na, da), (nb, db)]:
        d.link(s, t)
    fuse = d.node('Fusion<br>(avg / vote)', C - 40, 232, 80, 36, 'act')
    merge(da, db, fuse)
    d.link(fuse, pred)


@register(CAT, 'Whisper')
def whisper(d):
    # ---------------- encoder (left column, cx = 170) ----------------
    EX, EW = 90, 160
    mel = d.node('Log-Mel Spectrogram<br>(80 x 3000)', EX, 15, EW, 40, 'in_box')
    c1 = d.node('Conv1D (k=3) + GELU', EX, 80, EW, 30, 'conv_box')
    c2 = d.node('Conv1D (k=3, s=2) + GELU', EX, 135, EW, 30, 'conv_box')
    pe = d.op('+', 170, 200)
    sin = d.node('Sinusoidal<br>Pos. Encoding', 25, 182, 105, 36, 'pos')
    d.container('× N', 70, 240, 200, 135, 'container_repeat')
    esa = d.node('Self-Attention', EX, 260, EW, 30, 'attn')
    emlp = d.node('MLP', EX, 315, EW, 30, 'fc_box')
    eln = d.node('LayerNorm', EX, 400, EW, 30, 'norm')
    for a, b in [(mel, c1), (c1, c2), (c2, pe), (pe, esa), (esa, emlp), (emlp, eln)]:
        d.link(a, b)
    d.link(sin, pe, exit='right', entry='left')

    # ---------------- decoder (right column, cx = 470) ----------------
    chips = [('SOT', 'start', 315, 50), ('EN', 'lang.', 375, 40), ('TRANSCRIBE', 'task', 425, 90),
             ('0.0', 'time', 525, 40), ('The ...', 'text', 575, 50)]
    emb = d.node('Token Embedding', 400, 85, 140, 30, 'embed')
    for lab, role, x, w in chips:
        c = d.node(lab, x, 20, w, 30, 'in_box')
        d.text(role, x, 3, w, 14, 'dim_label')
        if c.cx == emb.cx:
            d.link(c, emb)
        else:
            d.link(c, emb, exit='bottom', entry='top', via=[(c.cx, 65), (emb.cx, 65)])
    pd = d.op('+', 470, 145)
    lpe = d.node('Learned<br>Pos. Embedding', 510, 127, 110, 36, 'pos')
    d.container('× N', 375, 180, 190, 190, 'container_repeat')
    dsa = d.node('Masked Self-Attention', 395, 200, 150, 30, 'attn')
    dca = d.node('Cross-Attention', 395, 255, 150, 30, 'cross_attn')
    dmlp = d.node('MLP', 395, 310, 150, 30, 'fc_box')
    head = d.node('Linear + Softmax', 395, 395, 150, 30, 'softmax')
    out = d.node('Next-Token Prediction', 395, 450, 150, 30, 'out_box')
    for a, b in [(emb, pd), (pd, dsa), (dsa, dca), (dca, dmlp), (dmlp, head), (head, out)]:
        d.link(a, b)
    d.link(lpe, pd, exit='left', entry='right')
    # encoder output feeds the cross-attention of every decoder block
    d.link(eln, dca, exit='right', entry='left', via=[(330, eln.cy), (330, dca.cy)], label='K, V')
