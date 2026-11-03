"""Diffusion and flow-based generative models."""

from nnlib.core import register

CAT = 'Diffusion Models'


@register(CAT, 'DDPM Process')
def ddpm_process(d):
    w, h, y = 60, 40, 60
    xT = d.node('x<sub>T</sub>', 0, y, w, h, 'noise')
    xt = d.node('x<sub>t</sub>', 160, y, w, h, 'reshape')
    xt1 = d.node('x<sub>t-1</sub>', 320, y, w, h, 'reshape')
    x0 = d.node('x<sub>0</sub>', 480, y, w, h, 'in_box')
    d.link(xt, xt1, exit='right', entry='left')
    for a, b in [(xT, xt), (xt1, x0)]:
        dots = d.text('···', a.r + 25, y + 10, 50, 20, 'ellipsis')
        d.link(a, dots, exit='right', entry='left', style='line')
        d.link(dots, b, exit='right', entry='left')
    d.text('p<sub>θ</sub>(x<sub>t-1</sub> | x<sub>t</sub>)<br>network ε<sub>θ</sub>(x<sub>t</sub>, t)',
           215, 16, 110, 36, 'label')
    d.link(xt1, xt, exit='bottom', entry='bottom', via=[(xt1.cx, 135), (xt.cx, 135)], style='skip')
    d.text('q(x<sub>t</sub> | x<sub>t-1</sub>)', 215, 140, 110, 16, 'label')
    d.text('reverse (denoising) process: left to right; forward (noising) process q: dashed',
           0, 175, 540, 14, 'dim_label')


@register(CAT, 'Diffusion U-Net')
def diffusion_unet(d):
    E, D, W = 150, 470, 130            # encoder / decoder column centres, block width
    # time embedding (top centre)
    t = d.node('Timestep t', 260, -115, 100, 28, 'pos')
    sin = d.node('Sinusoidal Emb.', 250, -70, 120, 26, 'embed')
    mlp = d.node('MLP', 260, -25, 100, 26, 'fc_box')
    d.link(t, sin)
    d.link(sin, mlp)
    d.text('t-emb', 45, -30, 40, 14, 'dim_label')
    d.text('t-emb', 535, -30, 40, 14, 'dim_label')

    def blk(label, cx, y, style, w=W, h=30):
        return d.node(label, cx - w / 2, y, w, h, style)

    xt = blk('Noisy image x<sub>t</sub>', E, 20, 'in_box')
    cin = blk('Conv 3x3', E, 70, 'conv_box', h=26)
    r1 = blk('ResBlock, 64', E, 115, 'conv_box')
    d1 = blk('Downsample', E, 165, 'pool_box', w=100, h=26)
    r2 = blk('ResBlock, 128', E, 210, 'conv_box')
    d2 = blk('Downsample', E, 260, 'pool_box', w=100, h=26)
    r3 = blk('ResBlock + Attn<br>256', E, 305, 'conv_box', h=40)
    mid = d.node('Middle Block<br>ResBlock, Attn, ResBlock', 225, 380, 170, 40, 'attn')
    r3u = blk('ResBlock + Attn<br>256', D, 305, 'conv_box', h=40)
    u2 = blk('Upsample', D, 260, 'deconv_box', w=100, h=26)
    r2u = blk('ResBlock, 128', D, 210, 'conv_box')
    u1 = blk('Upsample', D, 165, 'deconv_box', w=100, h=26)
    r1u = blk('ResBlock, 64', D, 115, 'conv_box')
    cout = blk('GN, SiLU, Conv 3x3', D, 70, 'conv_box', h=26)
    out = blk('Predicted noise ε<sub>θ</sub>', D, 20, 'out_box')
    for a, b in [(xt, cin), (cin, r1), (r1, d1), (d1, r2), (r2, d2), (d2, r3),
                 (r3u, u2), (u2, r2u), (r2u, u1), (u1, r1u), (r1u, cout), (cout, out)]:
        d.link(a, b)
    d.link(r3, mid, exit='bottom', entry='left', via=[(E, mid.cy)])
    d.link(mid, r3u, exit='right', entry='bottom', via=[(D, mid.cy)])
    for a, b in [(r1, r1u), (r2, r2u), (r3, r3u)]:
        d.link(a, b, exit='right', entry='left', style='skip')
    d.text('skip (concat)', 255, r1.cy - 17, 110, 14, 'dim_label')
    # time-embedding buses: left lane feeds the encoder, right lane the decoder
    for n in (r1, r2, r3):
        d.link(mlp, n, exit='left', entry='left', via=[(40, mlp.cy), (40, n.cy)])
    d.link(mlp, mid, exit='left', entry='bottom', via=[(40, mlp.cy), (40, 440), (mid.cx, 440)])
    for n in (r1u, r2u, r3u):
        d.link(mlp, n, exit='right', entry='right', via=[(580, mlp.cy), (580, n.cy)])


@register(CAT, 'Time-Conditioned ResBlock')
def time_resblock(d):
    cx = 220
    x = d.node('x  (C<sub>in</sub>)', cx - 60, 0, 120, 30, 'in_box')
    gn1 = d.node('GroupNorm', cx - 50, 55, 100, 26, 'norm')
    s1 = d.node('SiLU', cx - 40, 105, 80, 24, 'act')
    c1 = d.node('Conv 3x3', cx - 60, 153, 120, 28, 'conv_box')
    addt = d.op('+', cx, 220)
    gn2 = d.node('GroupNorm', cx - 50, 260, 100, 26, 'norm')
    s2 = d.node('SiLU', cx - 40, 310, 80, 24, 'act')
    dp = d.node('Dropout', cx - 40, 358, 80, 24, 'act')
    c2 = d.node('Conv 3x3', cx - 60, 406, 120, 28, 'conv_box')
    addr = d.op('+', cx, 470)
    out = d.node('h  (C<sub>out</sub>)', cx - 60, 510, 120, 30, 'out_box')
    for a, b in [(x, gn1), (gn1, s1), (s1, c1), (c1, addt), (addt, gn2), (gn2, s2), (s2, dp),
                 (dp, c2), (c2, addr), (addr, out)]:
        d.link(a, b)
    te = d.node('Time embedding', 345, 60, 120, 30, 'pos')
    st = d.node('SiLU', 365, 110, 80, 24, 'act')
    lt = d.node('Linear (C<sub>out</sub>)', 345, 155, 120, 28, 'fc_box')
    d.link(te, st)
    d.link(st, lt)
    d.link(lt, addt, exit='bottom', entry='right', via=[(lt.cx, addt.cy)])
    sc = d.node('Conv 1x1<br>(if C<sub>in</sub> ≠ C<sub>out</sub>)', 35, 285, 110, 36, 'conv_box')
    d.link(x, sc, exit='left', entry='top', via=[(90, x.cy)], style='skip')
    d.link(sc, addr, exit='bottom', entry='left', via=[(90, addr.cy)], style='skip')


@register(CAT, 'Latent Diffusion')
def latent_diffusion(d):
    d.container('Pixel Space', 5, 0, 165, 300, 'container_red')
    d.container('Latent Space', 190, 0, 345, 300, 'container_green')
    d.container('Conditioning', 555, 0, 150, 300, 'container_gray')
    x = d.node('x', 20, 62, 50, 36, 'in_box')
    enc = d.node('E', 90, 47, 65, 66, 'trapezoid_enc')
    z = d.node('z', 210, 60, 50, 40, 'latent')
    diff = d.node('Diffusion<br>Process', 300, 60, 100, 40, 'noise')
    zT = d.node('z<sub>T</sub>', 445, 60, 50, 40, 'latent')
    for a, b in [(x, enc), (enc, z), (z, diff), (diff, zT)]:
        d.link(a, b, exit='right', entry='left')
    d.container('× T', 270, 165, 245, 115, 'container_repeat')
    unet = d.node('Denoising U-Net ε<sub>θ</sub><br>cross-attention:<br>Q from z<sub>t</sub>, K / V from τ<sub>θ</sub>(y)',
                  285, 180, 210, 64, 'attn')
    z0 = d.node('z', 210, 192, 50, 40, 'latent')
    dec = d.node('D', 90, 179, 65, 66, 'trapezoid_dec_rev')
    xr = d.node('x̃', 20, 194, 50, 36, 'out_box')
    d.link(zT, unet, exit='bottom', entry='top', entry_pos=(zT.cx - unet.x) / unet.w)
    d.link(unet, z0, exit='left', entry='right', exit_pos=(z0.cy - unet.y) / unet.h)
    d.link(z0, dec, exit='left', entry='right')
    d.link(dec, xr, exit='left', entry='right')
    y = d.node('Text / image /<br>semantic map y', 570, 55, 120, 50, 'in_box')
    tau = d.node('Encoder τ<sub>θ</sub><br>(e.g. CLIP text)', 570, 182, 120, 60, 'module')
    d.link(y, tau)
    d.link(tau, unet, exit='left', entry='right', entry_pos=(tau.cy - unet.y) / unet.h)


@register(CAT, 'DiT Block')
def dit_block(d):
    cx = 200

    def box(label, y, w, h, style):
        return d.node(label, cx - w / 2, y, w, h, style)

    inp = box('Input tokens', 0, 140, 30, 'in_box')
    d.container('DiT Block', 75, 45, 235, 590, 'container_purple')
    j1 = d.op('', cx, 90, 8, 'dot')
    ln1 = box('Layer Norm', 110, 110, 26, 'norm')
    ss1 = box('Scale, Shift', 158, 110, 26, 'gate')
    at = box('Multi-Head<br>Self-Attention', 206, 140, 38, 'attn')
    sc1 = box('Scale', 266, 110, 26, 'gate')
    a1 = d.op('+', cx, 330)
    j2 = d.op('', cx, 370, 8, 'dot')
    ln2 = box('Layer Norm', 390, 110, 26, 'norm')
    ss2 = box('Scale, Shift', 438, 110, 26, 'gate')
    ff = box('Pointwise<br>Feedforward', 486, 140, 38, 'fc_box')
    sc2 = box('Scale', 546, 110, 26, 'gate')
    a2 = d.op('+', cx, 610)
    out = box('Output tokens', 660, 140, 30, 'out_box')
    d.link(inp, j1, style='line')
    seq = [j1, ln1, ss1, at, sc1, a1]
    for a, b in zip(seq, seq[1:]):
        d.link(a, b)
    d.link(a1, j2, style='line')
    seq = [j2, ln2, ss2, ff, sc2, a2]
    for a, b in zip(seq, seq[1:]):
        d.link(a, b)
    d.link(a2, out)
    for j, a in [(j1, a1), (j2, a2)]:
        d.link(j, a, exit='left', entry='left', via=[(100, j.cy), (100, a.cy)], style='skip')
    # conditioning
    t = d.node('Timestep t', 345, 0, 90, 30, 'pos')
    yl = d.node('Label y', 465, 0, 90, 30, 'pos')
    s = d.op('+', 450, 70)
    mlp = d.node('MLP', 400, 110, 100, 28, 'fc_box')
    d.link(t, s, exit='bottom', entry='left', via=[(t.cx, s.cy)])
    d.link(yl, s, exit='bottom', entry='right', via=[(yl.cx, s.cy)])
    d.link(s, mlp)
    d.text('adaLN-Zero', 505, 117, 60, 14, 'dim_label')
    for n, lab in [(ss1, 'γ<sub>1</sub>, β<sub>1</sub>'), (sc1, 'α<sub>1</sub>'),
                   (ss2, 'γ<sub>2</sub>, β<sub>2</sub>'), (sc2, 'α<sub>2</sub>')]:
        d.link(mlp, n, exit='bottom', entry='right', via=[(mlp.cx, n.cy)])
        d.text(lab, 330, n.cy - 17, 70, 14, 'dim_label')


@register(CAT, 'ControlNet')
def controlnet(d):
    x = d.node('x', 90, 0, 120, 30, 'in_box')
    locked = d.node('Neural Network Block<br>(locked, frozen)', 75, 135, 150, 50, 'frozen')
    add_out = d.op('+', 150, 290)
    y = d.node('y<sub>c</sub>', 90, 330, 120, 30, 'out_box')
    c = d.node('Condition c', 480, 0, 120, 30, 'in_box')
    zc1 = d.node('Zero Conv<br>(1x1, init 0)', 480, 77, 120, 36, 'conv_box')
    add_in = d.op('+', 380, 95)
    train = d.node('Neural Network Block<br>(trainable copy)', 305, 135, 150, 50, 'module')
    zc2 = d.node('Zero Conv<br>(1x1, init 0)', 320, 210, 120, 36, 'conv_box')
    d.link(x, locked)
    d.link(locked, add_out)
    d.link(add_out, y)
    d.link(x, add_in, exit='right', entry='top', via=[(380, x.cy)])
    d.link(c, zc1)
    d.link(zc1, add_in, exit='left', entry='right')
    d.link(add_in, train)
    d.link(train, zc2)
    d.link(zc2, add_out, exit='bottom', entry='right', via=[(380, add_out.cy)])



@register(CAT, 'Classifier-Free Guidance')
def cfg(d):
    xt = d.node('x<sub>t</sub>, t', 0, 95, 90, 36, 'in_box')
    dc = d.node('Denoiser<br>ε<sub>θ</sub>(x<sub>t</sub>, t, c)', 150, 30, 150, 40, 'module')
    du = d.node('Denoiser<br>ε<sub>θ</sub>(x<sub>t</sub>, t, ∅)', 150, 155, 150, 40, 'module')
    c = d.node('Condition c', 175, -30, 100, 28, 'pos')
    nul = d.node('Null token ∅', 175, 225, 100, 28, 'pos')
    d.link(c, dc)
    d.link(nul, du)
    d.link(xt, dc, exit='right', entry='left', via=[(120, xt.cy), (120, dc.cy)])
    d.link(xt, du, exit='right', entry='left', via=[(120, xt.cy), (120, du.cy)])
    d.text('shared weights', 180, 105, 90, 14, 'dim_label')
    comb = d.node('ε̃ = ε<sub>u</sub> + w (ε<sub>c</sub> − ε<sub>u</sub>)', 360, 93, 170, 40, 'act')
    d.link(dc, comb, exit='right', entry='left', via=[(330, dc.cy), (330, comb.y + 10)], entry_pos=0.25)
    d.link(du, comb, exit='right', entry='left', via=[(330, du.cy), (330, comb.y + 30)], entry_pos=0.75)
    d.text('ε<sub>c</sub>', 305, dc.cy - 18, 20, 14, 'dim_label')
    d.text('ε<sub>u</sub>', 305, du.cy + 4, 20, 14, 'dim_label')
    w = d.node('Guidance scale w', 385, 25, 120, 28, 'pos')
    d.link(w, comb)
    smp = d.node('Sampler step<br>(DDPM / DDIM)', 580, 93, 110, 40, 'module_alt')
    out = d.node('x<sub>t-1</sub>', 740, 95, 80, 36, 'out_box')
    d.link(comb, smp, exit='right', entry='left')
    d.link(smp, out, exit='right', entry='left')


@register(CAT, 'Flow Matching')
def flow_matching(d):
    d.container('Training', 5, 0, 770, 215, 'container_blue')
    x0 = d.node('x<sub>0</sub> ~ N(0, I)', 20, 55, 110, 30, 'noise')
    x1 = d.node('x<sub>1</sub> ~ data', 20, 105, 110, 30, 'in_box')
    t = d.node('t ~ U[0, 1]', 20, 155, 110, 30, 'pos')
    it = d.node('Interpolate<br>x<sub>t</sub> = (1 − t) x<sub>0</sub> + t x<sub>1</sub>', 180, 95, 170, 50, 'act')
    d.link(x0, it, exit='right', entry='left', via=[(155, x0.cy), (155, 105)], entry_pos=0.2)
    d.link(x1, it, exit='right', entry='left')
    d.link(t, it, exit='right', entry='left', via=[(155, t.cy), (155, 135)], entry_pos=0.8)
    v = d.node('v<sub>θ</sub>(x<sub>t</sub>, t)', 400, 100, 120, 40, 'module')
    d.link(it, v, exit='right', entry='left')
    d.link(t, v, exit='bottom', entry='bottom', via=[(t.cx, 200), (v.cx, 200)])
    loss = d.node('L = ‖v<sub>θ</sub>(x<sub>t</sub>, t) − (x<sub>1</sub> − x<sub>0</sub>)‖<sup>2</sup>',
                  570, 100, 190, 40, 'loss')
    d.link(v, loss, exit='right', entry='left')
    d.text('target velocity x<sub>1</sub> − x<sub>0</sub>', 570, 145, 190, 14, 'dim_label')
    d.container('Sampling', 5, 235, 770, 100, 'container_green')
    n = d.node('x<sub>0</sub> ~ N(0, I)', 20, 280, 130, 36, 'noise')
    ode = d.node('ODE solver (Euler / RK)<br>dx/dt = v<sub>θ</sub>(x, t), t: 0 → 1', 220, 275, 230, 46, 'module_alt')
    s = d.node('Sample x<sub>1</sub>', 520, 280, 130, 36, 'out_box')
    d.link(n, ode, exit='right', entry='left')
    d.link(ode, s, exit='right', entry='left')
