"""Generative models: vector-quantised autoencoders, GAN variants, flows and autoregressive models."""

from nnlib.core import register

CAT = 'Generative Models'


@register(CAT, 'VQ-VAE')
def vq_vae(d):
    # main row, left to right, centre y = 150
    x = d.node('x', 0, 132, 50, 36, 'in_box')
    enc = d.node('Encoder', 85, 110, 90, 80, 'trapezoid_enc')
    ze = d.node('z<sub>e</sub>(x)', 210, 132, 70, 36, 'embed')
    nn = d.node('Nearest Neighbour<br>argmin<sub>k</sub> ||z<sub>e</sub> - e<sub>k</sub>||',
                315, 127, 130, 46, 'gate')
    zq = d.node('z<sub>q</sub>(x)', 480, 132, 70, 36, 'embed')
    dec = d.node('Decoder', 585, 110, 90, 80, 'trapezoid_dec')
    xr = d.node('x&#770;', 710, 132, 50, 36, 'out_box')
    for a, b in [(x, enc), (enc, ze), (ze, nn), (nn, zq), (zq, dec), (dec, xr)]:
        d.link(a, b, exit='right', entry='left')

    # codebook (embedding space) feeds the nearest-neighbour lookup from below
    cb = d.node('Codebook<br>e<sub>1</sub>, ..., e<sub>K</sub> ∈ R<sup>D</sup>', 320, 205, 120, 40, 'gate')
    d.link(cb, nn, exit='top', entry='bottom')

    # straight-through estimator: decoder gradient copied from z_q to z_e
    d.link(zq, ze, exit='top', entry='top', via=[(515, 75), (245, 75)], style='grad',
           label='straight-through gradient')

    # loss terms
    commit = d.node('Commitment loss<br>β ||z<sub>e</sub> - sg[e]||<sup>2</sup>', 60, 275, 120, 40, 'loss')
    cbl = d.node('Codebook loss<br>||sg[z<sub>e</sub>] - e||<sup>2</sup>', 320, 275, 120, 40, 'loss')
    rec = d.node('Recon. loss<br>||x - x&#770;||<sup>2</sup>', 580, 275, 120, 40, 'loss')
    d.link(ze, commit, exit='bottom', entry='right', via=[(245, 295)], style='loss_flow')
    d.link(cb, cbl, exit='bottom', entry='top', style='loss_flow')
    d.link(xr, rec, exit='bottom', entry='right', via=[(735, 295)], style='loss_flow')
    d.text('sg[·]: stop-gradient', 315, 325, 130, 14, 'dim_label')


@register(CAT, 'VQGAN')
def vqgan(d):
    d.container('Stage 1: VQGAN', -20, 15, 800, 195, 'container_blue')
    x = d.node('x', 0, 132, 50, 36, 'in_box')
    enc = d.node('CNN<br>Encoder E', 80, 110, 90, 80, 'trapezoid_enc')
    q = d.node('Quantization<br>q(·)', 200, 127, 110, 46, 'gate')
    dec = d.node('CNN<br>Decoder G', 340, 110, 90, 80, 'trapezoid_dec')
    xr = d.node('x&#770;', 460, 132, 50, 36, 'out_box')
    disc = d.node('Patch-based<br>Discriminator D', 540, 127, 110, 46, 'module_alt')
    rf = d.node('real / fake<br>per patch', 680, 132, 80, 36, 'out_box')
    for a, b in [(x, enc), (enc, q), (q, dec), (dec, xr), (xr, disc), (disc, rf)]:
        d.link(a, b, exit='right', entry='left')
    d.text('z&#770;', 173, 130, 24, 14, 'dim_label')
    d.text('z<sub>q</sub>', 313, 130, 24, 14, 'dim_label')

    cb = d.node('Codebook Z', 200, 55, 110, 36, 'gate')
    d.link(cb, q, exit='bottom', entry='top')
    real = d.node('x (real)', 555, 55, 80, 36, 'in_box')
    d.link(real, disc, exit='bottom', entry='top')

    d.container('Stage 2', 160, 235, 490, 95, 'container_purple')
    idx = d.node('Code indices s<br>h x w, raster order', 190, 275, 130, 36, 'reshape')
    tf = d.node('Transformer<br>(autoregressive)', 355, 275, 120, 36, 'module_gen')
    ps = d.node('p(s<sub>i</sub> | s<sub>&lt;i</sub>)', 510, 275, 110, 36, 'softmax')
    d.link(q, idx, exit='bottom', entry='top')
    d.link(idx, tf, exit='right', entry='left')
    d.link(tf, ps, exit='right', entry='left')


@register(CAT, 'StyleGAN Generator')
def stylegan_generator(d):
    # ---- mapping network (left column, cx 85) ----
    d.container('Mapping network f', 0, 40, 170, 508, 'container_orange')
    z = d.node('Latent z ∈ Z', 30, 80, 110, 30, 'in_box')
    norm = d.node('Normalize', 30, 134, 110, 26, 'norm')
    chain = [z, norm]
    for i in range(8):
        chain.append(d.node('FC', 30, 180 + 46 * i, 110, 26, 'fc_box'))
    w = d.node('w ∈ W', 30, 575, 110, 30, 'pos')
    chain.append(w)
    for a, b in zip(chain, chain[1:]):
        d.link(a, b)

    # ---- synthesis network (main column cx 350, A at cx 245, B / noise on the right) ----
    d.container('Synthesis 4 x 4', 215, 40, 365, 326, 'container_blue')
    d.container('× 8  (8² to 1024²)', 215, 385, 365, 361, 'container_repeat')

    def adain(y):
        n = d.node('AdaIN', 290, y, 120, 26, 'norm')
        a = d.node('A', 230, y, 30, 26, 'fc_box')
        d.link(a, n, exit='right', entry='left')
        d.link(w, a, exit='right', entry='left', via=[(200, w.cy), (200, a.cy)])
        return n

    def noise_add(cy):
        o = d.op('+', 350, cy, 26)
        b = d.node('B', 440, cy - 13, 30, 26, 'fc_box')
        nz = d.node('Noise', 500, cy - 13, 60, 26, 'noise')
        d.link(nz, b, exit='left', entry='right')
        d.link(b, o, exit='left', entry='right')
        return o

    const = d.node('Const 4 x 4 x 512', 290, 80, 120, 30, 'in_box')
    seq = [const, noise_add(145), adain(180),
           d.node('Conv 3 x 3', 290, 230, 120, 26, 'conv_box'), noise_add(285), adain(320),
           d.node('Upsample', 290, 410, 120, 26, 'deconv_box'),
           d.node('Conv 3 x 3', 290, 460, 120, 26, 'conv_box'), noise_add(515), adain(550),
           d.node('Conv 3 x 3', 290, 600, 120, 26, 'conv_box'), noise_add(655), adain(690)]
    for a, b in zip(seq, seq[1:]):
        d.link(a, b)


@register(CAT, 'CycleGAN')
def cyclegan(d):
    def cycle(cy, src, g1, mid, g2, rec, loss):
        y0 = cy - 18
        a = d.node(src, 0, y0, 60, 36, 'in_box')
        b = d.node(g1, 95, cy - 20, 90, 40, 'module_gen')
        c = d.node(mid, 220, y0, 70, 36, 'out_box')
        e = d.node(g2, 325, cy - 20, 90, 40, 'module_gen')
        f = d.node(rec, 450, y0, 80, 36, 'out_box')
        lo = d.node(loss, 565, cy - 20, 140, 40, 'loss')
        for p, q in [(a, b), (b, c), (c, e), (e, f)]:
            d.link(p, q, exit='right', entry='left')
        d.link(f, lo, exit='right', entry='left', style='loss_flow')
        return a, c, lo

    def disc(cy, real, name):
        r = d.node(real, 105, cy - 18, 70, 36, 'in_box')
        dn = d.node(name, 210, cy - 18, 90, 36, 'module_alt')
        o = d.node('real / fake', 330, cy - 18, 80, 36, 'out_box')
        d.link(r, dn, exit='right', entry='left')
        d.link(dn, o, exit='right', entry='left')
        return dn

    # forward cycle  x -> G(x) -> F(G(x)) ~ x
    x, gx, l1 = cycle(75, 'x', 'G: X → Y', 'G(x)', 'F: Y → X', 'F(G(x))',
                      'Cycle loss<br>||F(G(x)) - x||<sub>1</sub>')
    d.link(x, l1, exit='top', entry='top', via=[(30, 25), (635, 25)], style='loss_flow')
    # backward cycle  y -> F(y) -> G(F(y)) ~ y
    y, fy, l2 = cycle(285, 'y', 'F: Y → X', 'F(y)', 'G: X → Y', 'G(F(y))',
                      'Cycle loss<br>||G(F(y)) - y||<sub>1</sub>')
    d.link(y, l2, exit='bottom', entry='bottom', via=[(30, 335), (635, 335)], style='loss_flow')

    dy = disc(145, 'y (real)', 'D<sub>Y</sub>')
    dx = disc(215, 'x (real)', 'D<sub>X</sub>')
    d.link(gx, dy, exit='bottom', entry='top')
    d.link(fy, dx, exit='top', entry='bottom')
    d.text('L = L<sub>GAN</sub>(G, D<sub>Y</sub>) + L<sub>GAN</sub>(F, D<sub>X</sub>)<br>+ λ L<sub>cyc</sub>(G, F)',
           450, 160, 250, 40, 'label')


@register(CAT, 'Pix2Pix')
def pix2pix(d):
    x = d.node('x', 0, 62, 50, 36, 'in_box')

    # ---- U-Net generator: encoder column (cx 140) down, decoder column (cx 280) up ----
    d.container('U-Net Generator G', 75, 30, 270, 285, 'container_blue')
    enc_lbl = ['Conv 64', 'Conv 128', 'Conv 256', 'Conv 512']
    dec_lbl = ['Deconv 64', 'Deconv 128', 'Deconv 256', 'Deconv 512']
    enc, dec = [], []
    for i in range(4):
        y = 65 + 50 * i
        enc.append(d.node(enc_lbl[i], 100, y, 80, 30, 'conv_box'))
        dec.append(d.node(dec_lbl[i], 240, y, 80, 30, 'deconv_box'))
    bott = d.node('Conv 512', 170, 265, 80, 30, 'conv_box')
    d.link(x, enc[0], exit='right', entry='left')
    for i in range(3):
        d.link(enc[i], enc[i + 1])
        d.link(dec[i + 1], dec[i], exit='top', entry='bottom')
    d.link(enc[3], bott, exit='bottom', entry='left', via=[(140, 280)])
    d.link(bott, dec[3], exit='right', entry='bottom', via=[(280, 280)])
    for e, u in zip(enc, dec):
        d.link(e, u, exit='right', entry='left', style='skip')

    # ---- PatchGAN discriminator on (x, G(x)) or (x, y) ----
    gx = d.node('G(x)', 380, 62, 60, 36, 'out_box')
    d.link(dec[0], gx, exit='right', entry='left')
    cat = d.op('C', 480, 80)
    d.link(gx, cat, exit='right', entry='left')
    d.link(x, cat, exit='top', entry='top', via=[(25, 15), (480, 15)])
    disc = d.node('PatchGAN<br>Discriminator D', 530, 57, 120, 46, 'module_alt')
    d.link(cat, disc, exit='right', entry='left')
    out = d.node('N x N patch<br>real / fake', 540, 145, 100, 40, 'out_box')
    d.link(disc, out, exit='bottom', entry='top')
    y = d.node('y (target)', 445, 147, 70, 36, 'in_box')
    d.link(y, cat, exit='top', entry='bottom')
    d.text('fake pair: (x, G(x))<br>real pair: (x, y)', 525, 197, 130, 28, 'dim_label')

    # ---- L1 reconstruction loss ----
    l1 = d.node('L1 loss<br>||y - G(x)||<sub>1</sub>', 365, 225, 130, 40, 'loss')
    d.link(gx, l1, exit='bottom', entry='top', entry_pos=(410 - 365) / 130, style='loss_flow')
    d.link(y, l1, exit='bottom', entry='top', entry_pos=(480 - 365) / 130, style='loss_flow')


@register(CAT, 'RealNVP Coupling Layer')
def realnvp_coupling(d):
    x = d.node('x', 215, 25, 80, 36, 'in_box')
    split = d.node('Split', 195, 85, 120, 30, 'reshape')
    d.link(x, split)
    x1 = d.node('x<sub>1</sub>', 45, 142, 70, 36, 'in_box')
    x2 = d.node('x<sub>2</sub>', 395, 142, 70, 36, 'in_box')
    d.link(split, x1, exit='left', entry='top', via=[(80, 100)])
    d.link(split, x2, exit='right', entry='top', via=[(430, 100)])

    s = d.node('s(·)', 165, 215, 90, 30, 'fc_box')
    ex = d.node('exp', 290, 217, 60, 26, 'act')
    t = d.node('t(·)', 165, 285, 90, 30, 'fc_box')
    mul = d.op('⊙', 430, 230)
    add = d.op('+', 430, 300)
    y1 = d.node('y<sub>1</sub> = x<sub>1</sub>', 30, 345, 100, 36, 'out_box')
    y2 = d.node('y<sub>2</sub> = x<sub>2</sub> ⊙ exp(s) + t', 355, 345, 150, 36, 'out_box')

    # x1 is copied unchanged and also conditions s and t
    d.link(x1, s, exit='bottom', entry='left', via=[(80, 230)])
    d.link(x1, t, exit='bottom', entry='left', via=[(80, 300)])
    d.link(x1, y1, exit='bottom', entry='top')
    d.link(s, ex, exit='right', entry='left')
    d.link(ex, mul, exit='right', entry='left')
    d.link(t, add, exit='right', entry='left')
    d.link(x2, mul)
    d.link(mul, add)
    d.link(add, y2)

    cat = d.op('C', 255, 420)
    d.link(y1, cat, exit='bottom', entry='left', via=[(80, 420)])
    d.link(y2, cat, exit='bottom', entry='right', via=[(430, 420)])
    y = d.node('y', 215, 465, 80, 36, 'out_box')
    d.link(cat, y)
    d.text('log |det ∂y/∂x| = Σ<sub>j</sub> s(x<sub>1</sub>)<sub>j</sub>', 310, 475, 190, 16, 'label')


@register(CAT, 'PixelCNN')
def pixelcnn(d):
    # ---- main stack (cx 120) ----
    x = d.node('Image x<br>H x W x 3', 35, 30, 170, 36, 'in_box')
    ca = d.node('7x7 Conv, mask A', 35, 90, 170, 30, 'conv_box')
    d.container('× 15', 15, 140, 240, 240, 'container_repeat')
    r1 = d.node('ReLU + 1x1 Conv (h)', 35, 160, 170, 30, 'conv_box')
    r2 = d.node('ReLU + 3x3 Conv mask B (h)', 35, 214, 170, 30, 'conv_box')
    r3 = d.node('ReLU + 1x1 Conv (2h)', 35, 268, 170, 30, 'conv_box')
    add = d.op('+', 120, 335)
    d.container('× 2', 15, 400, 240, 126, 'container_repeat')
    relu = d.node('ReLU', 70, 420, 100, 26, 'act')
    c1 = d.node('1x1 Conv, mask B', 35, 466, 170, 30, 'conv_box')
    sm = d.node('256-way Softmax', 35, 550, 170, 30, 'softmax')
    out = d.node('p(x<sub>i</sub> | x<sub>&lt;i</sub>)<br>H x W x 3 x 256', 35, 604, 170, 36, 'out_box')
    for a, b in [(x, ca), (ca, r1), (r1, r2), (r2, r3), (r3, add), (add, relu), (relu, c1), (c1, sm), (sm, out)]:
        d.link(a, b)
    d.link(ca, add, exit='right', entry='right', via=[(230, ca.cy), (230, add.cy)], style='skip')

    # ---- causal mask on a 5 x 5 kernel ----
    d.text('Causal mask (5 x 5 kernel)', 290, 148, 150, 16, 'label')
    for i in range(5):
        for j in range(5):
            if i < 2 or (i == 2 and j < 2):
                st = 'conv_box'
            elif i == 2 and j == 2:
                st = 'gate'
            else:
                st = 'act'
            d.node('', 300 + 28 * j, 175 + 28 * i, 20, 20, st)
    for k, (st, txt) in enumerate([('conv_box', 'visible (weight kept)'),
                                   ('act', 'masked (weight = 0)'),
                                   ('gate', 'centre: A = 0, B = 1')]):
        yy = 330 + 30 * k
        d.node('', 300, yy, 20, 20, st)
        d.text(txt, 328, yy + 2, 130, 16, 'label')
