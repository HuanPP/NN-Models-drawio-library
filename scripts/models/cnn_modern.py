"""Modern CNN building blocks."""

from nnlib.core import register

CAT = 'CNN Architectures'


@register(CAT, 'SE Block')
def se_block(d):
    d.container('Squeeze-and-Excitation', 80, 10, 210, 440, 'container_blue')
    inp = d.node('Input<br>H x W x C', 130, 50, 120, 36, 'in_box')
    gap = d.node('Global AvgPool', 130, 110, 120, 26, 'pool_box')
    fc1 = d.node('FC (C/r)', 130, 160, 120, 26, 'fc_box')
    relu = d.node('ReLU', 145, 210, 90, 24, 'act')
    fc2 = d.node('FC (C)', 130, 258, 120, 26, 'fc_box')
    sig = d.node('Sigmoid', 145, 308, 90, 24, 'act')
    scale = d.op('×', 190, 370)
    out = d.node('Output<br>H x W x C', 130, 405, 120, 36, 'out_box')
    for a, b in [(inp, gap), (gap, fc1), (fc1, relu), (relu, fc2), (fc2, sig), (sig, scale), (scale, out)]:
        d.link(a, b)
    # identity path: leave the input on the left, run down, enter the scale op from the left
    d.link(inp, scale, exit='left', entry='left', via=[(105, inp.cy), (105, scale.cy)])
    d.text('1 x 1 x C', 196, 335, 50, 14, 'dim_label')


@register(CAT, 'CBAM')
def cbam(d):
    f = d.node('Input F', 20, 0, 120, 30, 'in_box')
    j1 = d.op('', 80, 75, 8, 'dot')
    mc = d.op('×', 80, 205)
    j2 = d.op('', 80, 290, 8, 'dot')
    ms = d.op('×', 80, 430)
    out = d.node('Refined F″', 20, 470, 120, 30, 'out_box')
    d.link(f, j1, style='line')
    d.link(j1, mc)
    d.link(mc, j2, style='line')
    d.link(j2, ms)
    d.link(ms, out)
    # channel attention
    d.container('Channel Attention', 150, 30, 410, 200, 'container_purple')
    mx = d.node('MaxPool', 190, 62, 90, 26, 'pool_box')
    av = d.node('AvgPool', 190, 112, 90, 26, 'pool_box')
    for p in (mx, av):
        d.link(j1, p, exit='right', entry='left', via=[(170, j1.cy), (170, p.cy)])
    mlp = d.node('Shared<br>MLP', 310, 62, 90, 76, 'fc_box')
    d.link(mx, mlp, exit='right', entry='left', entry_pos=(mx.cy - mlp.y) / mlp.h)
    d.link(av, mlp, exit='right', entry='left', entry_pos=(av.cy - mlp.y) / mlp.h)
    add = d.op('+', 450, 100)
    for p in (mx, av):
        pos = (p.cy - mlp.y) / mlp.h
        d.link(mlp, add, exit='right', entry='left', exit_pos=pos, via=[(425, p.cy), (425, add.cy)])
    s1 = d.node('σ', 485, 87, 50, 26, 'act')
    d.link(add, s1, exit='right', entry='left')
    d.link(s1, mc, exit='bottom', entry='right', via=[(s1.cx, mc.cy)])
    d.text('M<sub>c</sub>', 515, 160, 30, 14, 'dim_label')
    # spatial attention
    d.container('Spatial Attention', 150, 250, 410, 200, 'container_purple')
    mx2 = d.node('Max (channel)', 190, 287, 100, 26, 'pool_box')
    av2 = d.node('Avg (channel)', 190, 337, 100, 26, 'pool_box')
    for p in (mx2, av2):
        d.link(j2, p, exit='right', entry='left', via=[(170, j2.cy), (170, p.cy)])
    cat = d.op('C', 335, 325)
    d.link(mx2, cat, exit='right', entry='top', via=[(cat.cx, mx2.cy)])
    d.link(av2, cat, exit='right', entry='bottom', via=[(cat.cx, av2.cy)])
    cv = d.node('Conv 7x7', 375, 312, 80, 26, 'conv_box')
    s2 = d.node('σ', 485, 312, 50, 26, 'act')
    d.link(cat, cv, exit='right', entry='left')
    d.link(cv, s2, exit='right', entry='left')
    d.link(s2, ms, exit='bottom', entry='right', via=[(s2.cx, ms.cy)])
    d.text('M<sub>s</sub>', 515, 385, 30, 14, 'dim_label')


@register(CAT, 'ConvNeXt Block')
def convnext_block(d):
    cx = 200
    inp = d.node('Input (H x W x C)', cx - 75, 0, 150, 30, 'in_box')
    d.container('ConvNeXt Block', 60, 45, 280, 400, 'container_blue')
    j = d.op('', cx, 85, 8, 'dot')
    seq = [j] + [d.node(lab, cx - w / 2, y, w, h, st) for lab, y, w, h, st in [
        ('Depthwise Conv 7x7', 105, 150, 28, 'conv_box'), ('LayerNorm', 157, 110, 26, 'norm'),
        ('1x1 Conv (C → 4C)', 207, 150, 28, 'conv_box'), ('GELU', 259, 80, 24, 'act'),
        ('1x1 Conv (4C → C)', 307, 150, 28, 'conv_box'), ('Layer Scale γ', 359, 110, 26, 'act')]]
    add = d.op('+', cx, 420)
    out = d.node('Output (H x W x C)', cx - 75, 470, 150, 30, 'out_box')
    d.link(inp, j, style='line')
    for a, b in zip(seq + [add], seq[1:] + [add, out]):
        d.link(a, b)
    d.link(j, add, exit='left', entry='left', via=[(90, j.cy), (90, add.cy)], style='skip')


@register(CAT, 'ResNeXt Block')
def resnext_block(d):
    inp = d.node('Input (256-d)', 155, 0, 130, 30, 'in_box')
    j = d.op('', 220, 55, 8, 'dot')
    d.link(inp, j, style='line')
    agg = d.op('+', 220, 260)
    for cx in (80, 190, 350):
        b1 = d.node('1x1, 4', cx - 40, 90, 80, 26, 'conv_box')
        b2 = d.node('3x3, 4', cx - 40, 136, 80, 26, 'conv_box')
        b3 = d.node('1x1, 256', cx - 40, 182, 80, 26, 'conv_box')
        d.link(j, b1, exit='bottom', entry='top', via=[(220, 75), (cx, 75)])
        d.link(b1, b2)
        d.link(b2, b3)
        d.link(b3, agg, exit='bottom', entry='top', via=[(cx, 230), (220, 230)])
    d.text('···', 245, 135, 50, 30, 'ellipsis')
    d.text('32 paths', 240, 112, 60, 14, 'dim_label')
    res = d.op('+', 220, 315)
    relu = d.node('ReLU', 180, 350, 80, 24, 'act')
    out = d.node('Output (256-d)', 155, 398, 130, 30, 'out_box')
    d.link(agg, res)
    d.link(res, relu)
    d.link(relu, out)
    d.link(j, res, exit='right', entry='right', via=[(430, j.cy), (430, res.cy)], style='skip')
    d.text('aggregate', 240, 245, 60, 14, 'dim_label')


@register(CAT, 'Depthwise Separable Conv')
def dw_separable(d):
    d.container('Standard Conv', 0, 0, 190, 300, 'container_gray')
    left = [d.node(lab, 95 - w / 2, y, w, h, st) for lab, y, w, h, st in [
        ('Input H x W x C', 40, 140, 30, 'in_box'), ('Conv 3x3<br>C → C′', 95, 140, 36, 'conv_box'),
        ('BN', 155, 110, 26, 'norm'), ('ReLU', 205, 90, 24, 'act'),
        ('Output H x W x C′', 253, 140, 30, 'out_box')]]
    for a, b in zip(left, left[1:]):
        d.link(a, b)
    d.text('k² · C · C′ · HW mult-adds', 0, 310, 190, 14, 'dim_label')
    d.container('Depthwise Separable', 220, 0, 190, 465, 'container_blue')
    right = [d.node(lab, 315 - w / 2, y, w, h, st) for lab, y, w, h, st in [
        ('Input H x W x C', 40, 140, 30, 'in_box'),
        ('Depthwise Conv 3x3<br>(one filter per channel)', 95, 160, 40, 'conv_box'),
        ('BN', 159, 110, 26, 'norm'), ('ReLU', 209, 90, 24, 'act'),
        ('Pointwise Conv 1x1<br>C → C′', 257, 140, 36, 'conv_box'),
        ('BN', 317, 110, 26, 'norm'), ('ReLU', 367, 90, 24, 'act'),
        ('Output H x W x C′', 415, 140, 30, 'out_box')]]
    for a, b in zip(right, right[1:]):
        d.link(a, b)
    d.text('k² · C · HW + C · C′ · HW mult-adds', 220, 475, 190, 14, 'dim_label')


@register(CAT, 'ShuffleNet V2 Unit')
def shufflenet_v2(d):
    inp = d.node('Input', 130, 0, 140, 30, 'in_box')
    sp = d.node('Channel Split', 140, 55, 120, 26, 'reshape')
    d.link(inp, sp)
    br = [d.node(lab, 230, y, 140, 26, st) for lab, y, st in [
        ('1x1 Conv, BN, ReLU', 110, 'conv_box'), ('3x3 DWConv, BN', 160, 'conv_box'),
        ('1x1 Conv, BN, ReLU', 210, 'conv_box')]]
    d.link(sp, br[0], exit='bottom', entry='top', exit_pos=0.75, via=[(230, 95), (300, 95)])
    for a, b in zip(br, br[1:]):
        d.link(a, b)
    cat = d.op('C', 200, 280)
    d.link(sp, cat, exit='bottom', entry='left', exit_pos=0.25, via=[(170, cat.cy)])
    d.link(br[-1], cat, exit='bottom', entry='right', via=[(300, cat.cy)])
    d.text('identity', 115, 170, 50, 14, 'dim_label')
    sh = d.node('Channel Shuffle', 140, 315, 120, 26, 'reshape')
    out = d.node('Output', 130, 365, 140, 30, 'out_box')
    d.link(cat, sh)
    d.link(sh, out)


@register(CAT, 'RepVGG Block')
def repvgg_block(d):
    tr = d.container('Training-time', 0, 0, 330, 290, 'container_gray')
    inp = d.node('Input', 110, 40, 110, 30, 'in_box')
    add = d.op('+', 165, 215)
    for cx, convl in [(60, '3x3 Conv'), (165, '1x1 Conv'), (270, None)]:
        bn = d.node('BN', cx - 40, 151, 80, 26, 'norm')
        if convl:
            cv = d.node(convl, cx - 40, 105, 80, 26, 'conv_box')
            d.link(inp, cv, exit='bottom', entry='top', via=[(165, 88), (cx, 88)])
            d.link(cv, bn)
        else:
            d.link(inp, bn, exit='bottom', entry='top', via=[(165, 88), (cx, 88)])
        d.link(bn, add, exit='bottom', entry='top', via=[(cx, 188), (165, 188)])
    relu = d.node('ReLU', 125, 245, 80, 24, 'act')
    d.link(add, relu)
    inf = d.container('Inference-time', 440, 0, 190, 290, 'container_blue')
    inp2 = d.node('Input', 480, 40, 110, 30, 'in_box')
    cv2 = d.node('3x3 Conv', 485, 141, 100, 26, 'conv_box')
    relu2 = d.node('ReLU', 495, 245, 80, 24, 'act')
    d.link(inp2, cv2)
    d.link(cv2, relu2)
    d.link(tr, inf, exit='right', entry='left', label='re-param')


@register(CAT, 'Ghost Module')
def ghost_module(d):
    x = d.node('Input X', 0, 85, 80, 36, 'in_box')
    pc = d.node('Primary Conv<br>(1x1)', 110, 85, 100, 36, 'conv_box')
    im = d.node('Intrinsic<br>maps (m)', 240, 85, 90, 36, 'reshape')
    ch = d.node('Cheap ops Φ<br>(DW Conv 3x3)', 370, 150, 120, 36, 'conv_box')
    gh = d.node('Ghost maps', 520, 150, 90, 36, 'reshape')
    cat = d.op('C', 660, 103)
    out = d.node('Output<br>(n = m · s)', 700, 85, 100, 36, 'out_box')
    d.link(x, pc, exit='right', entry='left')
    d.link(pc, im, exit='right', entry='left')
    d.link(im, cat, exit='right', entry='left')
    d.link(im, ch, exit='bottom', entry='left', via=[(im.cx, ch.cy)])
    d.link(ch, gh, exit='right', entry='left')
    d.link(gh, cat, exit='right', entry='bottom', via=[(cat.cx, gh.cy)])
    d.link(cat, out, exit='right', entry='left')
    d.text('identity', 460, 86, 60, 14, 'dim_label')


@register(CAT, 'HRNet Stage')
def hrnet_stage(d):
    rows = [(70, 'C, 1/4'), (150, '2C, 1/8'), (230, '4C, 1/16')]
    d.container('× 4 modules', 95, 25, 400, 265, 'container_repeat')
    fu = d.node('Multi-Resolution<br>Fusion<br><br>↓ strided 3x3 conv<br>↑ upsample + 1x1 conv<br>Σ per resolution',
                330, 40, 150, 220, 'gate')
    for cy, lab in rows:
        i = d.node(lab, 0, cy - 15, 75, 30, 'in_box')
        b1 = d.node('Conv Block', 110, cy - 15, 90, 30, 'conv_box')
        b2 = d.node('Conv Block', 225, cy - 15, 90, 30, 'conv_box')
        o = d.node(lab, 520, cy - 15, 75, 30, 'out_box')
        pos = (cy - fu.y) / fu.h
        d.link(i, b1, exit='right', entry='left')
        d.link(b1, b2, exit='right', entry='left')
        d.link(b2, fu, exit='right', entry='left', entry_pos=pos)
        d.link(fu, o, exit='right', entry='left', exit_pos=pos)
