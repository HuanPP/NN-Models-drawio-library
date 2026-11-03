"""Object detection architectures and building blocks."""

from nnlib.core import register

CAT = 'Object Detection'


@register(CAT, 'DETR')
def detr(d):
    cx, W = 260, 140
    X = cx - W // 2
    img = d.node('Image<br>3 x H<sub>0</sub> x W<sub>0</sub>', X, 40, W, 40, 'in_box')
    bb = d.node('CNN Backbone', X, 108, W, 30, 'module')
    proj = d.node('1 x 1 Conv', X, 166, W, 30, 'conv_box')
    add = d.op('+', cx, 239)
    pe = d.node('Positional<br>Encoding', 50, 219, 110, 40, 'pos')
    enc = d.node('Transformer Encoder<br>(× 6)', X, 282, W, 40, 'module_gen')
    dec = d.node('Transformer Decoder<br>(× 6)', X, 350, W, 40, 'module_gen')
    qry = d.node('Object Queries<br>(N learned)', 50, 350, 110, 40, 'embed')
    ffn = d.node('Shared FFN', X, 418, W, 30, 'fc_box')
    cls = d.node('Class<br>(K + 1)', 145, 484, 100, 40, 'out_box')
    box = d.node('Box<br>(c<sub>x</sub>, c<sub>y</sub>, w, h)', 275, 484, 100, 40, 'out_box')
    loss = d.node('Bipartite<br>Matching Loss', X, 560, W, 40, 'loss')
    gt = d.node('Ground Truth<br>(c<sub>i</sub>, b<sub>i</sub>)', 360, 560, 110, 40, 'in_box')

    for a, b in [(img, bb), (bb, proj), (proj, add), (add, enc), (enc, dec), (dec, ffn)]:
        d.link(a, b)
    d.link(pe, add, exit='right', entry='left')
    d.link(qry, dec, exit='right', entry='left')
    # shared FFN fans out to the two prediction heads
    d.link(ffn, cls, exit='bottom', entry='top', via=[(cx, 466), (cls.cx, 466)])
    d.link(ffn, box, exit='bottom', entry='top', via=[(cx, 466), (box.cx, 466)])
    # both predictions are matched against the ground truth
    d.link(cls, loss, exit='bottom', entry='top', via=[(cls.cx, 542), (cx, 542)])
    d.link(box, loss, exit='bottom', entry='top', via=[(box.cx, 542), (cx, 542)])
    d.link(gt, loss, exit='left', entry='right')

    d.text('C x H x W', 268, 145, 70, 14, 'dim_label')
    d.text('d x H x W', 268, 203, 70, 14, 'dim_label')
    d.text('HW x d', 268, 261, 50, 14, 'dim_label')
    d.text('N x d', 268, 397, 50, 14, 'dim_label')


@register(CAT, 'Mask R-CNN')
def mask_rcnn(d):
    cx, W = 250, 130
    X = cx - W // 2
    img = d.node('Image', X, 40, W, 30, 'in_box')
    bb = d.node('ResNet + FPN', X, 98, W, 30, 'module')
    rpn = d.node('RPN', 345, 98, 90, 30, 'conv_box')
    roi = d.node('RoIAlign<br>(7 x 7 | 14 x 14)', X, 156, W, 40, 'pool_box')
    d.link(img, bb)
    d.link(bb, roi)
    d.link(bb, rpn, exit='right', entry='left')
    d.link(rpn, roi, exit='bottom', entry='right', via=[(rpn.cx, roi.cy)])
    d.text('proposals', 398, 140, 55, 14, 'dim_label')

    # box head (left column) and mask head (right column)
    bc, mc, hw = 150, 350, 130
    y0 = 232
    fc1 = d.node('FC, 1024', bc - hw // 2, y0, hw, 30, 'fc_box')
    fc2 = d.node('FC, 1024', bc - hw // 2, y0 + 54, hw, 30, 'fc_box')
    cls = d.node('Class<br>K + 1', 80, y0 + 120, 60, 40, 'out_box')
    reg = d.node('Box<br>4K', 160, y0 + 120, 60, 40, 'out_box')
    m1 = d.node('Conv 3x3, 256 (× 4)', mc - hw // 2, y0, hw, 30, 'conv_box')
    m2 = d.node('Deconv 2x2, s2', mc - hw // 2, y0 + 54, hw, 30, 'deconv_box')
    m3 = d.node('Conv 1x1, K', mc - hw // 2, y0 + 108, hw, 30, 'conv_box')
    m4 = d.node('Mask<br>K x 28 x 28', mc - hw // 2, y0 + 162, hw, 40, 'out_box')

    split = roi.b + 18
    d.link(roi, fc1, exit='bottom', entry='top', via=[(cx, split), (bc, split)])
    d.link(roi, m1, exit='bottom', entry='top', via=[(cx, split), (mc, split)])
    d.link(fc1, fc2)
    fy = fc2.b + 18
    d.link(fc2, cls, exit='bottom', entry='top', via=[(bc, fy), (cls.cx, fy)])
    d.link(fc2, reg, exit='bottom', entry='top', via=[(bc, fy), (reg.cx, fy)])
    for a, b in [(m1, m2), (m2, m3), (m3, m4)]:
        d.link(a, b)
    d.text('14 x 14 x 256', mc + 8, y0 + 33, 75, 14, 'dim_label')
    d.text('28 x 28 x 256', mc + 8, y0 + 87, 75, 14, 'dim_label')


@register(CAT, 'RetinaNet')
def retinanet(d):
    rows = {7: 50, 6: 110, 5: 170, 4: 230, 3: 290}
    c = {i: d.node(f'C<sub>{i}</sub>', 20, rows[i] - 15, 60, 30, 'conv_box') for i in (3, 4, 5)}
    p = {i: d.node(f'P<sub>{i}</sub>', 120, rows[i] - 15, 60, 30, 'conv_box') for i in (3, 4, 5, 6, 7)}
    d.link(c[3], c[4])
    d.link(c[4], c[5])
    for i in (3, 4, 5):
        d.link(c[i], p[i], exit='right', entry='left')
    d.link(c[5], p[6], exit='top', entry='left', via=[(c[5].cx, rows[6])])
    d.link(p[6], p[7])
    d.link(p[5], p[4])
    d.link(p[4], p[3])
    d.text('ResNet', 20, 318, 60, 16, 'label')
    d.text('FPN', 120, 318, 60, 16, 'label')

    # every pyramid level feeds the same (shared) subnets
    bx = 205
    dot = d.op('', bx, rows[5], size=8, style='dot')
    for i in (7, 6):
        d.link(p[i], dot, exit='right', entry='top', via=[(bx, rows[i])], style='line')
    d.link(p[5], dot, exit='right', entry='left', style='line')
    for i in (4, 3):
        d.link(p[i], dot, exit='right', entry='bottom', via=[(bx, rows[i])], style='line')

    ycls, ybox = 80, 260
    d.container('Class Subnet', 240, ycls - 55, 260, 95, 'container_blue')
    d.container('Box Subnet', 240, ybox - 55, 260, 95, 'container_blue')
    cs1 = d.node('Conv 3x3, 256<br>+ ReLU (× 4)', 255, ycls - 20, 110, 40, 'conv_box')
    cs2 = d.node('Conv 3x3, KA<br>+ Sigmoid', 395, ycls - 20, 90, 40, 'conv_box')
    bs1 = d.node('Conv 3x3, 256<br>+ ReLU (× 4)', 255, ybox - 20, 110, 40, 'conv_box')
    bs2 = d.node('Conv 3x3, 4A', 395, ybox - 20, 90, 40, 'conv_box')
    oc = d.node('Class<br>W x H x KA', 530, ycls - 20, 80, 40, 'out_box')
    ob = d.node('Box<br>W x H x 4A', 530, ybox - 20, 80, 40, 'out_box')
    loss = d.node('Focal Loss (cls)<br>Smooth L1 (box)', 510, rows[5] - 20, 120, 40, 'loss')

    tx = 225
    d.link(dot, cs1, exit='right', entry='left', via=[(tx, rows[5]), (tx, ycls)])
    d.link(dot, bs1, exit='right', entry='left', via=[(tx, rows[5]), (tx, ybox)])
    for a, b in [(cs1, cs2), (cs2, oc), (bs1, bs2), (bs2, ob)]:
        d.link(a, b, exit='right', entry='left')
    d.link(oc, loss, style='loss_flow')
    d.link(ob, loss, style='loss_flow')
    d.text('shared across P<sub>3</sub> - P<sub>7</sub>', 240, rows[5] - 7, 110, 14, 'dim_label')


@register(CAT, 'SSD')
def ssd(d):
    cy, w = 140, 75
    xs = [20, 120, 220, 320, 420, 520, 620]
    hs = [130, 110, 90, 70, 50, 40, 30]
    names = ['Image', 'VGG-16<br>Conv4_3', 'Conv7<br>(FC7)', 'Conv8_2', 'Conv9_2', 'Conv10_2', 'Conv11_2']
    dims = ['300 x 300 x 3', '38 x 38 x 512', '19 x 19 x 1024', '10 x 10 x 512',
            '5 x 5 x 256', '3 x 3 x 256', '1 x 1 x 256']
    feats = []
    for i, (x, h, nm, dm) in enumerate(zip(xs, hs, names, dims)):
        feats.append(d.node(nm, x, cy - h // 2, w, h, 'input' if i == 0 else 'conv'))
        d.text(dm, x - 7.5, 52, 90, 14, 'dim_label')
    for a, b in zip(feats, feats[1:]):
        d.link(a, b, exit='right', entry='left')
    d.text('VGG-16 base', 120, 25, 175, 16, 'label')
    d.text('Extra feature layers', 320, 25, 375, 16, 'label')

    heads = [d.node('Conv 3x3<br>k (c + 4)', x, 235, w, 40, 'conv_box') for x in xs[1:]]
    for f, hd in zip(feats[1:], heads):
        d.link(f, hd)

    drop = heads[1].cx
    cat = d.node('Concat<br>8732 per class', drop - 57.5, 315, 115, 40, 'reshape')
    nms = d.node('Non-Max<br>Suppression', 345, 315, 90, 40, 'gate')
    out = d.node('Detections', 465, 315, 100, 40, 'out_box')
    for hd in heads:
        if hd.cx == drop:
            d.link(hd, cat)
        else:
            d.link(hd, cat, exit='bottom', entry='top', via=[(hd.cx, 295), (drop, 295)])
    d.link(cat, nms, exit='right', entry='left')
    d.link(nms, out, exit='right', entry='left')


@register(CAT, 'YOLOv8')
def yolov8(d):
    # ---- backbone (left column, top to bottom)
    bx, bw = 20, 120
    labels = ['Input 640 x 640', 'Conv, s2 (P1/2)', 'Conv, s2 (P2/4)', 'C2f (n=3)',
              'Conv, s2 (P3/8)', 'C2f (n=6)']
    styles = ['in_box', 'conv_box', 'conv_box', 'module_gen', 'conv_box', 'module_gen']
    bb = [d.node(t, bx, 20 + 50 * i, bw, 26, s) for i, (t, s) in enumerate(zip(labels, styles))]
    bb.append(d.node('Conv, s2 (P4/16)', bx, 320, bw, 26, 'conv_box'))
    bb.append(d.node('C2f (n=6)', bx, 370, bw, 26, 'module_gen'))
    bb.append(d.node('Conv, s2 (P5/32)', bx, 420, bw, 26, 'conv_box'))
    bb.append(d.node('C2f (n=3)', bx, 470, bw, 26, 'module_gen'))
    for a, b in zip(bb, bb[1:]):
        d.link(a, b)
    b3, b4, b5 = bb[5], bb[7], bb[9]
    r3, r4, r5 = b3.cy, b4.cy, b5.cy          # pyramid level rows

    # ---- neck columns
    c1, c2, c3, c4 = 205, 305, 405, 505
    sppf = d.node('SPPF', c1 - 35, r5 - 13, 70, 26, 'pool_box')
    up45 = d.node('Upsample', c1 - 35, 420, 70, 26, 'deconv_box')
    cat_t4 = d.op('C', c1, r4)
    up34 = d.node('Upsample', c1 - 35, 320, 70, 26, 'deconv_box')
    cat_t3 = d.op('C', c1, r3)
    t4 = d.node('C2f', c2 - 35, r4 - 13, 70, 26, 'module_gen')
    t3 = d.node('C2f', c2 - 35, r3 - 13, 70, 26, 'module_gen')
    ds34 = d.node('Conv, s2', c3 - 35, 320, 70, 26, 'conv_box')
    cat_u4 = d.op('C', c3, r4)
    ds45 = d.node('Conv, s2', c3 - 35, 420, 70, 26, 'conv_box')
    cat_u5 = d.op('C', c3, r5)
    u4 = d.node('C2f', c4 - 35, r4 - 13, 70, 26, 'module_gen')
    u5 = d.node('C2f', c4 - 35, r5 - 13, 70, 26, 'module_gen')
    h3 = d.node('Detect P3<br>80 x 80', 570, r3 - 20, 80, 40, 'fc_box')
    h4 = d.node('Detect P4<br>40 x 40', 570, r4 - 20, 80, 40, 'fc_box')
    h5 = d.node('Detect P5<br>20 x 20', 570, r5 - 20, 80, 40, 'fc_box')

    R = dict(exit='right', entry='left')
    d.link(b3, cat_t3, **R)
    d.link(b4, cat_t4, **R)
    d.link(b5, sppf, **R)
    # top-down path
    d.link(sppf, up45)
    d.link(up45, cat_t4)
    d.link(cat_t4, t4, **R)
    d.link(t4, up34, exit='top', entry='right', via=[(c2, up34.cy)])
    d.link(up34, cat_t3)
    d.link(cat_t3, t3, **R)
    # bottom-up path
    d.link(t3, h3, **R)
    d.link(t3, ds34, exit='right', entry='top', via=[(c3, r3)])
    d.link(ds34, cat_u4)
    d.link(t4, cat_u4, **R)
    d.link(cat_u4, u4, **R)
    d.link(u4, h4, **R)
    d.link(u4, ds45, exit='bottom', entry='right', via=[(c4, ds45.cy)])
    d.link(ds45, cat_u5)
    d.link(sppf, cat_u5, **R)
    d.link(cat_u5, u5, **R)
    d.link(u5, h5, **R)

    # ---- detail of one decoupled, anchor-free Detect head
    d.container('Decoupled Head (Detect)', 170, 20, 480, 205, 'container_orange')
    yb, yc = 95, 185
    pin = d.node('P<sub>i</sub>', 190, 127, 40, 26, 'in_box')
    b1 = d.node('Conv 3x3', 270, yb - 13, 60, 26, 'conv_box')
    b2 = d.node('Conv 3x3', 350, yb - 13, 60, 26, 'conv_box')
    b3x = d.node('Conv 1x1<br>4 x 16', 430, yb - 20, 70, 40, 'conv_box')
    dfl = d.node('DFL', 520, yb - 13, 45, 26, 'reshape')
    obox = d.node('Box', 585, yb - 13, 45, 26, 'out_box')
    k1 = d.node('Conv 3x3', 270, yc - 13, 60, 26, 'conv_box')
    k2 = d.node('Conv 3x3', 350, yc - 13, 60, 26, 'conv_box')
    k3 = d.node('Conv 1x1<br>nc', 430, yc - 20, 70, 40, 'conv_box')
    ocls = d.node('Cls', 585, yc - 13, 45, 26, 'out_box')
    d.link(pin, b1, exit='right', entry='left', via=[(250, pin.cy), (250, yb)])
    d.link(pin, k1, exit='right', entry='left', via=[(250, pin.cy), (250, yc)])
    for a, b in [(b1, b2), (b2, b3x), (b3x, dfl), (dfl, obox), (k1, k2), (k2, k3), (k3, ocls)]:
        d.link(a, b, **R)


@register(CAT, 'C2f Block')
def c2f_block(d):
    cx, W = 200, 130
    X = cx - W // 2
    inp = d.node('Input<br>c<sub>1</sub> x H x W', X, 20, W, 36, 'in_box')
    cv1 = d.node('Conv 1x1, 2c', X, 80, W, 30, 'conv_box')
    split = d.node('Split', X, 134, W, 30, 'reshape')
    b1 = d.node('Bottleneck 1', X, 192, W, 30, 'module')
    b2 = d.node('Bottleneck 2', X, 246, W, 30, 'module')
    bn = d.node('Bottleneck n', X, 336, W, 30, 'module')
    cat = d.op('C', cx, 405)
    cv2 = d.node('Conv 1x1, c<sub>2</sub>', X, 445, W, 30, 'conv_box')
    out = d.node('Output<br>c<sub>2</sub> x H x W', X, 499, W, 36, 'out_box')
    for a, b in [(inp, cv1), (cv1, split), (split, b1), (b1, b2), (b2, bn), (bn, cat), (cat, cv2), (cv2, out)]:
        d.link(a, b)
    d.text('⋮', 208, 294, 24, 24, 'ellipsis')
    # first half bypasses all bottlenecks
    d.link(split, cat, exit='left', entry='left', via=[(110, split.cy), (110, cat.cy)])
    # second half and every bottleneck output are also concatenated
    for s in (split, b1, b2):
        d.link(s, cat, exit='right', entry='right', via=[(290, s.cy), (290, cat.cy)])
    d.text('2c', 208, 115, 30, 14, 'dim_label')
    d.text('c', 208, 170, 20, 14, 'dim_label')
    d.text('c', 84, 270, 18, 14, 'dim_label')
    d.text('(2 + n) c', 208, 426, 60, 14, 'dim_label')

    # Bottleneck detail
    d.container('Bottleneck', 330, 130, 175, 295, 'container_blue')
    bx_ = d.node('x (c)', 355, 170, 100, 26, 'in_box')
    bc1 = d.node('Conv 3x3', 355, 220, 100, 30, 'conv_box')
    bc2 = d.node('Conv 3x3', 355, 274, 100, 30, 'conv_box')
    badd = d.op('+', 405, 343)
    by = d.node('y (c)', 355, 382, 100, 26, 'out_box')
    for a, b in [(bx_, bc1), (bc1, bc2), (bc2, badd), (badd, by)]:
        d.link(a, b)
    d.link(bx_, badd, exit='right', entry='right', via=[(480, bx_.cy), (480, badd.cy)], style='skip')


@register(CAT, 'PANet')
def panet(d):
    rows = {5: 60, 4: 160, 3: 260, 2: 360}
    mids = {5: 110, 4: 210, 3: 310}      # stride-2 conv producing level i from level i-1
    d.container('FPN', 5, 10, 190, 380, 'container_blue')
    d.container('Bottom-up Path', 205, 10, 200, 380, 'container_green')
    d.container('Adaptive Pooling', 415, 10, 160, 380, 'container_orange')

    C = {i: d.node(f'C<sub>{i}</sub>', 20, rows[i] - 15, 60, 30, 'conv_box') for i in rows}
    P = {i: d.node(f'P<sub>{i}</sub>', 120, rows[i] - 15, 60, 30, 'conv_box') for i in rows}
    N = {i: d.node(f'N<sub>{i}</sub>', 330, rows[i] - 15, 60, 30, 'conv_box') for i in rows}
    A = {i: d.op('+', 260, rows[i]) for i in (3, 4, 5)}
    S = {i: d.node('Conv 3x3, s2', 220, mids[i] - 13, 80, 26, 'conv_box') for i in (3, 4, 5)}
    Rr = {i: d.node('RoIAlign', 430, rows[i] - 15, 80, 30, 'pool_box') for i in rows}
    R = dict(exit='right', entry='left')

    for i in (2, 3, 4):
        d.link(C[i], C[i + 1])            # backbone, bottom-up
    for i in (5, 4, 3):
        d.link(P[i], P[i - 1])            # top-down (2x upsample + add)
    for i in rows:
        d.link(C[i], P[i], **R)           # lateral 1x1
    d.link(P[2], N[2], **R)               # N2 = P2
    for i in (3, 4, 5):
        d.link(P[i], A[i], **R)
        d.link(A[i], N[i], **R)
        d.link(N[i - 1], S[i], exit='top', entry='right', via=[(N[i - 1].cx, mids[i])])
        d.link(S[i], A[i])
    for i in rows:
        d.link(N[i], Rr[i], **R)

    bus = 540
    fuse = d.op('max', bus, 210, style='op_sm')
    for i in (5, 4):
        d.link(Rr[i], fuse, exit='right', entry='top', via=[(bus, rows[i])])
    for i in (3, 2):
        d.link(Rr[i], fuse, exit='right', entry='bottom', via=[(bus, rows[i])])
    head = d.node('Box Branch<br>(cls, box)', 605, 190, 90, 40, 'fc_box')
    d.link(fuse, head, **R)


@register(CAT, 'CenterNet')
def centernet(d):
    cy = 140
    img = d.node('Image<br>3 x H x W', 20, cy - 20, 80, 40, 'in_box')
    bb = d.node('Backbone<br>(ResNet)', 130, cy - 20, 110, 40, 'module')
    up = d.node('Upsample<br>3 x Deconv', 270, cy - 20, 100, 40, 'deconv_box')
    d.text('512 x H/32 x W/32', 130, cy + 25, 110, 14, 'dim_label')
    d.text('64 x H/4 x W/4', 270, cy + 25, 100, 14, 'dim_label')
    d.link(img, bb, exit='right', entry='left')
    d.link(bb, up, exit='right', entry='left')

    hm = d.node('Heatmap<br>K x H/4 x W/4', 410, 50, 110, 40, 'fc_box')
    sz = d.node('Size (w, h)<br>2 x H/4 x W/4', 410, 120, 110, 40, 'fc_box')
    off = d.node('Offset<br>2 x H/4 x W/4', 410, 190, 110, 40, 'fc_box')
    d.text('each head: Conv 3x3, ReLU, Conv 1x1', 380, 240, 170, 14, 'dim_label')
    tx = 390
    d.link(up, hm, exit='right', entry='left', via=[(tx, cy), (tx, hm.cy)])
    d.link(up, sz, exit='right', entry='left')
    d.link(up, off, exit='right', entry='left', via=[(tx, cy), (tx, off.cy)])

    peak = d.node('Peak Extraction<br>(3x3 max-pool)', 560, 50, 110, 40, 'gate')
    boxes = d.node('Boxes<br>(top-100 peaks)', 560, 120, 110, 40, 'out_box')
    d.link(hm, peak, exit='right', entry='left')
    d.link(peak, boxes)
    d.link(sz, boxes, exit='right', entry='left')
    d.link(off, boxes, exit='right', entry='bottom', via=[(boxes.cx, off.cy)])
