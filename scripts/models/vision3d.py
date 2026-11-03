"""3D vision and neural rendering: point-cloud networks, voxel / multi-view CNNs, NeRF, 3DGS."""

from nnlib.core import register

CAT = '3D Vision & Neural Rendering'


@register(CAT, 'PointNet')
def pointnet(d):
    # classification column (A) and segmentation column (B)
    ax, bx, w = 75, 285, 150
    inp = d.node('Input Points<br>n x 3', ax, 40, w, 40, 'in_box')
    itr = d.node('Input Transform<br>(T-Net 3 x 3)', ax, 105, w, 40, 'module')
    mlp1 = d.node('Shared MLP (64, 64)', ax, 170, w, 30, 'fc_box')
    ftr = d.node('Feature Transform<br>(T-Net 64 x 64)', ax, 225, w, 40, 'module')
    mlp2 = d.node('Shared MLP<br>(64, 128, 1024)', ax, 290, w, 40, 'fc_box')
    pool = d.node('Max Pool', ax, 355, w, 30, 'pool_box')
    glob = d.node('Global Feature (1024)', ax, 410, w, 30, 'reshape')
    mlp3 = d.node('MLP<br>(512, 256, k)', ax, 465, w, 40, 'fc_box')
    cls = d.node('Class Scores<br>k', ax, 530, w, 40, 'out_box')
    for a, b in [(inp, itr), (itr, mlp1), (mlp1, ftr), (ftr, mlp2), (mlp2, pool),
                 (pool, glob), (glob, mlp3), (mlp3, cls)]:
        d.link(a, b)

    bcx = bx + w / 2
    cat = d.op('C', bcx, glob.cy)
    seg1 = d.node('Shared MLP<br>(512, 256, 128)', bx, 465, w, 40, 'fc_box')
    seg2 = d.node('Shared MLP<br>(128, m)', bx, 530, w, 40, 'fc_box')
    seg = d.node('Per-point Scores<br>n x m', bx, 595, w, 40, 'out_box')
    d.link(ftr, cat, exit='right', entry='top', via=[(bcx, ftr.cy)])
    d.link(glob, cat, exit='right', entry='left')
    for a, b in [(cat, seg1), (seg1, seg2), (seg2, seg)]:
        d.link(a, b)

    lx = inp.cx + 6
    d.text('n x 3', lx, 150, 50, 14, 'dim_label')
    d.text('n x 64', lx, 205, 50, 14, 'dim_label')
    d.text('n x 64', lx, 270, 50, 14, 'dim_label')
    d.text('n x 1024', lx, 335, 50, 14, 'dim_label')
    d.text('local: n x 64', 245, 228, 80, 14, 'dim_label')
    d.text('n x 1088', bcx + 6, 445, 50, 14, 'dim_label')
    d.text('Classification', ax, 580, w, 20, 'caption')
    d.text('Segmentation', bx, 645, w, 20, 'caption')


@register(CAT, 'PointNet++ Set Abstraction')
def pointnet_pp(d):
    # ---- top: one set abstraction level in detail ----
    d.container('Set Abstraction Level', 20, 20, 680, 150, 'container_blue')
    bw, by, bh = 100, 90, 40
    xs = [50, 180, 310, 440, 570]
    inp = d.node('Points<br>N x (d + C)', xs[0], by, bw, bh, 'in_box')
    smp = d.node('Sampling<br>(FPS)', xs[1], by, bw, bh, 'pool_box')
    grp = d.node('Grouping<br>(ball query)', xs[2], by, bw, bh, 'reshape')
    pn = d.node('PointNet<br>(MLP + max)', xs[3], by, bw, bh, 'fc_box')
    out = d.node("Features<br>N' x (d + C')", xs[4], by, bw, bh, 'out_box')
    for a, b in [(inp, smp), (smp, grp), (grp, pn), (pn, out)]:
        d.link(a, b)
    d.link(inp, grp, exit='top', entry='top', via=[(inp.cx, 70), (grp.cx, 70)], label='all points')
    d.text("N' x d centroids", smp.x - 5, 137, bw + 10, 14, 'dim_label')
    d.text("N' x K x (d + C)", grp.x - 5, 137, bw + 10, 14, 'dim_label')
    d.text("N' x C'", pn.x, 137, bw, 14, 'dim_label')

    # ---- bottom: hierarchical network with classification and segmentation heads ----
    d.container('Hierarchical Point Set Feature Learning', 20, 185, 680, 285, 'container_gray')
    cw, ch = 120, 40
    cx = [60, 220, 380, 540]
    r1, r2, r3 = 230, 320, 410
    pin = d.node('Points<br>N x (d + C)', cx[0], r1, cw, ch, 'in_box')
    sa1 = d.node('SA Level 1<br>N<sub>1</sub> x (d + C<sub>1</sub>)', cx[1], r1, cw, ch, 'module')
    sa2 = d.node('SA Level 2<br>N<sub>2</sub> x (d + C<sub>2</sub>)', cx[2], r1, cw, ch, 'module')
    sa3 = d.node('SA (group all)<br>1 x C<sub>3</sub>', cx[3], r1, cw, ch, 'module')
    fc = d.node('FC Layers<br>(512, 256, k)', cx[3], r2, cw, ch, 'fc_box')
    cls = d.node('Class Scores<br>k', cx[3], r3, cw, ch, 'out_box')
    fp2 = d.node('Feature Prop.<br>N<sub>1</sub> x C<sub>4</sub>', cx[1], r2, cw, ch, 'module_alt')
    fp1 = d.node('Feature Prop.<br>N x C<sub>5</sub>', cx[0], r2, cw, ch, 'module_alt')
    seg = d.node('Per-point Scores<br>N x m', cx[0], r3, cw, ch, 'out_box')
    for a, b in [(pin, sa1), (sa1, sa2), (sa2, sa3), (sa3, fc), (fc, cls), (fp1, seg)]:
        d.link(a, b)
    d.link(sa2, fp2, exit='bottom', entry='right', via=[(sa2.cx, fp2.cy)])
    d.link(fp2, fp1, exit='left', entry='right')
    d.link(sa1, fp2, style='skip')
    d.link(pin, fp1, style='skip')
    d.text('skip link<br>concat', sa1.cx + 6, 278, 60, 28, 'dim_label')
    d.text('skip link<br>concat', pin.cx + 6, 278, 60, 28, 'dim_label')
    d.text('interpolate', sa2.cx - 75, 322, 65, 14, 'dim_label')
    d.text('Segmentation', cx[1] - 5, r3 + 10, cw + 10, 20, 'caption')
    d.text('Classification', cx[2], r3 + 10, cw + 10, 20, 'caption')


@register(CAT, 'DGCNN EdgeConv')
def dgcnn_edgeconv(d):
    cx, w = 200, 140
    inp = d.node('Point Features<br>n x F', cx - w / 2, 20, w, 40, 'in_box')
    d.container('EdgeConv Layer', 50, 80, 300, 350, 'container_blue')
    knn = d.node('k-NN Graph<br>(feature space)', cx - w / 2, 120, w, 40, 'gate')
    bw = 120
    xi = d.node('x<sub>i</sub><br>(central point)', 125 - bw / 2, 195, bw, 40, 'reshape')
    xj = d.node('x<sub>j</sub> - x<sub>i</sub><br>(edge vector)', 275 - bw / 2, 195, bw, 40, 'reshape')
    cat = d.op('C', cx, 265)
    mlp = d.node('Shared MLP h<sub>Θ</sub><br>(a<sub>1</sub>, ..., a<sub>L</sub>)', cx - w / 2, 305, w, 40, 'fc_box')
    mx = d.node('Max Aggregation<br>over k neighbours', cx - w / 2, 370, w, 40, 'pool_box')
    out = d.node('New Point Features<br>n x a<sub>L</sub>', cx - w / 2, 450, w, 40, 'out_box')

    d.link(inp, knn)
    d.link(knn, xi, via=[(cx, 178), (xi.cx, 178)])
    d.link(knn, xj, via=[(cx, 178), (xj.cx, 178)])
    d.link(xi, cat, exit='bottom', entry='left', via=[(xi.cx, cat.cy)])
    d.link(xj, cat, exit='bottom', entry='right', via=[(xj.cx, cat.cy)])
    for a, b in [(cat, mlp), (mlp, mx), (mx, out)]:
        d.link(a, b)
    d.link(out, knn, exit='right', entry='right', via=[(380, out.cy), (380, knn.cy)], style='skip')

    d.text('n x k x 2F', cx + 6, 285, 60, 14, 'dim_label')
    d.text('n x k x a<sub>L</sub>', cx + 6, 350, 60, 14, 'dim_label')
    d.text('dynamic graph:<br>recompute k-NN<br>in next layer', 388, 260, 85, 42, 'dim_label')


@register(CAT, 'NeRF')
def nerf(d):
    ax, aw = 160, 160          # MLP column
    acx = ax + aw / 2
    bx, bw = 360, 160          # rendering column
    bcx = bx + bw / 2
    smp = d.node('Ray Sampling<br>(x, y, z, θ, φ)', ax, 20, aw, 40, 'in_box')
    gx = d.node('γ(x)  (60-D)', ax, 90, aw, 30, 'pos')
    gd = d.node('γ(d)  (24-D)', 10, 90, 90, 30, 'pos')
    fc5 = d.node('5 × FC-256 + ReLU', ax, 145, aw, 30, 'fc_box')
    cat1 = d.op('C', acx, 210)
    fc3 = d.node('3 × FC-256 + ReLU', ax, 250, aw, 30, 'fc_box')
    feat = d.node('Feature (256-D)', ax, 310, aw, 30, 'fc_box')
    sig = d.node('Density σ', bx, 310, bw, 30, 'out_box')
    cat2 = d.op('C', acx, 375)
    fc1 = d.node('FC-128 + ReLU', ax, 415, aw, 30, 'fc_box')
    rgb = d.node('RGB c (sigmoid)', ax, 470, aw, 30, 'out_box')
    vr = d.node('Volume Rendering<br>Ĉ(r) = Σ T<sub>i</sub> α<sub>i</sub> c<sub>i</sub>', bx, 465, bw, 40, 'module')
    gt = d.node('Ground-Truth Pixel<br>C(r)', ax, 535, aw, 40, 'in_box')
    loss = d.node('Rendering Loss<br>||Ĉ(r) - C(r)||²', bx, 535, bw, 40, 'loss')

    d.link(smp, gx)
    d.link(smp, gd, via=[(acx, 75), (gd.cx, 75)])
    d.link(gx, fc5)
    d.link(gx, cat1, exit='left', entry='left', via=[(135, gx.cy), (135, cat1.cy)], style='skip')
    for a, b in [(fc5, cat1), (cat1, fc3), (fc3, feat), (feat, cat2), (cat2, fc1), (fc1, rgb)]:
        d.link(a, b)
    d.link(fc3, sig, via=[(acx, 295), (bcx, 295)])
    d.link(gd, cat2, exit='bottom', entry='left', via=[(gd.cx, cat2.cy)])
    d.link(sig, vr)
    d.link(rgb, vr, exit='right', entry='left')
    d.link(vr, loss)
    d.link(gt, loss, exit='right', entry='left')

    d.text('x', acx + 6, 68, 20, 14, 'dim_label')
    d.text('d', gd.cx + 6, 125, 20, 14, 'dim_label')
    d.text('skip', 100, 160, 28, 14, 'dim_label')
    d.text('pixel colour Ĉ(r)', bcx + 6, 513, 90, 14, 'dim_label')


@register(CAT, '3D Gaussian Splatting')
def gaussian_splatting(d):
    w, h = 110, 40
    c = [70, 210, 350, 490, 630]
    r0, r1, r2 = 20, 110, 200
    adc = d.node('Adaptive Density<br>Control', c[1] - w / 2, r0, w, h, 'gate')
    cam = d.node('Camera<br>(pose, intrinsics)', c[2] - w / 2, r0, w, h, 'pos')
    sfm = d.node('SfM Points<br>(initialization)', c[0] - w / 2, r1, w, h, 'in_box')
    gs = d.node('3D Gaussians<br>μ, Σ, α, SH', c[1] - w / 2, r1, w, h, 'embed')
    prj = d.node('Projection<br>(3D → 2D)', c[2] - w / 2, r1, w, h, 'reshape')
    ras = d.node('Differentiable<br>Tile Rasterizer', c[3] - w / 2, r1, w, h, 'module')
    img = d.node('Rendered<br>Image', c[4] - w / 2, r1, w, h, 'out_box')
    gt = d.node('Ground-Truth<br>Image', c[3] - w / 2, r2, w, h, 'in_box')
    loss = d.node('L1 + D-SSIM<br>Loss', c[4] - w / 2, r2, w, h, 'loss')

    for a, b in [(sfm, gs), (gs, prj), (prj, ras), (ras, img)]:
        d.link(a, b, exit='right', entry='left')
    d.link(cam, prj)
    d.link(img, loss)
    d.link(gt, loss, exit='right', entry='left')
    # density control loop: ADC -> Gaussians (left), Gaussians -> ADC (right)
    pl, pr = 0.5 - 20 / w, 0.5 + 20 / w
    d.link(adc, gs, exit='bottom', entry='top', exit_pos=pl, entry_pos=pl)
    d.link(gs, adc, exit='top', entry='bottom', exit_pos=pr, entry_pos=pr)
    # gradient flow back to the Gaussian parameters
    d.link(loss, gs, exit='bottom', entry='bottom', via=[(loss.cx, 270), (gs.cx, 270)],
           style='grad', label='gradient flow')

    d.text('clone / split / prune', 75, 78, 110, 14, 'dim_label')
    d.text('view-space<br>gradients', 237, 71, 55, 28, 'dim_label')


@register(CAT, 'VoxNet')
def voxnet(d):
    x, w, h = 40, 130, 55
    ys = [20, 105, 190, 275, 360, 445]
    inp = d.node('Occupancy Grid<br>32 x 32 x 32', x, ys[0], w, h, 'input')
    c1 = d.node('Conv3D<br>32 @ 5³, s 2', x, ys[1], w, h, 'conv')
    c2 = d.node('Conv3D<br>32 @ 3³', x, ys[2], w, h, 'conv')
    p = d.node('MaxPool3D<br>2³', x, ys[3], w, h, 'pool')
    f1 = d.node('FC 128', x, ys[4], w, h, 'fc')
    out = d.node('FC K<br>(class scores)', x, ys[5], w, h, 'output')
    nodes = [inp, c1, c2, p, f1, out]
    for a, b in zip(nodes, nodes[1:]):
        d.link(a, b)
    dims = ['1 x 32³', '32 x 14³', '32 x 12³', '32 x 6³', '128']
    for a, t in zip(nodes, dims):
        d.text(t, a.cx + 6, a.b + 8, 60, 14, 'dim_label')


@register(CAT, 'Multi-View CNN')
def multi_view_cnn(d):
    rows = [40, 90, 140, 190]          # view 1, view 2, ellipsis, view 12
    rh = 30
    mid = (rows[0] + rows[-1] + rh) / 2
    shape = d.node('3D Shape<br>(mesh)', 20, mid - 20, 80, 40, 'in_box')
    d.container('CNN<sub>1</sub> (shared)', 245, 5, 120, 230, 'container_blue')
    pool = d.node('View<br>Pooling<br>(max)', 400, rows[0], 70, rows[-1] + rh - rows[0], 'pool_box')
    cnn2 = d.node('CNN<sub>2</sub>', 510, mid - 15, 80, 30, 'conv_box')
    out = d.node('Class Scores<br>(softmax)', 630, mid - 20, 90, 40, 'out_box')
    names = ['View 1', 'View 2', None, 'View 12']
    for y, nm in zip(rows, names):
        if nm is None:
            d.text('⋮', 140, y, 80, rh, 'ellipsis')
            d.text('⋮', 260, y, 90, rh, 'ellipsis')
            continue
        v = d.node(nm, 140, y, 80, rh, 'reshape')
        c1 = d.node('CNN<sub>1</sub>', 260, y, 90, rh, 'conv_box')
        d.link(shape, v, exit='right', entry='left', via=[(120, shape.cy), (120, v.cy)])
        d.link(v, c1, exit='right', entry='left')
        d.link(c1, pool, exit='right', entry='left', entry_pos=(v.cy - pool.y) / pool.h)
    d.link(pool, cnn2, exit='right', entry='left')
    d.link(cnn2, out, exit='right', entry='left')
    d.text('rendered views', 130, 15, 100, 14, 'dim_label')
