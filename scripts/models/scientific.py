"""Scientific machine learning: physics-informed and operator-learning models."""

from nnlib.core import register

CAT = 'Scientific ML'


@register(CAT, 'PINN')
def pinn(d):
    xt = d.node('(x, t)', 0, 70, 80, 36, 'in_box')
    nn = d.node('Fully connected<br>network u<sub>θ</sub><br>(tanh)', 110, 55, 120, 66, 'fc_box')
    u = d.node('u(x, t)', 260, 70, 90, 36, 'out_box')
    ad = d.node('Autodiff<br>u<sub>t</sub>, u<sub>x</sub>, u<sub>xx</sub>', 380, 70, 130, 36, 'gate')
    res = d.node('PDE residual<br>f = u<sub>t</sub> + N[u]', 540, 70, 140, 36, 'act')
    for a, b in [(xt, nn), (nn, u), (u, ad), (ad, res)]:
        d.link(a, b, exit='right', entry='left')
    lp = d.node('L<sub>PDE</sub><br>(collocation points)', 540, 150, 140, 36, 'loss')
    d.link(res, lp)
    lb = d.node('L<sub>IC/BC</sub>', 240, 150, 130, 36, 'loss')
    bc = d.node('IC / BC<br>targets', 390, 150, 120, 36, 'in_box')
    d.link(u, lb, entry_pos=(u.cx - lb.x) / lb.w)
    d.link(bc, lb, exit='left', entry='right')
    tot = d.node('L = L<sub>PDE</sub> + L<sub>IC/BC</sub>', 380, 230, 160, 36, 'loss')
    d.link(lb, tot, exit='bottom', entry='left', via=[(u.cx, tot.cy)], exit_pos=(u.cx - lb.x) / lb.w)
    d.link(lp, tot, exit='bottom', entry='right', via=[(lp.cx, tot.cy)])
    opt = d.node('Optimizer<br>(Adam / L-BFGS)', 400, 300, 120, 36, 'act')
    d.link(tot, opt)
    d.link(opt, nn, exit='left', entry='bottom', via=[(nn.cx, opt.cy)], style='grad', label='update θ')


@register(CAT, 'Neural ODE')
def neural_ode(d):
    f = d.node('f<sub>θ</sub>(h(t), t)<br>dynamics network', 150, 20, 160, 40, 'fc_box')
    h0 = d.node('h(t<sub>0</sub>)', 0, 120, 90, 36, 'in_box')
    sv = d.node('ODE solver<br>h(t<sub>1</sub>) = h(t<sub>0</sub>) + ∫ f<sub>θ</sub> dt', 130, 110, 200, 56, 'module')
    h1 = d.node('h(t<sub>1</sub>)', 370, 120, 90, 36, 'out_box')
    loss = d.node('L', 500, 120, 60, 36, 'loss')
    for a, b in [(h0, sv), (sv, h1), (h1, loss)]:
        d.link(a, b, exit='right', entry='left')
    d.link(sv, f, exit='top', entry='bottom', exit_pos=0.3, entry_pos=0.25)
    d.link(f, sv, exit='bottom', entry='top', exit_pos=0.75, entry_pos=0.7)
    d.text('h(t), t', 130, 75, 55, 14, 'dim_label')
    d.text('dh/dt', 275, 75, 40, 14, 'dim_label')
    adj = d.node('Adjoint ODE solve<br>(backward, t<sub>1</sub> → t<sub>0</sub>)', 130, 210, 200, 46, 'gate')
    gr = d.node('∂L/∂θ,<br>∂L/∂h(t<sub>0</sub>)', 0, 215, 100, 36, 'reshape')
    d.link(loss, adj, exit='bottom', entry='right', via=[(loss.cx, adj.cy)], style='grad')
    d.link(adj, gr, exit='left', entry='right', style='grad')
    d.text('constant memory in the number of solver steps', 110, 265, 240, 14, 'dim_label')


@register(CAT, 'KAN Layer')
def kan_layer(d):
    for x0, title, edge_style, node_label in [(0, 'MLP', 'act', 'w'), (290, 'KAN', 'gate', 'φ')]:
        d.container(title, x0, 0, 260, 300, 'container_gray' if title == 'MLP' else 'container_purple')
        cxs = [x0 + 50, x0 + 130, x0 + 210]
        s = d.op('Σ', x0 + 130, 170)
        for k, cx in enumerate(cxs, 1):
            xi = d.node(f'x<sub>{k}</sub>', cx - 25, 45, 50, 28, 'in_box')
            if title == 'MLP':
                e = d.node(f'w<sub>{k}</sub>', cx - 25, 100, 50, 26, edge_style)
            else:
                e = d.node(f'φ<sub>{k}</sub>(·)', cx - 30, 98, 60, 30, edge_style)
            d.link(xi, e)
            side = 'left' if k == 1 else ('top' if k == 2 else 'right')
            if side == 'top':
                d.link(e, s)
            else:
                d.link(e, s, exit='bottom', entry=side, via=[(cx, s.cy)])
        if title == 'MLP':
            a = d.node('σ (fixed)', x0 + 95, 205, 70, 26, 'act')
            d.link(s, a)
            y = d.node('y', x0 + 100, 255, 60, 28, 'out_box')
            d.link(a, y)
        else:
            y = d.node('y', x0 + 100, 255, 60, 28, 'out_box')
            d.link(s, y)
    d.text('fixed activation on nodes,<br>learnable linear weights on edges', 0, 310, 260, 28, 'dim_label')
    d.text('learnable univariate functions φ<br>(B-spline + SiLU) on edges, sum on nodes', 290, 310, 260, 28, 'dim_label')


@register(CAT, 'Fourier Neural Operator')
def fno(d):
    a = d.node('a(x)', 0, 95, 70, 36, 'in_box')
    p = d.node('Lift P', 95, 100, 60, 26, 'fc_box')
    d.link(a, p, exit='right', entry='left')
    d.container('× T Fourier layers', 180, 25, 470, 170, 'container_repeat')
    j = d.op('', 210, 113, 8, 'dot')
    d.link(p, j, exit='right', entry='left', style='line')
    fft = d.node('FFT', 250, 50, 70, 26, 'reshape')
    r = d.node('R: keep k<sub>max</sub> modes', 335, 50, 130, 26, 'gate')
    ifft = d.node('IFFT', 480, 50, 60, 26, 'reshape')
    w = d.node('W: local linear', 335, 140, 130, 26, 'fc_box')
    add = d.op('+', 565, 113)
    sg = d.node('σ', 595, 100, 40, 26, 'act')
    d.link(j, fft, exit='top', entry='left', via=[(j.cx, fft.cy)])
    d.link(j, w, exit='bottom', entry='left', via=[(j.cx, w.cy)])
    d.link(fft, r, exit='right', entry='left')
    d.link(r, ifft, exit='right', entry='left')
    d.link(ifft, add, exit='right', entry='top', via=[(add.cx, ifft.cy)])
    d.link(w, add, exit='right', entry='bottom', via=[(add.cx, w.cy)])
    d.link(add, sg, exit='right', entry='left')
    q = d.node('Project Q', 675, 100, 80, 26, 'fc_box')
    u = d.node('u(x)', 780, 95, 70, 36, 'out_box')
    d.link(sg, q, exit='right', entry='left')
    d.link(q, u, exit='right', entry='left')


@register(CAT, 'DeepONet')
def deeponet(d):
    u = d.node('u(x<sub>1</sub>), …, u(x<sub>m</sub>)', 0, 22, 130, 36, 'in_box')
    bn = d.node('Branch net', 160, 22, 120, 36, 'module')
    bv = d.node('b<sub>1</sub>, …, b<sub>p</sub>', 310, 22, 110, 36, 'embed')
    y = d.node('Location y', 0, 122, 130, 36, 'in_box')
    tn = d.node('Trunk net', 160, 122, 120, 36, 'module_alt')
    tv = d.node('t<sub>1</sub>, …, t<sub>p</sub>', 310, 122, 110, 36, 'embed')
    for a_, b_ in [(u, bn), (bn, bv), (y, tn), (tn, tv)]:
        d.link(a_, b_, exit='right', entry='left')
    s = d.op('Σ', 480, 90)
    d.link(bv, s, exit='right', entry='top', via=[(s.cx, bv.cy)])
    d.link(tv, s, exit='right', entry='bottom', via=[(s.cx, tv.cy)])
    out = d.node('G(u)(y) = Σ<sub>k</sub> b<sub>k</sub> t<sub>k</sub> + b<sub>0</sub>', 525, 72, 180, 36, 'out_box')
    d.link(s, out, exit='right', entry='left')


@register(CAT, 'Capsule Network')
def capsnet(d):
    img = d.node('Input<br>28 x 28', 0, 30, 75, 70, 'input')
    cv = d.node('Conv1<br>256@9x9', 100, 30, 85, 70, 'conv')
    pc = d.node('PrimaryCaps<br>32 x 6 x 6<br>8D', 210, 30, 100, 70, 'feature')
    rt = d.node('Dynamic routing<br>(by agreement, r = 3)', 335, 45, 140, 40, 'gate')
    dc = d.node('DigitCaps<br>10 x 16D', 490, 30, 90, 70, 'latent')
    ln = d.node('Length ‖v<sub>j</sub>‖<br>class probability', 610, 47, 130, 36, 'out_box')
    for a, b in [(img, cv), (cv, pc), (pc, rt), (rt, dc), (dc, ln)]:
        d.link(a, b, exit='right', entry='left')
    f1 = d.node('FC 512', 410, 160, 90, 26, 'fc_box')
    f2 = d.node('FC 1024', 295, 160, 90, 26, 'fc_box')
    f3 = d.node('FC 784, sigmoid', 160, 160, 110, 26, 'fc_box')
    rc = d.node('Reconstruction', 20, 155, 115, 36, 'out_box')
    d.link(dc, f1, exit='bottom', entry='right', via=[(535, f1.cy)])
    for a, b in [(f1, f2), (f2, f3), (f3, rc)]:
        d.link(a, b, exit='left', entry='right')
    d.text('decoder (masked by the true class)', 160, 195, 340, 14, 'dim_label')
