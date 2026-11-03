"""Reinforcement learning architectures and training loops."""

from nnlib.core import register

CAT = 'Reinforcement Learning'


@register(CAT, 'Agent-Environment Loop')
def agent_env_loop(d):
    agent = d.node('Agent<br>policy π(a | s)', 200, 40, 160, 50, 'module')
    env = d.node('Environment', 200, 200, 160, 50, 'env')
    # action: out of the agent on the right, down, into the environment
    d.link(agent, env, exit='right', entry='right',
           via=[(430, agent.cy), (430, env.cy)], label='action<br>a<sub>t</sub>')
    # state (inner loop) and reward (outer loop) return on the left
    d.link(env, agent, exit='left', entry='left', exit_pos=0.3, entry_pos=0.7,
           via=[(140, env.port('left', 0.3)[1]), (140, agent.port('left', 0.7)[1])],
           label='state<br>s<sub>t+1</sub>')
    d.link(env, agent, exit='left', entry='left', exit_pos=0.7, entry_pos=0.3,
           via=[(80, env.port('left', 0.7)[1]), (80, agent.port('left', 0.3)[1])],
           label='reward<br>r<sub>t+1</sub>')


@register(CAT, 'DQN')
def dqn(d):
    env = d.node('Environment', 265, 20, 130, 36, 'env')
    mem = d.node('Replay memory D', 265, 85, 130, 60, 'db')
    d.link(env, mem)
    d.text('store (s, a, r, s′)', 340, 63, 100, 14, 'dim_label')

    # online network (left) and target network (right), identical layer stacks
    layers = ['Conv 8x8, 32, /4, ReLU', 'Conv 4x4, 64, /2, ReLU', 'Conv 3x3, 64, /1, ReLU',
              'FC 512, ReLU', 'FC |A|']
    d.container('Online Q-network θ', 90, 170, 180, 360, 'container_blue')
    d.container('Target network θ⁻', 390, 170, 180, 360, 'container_gray')
    s = d.node('State s<br>84 x 84 x 4', 110, 210, 140, 36, 'in_box')
    s2 = d.node('Next state s′<br>84 x 84 x 4', 410, 210, 140, 36, 'in_box')
    on, tg = [s], [s2]
    y = 270
    for i, lab in enumerate(layers):
        on.append(d.node(lab, 110, y, 140, 28, 'conv_box' if i < 3 else 'fc_box'))
        tg.append(d.node(lab, 410, y, 140, 28, 'frozen'))
        y += 52
    for col in (on, tg):
        for a, b in zip(col, col[1:]):
            d.link(a, b)

    q = d.node('Q(s, a; θ)<br>of taken action a', 95, 550, 170, 40, 'out_box')
    y_t = d.node('TD target y<br>r + γ max<sub>a′</sub> Q(s′, a′; θ<sup>−</sup>)', 395, 550, 170, 40, 'out_box')
    d.link(on[-1], q)
    d.link(tg[-1], y_t)
    loss = d.node('TD loss<br>(y − Q(s, a; θ))<sup>2</sup>', 260, 620, 140, 40, 'loss')
    d.link(q, loss, exit='bottom', entry='left', via=[(q.cx, loss.cy)])
    d.link(y_t, loss, exit='bottom', entry='right', via=[(y_t.cx, loss.cy)])

    # minibatch (s, a, r, s') sampled from D
    yl, yh = mem.port('left', 0.25)[1], mem.port('left', 0.75)[1]
    d.link(mem, q, exit='left', entry='left', exit_pos=0.25, via=[(40, yl), (40, q.cy)], label='a')
    d.link(mem, s, exit='left', entry='left', exit_pos=0.75, via=[(60, yh), (60, s.cy)])
    d.link(mem, y_t, exit='right', entry='right', exit_pos=0.25, via=[(620, yl), (620, y_t.cy)], label='r')
    d.link(mem, s2, exit='right', entry='right', exit_pos=0.75, via=[(600, yh), (600, s2.cy)])

    # periodic hard copy of the online weights into the target network
    d.link(on[3], tg[3], exit='right', entry='left', style='grad')
    d.text('θ<sup>−</sup> ← θ<br>every C steps', 285, 350, 90, 28, 'label')


@register(CAT, 'Dueling DQN')
def dueling_dqn(d):
    cx = 250
    s = d.node('State s<br>84 x 84 x 4', cx - 70, 20, 140, 36, 'in_box')
    c1 = d.node('Conv 8x8, 32, /4, ReLU', cx - 75, 80, 150, 28, 'conv_box')
    c2 = d.node('Conv 4x4, 64, /2, ReLU', cx - 75, 132, 150, 28, 'conv_box')
    c3 = d.node('Conv 3x3, 64, /1, ReLU', cx - 75, 184, 150, 28, 'conv_box')
    for a, b in [(s, c1), (c1, c2), (c2, c3)]:
        d.link(a, b)

    lx, rx = 160, 340
    v1 = d.node('FC 512, ReLU', lx - 70, 256, 140, 28, 'fc_box')
    a1 = d.node('FC 512, ReLU', rx - 70, 256, 140, 28, 'fc_box')
    v2 = d.node('FC 1', lx - 70, 308, 140, 28, 'fc_box')
    a2 = d.node('FC |A|', rx - 70, 308, 140, 28, 'fc_box')
    a3 = d.node('A − mean<sub>a′</sub> A', rx - 70, 360, 140, 28, 'reshape')
    fy = 232
    d.link(c3, v1, via=[(cx, fy), (lx, fy)])
    d.link(c3, a1, via=[(cx, fy), (rx, fy)])
    for a, b in [(v1, v2), (a1, a2), (a2, a3)]:
        d.link(a, b)
    d.text('Value<br>stream', 30, 256, 50, 28, 'label')
    d.text('Advantage<br>stream', 420, 256, 60, 28, 'label')

    add = d.op('+', cx, 425)
    d.link(v2, add, exit='bottom', entry='left', via=[(lx, add.cy)])
    d.link(a3, add, exit='bottom', entry='right', via=[(rx, add.cy)])
    d.text('V(s)', lx + 5, 370, 40, 14, 'dim_label')
    d.text('A(s, a)', rx + 5, 341, 50, 14, 'dim_label')
    out = d.node('Q(s, a) = V + (A − mean<sub>a′</sub> A)', cx - 95, 460, 190, 30, 'out_box')
    d.link(add, out)


@register(CAT, 'Actor-Critic')
def actor_critic(d):
    s = d.node('State s<sub>t</sub>', 20, 142, 100, 36, 'in_box')
    actor = d.node('Actor<br>π<sub>θ</sub>(a | s)', 180, 60, 140, 40, 'module')
    critic = d.node('Critic<br>V<sub>w</sub>(s)', 180, 220, 140, 40, 'module_alt')
    env = d.node('Environment', 400, 60, 140, 40, 'env')
    td = d.node('TD error δ<sub>t</sub><br>r + γV(s′) − V(s)', 400, 220, 140, 40, 'loss')

    d.link(s, actor, exit='right', entry='left', via=[(150, s.cy), (150, actor.cy)])
    d.link(s, critic, exit='right', entry='left', via=[(150, s.cy), (150, critic.cy)])
    d.link(actor, env)
    d.link(env, td)
    d.link(critic, td)
    d.link(env, s, exit='top', entry='top', via=[(env.cx, 30), (s.cx, 30)])
    d.text('a<sub>t</sub>', 340, 62, 40, 14, 'label')
    d.text('r<sub>t+1</sub>', 478, 150, 40, 16, 'label')
    d.text('V(s<sub>t</sub>)', 335, 222, 50, 14, 'label')
    d.text('s<sub>t+1</sub>', 245, 12, 50, 14, 'label')

    # the TD error trains both networks
    px = td.port('top', 0.25)[0]
    d.link(td, actor, exit='top', entry='bottom', exit_pos=0.25, entry_pos=0.75,
           via=[(px, 160), (actor.port('bottom', 0.75)[0], 160)], style='grad')
    d.link(td, critic, exit='bottom', entry='bottom', via=[(td.cx, 290), (critic.cx, 290)], style='grad')
    d.text('∇<sub>θ</sub> log π · δ', 320, 140, 80, 16, 'label')
    d.text('∇<sub>w</sub> δ<sup>2</sup>', 330, 294, 60, 16, 'label')


@register(CAT, 'PPO')
def ppo(d):
    c1, c2, c3, c4 = 80, 260, 430, 600

    # rollout with the old policy
    d.container('Rollout with π<sub>old</sub>', 5, 20, 340, 100, 'container_green')
    env = d.node('Environment', c1 - 60, 60, 120, 40, 'env')
    pold = d.node('Policy π<sub>θ<sub>old</sub></sub>', c2 - 70, 60, 140, 40, 'frozen')
    d.link(pold, env, exit='left', entry='right', exit_pos=0.3, entry_pos=0.3)
    d.link(env, pold, exit='right', entry='left', exit_pos=0.7, entry_pos=0.7)
    d.text('a<sub>t</sub>', 150, 52, 30, 14, 'label')
    d.text('s<sub>t</sub>, r<sub>t</sub>', 145, 94, 40, 14, 'label')
    buf = d.node('Rollout buffer<br>(s, a, r, log π<sub>old</sub>)', c2 - 70, 145, 140, 60, 'db')
    d.link(pold, buf)

    gae = d.node('GAE (γ, λ)<br>Â<sub>t</sub>, R̂<sub>t</sub>', c3 - 70, 225, 140, 40, 'reshape')
    d.text('θ<sub>old</sub> ← θ after each update', 380, 70, 200, 16, 'label')

    # K epochs of minibatch updates
    d.container('K epochs × minibatches', 5, 290, 685, 360, 'container_repeat')
    pol = d.node('Policy π<sub>θ</sub>(a | s)', c2 - 70, 315, 140, 40, 'module')
    val = d.node('Value net V<sub>φ</sub>(s)', c4 - 70, 315, 140, 40, 'module_alt')
    ratio = d.node('Ratio r<sub>t</sub>(θ)<br>π<sub>θ</sub> / π<sub>θ<sub>old</sub></sub>', c2 - 70, 379, 140, 40, 'reshape')
    clip = d.node('clip(r<sub>t</sub>, 1 ± ε)', c2 - 70, 443, 140, 28, 'act')
    lclip = d.node('Surrogate L<sup>CLIP</sup><br>min(r<sub>t</sub>Â<sub>t</sub>, clip · Â<sub>t</sub>)',
                   c2 - 70, 495, 140, 40, 'loss')
    ent = d.node('Entropy bonus<br>S[π<sub>θ</sub>](s)', c1 - 60, 495, 120, 40, 'loss')
    lvf = d.node('Value loss L<sup>VF</sup><br>(V<sub>φ</sub>(s) − R̂<sub>t</sub>)<sup>2</sup>', c4 - 70, 495, 140, 40, 'loss')
    tot = d.node('Total loss<br>−L<sup>CLIP</sup> + c<sub>1</sub>L<sup>VF</sup> − c<sub>2</sub>S',
                 c3 - 70, 555, 140, 40, 'loss')

    ty = buf.cy
    d.link(buf, gae, exit='right', entry='top', via=[(c3, ty)])
    vx_in = val.port('top', 0.75)[0]
    d.link(buf, val, exit='right', entry='top', entry_pos=0.75, via=[(vx_in, ty)])
    d.link(buf, pol)
    vx_out = val.port('top', 0.25)[0]
    d.link(val, gae, exit='top', entry='right', exit_pos=0.25, via=[(vx_out, gae.cy)])

    for a, b in [(pol, ratio), (ratio, clip), (clip, lclip), (val, lvf)]:
        d.link(a, b)
    d.link(ratio, lclip, exit='left', entry='left', via=[(165, ratio.cy), (165, lclip.cy)])
    d.link(pol, ent, exit='left', entry='top', via=[(c1, pol.cy)])
    ax = gae.port('bottom', 0.3)[0]
    rx = gae.port('bottom', 0.7)[0]
    d.link(gae, lclip, exit='bottom', entry='right', exit_pos=0.3, via=[(ax, lclip.cy)],
           label='Â<sub>t</sub>')
    d.link(gae, lvf, exit='bottom', entry='left', exit_pos=0.7, via=[(rx, lvf.cy)],
           label='R̂<sub>t</sub>')
    d.link(lclip, tot, exit='bottom', entry='left', via=[(c2, tot.cy)])
    d.link(lvf, tot, exit='bottom', entry='right', via=[(c4, tot.cy)])
    d.link(ent, tot, exit='bottom', entry='bottom', via=[(c1, 620), (c3, 620)])


@register(CAT, 'AlphaZero')
def alphazero(d):
    # ---- self-play ----
    lx = 140
    d.container('Self-play', 15, 15, 345, 200, 'container_teal')
    st = d.node('Game state s<sub>t</sub>', lx - 65, 55, 130, 30, 'in_box')
    mcts = d.node('MCTS search<br>guided by f<sub>θ</sub>', lx - 65, 105, 130, 40, 'gate')
    pi = d.node('π<sub>t</sub> (visit counts)', lx - 65, 165, 130, 30, 'out_box')
    fev = d.node('f<sub>θ</sub>(s<sub>L</sub>)<br>→ (p, v)', 245, 105, 100, 40, 'module')
    d.link(st, mcts)
    d.link(mcts, pi)
    d.link(mcts, fev, exit='right', entry='left', exit_pos=0.25, entry_pos=0.25)
    d.link(fev, mcts, exit='left', entry='right', exit_pos=0.75, entry_pos=0.75)
    d.text('leaf s<sub>L</sub>', 208, 96, 34, 14, 'dim_label')
    d.text('(p, v)', 208, 139, 34, 14, 'dim_label')
    d.link(pi, st, exit='left', entry='left', via=[(35, pi.cy), (35, st.cy)],
           label='a<sub>t</sub> ~ π<sub>t</sub>')

    buf = d.node('Replay buffer<br>(s<sub>t</sub>, π<sub>t</sub>, z)', lx - 65, 240, 130, 60, 'db')
    d.link(pi, buf)
    d.text('+ outcome z at game end', 148, 220, 120, 14, 'dim_label')

    # ---- network f_theta ----
    nx = 490
    inp = d.node('Input s<sub>t</sub><br>19 x 19 x 17', nx - 70, 50, 140, 40, 'in_box')
    d.link(buf, inp, exit='right', entry='left', via=[(370, buf.cy), (370, inp.cy)])
    conv = d.node('Conv 3x3, 256, BN, ReLU', nx - 85, 110, 170, 28, 'conv_box')
    d.link(inp, conv)
    d.container('× 19', 380, 155, 245, 230, 'container_repeat')
    dot = d.op('', nx, 173, size=8, style='dot')
    r1 = d.node('Conv 3x3, 256, BN, ReLU', nx - 85, 189, 170, 28, 'conv_box')
    r2 = d.node('Conv 3x3, 256, BN', nx - 85, 237, 170, 28, 'conv_box')
    add = d.op('+', nx, 300)
    relu = d.node('ReLU', nx - 45, 335, 90, 24, 'act')
    d.link(conv, dot, style='line')
    for a, b in [(dot, r1), (r1, r2), (r2, add), (add, relu)]:
        d.link(a, b)
    d.link(dot, add, exit='right', entry='right', via=[(600, dot.cy), (600, add.cy)], style='skip')
    d.text('residual block', 395, 360, 80, 14, 'dim_label')

    px, vx = nx - 85, nx + 85
    fy = 400
    ph1 = d.node('Conv 1x1, 2, BN, ReLU', px - 70, 415, 140, 28, 'conv_box')
    vh1 = d.node('Conv 1x1, 1, BN, ReLU', vx - 70, 415, 140, 28, 'conv_box')
    ph2 = d.node('FC, softmax', px - 70, 463, 140, 36, 'fc_box')
    vh2 = d.node('FC 256, ReLU<br>FC 1, tanh', vx - 70, 463, 140, 36, 'fc_box')
    p = d.node('Move probs p', px - 70, 519, 140, 28, 'out_box')
    v = d.node('Value v', vx - 70, 519, 140, 28, 'out_box')
    d.link(relu, ph1, via=[(nx, fy), (px, fy)])
    d.link(relu, vh1, via=[(nx, fy), (vx, fy)])
    for a, b in [(ph1, ph2), (ph2, p), (vh1, vh2), (vh2, v)]:
        d.link(a, b)
    d.text('policy head', px - 140, 422, 60, 14, 'dim_label')
    d.text('value head', vx + 75, 422, 55, 14, 'dim_label')

    loss = d.node('Loss<br>(z − v)<sup>2</sup> − π<sup>T</sup> log p<br>+ c‖θ‖<sup>2</sup>',
                  lx - 75, p.cy - 27, 150, 54, 'loss')
    d.link(p, loss)
    d.link(buf, loss, label='targets π<sub>t</sub>, z')
    d.link(v, loss, exit='bottom', entry='bottom', via=[(vx, 590), (lx, 590)])


@register(CAT, 'Dreamer')
def dreamer(d):
    # ---- world model learning (one time step of the RSSM) ----
    A, B, C, E = 85, 230, 375, 405
    d.container('World model learning', 5, 5, 475, 490, 'container_blue')
    buf = d.node('Replay buffer', E - 60, 45, 120, 50, 'db')
    obs = d.node('Observation o<sub>t</sub>', E - 60, 120, 120, 36, 'in_box')
    prev = d.node('h<sub>t−1</sub>, z<sub>t−1</sub>, a<sub>t−1</sub>', B - 60, 120, 120, 36, 'in_box')
    gru = d.node('GRU<br>deterministic h<sub>t</sub>', B - 60, 180, 120, 36, 'ssm')
    enc = d.node('CNN encoder<br>e<sub>t</sub>', E - 60, 180, 120, 36, 'conv_box')
    prior = d.node('Prior<br>p(ẑ<sub>t</sub> | h<sub>t</sub>)', A - 60, 256, 120, 40, 'gate')
    post = d.node('Posterior<br>q(z<sub>t</sub> | h<sub>t</sub>, e<sub>t</sub>)', C - 60, 256, 120, 40, 'gate')
    state = d.node('Model state<br>s<sub>t</sub> = (h<sub>t</sub>, z<sub>t</sub>)', B - 60, 320, 120, 36, 'ssm')
    dec = d.node('Decoder → ô<sub>t</sub>', 115, 390, 110, 30, 'deconv_box')
    rew = d.node('Reward → r̂<sub>t</sub>', 235, 390, 110, 30, 'fc_box')
    kl = d.node('KL[ q ‖ p ]', B - 60, 445, 120, 30, 'loss')

    for a, b in [(buf, obs), (obs, enc), (prev, gru)]:
        d.link(a, b)
    fy = 236
    d.link(gru, prior, via=[(B, fy), (A, fy)])
    d.link(gru, post, entry_pos=0.25, via=[(B, fy), (post.port('top', 0.25)[0], fy)])
    d.link(gru, state)
    d.link(enc, post, entry_pos=0.75)
    d.link(post, state, exit='bottom', entry='right', via=[(C, state.cy)])
    sy = 370
    d.link(state, dec, via=[(B, sy), (dec.cx, sy)])
    d.link(state, rew, via=[(B, sy), (rew.cx, sy)])
    d.link(prior, kl, exit='bottom', entry='left', via=[(A, kl.cy)])
    d.link(post, kl, exit='bottom', entry='right', via=[(C, kl.cy)])
    d.text('h<sub>t</sub>', B + 5, 296, 30, 14, 'dim_label')
    d.text('z<sub>t</sub>', C + 5, 300, 30, 14, 'dim_label')

    # ---- behaviour learning in imagination ----
    EX, FX = 620, 780
    d.container('Behavior learning in imagination', 510, 140, 375, 380, 'container_purple')
    lat = d.node('Latent state<br>ŝ<sub>τ</sub> = (h<sub>τ</sub>, ẑ<sub>τ</sub>)', EX - 70, 180, 140, 40, 'ssm')
    actor = d.node('Actor<br>π<sub>φ</sub>(a<sub>τ</sub> | ŝ<sub>τ</sub>)', EX - 70, 250, 140, 40, 'module')
    critic = d.node('Critic<br>v<sub>ψ</sub>(ŝ<sub>τ</sub>)', FX - 60, 250, 120, 40, 'module_alt')
    trans = d.node('World model (frozen)<br>GRU + prior → ŝ<sub>τ+1</sub>', EX - 70, 320, 140, 40, 'frozen')
    rhat = d.node('Reward head<br>r̂<sub>τ</sub>', FX - 60, 320, 120, 40, 'fc_box')
    lam = d.node('λ-return<br>V<sub>λ</sub>(ŝ<sub>τ</sub>)', FX - 60, 390, 120, 40, 'reshape')
    aloss = d.node('Actor loss<br>−V<sub>λ</sub>(ŝ<sub>τ</sub>)', EX - 70, 460, 140, 40, 'loss')
    closs = d.node('Critic loss<br>(v<sub>ψ</sub> − V<sub>λ</sub>)<sup>2</sup>', FX - 60, 460, 120, 40, 'loss')

    sy0 = lat.port('left', 0.25)[1]
    d.link(post, lat, exit='right', entry='left', entry_pos=0.25, via=[(495, post.cy), (495, sy0)])
    d.link(lat, actor)
    d.link(lat, critic, via=[(EX, 235), (FX, 235)])
    d.link(actor, trans)
    d.link(trans, rhat)
    d.link(trans, lat, exit='left', entry='left', entry_pos=0.75,
           via=[(530, trans.cy), (530, lat.port('left', 0.75)[1])])
    d.link(rhat, lam)
    d.link(critic, lam, exit='right', entry='right', via=[(865, critic.cy), (865, lam.cy)])
    d.link(lam, aloss, via=[(FX, 445), (EX, 445)])
    d.link(lam, closs)
    d.text('a<sub>τ</sub>', EX + 5, 298, 30, 14, 'dim_label')
    d.text('imagine<br>H steps', 455 + 0, 330, 1, 1, 'dim_label') if False else None
