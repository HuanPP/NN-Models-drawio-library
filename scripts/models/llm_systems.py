"""LLM training recipes, inference systems and agent pipelines."""

from nnlib.core import register

CAT = 'Training & Systems'


@register(CAT, 'LoRA')
def lora(d):
    x = d.node('Input x', 130, 0, 120, 30, 'in_box')
    w = d.node('Pretrained<br>weights W<br>(frozen)<br>d x d', 40, 80, 120, 110, 'frozen')
    a = d.node('A = N(0, σ²)<br>d → r', 230, 80, 120, 36, 'fc_box')
    b = d.node('B = 0<br>r → d', 230, 150, 120, 36, 'fc_box')
    d.link(x, w, exit='bottom', entry='top', via=[(190, 55), (w.cx, 55)])
    d.link(x, a, exit='bottom', entry='top', via=[(190, 55), (a.cx, 55)])
    d.link(a, b)
    add = d.op('+', 190, 240)
    d.link(w, add, exit='bottom', entry='left', via=[(w.cx, add.cy)])
    d.link(b, add, exit='bottom', entry='right', via=[(b.cx, add.cy)])
    h = d.node('h = Wx + BAx', 130, 280, 120, 30, 'out_box')
    d.link(add, h)
    d.text('rank r ≪ d<br>(trainable)', 355, 120, 70, 28, 'dim_label')


@register(CAT, 'KV Cache Decoding')
def kv_cache(d):
    xt = d.node('Token x<sub>t</sub>', 150, 0, 120, 30, 'in_box')
    q = d.node('q<sub>t</sub>', 60, 70, 60, 30, 'q')
    k = d.node('k<sub>t</sub>', 180, 70, 60, 30, 'k')
    v = d.node('v<sub>t</sub>', 300, 70, 60, 30, 'v')
    for n in (q, k, v):
        d.link(xt, n, exit='bottom', entry='top', via=[(210, 50), (n.cx, 50)])
    cache = d.node('KV Cache<br>K<sub>1:t</sub>, V<sub>1:t</sub>', 420, 120, 150, 60, 'db')
    d.link(v, cache, exit='bottom', entry='left', via=[(v.cx, 135)], entry_pos=0.25)
    d.link(k, cache, exit='bottom', entry='left', via=[(k.cx, 165)], entry_pos=0.75)
    d.text('append', 365, 116, 40, 14, 'dim_label')
    att = d.node('Attention(q<sub>t</sub>, K<sub>1:t</sub>, V<sub>1:t</sub>)', 40, 220, 300, 36, 'attn')
    d.link(q, att, exit='bottom', entry='top', entry_pos=(q.cx - att.x) / att.w)
    d.link(cache, att, exit='bottom', entry='right', via=[(cache.cx, att.cy)])
    out = d.node('Next token x<sub>t+1</sub>', 100, 290, 180, 30, 'out_box')
    d.link(att, out, exit='bottom', entry='top', exit_pos=(190 - att.x) / att.w)
    d.link(out, xt, exit='left', entry='left', via=[(20, out.cy), (20, xt.cy)], label='next step')
    d.text('only the new token is projected; past keys and values are read from the cache',
           0, 335, 570, 14, 'dim_label')


@register(CAT, 'Speculative Decoding')
def speculative(d):
    pre = d.node('Prefix', 0, 80, 90, 36, 'in_box')
    dr = d.node('Draft model q<br>(small, fast)', 120, 70, 120, 56, 'module_alt')
    tk = d.node('γ draft tokens<br>x̃<sub>1</sub> … x̃<sub>γ</sub>', 270, 80, 110, 36, 'reshape')
    tg = d.node('Target model p<br>(one parallel pass)', 410, 70, 140, 56, 'module')
    ok = d.node('accept?', 570, 63, 90, 70, 'decision')
    out = d.node('Accepted tokens<br>+ 1 new token', 690, 80, 120, 36, 'out_box')
    for a, b in [(pre, dr), (dr, tk), (tk, tg), (tg, ok)]:
        d.link(a, b, exit='right', entry='left')
    d.link(ok, out, exit='right', entry='left', label='yes')
    rs = d.node('Resample from<br>norm(max(0, p − q))', 550, 175, 130, 40, 'act')
    d.link(ok, rs, label='reject')
    d.link(rs, out, exit='right', entry='bottom', via=[(out.cx, rs.cy)])
    d.link(out, pre, exit='top', entry='top', via=[(out.cx, 20), (pre.cx, 20)], label='append and repeat')


@register(CAT, 'RLHF Pipeline')
def rlhf(d):
    c1 = d.container('Step 1: SFT', 0, 0, 200, 260, 'container_blue')
    a = d.node('Demonstration<br>data', 40, 40, 120, 60, 'db')
    b = d.node('Supervised<br>fine-tuning', 40, 125, 120, 40, 'module')
    c = d.node('SFT model π<sup>SFT</sup>', 40, 195, 120, 36, 'out_box')
    d.link(a, b)
    d.link(b, c)
    c2 = d.container('Step 2: Reward Model', 230, 0, 200, 260, 'container_green')
    a2 = d.node('Human preference<br>comparisons', 270, 40, 120, 60, 'db')
    b2 = d.node('Train RM<br>(ranking loss)', 270, 125, 120, 40, 'module_alt')
    r = d.node('Reward model r<sub>φ</sub>', 270, 195, 120, 36, 'out_box')
    d.link(a2, b2)
    d.link(b2, r)
    c3 = d.container('Step 3: PPO', 460, 0, 325, 260, 'container_purple')
    pr = d.node('Prompts', 500, 40, 100, 50, 'db')
    pol = d.node('Policy π<sub>θ</sub><br>(init: SFT)', 500, 115, 100, 40, 'module_gen')
    rm = d.node('r<sub>φ</sub> (frozen)', 635, 40, 120, 40, 'frozen')
    ref = d.node('Reference π<sup>SFT</sup><br>(frozen)', 635, 115, 120, 40, 'frozen')
    obj = d.node('r<sub>φ</sub>(x, y) − β KL(π<sub>θ</sub> ‖ π<sup>SFT</sup>)', 505, 195, 250, 36, 'loss')
    d.link(pr, pol)
    d.link(pol, obj, entry_pos=(pol.cx - obj.x) / obj.w)
    d.link(ref, obj, entry_pos=(ref.cx - obj.x) / obj.w)
    d.link(rm, obj, exit='right', entry='right', via=[(770, rm.cy), (770, obj.cy)])
    d.link(obj, pol, exit='left', entry='left', via=[(482, obj.cy), (482, pol.cy)], style='grad')
    d.text('PPO', 488, 162, 30, 14, 'dim_label')
    d.link(c1, c2, exit='right', entry='left')
    d.link(c2, c3, exit='right', entry='left')


@register(CAT, 'DPO')
def dpo(d):
    data = d.node('Preference data<br>(x, y<sub>w</sub>, y<sub>l</sub>)', 150, 0, 160, 60, 'db')
    pol = d.node('Policy π<sub>θ</sub>', 60, 100, 140, 40, 'module_gen')
    ref = d.node('Reference π<sub>ref</sub><br>(frozen)', 260, 100, 140, 40, 'frozen')
    for m in (pol, ref):
        d.link(data, m, exit='bottom', entry='top', via=[(230, 80), (m.cx, 80)])
    lp = d.node('log π<sub>θ</sub>(y<sub>w</sub>|x), log π<sub>θ</sub>(y<sub>l</sub>|x)', 35, 175, 190, 36, 'act')
    lr = d.node('log π<sub>ref</sub>(y<sub>w</sub>|x), log π<sub>ref</sub>(y<sub>l</sub>|x)', 235, 175, 190, 36, 'act')
    d.link(pol, lp)
    d.link(ref, lr)
    loss = d.node('L<sub>DPO</sub> = −log σ(β · Δ)<br>Δ = log π<sub>θ</sub>/π<sub>ref</sub>(y<sub>w</sub>) − '
                  'log π<sub>θ</sub>/π<sub>ref</sub>(y<sub>l</sub>)', 80, 250, 300, 50, 'loss')
    d.link(lp, loss, entry_pos=(lp.cx - loss.x) / loss.w)
    d.link(lr, loss, entry_pos=(lr.cx - loss.x) / loss.w)
    d.link(loss, pol, exit='left', entry='left', via=[(20, loss.cy), (20, pol.cy)], style='grad')
    d.text('no reward model, no RL sampling: a classification loss on preference pairs',
           0, 315, 460, 14, 'dim_label')


@register(CAT, 'RAG Pipeline')
def rag(d):
    d.container('Offline indexing', 0, 0, 470, 110, 'container_gray')
    docs = d.node('Documents', 20, 40, 90, 50, 'db')
    ch = d.node('Chunking', 130, 47, 80, 36, 'reshape')
    e1 = d.node('Embedding<br>model', 230, 47, 100, 36, 'embed')
    vs = d.node('Vector store', 350, 35, 100, 60, 'db')
    for a, b in [(docs, ch), (ch, e1), (e1, vs)]:
        d.link(a, b, exit='right', entry='left')
    d.container('Online query', 0, 135, 840, 135, 'container_blue')
    q = d.node('Query', 20, 180, 90, 36, 'in_box')
    e2 = d.node('Embedding<br>model', 130, 180, 110, 36, 'embed')
    rt = d.node('Top-k<br>retrieval', 350, 180, 100, 36, 'gate')
    pr = d.node('Prompt =<br>query + docs', 480, 180, 110, 36, 'act')
    llm = d.node('LLM', 620, 180, 90, 36, 'module_gen')
    ans = d.node('Answer', 740, 180, 80, 36, 'out_box')
    for a, b in [(q, e2), (e2, rt), (rt, pr), (pr, llm), (llm, ans)]:
        d.link(a, b, exit='right', entry='left')
    d.link(vs, rt)
    d.link(q, pr, exit='bottom', entry='bottom', via=[(q.cx, 250), (pr.cx, 250)])


@register(CAT, 'ReAct Agent Loop')
def react(d):
    q = d.node('Task / user<br>query', 0, 100, 100, 36, 'in_box')
    mem = d.node('Context<br>(history)', 130, 90, 120, 56, 'db')
    llm = d.node('LLM<br>Thought', 290, 95, 120, 46, 'module_gen')
    dn = d.node('done?', 450, 83, 90, 70, 'decision')
    ans = d.node('Final answer', 580, 100, 100, 36, 'out_box')
    for a, b in [(q, mem), (mem, llm), (llm, dn)]:
        d.link(a, b, exit='right', entry='left')
    d.link(dn, ans, exit='right', entry='left', label='yes')
    act = d.node('Action<br>(tool call)', 435, 200, 120, 36, 'gate')
    tools = d.node('Tools /<br>environment', 270, 200, 140, 36, 'env')
    obs = d.node('Observation', 130, 200, 120, 36, 'reshape')
    d.link(dn, act, label='no')
    d.link(act, tools, exit='left', entry='right')
    d.link(tools, obs, exit='left', entry='right')
    d.link(obs, mem)


@register(CAT, 'Knowledge Distillation')
def distillation(d):
    x = d.node('Input x', 180, 0, 100, 30, 'in_box')
    te = d.node('Teacher<br>(frozen, large)', 40, 70, 140, 40, 'frozen')
    st = d.node('Student<br>(small)', 280, 70, 140, 40, 'module')
    for m in (te, st):
        d.link(x, m, exit='bottom', entry='top', via=[(230, 50), (m.cx, 50)])
    ts = d.node('softmax(z<sub>t</sub> / T)', 40, 140, 140, 30, 'softmax')
    ss = d.node('softmax(z<sub>s</sub> / T)', 280, 140, 140, 30, 'softmax')
    sh = d.node('softmax(z<sub>s</sub>)', 450, 140, 120, 30, 'softmax')
    d.link(te, ts)
    d.link(st, ss)
    d.link(st, sh, exit='right', entry='top', via=[(sh.cx, st.cy)])
    kd = d.node('Distillation loss<br>T² · KL(p<sub>t</sub> ‖ p<sub>s</sub>)', 140, 210, 180, 36, 'loss')
    d.link(ts, kd, exit='bottom', entry='left', via=[(ts.cx, kd.cy)])
    d.link(ss, kd, exit='bottom', entry='right', via=[(ss.cx, kd.cy)])
    y = d.node('Label y', 600, 140, 80, 30, 'in_box')
    ce = d.node('CE loss<br>H(y, p<sub>s</sub>)', 440, 210, 140, 36, 'loss')
    d.link(sh, ce, entry_pos=(sh.cx - ce.x) / ce.w)
    d.link(y, ce, exit='bottom', entry='right', via=[(y.cx, ce.cy)])
    tot = d.node('L = α · L<sub>KD</sub> + (1 − α) · L<sub>CE</sub>', 240, 280, 220, 36, 'loss')
    d.link(kd, tot, exit='bottom', entry='left', via=[(kd.cx, tot.cy)])
    d.link(ce, tot, exit='bottom', entry='right', via=[(ce.cx, tot.cy)])
