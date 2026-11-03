# Template Authoring Guide

Every template is a Python function that draws onto a `Diagram`. The build
turns it into a draw.io library entry, a standalone `.drawio` file and PNG/SVG
previews. A template is accepted only when it is **technically correct**, passes
the **automatic layout checks** with zero errors, and **looks like a figure from
a top-venue paper** in its rendered PNG.

## 1. Workflow

```bash
cd /data/NN-Models-drawio-library
python3 scripts/render.py --module <module>                 # all models in scripts/models/<module>.py
python3 scripts/render.py --module <module> --title "DiT Block"   # one model
```

The script builds, runs the static checks, renders through draw.io and then
compares the rendered edge routes against the intended ones. Output:
`/tmp/nnlib/review/<module>/<slug>.png` (+ `.svg`, `.drawio`). Exit code 0 means
no ERROR. Then **open every PNG with the Read tool and critique it** (section 5).
Iterate until both the checks and your own visual review are clean.

Coordinates printed in check messages are the coordinates you wrote in the
code (except `verify_render` waypoint messages, which are shifted so the
diagram starts at (10, 10)).

## 2. API (`scripts/nnlib/core.py`)

```python
from nnlib.core import register

CAT = 'Diffusion Models'

@register(CAT, 'DiT Block')          # title must be unique across the library
def dit_block(d):
    c   = d.container('DiT Block', x, y, w, h, 'container_purple')
    n   = d.node('Multi-Head<br>Self-Attention', x, y, w, h, 'attn')
    o   = d.op('+', cx, cy)          # circle centred at (cx, cy), default size 30
    t   = d.text('N x C', x, y, w, h, 'dim_label')
    d.link(a, b)                                       # sides inferred (vertical if b is below a)
    d.link(a, b, exit='right', entry='left')           # explicit sides
    d.link(a, b, exit='left', entry='left',            # explicit route
           via=[(x1, a.cy), (x1, b.cy)], style='skip')
    d.link(a, b, exit='bottom', entry='top', exit_pos=0.25)   # port at 25 % of the side
    d.arrow([(x0, y0), (x0, y1)])                      # free arrow not bound to nodes (rare)
```

Node helpers: `n.x n.y n.w n.h n.cx n.cy n.r n.b` and `n.port(side, pos)`.

Rules the code enforces:
* Every pair of consecutive route points (exit port, waypoints..., entry port)
  must share x or y: routes are strictly orthogonal and fully specified by you.
* The first segment must leave through the `exit` side, the last must enter
  through the `entry` side.
* Ports on circular operators must be centred (`pos=0.5`).
* Only the style keys below may be used. Do not edit `scripts/nnlib/`; if a
  style is genuinely missing, use the closest one and say so in your report.

### Node styles and colour semantics

| key | colour | use for |
|---|---|---|
| `input` / `in_box` | red | inputs: image, tokens, x, noise image x_t |
| `output` / `out_box` | yellow | outputs, predictions, logits |
| `conv` `conv_sm` / `conv_box` | blue | convolutions |
| `deconv` `deconv_sm` / `deconv_box` | sky | upsampling, transposed conv |
| `pool` `pool_sm` / `pool_box` | green | pooling, downsampling, patch merging |
| `fc` / `fc_box` | orange | Linear, MLP, FFN, experts, heads |
| `attn` | purple | self-attention, attention blocks |
| `cross_attn` | red | cross-attention |
| `embed` | purple | token / patch / time / class embeddings |
| `pos` | yellow | positional encoding, conditioning signals (t, y, text c) |
| `norm` | gray | BN, LN, RMSNorm, GN, AdaLN |
| `reshape` | gray | flatten, reshape, patchify, split, shuffle |
| `act` | white | activations (ReLU, GELU, SiLU, Sigmoid), dropout |
| `softmax` | yellow | softmax |
| `ssm` | teal | SSM / selective scan / recurrence / memory state |
| `rnn` | teal | LSTM / GRU cells |
| `gate` | pink | routers, gates, quantiser, codebook, top-k selection |
| `q` `k` `v` | orange/blue/green | query / key / value projections |
| `loss` | red dashed | loss terms and objectives |
| `frozen` | gray italic | frozen / pre-trained modules (write "(frozen)") |
| `module` / `module_alt` / `module_gen` | blue / green / purple | large opaque sub-networks (Encoder, Decoder, LLM, Critic, Generator ...) |
| `noise` | gray dashed | noise / random sampling |
| `mask` | dark | masks |
| `env` | green | RL environment, external world |
| `db` | gray cylinder | memory bank, replay buffer, vector store, dataset |
| `trapezoid_enc` / `trapezoid_dec` | blue / green | encoder (narrowing) / decoder (widening) |
| `decision` | white rhombus | branching / decision |
| `op` / `op_sm` | white circle | `+` add, `×` multiply, `C` concat, `σ`, `@` matmul |
| `dot` | small black dot | junction marker |
| `container_<gray/blue/green/red/purple/orange/teal>` | dashed frame | groups, title top-left |
| `container_repeat` | dashed gray frame | "× N" repeated blocks, title bottom-right |
| `label` / `label_left` / `dim_label` / `caption` / `ellipsis` | text | annotations, tensor shapes, captions, "..." |

Use cubes (`input`, `conv`, `pool`, `fc`, `output`, `feature`, `latent`) for
tensor-stage diagrams of CNN pipelines (as in LeNet / U-Net); use flat boxes for
operator-level block diagrams (as in Transformer blocks). Do not mix both
styles randomly inside one figure.

### Edge styles

| key | use |
|---|---|
| `arrow` | normal data flow |
| `skip` | dashed residual / skip / identity connection |
| `line` / `skip_line` | no arrow head (e.g. a trunk that later splits) |
| `bidir` | two-way |
| `loss_flow` / `grad` | red dotted: losses, gradients, EMA/stop-grad annotations |

## 3. Layout rules

1. **Direction.** Single blocks and layer stacks flow top to bottom.
   Multi-stage architectures and pipelines flow left to right. Pick one main
   axis per figure and keep it.
2. **Grid and alignment.** All coordinates are multiples of 5. Nodes in one
   column share the same centre x; nodes in one row share the same centre y.
   Nodes of the same role have the same size.
3. **Size.** Typical box: width 100-140, height 26-36 (single line 26-30, two
   lines 36-40). Activations / norms may be narrower (70-100). Operators 26-30.
   Keep the whole figure compact: aim for <= 700 px wide and <= 650 px tall; a
   figure larger than 900 px in either direction needs a good reason.
4. **Spacing.** Vertical gap between consecutive boxes 20-28 px (24 default).
   Horizontal gap between columns 30-50 px. Use the same gap throughout a
   figure.
5. **Containers.** Inner padding >= 15 px on all sides; the title occupies the
   top 28 px, so the first child starts at least 35 px below the container
   top. Containers must not be lopsided: centre the content, leave the same
   margin left and right unless a skip connection needs room on one side (then
   keep that side's margin to ~25 px beyond the skip line).
6. **Edges.** Straight lines wherever possible; at most 2 bends per edge unless
   routing around something. No crossings: rearrange the nodes first. Only if a
   crossing is truly unavoidable set `jump=True` on one edge and mention it in
   your report. Never route along another box's border or through a container
   title. Parallel edges must be >= 12 px apart (checker hard limit 6 px).
7. **Skip / residual connections** run on one side of the main column (right
   side by default), 20-30 px outside the widest box; nested skips are
   staggered by 15 px.
8. **Fan-out / fan-in.** Edges from the same source port may share a trunk and
   split at a common point; edges into the same port may merge into a shared
   final segment. Edges with different sources and different targets must never
   share a segment.
9. **Labels.** Use `<br>` for line breaks, `<sub>`/`<sup>` for indices
   (x<sub>t</sub>). ASCII and Greek / math symbols are fine; no emoji. Tensor
   shapes go in `dim_label` text next to an arrow, never on top of a line or
   box. Edge labels (`label=`) only on long straight segments.
10. **Margins.** The diagram is auto-translated so it starts at (10, 10); you
    may start at any coordinate.

## 4. Correctness

Draw the architecture as described in the original paper (or the canonical
implementation). Use the paper's names for components and keep the
granularity useful for someone drawing their own variant: show the key
operators (norms, residuals, attention, gating) but collapse repeated layers
into a `container_repeat` "× N" frame. If a choice is ambiguous (e.g. pre-norm
vs post-norm), follow the most cited version and note it in the report.

## 5. Visual self-review checklist (open every PNG)

* Any line crossing, touching or overlapping another line or box?
* Any arrow head hidden, cramped or pointing at the wrong side?
* Text fits inside every box with breathing room; no clipped or wrapped label.
* Whitespace balanced: content centred in containers, no large empty areas,
  no cramped corners; consistent gaps.
* Colours follow the semantics table; at most ~5 fill colours per figure.
* The figure is readable at the size it would appear in a two-column paper.
* Would a reviewer at CVPR/NeurIPS accept this figure as-is? If not, fix it.
