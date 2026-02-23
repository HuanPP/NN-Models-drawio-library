# Fix Edge Crossings in Neural Network Model Library

## TL;DR

> **Quick Summary**: Fix edge/line crossings and overlaps in 12+ model diagrams by redesigning edge routing to use explicit multi-segment paths that go AROUND obstacles rather than through them.
> 
> **Deliverables**:
> - Fixed `model_unet()` with skip connections routed to far right
> - Fixed `model_faster_rcnn()` with backbone→roi routed below RPN
> - Fixed `model_conditional_gan()` with condition lines routed to far left
> - Fixed 9+ other models with similar crossing issues
> - Regenerated `ieee_trans_nn_model_library.xml`
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1-3 (high priority) → Task 4-12 (audit) → Task 13 (regenerate)

---

## Context

### Original Request
User requested fixes for edge crossings in neural network model diagrams. Specific issues identified:
- U-Net: Skip connections cross through decoder path
- Faster R-CNN: backbone→roi line crosses through RPN cls/reg
- Conditional GAN: c→concat lines cross through Fake x module

### Previous Work
- Created `gen_model_library.py` (~1176 lines) generating 38 model diagrams
- Added routing helpers: `connect_points_via_x()`, `connect_points_via_y()`, `connect_skip_vertical()`, `connect_skip_horizontal()`
- Added edge styles: `arrow_seg`, `arrow_skip_seg` for controlled routing

### Key Insight
The `orthogonalEdgeStyle` auto-router in draw.io doesn't respect intended paths. We must use explicit 3+ segment routing with `connect_points_via_x/y` for ALL multi-hop connections, calculating via-points that route AROUND obstacles.

---

## Work Objectives

### Core Objective
Eliminate all visible edge crossings and module overlaps in the 38 neural network model diagrams.

### Concrete Deliverables
- Modified `gen_model_library.py` with fixed routing for all 12+ problematic models
- Regenerated `ieee_trans_nn_model_library.xml` (38 entries, ~50KB)

### Definition of Done
- [ ] `python gen_model_library.py` runs without error
- [ ] Visual inspection shows NO edge crossings in any of 38 diagrams
- [ ] All skip connections route around decoder/discriminator columns, not through

### Must Have
- Skip connections in U-Net route to FAR RIGHT of decoder column
- Backbone→roi in Faster R-CNN routes BELOW RPN container
- Condition lines in Conditional GAN route to FAR LEFT before curving

### Must NOT Have (Guardrails)
- No edges passing through modules/containers
- No edges crossing other edges unnecessarily
- No overlapping elements
- No auto-router reliance - all multi-hop connections must be explicit segments

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (Python script)
- **Automated tests**: None - visual verification required
- **Framework**: N/A

### QA Policy
Every task includes visual verification via importing into draw.io.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — 3 critical fixes):
├── Task 1: Fix U-Net skip connections [deep]
├── Task 2: Fix Faster R-CNN routing [deep]
└── Task 3: Fix Conditional GAN routing [deep]

Wave 2 (After Wave 1 — audit and fix remaining models):
├── Task 4: Fix FCN Decoder edge overlaps [quick]
├── Task 5: Fix VAE edge crossings [quick]
├── Task 6: Fix GAN curved arrow crossing [quick]
├── Task 7: Fix Multi-Head Attention fan-out [quick]
├── Task 8: Fix BiLSTM input fan-out [quick]
├── Task 9: Fix Siamese Network emb→dist [quick]
├── Task 10: Fix GAT Layer multi-input [quick]
├── Task 11: Fix EfficientNet MBConv SE block [quick]
└── Task 12: Audit remaining 27 models [unspecified-high]

Wave 3 (After Wave 2 — verification):
└── Task 13: Regenerate XML and visual verification [unspecified-high]

Critical Path: Tasks 1-3 → Tasks 4-12 → Task 13
```

### Dependency Matrix
| Task | Depends On | Blocks |
|------|------------|--------|
| 1-3  | None       | 13     |
| 4-12 | None       | 13     |
| 13   | 1-12       | Done   |

---

## TODOs

 [x] 1. Fix U-Net Skip Connections

  **What to do**:
  - Modify `model_unet()` function (line ~415-452)
  - Route ALL three skip connections through a vertical bus on the FAR RIGHT side (x ≈ 536, past all decoder elements)
  - Use 4-segment routing for each skip:
    1. H segment: `enc[N].right` → `skip_bus_x`
    2. V segment: down to `cat[N].y - 10`
    3. H segment: left to `cat[N].x`
    4. V segment: down to `cat[N].top` (with arrow)
  - Stagger the three bus lines (x=536, x=521, x=506) to prevent overlap
  - Use `arrow_skip_seg` style for non-final segments, `arrow_skip` for final

  **Implementation snippet**:
  ```python
  # Skip bus on far right of diagram
  skip_bus_x = dec3['x'] + dec3['w'] + 30  # ~536
  
  # Skip 1: enc1 -> cat1
  enc1_rx, enc1_ry = right(enc1)
  cat1_tx, cat1_ty = cx(cat1), cat1['y']
  seg_style = 'arrow_skip_seg'
  connect_points(d, enc1_rx, enc1_ry, skip_bus_x, enc1_ry, seg_style)
  connect_points(d, skip_bus_x, enc1_ry, skip_bus_x, cat1_ty - 10, seg_style)
  connect_points(d, skip_bus_x, cat1_ty - 10, cat1_tx, cat1_ty - 10, seg_style)
  connect_points(d, cat1_tx, cat1_ty - 10, cat1_tx, cat1_ty, 'arrow_skip')
  
  # Similar for enc2->cat2 at skip_bus_x-15
  # Similar for enc3->cat3 at skip_bus_x-30
  ```

  **Must NOT do**:
  - Do not use `connect_skip_vertical()` - it routes too close to decoder
  - Do not rely on orthogonalEdgeStyle auto-router

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires careful geometric calculation and understanding of existing routing helpers
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3)
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:415-452` - Current model_unet() implementation
  - `gen_model_library.py:210-248` - Routing helper functions
  - `gen_model_library.py:75-78` - Edge styles including arrow_skip_seg

  **Acceptance Criteria**:
  - [ ] Three skip connections visible as dashed lines
  - [ ] All three route to the RIGHT of dec1, dec2, dec3 columns
  - [ ] No crossing with up1, up2, up3, cat1, cat2, cat3, dec1, dec2, dec3

  **QA Scenarios**:
  ```
  Scenario: U-Net skip connections route correctly
    Tool: Bash (python + manual visual check)
    Steps:
      1. Run `python gen_model_library.py`
      2. Import ieee_trans_nn_model_library.xml into draw.io
      3. Find "U-Net" diagram
      4. Verify: enc1→cat1 skip line goes RIGHT past dec1 column before connecting
      5. Verify: enc2→cat2 skip line goes RIGHT past dec2 column before connecting
      6. Verify: enc3→cat3 skip line goes RIGHT past dec3 column before connecting
      7. Verify: No dashed lines cross through decoder elements
    Expected Result: All 3 skip connections route around decoder column on the right
    Evidence: .sisyphus/evidence/task-1-unet-skip.png
  ```

  **Commit**: YES (group with 2, 3)
  - Message: `fix(models): route U-Net/Faster-RCNN/cGAN edges around obstacles`
  - Files: `gen_model_library.py`

---

 [x] 2. Fix Faster R-CNN Routing

  **What to do**:
  - Modify `model_faster_rcnn()` function (line ~507-528)
  - Current issue: `backbone→roi` line (line 522) crosses through RPN cls/reg outputs
  - Solution: Route backbone→roi BELOW the RPN container
  - Add explicit 3-segment routing:
    1. backbone.right → via_x (just past RPN right edge)
    2. via_x down to below RPN container (y ≈ 175)
    3. via_x right to roi.left

  **Implementation snippet**:
  ```python
  # Route backbone->roi BELOW RPN container
  rpn_container_bottom = 175  # RPN container ends at y=175
  rpn_container_right = 274   # RPN container ends at x=274
  
  bk_rx, bk_ry = right(backbone)
  roi_lx, roi_ly = left(roi)
  
  # Don't use direct connect_right - route around
  via_x = rpn_container_right + 10
  via_y = rpn_container_bottom + 15
  connect_points(d, bk_rx, bk_ry, via_x, bk_ry, 'arrow_seg')
  connect_points(d, via_x, bk_ry, via_x, via_y, 'arrow_seg')
  connect_points(d, via_x, via_y, roi_lx, via_y, 'arrow_seg')
  connect_points(d, roi_lx, via_y, roi_lx, roi_ly, 'arrow')
  ```

  **Must NOT do**:
  - Do not use simple `connect_right(backbone, roi)` - it crosses RPN

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3)
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:507-528` - Current model_faster_rcnn() implementation
  - RPN container: x=112, y=50, w=162, h=125 (bottom at y=175, right at x=274)

  **Acceptance Criteria**:
  - [ ] backbone→roi line visible
  - [ ] Line does NOT cross through RPN cls or RPN reg boxes
  - [ ] Line routes around (below or to the right of) RPN container

  **QA Scenarios**:
  ```
  Scenario: Faster R-CNN backbone-to-roi routes around RPN
    Tool: Bash + visual
    Steps:
      1. Run `python gen_model_library.py`
      2. Import into draw.io, find "Faster R-CNN"
      3. Trace backbone→roi line
      4. Verify: line does not intersect RPN cls or RPN reg boxes
    Expected Result: backbone→roi routes below or around RPN container
    Evidence: .sisyphus/evidence/task-2-fasterrcnn-routing.png
  ```

  **Commit**: YES (group with 1, 3)

---

 [x] 3. Fix Conditional GAN Routing

  **What to do**:
  - Modify `model_conditional_gan()` function (line ~848-880)
  - Current issue: c→concat_fake and c→concat_real lines (lines 870-871) cross through "Fake x" module
  - Solution: Route condition lines to FAR LEFT first, then down, then right
  - Use vertical bus at x ≈ 5 (left of everything)

  **Implementation snippet**:
  ```python
  # Route condition 'c' to concat nodes via LEFT side
  cond_bus_x = 5  # Far left of diagram
  
  cond_rx, cond_ry = right(cond)
  # First go LEFT from cond
  cond_lx = cond['x']
  
  # c -> concat_fake (route: left, down, right)
  cf_lx, cf_ly = left(concat_fake)
  connect_points(d, cond_lx, cond_ry, cond_bus_x, cond_ry, 'arrow_seg')
  connect_points(d, cond_bus_x, cond_ry, cond_bus_x, cf_ly, 'arrow_seg')
  connect_points(d, cond_bus_x, cf_ly, cf_lx, cf_ly, 'arrow')
  
  # c -> concat_real (route: left, down further, right)
  cr_lx, cr_ly = left(concat_real)
  connect_points(d, cond_bus_x, cond_ry, cond_bus_x, cr_ly, 'arrow_seg')
  connect_points(d, cond_bus_x, cr_ly, cr_lx, cr_ly, 'arrow')
  ```

  **Must NOT do**:
  - Do not route through x=246 (cond_bus_x = concat_fake['x'] - 34) as in current code

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:848-880` - Current model_conditional_gan() implementation
  - Current cond_bus_x calculation at line 869

  **Acceptance Criteria**:
  - [ ] c→concat_fake line visible
  - [ ] c→concat_real line visible
  - [ ] Neither line crosses through "Fake x" box
  - [ ] Lines route to LEFT of generator, not RIGHT

  **QA Scenarios**:
  ```
  Scenario: Conditional GAN condition routing avoids Fake x
    Tool: Bash + visual
    Steps:
      1. Run `python gen_model_library.py`
      2. Import into draw.io, find "Conditional GAN"
      3. Trace c→concat_fake and c→concat_real lines
      4. Verify: neither line intersects "Fake x" module
    Expected Result: Condition lines route to far left, then down, then right to concat nodes
    Evidence: .sisyphus/evidence/task-3-cgan-routing.png
  ```

  **Commit**: YES (group with 1, 2)

---

 [x] 4. Fix FCN Decoder Edge Overlaps

  **What to do**:
  - Review `model_fcn_decoder()` (line ~531-551)
  - Issue: Multiple edges converge at add nodes from different Y positions
  - Add staggered via-points to prevent overlap

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:531-551` - model_fcn_decoder()

  **QA Scenarios**:
  ```
  Scenario: FCN Decoder edges don't overlap
    Tool: Visual inspection
    Steps:
      1. Check FCN Decoder diagram in draw.io
      2. Verify no edges overlap at add nodes
    Expected Result: Clean routing at add nodes
    Evidence: .sisyphus/evidence/task-4-fcn-decoder.png
  ```

  **Commit**: YES (group with 5-12)

---

 [x] 5. Fix VAE Edge Crossings

  **What to do**:
  - Review `model_vae()` (line ~763-791)
  - Issue: Edges from enc to mu/sigma may overlap; loss connections cross main path
  - Route loss connections (dashed) to TOP of diagram, not crossing main flow

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:763-791` - model_vae()

  **QA Scenarios**:
  ```
  Scenario: VAE edges don't cross
    Tool: Visual inspection
    Steps:
      1. Check VAE diagram
      2. Verify loss connections (dashed) route cleanly to loss box
    Expected Result: No crossing between main flow and loss connections
    Evidence: .sisyphus/evidence/task-5-vae.png
  ```

  **Commit**: YES (group with 4, 6-12)

---

 [x] 6. Fix GAN Curved Arrow Crossing

  **What to do**:
  - Review `model_gan()` (line ~794-814)
  - Issue: `fake→d1` uses `arrow_curved` which may cross through `real` box
  - Change to explicit segmented routing that goes ABOVE real box

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:794-814` - model_gan()
  - Line 811: `connect_points(..., 'arrow_curved')`

  **QA Scenarios**:
  ```
  Scenario: GAN fake→discriminator doesn't cross real
    Tool: Visual inspection
    Steps:
      1. Check GAN diagram
      2. Verify fake→d1 line doesn't cross through "Real x" box
    Expected Result: Fake→discriminator routes around real input
    Evidence: .sisyphus/evidence/task-6-gan.png
  ```

  **Commit**: YES (group with 4-5, 7-12)

---

 [x] 7. Fix Multi-Head Attention Fan-out

  **What to do**:
  - Review `model_multi_head_attention_detail()` (line ~724-748)
  - Issue: Input fans out to Q, K, V - these 3 edges may overlap
  - Use staggered via-points for clean fan-out

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:724-748` - model_multi_head_attention_detail()

  **QA Scenarios**:
  ```
  Scenario: MHA input fan-out is clean
    Tool: Visual inspection
    Steps:
      1. Check Multi-Head Attn Detail diagram
      2. Verify input→Q, input→K, input→V edges don't overlap
    Expected Result: Clean 3-way fan-out with staggered routing
    Evidence: .sisyphus/evidence/task-7-mha.png
  ```

  **Commit**: YES (group with 4-6, 8-12)

---

 [x] 8. Fix BiLSTM Input Fan-out

  **What to do**:
  - Review `model_bilstm()` (line ~920-934)
  - Issue: Input fans to Forward LSTM and Backward LSTM - may overlap
  - Add staggered via-points

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:920-934` - model_bilstm()

  **QA Scenarios**:
  ```
  Scenario: BiLSTM fan-out is clean
    Tool: Visual inspection
    Steps:
      1. Check Bi-LSTM diagram
      2. Verify input→fwd and input→bwd edges don't overlap
    Expected Result: Clean 2-way fan-out
    Evidence: .sisyphus/evidence/task-8-bilstm.png
  ```

  **Commit**: YES (group with 4-7, 9-12)

---

 [x] 9. Fix Siamese Network emb→dist

  **What to do**:
  - Review `model_siamese_network()` (line ~1004-1023)
  - Issue: emb1→dist and emb2→dist may overlap at dist input
  - Add staggered via-points

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:1004-1023` - model_siamese_network()

  **QA Scenarios**:
  ```
  Scenario: Siamese embedding→distance is clean
    Tool: Visual inspection
    Steps:
      1. Check Siamese Network diagram
      2. Verify emb A→dist and emb B→dist edges are distinct
    Expected Result: Clean convergence at distance node
    Evidence: .sisyphus/evidence/task-9-siamese.png
  ```

  **Commit**: YES (group with 4-8, 10-12)

---

 [x] 10. Fix GAT Layer Multi-input

  **What to do**:
  - Review `model_gat_layer()` (line ~1097-1116)
  - Issue: Multiple node inputs fan into attention and multiply nodes
  - Add staggered via-points for clean fan-in

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:1097-1116` - model_gat_layer()

  **QA Scenarios**:
  ```
  Scenario: GAT Layer fan-in is clean
    Tool: Visual inspection
    Steps:
      1. Check GAT Layer diagram
      2. Verify ni, nj, nk edges don't overlap
    Expected Result: Clean multi-input routing
    Evidence: .sisyphus/evidence/task-10-gat.png
  ```

  **Commit**: YES (group with 4-9, 11-12)

---

 [x] 11. Fix EfficientNet MBConv SE Block

  **What to do**:
  - Review `model_efficientnet_mbconv()` (line ~1069-1094)
  - Issue: SE block (GAP→FC→Sigmoid) routing and skip connection may cross
  - Ensure skip connection routes FAR RIGHT, SE block routes FAR LEFT

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py:1069-1094` - model_efficientnet_mbconv()

  **QA Scenarios**:
  ```
  Scenario: MBConv SE block and skip don't cross
    Tool: Visual inspection
    Steps:
      1. Check EfficientNet MBConv diagram
      2. Verify SE block routing and skip connection are separate
    Expected Result: Clean SE block path and skip path
    Evidence: .sisyphus/evidence/task-11-mbconv.png
  ```

  **Commit**: YES (group with 4-10, 12)

---

 [x] 12. Audit Remaining 27 Models

  **What to do**:
  - Visually inspect all remaining models not covered by Tasks 1-11
  - Models to check: LeNet-5, AlexNet, VGG Block, ResNet Block, ResNet Bottleneck, DenseNet Block, Inception Module, MobileNet Block, FPN, YOLO Head, ASPP Module, Transformer Encoder, Transformer Decoder, Full Transformer, ViT, BERT Block, GPT Block, Autoencoder, DCGAN Generator, DCGAN Discriminator, Diffusion Model, Stacked LSTM, Stacked GRU, Seq2Seq, Seq2Seq+Attention, MLP-Mixer, Swin Transformer Block
  - Fix any crossing/overlap issues found using same routing techniques

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 13
  - **Blocked By**: None

  **References**:
  - `gen_model_library.py` - All model functions

  **QA Scenarios**:
  ```
  Scenario: All 38 models have clean routing
    Tool: Visual inspection
    Steps:
      1. Import library into draw.io
      2. Check each of 38 diagrams
      3. Flag any remaining crossings
    Expected Result: Zero edge crossings in any diagram
    Evidence: .sisyphus/evidence/task-12-audit-all.md
  ```

  **Commit**: YES (group with 4-11)

---

- [ ] 13. Regenerate XML and Final Verification

  **What to do**:
  - Run `python gen_model_library.py` to regenerate library
  - Import `ieee_trans_nn_model_library.xml` into draw.io
  - Verify all 38 diagrams visually
  - Capture screenshots of fixed diagrams as evidence

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`dev-browser`]
    - `dev-browser`: For importing XML into draw.io and visual verification

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (Sequential after Wave 2)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 1-12

  **References**:
  - `gen_model_library.py:1165-1175` - main() function
  - Output: `ieee_trans_nn_model_library.xml`

  **Acceptance Criteria**:
  - [ ] `python gen_model_library.py` outputs "Wrote ... with 38 models"
  - [ ] XML file size ~50KB
  - [ ] All 38 diagrams render correctly in draw.io
  - [ ] Zero visible edge crossings

  **QA Scenarios**:
  ```
  Scenario: Final library verification
    Tool: Bash + dev-browser
    Steps:
      1. cd /Users/huan/Downloads/drawio-hub && python gen_model_library.py
      2. Verify output: "Wrote ieee_trans_nn_model_library.xml with 38 models"
      3. Open draw.io, import library
      4. Check U-Net, Faster R-CNN, Conditional GAN specifically
      5. Spot-check 10 other random diagrams
    Expected Result: All diagrams render with clean edge routing
    Evidence: .sisyphus/evidence/task-13-final-verification.png
  ```

  **Commit**: NO (already committed in earlier tasks)

---

## Final Verification Wave

> After ALL implementation tasks complete, run final verification.

- [ ] F1. **Visual Verification of All 38 Diagrams**
  Import library into draw.io. Check each diagram for edge crossings. Document any remaining issues.
  Output: `Pass/Fail per diagram | VERDICT: APPROVE/REJECT`

---

## Commit Strategy

| Task(s) | Message | Files |
|---------|---------|-------|
| 1-3 | `fix(models): route U-Net/Faster-RCNN/cGAN edges around obstacles` | `gen_model_library.py` |
| 4-12 | `fix(models): clean up remaining edge crossings in 9+ diagrams` | `gen_model_library.py` |

---

## Success Criteria

### Verification Commands
```bash
cd /Users/huan/Downloads/drawio-hub
python gen_model_library.py
# Expected: Wrote ieee_trans_nn_model_library.xml with 38 models
ls -la ieee_trans_nn_model_library.xml
# Expected: ~50KB file
```

### Final Checklist
- [ ] All 3 high-priority models (U-Net, Faster R-CNN, cGAN) have clean routing
- [ ] All 9 medium-priority models audited and fixed
- [ ] Remaining 26 models audited
- [ ] Zero edge crossings visible in any diagram
- [ ] XML regenerated successfully with 38 entries
