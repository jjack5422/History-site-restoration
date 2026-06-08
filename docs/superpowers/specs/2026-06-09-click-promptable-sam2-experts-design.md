# Click-promptable SAM2 experts (craquelure / crack) — Design

Date: 2026-06-09
Status: approved (design), pending spec review
Owner: zzz90

## 0. Context and project decomposition

Origin: the user wants to self-host an annotation website (CVAT) on the local WSL2 box
(single user) and label heritage craquelure/crack by **clicking a point in the CVAT UI and
having the model grab the region, SAM2-style, using the user's own fine-tuned model**.

Two findings reshaped this into a two-sub-project effort:

1. **Self-hosted open-source CVAT (Community edition) does NOT support the cloud AI-agents
   API** (`cvat-cli function create-native` / `run-agent`). Verified on docs.cvat.ai
   (2026-06): native functions / AI agents are "only ... CVAT Enterprise or CVAT Cloud,
   version 2.25+". In-UI **interactive** point-click segmentation on self-hosted CE is served
   exclusively by **Nuclio serverless interactor functions**.
2. **None of the existing fine-tuned checkpoints accept a user click.** `model_seg.py`
   (backbone-only dense), `model_decoder_seg.py` (decoder-only, prompt encoder dropped), and
   `model_prompt_seg.py` (learnable prompt tokens that *replace* manual points) are all dense
   / non-interactive. A user-click prompt needs SAM2's real prompt encoder driven by the
   clicked coordinates.

Decomposition (each gets its own spec → plan → implementation):

- **Sub-project 1 (this spec):** train click-promptable SAM2 experts (craq + crack) — a model
  that consumes `(image, positive/negative points, optional previous mask) -> mask`.
- **Sub-project 2 (later):** self-host CVAT CE via Docker on WSL2 and wrap sub-project 1's
  checkpoints as Nuclio **interactor** functions, so a UI click runs the user's model.

Without sub-project 1 there is no promptable model for the interactor to call, so it is first.

## 1. Goal

For each class (craquelure, crack) produce a checkpoint that, given an image plus one or more
user click points (positive and negative, added iteratively, with optional previous-mask
refinement), returns a binary mask of the clicked structure. This is the standard
SAM/SAM2 *interactive segmentation* setting, fine-tuned on the heritage data so quality beats
SAM2 zero-shot (which prior CMC results showed is weak on these cracks).

Two separate per-class experts (chosen over one class-agnostic model): craquelure (dense
filled texture) and crack (thin lines) have very different morphology; the CVAT label the user
selects picks which expert/interactor is invoked.

## 2. Current assets reused (no logic duplication)

- `crack_detection_sam2/model.py`: `build_sam2_model(variant, device, mode)` — builds the SAM2
  stack and exposes `image_encoder`, `sam_prompt_encoder` (with `.pe_layer`), `sam_mask_decoder`,
  `backbone_stride` (16), `hidden_dim`, `use_high_res_features_in_sam`, `image_size`.
- `crack_detection_sam2/model_prompt_seg.py`: `SAM2PromptSeg` already wires
  image_encoder -> sam_prompt_encoder -> sam_mask_decoder for non-1024 input, including the
  prompt-encoder size overrides (`input_image_size`, `image_embedding_size`, `mask_input_size`).
  The new model copies that wiring but feeds **real** point coords/labels instead of learnable
  tokens. `model_prompt_seg.py` is left unchanged.
- `crack_detection_sam2/train_prompt.py`: training loop, BCE+Dice loss, dataset/augment wiring,
  checkpoint save. The new trainer reuses these and adds only point sampling + the iterative loop.
- `crackseg_common`: `dataset` (`TileSegDataset`, `load_tile_index`, `set_class_names`,
  `compute_class_weights`), `augment` (`train_transforms`, `val_transforms`).

## 3. Data

- craquelure: `/home/zzz90/research/_data/labeled32_craq_v3/tiles_512`
- crack:      `/home/zzz90/research/_data/labeled32_crack_v3/tiles_512`
- Both: 512x512 tiles, binary mask {0,1}, split `group_split_stem.json` (4-fold LOSO by stem).
- Input fixed at **512** for v0 (matches existing tiles). Mapping CVAT's arbitrary-size image
  to model input (resize + coordinate scaling) is deferred to sub-project 2.

## 4. Components

### 4.1 `crack_detection_sam2/model_click_seg.py` (new)
`SAM2ClickSeg(nn.Module)`:
- `__init__(variant="small", image_size=512, freeze_image_encoder=True, device=None)` —
  build via `build_sam2_model`, keep `image_encoder` (frozen), `sam_prompt_encoder`,
  `sam_mask_decoder`; apply the same non-1024 size overrides as `SAM2PromptSeg`.
- `forward(image, point_coords, point_labels, prev_mask=None) -> logits [B,1,H,W]`:
  1. `feats = image_encoder(image)` (no grad).
  2. `sparse, dense = sam_prompt_encoder(points=(point_coords, point_labels), boxes=None,
     masks=prev_mask)` — `prev_mask` is the previous low-res logits (dense prompt) or None.
  3. `low_res_logits, iou_pred = sam_mask_decoder(image_embeddings=feats.embed,
     image_pe=sam_prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
     dense_prompt_embeddings=dense, multimask_output=False, ...)` (high-res feats passed when
     `use_high_res_features_in_sam`, mirroring `model_prompt_seg.py`).
  4. upsample `low_res_logits` to `[B,1,H,W]`. Also return the low-res logits for the next
     iteration's `prev_mask`.
- `count_params()` helper (parity with sibling model modules).
- v0: `multimask_output=False` (single mask). Multimask + IoU-token selection deferred.

### 4.2 `crack_detection_sam2/train_click.py` (new)
SAM-style interactive training; per tile run `n_clicks` (default 8) iterations:
1. iter 0: sample one **positive** point uniformly from GT foreground; `prev_mask=None`.
2. forward -> pred mask + low-res logits.
3. error region = symmetric difference (pred vs GT). On the largest error connected component:
   if it is a false-negative (missed GT) sample a **positive** point at/near its center;
   if false-positive sample a **negative** point. Append to the running point set.
4. set `prev_mask` = this iteration's low-res logits (detached); forward again.
5. loss = BCE + Dice on the upsampled logits, summed/averaged over iterations (reuse
   `train_prompt.py`'s loss). Backprop once per tile (accumulated over iters).
- Trainable: prompt encoder + mask decoder. Image encoder frozen.
- CLI mirrors `train_decoder.py` / `train_prompt.py`: `--tiles_root --split --fold --variant
  --epochs --batch_size --class_names --output_dir --n_clicks`. Run craq and crack separately.
- Edge cases: empty-GT tile -> skip click sampling (no positive available), treat as all-negative
  background target; degenerate error region -> stop early for that tile.

### 4.3 `crack_detection_sam2/predict_click.py` (new — consumed by sub-project 2)
- `load_model_from_ckpt(ckpt, device) -> (model, payload)` (same contract name as the existing
  `predict_full.py` modules).
- `predict_click(model, img, pos_points, neg_points, prev_mask, device) -> (mask, low_res_logits)`:
  build point_coords/labels tensors, forward once, threshold to bool mask, return mask plus
  low-res logits so the caller can pass them back as `prev_mask` for the next click. No training
  deps. This is the single inference entry the Nuclio interactor will call.

## 5. Evaluation (interactive-segmentation metrics)

Standard click metrics, not plain IoU, computed per fold on val with **simulated** clicks
(same sampler as training, deterministic seed):
- **IoU@{1,3,5} clicks** — mean mask IoU after k simulated clicks.
- **NoC@0.8** — mean number of clicks to reach IoU >= 0.8 (capped at `n_clicks`; report the cap
  hit-rate).
Written to `metrics.json` alongside per-fold values. These answer "how many clicks to label one
structure", the quantity that matters for the labeling UX.

## 6. Experiment tracking

Per the experiment-tracking convention:
- `crack_detection_sam2/runs/2026-06-09-click-craq-4fold/` and `...-click-crack-4fold/`.
- Each holds `code_snapshot/`, `manifest.md` (exact commands, data path, env, git SHA),
  `metrics.json`, and per-fold subdirs with checkpoints. Use the dedicated training env already
  used for the sam2 dense experts (torch cu128 + editable SAM-2).

## 7. Testing

1. **Shape/forward smoke (offline):** instantiate `SAM2ClickSeg`, forward a 512 tile with 1
   positive point -> assert logits `[1,1,512,512]`, finite; forward again with an added negative
   point + `prev_mask` -> assert shape stable.
2. **Overfit-one-tile sanity:** train on a single craq tile for a few hundred steps -> IoU@5
   should approach ~1.0, confirming the prompt path actually drives the mask.
3. **Eval harness:** run IoU@{1,3,5}/NoC@0.8 on one fold; assert metrics are produced and within
   [0,1] / clicks in `[1, n_clicks]`.
4. **predict_click round-trip:** `load_model_from_ckpt` then `predict_click` on a tile with a
   hand-picked positive point -> returns a bool mask of tile size and reusable low-res logits.

## 8. Constraints / non-goals

- Only craquelure + crack; two separate experts.
- Input fixed at 512 tiles; CVAT arbitrary-size mapping is sub-project 2.
- No box prompt, no multimask output in v0.
- 8 GB RTX 5060: one expert trained at a time; image encoder frozen keeps memory modest.
- **Primary risk — data scarcity:** only ~228 labeled tiles per class. Mitigation: each mask
  yields many sampled point configurations (effective augmentation) plus the existing augment
  pipeline. Fallback if a class fails to learn a usable click response: serve that class's
  existing dense expert via a "click-selects-connected-component" Nuclio interactor (the earlier
  "path 1") instead of a promptable model — sub-project 2 can mix the two.
- Quality is an accelerator for human labeling, not ground truth; the user reviews every mask.

## 9. Open items to confirm before implementation

- Exact SAM2 mask-decoder call signature for high-res features in this repo's vendored SAM-2
  (read `model_prompt_seg.py`'s decoder call and reuse verbatim).
- Whether `n_clicks=8` and per-iteration loss averaging are kept, or only final-iteration loss
  (decide during plan; default = average over iterations).
