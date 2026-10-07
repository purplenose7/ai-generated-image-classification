# AI-Image-Detector Robustness Audit — Full 20-Day Build Guide (Path A)
### GenImage · ResNet-50 vs ViT-B/16 · OOD + degradation failure analysis · calibrated demo
*~1.5–2 hrs/day · prereqs (theory) done · starts at dataset download · every day self-contained — you should never need to ask another tool "what do I do?"*

**The thesis (this is what makes it not generic):** transfer-learned CNN and ViT detectors hit high in-distribution accuracy, but that reliability is an illusion — they collapse on (a) generators never seen in training and (b) ordinary image degradation (recompression, rescaling, blur, screenshotting). This project **quantifies both failure modes, explains the mechanism** with frequency-domain + interpretability analysis, and ships an **honestly-calibrated demo** that surfaces its own uncertainty. Your resume verb is *audited the reliability of*, not *built a classifier*.

---

## How to use this (read once)

- **Read only today's box.** The file is long because it's complete, not because a day is. One box = one calm 1.5–2 hour session.
- **No code, ever — but no vagueness either.** You get exact **folder names, file names, function names, inputs, returns, and a numbered plain-English algorithm**. You write the actual code. That gap is the learning and the interview defense.
- **I never hand you a value; I hand you the *method*.** Learning rate, batch size, epochs, dropout, degradation severities — you choose them via the **"How to choose values"** box and log the reasoning.
- **`LOG.md` and `README.md` prompts are in every day.** Write them the same day; by Day 19 the README is 80% assembled.
- **Buffers are permission to fall behind.** Days 8, 12, 15, and 20 are slack. Given a cramped 1.5-hr norm, they're your safety margin — the two things they *never* absorb are the OOD why-analysis and the robustness arm (Days 13–15), your crown jewels.

### Project structure (create in your GitHub repo now)

```
ai-image-forensics/
├── README.md
├── LOG.md
├── requirements.txt
├── .gitignore                # never commit images; manifests OK; checkpoints/ figures/ optional
├── app.py                    # Gradio app for Hugging Face Spaces (Day 17–18)
├── config/
│   └── settings.py           # seed, class names, paths, model input size
├── data/
│   ├── build_working_set.py  # raw GenImage → balanced manifest (paths only)
│   ├── make_splits.py        # stratified train/val/test + OOD manifests + leakage checks
│   └── manifests/            # train.csv, val.csv, test.csv, ood.csv  (commit these)
├── src/
│   ├── dataset.py            # transforms + Dataset + dataloader factory
│   ├── eda_fft.py            # frequency-domain analysis
│   ├── models.py             # build_resnet, build_vit  (NO scratch baseline in Path A)
│   ├── engine.py             # train_one_epoch, evaluate, fit
│   ├── metrics.py            # f1, confusion, ood_report
│   ├── robustness.py         # degradations, degraded loaders, sweep, reliability curves
│   ├── gradcam_utils.py      # misclassified finder, grad-cam, attention rollout
│   └── viz.py                # sample grids, confusion plots
├── notebooks/
│   ├── 01_pipeline_and_fft.ipynb
│   ├── 02_resnet.ipynb
│   ├── 03_vit.ipynb
│   ├── 04_audit_ood_robustness.ipynb
│   └── 05_eval_and_deploy.ipynb
├── checkpoints/              # gitignored
└── figures/                  # plots for README
```

**Kaggle workflow:** develop each function in a notebook cell; once it works, save it into its `src/`/`data/` file with the `%%writefile path/to/file.py` cell magic, then `import` it back — this keeps your Kaggle notebook and GitHub repo identical. Attach GenImage subsets via **Add Data**; never copy images into `/kaggle/working` — manifests store paths into `/kaggle/input/...`.

### `config/settings.py` — build first (Day 1, Hour 1)
Constants: `SEED`; `CLASS_NAMES = ["real", "gan", "diffusion"]`; `CLASS_TO_IDX`; `MODEL_INPUT_SIZE`; path variables for each GenImage subset root, `MANIFEST_DIR`, `CHECKPOINT_DIR`, `FIGURE_DIR`, `DEGRADED_DIR`.
One function: **`set_global_seed(seed)`** → seeds Python `random`, NumPy, PyTorch (CPU+CUDA), sets deterministic flags. Call at the top of every notebook.

---

## How to choose values (methods, not numbers) — your reference box

- **Learning rate.** Either (1) *LR range test:* ramp LR up over a few hundred batches, plot loss vs LR, use the value one notch below where loss stops falling / spikes; or (2) *modest sweep:* try 3–4 LRs across ~2 orders of magnitude, keep the best **validation** curve. For transfer learning: head learns at a normal LR, backbone much slower — find the ratio by trying a couple and watching for early val-loss jumps (= backbone LR too high).
- **Batch size.** Largest that fits Kaggle GPU memory without OOM; if you change it a lot, re-check LR.
- **Epoch budget.** Don't hand-pick — set a generous max, let **early stopping** decide.
- **Dropout / weight decay.** Start modest; if the train-vs-val gap widens (overfitting), increase. Expect **ViT to need more** than ResNet at your ~10k scale — plan to sweep it up.
- **Early-stopping patience.** Enough that noise doesn't trigger it, few enough to save GPU quota.
- **Class weighting.** Balanced set → minor; if used, weight inversely to training-manifest frequency.
- **Degradation severity (robustness arm).** For each degradation, pick **3–4 levels** spanning "barely visible" → "clearly degraded but content still obvious." Method: apply each to a few sample images, eyeball them, choose levels where a human still easily reads the image but quality visibly drops. The levels become the x-axis of your reliability curve — so space them to show a *trend*, not one cliff.

---

# LEVEL 1 — DATA (Days 1–4)
*Goal: a balanced, leak-free GenImage manifest with a GLIDE OOD hold-out, a working DataLoader, and an FFT figure.*

---

### Day 1 · Acquire + inspect GenImage, lock the label map
**Mission:** understand exactly what GenImage gives you and commit your class definitions in code.

**Files & functions today:** `config/settings.py` (finish) · `notebooks/01_pipeline_and_fft.ipynb`

**Hour 1 — Attach + map the terrain.** On Kaggle, **Add Data** for: **BigGAN**, **Stable Diffusion v1.4**, **ADM**, and **GLIDE** (GLIDE is your OOD hold-out — attach it, never train on it). Confirm the layout with your own eyes: GenImage subsets follow `imagenet_<generator>/{train,val}/{ai,nature}/`. Record each subset's path root and fill in `config/settings.py`.

**Hour 2 — Verify the framing.** In a markdown cell answer: which subsets are GAN (BigGAN) vs diffusion (SD1.4, ADM, GLIDE)? Are `nature` images consistent across subsets? Does the Real/GAN/Diffusion split hold cleanly (BigGAN→gan, SD1.4+ADM→diffusion, nature→real, GLIDE→OOD)?

**Decisions to make:** none numeric — verification day.

**Boss Check (→ LOG.md):** why does holding GLIDE out entirely make the OOD test *honest*? What would be dishonest about testing on a generator you'd trained on?

**LOG.md — write:** the four subsets + their roles; confirmed folder layout; one-line label map.
**README.md — draft:** a "Dataset" stub — GenImage, the generators, the Real/GAN/Diffusion definition.

**Loot:** verified data source + locked class map. **XP:** you can navigate GenImage blindfolded.

---

### Day 2 · Build the balanced working-set manifest (+ OOD manifest)
**Mission:** turn ~1.3M raw images into a balanced 15,000-path manifest (no copying) + a separate OOD manifest.

**Files & functions today:** `data/build_working_set.py`

**Target sizes (defensible; scale down if Kaggle time is tight):** ID **15,000 = 5,000/class** — Real: 5,000 `nature`; GAN: 5,000 BigGAN `ai`; Diffusion: 2,500 SD1.4 `ai` + 2,500 ADM `ai`. OOD **3,000 = 1,500 GLIDE `ai`** (truth = diffusion) **+ 1,500 fresh `nature`** not in the ID set.

**Functions (name · inputs → returns · algorithm):**
1. **`list_source_images(subset_root, split, label_folder)`** · → sorted list of paths. Algorithm: join `subset_root/split/label_folder`; collect image files; return sorted (sorting + seed = reproducibility).
2. **`sample_paths(paths, n, seed)`** · → n paths. Algorithm: seeded RNG; if fewer than n, warn + take all; else shuffle a copy, take first n.
3. **`build_working_set(class_spec, out_csv)`** · → DataFrame + CSV. Algorithm: per class, read its sources/counts; `list_source_images` → `sample_paths`; build rows `[filepath, class_name, class_idx, generator, source_split]`; concat; assert balance; save; return. No image copying.
4. **`build_ood_set(glide_root, nature_source, n_ai, n_real, out_csv)`** · → DataFrame + CSV. Algorithm: sample `n_ai` GLIDE `ai` (label `diffusion`, `generator="glide"`, `is_ood=True`); sample `n_real` `nature` **not** in the ID set (label `real`); concat; save.

**Decisions to make:** total size (defaults above — change only with a logged reason).

**Boss Check (→ LOG.md):** why sort file lists *before* seeded sampling instead of trusting directory order?

**LOG.md — write:** final per-class/per-generator counts; your `class_spec`; confirmation ID and OOD reals don't overlap.
**README.md — append:** finalize "Dataset" with real counts + the OOD design (GLIDE held out to test unseen-generator generalization).

**Loot:** `working_set.csv` + `ood.csv`. **XP:** you turned an unusable dump into a clean experimental dataset.

---

### Day 3 · Stratified splits, leakage checks, DataLoader
**Mission:** carve trustworthy splits and build the pipe that feeds the models.

**Files & functions today:** `data/make_splits.py` · `src/dataset.py`

**Functions — `data/make_splits.py`:**
1. **`make_stratified_splits(working_csv, ratios, seed, out_dir)`** · → writes train/val/test CSVs. Algorithm: read; group by **(class_name, generator)** so every split sees every class *and* generator proportionally; within each group shuffle (seeded) + slice by `ratios`; concat per split; add `split` column; save.
2. **`assert_no_leakage(train_csv, val_csv, test_csv, ood_csv)`** · → raises or prints "clean". Algorithm: load each `filepath` set; assert pairwise intersections empty; assert `glide` appears in none of train/val/test; assert no OOD real path is in ID reals.
3. **`summarize_splits(out_dir)`** · → prints per-split class + generator counts.

**Functions — `src/dataset.py`:**
4. **`build_transforms(train)`** · → transform pipeline. Algorithm: always resize to `MODEL_INPUT_SIZE`, to-tensor, normalize with the **mean/std your pretrained backbones expect** (a lookup, not a hyperparameter); if `train`, insert your chosen augmentations before tensor/normalize.
5. **`class ForensicsDataset(Dataset)`** · (manifest_csv, transform) → `(image_tensor, class_idx)`. Algorithm: `__init__` reads CSV + stores transform; `__len__` = row count; `__getitem__` loads image (RGB), transforms, returns tensor + `class_idx`.
6. **`make_dataloader(manifest_csv, train, batch_size, num_workers)`** · → DataLoader. Algorithm: build dataset with `build_transforms(train)`; DataLoader with `shuffle=train`; small `num_workers` on Kaggle.

**Decisions to make:** split `ratios`; `batch_size` (values box); which augmentations.

**Boss Check (→ LOG.md):** (1) why stratify on **generator**, not just class? (2) which augmentations did you *reject*, and which could **erase the high-frequency artifact** you're detecting? (The single most important reasoning of the data phase.)

**LOG.md — write:** ratios + why; augmentation choices + rejected ones + reasons; "clean" leakage confirmation; batch size + why.
**README.md — append:** "Data pipeline" note — input size, normalization, train-only augmentation, leakage checks pass.

**Loot:** committed manifests + working DataLoader. **XP:** you know why most beginner accuracy numbers are inflated — yours won't be.

---

### Day 4 · Frequency-domain analysis (look first) + buffer
**Mission:** reveal what the eye can't; form your *own* interpretation before verifying.

**Files & functions today:** `src/eda_fft.py` · `src/viz.py`

**Functions — `src/eda_fft.py`:**
1. **`compute_avg_spectrum(manifest_csv, class_name, n_samples, seed)`** · → 2D array. Algorithm: sample n rows of the class; per image → grayscale → `np.fft.fft2` → `np.fft.fftshift` → magnitude → `log1p`; average the stack; return.
2. **`plot_class_spectra(spectra_by_class, out_png)`** · → side-by-side figure, same color scale, saved to `figures/`.

**Functions — `src/viz.py`:**
3. **`show_samples(manifest_csv, n_per_class, out_png)`** · → labeled grid saved to `figures/`.

**Order:** run `show_samples` first (just look), *then* FFT. **Write what you see in your own words before reading about GAN artifacts** — interpretation first.

**Decisions to make:** `n_samples` (enough to average noise, few enough to be quick).

**Buffer:** if Day 3 ran long, FFT slides here guilt-free.

**Boss Check (→ LOG.md):** in your own words, what does each class's spectrum look like and what might physically explain it (think GAN upsampling)? Be honest if the diffusion signature is subtle.

**LOG.md — write:** pre-reading interpretation, then post-verification correction.
**README.md — append:** embed `fft_spectra.png` with a 2–3 sentence caption on why frequency matters to detection.

**Loot:** two README-grade figures. **XP:** Level 1 clear — data you trust + an insight most projects lack.

---

# LEVEL 2 — RESNET (Days 5–8)
*Goal: build the reusable training engine, run ResNet transfer learning properly (two runs), Grad-CAM as a diagnostic, first OOD gap.*

---

### Day 5 · Build the engine + ResNet feature-extraction (Run 1)
**Mission:** build the machine every model reuses, then run your first real detector. **This is the meatiest build day — the Level-2 buffer (Day 8) exists for exactly this.**

**Files & functions today:** `src/models.py` · `src/engine.py` · `src/metrics.py` · `notebooks/02_resnet.ipynb`

**Functions — `src/metrics.py`:**
1. **`per_class_f1(y_true, y_pred, class_names)`** · → dict class→F1.
2. **`confusion(y_true, y_pred, num_classes)`** · → confusion-matrix array.

**Functions — `src/engine.py`:**
3. **`train_one_epoch(model, loader, optimizer, loss_fn, device)`** · → avg train loss. Algorithm: train mode; per batch → to device, zero grads, forward, loss, backward, step, accumulate; return mean.
4. **`evaluate(model, loader, loss_fn, device)`** · → `{loss, accuracy, per_class_f1, confusion, y_true, y_pred}`. Algorithm: eval + no-grad; collect preds/truths; compute metrics.
5. **`fit(model, train_loader, val_loader, optimizer, scheduler, loss_fn, device, max_epochs, patience, ckpt_path, wandb_run)`** · → best val metrics + best checkpoint path. Algorithm: per epoch → `train_one_epoch` → `evaluate` on val → `wandb.log` losses/accuracy/F1 → step scheduler → save checkpoint if val improved (to `/kaggle/working`) → early-stop after `patience` no-improve epochs → return best.

**Functions — `src/models.py`:**
6. **`build_resnet(num_classes, mode)`** · (`mode` in `{"feature_extraction","finetune"}`) → model. Algorithm: load ResNet-50 pretrained; find its final-layer input feature count yourself + swap in a fresh head for `num_classes` (add head regularization if you choose); if `feature_extraction`, freeze backbone (head trainable); if `finetune`, all trainable; return.
7. **`count_parameters(model)`** · → (trainable, total).

**The work:** build ResNet in feature-extraction mode; call `fit`; checkpoint each epoch + Save Version. Pick head LR via the values-box method (fresh head takes a normal LR). Then `evaluate` on the **test** split.

**Decisions to make:** head regularization; optimizer/LR/schedule/epochs/patience (values box).

**Boss Check (→ LOG.md):** why does feature extraction come *before* fine-tuning? How would the val curve reveal your head LR is too high/low?

**LOG.md — write:** architecture-head choice; LR method + value; ResNet-FE test accuracy + F1.
**README.md — append:** start a results table with a `resnet-FE` row (columns: clean-ID accuracy, macro-F1 — OOD + degraded columns come later).

**Loot:** a training engine + your first detector. **XP:** you built the machine, not just a model.

---

### Day 6 · ResNet fine-tuning (Run 2) + first OOD gap
**Mission:** unlock the backbone without destroying its knowledge; get your anchor generalization number.

**Files & functions today:** `src/metrics.py` (add OOD) · `notebooks/02_resnet.ipynb`

**Function — `src/metrics.py`:**
1. **`ood_report(model, ood_loader, device, class_names)`** · → `{ood_accuracy_on_true_class, prediction_distribution, fooled_rate}`. Algorithm: run over the OOD loader; GLIDE's truth is `diffusion`, so compute how often it's called diffusion, the full prediction distribution, and the fraction of OOD fakes classified as **real** (the "fooled" rate — your headline generalization number).

**The work:** rebuild via `build_resnet(..., "finetune")` from the best FE checkpoint. **Decide (values box):** single LR or **parameter groups** with a smaller backbone LR? Find the backbone/head ratio by the transfer-learning method. Set regularization; expect to tune. Re-run `fit`; then `evaluate` (test) + `ood_report` (OOD). Note the gap.
- *Encouragement:* first tricky training day. If it gets worse before better, that's information — read the curve, adjust, don't panic.

**Decisions to make:** single-LR vs parameter groups; backbone/head ratio (method); weight decay.

**Boss Check (→ LOG.md):** what failure are you preventing by keeping the backbone LR low relative to the head? Describe it from your curves (this is catastrophic forgetting — in your own words).

**LOG.md — write:** LR strategy + ratio method + choice; ResNet-FT test + OOD numbers; the ID→OOD gap; the jump over FE and your explanation.
**README.md — append:** `resnet-FT` row; add an **OOD accuracy** column and fill both ResNet rows.

**Loot:** your best ResNet + first generalization gap. **XP:** real fine-tuning done, and you've quantified generalization.

---

### Day 7 · Grad-CAM — detective mode
**Mission:** find *where* the model looks when it's wrong. Diagnosis, not decoration.

**Files & functions today:** `src/gradcam_utils.py` · `notebooks/02_resnet.ipynb`

**Functions — `src/gradcam_utils.py`:**
1. **`get_misclassified(model, loader, device, k_per_class)`** · → list of `{image, true, pred}`. Algorithm: run over a loader; keep wrong predictions, up to `k_per_class` per class; retain image tensors/paths.
2. **`run_gradcam(model, target_layer, samples, out_dir)`** · → saved overlays. Algorithm: attach Grad-CAM to `target_layer`; per sample compute the heatmap for the *predicted* class; overlay on original; save to `figures/gradcam/`. **You choose `target_layer` and justify it.**

**The work:** run on **misclassified** test images across all three classes. Save a handful.

**Decisions to make:** `k_per_class`; `target_layer` + why.

**Boss Check (→ LOG.md):** what does Grad-CAM actually compute? Name one failure mode you can *see* (background vs texture? GAN/diffusion confusion from shared high-freq noise?).

**LOG.md — write:** why that target layer; your failure-mode insight.
**README.md — append:** 2–3 Grad-CAM overlays with a caption on what they reveal.

**Loot:** "here's why it fails" images. **XP:** you can interrogate a model, not just score it.

---

### Day 8 · Buffer / ResNet consolidation
**Mission:** slack day. Absorb overrun; write **`viz.plot_confusion(cm, class_names, out_png)`** (→ labeled heatmap) if not done; finalize ResNet rows + confusion figure in the README.
**Boss Check (→ LOG.md):** which two classes confuse most, and your hypothesis why — tie to the FFT plot.
**Loot:** a fully documented ResNet. **XP:** Level 2 clear.

---

# LEVEL 3 — ViT (Days 9–12)
*Goal: the same rigor on ViT, the attention-vs-Grad-CAM comparison (an intellectual centerpiece), and ViT's OOD gap.*

---

### Day 9 · ViT-B/16 feature extraction (Run 1)
**Mission:** give the transformer the same fair shot ResNet got.

**Files & functions today:** `src/models.py` (add ViT) · `notebooks/03_vit.ipynb`

**Function — `src/models.py`:**
1. **`build_vit(num_classes, mode)`** · → model. Algorithm: create a pretrained ViT-Base/16 via `timm` for `num_classes`; locate the classifier head (timm's `head`); `feature_extraction` → freeze all but head; `finetune` → all trainable; return.

**The work:** FE run using your existing `fit`; same checkpoint/W&B discipline. Confirm ViT gets the **same normalization** it was pretrained with. `evaluate` (test) + `ood_report` (OOD).

**Decisions to make:** head regularization; training values (values box).

**Boss Check (→ LOG.md):** why must ViT get the same input normalization it was pretrained with? How does its fixed patch size set its sequence length?

**LOG.md — write:** ViT-FE test + OOD numbers.
**README.md — append:** `vit-FE` row.

**Loot:** a ViT baseline. **XP:** two model families, head-to-head.

---

### Day 10 · ViT-B/16 fine-tuning (Run 2) + settle the bet
**Mission:** fine-tune ViT and reason about why it differs from ResNet.

**Files & functions today:** `notebooks/03_vit.ipynb` (reuses `build_vit(..., "finetune")`, `fit`, `evaluate`, `ood_report`)

**The work:** rebuild in `finetune` from the best FE checkpoint. Decide backbone/head LR strategy again. Key reasoning: **does ViT need more regularization than ResNet at ~10k scale?** Decide from inductive-bias reasoning; sweep it up if the overfitting gap widens. `fit` → `evaluate` (test) → `ood_report` (OOD).

**Decisions to make:** LR strategy; regularization (expect higher than ResNet — justify).

**Boss Check (→ LOG.md):** state your regularization reasoning. Then **settle your original ResNet-vs-ViT bet** — who won in-distribution, does it match your prediction? And which generalized better to **OOD** — surprised?

**LOG.md — write:** ViT-FT test + OOD numbers; the resolved bet + explanation.
**README.md — append:** `vit-FT` row — you now have a full 4-model table (resnet-FE/FT, vit-FE/FT) with clean + OOD columns.

**Loot:** prediction resolved against reality. **XP:** theory connected to measurement.

---

### Day 11 · Attention maps vs Grad-CAM — the centerpiece
**Mission:** the most interesting *interpretability* comparison in the project.

**Files & functions today:** `src/gradcam_utils.py` (add rollout) · `notebooks/03_vit.ipynb`

**Function — `src/gradcam_utils.py`:**
1. **`run_attention_rollout(vit_model, samples, out_dir)`** · → saved attention overlays. Algorithm: compute attention rollout (combine attention across layers); overlay on original; save to `figures/attention/`. Run on the **exact same misclassified images** you used for ResNet Grad-CAM on Day 7.

**Boss Check (→ LOG.md):** on identical wrong predictions, what's the most striking difference between where ResNet looks (Grad-CAM) vs where ViT looks (attention)? Spend your real thinking here.

**LOG.md — write:** the comparison, a full paragraph.
**README.md — append:** side-by-side Grad-CAM vs attention figures with that paragraph as caption.

**Loot:** a "two architectures see differently" figure. **XP:** something that feels publishable.

---

### Day 12 · Buffer / ViT consolidation
**Mission:** slack day — finish ViT figures; confirm all four rows + both interpretability sections are in the README. **XP:** Level 3 clear — modeling done; the distinctive audit is next.

---

# LEVEL 4 — THE AUDIT (Days 13–15) · PROTECTED CORE
*Goal: the arms that make this project original — explain the OOD failure, then stress-test both trained models under real-world degradation, and chart reliability curves. These days do not get cut.*

---

### Day 13 · OOD why-analysis (not just the number)
**Mission:** explain *why* your detectors fail on GLIDE, turning a gap number into a research-grade finding.

**Files & functions today:** `notebooks/04_audit_ood_robustness.ipynb` (reuses `get_misclassified`, `run_gradcam`, `run_attention_rollout`, `compute_avg_spectrum`)

**The work:** on the **GLIDE (OOD)** images specifically: (1) pull misclassified OOD samples; (2) run Grad-CAM (ResNet) + attention rollout (ViT) on them; (3) compute the averaged FFT spectrum of GLIDE images and compare it to your training diffusion classes (SD1.4/ADM). Form a hypothesis: *is the model keying on a generator-specific frequency fingerprint that GLIDE lacks/differs on?*

**Decisions to make:** how many OOD samples to inspect (enough to see a pattern).

**Boss Check (→ LOG.md):** write the analysis paragraph. "Accuracy dropped X%" is a student result; "…*because* the model relied on a fingerprint absent in GLIDE, as the FFT + attention show" is a researcher's.

**LOG.md — write:** the hypothesis + the evidence (which figures support it).
**README.md — append:** an "Why it fails on unseen generators" subsection with the OOD FFT comparison + a Grad-CAM/attention example.

**Loot:** your headline finding, explained. **XP:** the paragraph that makes interviewers lean in.

---

### Day 14 · Robustness stress test — build it and sweep
**Mission:** measure how both detectors hold up under everyday image degradation. **No retraining** — you re-evaluate your existing best models.

**Files & functions today:** `src/robustness.py` · `notebooks/04_audit_ood_robustness.ipynb`

**Functions — `src/robustness.py`:**
1. **`build_degradations(severity_spec)`** · → a structure mapping `(degradation_name, severity)` → a transform callable. Algorithm: define one callable per degradation — **JPEG recompression** (re-encode at a quality factor, decode back), **rescale** (downscale to a fraction, upscale back to input size), **Gaussian blur** (a chosen radius/sigma), **screenshot-style resave** (mild rescale + moderate-quality JPEG combined) — each parameterized by a severity level you choose via the values box.
2. **`make_degraded_loader(manifest_csv, degradation_transform, batch_size)`** · → DataLoader. Algorithm: build a dataset whose transform = apply the degradation **first**, then the standard eval resize/normalize; wrap in a loader. On-the-fly — no disk copies.
3. **`run_robustness_sweep(model, manifest_csv, degradations, device)`** · → tidy results (per degradation × severity: accuracy + fooled-rate). Algorithm: for each `(degradation, severity)`, build a degraded loader, run `evaluate` (and `ood_report` if the manifest is the OOD one), record metrics; return.

**The work:** run `run_robustness_sweep` for **both** best models on **two** manifests — the **ID test set** and the **OOD (GLIDE) set** (the confirmed worst-case cell: "unseen generator *and* degraded"). That's 2 models × 2 manifests × several degradations × several severities — all evaluation, all background compute.

**Decisions to make:** which degradations to include (the four above are the default set), and the severity levels for each (values box — space them to reveal a trend).

**Boss Check (→ LOG.md):** before looking, predict which degradation hurts most and which model is more robust — then check. Frequency-artifact detectors often crumble under recompression; does your data agree?

**LOG.md — write:** your prediction, then the observed ranking of degradations by harm, per model.
**README.md — append:** hold the numbers for tomorrow's curve; note the setup (degradations, severities, worst-case cell).

**Loot:** a full robustness results structure. **XP:** you built an evaluation most detector projects never attempt.

---

### Day 15 · Reliability curves + the "where detectors break" view
**Mission:** turn the sweep into the visual and table that *are* your project's identity.

**Files & functions today:** `src/robustness.py` (add plot) · `src/viz.py` · `notebooks/04_audit_ood_robustness.ipynb`

**Function — `src/robustness.py`:**
1. **`plot_reliability_curves(sweep_results, out_png)`** · → saved figure. Algorithm: x-axis = severity, y-axis = accuracy (and a second panel for fooled-rate); one line per degradation; separate curves/panels per model; label clearly; save to `figures/reliability/`.

**The work:** build the **master audit table** — rows = each model; columns = **clean-ID / unseen-generator (OOD) / degraded-ID (worst degradation) / unseen+degraded (worst-case cell)**, with accuracy + fooled-rate. Plus the reliability-curve figure. This single table + figure is the thesis made visible. This day has built-in slack — if Day 14's sweep overran, finish it here.

**Boss Check (→ LOG.md):** in one paragraph — what's the single most important takeaway about when these detectors can and can't be trusted?

**LOG.md — write:** the takeaway paragraph (this becomes your README's headline claim + a resume bullet).
**README.md — append:** the master audit table + `reliability_curves.png` with a caption stating the headline finding.

**Loot:** the centerpiece of the whole project. **XP:** Level 4 clear — you have a genuine, defensible research contribution.

---

# LEVEL 5 — EVALUATION + DEPLOY (Days 16–18)
*Goal: one clean scoreboard, then a live, honestly-calibrated demo.*

---

### Day 16 · Evaluation harness + error analysis
**Mission:** consolidate everything into one comparison and show failures honestly.

**Files & functions today:** `notebooks/05_eval_and_deploy.ipynb` (reuses `evaluate`, `ood_report`, `run_robustness_sweep`, `viz.plot_confusion`, `count_parameters`)

**The work:** one clean script/notebook producing: the full model comparison (clean / OOD / degraded / worst-case × accuracy, per-class F1, **inference latency**, **parameter count**); confusion matrices for ResNet-FT vs ViT-FT side by side; a false-positive/negative gallery (a real called fake; a fake that passed as real, on both clean and degraded inputs); export a shareable **W&B report** URL of your training curves.

**Decisions to make:** which metrics headline the table; how to measure latency fairly (same batch size/device — a method choice).

**Boss Check (→ LOG.md):** given the full table, which model would you actually deploy and why — and under what conditions would you *not* trust either?

**LOG.md — write:** the deploy decision + its caveats.
**README.md — append:** the consolidated table, confusion matrices, error gallery, W&B link.

**Loot:** a complete scoreboard. **XP:** Level 5 begun; you reason about trade-offs, not just accuracy.

---

### Day 17 · Gradio app — local, calibrated, honest
**Mission:** build a demo that surfaces its own uncertainty instead of pretending to be an oracle.

**Files & functions today:** `app.py` · `notebooks/05_eval_and_deploy.ipynb`

**Functions — `app.py`:**
1. **`predict_with_confidence(model, image)`** · → `(class_name, confidence_scores, is_unsure)`. Algorithm: preprocess **exactly as in training**; forward; softmax; take top class + scores; if top confidence < a threshold **you choose**, set `is_unsure=True`.
2. **`build_interface(model)`** · → a Gradio interface (described in plain English, no code): image upload → `predict_with_confidence` → display predicted class, a confidence bar, a Grad-CAM overlay, an **"unsure" flag** when confidence is low, and a fixed **disclaimer banner** carrying your measured OOD + degradation reliability numbers.

**Decisions to make:** the confidence threshold for "unsure" (method: pick it from your validation confidence distribution — e.g., where correct and incorrect predictions separate — not a guessed number).

**Boss Check (→ LOG.md):** why must inference preprocessing exactly match training? And why is an "unsure" flag more honest than always forcing a 3-way guess?

**LOG.md — write:** your threshold method + value; how the demo communicates uncertainty.
**README.md — append:** a "Demo" stub (screenshot placeholder) noting the calibration + disclaimer.

**Loot:** a working, honest local demo. **XP:** the "engineering + integrity" signal recruiters notice.

---

### Day 18 · Deploy to Hugging Face Spaces
**Mission:** ship a public, one-click demo with its limits stated up front.

**Files & functions today:** `app.py` (finalize) + a `requirements.txt`

**The work:** push to HF Spaces (free, public URL). Ensure the disclaimer banner states, in plain numbers, that reliability drops on unseen generators and under recompression (cite your own audit figures). Verify it runs from a cold load with no setup.

**Boss Check (→ LOG.md):** why would omitting the reliability disclaimer be dishonest given your own results? (This restraint reads as maturity to a technical reviewer.)

**LOG.md — write:** the live URL; final deployment gotchas.
**README.md — append:** the live demo link + a one-line honest summary of what it can and can't do.

**Loot:** a live link for your resume and LinkedIn. **XP:** Level 5 clear — you deployed, which most students never do.

---

# LEVEL 6 — THE WRITEUP (Days 19–20)
*Goal: turn 18 days of work into a research-grade README and resume weapon.*

---

### Day 19 · README + literature framing + resume bullets
**Mission:** frame the work like a researcher now that you have results to stand on.

**The work:**
- **Read the two key papers now (with your numbers in hand):** skim **CNNDet (Wang 2020)** and **UnivFD (Ojha 2023)** — abstracts, intros, results only. Reading them here (not earlier) makes positioning concrete. Write three sentences: what's been tried, where it fails, how your **robustness-audit** framing extends it (you add the *degradation* failure axis and a calibrated demo, not just unseen-generator generalization).
- **Assemble the README** (mostly paste from `LOG.md`): reframed problem + 2026 context; dataset + OOD design; methods (ResNet vs ViT, two runs each); the **master audit table + reliability curves** as the headline; OOD why-analysis; Grad-CAM vs attention; honest limitations; live demo; future work.
- **Draft 2–3 resume bullets** leading with *audited / quantified / generalization / robustness*, each with a number and the live link.

**Boss Check (→ LOG.md):** state your project's single-sentence contribution — the thing no generic "AI image detector" project claims.

**LOG.md — write:** that one-sentence contribution + your final resume bullets.
**README.md — finalize:** the full document.

**Loot:** a research-grade README + resume bullets. **XP:** the project now has a story, not just code.

---

### Day 20 · Buffer / repo polish / rehearsal
**Mission:** land the plane.
- Code cleanup; seed everything; `requirements.txt`; sensible structure; `.gitignore` verified; final read-through.
- **Rehearse explaining the whole project out loud from `LOG.md`, end to end** — your interview dry run. If you can narrate every decision (why these generators, why the backbone LR ratio, why ViT needed more regularization, why recompression breaks detection) without notes, nothing is a black box.

**Loot:** a project you can defend cold, with a live demo and a genuine finding. **XP:** all levels clear — a solo, end-to-end reliability audit that reads like research and shares zero DNA with your autism coursework project.

---

## Carry-over pointer reference (surfaced above on their day)
- **Kaggle:** GPU + Internet on; manifests store `/kaggle/input` paths (never copy images); checkpoint each epoch to `/kaggle/working` + Save Version; GPU quota is a weekly budget.
- **Reproducibility:** `set_global_seed` at the top of every notebook; sort before seeded sampling; commit manifest CSVs.
- **Leakage:** `assert_no_leakage` prints clean before any training; GLIDE never enters train/val.
- **Robustness = evaluation, not training:** degradations apply on-the-fly to already-trained models — no new runs, which is why the audit is cheap.
- **Honesty:** never report a clean accuracy without its OOD and degraded numbers beside it — that pairing *is* your thesis.
- **`%%writefile`:** the bridge between your Kaggle notebook and your GitHub `.py` files — keep them identical.

## Next check-in
Finish **Day 2** (the manifests) and paste me your per-class / per-generator counts + your `class_spec`; I'll stress-test it for balance and leakage before you build loaders on Day 3. I'll poke holes in the reasoning — I won't hand you values or code.
