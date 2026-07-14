# AI-Generated Image Forensics — The Build Guide
### Real / GAN / Diffusion · CNN vs ViT · cross-generator generalization
*2 hrs/day · picks up at ViT (your next step) · ~31 days with buffers · your first project of this scale — and that's completely fine*

---

## Read this once, then never feel overwhelmed again

- **Read ONLY today's box.** Do not read ahead. The whole file at once looks like a mountain; a single day is a calm two hours. This is deliberate — trust it.
- **Falling behind is planned for.** Buffer days exist precisely so a slow day costs you nothing. Slipping is not failure; it's what the buffers are for.
- **No code appears anywhere in here, on purpose.** You get the *what* and the *why* in plain English; you figure out the *how*. That struggle is the learning. (If you ever need attention/transformers again, your Yannic Kilcher + Illustrated Transformer notes are your reference — you've already cleared that.)
- **`LOG.md` is your save file.** Every day, write: (1) what you built, (2) every real choice + why, (3) one thing that confused you + how you beat it. A value you set but can't justify in the log is an open bug. This log is also your interview script and half your README.
- **Training runs while you rest.** A "2-hour day" is often ~40 min hands-on + a run training in the background. Attention time ≠ wall-clock time.

**Gamification legend:** each Phase is a **Level**. Each day has a **Mission**, a **Build**, and a **Boss Check** (your Prove-it questions → logged). **Loot** = the artifact you walk away with. **Pointer unlocked** = a pro tip surfaced exactly when it becomes relevant. **XP** = understanding banked. Keep a **streak** in your log.

**The one rule I'll never break:** I name what to decide; I never hand you the value. Learning rates, dropout, weight decay, epochs, batch size, layer widths, schedules — all yours to choose and justify. Architecture *definitions* (what a residual block is, what shapes ViT uses) I'll describe, because that's "what the thing is," not a knob you tune.

---

## The map (your levels)

- **Level 1 — Foundations:** finish ViT, get Kaggle + tooling battle-ready *(Days 1–2)*
- **Level 2 — Data:** choose your dataset, build the pipeline, read the frequency domain *(Days 3–6)*
- **Level 3 — Baseline:** your own CNN from scratch — first real results *(Days 7–9)*
- **Level 4 — ResNet:** transfer learning + Grad-CAM detective work *(Days 10–13)*
- **Level 5 — ViT:** the rematch + attention maps + settle your bet *(Days 14–17)*
- **Level 6 — Generalization:** the final-boss experiment *(Days 18–20)*
- **Level 7 — Evaluation:** one clean scoreboard *(Days 21–22)*
- **Level 8 — Deployment:** ship a live demo *(Days 23–25)*
- **Level 9 — The Writeup:** turn it into a resume weapon *(Days 26–31)*

---

## Dataset options (you commit on Day 3 — skim now, decide later)

Your framing is **Real / GAN / Diffusion** because GAN and diffusion leave *structurally different* fingerprints — the split is only meaningful if GAN images are actually present.

**1. GenImage — best fit.** GAN-era (ProGAN, BigGAN, StyleGAN) *and* diffusion generators + real images.
- *Pros:* clean, defensible 3-class split; recognized benchmark; per-generator labels make Level 6 clean.
- *Cons:* large (you'll subsample); reals are ImageNet-domain; download logistics.

**2. Defactify / MS-COCO-AI.** Modern generators only (SD3, SDXL, DALL·E 3, MJ v6) — **no GAN.**
- *Pros:* current generators; diverse natural reals.
- *Cons:* breaks your 3-class framing — you'd fall back to Real/Diffusion/Autoregressive (subtler) or binary.

**3. CIFAKE.** 60k real + 60k SD fakes.
- *Pros:* tiny, fast — perfect as the Day-2 sanity-check dataset.
- *Cons:* 32×32 (too small for 224-input models), binary, diffusion-only, overused. Not your main set.

**4. ArtiFact.** Large, many generators, multi-source.
- *Pros:* diversity, great for generalization.
- *Cons:* messy, heavy preprocessing, higher source-leakage risk.

**5. Combination (Defactify reals + separate GAN source).**
- *Pros:* modern + GAN coverage.
- *Cons:* **source-leakage trap** — if reals and fakes come from different origins, the model can cheat by learning the *source* (compression, resolution) instead of AI artifacts, and it collapses in the real world.

**Recommendation:** GenImage for the project, CIFAKE only for the Day-2 sanity test.

---

# LEVEL 1 — FOUNDATIONS

### Day 1 · Fri Jul 10 · Vision Transformer + patch embedding
**Mission:** finish the last theory brick — understand how an image becomes a sequence a transformer can read.

**Hour 1 — Read.** *An Image is Worth 16×16 Words* (ViT paper), Sections 1–3 only, then the HuggingFace ViT explainer.
- *Question to hold the whole time:* a CNN has two gifts baked in — *locality* (nearby pixels relate) and *translation equivariance* (a feature on the left is recognized on the right). ViT is handed neither. Find the term **inductive bias** in the paper and write, in your own words, what ViT gives up and what it gains by giving it up.
- *Second thing to hunt for:* patches fed to a transformer have no inherent order. How does the paper stop the model from treating a shuffled image as identical? (Look for positional embeddings; understand *why* they're needed, not just that they exist.)

**Hour 2 — Build (no code, plain steps).** Turn a random `(1, 3, 224, 224)` image into a sequence of shape `(1, 196, 768)`.
- Extract every non-overlapping 16×16 patch using `torch.Tensor.unfold` called twice — once per spatial dimension, size 16, step 16. **Check the shape after this step and say aloud what each dimension represents** before moving on.
- Reshape so each patch becomes a flat vector of length 768 (= 16×16×3), landing at 196 patches (a 14×14 grid). You'll need `permute`, `contiguous`, and `view` in *some* order — work out the order by reasoning about the shape you have vs. the shape you need, checking after every single operation.
- Pass through one linear projection layer; confirm the final shape.
- *You will hit shape errors.* Read them, reason from first principles about what's mismatched, fix deliberately — no random trial-and-error. That debugging *is* the exercise.

**Boss Check (→ LOG.md):**
1. If you shuffled the 196 patch tokens before feeding them in, what happens to a trained ViT's prediction, and why?
2. **Place your bet (this is a real wager you'll settle in Level 5):** with only ~10k training images, do you expect ResNet or ViT to win — and *exactly* why? Commit it in writing now.

**Loot:** a working patch-embedding layer + a locked prediction. **XP:** you now understand the single idea that makes transformers "see."

---

### Day 2 · Sat Jul 11 · Kaggle + tooling, battle-ready
**Mission:** make your workshop bulletproof *before* you're under pressure mid-training.

**Hour 1 — Kaggle setup (you're using Kaggle, so learn its rules now).**
- Create a Kaggle notebook. In settings, switch the **accelerator to GPU** (T4×2 or P100) and toggle **Internet on** (needed for pip installs and W&B).
- Learn the geography: `/kaggle/input` is **read-only** (where attached datasets live); `/kaggle/working` is your **writable** output folder. Anything you want to keep must be there.
- Understand the timers: interactive sessions expire, and an **idle tab disconnects** after a short spell — so long training must **checkpoint** and you must **"Save Version" (commit)** to persist outputs. Your GPU quota is limited per week — treat it as a budget, don't burn it babysitting.
- **Pointer unlocked (compute):** plan every future training run to **checkpoint each epoch into `/kaggle/working`**, then Save Version at the end. A disconnect should never cost you more than one epoch.

**Hour 2 — The rest of the kit + a real sanity test.**
- Install/verify `timm`, HuggingFace `datasets`, `matplotlib`, `seaborn`, `scikit-learn`, `wandb`, `gradio`, `pytorch-grad-cam`. Confirm a tensor operation runs on CUDA.
- Create a free **Weights & Biases** account; run their tiny quickstart so you *see* a metric appear in the dashboard.
- Create your **GitHub repo**: sensible folders, a README placeholder, and a `.gitignore` that excludes data and model weights.
- **Prove the whole loop works end-to-end**: train for a *single epoch* on CIFAKE or CIFAR-10. You're not chasing accuracy — you're proving data → model → loss → W&B log all connect before real data arrives.

**Boss Check (→ LOG.md):** in one line each — where do checkpoints go on Kaggle and why there; what does "Save Version" protect you from; did your sanity loop log to W&B?

**Loot:** a repo, a live W&B dashboard, and a proven training loop. **XP:** Level 1 complete — foundations done.

---

# LEVEL 2 — DATA

### Day 3 · Sun Jul 12 · Choose your dataset (own the decision)
**Mission:** make a deliberate, defensible dataset choice — this shapes everything downstream.

**Hour 1 — Inspect before you commit.** Open your top candidate(s) and load a *handful* of real samples. In a notebook markdown cell, answer explicitly: which generators are actually present? Which are GAN, which diffusion, which neither? Is a clean Real/GAN/Diffusion split truly possible here, or must the class structure change?

**Hour 2 — Decide and defend.** Pick from the five options above. Write **one paragraph** justifying the choice and naming the single biggest risk you're accepting.
- **Pointer unlocked (leakage #1):** if you ever combine sources, beware **source leakage** — the model scoring high by learning "which dataset an image came from" instead of AI artifacts. Note in your log how your choice avoids or controls this.
- Lock the label mapping as an **explicit dictionary in code**, never implicit folder names. Record counts per class and per generator.

**Boss Check (→ LOG.md):** which generators are GAN vs diffusion vs autoregressive? Why is your split scientifically meaningful? What's your biggest risk and your control for it?

**Loot:** a committed dataset + a written justification (this becomes a README paragraph). **XP:** you're now making research decisions, not following a tutorial.

---

### Day 4 · Mon Jul 13 · Pipeline part 1 — the splits
**Mission:** carve the data correctly so your results are trustworthy, not accidentally inflated.

**Hour 1 — Scale + inspect.** Decide your dataset size and per-class count (justify the compute-vs-signal trade-off in your log). Print the raw class distribution. Look at a few images per class with your own eyes — what visibly differs between real, GAN, and diffusion?

**Hour 2 — Split with discipline.** Decide your train/val/test proportions and justify them. Make the split **stratified** (so class ratios hold across splits) with a **fixed seed**, and save a **split manifest** (the exact file lists) so the split is reproducible forever.
- **Pointer unlocked (leakage #2 + reproducibility):** a source image or near-duplicate landing in two splits secretly inflates your score. And "I re-ran and got a different number" must never happen — seed everything, save the manifest.

**Boss Check (→ LOG.md):** why stratified? why a fixed seed? why split *before* any augmentation? what exact failure does a duplicate-across-splits cause?

**Loot:** a reproducible, honest split. **XP:** you now know why most beginner accuracy numbers are lies.

---

### Day 5 · Tue Jul 14 · Pipeline part 2 — loaders, normalization, augmentation
**Mission:** feed the model correctly — and dodge the trap that silently kills this specific project.

**Hour 1 — Dataset + DataLoader + normalization.** Build a PyTorch `Dataset` and `DataLoader`. Figure out **what input normalization your pretrained models expect** and match it exactly — then write down *why* mismatching it quietly wrecks predictions.

**Hour 2 — Augmentation, with real reasoning.** Choose your **train-only** augmentations. Reason hard before applying anything: **could a given augmentation erase the very high-frequency artifact that distinguishes AI images from real ones?** Decide accordingly, and apply nothing to val/test. Wire `wandb` logging into your loop template now so every future run is tracked automatically.
- **Pointer unlocked (W&B hygiene):** name runs meaningfully (so future-you reading the README knows which was which) and tag them by level.

**Boss Check (→ LOG.md):** why augment train but not val/test? Justify every augmentation you *kept* and every one you *rejected*, in terms of the artifacts you must preserve.

**Loot:** a clean, tracked data pipeline. **XP:** you understood a domain-specific trap most people walk straight into.

---

### Day 6 · Wed Jul 15 · Frequency domain (look first, explain second) + buffer
**Mission:** see what the human eye can't — and form your *own* interpretation before anyone tells you the answer.

**Hour 1 — Compute.** Take a sample of images per class (you pick how many, justify it). Compute the 2D FFT (`np.fft.fft2`), take the magnitude spectrum, center it (`np.fft.fftshift`), and average per class. Plot the three averaged spectra side by side.

**Hour 2 — Interpret, then verify.** *Look at your plots and write what you see, in your own words, before reading anything.* Only after you've committed an interpretation, sanity-check it against what's known about generator artifacts. The goal is analysis, not confirmation — see it first.
- If Hour 1 pipeline work ran long, this is also your **buffer** — the FFT can slide into tomorrow guilt-free.

**Boss Check (→ LOG.md):** in your own words, what does each class's spectrum look like and what might explain the differences? Be honest about how visible (or invisible) the diffusion signature is.

**Loot:** a striking FFT figure for your README. **XP:** Level 2 complete — you have data you trust and an insight most projects lack.

---

# LEVEL 3 — BASELINE

### Day 7 · Thu Jul 16 · Design your own CNN + the training loop
**Mission:** build a network that is *yours* — every choice defensible.

**Hour 1 — Architect it.** Design a small CNN from scratch. **You decide:** how deep (how many conv blocks), the channel progression, where normalization / pooling / regularization sit, how you pool spatially before the head, and the head's shape for 3 classes. The *pattern* of a block (convolution → normalization → activation → downsample) is a convention you can lean on; the *numbers* are yours.

**Hour 2 — Write the loop (plain steps).** Forward pass → loss → backward → optimizer step; a validation pass each epoch; W&B logging of train/val loss and val accuracy; and an early-stopping rule you define. Decide how you'll handle class balance (there's more than one way) and justify the pick.

**Boss Check (→ LOG.md):** justify every structural choice. If it *underfits*, what's the first thing you change? If it *overfits*?

**Loot:** your own architecture, ready to train. **XP:** you designed, not copied.

---

### Day 8 · Fri Jul 17 · Train the baseline + debug like an engineer
**Mission:** get your first real results — and read them, don't just collect them. *(Your internship wraps today; more breathing room ahead.)*

**Hour 1 — Predict, then train.** *Before* looking at accuracy, compute your **random-chance floor** yourself and write down the number that would convince you "the pipeline actually works." Then launch training and watch the W&B curves live.

**Hour 2 — Debug from first principles.** *Debugging heuristic (pointer unlocked):* if you land near random chance, suspect the **data pipeline first** — normalization, label mapping, DataLoader shuffle — before blaming the model. *Timeboxing (pointer unlocked):* stuck more than ~45 minutes on one bug? Log the symptom, take a buffer, move on — don't let one bug eat the level.

**Boss Check (→ LOG.md):** from the curves *alone* — underfitting or overfitting? State your evidence, not just the verdict.

**Loot:** your first honest results. **XP:** you can now read a loss curve, which most students can't.

---

### Day 9 · Sat Jul 18 · Buffer / baseline polish
**Mission:** breathe, finish, consolidate. This is a light day by design.
- Absorb any overrun. Save the checkpoint, confusion matrix, and per-class F1.
- **Boss Check (→ LOG.md):** which two classes confuse most, and your hypothesis *why* — tie it back to your FFT plot.
- **Loot:** a finished baseline. **XP:** Level 3 clear. You've shipped a working model from scratch — that alone beats a lot of resumes.

---

# LEVEL 4 — RESNET

### Day 10 · Sun Jul 19 · ResNet-50, Run 1 — feature extraction
**Mission:** stand on ImageNet's shoulders.

**Hour 1 — Set it up (plain steps).** Load pretrained ResNet-50; **freeze the backbone**; find the backbone's output feature dimension *yourself* and attach a fresh classifier head sized for 3 classes (decide whether/how to regularize it).

**Hour 2 — Train the head only.** *You choose* optimizer, learning rate, whether to schedule it, epoch budget, batch size. Checkpoint each epoch into `/kaggle/working`; Save Version at the end.

**Boss Check (→ LOG.md):** why does feature extraction come *before* fine-tuning? Why freeze the backbone at all? What made you pick your learning rate, and how would the curves reveal it's too high or too low?

**Loot:** a strong baseline-beating model. **XP:** you understand what "pretrained" actually buys you.

---

### Day 11 · Mon Jul 20 · ResNet-50, Run 2 — fine-tuning
**Mission:** unlock the backbone without destroying what it already knows.

**Build (plain steps).** Start from your best feature-extraction checkpoint; unfreeze the backbone. Reason it through: should the backbone and head train at the *same* learning rate or different ones? Implement your decision (research "parameter groups" if you go the different-rates route). Decide regularization and your stopping rule. Log everything.
- *Encouragement:* this is the first "tricky" training day. If it gets worse before it gets better, that's information, not failure — read the curves and adjust.

**Boss Check (→ LOG.md):** what specific failure are you guarding against by how you set the backbone's rate relative to the head's? How would that failure show up in the metrics? *(This is catastrophic forgetting — but define it in your own words from what you observe.)*

**Loot:** your best ResNet yet. **XP:** you've done real fine-tuning, the skill most "ML projects" skip.

---

### Day 12 · Tue Jul 21 · Grad-CAM — detective mode
**Mission:** find out *where* the model looks when it's wrong. This is diagnosis, not decoration.

**Build (plain steps).** Run Grad-CAM on your **misclassified** test images (not the easy correct ones). Choose the target layer and be ready to justify that choice. Save a handful of overlays across all three classes — all wrong predictions.

**Boss Check (→ LOG.md):** what does Grad-CAM actually compute? Name one concrete failure mode you can *see* (e.g., is it staring at the background instead of texture? confusing GAN and diffusion where both have similar high-frequency noise?).

**Loot:** a set of "here's why it fails" images — pure gold for your writeup. **XP:** you can interrogate a model, not just score it.

---

### Day 13 · Wed Jul 22 · Buffer / ResNet catch-up
**Mission:** slack day — absorb overrun, start the results table with your ResNet rows. Light. **XP:** Level 4 clear.

---

# LEVEL 5 — VIT (THE REMATCH)

### Day 14 · Thu Jul 23 · ViT-Base/16, Run 1 — feature extraction
**Mission:** give the transformer the same fair shot you gave ResNet.

**Build (plain steps).** Load a pretrained ViT-Base/16; freeze everything except the classifier head (locate where that head lives in the model); train the head. Same logging and checkpoint discipline. You set the training values.

**Boss Check (→ LOG.md):** why must ViT get the *same* input normalization it was pretrained with? How does its fixed patch size determine its sequence length?

**Loot:** a ViT feature-extraction baseline. **XP:** you're running two model families head-to-head — a genuinely strong comparison.

---

### Day 15 · Fri Jul 24 · ViT-Base/16, Run 2 — fine-tuning *(classes start — keep it light)*
**Mission:** fine-tune ViT, and reason about why it behaves differently from ResNet.

**Build (plain steps).** Unfreeze; decide your backbone-vs-head learning-rate strategy again. Then reason specifically: does ViT need a *different amount of regularization* than ResNet did on this same-size dataset? Decide from what you know about inductive biases, and justify. Log.

**Boss Check (→ LOG.md):** state your regularization reasoning explicitly. Then **settle your Day-1 bet** — did ResNet or ViT win in-distribution, and does the result match your prediction? Score yourself honestly.

**Loot:** your bet, resolved. **XP:** you connected a theory prediction to a measured result — that's real scientific thinking.

---

### Day 16 · Sat Jul 25 · Attention maps vs Grad-CAM — the centerpiece
**Mission:** the single most interesting comparison in the whole project.

**Build (plain steps).** Run attention rollout on the **exact same misclassified images** you used for Grad-CAM on Day 12.

**Boss Check (→ LOG.md):** on identical wrong predictions, what's the most striking difference between *where ResNet looks* and *where ViT looks*? Spend your real thinking time here — this paragraph is the intellectual heart of your writeup.

**Loot:** a side-by-side "how two architectures see differently" figure. **XP:** you have something genuinely publishable-feeling.

---

### Day 17 · Sun Jul 26 · Buffer / ViT catch-up
**Mission:** slack day — absorb overrun, add ViT rows to the results table. **XP:** Level 5 clear.

---

# LEVEL 6 — GENERALIZATION (FINAL BOSS)

### Day 18 · Mon Jul 27 · Design + run the hold-out experiment
**Mission:** test whether your detector *actually* generalizes — the question that separates a real finding from a lucky number.

**Build (plain steps).** Pick the generator your best model handled worst (from the confusion matrices). Re-split so **all** of that generator's images live only in a held-out test set — never train or val. Evaluate your best model on it (decide whether to retrain or evaluate directly, and justify). Report in-distribution vs held-out accuracy and the gap.

**Boss Check (→ LOG.md):** why does holding out an entire *generator* test generalization in a way a random split never could? Why might a large gap be expected?

**Loot:** the gap number — the honest core of your project. **XP:** you designed a real generalization experiment.

---

### Day 19 · Tue Jul 28 · Why the gap? (the analysis that makes it research-grade)
**Mission:** explain the failure, don't just measure it.

**Build (plain steps).** Run Grad-CAM + FFT on the held-out generator's images. Form a hypothesis for *why* the model stumbles on them.

**Boss Check (→ LOG.md):** write the analysis paragraph. "Accuracy dropped 30%" is a student result; "accuracy dropped 30% *because* the model keyed on a generator-specific fingerprint absent here" is a researcher's result. Aim for the second.

**Loot:** your headline finding, fully explained. **XP:** this is the paragraph that makes interviewers lean in.

---

### Day 20 · Wed Jul 29 · Buffer
**Mission:** slack day. Breathe. **XP:** Level 6 clear — the hard part is behind you.

---

# LEVEL 7 — EVALUATION

### Day 21 · Thu Jul 30 · One clean scoreboard
**Mission:** consolidate five models into a single honest comparison.

**Build (plain steps).** One notebook/script: a table across all five models (baseline, ResNet-FE, ResNet-FT, ViT-FE, ViT-FT) and the metrics you decide matter (accuracy, per-class F1, inference latency, parameter count); confusion matrices for your two best models side by side.

**Boss Check (→ LOG.md):** explain the latency-vs-accuracy trade-off. Which model would you actually deploy, and why?

**Loot:** the results table your README is built around. **XP:** you can reason about deployment trade-offs, not just accuracy.

---

### Day 22 · Fri Jul 31 · Error analysis + shareable report
**Mission:** show the failures honestly and package the curves.

**Build (plain steps).** Show false positives/negatives with the actual images (a real classified as fake; a fake that passed as real). Export a **shareable W&B report** of your training curves — that URL goes straight into your README.
- **Pointer unlocked (honesty):** never present a headline accuracy without the generalization gap beside it. Measured honesty beats overclaiming with any real reviewer.

**Boss Check (→ LOG.md):** which error type is most costly for a real-world detector, and why?

**Loot:** an error-analysis section + a public W&B link. **XP:** Level 7 clear.

---

# LEVEL 8 — DEPLOYMENT

### Day 23 · Sat Aug 1 · Gradio app (local first)
**Mission:** make your model something anyone can click.

**Build (plain steps).** A Gradio app: upload image → best model → predicted class + confidence + Grad-CAM overlay. Run it locally first.

**Boss Check (→ LOG.md):** why must inference preprocessing *exactly* match training preprocessing, or predictions silently break?

**Loot:** a working local demo. **XP:** the "SDE layer" recruiters react to.

---

### Day 24 · Sun Aug 2 · Deploy to Hugging Face Spaces + the honest disclaimer
**Mission:** ship a public, one-click demo — and tell the truth about its limits.

**Build (plain steps).** Push to HF Spaces (free, public URL). Add a **visible disclaimer**: this detector is brittle on generators it hasn't seen, is not forensic proof, and its reliability is bounded by the Level-6 gap you measured.

**Boss Check (→ LOG.md):** why would omitting that disclaimer be dishonest given your own results? (That restraint is itself a signal of maturity to a technical reviewer.)

**Loot:** a live link for your resume and LinkedIn. **XP:** you've deployed a model — most students never do.

---

### Day 25 · Mon Aug 3 · Buffer / deployment fixes
**Mission:** slack day — smooth out Spaces quirks. **XP:** Level 8 clear.

---

# LEVEL 9 — THE WRITEUP

### Day 26 · Tue Aug 4 · README part 1 + situate against the literature
**Mission:** frame the work like a researcher — *now* that you have results to stand on.

**Hour 1 — Read the two key papers (moved here on purpose).** Skim **CNNDet (Wang 2020)** and **UnivFD (Ojha 2023)** — abstracts, intros, results tables only. Reading them *now*, with your own numbers in hand, makes positioning concrete instead of abstract.
- Write **three sentences**: what's already been tried, where it fails, and how your Real/GAN/Diffusion framing + generalization experiment sits relative to it. This is your README intro and it will read like a researcher wrote it.

**Hour 2 — README part 1.** Problem + 2026 context; dataset (class definitions, why the split is meaningful, your FFT figure); what ResNet vs ViT is each built to capture.

**Loot:** a research-grade intro. **XP:** your project now has a story, not just code.

---

### Day 27 · Wed Aug 5 · README part 2
**Mission:** assemble the evidence (most of it already written in `LOG.md`).
- Results table + W&B link; Grad-CAM vs attention comparison on shared failures; the generalization experiment with its *why*-analysis; honest limitations; live demo link; future work (CLIP-based detection, newer generators).
- **Pointer unlocked (write-as-you-go pays off):** because you logged daily, this is assembly, not authorship. Paste, tighten, done.

**Loot:** a complete README. **XP:** the project is now legible to a stranger in 60 seconds.

---

### Day 28 · Thu Aug 6 · Repo polish + resume bullets
**Mission:** make the repo something you're proud to link.
- Code cleanup; seed everything; `requirements.txt`; sensible structure; `.gitignore` for data/checkpoints; final read-through.
- Draft your **2–3 resume bullets** for this project — decisions + metrics + the live link.
- **Pointer unlocked (git cadence):** you've been committing daily with one-line messages and never committing data/checkpoints — so this is a tidy-up, not a rescue.

**Loot:** a portfolio-ready repo + resume bullets. **XP:** almost there.

---

### Days 29–31 · Aug 7–9 · Final buffer + out-loud rehearsal
**Mission:** land the plane.
- Overflow room for anything unfinished.
- **Rehearse explaining the whole project out loud from `LOG.md`, end to end** — this is your interview dry run. If you can narrate every decision without notes, nothing is a black box.
- If you're ahead, an optional stretch goal (a second held-out generator, or a CLIP-feature baseline) lives here.

**Loot:** a project you can defend cold. **XP:** all levels clear. You built something real, at scale, for the first time — and you can prove every inch of it.

---

## Master pointer reference (all surfaced on their day above — here so nothing's lost)

- **Compute (Kaggle):** GPU + Internet on in settings; `/kaggle/input` read-only, `/kaggle/working` writable; idle tabs disconnect; **checkpoint every epoch** and **Save Version** to persist; GPU quota is a weekly budget.
- **Reproducibility:** seed Python/NumPy/PyTorch, deterministic flags, save the split manifest.
- **Leakage, two kinds:** duplicates across splits; source leakage across classes. Both fake your accuracy.
- **Debugging:** near random chance → suspect the pipeline first; timebox any single bug to ~45 min, then buffer it.
- **W&B hygiene:** meaningful run names, tag by level.
- **Git:** commit daily, one-liner messages; never commit data/checkpoints.
- **README as you go:** paste each figure the day you make it.
- **Honesty:** never a headline accuracy without the generalization gap beside it.
- **DSA block runs in parallel, untouched.**
- **Scope safety:** ViT fine-tuning is the riskiest stretch (buffers sit right after Levels 4 and 5). If time runs short late, cut *stretch goals* — never the generalization experiment or the writeup.

---

## Next check-in
When you finish **Day 3** and pick a dataset, come back and I'll try to *break* your label-mapping and split reasoning before you spend a minute of GPU on it. I won't hand you values — I'll poke holes in your thinking. That's the whole game.
