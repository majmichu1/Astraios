# Smart Processor — Object-Aware Roadmap

Living status doc for the "truly smart" object-aware processing effort. Goal: the
Smart Processor should **know the subject** (what it is, where it is, its shape)
and **edit it accordingly** — stretch, background, deconvolution, and contrast
all steered by the catalog + plate solve, not whole-image heuristics.

**Branch:** `smart-object-aware` (forked from `ui-redesign`). Each item is gated
and graceful — if the object mask / WCS isn't available, the pipeline falls back
to whole-image behaviour. `git checkout ui-redesign` reverts the whole effort.

---

## Architecture recap

- **Identify the target:** plate solve (WCS → catalog region query) or the
  user-typed name (`catalog.lookup`). Plate solving uses nova.astrometry.net
  online when an API key is set (Preferences) — the keystone that unlocks
  position-aware processing.
- **Catalog** (`cosmica/resources/catalog.json`, 139 hand-curated targets):
  per-target metadata + `processing_hints` consumed by `_build_plan`.
- **Object mask** (`cosmica/core/object_mask.py`): soft elliptical [0,1] mask of
  the subject, positioned via WCS or centred fallback, carried on
  `ProcessingPlan.object_mask`. Steers enhancement toward the subject.

---

## Status

### ✅ Done (committed on branch)

| Item | What | Commit |
|------|------|--------|
| Plate solving keystone | Smart Processor uses astrometry.net online solver with the Preferences API key | `ff494dc` |
| #1 object mask — module | `build_object_mask()` soft elliptical mask + tests | `786835b` |
| #1 object mask — plan | `_build_object_mask()` builds & carries it on the plan (WCS or centred) | `078770c` |
| #1b consumer — local contrast | HDR + star-aware CLAHE confined to the subject; sky grain left alone | `351ff48` |
| #1c consumer — deconvolution | deconvolve the subject, leave empty sky un-sharpened | `279e9b0` |
| #1d consumer — stretch | framed subject judged by its own region, not the noisy >0.02 heuristic | `050f473` |
| #2 richer recipes — hints | every catalog `processing_hint` now consumed (`bg_sensitive`, `ha_dominant`, `reflection_nebulosity` were dead) | `74ae583`, `3e81318` |

| #2b SIMBAD fallback | unknown target names resolved via SIMBAD (HTTP) → TargetInfo + recipe; verified live (M51) | `f716d37` |

**#1 (spatial object masks) is COMPLETE** — the mask steers local contrast,
deconvolution, and the stretch, all gated + graceful.
**#2 (richer recipes) is COMPLETE** — all hints consumed; any object identifiable
(catalog → SIMBAD).

The core of the vision is delivered: the processor now **knows what** the subject
is (catalog + SIMBAD), **where** it is (plate solve + object mask), and **how to
edit it** (per-type recipes + object-aware background/deconv/stretch/contrast).

| #3 reference-image prior | DSS2 cutout via hips2fits → shape-accurate object mask (preferred over the ellipse when plate-solved); verified live (M51) | `79e199c`, `be0371f` |

**#3 (reference-image prior) DONE for the mask use-case.** When a plate solve
succeeds, the object mask is now the object's REAL DSS2 outline (not an ellipse),
steering all the object-aware stages. Needs the astrometry key + network; falls
back to the ellipse otherwise. (A *tonal* prior from the reference is a possible
future extension, not built.)

### 🔜 Next

- [ ] **#4 — learning (LAST)**: train a small model on (raw stack → expert result)
      pairs + object metadata to predict processing parameters. Needs a dataset —
      the big open piece. Until then the rule-based + reference-image system stands.
- [ ] **#4 — learning (LAST)**: train a small model on (raw stack → expert result)
      pairs + object metadata to predict processing parameters. Needs a dataset.
- [ ] **#3 — reference-image prior**: fetch a DSS2/PanSTARRS cutout of the target
      as a "what it should look like" tonal/structure guide.
- [ ] **#4 — learning (LAST)**: train a small model on (raw stack → expert result)
      pairs + object metadata to predict processing parameters.

### Notes / caveats

- Object-aware steering mostly helps **framed** objects. M42 fills the frame, so
  its mask is ~100% → most steering is a no-op for it (verified).
- Position angle isn't in the catalog → ellipses are axis-aligned (WCS rotation
  applied when available).
- Residual vignette on frame-filling objects is a tiny linear gradient amplified
  by the aggressive stretch; a dedicated post-stretch gradient remover is a
  separate possible task (not on this roadmap unless requested).

---

## Polish fixes landed alongside (branch)

- Star dark-rings: additive recombine `working + (enhanced - starless)` — `9dbf909`
- Residual vignette: order-3 clamped background on object-dominated frames — `fcd3e2a`
- Deconvolution ringing on soft/large PSFs: gentler RL for FWHM > 5px — `19a70bb`
- Star-core deconv ringing: protect compact bright cores post-deconv — `df88d21`
- Colour overstretch: per-channel local contrast made bg-preserving — `b4b641e`
- Noisy/faint backgrounds: gentle background-floor pull for non-HDR frames — `2fae6f9`

- Post-stretch gradient / flat (vignette) tool: `cosmica/core/gradient_removal.py`
  + auto object-aware cleanup stage — `24562ef`. NGC 7023 corner spread 0.16 → 0.05.

### Known-hard cases / data-limited

- **NGC 7023 (faint OSC colour, dense field):** the LC, sky-veil, and gradient
  fixes darkened + flattened the background (corner spread 0.16 → 0.05) and the
  Iris nebula separates cleanly. Residual **colour noise** is the data floor
  (faint subs) — stronger colour denoise is the remaining lever.

- Chroma (colour) noise reduction for OSC: `cosmica/core/chroma_denoise.py` +
  post-stretch step — `7bf4ef4`. NGC 7023 chroma noise 0.085 → 0.018.
- Star-core deconv protection over-softened soft-PSF stars (M42) → gated to
  sharp PSFs only — `3c3c14f`.

- Recipe book: `cosmica/resources/recipes.json` + `cosmica/core/recipe.py` —
  per-type recipes + named overrides, resolved after plate solve, merged into
  hints (`ae56f72`). NGC 7023 → chroma 2.0 + StarNet + gentle. The seam for #4.

### 🔜 Next

- [ ] **Curate/tune recipes** as targets are tested (it's now one JSON edit per
      target — e.g. tune the Iris, add Veil/M31/M81 overrides). This is the
      ongoing knob now, replacing scattered heuristic edits.
- [ ] **#4 — learning (ML)**: self-supervised denoise fine-tune on the user's raw
      subs (thousands available, Noise2Self infra exists: prepare_data_v2 +
      train_n2s_v2). HDD-friendly via one-time patch extraction → cached memmap.
      Bounded POC first; uncertain it beats the stock model.
