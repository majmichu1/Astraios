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
| #1 consumer — local contrast | HDR + star-aware CLAHE confined to the subject; sky grain left alone | `351ff48` |
| #2 richer recipes — hints | every catalog `processing_hint` now consumed (`bg_sensitive`, `ha_dominant`, `reflection_nebulosity` were dead) | `74ae583`, `3e81318` |

### 🔜 Next (this is where we're working)

- [ ] **#1c — wire object mask into deconvolution**: deconvolve the subject,
      leave empty sky un-sharpened (less noise amplification).
- [ ] **#1d — wire object mask into the stretch**: target the *object* region's
      brightness so framed subjects are well-exposed (vs. whole-frame heuristic).
- [ ] **#2b — auto-populate recipes from SIMBAD/NED**: cover thousands of objects,
      not 139; derive type/size/surface-brightness → recipe.
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
