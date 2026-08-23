"""Tests for the guided processing workflow.

These assert the things a beginner actually depends on: that the order is the
professionally correct one, that each step does what its description claims,
and that a plain "run it all" produces a finished image from a linear stack.
"""

import numpy as np

from astraios.core.guided import (
    build_workflow,
    is_color,
    looks_linear,
    run_workflow,
    workflow_for,
)


def _linear_stack(color=True, seed=0, gradient=True, green_cast=True):
    """A synthetic frame that looks like a real stacked, unstretched image."""
    rng = np.random.default_rng(seed)
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w]
    neb = 0.012 * np.exp(-(((xx - 100) ** 2 + (yy - 95) ** 2) / (2 * 40.0**2)))
    if color:
        img = np.full((3, h, w), 0.004, np.float32)
        for c, amp in enumerate((1.0, 0.55, 0.5)):
            img[c] += (neb * amp).astype(np.float32)
    else:
        img = (np.full((h, w), 0.004, np.float32) + neb).astype(np.float32)
    for _ in range(25):
        x, y = rng.integers(15, h - 15, 2)
        star = rng.uniform(0.02, 0.3) * np.exp(
            -(((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.8**2))
        )
        img = img + star.astype(np.float32)
    if gradient:
        img = img + (0.010 * (xx / w) + 0.006 * (yy / h)).astype(np.float32)
    if color and green_cast:
        img[1] *= 1.25
    img = img + rng.normal(0, 0.0015, img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


class TestWorkflowShape:
    def test_order_is_the_professional_one(self):
        """Everything that belongs on linear data must precede the stretch,
        and saturation (which needs a stretched image) must follow it."""
        ids = [s.step_id for s in build_workflow()]
        assert ids.index("gradient") < ids.index("stretch")
        assert ids.index("color") < ids.index("stretch")
        assert ids.index("sharpen") < ids.index("stretch")
        assert ids.index("denoise") < ids.index("stretch")
        assert ids.index("saturation") > ids.index("stretch")
        assert ids.index("trim") == 0

    def test_every_step_is_self_describing(self):
        """The whole point is that a beginner is never shown a bare slider."""
        for step in build_workflow():
            assert step.title and step.summary and step.detail
            for control in step.controls:
                assert control.summary and control.higher and control.lower
                assert control.maximum > control.minimum

    def test_colour_steps_are_dropped_for_mono(self):
        mono = _linear_stack(color=False)
        ids = [s.step_id for s in workflow_for(mono)]
        assert "color" not in ids
        assert "saturation" not in ids
        assert "stretch" in ids and "gradient" in ids

    def test_colour_steps_are_present_for_colour(self):
        ids = [s.step_id for s in workflow_for(_linear_stack())]
        assert "color" in ids and "saturation" in ids


class TestStepsDoWhatTheyClaim:
    @staticmethod
    def _background_ramp(image):
        """Total background tilt across the frame, from a plane fit to the
        darkest 40% of pixels.

        Deliberately not "difference between corner brightnesses": the target
        itself is brightest in the middle and reaches the corners unevenly, so
        corner spread largely measures the nebula. A gradient-free version of
        this same scene still reads 35% of the injected value that way, which
        makes it useless for judging gradient removal.
        """
        lum = image.mean(axis=0) if image.ndim == 3 else image
        h, w = lum.shape
        yy, xx = np.mgrid[0:h, 0:w]
        sel = lum < np.percentile(lum, 40)
        coef, *_ = np.linalg.lstsq(
            np.c_[xx[sel], yy[sel], np.ones(sel.sum())], lum[sel], rcond=None
        )
        return float(abs(coef[0]) * w + abs(coef[1]) * h)

    def test_gradient_step_removes_the_gradient(self):
        img = _linear_stack(green_cast=False)
        step = {s.step_id: s for s in build_workflow()}["gradient"]
        out = step.apply(img, step.suggest(img))
        assert self._background_ramp(out) < self._background_ramp(img) * 0.1

    def test_gradient_step_reaches_the_gradient_free_baseline(self):
        """The strong form: after removal the tilt should match an image that
        never had a gradient injected, not merely be smaller."""
        with_gradient = _linear_stack(green_cast=False, gradient=True)
        without_gradient = _linear_stack(green_cast=False, gradient=False)
        step = {s.step_id: s for s in build_workflow()}["gradient"]
        corrected = step.apply(with_gradient, step.suggest(with_gradient))
        baseline = self._background_ramp(without_gradient)
        assert self._background_ramp(corrected) < max(baseline * 3, 1e-3)

    def test_colour_step_removes_a_green_cast(self):
        img = _linear_stack(green_cast=True)
        step = {s.step_id: s for s in build_workflow()}["color"]
        out = step.apply(img, step.suggest(img))

        def green_excess(a):
            m = [float(a[c].mean()) for c in range(3)]
            neutral = (m[0] + m[2]) / 2
            return (m[1] - neutral) / max(neutral, 1e-9)

        assert abs(green_excess(out)) < abs(green_excess(img)) * 0.5

    def test_colour_step_leaves_a_neutral_image_alone(self):
        """Narrowband palettes must not have their green stripped."""
        img = _linear_stack(green_cast=False)
        step = {s.step_id: s for s in build_workflow()}["color"]
        assert step.suggest(img)["green_amount"] < 0.1

    def test_stretch_step_makes_the_image_visible(self):
        img = _linear_stack()
        step = {s.step_id: s for s in build_workflow()}["stretch"]
        assert looks_linear(img)
        out = step.apply(img, step.suggest(img))
        assert not looks_linear(out)
        assert float(np.median(out)) > float(np.median(img)) * 3

    def test_denoise_suggestion_scales_with_actual_noise(self):
        steps = {s.step_id: s for s in build_workflow()}
        clean = _linear_stack(seed=1)
        noisy = np.clip(
            clean + np.random.default_rng(2).normal(0, 0.02, clean.shape), 0, 1
        ).astype(np.float32)
        assert steps["denoise"].suggest(noisy)["strength"] > \
            steps["denoise"].suggest(clean)["strength"]

    def test_trim_removes_the_border(self):
        img = _linear_stack()
        step = {s.step_id: s for s in build_workflow()}["trim"]
        out = step.apply(img, {"percent": 10.0})
        assert out.shape[-1] < img.shape[-1] and out.shape[-2] < img.shape[-2]

    def test_trim_of_zero_is_a_no_op(self):
        img = _linear_stack()
        step = {s.step_id: s for s in build_workflow()}["trim"]
        assert step.apply(img, {"percent": 0.0}).shape == img.shape

    def test_trim_never_destroys_the_image(self):
        img = _linear_stack()
        step = {s.step_id: s for s in build_workflow()}["trim"]
        out = step.apply(img, {"percent": 99.0})
        assert out.shape[-1] >= 8 and out.shape[-2] >= 8


class TestRunWorkflow:
    def test_turns_a_linear_stack_into_a_finished_image(self):
        img = _linear_stack()
        run = run_workflow(img)
        assert looks_linear(img) and not looks_linear(run.image)
        assert "stretch" in run.applied and "gradient" in run.applied
        assert np.isfinite(run.image).all()
        assert run.image.min() >= 0.0 and run.image.max() <= 1.0

    def test_preserves_star_highlights(self):
        img = _linear_stack()
        out = run_workflow(img).image
        # a stretched result should still have bright stars, not a blown white field
        assert out.max() > 0.5
        assert float(np.mean(out > 0.95)) < 0.05

    def test_trim_is_skipped_unless_asked_for(self):
        run = run_workflow(_linear_stack())
        assert "trim" in run.skipped
        assert run.image.shape[-1] == 200

    def test_overriding_trim_opts_it_in(self):
        run = run_workflow(_linear_stack(), overrides={"trim": {"percent": 5.0}})
        assert "trim" in run.applied
        assert run.image.shape[-1] < 200

    def test_explicit_skip_is_honoured(self):
        run = run_workflow(_linear_stack(), skip={"stretch"})
        assert "stretch" in run.skipped
        assert looks_linear(run.image)

    def test_mono_runs_without_the_colour_steps(self):
        run = run_workflow(_linear_stack(color=False))
        assert "color" not in run.applied and "saturation" not in run.applied
        assert not looks_linear(run.image)

    def test_progress_is_reported(self):
        seen = []
        run_workflow(_linear_stack(), progress=lambda f, m: seen.append((f, m)))
        assert seen and seen[-1][0] == 1.0

    def test_input_is_not_mutated(self):
        img = _linear_stack()
        before = img.copy()
        run_workflow(img)
        assert np.array_equal(img, before)

    def test_a_failing_step_is_skipped_not_fatal(self, monkeypatch):
        import astraios.core.guided as g

        def _boom(image, params):
            raise RuntimeError("simulated failure")

        real = g.build_workflow

        def patched():
            steps = real()
            for s in steps:
                if s.step_id == "sharpen":
                    s.apply_fn = _boom
            return steps

        monkeypatch.setattr(g, "build_workflow", patched)
        run = g.run_workflow(_linear_stack())
        assert "sharpen" in run.skipped
        assert not looks_linear(run.image)   # the rest still ran


class TestHelpers:
    def test_is_color(self):
        assert is_color(np.zeros((3, 10, 10), np.float32))
        assert not is_color(np.zeros((10, 10), np.float32))

    def test_looks_linear(self):
        assert looks_linear(np.full((32, 32), 0.01, np.float32))
        assert not looks_linear(np.full((32, 32), 0.35, np.float32))
