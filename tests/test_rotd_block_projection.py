"""Upstream RotD projection tests by Silvia Mazzoni, 2026."""

from __future__ import annotations

import sys
import types
import unittest

import numpy as np


try:
    import numba  # noqa: F401
except ImportError:
    fallback = types.ModuleType("numba")

    def identity_jit(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1:
            return args[0]
        return lambda function: function

    fallback.jit = identity_jit
    sys.modules["numba"] = fallback

import reqpy_M


class RotDBlockProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(2026)
        cls.dt = 0.01
        time = np.arange(384) * cls.dt
        cls.component_1 = (
            0.12 * np.sin(2 * np.pi * 1.15 * time)
            + 0.004 * rng.standard_normal(time.size)
        )
        cls.component_2 = (
            0.10 * np.sin(2 * np.pi * 0.85 * time + 0.45)
            + 0.004 * rng.standard_normal(time.size)
        )
        cls.periods = np.array([0.0, 0.08, 0.2, 0.5, 1.0, 2.0])
        cls.angles = np.array([0.0, 17.0, 42.0, 89.0, 137.0])

    def test_projection_matches_independent_rotated_time_histories(self):
        actual = reqpy_M.compute_rotated_spectra_pw(
            self.periods, self.component_1, self.component_2,
            0.05, self.dt, self.angles, angle_block_size=2,
        )
        expected = [np.zeros_like(quantity) for quantity in actual]
        for angle_index, angle in enumerate(np.deg2rad(self.angles)):
            rotated = (
                self.component_1 * np.cos(angle)
                + self.component_2 * np.sin(angle)
            )
            direct = reqpy_M.compute_spectrum_pw(
                self.periods, rotated, 0.05, self.dt
            )
            for expected_quantity, direct_quantity in zip(expected, direct):
                expected_quantity[angle_index, :] = direct_quantity

        for actual_quantity, expected_quantity in zip(actual, expected):
            np.testing.assert_allclose(
                actual_quantity, expected_quantity, rtol=1e-12, atol=1e-12
            )

    def test_block_size_does_not_change_results(self):
        reference = reqpy_M.compute_rotated_spectra_pw(
            self.periods, self.component_1, self.component_2,
            0.05, self.dt, self.angles, angle_block_size=1,
        )
        for block_size in (2, 7, 24, 181):
            actual = reqpy_M.compute_rotated_spectra_pw(
                self.periods, self.component_1, self.component_2,
                0.05, self.dt, self.angles, angle_block_size=block_size,
            )
            for actual_quantity, expected_quantity in zip(actual, reference):
                np.testing.assert_allclose(
                    actual_quantity, expected_quantity, rtol=0, atol=0
                )

    def test_block_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "angle_block_size"):
            reqpy_M.compute_rotated_spectra_pw(
                self.periods, self.component_1, self.component_2,
                0.05, self.dt, self.angles, angle_block_size=0,
            )


if __name__ == "__main__":
    unittest.main()
