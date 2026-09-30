from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from v1_core import (economic_profiles, filter_observations, make_observations, model_for,
                     overlap_jsd)
from v1_shadow import run_shadow


class V1Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, _, cls.mapping = model_for("original")
        cls.fixture = pd.read_csv(ROOT / "tests" / "fixtures" / "v06_2026_09_28_states.csv")
        cls.fixture_obs = pd.DataFrame(
            {"log_return": cls.fixture["log_return"].to_numpy()},
            index=pd.to_datetime(cls.fixture["timestamp"]),
        )
        cls.fixture_starts = cls.fixture["sequence_reset"].to_numpy(dtype=bool)

    def test_original_artifact_reproduces_saved_v06_posteriors(self):
        stable, _, _ = filter_observations(
            self.original, self.mapping, self.fixture_obs, self.fixture_starts
        )
        actual = np.empty_like(stable)
        actual[:, self.mapping] = stable
        expected = self.fixture[[f"p_state_{i}" for i in range(7)]].to_numpy()
        self.assertLess(float(np.max(np.abs(actual - expected))), 1e-10)

    def test_filter_is_causal_and_resets(self):
        full, _, _ = filter_observations(
            self.original, self.mapping, self.fixture_obs, self.fixture_starts
        )
        prefix, _, _ = filter_observations(
            self.original, self.mapping, self.fixture_obs.iloc[:40], self.fixture_starts[:40]
        )
        np.testing.assert_allclose(full[:40], prefix, atol=1e-12)
        obs = self.fixture_obs.iloc[:3]
        starts = np.array([True, False, True])
        reset, _, _ = filter_observations(self.original, self.mapping, obs, starts)
        alone, _, _ = filter_observations(self.original, self.mapping, obs.iloc[2:], np.array([True]))
        np.testing.assert_allclose(reset[2], alone[0], atol=1e-12)

    def test_emission_diagnostics_are_symmetric(self):
        strict, _, mapping = model_for("strict")
        metrics = overlap_jsd(strict, mapping)
        overlap = np.asarray(metrics["overlap_matrix"])
        jsd = np.asarray(metrics["jsd_nats_matrix"])
        np.testing.assert_allclose(overlap, overlap.T, atol=1e-12)
        np.testing.assert_allclose(jsd, jsd.T, atol=1e-12)
        self.assertEqual(len(metrics["pairs"]), 21)
        self.assertTrue(np.all((overlap >= 0) & (overlap <= 1.0001)))
        self.assertTrue(np.all((jsd >= -1e-12) & (jsd <= np.log(2) + 1e-4)))

    def test_shadow_scores_before_update(self):
        obs = self.fixture_obs.iloc[:20].copy()
        obs["log_return"] = self.fixture_obs["log_return"].iloc[:20]
        frozen_p, frozen_score, _ = filter_observations(
            self.original, self.mapping, obs, self.fixture_starts[:20]
        )
        self.assertEqual(len(frozen_p), 20)
        records, report = run_shadow(
            self.original, self.mapping, obs, self.fixture_starts[:20], frozen_score,
            "2026-09-28"
        )
        self.assertEqual(report["updates"], 20)
        self.assertAlmostEqual(
            float(records["shadow_log_density"].iloc[0]), float(frozen_score[0]), places=12
        )

    def test_next_minute_profile_does_not_cross_lunch(self):
        obs = pd.DataFrame(
            {"log_return": [0.01, 0.02], "volume": [100, 200],
             "session": ["AM", "PM"]},
            index=pd.to_datetime(["2026-09-30 11:30", "2026-09-30 13:01"]),
        )
        posterior = np.full((2, 7), 1 / 7)
        profiles = economic_profiles(obs, posterior, np.array([True, False]))
        self.assertTrue(all(row["next_observation_mean_return_pct"] is None
                            for row in profiles["profiles"]))

    def test_end_to_end_on_small_historical_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            times = pd.date_range("2026-09-30 09:00", periods=9, freq="min")
            close = np.array([1800.0, 1800.2, 1800.1, 1800.5, 1800.4,
                              1800.7, 1800.6, 1800.8, 1800.9])
            data = pd.DataFrame({
                "SYMBOL": "VN30F1M",
                "TRADING_DATE": [t.strftime("%Y%m%d") for t in times],
                "TRADING_TIME": [t.strftime("%H:%M:%S") for t in times],
                "OPEN_PX": close,
                "HIGH_PX": close,
                "LOW_PX": close,
                "CLOSE_PX": close,
                "VOL": np.arange(len(times)) + 100,
            })
            source = folder / "sample.csv"
            data.to_csv(source, index=False)
            command = [
                sys.executable, str(ROOT / "run_v1.py"), "--data", str(source),
                "--model", "strict", "--shadow-start", "2026-09-30",
                "--out", str(folder / "results"),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            output = folder / "results"
            report = json.loads((output / "diagnostics_strict.json").read_text(encoding="utf-8"))
            records = pd.read_csv(output / "states_strict.csv")
            self.assertEqual(report["status"], "historical_diagnostics_complete")
            self.assertEqual(len(records), 8)
            self.assertEqual(report["online_shadow"]["updates"], 8)
            self.assertTrue(records["available_at"].isna().all())
            np.testing.assert_allclose(
                records[[f"p_S{i}" for i in range(7)]].sum(axis=1), 1, atol=1e-10
            )
            self.assertEqual(records["event_bar_end"].iloc[0], "2026-09-30 09:02:00")


if __name__ == "__main__":
    unittest.main()
