import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arsh_v05 as arsh
from runtime import RegimeRuntime


def synthetic_minutes(days=2):
    rows=[]
    for d in pd.date_range("2024-01-02",periods=days,freq="B"):
        times=pd.date_range(f"{d.date()} 09:00",f"{d.date()} 11:30",freq="min").append(
            pd.date_range(f"{d.date()} 13:00",f"{d.date()} 14:30",freq="min")).append(
            pd.DatetimeIndex([pd.Timestamp(f"{d.date()} 14:45")]))
        for i,t in enumerate(times):
            p=100+.01*i+.1*(d.day-2)
            rows.append({"datetime":t,"CLOSE_PX":p})
    return pd.DataFrame(rows).set_index("datetime")


class TestV05(unittest.TestCase):
    def test_horizons_are_nonoverlapping_and_do_not_cross_breaks(self):
        frames,audit,rejected,gaps=arsh.build_returns(synthetic_minutes(),(5,10,15,30,60))
        expected_per_day={5:48,10:24,15:16,30:8,60:3}
        for h,frame in frames.items():
            self.assertEqual(len(frame),2*expected_per_day[h])
            self.assertTrue(((frame.index-frame.window_start)==pd.Timedelta(h,unit="min")).all())
            self.assertFalse(((frame.window_start.dt.hour<12)&(frame.index.hour>=12)).any())
        self.assertEqual(set(gaps.gap_type),{"lunch","closing_auction","overnight_open"})

    def test_strict_window_is_rejected_without_interpolation(self):
        raw=synthetic_minutes(1).drop(pd.Timestamp("2024-01-02 09:03"))
        frames,audit,rejected,_=arsh.build_returns(raw,(5,))
        self.assertEqual(len(frames[5]),47)
        self.assertEqual(len(rejected),1)
        self.assertEqual(int(rejected.iloc[0].observed_prices),5)

    def test_one_internal_missing_sensitivity_uses_endpoints_without_imputation(self):
        raw=synthetic_minutes(1).drop(pd.Timestamp("2024-01-02 09:03"))
        strict=arsh.build_returns(raw,(5,))[0][5]
        allowed=arsh.build_returns(raw,(5,),allow_one_internal_missing=True)[0][5]
        self.assertEqual(len(allowed),len(strict)+1)
        recovered=allowed.loc[pd.Timestamp("2024-01-02 09:05")]
        self.assertEqual(int(recovered.internal_missing_minutes),1)
        expected=np.log(raw.loc[pd.Timestamp("2024-01-02 09:05"),"CLOSE_PX"] /
                        raw.loc[pd.Timestamp("2024-01-02 09:00"),"CLOSE_PX"])
        self.assertAlmostEqual(float(recovered.log_return),float(expected))

    def test_overlapping_is_explicit_and_stays_inside_sessions(self):
        raw=synthetic_minutes(1)
        nonoverlap=arsh.build_returns(raw,(5,),overlapping=False)[0][5]
        overlap=arsh.build_returns(raw,(5,),overlapping=True)[0][5]
        self.assertEqual(len(nonoverlap),48)
        self.assertEqual(len(overlap),232)
        self.assertTrue(((overlap.index-overlap.window_start)==pd.Timedelta(5,unit="min")).all())
        self.assertFalse(((overlap.window_start.dt.hour<12)&(overlap.index.hour>=12)).any())

    def test_intraday_adjustment_uses_train_only(self):
        frame=arsh.build_returns(synthetic_minutes(5),(5,))[0][5]
        train=frame[frame.date<=frame.date.unique()[2]].copy()
        validation=frame[frame.date==frame.date.unique()[3]].copy()
        test=frame[frame.date==frame.date.unique()[4]].copy()
        first=arsh.intraday_adjust_split(train,validation,test)
        changed=test.copy();changed["log_return"]=changed.log_return*1000
        second=arsh.intraday_adjust_split(train,validation,changed)
        np.testing.assert_allclose(first[0].log_return,second[0].log_return)
        np.testing.assert_allclose(first[1].log_return,second[1].log_return)
        np.testing.assert_allclose(first[2].intraday_volatility_factor,
                                   second[2].intraday_volatility_factor)

    def test_duration_uses_bars_trading_and_calendar_minutes(self):
        test=arsh.build_returns(synthetic_minutes(1),(5,))[0][5].iloc[:4]
        out=arsh.duration_summary(test,np.array([0,0,1,1]),0,5)
        self.assertEqual(out["mean_run_bars"],2)
        self.assertEqual(out["mean_trading_minutes"],10)
        self.assertEqual(out["mean_calendar_minutes"],10)

    def test_filter_is_causal(self):
        rng=np.random.default_rng(3);x=np.r_[rng.normal(-1,.3,80),rng.normal(1,.4,80)]
        model=arsh.GaussianSequenceHMM(2,n_iter=60,random_state=3).fit(x)
        p1,_=arsh.causal_filter(model,x);changed=x.copy();changed[100:]+=100
        p2,_=arsh.causal_filter(model,changed)
        np.testing.assert_allclose(p1[:100],p2[:100])

    def test_sequence_policy_controls_em_and_filter_boundaries(self):
        frame=arsh.build_returns(synthetic_minutes(2),(5,))[0][5]
        self.assertIsNone(arsh.sequence_lengths(frame,"continuous_carry"))
        np.testing.assert_array_equal(arsh.sequence_lengths(frame,"daily_sequence"),[48,48])
        np.testing.assert_array_equal(arsh.sequence_lengths(frame,"session_sequence"),[30,18,30,18])
        starts=arsh.sequence_starts(frame,"session_sequence")
        self.assertEqual(int(starts.sum()),4)

    def test_removed_window_is_a_hard_sequence_boundary(self):
        frame=arsh.build_returns(synthetic_minutes(1),(5,))[0][5]
        removed=frame.drop(frame.index[5])
        lengths=arsh.sequence_lengths(removed,"continuous_carry")
        np.testing.assert_array_equal(lengths,[5,len(removed)-5])

    def test_filter_resets_exactly_at_requested_boundary(self):
        rng=np.random.default_rng(8);x=rng.normal(size=120)
        model=arsh.GaussianSequenceHMM(2,n_iter=20,random_state=8).fit(x)
        starts=np.zeros(len(x),dtype=bool);starts[60]=True
        full,_=arsh.causal_filter(model,x,starts=starts)
        tail,_=arsh.causal_filter(model,x[60:])
        np.testing.assert_allclose(full[60:],tail)

    def test_runtime_roundtrip_with_history(self):
        rng=np.random.default_rng(2);x=rng.normal(size=150)
        model=arsh.GaussianSequenceHMM(2,n_iter=30,random_state=2).fit(x)
        scaler=arsh.StandardScaler().fit(pd.DataFrame({"log_return":x}))
        runtime=RegimeRuntime(model,scaler,"continuous_carry","v0.5-test",{0:"A",1:"B"},
                              posterior=np.array([.4,.6]),last_timestamp=pd.Timestamp("2024-01-02 09:05"))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"runtime.joblib";runtime.save(path);loaded=RegimeRuntime.load(path)
            result=loaded.predict_one(.1,"2024-01-02 09:10")
        self.assertAlmostEqual(sum(result["posterior"]),1.)
        self.assertFalse(result["reset"])


if __name__=="__main__":unittest.main()
