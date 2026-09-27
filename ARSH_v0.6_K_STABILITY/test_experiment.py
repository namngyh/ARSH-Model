"""Behavioral checks: causality, convergence, restart integrity and model locks."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'reference'))
import numpy as np
import pandas as pd
from label_confirmation import ConfirmationRule,confirm_labels
from stability_fit import convergence,fit_seed,match_states
from run_experiment import read_config,prepare_frames,select_k,run,sha,evaluate_future

BASE_CFG = read_config(HERE/'experiment_config.json')
FIT_CFG = {k:BASE_CFG[k] for k in ['max_updates','min_updates','absolute_ll_tolerance',
    'll_tolerance_per_observation','consecutive_small_improvements','negative_ll_tolerance','checkpoint_every']}


class ConfirmationTests(unittest.TestCase):
    def test_uncertain_candidate_does_not_switch_and_reports_actual_probability(self):
        p = np.array([[.8,.2],[.45,.55],[.3,.7],[.2,.8]])
        result = confirm_labels(p,[True,False,False,False],ConfirmationRule('test',.6,.1,2))
        np.testing.assert_array_equal(result['confirmed_label'],[0,0,0,1])
        np.testing.assert_array_equal(result['display_label'],[0,-1,-1,1])
        np.testing.assert_allclose(result['display_probability'],[.8,.45,.3,.8])
        self.assertTrue(result['uncertain'][2])
        self.assertEqual(result['confirmation_wait_observations'][3],1)

    def test_reset_discards_pending_confirmation(self):
        p = np.array([[.8,.2],[.2,.8],[.8,.2],[.2,.8]])
        result = confirm_labels(p,[True,False,True,False],ConfirmationRule('test',.6,.1,2))
        np.testing.assert_array_equal(result['confirmed_label'],[0,0,0,0])
        np.testing.assert_array_equal(result['display_label'],[0,-1,0,-1])

    def test_future_observations_cannot_change_previous_labels(self):
        rng = np.random.default_rng(3)
        p = rng.dirichlet([1,1,1],size=100)
        original = p.copy()
        starts = np.zeros(100,dtype=bool); starts[0]=True; starts[30]=True
        rule = ConfirmationRule('test',.6,.1,3)
        a = confirm_labels(p[:50],starts[:50],rule)
        b = confirm_labels(p,starts,rule)
        np.testing.assert_array_equal(a['display_label'],b['display_label'][:50])
        np.testing.assert_array_equal(p,original)

    def test_raw_rule_is_exact_argmax(self):
        p = np.array([[.6,.4],[.49,.51],[.9,.1]])
        r = confirm_labels(p,[True,False,False],ConfirmationRule('raw',0,0,1))
        np.testing.assert_array_equal(r['display_label'],p.argmax(1))


class FitTests(unittest.TestCase):
    def test_plateau_requires_patience_and_both_thresholds(self):
        c = dict(FIT_CFG,min_updates=1,consecutive_small_improvements=3)
        self.assertTrue(convergence([0,.001,.002,.003],10000,c))
        self.assertFalse(convergence([0,.001,.002,.003],100,c))
        self.assertFalse(convergence([0,.001,.002,.003],10000,dict(c,min_updates=5)))
        self.assertFalse(convergence([0,.001,-1,.003],10000,c))

    def test_resumed_fit_matches_uninterrupted_and_scores_current_parameters(self):
        rng=np.random.default_rng(9)
        x=np.r_[rng.normal(-1,.3,60),rng.normal(1,.5,60)]
        cfg=dict(FIT_CFG,max_updates=12,min_updates=12,checkpoint_every=4)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            whole,dw=fit_seed(x,[20]*6,2,42,cfg,p/'whole.joblib')
            _,interrupted=fit_seed(x,[20]*6,2,42,cfg,p/'resume.joblib',max_evaluations_this_call=5)
            self.assertFalse(interrupted['finished'])
            resumed,dr=fit_seed(x,[20]*6,2,42,cfg,p/'resume.joblib')
            self.assertEqual(resumed.n_iter,cfg['max_updates'])
            self.assertEqual(dr['device'],'cpu')
            np.testing.assert_allclose(whole.means_,resumed.means_,atol=1e-12)
            np.testing.assert_allclose(whole.transmat_,resumed.transmat_,atol=1e-12)
            self.assertAlmostEqual(dw['train_ll'],dr['train_ll'],places=10)
            self.assertAlmostEqual(resumed.score(x,[20]*6),dr['train_ll'],places=8)
            with self.assertRaisesRegex(ValueError,'Checkpoint inputs differ'):
                fit_seed(x+.01,[20]*6,2,42,cfg,p/'resume.joblib')

    def test_alignment_handles_permuted_state_ids(self):
        from student_t_hmm import StudentTHMM
        a=StudentTHMM(3);a.means_=np.array([-1.,0.,2.]);a.scale2_=np.array([.2,.5,1.]);a.df_=np.ones(3)*8
        b=copy.deepcopy(a)
        b.means_=a.means_[[2,0,1]];b.scale2_=a.scale2_[[2,0,1]]
        permutation,cost=match_states(a,b)
        np.testing.assert_array_equal(permutation,[1,2,0])
        self.assertEqual(cost,0)

    def test_selection_ignores_unconverged_k_and_prefers_smaller_k_in_tie(self):
        table=pd.DataFrame([{'k':5,'eligible':True,'validation_log_density':6.1},
            {'k':6,'eligible':True,'validation_log_density':6.101},
            {'k':8,'eligible':False,'validation_log_density':6.2}])
        self.assertEqual(select_k(table,BASE_CFG)['chosen_k'],5)
        table.eligible=False
        self.assertIsNone(select_k(table,BASE_CFG)['chosen_k'])


def synthetic_source(path):
    rng=np.random.default_rng(22)
    rows=[]
    # Sparse dates are deliberate synthetic fixtures, never a research dataset.
    for date in ['2023-02-01','2026-01-20','2026-02-02','2026-07-31']:
        price=1500.
        for session in [('09:00',150),('13:00',90)]:
            for t in pd.date_range(date+' '+session[0],periods=session[1],freq='min'):
                price*=np.exp(rng.normal((1 if t.minute%2 else -1)*.00015,.0004))
                rows.append({'SYMBOL':'VN30F1M','TRADING_DATE':t.strftime('%Y%m%d'),
                    'TRADING_TIME':t.strftime('%H:%M:%S'),'OPEN_PX':price,'HIGH_PX':price,
                    'LOW_PX':price,'CLOSE_PX':price,'VOL':1})
        t=pd.Timestamp(date+' 14:45')
        rows.append({'SYMBOL':'VN30F1M','TRADING_DATE':t.strftime('%Y%m%d'),
            'TRADING_TIME':'14:45:00','OPEN_PX':price,'HIGH_PX':price,'LOW_PX':price,'CLOSE_PX':price,'VOL':1})
    pd.DataFrame(rows).to_csv(path,index=False)


class IntegrationTests(unittest.TestCase):
    def test_full_pipeline_resume_preserves_locked_artifacts_and_rejects_seen_holdout(self):
        cfg=copy.deepcopy(BASE_CFG)
        cfg.update(experiment='SYNTHETIC_SMOKE_ONLY',selection_seeds=[62],final_seeds=[142],
            max_updates=8,min_updates=5,consecutive_small_improvements=2,
            absolute_ll_tolerance=1e6,ll_tolerance_per_observation=1e6,checkpoint_every=4)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=root/'source.csv';out=root/'outputs'
            synthetic_source(data)
            train,valid,final,audit=prepare_frames(data,cfg)
            self.assertTrue(valid.index.max()<pd.Timestamp('2026-08-01'))
            args=argparse.Namespace(out=out,backend='cpu',device='cuda:0',data=data,fetch_db=False,phase='all')
            run(args,cfg)
            artifacts={k:out/f'final_models/k{k}/runtime_research_candidate.joblib' for k in cfg['k_values']}
            self.assertTrue(all(p.exists() for p in artifacts.values()))
            hashes={k:sha(p) for k,p in artifacts.items()}
            manifests={k:(p.parent/'model_manifest.json').read_bytes() for k,p in artifacts.items()}
            run(args,cfg)
            self.assertEqual(hashes,{k:sha(p) for k,p in artifacts.items()})
            self.assertEqual(manifests,{k:(p.parent/'model_manifest.json').read_bytes() for k,p in artifacts.items()})
            with self.assertRaisesRegex(ValueError,'No complete unseen days'):
                evaluate_future(argparse.Namespace(model=artifacts[5].parent,data=data,out=root/'future'),cfg)


if __name__=='__main__':
    unittest.main(verbosity=2)
