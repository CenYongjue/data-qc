"""合成案例测试，不重复扫描真实数据。"""
import unittest
import json
from pathlib import Path
import numpy as np
from agibot_quality_report import runs, propose_trim, image_metrics, JOINT, END, GRIP


class QualityRulesTest(unittest.TestCase):
    def setUp(self):
        self.config=json.loads(Path(__file__).with_name('agibot_qc_config.json').read_text())

    def test_intervals_are_exclusive(self):
        spans=runs([False,True,True,False,True],10)
        self.assertEqual([(r['start_frame'],r['end_frame_exclusive']) for r in spans],[(1,3),(4,5)])

    def test_annotation_protection_and_static_clip(self):
        x={JOINT:np.zeros((120,14)),END:np.zeros((120,2,3)),GRIP:np.zeros((120,2))}
        scales={JOINT:np.ones(14),END:np.ones(6),GRIP:np.ones(2)}
        trim,_=propose_trim(x,{'action_config':[]},scales,self.config,30)
        self.assertTrue(trim['no_motion']);self.assertEqual(trim['end_frame_exclusive'],120)
        for a in x.values():
            a[40:80]=np.linspace(0,1,40).reshape((40,)+(1,)*(a.ndim-1));a[80:]=1
        trim,_=propose_trim(x,{'action_config':[{'start_frame':0,'end_frame':119}]},scales,self.config,30)
        self.assertEqual((trim['start_frame'],trim['end_frame_exclusive']),(0,120))

    def test_low_texture_does_not_mean_blur(self):
        m,_=image_metrics(np.full((480,640,3),128,np.uint8),self.config)
        self.assertTrue(m['low_texture']);self.assertFalse(m['blur']);self.assertIsNone(m['edge_laplacian'])


if __name__=='__main__': unittest.main()
