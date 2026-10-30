"""Small integration test: inclusive boundaries, exact floats, corrupt clip exclusion."""
import json
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numcodecs import Zstd

from syndata_to_lerobot_v3 import convert, dump, Stats


class ConversionTest(unittest.TestCase):
    def test_histogram_and_merged_statistics_equal_direct_statistics(self):
        rgb = np.random.default_rng(42).integers(0,256,size=(5,8,8,3),dtype=np.uint8)
        left,right = Stats(),Stats()
        left.update('rgb',rgb[:2],video=True)
        right.update('rgb',rgb[2:],video=True)
        left.merge(right)
        actual = left.export(5)['rgb']
        pixels = rgb.reshape(-1,3).astype(np.float64)/255
        expected = {'min':pixels.min(0),'max':pixels.max(0),'mean':pixels.mean(0),'std':pixels.std(0)}
        for name,value in expected.items():
            np.testing.assert_allclose(np.asarray(actual[name]).reshape(3),value,atol=1e-12)
        self.assertEqual(actual['count'],[5])

    def test_missing_chunk_rejects_only_affected_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            subset = base/'glove-test'
            volume = subset/'tasks/task_0001/000001.zarr'
            rgb = np.arange(6*8*8*3,dtype=np.uint8).reshape(6,8,8,3)
            depth = np.arange(6*8*8,dtype=np.float32).reshape(6,8,8)/7
            qpos = np.arange(6*30,dtype=np.float32).reshape(6,30)/11
            data = {'head_rgb':rgb,'head_depth':depth,'left_qpos':qpos}
            dump(volume/'zarr.json',{'zarr_format':3,'node_type':'group','attributes':{'modalities':list(data)}})
            dump(volume/'data/zarr.json',{'zarr_format':3,'node_type':'group'})
            for name,x in data.items():
                root = volume/'data'/name
                chunks = [2,*x.shape[1:]]
                dump(root/'zarr.json',{'zarr_format':3,'node_type':'array','shape':list(x.shape),
                    'data_type':x.dtype.name,'chunk_grid':{'name':'regular','configuration':{'chunk_shape':chunks}},
                    'chunk_key_encoding':{'name':'default','configuration':{'separator':'/'}},
                    'codecs':[{'name':'bytes'},{'name':'zstd'}]})
                for i in range(3):
                    if name=='head_depth' and i==1:
                        continue
                    p = root/'c'/str(i)
                    for _ in x.shape[1:]:
                        p /= '0'
                    p.parent.mkdir(parents=True,exist_ok=True)
                    p.write_bytes(Zstd().encode(x[i*2:i*2+2].tobytes()))
            (subset/'index').mkdir()
            (subset/'annotations').mkdir()
            pq.write_table(pa.Table.from_pylist([{'clip_id':f'c{i}','task_key':'task_0001','volume_id':'000001',
                'rel_path':'tasks/task_0001/000001.zarr','start_idx':2*i,'end_idx':2*i+1,'num_frames':2}
                for i in range(3)]),subset/'index/clips.parquet')
            pq.write_table(pa.Table.from_pylist([{'clip_id':f'c{i}','task_summary_en':'test'} for i in range(3)]),
                subset/'annotations/clip_annotations.parquet')
            dump(subset/'task.json',[{'task_id':'task_0001','task_name':'test'}])
            output = base/'output'
            args = Namespace(source=volume,output=output,fps=10,batch_frames=2,data_mb=100,video_mb=200)
            convert(args)
            report = json.loads((output/'qc/report.json').read_text())
            self.assertEqual(report['accepted_clips'],2)
            self.assertEqual(report['rejected_clips'][0]['clip_id'],'c1')
            self.assertEqual(pq.read_table(output/'data')['source_frame_index'].to_pylist(),[0,1,4,5])
            info = json.loads((output/'meta/info.json').read_text())
            self.assertEqual(info['total_frames'],4)
            self.assertNotIn('action',info['features'])
            with self.assertRaises(FileExistsError):
                convert(args)


if __name__ == '__main__':
    unittest.main()
