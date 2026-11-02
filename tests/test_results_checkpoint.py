"""Subgroup counts must preserve full-image one-to-one matching."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_test import subgroup_rows


class SubgroupTests(unittest.TestCase):
    def test_one_prediction_cannot_count_in_two_size_groups(self):
        images = [dict(image='a', targets=[[0,0,10,10,0],[0,0,9,9,0]],
                       predictions=[[0,0,10,10,.8,0]])]
        objects = [dict(image='a', target_index=i, class_name='helmet', size=size,
                        visibility='clear', truncated='no') for i,size in enumerate(['medium','small'])]
        rows = subgroup_rows(images, objects, .2, 'model', 42)
        sizes = {r['group']:r for r in rows if r['dimension']=='size' and r['class_name']=='ALL'}
        self.assertEqual(sizes['medium']['tp'],1)
        self.assertEqual(sizes['small']['tp'],0)
        self.assertEqual(sizes['small']['support'],1)

    def test_uncertain_visibility_does_not_enter_clear_occluded(self):
        images = [dict(image='a',targets=[[0,0,10,10,0]],predictions=[])]
        objects = [dict(image='a',target_index=0,class_name='helmet',size='small',
                        visibility='uncertain',truncated='uncertain')]
        rows = subgroup_rows(images,objects,.2,'model',42)
        groups = {r['group'] for r in rows if r['dimension']=='visibility'}
        self.assertEqual(groups,{'uncertain'})

if __name__ == '__main__':
    unittest.main()
