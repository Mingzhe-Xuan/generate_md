from types import SimpleNamespace as Node
import unittest
from paper_dft import leaf_jobs


class WorkflowTests(unittest.TestCase):
    def test_nested_double_relax_and_static_keep_stage_order(self):
        relax1, relax2, static = [Node(name=n) for n in ('relax1','relax2','static')]
        flow = Node(jobs=[Node(jobs=[relax1,relax2]),static])
        self.assertEqual(list(leaf_jobs(flow)), [relax1,relax2,static])


if __name__ == '__main__':
    unittest.main()
