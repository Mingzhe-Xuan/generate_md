from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from safe_custodian import terminate_launchers


class SafeTerminationTests(unittest.TestCase):
    def test_only_owned_launcher_in_exact_stage_can_be_terminated(self):
        with tempfile.TemporaryDirectory() as directory:
            stage=Path(directory)/'stage';stage.mkdir()
            def child(pid, cwd=stage, job='123', command=None):
                process=Mock(pid=pid)
                process.cmdline.return_value=command or ['srun','/opt/vasp.5.4.4/bin/vasp_std']
                process.cwd.return_value=str(cwd)
                process.environ.return_value={'SLURM_JOB_ID':job}
                return process
            owned=child(100)
            other_stage=child(101,cwd=directory)
            other_job=child(102,job='124')
            unrelated=child(103,command=['srun','python'])
            processes=[owned,other_stage,other_job,unrelated]
            actions=terminate_launchers(stage,directory,'123',processes)
            self.assertEqual([a['pid'] for a in actions],[100])
            owned.terminate.assert_called_once()
            for process in processes[1:]:process.terminate.assert_not_called()
            with self.assertRaises(ValueError):
                terminate_launchers(directory,stage,'123',processes)


if __name__ == '__main__':
    unittest.main()
