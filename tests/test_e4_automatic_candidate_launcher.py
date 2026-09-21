"""Real Slurm spool dispatch reaches the intended candidate source."""
import os
import shutil
import subprocess
from pathlib import Path
import pytest

CODE=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('source',[None,'relative',str(CODE)])
def test_candidate_launcher_copied_into_slurm_spool(tmp_path,source):
    spool=tmp_path/'slurm/job1';spool.mkdir(parents=True)
    script=spool/'slurm_script'
    shutil.copyfile(CODE/'run/icra2027/run_e4_automatic_candidates.sh',script)
    env=dict(os.environ);env.pop('E4_CODE',None)
    if source is not None:env['E4_CODE']=source
    result=subprocess.run(['bash',str(script),'invalid-phase'],cwd=spool,env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'No module named' not in result.stderr
    if source==str(CODE):assert "invalid choice: 'invalid-phase'" in result.stderr
    assert not (spool/'outputs').exists()
