# Publication status

Final-scope execution, observation synchronization, result projection,
configuration and task documentation are implemented in this revision.

## Validation performed

Local CPU/numerical validation: **81 passed**, comprising 38 new final-experiment
checks and 43 existing campaign checks. The final recorded invocation completed
in 8.97 seconds. Python compileall and shell syntax checks passed.

New source-file Git blob identities were read back from GitHub and match the
locally tested files:
- robo/campaign/finalize.py: 9791ed836080f6523e636a7706d393f10f89bed6
- robo/eval/final_release.py: 5abc536dc9003e052f04c564db11dba7bc978e5e
- robo/rendering/final_observation.py: d9587362ff247374e572bb9a02e21e62c3c4233f
- tests/test_final_experiments.py: 9ee440644ea37faed4dd2bed1b99dbbab464f817

GitHub Actions run34552485615 at source929538f5 returned failure with no step
records; job103118094610 had no downloadable log (404 BlobNotFound). Therefore
**remote CI success is NOT claimed** and its cause is not inferred. The expanded
workflow is committed so it can be rerun when the runner is available.

## Not executed by this publication

No new pretrained-model experiment, native policy rollout, data download, paid
API request or Slurm job was started. The narrow native worker/final-arm/renderer
integration in SOURCE_INTEGRATION.md still requires actual cluster validation.
These CPU tests use controlled fixtures, not measured robot/model performance.
This is not a completed scientific-result release.

Execution owners append exact commit, commands, real DEV admission, job IDs,
measured/planned counts, output paths and limitations to their own status files.
Only05_release closes the final scientific evidence set.
