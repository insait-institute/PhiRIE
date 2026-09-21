#!/usr/bin/env bash
# Source from an allocated native-render worker. Does not select extra GPUs.
python_native=${SR_NATIVE_PY:-${python_native:-/group/worldcept/PhiRIE/code/SimAny-wt/sr0-native/.venv-native/bin/python}}
export SR_NATIVE_PY="$python_native"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
unset __EGL_VENDOR_LIBRARY_FILENAMES EGL_PLATFORM
MUJOCO_EGL_DEVICE_ID=$("$python_native" - <<'PY'
import os, subprocess
selector=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
if not selector:
    raise RuntimeError('native EGL worker requires an allocated CUDA_VISIBLE_DEVICES')
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
pairs=[tuple(x.strip() for x in row.split(',')) for row in rows]
matches=[index for index,uuid in pairs if selector==uuid or selector==index]
if len(matches)!=1:raise RuntimeError(f'cannot bind allocated GPU selector {selector!r}')
print(matches[0])
PY
) || return 1
export MUJOCO_EGL_DEVICE_ID
# Pinned robosuite validates numeric EGL ID against CUDA_VISIBLE_DEVICES.
# This changes the spelling of the same allocated GPU, never its identity.
export CUDA_VISIBLE_DEVICES="$MUJOCO_EGL_DEVICE_ID"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export TMPDIR=${TMPDIR:-/group/worldcept/PhiRIE/code/SimAny-wt/sr0-native/outputs/sr0-setup/tmp}
