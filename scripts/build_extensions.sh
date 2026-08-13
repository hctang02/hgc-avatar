#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"

"${python_bin}" -m pip install --no-build-isolation \
  "${project_dir}/gaussians/diff_gaussian_rasterization_depth_alpha"
"${python_bin}" -m pip install --no-build-isolation \
  "${project_dir}/network/styleunet"
"${python_bin}" -m pip install --no-build-isolation \
  "${project_dir}/utils/posevocab_custom_ops"
"${python_bin}" -m pip install --no-build-isolation \
  "${project_dir}/utils/root_finding"

dcvc_dir="${project_dir}/compress_part/codec/DCVC-DC"
cmake -S "${dcvc_dir}/src/cpp" -B "${dcvc_dir}/src/build" \
  -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="$(command -v "${python_bin}")"
cmake --build "${dcvc_dir}/src/build" --config Release -j "$(nproc)"

echo "HGC-Avatar native extensions built successfully."
