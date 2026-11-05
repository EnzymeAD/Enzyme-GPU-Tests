#!/bin/bash
# Build runtime paper58: as paper57, with the element-chunk barrier form in uneven chunks (rextra-src 90c01fcb7).
E=/home/wmoses/ejwork
cd /home/wmoses/git/Reactant.jl/deps/ReactantExtra && nice -n 19 bazel build --jobs=32 --config=cuda12 --repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_60,sm_70,sm_80,sm_90,sm_100,compute_120" --check_visibility=false --repo_env=HERMETIC_PYTHON_VERSION=3.11 --copt=-Wno-error=unused-command-line-argument --host_copt=-Wno-error=unused-command-line-argument --override_repository=enzyme_ad=$E/wt/rextra-src //:libReactantExtra.so > $E/logs/build_rextra_paper58.log 2>&1 || { echo PAPER58-BUILD-FAILED; exit 1; }
B=/mnt3/wmoses/.cache/bazel/_bazel_wmoses/938daab9a72fc0d5443aa9a534262d60/execroot/__main__/bazel-out/k8-opt/bin
rm -f $B/libReactantExtra_paper58.so; cp $B/libReactantExtra.so $B/libReactantExtra_paper58.so && mkdir -p $E/rt/paper58 && ln -sf $B/libReactantExtra_paper58.so $E/rt/paper58/libReactantExtra_stable.so && ln -sfn $B/_solib_local $E/rt/paper58/_solib_local
cd $E
echo PAPER58-BUILT
