#!/bin/bash
# mfem3_build2.sh <name> <jobs> [fallback-file]: like mfem3_build.sh, with the mfem3_launch.sh compiler launcher
# (fallback list + retry-once). Log logs/mfem3_build2_<name>.log, launcher log logs/mfem3_launch_<name>.log.
E=/home/wmoses/ejwork; n=$1; j=$2; fb=${3:-/dev/null}; L=$E/logs/mfem3_build2_$n.log
cmake -S /mnt3/wmoses/git/mfem -B $E/mfem_builds/$n -DCMAKE_CUDA_COMPILER_LAUNCHER=$E/scripts/mfem3_launch.sh -DCMAKE_CXX_COMPILER_LAUNCHER=$E/scripts/mfem3_launch.sh > $L 2>&1 || { echo "MFEM3-BUILD-DONE $n cmake failed" >> $L; exit 1; }
start=$(date +%s)
extra=(); [ "$n" = cuda ] && extra=(PATH=/usr/local/cuda-12.9/bin:$PATH POLYGEIST_GPU_KERNEL_BLOCK_SIZE=128)
env -u EXPORT_REACTANT "${extra[@]}" MFEM3_FALLBACK=$fb MFEM3_LAUNCH_LOG=$E/logs/mfem3_launch_$n.log \
  REACTANT_LOWER_INVOKE=0 REACTANT_SPECIALIZE_INDEX_STRIDES=1 REACTANT_SPECIALIZE_INT_SCALARS=1 \
  REACTANT_UNROLL_MAX_TRIP_COUNT=0 REACTANT_UNROLL_MAX_NESTED_OPS=4096 \
  ninja -C $E/mfem_builds/$n -k 0 -j $j mfem examples benchmarks >> $L 2>&1 || { echo "MFEM3-BUILD-DONE $n cmake failed" >> $L; exit 1; }
rc=$?
echo "MFEM3-BUILD-DONE $n rc=$rc secs=$(( $(date +%s) - start )) failed=$(grep -c '^FAILED:' $L)" >> $L
