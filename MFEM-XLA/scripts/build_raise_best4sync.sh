#!/bin/bash
# build_raise_best4sync.sh: libRaise at tmp/best4 (the commit behind so/libRaise_best4.so, which built mfem_builds/xla2) plus
# Enzyme-JAX #3641 (cudaDeviceSynchronize -> reactantXLASync on xla targets), built in the warm wt/sliceconcat tree on
# local branch tmp/best4sync -> so/libRaise_best4sync.so. Log logs/build_best4sync.log, marker RAISE-BEST4SYNC-BUILT.
E=/home/wmoses/ejwork; L=$E/logs/build_best4sync.log; cd $E/wt/sliceconcat || exit 1
prev=$(git branch --show-current); echo "prev branch $prev" > $L
git checkout -q -B tmp/best4sync tmp/best4 >> $L 2>&1 || { echo CHECKOUT-FAILED >> $L; exit 1; }
git apply $E/scratch/sync_lowering.diff >> $L 2>&1 || { echo PATCH-FAILED >> $L; exit 1; }
git commit -qam 'xla targets: lower cudaDeviceSynchronize to reactantXLASync (best4 + #3641)' >> $L 2>&1
nice -n 19 bazel build --jobs=16 //:libRaise.so >> $L 2>&1; rc=$?; echo "BUILD-RC $rc" >> $L
[ $rc -eq 0 ] && cp bazel-bin/libRaise.so $E/so/libRaise_best4sync.so && echo RAISE-BEST4SYNC-BUILT >> $L
