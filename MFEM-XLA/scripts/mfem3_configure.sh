#!/bin/bash
# mfem3_configure.sh <native|cuda|xla>: configure ejwork/mfem_builds/<name> from the current mfem working tree
# (same options as mfem/build_cuda_{llvm,mlir}, plus examples and Google-Benchmark benchmarks).
# xla2 = xla with the plugin and raising library at LLVM b08e0059 (shim-xla2).
# native = plain clang CUDA + ClangEnzyme; cuda = Reactant plugin, MLIR round trip, CUDA backend; xla = Reactant plugin, xla-gpu backend.
E=/home/wmoses/ejwork; MB=$E/mfem_builds; name=$1; CL=/home/wmoses/git/Enzyme/llvm-project/buildRelease/bin/clang++
case $name in
  native) ED=/home/wmoses/git/Enzyme/enzyme/buildRelease ;;
  cuda|xla|xla2) ED=$MB/shim-$name ;;
  *) echo "unknown build $name"; exit 1 ;;
esac
cmake -S /mnt3/wmoses/git/mfem -B $MB/$name -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$CL -DCMAKE_CUDA_COMPILER=$CL -DCMAKE_CUDA_FLAGS=--cuda-path=/usr/local/cuda-12.9 -DCUDA_ARCH=sm_120 \
  -DMFEM_USE_CUDA=ON -DMFEM_USE_MPI=ON -DMFEM_USE_METIS=ON -DMFEM_USE_ENZYME=ON -DMFEM_USE_MEMALLOC=ON -DMFEM_EXPORT_GPU_CONFIG=ON \
  -DHYPRE_INCLUDE_DIRS=/usr/include/hypre -DHYPRE_LIBRARIES=/usr/lib/x86_64-linux-gnu/libHYPRE.so \
  -DMETIS_INCLUDE_DIRS=/usr/include -DMETIS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libmetis.so -DMETIS_VERSION_5=ON \
  -DEnzyme_DIR=$ED -DMFEM_ENABLE_TESTING=ON -DMFEM_ENABLE_EXAMPLES=ON -DMFEM_ENABLE_BENCHMARKS=ON \
  -DMFEM_USE_BENCHMARK=ON -DBENCHMARK_DIR=$E/deps/google-benchmark
