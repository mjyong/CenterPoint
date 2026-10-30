set -e
echo build_begin
cmake -S deploy/cpp -B deploy/cpp/build_cuda -G Ninja -DDOGPERC_WITH_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build deploy/cpp/build_cuda -j16
echo build_ok
echo ctest_begin
DOGPERC_MODEL_DIR=$PWD/data/deploy/nusc_pp_dyn ctest --test-dir deploy/cpp/build_cuda --output-on-failure -V
echo ctest_exit=$?
