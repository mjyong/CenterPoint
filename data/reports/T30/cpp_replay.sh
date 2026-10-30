set -e
N=data/deploy/T30
M=data/deploy/nusc_pp_dyn
echo "=== cpp_dets ==="
deploy/cpp/build_cuda/dogperc_replay --messages $N/msgs.npz --config $N/kygi.yaml \
  --detections $N/py_dets.npz --out $N/cpp_dets.npz
echo "=== cpp_gpu ==="
deploy/cpp/build_cuda/dogperc_replay --messages $N/msgs.npz --config $N/kygi.yaml \
  --model-dir $M --precision fp16 --gpu --out $N/cpp.npz
echo replay_ok
