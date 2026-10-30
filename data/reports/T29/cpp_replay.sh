set -e
N=data/deploy/T29
M=data/deploy/nusc_pp_dyn
for n in rosbag2_2026_09_21-16_41_27_moving rosbag2_2026_09_21-16_41_27; do
  echo "=== cpp_dets $n ==="
  deploy/cpp/build_cuda/dogperc_replay --messages $N/$n/msgs.npz --config $N/$n/kygi.yaml \
    --detections $N/$n/py_dets.npz --out $N/$n/cpp_dets.npz
  echo "=== cpp_gpu $n ==="
  deploy/cpp/build_cuda/dogperc_replay --messages $N/$n/msgs.npz --config $N/$n/kygi.yaml \
    --model-dir $M --precision fp16 --gpu --out $N/$n/cpp.npz
done
echo replay_ok
