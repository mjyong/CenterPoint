#include "dogperc/detection/decode.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace dogperc {

namespace {

int key_index(const ModelMeta& m, const std::string& key) {
  for (size_t i = 0; i < m.head_keys.size(); ++i)
    if (m.head_keys[i] == key) return static_cast<int>(i);
  return -1;
}

void decode_task(const std::vector<HeadMap>& maps, const ModelMeta& meta, int task, int label_offset, RawBoxes& out) {
  const DecoderConfig& d = meta.decoder;
  const int i_hm = key_index(meta, "hm"), i_reg = key_index(meta, "reg"), i_h = key_index(meta, "height");
  const int i_dim = key_index(meta, "dim"), i_rot = key_index(meta, "rot"), i_vel = key_index(meta, "vel");
  if (i_hm < 0 || i_reg < 0 || i_h < 0 || i_dim < 0 || i_rot < 0)
    throw Error("decode: the model needs reg, height, dim, rot and hm heads");
  if (i_vel < 0) throw Error("decode: the deployed models have a velocity head; this one has none");
  const HeadMap& hm = maps[i_hm];
  const int C = hm.channels, W = hm.width, HW = hm.height * hm.width;
  // sigmoid of the clipped logits, float32
  std::vector<float> scores(HW);
  std::vector<int> labels(HW);
  for (int i = 0; i < HW; ++i) {
    float best = 0.0f;
    int arg = 0;
    for (int c = 0; c < C; ++c) {
      const float x = std::min(std::max(hm.at(c, i), -60.0f), 60.0f);
      const float s = 1.0f / (1.0f + std::exp(-x));
      if (c == 0 || s > best) {
        best = s;
        arg = c;
      }
    }
    scores[i] = best;
    labels[i] = arg;
  }
  const float thr = static_cast<float>(d.score_threshold);
  std::vector<int> cand;
  for (int i = 0; i < HW; ++i)
    if (scores[i] > thr) cand.push_back(i);
  if (d.pre_max_size && static_cast<int>(cand.size()) > d.pre_max_size) {
    std::vector<int> order(cand.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return -scores[cand[a]] < -scores[cand[b]]; });
    std::vector<int> top;
    for (int k = 0; k < d.pre_max_size; ++k) top.push_back(cand[order[k]]);
    cand = top;
  }
  const double lo[3] = {d.post_center_limit_range[0], d.post_center_limit_range[1], d.post_center_limit_range[2]};
  const double hi[3] = {d.post_center_limit_range[3], d.post_center_limit_range[4], d.post_center_limit_range[5]};
  std::vector<std::array<double, 9>> boxes;
  std::vector<double> sc;
  std::vector<int> lb;
  for (int i : cand) {
    const int ys = i / W, xs = i % W;
    // int64 + float32 -> float64 in NumPy
    const double x = ((static_cast<double>(xs) + static_cast<double>(maps[i_reg].at(0, i))) * d.out_size_factor) * d.voxel_size[0] + d.pc_range[0];
    const double y = ((static_cast<double>(ys) + static_cast<double>(maps[i_reg].at(1, i))) * d.out_size_factor) * d.voxel_size[1] + d.pc_range[1];
    const float r = std::atan2(maps[i_rot].at(0, i), maps[i_rot].at(1, i));
    std::array<double, 9> b = {x, y, maps[i_h].at(0, i), std::exp(maps[i_dim].at(0, i)), std::exp(maps[i_dim].at(1, i)),
                               std::exp(maps[i_dim].at(2, i)), maps[i_vel].at(0, i), maps[i_vel].at(1, i), r};
    if (!(b[0] >= lo[0] && b[1] >= lo[1] && b[2] >= lo[2] && b[0] <= hi[0] && b[1] <= hi[1] && b[2] <= hi[2])) continue;
    boxes.push_back(b);
    sc.push_back(scores[i]);
    lb.push_back(labels[i]);
  }
  std::vector<std::array<double, 2>> xy;
  for (const auto& b : boxes) xy.push_back({b[0], b[1]});
  std::vector<int> keep = circle_nms(xy, sc, std::sqrt(d.min_radius[task]));
  if (static_cast<int>(keep.size()) > d.nms_post_max_size) keep.resize(d.nms_post_max_size);
  for (int k : keep) {
    out.box9.push_back(boxes[k]);
    out.scores.push_back(sc[k]);
    out.labels.push_back(lb[k] + label_offset);
  }
}

}  // namespace

TaskMaps split_head_outputs(const std::vector<std::vector<float>>& outputs, const ModelMeta& meta) {
  const int K = static_cast<int>(meta.head_keys.size());
  if (static_cast<int>(outputs.size()) != meta.num_tasks() * K) throw Error("split_head_outputs: wrong number of maps");
  TaskMaps maps(meta.num_tasks());
  const int H = meta.feature_y(), W = meta.feature_x();
  for (int t = 0; t < meta.num_tasks(); ++t)
    for (int k = 0; k < K; ++k) {
      HeadMap m;
      m.channels = meta.head_channels(t, meta.head_keys[k]);
      m.height = H;
      m.width = W;
      m.data = outputs[t * K + k];
      if (m.data.size() != static_cast<size_t>(m.channels) * H * W)
        throw Error("split_head_outputs: " + meta.rpn_outputs[t * K + k] + " has the wrong size");
      maps[t].push_back(std::move(m));
    }
  return maps;
}

RawBoxes decode_centerpoint(const TaskMaps& maps, const ModelMeta& meta) {
  RawBoxes out;
  int offset = 0;
  const auto nc = meta.num_classes();
  for (int t = 0; t < meta.num_tasks(); ++t) {
    decode_task(maps[t], meta, t, offset, out);
    offset += nc[t];
  }
  return out;
}

Detections postprocess(const RawBoxes& raw, const ModelMeta& meta, double stamp) {
  Detections d;
  d.frame = "det";
  d.stamp = stamp;
  for (size_t i = 0; i < raw.scores.size(); ++i) {
    const int ml = raw.labels[i];
    const int label = meta.label_map.at(ml);
    if (label < 0) continue;
    Box b;
    std::array<double, 2> v;
    det3d_to_standard(raw.box9[i].data(), b, v);
    meta.frame.box_from_model(b, v);
    d.push(b, v, raw.scores[i], label, meta.subclass_map.at(ml));
  }
  return classwise_circle_nms(d, meta.merged_nms);
}

}  // namespace dogperc
