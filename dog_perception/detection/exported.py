"""CenterPoint-Pillar from its two exported graphs, without det3d or PyTorch.

Same call and output as ``CenterPointDetector`` (crop, model frame, voxelize, PFN, scatter, RPN +
heads, decode, class mapping, merged NMS, geometric filters), with the networks run by

* ONNX Runtime (``runtime="onnx"``: any machine, the reference for the engines), or
* TensorRT engines (``runtime="trt"``: built on the machine that runs them, the server's GPU or
  the Orin, ``tools/deploy/build_engines.py``).

    det = ExportedPillarDetector("data/deploy/nusc_pp", runtime="trt", precision="fp16")
    dets = det(points_det, stamp)

The directory holds what ``tools/dog/export_pillar_onnx.py`` writes (``model_meta.json``,
``pfn.onnx``, ``rpn_head.onnx``) and the engines, named ``<graph>.<precision>.engine``
(``pfn.fp16.engine``, ``rpn_head.int8.engine``, ...). Everything here is the host reference of the
C++ / CUDA pipeline (``deploy/cpp``); its timings are not the deployed ones.
"""
import ctypes
import ctypes.util
import glob
import os
import time

import numpy as np

from .boxes import CLASSES, Detections
from .decode import decode_centerpoint
from .filters import DetectionFilterConfig, filter_detections
from .model_meta import ModelMeta, postprocess
from .pillar_host import decorate_pillars, pad_pillars, scatter_pillars, split_head_outputs
from .voxelize import voxelize


class OnnxRunner:
    """One ONNX graph in ONNX Runtime: ``runner({input: array}) -> {output: array}``."""

    def __init__(self, path, providers=None):
        import onnxruntime as ort
        self.path = path
        self.session = ort.InferenceSession(path, providers=providers or ["CPUExecutionProvider"])
        self.outputs = [o.name for o in self.session.get_outputs()]

    def __call__(self, feeds):
        return dict(zip(self.outputs, self.session.run(None, feeds)))


class _CudaRuntime:
    """The few CUDA runtime calls the engine runner needs, through ctypes (no pycuda / cuda-python)."""

    H2D, D2H = 1, 2

    def __init__(self):
        self.lib = None
        for name in self._candidates():
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            raise OSError("libcudart not found; set DOGPERC_CUDART to its path")
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    @staticmethod
    def _candidates():
        if os.environ.get("DOGPERC_CUDART"):
            yield os.environ["DOGPERC_CUDART"]
        for name in ("libcudart.so", "libcudart.so.12", ctypes.util.find_library("cudart")):
            if name:
                yield name
        for pattern in ("/usr/local/cuda/lib64/libcudart.so*", "/usr/local/cuda/targets/*/lib/libcudart.so*"):
            yield from sorted(glob.glob(pattern))
        try:                                           # pip's nvidia-cuda-runtime-cu12 (next to pip TensorRT)
            import nvidia.cuda_runtime as cr
            yield from sorted(glob.glob(os.path.join(os.path.dirname(cr.__file__), "lib", "libcudart.so*")))
        except ImportError:
            pass

    def check(self, err, what):
        if err != 0:
            raise RuntimeError("%s: CUDA error %d (%s)" % (what, err, self.lib.cudaGetErrorString(err).decode()))

    def malloc(self, nbytes):
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(max(nbytes, 1))), "cudaMalloc")
        return ptr.value

    def free(self, ptr):
        self.lib.cudaFree(ctypes.c_void_p(ptr))

    def h2d(self, ptr, array):
        self.check(self.lib.cudaMemcpy(ctypes.c_void_p(ptr), array.ctypes.data_as(ctypes.c_void_p),
                                       ctypes.c_size_t(array.nbytes), self.H2D), "cudaMemcpy H2D")

    def d2h(self, array, ptr):
        self.check(self.lib.cudaMemcpy(array.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ptr),
                                       ctypes.c_size_t(array.nbytes), self.D2H), "cudaMemcpy D2H")

    def stream(self):
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s.value

    def sync(self, stream):
        self.check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize")


class TrtRunner:
    """One TensorRT engine (TensorRT 10 API, static shapes): ``runner({input: array}) -> {output: array}``.
    Device buffers are allocated once; inputs and outputs are copied synchronously."""

    def __init__(self, path):
        import tensorrt as trt
        self.path = path
        self.cuda = _CudaRuntime()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("%s: TensorRT could not load the engine (built with another version / GPU?)" % path)
        self.context = self.engine.create_execution_context()
        self.stream = self.cuda.stream()
        self.inputs, self.outputs, self.host, self.device = [], [], {}, {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise ValueError("%s: tensor %s has a dynamic shape %s; export static shapes" % (path, name, shape))
            arr = np.zeros(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            self.host[name], self.device[name] = arr, self.cuda.malloc(arr.nbytes)
            self.context.set_tensor_address(name, self.device[name])
            (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(name)

    def __call__(self, feeds):
        for name in self.inputs:
            self.cuda.h2d(self.device[name], np.ascontiguousarray(feeds[name], dtype=self.host[name].dtype))
        if not self.context.execute_async_v3(self.stream):
            raise RuntimeError("%s: execute_async_v3 failed" % self.path)
        self.cuda.sync(self.stream)
        out = {}
        for name in self.outputs:
            self.cuda.d2h(self.host[name], self.device[name])
            out[name] = self.host[name].astype(np.float32, copy=True)
        return out

    def __del__(self):
        for ptr in getattr(self, "device", {}).values():
            try:
                self.cuda.free(ptr)
            except Exception:
                pass


def graph_file(model_dir, graph, runtime, precision):
    """Path of a graph: ``<graph>.onnx`` for ONNX Runtime, ``<graph>.<precision>.engine`` for TensorRT."""
    return os.path.join(model_dir, graph + (".onnx" if runtime == "onnx" else ".%s.engine" % precision))


class ExportedPillarDetector:
    def __init__(self, model_dir, runtime="onnx", precision="fp16", pfn=None, rpn=None, filter_cfg="default",
                 keep_stages=False):
        """``pfn`` / ``rpn``: graph files (relative to ``model_dir``) instead of the default names,
        e.g. an FP16 PFN with an INT8 RPN. ``keep_stages``: keep the last frame's intermediate tensors
        in ``last_stages`` (tools/deploy/compare_backends.py)."""
        self.meta = ModelMeta.load(os.path.join(model_dir, "model_meta.json"))
        make = OnnxRunner if runtime == "onnx" else TrtRunner
        self.pfn = make(os.path.join(model_dir, pfn) if pfn else graph_file(model_dir, "pfn", runtime, precision))
        self.rpn = make(os.path.join(model_dir, rpn) if rpn else graph_file(model_dir, "rpn_head", runtime, precision))
        self.runtime, self.precision = runtime, precision
        self.filter_cfg = DetectionFilterConfig() if filter_cfg == "default" else filter_cfg
        self.frame = self.meta.model_frame
        self.z_range = tuple(self.meta.z_range)
        self.keep_stages = keep_stages
        self.tta = False
        self.last_timing, self.last_raw, self.last_raw_count, self.last_stages = {}, None, 0, None
        self.last_pillars = (0, 0)               # (used, total): total > max_pillars means some were dropped

    def _crop(self, points_det):
        z = points_det[:, 2]
        return points_det[(z >= self.z_range[0]) & (z <= self.z_range[1])]

    def network(self, points_model):
        """Model-frame points -> per-task head maps, and the intermediate tensors."""
        m = self.meta
        tv = time.perf_counter()
        pts = points_model[:, :m.num_point_features].astype(np.float32)
        v, c, n = voxelize(pts, m.voxel_size, m.pc_range, m.max_points, m.max_voxels)
        feats = decorate_pillars(v, n, c, m.voxel_size, m.pc_range)
        x, cpad, nv = pad_pillars(feats, c, m.max_pillars)
        self.last_pillars = (nv, len(v))
        t0 = time.perf_counter()
        pf = self.pfn({m.onnx["pfn"]["input"]: x})[m.onnx["pfn"]["output"]][0, :, :, 0]
        t1 = time.perf_counter()
        canvas = scatter_pillars(pf, cpad, nv, m.grid_size)
        t2 = time.perf_counter()
        outs = self.rpn({m.onnx["rpn"]["input"]: canvas})
        t3 = time.perf_counter()
        flat = [outs[name] for name in m.rpn_outputs]
        maps = split_head_outputs(flat, m.head_keys, len(m.tasks))
        stages = dict(voxels=v, coords=c, num_points=n, pfn_in=x, pfn_out=pf[:, :nv], bev=canvas, heads=flat)
        return maps, stages, dict(voxelize_ms=1e3 * (t0 - tv), pfn_ms=1e3 * (t1 - t0), scatter_ms=1e3 * (t2 - t1),
                                  rpn_ms=1e3 * (t3 - t2))

    def __call__(self, points_det, stamp=None):
        t0 = time.perf_counter()
        pts = self.frame.points_to_model(self._crop(points_det))
        if len(pts) == 0:
            self.last_raw = Detections(stamp=stamp)
            return Detections(stamp=stamp)
        t1 = time.perf_counter()
        maps, stages, ms = self.network(pts)
        t2 = time.perf_counter()
        m = self.meta
        box9, scores, labels = decode_centerpoint(maps, m.num_classes, m.test_cfg)
        dets = postprocess(box9, scores, labels, m.label_map, m.subclass_map, self.frame, m.merged_nms, stamp)
        self.last_raw_count = len(dets)
        self.last_raw = dets
        if self.filter_cfg is not None:
            dets = filter_detections(dets, points_det, self.filter_cfg)
        t3 = time.perf_counter()
        if self.keep_stages:
            stages.update(box9=box9, scores=scores, labels=labels)
            self.last_stages = stages
        net = ms["pfn_ms"] + ms["scatter_ms"] + ms["rpn_ms"]
        self.last_timing = dict(voxelize_ms=1e3 * (t1 - t0) + ms["voxelize_ms"], network_ms=net,
                                post_ms=1e3 * (t3 - t2), pfn_ms=ms["pfn_ms"], scatter_ms=ms["scatter_ms"],
                                rpn_ms=ms["rpn_ms"])
        return dets

    @property
    def classes(self):
        return CLASSES
