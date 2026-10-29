// Data-parallel loops for the per-point steps of the preprocessing. Every point is computed on its own,
// so the results do not depend on the thread count. A small pool of persistent workers runs the chunks
// (starting threads per call costs more than the loops on a 100k-point sweep); one thread (the default)
// runs everything inline.
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>

namespace dogperc {

void set_num_threads(int n);   // >= 1; the workers are started on the first call that needs them
int num_threads();

namespace detail {
// fn(chunk) for chunk in [0, chunks) on the pool and the calling thread; returns when all are done.
// fn must not throw. Calls from several threads take turns; a parallel_for inside a chunk deadlocks.
void run_chunks(size_t chunks, const std::function<void(size_t)>& fn);
}  // namespace detail

// f(begin, end) over contiguous ranges covering [0, n); ranges of fewer than `grain` items are not split
template <class F>
void parallel_for(size_t n, F&& f, size_t grain = 8192) {
  const size_t chunks = std::min<size_t>(static_cast<size_t>(num_threads()), n / grain);
  if (chunks <= 1) {
    f(size_t(0), n);
    return;
  }
  const size_t step = (n + chunks - 1) / chunks;
  detail::run_chunks(chunks, [&](size_t c) { f(c * step, std::min(n, (c + 1) * step)); });
}

}  // namespace dogperc
