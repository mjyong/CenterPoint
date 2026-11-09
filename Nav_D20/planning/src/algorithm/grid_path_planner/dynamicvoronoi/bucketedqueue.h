/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2025-01-07 16:19:00
 * @LastEditors: lq qiang.li@seres.cn
 * @LastEditTime: 2025-08-28 14:10:35
 */
#ifndef _PRIORITYQUEUE2_H_
#define _PRIORITYQUEUE2_H_

#include "src/algorithm/grid_path_planner/dynamicvoronoi/point.h"
#include <assert.h>
#include <map>
#include <queue>
#include <set>
#include <vector>

//! Priority queue for integer coordinates with squared distances as priority.
/** A priority queue that uses buckets to group elements with the same priority.
 *  The individual buckets are unsorted, which increases efficiency if these
 * groups are large. The elements are assumed to be integer coordinates, and the
 * priorities are assumed to be squared Euclidean distances (integers).
 */
template <typename T> class BucketPrioQueue {

public:
  //! Standard constructor
  /** Standard constructor. When called for the first time it creates a look up
   * table that maps square distances to bucket numbers, which might take some
   * time...
   */
  BucketPrioQueue();

  void clear() {
    buckets.clear();
    count = 0;
    nextPop = buckets.end();
  }

  //! Checks whether the Queue is empty
  bool empty();
  //! push an element
  void push(int prio, T t);
  //! return and pop the element with the lowest squared distance */
  T pop();

  int size() { return count; }
  int getNumBuckets() { return buckets.size(); }

  int getTopPriority() { return nextPop->first; }

private:
  int count;

  typedef std::map<int, std::queue<T>> BucketType;
  BucketType buckets;
  typename BucketType::iterator nextPop;
};

#include "src/algorithm/grid_path_planner/dynamicvoronoi/bucketedqueue.hxx"

#endif
