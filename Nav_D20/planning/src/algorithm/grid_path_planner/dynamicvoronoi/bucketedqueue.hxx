/*
 * @Author: tian.zhao tian.zhao@seres.cn
 * @Date: 2025-01-07 16:19:00
 * @LastEditors: tian.zhao tian.zhao@seres.cn
 * @LastEditTime: 2025-01-07 16:30:48
 */
#include "limits.h"
#include "src/algorithm/grid_path_planner/dynamicvoronoi/bucketedqueue.h"
#include <stdio.h>
#include <stdlib.h>

template <class T> BucketPrioQueue<T>::BucketPrioQueue() { clear(); }

template <class T> bool BucketPrioQueue<T>::empty() { return (count == 0); }

template <class T> void BucketPrioQueue<T>::push(int prio, T t) {
  buckets[prio].push(t);
  if (nextPop == buckets.end() || prio < nextPop->first)
    nextPop = buckets.find(prio);
  count++;
}

template <class T> T BucketPrioQueue<T>::pop() {
  while (nextPop != buckets.end() && nextPop->second.empty())
    ++nextPop;

  T p = nextPop->second.front();
  nextPop->second.pop();
  if (nextPop->second.empty()) {
    typename BucketType::iterator it = nextPop;
    nextPop++;
    buckets.erase(it);
  }
  count--;
  return p;
}
