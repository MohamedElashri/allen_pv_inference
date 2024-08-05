/*****************************************************************************\
* (c) Copyright 2018-2020 CERN for the benefit of the LHCb Collaboration      *
*                                                                             *
* This software is distributed under the terms of the Apache License          *
* version 2 (Apache-2.0), copied verbatim in the file "LICENSE".              *
*                                                                             *
* In applying this licence, CERN does not waive the privileges and immunities *
* granted to it by virtue of its status as an Intergovernmental Organization  *
* or submit itself to any jurisdiction.                                       *
\*****************************************************************************/


#pragma once

#include <algorithm>
#include <cstdint>

#ifndef TARGET_DEVICE_CPU
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#endif

namespace SmartSort {

// Forward declarations
template<typename KeyType>
void segmented_sort(KeyType* keys, const unsigned* seg_offsets, unsigned n_segments);

template<typename KeyType>
void segmented_sort_with_permutation(const KeyType* keys, const unsigned* seg_offsets, unsigned n_segments, unsigned* permutations);

#ifndef TARGET_DEVICE_CPU
// GPU-specific helper functions (declared here, defined in .cu file)
template<typename KeyType>
__device__ void bitonic_sort(KeyType* keys, unsigned size);

template<typename KeyType>
__global__ void segmented_sort_kernel(KeyType* keys, const unsigned* seg_offsets, unsigned n_segments);

template<typename KeyType>
__global__ void segmented_sort_with_permutation_kernel(const KeyType* keys, const unsigned* seg_offsets, unsigned n_segments, unsigned* permutations);
#endif

} // namespace SmartSort
