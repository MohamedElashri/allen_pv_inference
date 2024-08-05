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

#include "smart_sort.cuh"

namespace SmartSort {

// CPU implementations
#if defined(TARGET_DEVICE_CPU) || !defined(__CUDACC__)

template<typename KeyType>
void segmented_sort(KeyType* keys, const unsigned* seg_offsets, unsigned n_segments)
{
    for (unsigned s = 0; s < n_segments; s++) {
        unsigned start = seg_offsets[s];
        unsigned end = seg_offsets[s + 1];
        std::sort(keys + start, keys + end);
    }
}

template<typename KeyType>
void segmented_sort_with_permutation(const KeyType* keys, const unsigned* seg_offsets, unsigned n_segments, unsigned* permutations)
{
    unsigned n_keys = seg_offsets[n_segments];
    for (unsigned i = 0; i < n_keys; i++) {
        permutations[i] = i;
    }
    for (unsigned s = 0; s < n_segments; s++) {
        unsigned start = seg_offsets[s];
        unsigned end = seg_offsets[s + 1];
        std::sort(
            permutations + start, 
            permutations + end, 
            [&](const auto i, const auto j) { return keys[i] < keys[j]; }
        );
    }
}

#else // GPU implementations

// GPU helper functions
template<typename KeyType>
__device__ void bitonic_sort(KeyType* keys, unsigned size)
{
    for (unsigned k = 2; k <= size; k *= 2) {
        for (unsigned j = k / 2; j > 0; j /= 2) {
            int ixj = threadIdx.x ^ j;
            if (ixj > threadIdx.x) {
                if ((threadIdx.x & k) == 0) {
                    if (keys[threadIdx.x] > keys[ixj]) {
                        KeyType temp = keys[threadIdx.x];
                        keys[threadIdx.x] = keys[ixj];
                        keys[ixj] = temp;
                    }
                } else {
                    if (keys[threadIdx.x] < keys[ixj]) {
                        KeyType temp = keys[threadIdx.x];
                        keys[threadIdx.x] = keys[ixj];
                        keys[ixj] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
}

template<typename KeyType>
__global__ void segmented_sort_kernel(KeyType* keys, const unsigned* seg_offsets, unsigned n_segments)
{
    extern __shared__ KeyType shared_keys[];
    
    unsigned seg = blockIdx.x;
    if (seg >= n_segments) return;

    unsigned start = seg_offsets[seg];
    unsigned size = seg_offsets[seg + 1] - start;

    // Load data into shared memory
    for (unsigned i = threadIdx.x; i < size; i += blockDim.x) {
        shared_keys[i] = keys[start + i];
    }
    __syncthreads();

    // Sort using bitonic sort
    if (threadIdx.x < size) {
        bitonic_sort(shared_keys, size);
    }
    __syncthreads();

    // Store sorted data back to global memory
    for (unsigned i = threadIdx.x; i < size; i += blockDim.x) {
        keys[start + i] = shared_keys[i];
    }
}

template<typename KeyType>
__global__ void segmented_sort_with_permutation_kernel(const KeyType* keys, const unsigned* seg_offsets, unsigned n_segments, unsigned* permutations)
{
    extern __shared__ unsigned shared_permutations[];
    
    unsigned seg = blockIdx.x;
    if (seg >= n_segments) return;

    unsigned start = seg_offsets[seg];
    unsigned size = seg_offsets[seg + 1] - start;

    // Initialize permutations
    for (unsigned i = threadIdx.x; i < size; i += blockDim.x) {
        shared_permutations[i] = start + i;
    }
    __syncthreads();

    // Sort permutations based on keys
    for (unsigned i = 0; i < size; i++) {
        for (unsigned j = threadIdx.x; j < size - 1; j += blockDim.x) {
            if (keys[shared_permutations[j]] > keys[shared_permutations[j + 1]]) {
                unsigned temp = shared_permutations[j];
                shared_permutations[j] = shared_permutations[j + 1];
                shared_permutations[j + 1] = temp;
            }
        }
        __syncthreads();
    }

    // Store sorted permutations back to global memory
    for (unsigned i = threadIdx.x; i < size; i += blockDim.x) {
        permutations[start + i] = shared_permutations[i];
    }
}

// Host functions to launch kernels
template<typename KeyType>
void segmented_sort(KeyType* keys, const unsigned* seg_offsets, unsigned n_segments)
{
    dim3 grid(n_segments);
    dim3 block(256);  // Adjust based in RUN optimization later
    
    size_t shared_mem_size = 256 * sizeof(KeyType);  // Adjust based in RUN optimization later
    
    segmented_sort_kernel<<<grid, block, shared_mem_size>>>(keys, seg_offsets, n_segments);
}

template<typename KeyType>
void segmented_sort_with_permutation(const KeyType* keys, const unsigned* seg_offsets, unsigned n_segments, unsigned* permutations)
{
    dim3 grid(n_segments);
    dim3 block(256);  // Adjust based in RUN optimization later
    
    size_t shared_mem_size = 256 * sizeof(unsigned);  // Adjust based in RUN optimization later
    
    segmented_sort_with_permutation_kernel<<<grid, block, shared_mem_size>>>(keys, seg_offsets, n_segments, permutations);
}

#endif // TARGET_DEVICE_CPU

// Explicit instantiations for common types
template void segmented_sort<int>(int*, const unsigned*, unsigned);
template void segmented_sort<float>(float*, const unsigned*, unsigned);
template void segmented_sort_with_permutation<int>(const int*, const unsigned*, unsigned, unsigned*);
template void segmented_sort_with_permutation<float>(const float*, const unsigned*, unsigned, unsigned*);

} // namespace SmartSort
