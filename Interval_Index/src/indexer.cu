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

#include "indexer.cuh"


/*** This part should be better ****/

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#else
#define DEVICE
#define HOST_DEVICE
#include <cmath>
#endif

/**************************************/



INSTANTIATE_ALGORITHM(interval_indexer::interval_indexer_t) // This is a macro that instantiates the algorithm in the Allen namespace

// Constants 

__host__ void interval_indexer::interval_indexer_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  printf("Entering set_arguments_size function\n");

  // Get the number of events from the arguments on the host
  auto host_number_of_events = first<host_number_of_events_t>(arguments);
  printf("host_number_of_events: %u\n", host_number_of_events);

  // Get the offsets of all velo tracks from the arguments
  auto dev_offsets_all_velo_tracks = data<dev_offsets_all_velo_tracks_t>(arguments);
  printf("dev_offsets_all_velo_tracks pointer: %p\n", (void*)dev_offsets_all_velo_tracks);

  // Check if dev_offsets_all_velo_tracks is valid
  if (dev_offsets_all_velo_tracks == nullptr) {
    printf("Error: dev_offsets_all_velo_tracks is null\n");
    return;
  }

  // Get total_tracks
  unsigned total_tracks = 0;
  #ifdef __CUDACC__
    // GPU version
    cudaError_t cuda_err = cudaMemcpy(&total_tracks, &dev_offsets_all_velo_tracks[host_number_of_events], sizeof(unsigned), cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
      printf("CUDA error when copying total_tracks: %s\n", cudaGetErrorString(cuda_err));
      return;
    }
  #else
    // CPU version
    total_tracks = dev_offsets_all_velo_tracks[host_number_of_events];
  #endif
  printf("total_tracks: %u\n", total_tracks);

  // Check for potential overflow when calculating sizes
  if (total_tracks > (UINT_MAX / 3) || total_tracks > (UINT_MAX / 6)) {
    printf("Error: Potential integer overflow when calculating sizes\n");
    return;
  }

  // Set the size of the intervals array
  printf("Setting size for dev_intervals_t: %u\n", total_tracks * 3);
  set_size<dev_intervals_t>(arguments, total_tracks * 3);

  // Set the size of the interval counts array
  printf("Setting size for dev_interval_counts_t: %u\n", total_tracks);
  set_size<dev_interval_counts_t>(arguments, total_tracks);

  // Set the size of the sorted track indices array
  printf("Setting size for dev_sorted_track_indices_t: %u\n", total_tracks);
  set_size<dev_sorted_track_indices_t>(arguments, total_tracks);

  // Set the size of the ellipsoid parameters array
  printf("Setting size for dev_ellipsoid_params_t: %u\n", total_tracks * 6);
  set_size<dev_ellipsoid_params_t>(arguments, total_tracks * 6);

  // Set the size of the z_pocas array
  printf("Setting size for dev_z_pocas_t: %u\n", total_tracks);
  set_size<dev_z_pocas_t>(arguments, total_tracks);

  // Set the size of the track indices array
  printf("Setting size for dev_track_indices_t: %u\n", total_tracks);
  set_size<dev_track_indices_t>(arguments, total_tracks);

  printf("Exiting set_arguments_size function\n");
}

// Main function for the Algorithm
void interval_indexer::interval_indexer_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  global_function(interval_indexer)(
    dim3(first<host_number_of_events_t>(arguments)), dim3(256), context)(arguments); // Call the global function with the number of events and the block size
}

// Function to assign intervals based on z_poca
DEVICE void interval_indexer::assign_intervals(float z_poca, int* intervals, int& num_intervals)
{
    // Shift z_poca to [0, 400] range for easier interval calculation
    z_poca += 100.0f;

    // Handle edge cases: z_poca outside [0, 400] range
    if (z_poca < 0.0f || z_poca > 400.0f) { // Check if z_poca is outside the range [0, 200]
        num_intervals = 1;
        intervals[0] = z_poca < 0.0f ? 0 : 39; // Assign to first or last interval
        return;
    }

    // Calculate base interval
    float interval_float = z_poca / 10.0f;
    int base_interval = floorf(interval_float);
    
    num_intervals = 0;

    // Check if z_poca is within OVERLAP_WIDTH of lower boundary
    if (interval_float - base_interval <= OVERLAP_WIDTH && base_interval > 0) {
        intervals[num_intervals++] = base_interval - 1;
    }

    // Always add the base interval
    intervals[num_intervals++] = base_interval;

    // Check if z_poca is within OVERLAP_WIDTH of upper boundary
    if (base_interval + 1 - interval_float <= OVERLAP_WIDTH && base_interval < 39) {
        intervals[num_intervals++] = base_interval + 1;
    }

    // Ensure all intervals are within [0, 39]
    for (int i = 0; i < num_intervals; ++i) {
        intervals[i] = min(39, max(0, intervals[i]));
    }
}

// Bubble sort implementation for GPU
// Todo: Replace with a more efficient sorting algorithm
DEVICE void interval_indexer::bubble_sort(float* z_pocas, int* indices, int n)
{
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (z_pocas[j] > z_pocas[j + 1]) {
                // Swap z_pocas
                float temp_z = z_pocas[j];
                z_pocas[j] = z_pocas[j + 1];
                z_pocas[j + 1] = temp_z;
                // Swap corresponding indices
                int temp_index = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp_index;
            }
        }
    }
}

//// ********************************************************************************************************************  ////
//// I use HOST_DEVICE not DEVICE for now because I want to be able to call this function from the host for debugging      ////
//// Todo: change to DEVICE when done with debugging                                                                       ////
//// ********************************************************************************************************************  ////

// Function to calculate ellipsoid parameters from track state
HOST_DEVICE void interval_indexer::calculate_ellipsoid_params(
    float x, float y, float z, float tx, float ty,
    float c00, float c20, float c22, float c11, float c31, float c33,
    float* ellipsoid_params, int event_number, int track_index) // Add event_number and track_index for debugging (remove later - just need it now in this scope)
{

    /*
    Mathematical background:
    
    1. The equation of an ellipsoid is: Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz = 1
    
    2. In statistical terms, this is related to the Mahalanobis distance:
       (X - μ)^T Σ^(-1) (X - μ) = 1
       where X is the position vector, μ is the mean (center of ellipsoid),
       and Σ is the covariance matrix.
    
    3. The inverse of the covariance matrix Σ^(-1) directly gives us the 
       coefficients of the ellipsoid equation:
       
       | A   D   E |
       | D   B   F | = Σ^(-1)
       | E   F   C |
    
    4. In our case, the covariance matrix is already calculated at the point 
       of closest approach (POCA), so μ is effectively (0,0,0) in this coordinate system.
    
    Assumptions:
    1. The covariance matrix elements provided (c00, c20, c22, c11, c31, c33) 
       are calculated at the POCA.
    2. The coordinate system is centered at the POCA, so we don't need to use x, y, z 
       in our calculations here.
    3. The covariance matrix is symmetric, so we can fill in the missing elements.
    */


    // Construct the covariance matrix
    float cov[3][3] = {
        {c00, c20, c31},
        {c20, c11, c31},
        {c31, c31, c33}
    };

    // Add a small regularization term to improve numerical stability
    const float epsilon = 1e-8f;
    for (int i = 0; i < 3; ++i) {
        cov[i][i] += epsilon;
    }

    // Invert the matrix using Gauss-Jordan elimination
    float inv[3][3] = {{1,0,0}, {0,1,0}, {0,0,1}}; // Start with identity matrix

    // Perform Gauss-Jordan elimination
    for (int i = 0; i < 3; ++i) {
        // Find the pivot
        int pivot = i;
        for (int j = i + 1; j < 3; ++j) {
            if (fabsf(cov[j][i]) > fabsf(cov[pivot][i])) {
                pivot = j;
            }
        }

        // Swap rows if necessary
        if (pivot != i) {
            for (int j = 0; j < 3; ++j) {
                float temp = cov[i][j];
                cov[i][j] = cov[pivot][j];
                cov[pivot][j] = temp;

                temp = inv[i][j];
                inv[i][j] = inv[pivot][j];
                inv[pivot][j] = temp;
            }
        }

        // Scale row to have a 1 on the diagonal
        float scale = 1.0f / cov[i][i];
        for (int j = 0; j < 3; ++j) {
            cov[i][j] *= scale;
            inv[i][j] *= scale;
        }

        // Subtract this row from all other rows
        for (int k = 0; k < 3; ++k) {
            if (k != i) {
                float factor = cov[k][i];
                for (int j = 0; j < 3; ++j) {
                    cov[k][j] -= factor * cov[i][j];
                    inv[k][j] -= factor * inv[i][j];
                }
            }
        }
    }

    // Assign the results to ellipsoid parameters
    ellipsoid_params[0] = inv[0][0];
    ellipsoid_params[1] = inv[1][1];
    ellipsoid_params[2] = inv[2][2];
    ellipsoid_params[3] = inv[0][1];
    ellipsoid_params[4] = inv[0][2];
    ellipsoid_params[5] = inv[1][2];

    // Debug output:
    printf("Debug: Covariance matrix:\n");
    for (int i = 0; i < 3; ++i) {
        printf("%f %f %f\n", (double)cov[i][0], (double)cov[i][1], (double)cov[i][2]);
    }
    printf("Debug: Inverse matrix (Ellipsoid parameters):\n");
    for (int i = 0; i < 3; ++i) {
        printf("%f %f %f\n", (double)inv[i][0], (double)inv[i][1], (double)inv[i][2]);
    }

    // Dev output:
    // Print A-F values for each track
    printf("Event %d, Track %d: A=%f, B=%f, C=%f, D=%f, E=%f, F=%f\n",
        event_number, track_index, 
        (double)ellipsoid_params[0], (double)ellipsoid_params[1], (double)ellipsoid_params[2],
        (double)ellipsoid_params[3], (double)ellipsoid_params[4], (double)ellipsoid_params[5]);

    // Note: x, y, z, tx, ty are currently unused but kept for potential future use
    // Todo: Check if these values are needed for the ellipsoid calculation

}

// Kernel function for interval indexing and ellipsoid parameter calculation
__global__ void interval_indexer::interval_indexer(Parameters parameters)
{  
    printf("Entering interval_indexer kernel\n");

    // Get the current event number from the block index
    const unsigned event_number = blockIdx.x;
    printf("event_number: %u\n", event_number);

    // Get the total number of events from the parameters
    const unsigned number_of_events = parameters.dev_number_of_events[0];
    printf("number_of_events: %u\n", number_of_events);

    // Early exit if this block is beyond the number of events
    if (event_number >= number_of_events) {
        printf("Early exit: event_number (%u) >= number_of_events (%u)\n", event_number, number_of_events);
        return;
    }

    // Get views for VELO tracks and states for the current event
    const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];
    const auto velo_states_view = parameters.dev_velo_states_view[event_number];
    // Get the number of tracks in this event
    const unsigned num_tracks = velo_tracks_view.size();
    // Calculate the offset for this event's tracks in global arrays
    const unsigned event_track_offset = velo_tracks_view.offset();

    // Step 1 & 2: Read tracks and prepare for sorting
    // Each thread processes tracks in a strided manner
    for (unsigned i = threadIdx.x; i < num_tracks; i += blockDim.x) {
        const auto track = velo_tracks_view.track(i);
        const auto state = velo_states_view.state(track.track_index());
        // Store z_poca (z-coordinate of point of closest approach) for sorting
        parameters.dev_z_pocas[event_track_offset + i] = state.z();
        // Store original track indices for sorting
        parameters.dev_track_indices[event_track_offset + i] = i;
    }
    // Synchronize threads to ensure all data is written before sorting
    __syncthreads();

    // Sort tracks based on z_poca
    // Only the first thread in the block performs sorting to avoid race conditions
    if (threadIdx.x == 0) {
        bubble_sort(&parameters.dev_z_pocas[event_track_offset], 
                    &parameters.dev_track_indices[event_track_offset], 
                    num_tracks);
    }
    // Wait for sorting to complete before proceeding
    __syncthreads();

    // Step 3, 4, 5: Assign intervals, read state info, and calculate ellipsoid parameters    
    for (unsigned i = threadIdx.x; i < num_tracks; i += blockDim.x) {
        // Get the original track index from the sorted array
        int track_index = parameters.dev_track_indices[event_track_offset + i];
        const auto track = velo_tracks_view.track(track_index);
        const auto state = velo_states_view.state(track.track_index());

        // Get z_poca for interval assignment
        float z_poca = state.z();
        int intervals[3];
        int num_intervals;
        // Assign intervals based on z_poca
        assign_intervals(z_poca, intervals, num_intervals);

        // Calculate output index for this track
        unsigned output_index = event_track_offset + track_index;
        // Store the number of intervals this track belongs to
        parameters.dev_interval_counts[output_index] = num_intervals;
        // Store the interval indices for this track
        for (int j = 0; j < num_intervals; ++j) {
            parameters.dev_intervals[output_index * 3 + j] = intervals[j];
        }

        // Store track index in sorted order for later use
        parameters.dev_sorted_track_indices[output_index] = track_index;

        // Calculate ellipsoid parameters and store them
        float ellipsoid_params[6];
        calculate_ellipsoid_params(
            state.x(), state.y(), state.z(), state.tx(), state.ty(),
            state.c00(), state.c20(), state.c22(), state.c11(), state.c31(), state.c33(),
            ellipsoid_params, event_number, track_index); // Add event_number and track_index for debugging (remove later - just need it now in this scope)

        // Store ellipsoid parameters for each track in an array containing 6 values
        for (int j = 0; j < 6; ++j) {
            parameters.dev_ellipsoid_params[output_index * 6 + j] = ellipsoid_params[j];
        }
        
    }
}
