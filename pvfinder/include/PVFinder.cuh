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

#include "AlgorithmTypes.cuh"
#include "States.cuh"
#include "VeloConsolidated.cuh"

namespace PVFinder {
  constexpr float OVERLAP_WIDTH = 0.05f;
  constexpr unsigned MAX_LAYERS = 6;
  constexpr unsigned MAX_LAYER_SIZE = 100;

  struct BeamLine {
    __host__ __device__ float x() const { return 0.f; }
    __host__ __device__ float y() const { return 0.f; }
    __host__ __device__ float z() const { return 0.f; }
    __host__ __device__ float tx() const { return 0.f; }
    __host__ __device__ float ty() const { return 0.f; }
  };

  struct Parameters {
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_velo_states_view_t, Allen::Views::Physics::KalmanStates) dev_velo_states_view;
    DEVICE_OUTPUT(dev_pv_positions_t, float) dev_pv_positions;
    DEVICE_OUTPUT(dev_pv_number_t, unsigned) dev_pv_number;
    DEVICE_OUTPUT(dev_output_histogram_t, float) dev_output_histogram;
    DEVICE_OUTPUT(dev_layer_weights_t, float) dev_layer_weights;
    DEVICE_OUTPUT(dev_layer_biases_t, float) dev_layer_biases;
    DEVICE_OUTPUT(dev_layer_sizes_t, unsigned) dev_layer_sizes;
    DEVICE_OUTPUT(dev_input_mean_t, float) dev_input_mean;
    DEVICE_OUTPUT(dev_input_std_t, float) dev_input_std;
    DEVICE_OUTPUT(dev_ellipsoid_params_t, float) dev_ellipsoid_params;  // Array to hold ellipsoid params (A-F)
    DEVICE_OUTPUT(dev_major_axes_t, float) dev_major_axes;
    DEVICE_OUTPUT(dev_minor_axes_t, float) dev_minor_axes;
    DEVICE_OUTPUT(dev_poca_coords_t, float) dev_poca_coords;
    DEVICE_OUTPUT(dev_intervals_t, int) dev_intervals;
    DEVICE_OUTPUT(dev_interval_counts_t, int) dev_interval_counts;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
  };

  __global__ void pv_finder_kernel(Parameters parameters);

  struct PVFinder_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;
    std::pair<size_t, size_t> load_model_parameters(const ArgumentReferences<Parameters>& arguments) const;
    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{256, 1, 1}}};
    std::vector<unsigned> layer_sizes = {9, 20, 20, 20, 20, 20, 100};
  };

  // Helper functions
  __device__ void assign_intervals(float z_poca, int* intervals, int* num_intervals);
  __device__ bool state_poca(
    const Allen::Views::Physics::KalmanState& state,
    const BeamLine& beam_line,
    float& poca_x, float& poca_y, float& poca_z);
  __device__ void calculate_ellipsoid_params(
    const Allen::Views::Physics::KalmanState& state,
    float* ellipsoid_params,
    float* major_axis,
    float* minor_axes,
    float& poca_x, float& poca_y, float& poca_z,
    unsigned event_number,
    unsigned track_index);
  __device__ float leaky_relu(float x);
  __device__ float softplus(float x);
  __device__ void neural_network_forward(const float* input, float* output, const float* weights, const float* biases, const unsigned* layer_sizes);
  __device__ void normalize_input(float* input, const float* mean, const float* std);
  __device__ void pv_locations_updated_res(
    float* targets,
    float threshold,
    float integral_threshold,
    int min_width,
    float* pv_locations,
    int* num_pvs);
  __device__ void get_reco_resolution(
    float* pv_locations,
    int num_pvs,
    float* output_histogram,
    float* resolutions);

  // Vector operation helper functions
  __device__ float3 normalize(float3 v);
  __device__ float3 cross(float3 a, float3 b);
  __device__ float dot(float3 a, float3 b);

} // namespace PVFinder