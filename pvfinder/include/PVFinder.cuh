#pragma once

#include "AlgorithmTypes.cuh"
#include "States.cuh"
#include "VeloConsolidated.cuh"

namespace PVFinder {
  constexpr float OVERLAP_WIDTH = 0.05f;
  constexpr unsigned MAX_LAYERS = 6;
  constexpr unsigned MAX_LAYER_SIZE = 100;

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
  __device__ void assign_intervals(float z_poca, int* intervals, int& num_intervals);
  __device__ void calculate_ellipsoid_params(
    float x, float y, float z, float tx, float ty,
    float c00, float c20, float c22, float c11, float c31, float c33,
    float* ellipsoid_params);
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
}
