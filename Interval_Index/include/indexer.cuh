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
#include "VeloConsolidated.cuh"
#include "AlgorithmTypes.cuh"
#include "States.cuh"
#include <cassert>
#include <vector>

constexpr float OVERLAP_WIDTH = 0.05f;  // 0.5 units overlap on each side
constexpr int MAX_INTERVALS = 40;  // Maximum number of intervals

namespace interval_indexer {
  struct BeamLine {
    __host__ __device__ float x() const { return 0.f; }
    __host__ __device__ float y() const { return 0.f; }
    __host__ __device__ float z() const { return 0.f; }
    __host__ __device__ float tx() const { return 0.f; }
    __host__ __device__ float ty() const { return 0.f; }
  };

  struct Parameters {
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    DEVICE_INPUT(dev_offsets_all_velo_tracks_t, unsigned) dev_offsets_all_velo_tracks;
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_velo_states_view_t, Allen::Views::Physics::KalmanStates) dev_velo_states_view;
    DEVICE_OUTPUT(dev_intervals_t, int) dev_intervals;
    DEVICE_OUTPUT(dev_interval_counts_t, int) dev_interval_counts;
    DEVICE_OUTPUT(dev_sorted_track_indices_t, int) dev_sorted_track_indices;
    DEVICE_OUTPUT(dev_ellipsoid_params_t, float) dev_ellipsoid_params;
    DEVICE_OUTPUT(dev_z_pocas_t, float) dev_z_pocas;
    DEVICE_OUTPUT(dev_track_indices_t, int) dev_track_indices;
    DEVICE_OUTPUT(dev_track_x_t, float) dev_track_x;
    DEVICE_OUTPUT(dev_track_y_t, float) dev_track_y;
    DEVICE_OUTPUT(dev_track_z_t, float) dev_track_z;
    DEVICE_OUTPUT(dev_track_tx_t, float) dev_track_tx;
    DEVICE_OUTPUT(dev_track_ty_t, float) dev_track_ty;
    DEVICE_OUTPUT(dev_u1_x_t, float) dev_u1_x;
    DEVICE_OUTPUT(dev_u1_y_t, float) dev_u1_y;
    DEVICE_OUTPUT(dev_u1_z_t, float) dev_u1_z;
    DEVICE_OUTPUT(dev_u2_x_t, float) dev_u2_x;
    DEVICE_OUTPUT(dev_u2_y_t, float) dev_u2_y;
    DEVICE_OUTPUT(dev_u2_z_t, float) dev_u2_z;
    DEVICE_OUTPUT(dev_u3_x_t, float) dev_u3_x;
    DEVICE_OUTPUT(dev_u3_y_t, float) dev_u3_y;
    DEVICE_OUTPUT(dev_u3_z_t, float) dev_u3_z;    
    HOST_OUTPUT(host_track_info_t, float) host_track_info;
    PROPERTY(block_dim_t, "block_dim", "block dimensions", DeviceDimensions) block_dim;
  };

  __global__ void interval_indexer(Parameters parameters);

  struct interval_indexer_t : public DeviceAlgorithm, Parameters {
    void set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const;
    void operator()(
      const ArgumentReferences<Parameters>& arguments,
      const RuntimeOptions&,
      const Constants&,
      const Allen::Context& context) const;

    void write_output(const ArgumentReferences<Parameters>& arguments) const;

  private:
    Property<block_dim_t> m_block_dim {this, {{256, 1, 1}}};
  };

  // Helper functions
  __device__ void assign_intervals(float z_poca, int* intervals, int& num_intervals);
  __device__ bool state_poca(
    const Allen::Views::Physics::KalmanState& state,
    const BeamLine& beam_line,
    float& poca_x, float& poca_y, float& poca_z);
  __device__ void bubble_sort(float* z_pocas, int* indices, int n);
  __device__ void calculate_ellipsoid_params(
    const Allen::Views::Physics::KalmanState& state,
    float* ellipsoid_params,
    float& poca_x, float& poca_y, float& poca_z,
    float3& u1, float3& u2, float3& u3);
}