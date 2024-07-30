#pragma once

#include "VeloConsolidated.cuh"
#include "AlgorithmTypes.cuh"
#include "States.cuh"
#include <cassert>
#include <vector>

constexpr float OVERLAP_WIDTH = 0.05f;  // 0.5 units overlap on each side

namespace interval_indexer {
  struct Parameters {
    HOST_INPUT(host_number_of_reconstructed_velo_tracks_t, unsigned) host_number_of_reconstructed_velo_tracks;
    HOST_INPUT(host_number_of_events_t, unsigned) host_number_of_events;
    DEVICE_INPUT(dev_number_of_events_t, unsigned) dev_number_of_events;
    DEVICE_INPUT(dev_offsets_all_velo_tracks_t, unsigned) dev_offsets_all_velo_tracks;
    DEVICE_INPUT(dev_velo_tracks_view_t, Allen::Views::Velo::Consolidated::Tracks) dev_velo_tracks_view;
    DEVICE_INPUT(dev_velo_states_view_t, Allen::Views::Physics::KalmanStates) dev_velo_states_view;
    //DEVICE_INPUT(dev_velo_kalman_beamline_states_t, char) dev_velo_kalman_beamline_states;
    //DEVICE_INPUT(dev_offsets_velo_track_hit_number_t, unsigned) dev_velo_track_hit_number;
    DEVICE_OUTPUT(dev_intervals_t, int) dev_intervals;
    DEVICE_OUTPUT(dev_interval_counts_t, int) dev_interval_counts;
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

  private:
    Property<block_dim_t> m_block_dim {this, {{256, 1, 1}}};
  };
} // namespace interval_indexer
