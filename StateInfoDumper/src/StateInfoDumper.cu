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
#include "StateInfoDumper.cuh"
#include <stdio.h>

INSTANTIATE_ALGORITHM(state_info_dumper::state_info_dumper_t)

void state_info_dumper::state_info_dumper_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  // No output buffer to set size for in this algorithm
}

void state_info_dumper::state_info_dumper_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  global_function(state_info_dumper)(
    dim3(first<host_number_of_events_t>(arguments)), property<block_dim_t>(), context)(arguments);
}

__global__ void state_info_dumper::state_info_dumper(state_info_dumper::Parameters parameters)
{
  const unsigned event_number = blockIdx.x;
  const unsigned number_of_events = parameters.dev_number_of_events[0];

  if (event_number < number_of_events) {
    const auto velo_states_view = parameters.dev_velo_states_view[event_number];
    const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];
    const unsigned num_tracks = velo_tracks_view.size();

    for (unsigned track_index = threadIdx.x; track_index < num_tracks; track_index += blockDim.x) {
      const auto track = velo_tracks_view.track(track_index);
      const auto state = velo_states_view.state(track.track_index());

    // Print state information
    printf("Event %d, Track %d: x=%.6f, y=%.6f, z=%.6f, tx=%.6f, ty=%.6f, qop=%.6f\n",
          event_number, track_index, state.x(), state.y(), state.z(), state.tx(), state.ty(), state.qop());

    // Print covariance matrix elements
    printf("Covariance: c00=%.6f, c20=%.6f, c22=%.6f, c11=%.6f, c31=%.6f, c33=%.6f\n",
          state.c00(), state.c20(), state.c22(), state.c11(), state.c31(), state.c33());
    }
  }
}
