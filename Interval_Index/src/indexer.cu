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
#include <fstream>
#include <iomanip>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define DEVICE __device__
#define HOST_DEVICE __host__ __device__
#else
#define DEVICE
#define HOST_DEVICE
#include <cmath>
#endif

INSTANTIATE_ALGORITHM(interval_indexer::interval_indexer_t)

namespace interval_indexer {

__host__ void interval_indexer_t::set_arguments_size(
  ArgumentReferences<Parameters> arguments,
  const RuntimeOptions&,
  const Constants&) const
{
  auto host_number_of_events = first<host_number_of_events_t>(arguments);
  auto dev_offsets_all_velo_tracks = data<dev_offsets_all_velo_tracks_t>(arguments);

  if (dev_offsets_all_velo_tracks == nullptr) {
    throw std::runtime_error("dev_offsets_all_velo_tracks is null");
  }

  unsigned total_tracks;
#ifdef __CUDACC__
  cudaMemcpy(&total_tracks, &dev_offsets_all_velo_tracks[host_number_of_events], sizeof(unsigned), cudaMemcpyDeviceToHost);
#else
  total_tracks = dev_offsets_all_velo_tracks[host_number_of_events];
#endif

  set_size<dev_intervals_t>(arguments, total_tracks * 3);
  set_size<dev_interval_counts_t>(arguments, total_tracks);
  set_size<dev_sorted_track_indices_t>(arguments, total_tracks);
  set_size<dev_ellipsoid_params_t>(arguments, total_tracks * 6);
  set_size<dev_z_pocas_t>(arguments, total_tracks);
  set_size<dev_track_indices_t>(arguments, total_tracks);
  set_size<dev_track_x_t>(arguments, total_tracks);
  set_size<dev_track_y_t>(arguments, total_tracks);
  set_size<dev_track_z_t>(arguments, total_tracks);
  set_size<dev_track_tx_t>(arguments, total_tracks);
  set_size<dev_track_ty_t>(arguments, total_tracks);
  set_size<dev_u1_x_t>(arguments, total_tracks);
  set_size<dev_u1_y_t>(arguments, total_tracks);
  set_size<dev_u1_z_t>(arguments, total_tracks);
  set_size<dev_u2_x_t>(arguments, total_tracks);
  set_size<dev_u2_y_t>(arguments, total_tracks);
  set_size<dev_u2_z_t>(arguments, total_tracks);
  set_size<dev_u3_x_t>(arguments, total_tracks);
  set_size<dev_u3_y_t>(arguments, total_tracks);
  set_size<dev_u3_z_t>(arguments, total_tracks);
  set_size<host_track_info_t>(arguments, total_tracks * 23); // 23 values per track
}

void interval_indexer_t::operator()(
  const ArgumentReferences<Parameters>& arguments,
  const RuntimeOptions&,
  const Constants&,
  const Allen::Context& context) const
{
  global_function(interval_indexer)(
    dim3(first<host_number_of_events_t>(arguments)), dim3(256), context)(arguments);
  
  write_output(arguments);
}

DEVICE void assign_intervals(float z_poca, int* intervals, int& num_intervals)
{
    z_poca += 100.0f;

    if (z_poca < 0.0f || z_poca > 400.0f) {
        num_intervals = 1;
        intervals[0] = z_poca < 0.0f ? 0 : 39;
        return;
    }

    float interval_float = z_poca / 10.0f;
    int base_interval = floorf(interval_float);
    
    num_intervals = 0;

    if (interval_float - base_interval <= OVERLAP_WIDTH && base_interval > 0) {
        intervals[num_intervals++] = base_interval - 1;
    }

    intervals[num_intervals++] = base_interval;

    if (base_interval + 1 - interval_float <= OVERLAP_WIDTH && base_interval < 39) {
        intervals[num_intervals++] = base_interval + 1;
    }

    for (int i = 0; i < num_intervals; ++i) {
        intervals[i] = min(39, max(0, intervals[i]));
    }
}

DEVICE void bubble_sort(float* z_pocas, int* indices, int n)
{
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (z_pocas[j] > z_pocas[j + 1]) {
                float temp_z = z_pocas[j];
                z_pocas[j] = z_pocas[j + 1];
                z_pocas[j + 1] = temp_z;
                int temp_index = indices[j];
                indices[j] = indices[j + 1];
                indices[j + 1] = temp_index;
            }
        }
    }
}

DEVICE bool state_poca(
  const Allen::Views::Physics::KalmanState& state,
  const BeamLine& beam_line,
  float& poca_x, float& poca_y, float& poca_z)
{
  float dz = -state.z();
  float dx = state.x() + dz * state.tx();
  float dy = state.y() + dz * state.ty();

  poca_x = dx;
  poca_y = dy;
  poca_z = 0.f;

  float distance = sqrtf(dx * dx + dy * dy);

  return distance < 1000.f;
}

DEVICE float3 normalize(const float3& v) {
    float mag = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (mag > 0.0f) {
        return make_float3(v.x / mag, v.y / mag, v.z / mag);
    } else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

DEVICE float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

DEVICE float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

DEVICE void calculate_ellipsoid_params(
  const Allen::Views::Physics::KalmanState& state,
  float* ellipsoid_params,
  float& poca_x, float& poca_y, float& poca_z,
  float3& u1, float3& u2, float3& u3)
{
  BeamLine beam_line;

  bool poca_success = state_poca(state, beam_line, poca_x, poca_y, poca_z);
  if (!poca_success) {
    poca_x = poca_y = poca_z = 0.0f;
    for (int i = 0; i < 6; ++i) {
        ellipsoid_params[i] = 0.0f;
    }
    u1 = u2 = u3 = make_float3(0.0f, 0.0f, 0.0f);
    return;
  }

  float3 center = make_float3(poca_x, poca_y, poca_z);
  float3 track_dir = make_float3(state.tx(), state.ty(), 1.0f);
  track_dir = normalize(track_dir);

  float3 zhat = normalize(center);
  float3 xhat = track_dir;
  float3 yhat = cross(zhat, xhat);
  yhat = normalize(yhat);

  float road_error = sqrtf(state.c00());

  u1.x = road_error * zhat.x;
  u1.y = road_error * zhat.y;
  u1.z = road_error * zhat.z;

  u2.x = road_error * yhat.x;
  u2.y = road_error * yhat.y;
  u2.z = road_error * yhat.z;

  float arg = dot(xhat, track_dir);
  arg = fminf(arg, 0.9999f);
  float u3_scale = (road_error * arg) / sqrtf(1.0f - arg * arg);
  u3.x = u3_scale * xhat.x;
  u3.y = u3_scale * xhat.y;
  u3.z = u3_scale * xhat.z;

  ellipsoid_params[0] = u1.x * u1.x + u2.x * u2.x + u3.x * u3.x;
  ellipsoid_params[1] = u1.y * u1.y + u2.y * u2.y + u3.y * u3.y;
  ellipsoid_params[2] = u1.z * u1.z + u2.z * u2.z + u3.z * u3.z;
  ellipsoid_params[3] = u1.x * u1.y + u2.x * u2.y + u3.x * u3.y;
  ellipsoid_params[4] = u1.x * u1.z + u2.x * u2.z + u3.x * u3.z;
  ellipsoid_params[5] = u1.y * u1.z + u2.y * u2.z + u3.y * u3.z;
}

__global__ void interval_indexer(Parameters parameters)
{
    const unsigned event_number = blockIdx.x;
    const unsigned number_of_events = parameters.dev_number_of_events[0];

    if (event_number >= number_of_events) return;

    const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];
    const auto velo_states_view = parameters.dev_velo_states_view[event_number];
    const unsigned num_tracks = velo_tracks_view.size();
    const unsigned event_track_offset = velo_tracks_view.offset();

    for (unsigned i = threadIdx.x; i < num_tracks; i += blockDim.x) {
        const auto track = velo_tracks_view.track(i);
        const auto state = velo_states_view.state(track.track_index());

        unsigned output_index = event_track_offset + i;

        // Calculate all required values
        float poca_x, poca_y, poca_z;
        float3 u1, u2, u3;
        float ellipsoid_params[6];
        calculate_ellipsoid_params(state, ellipsoid_params, poca_x, poca_y, poca_z, u1, u2, u3);

        // Store all values in the host_track_info array
        unsigned info_index = output_index * 23;
        parameters.host_track_info[info_index++] = state.x();
        parameters.host_track_info[info_index++] = state.y();
        parameters.host_track_info[info_index++] = state.z();
        parameters.host_track_info[info_index++] = state.tx();
        parameters.host_track_info[info_index++] = state.ty();
        parameters.host_track_info[info_index++] = poca_x;
        parameters.host_track_info[info_index++] = poca_y;
        parameters.host_track_info[info_index++] = poca_z;
        parameters.host_track_info[info_index++] = u1.x;
        parameters.host_track_info[info_index++] = u1.y;
        parameters.host_track_info[info_index++] = u1.z;
        parameters.host_track_info[info_index++] = u2.x;
        parameters.host_track_info[info_index++] = u2.y;
        parameters.host_track_info[info_index++] = u2.z;
        parameters.host_track_info[info_index++] = u3.x;
        parameters.host_track_info[info_index++] = u3.y;
        parameters.host_track_info[info_index++] = u3.z;
        parameters.host_track_info[info_index++] = ellipsoid_params[0];
        parameters.host_track_info[info_index++] = ellipsoid_params[1];
        parameters.host_track_info[info_index++] = ellipsoid_params[2];
        parameters.host_track_info[info_index++] = ellipsoid_params[3];
        parameters.host_track_info[info_index++] = ellipsoid_params[4];
        parameters.host_track_info[info_index] = ellipsoid_params[5];

        // Store basic track parameters
        parameters.dev_track_x[output_index] = state.x();
        parameters.dev_track_y[output_index] = state.y();
        parameters.dev_track_z[output_index] = state.z();
        parameters.dev_track_tx[output_index] = state.tx();
        parameters.dev_track_ty[output_index] = state.ty();

        // Compute z_poca and store
        float z_poca = state.z();
        parameters.dev_z_pocas[output_index] = z_poca;
        parameters.dev_track_indices[output_index] = i;

        // Assign intervals
        int intervals[3];
        int num_intervals;
        assign_intervals(z_poca, intervals, num_intervals);

        parameters.dev_interval_counts[output_index] = num_intervals;
        for (int j = 0; j < num_intervals; ++j) {
            parameters.dev_intervals[output_index * 3 + j] = intervals[j];
        }

        parameters.dev_sorted_track_indices[output_index] = i;

        parameters.dev_u1_x[output_index] = u1.x;
        parameters.dev_u1_y[output_index] = u1.y;
        parameters.dev_u1_z[output_index] = u1.z;
        parameters.dev_u2_x[output_index] = u2.x;
        parameters.dev_u2_y[output_index] = u2.y;
        parameters.dev_u2_z[output_index] = u2.z;
        parameters.dev_u3_x[output_index] = u3.x;
        parameters.dev_u3_y[output_index] = u3.y;
        parameters.dev_u3_z[output_index] = u3.z;

        for (int j = 0; j < 6; ++j) {
            parameters.dev_ellipsoid_params[output_index * 6 + j] = ellipsoid_params[j];
        }
    }
}

void interval_indexer_t::write_output(const ArgumentReferences<Parameters>& arguments) const
{
    auto host_number_of_events = first<host_number_of_events_t>(arguments);
    auto dev_offsets_all_velo_tracks = data<dev_offsets_all_velo_tracks_t>(arguments);

    unsigned total_tracks;
    cudaMemcpy(&total_tracks, &dev_offsets_all_velo_tracks[host_number_of_events], sizeof(unsigned), cudaMemcpyDeviceToHost);

    auto host_track_info = data<host_track_info_t>(arguments);

    std::ofstream outfile("track_info.csv");
    outfile << std::setprecision(6) << std::fixed;
    outfile << "x,y,z,tx,ty,poca_x,poca_y,poca_z,u1.x,u1.y,u1.z,u2.x,u2.y,u2.z,u3.x,u3.y,u3.z,A,B,C,D,E,F\n";

    for (unsigned i = 0; i < total_tracks; ++i) {
        for (unsigned j = 0; j < 23; ++j) {
            outfile << host_track_info[i * 23 + j];
            if (j < 22) outfile << ",";
        }
        outfile << "\n";
    }

    outfile.close();
}

} // namespace interval_indexer