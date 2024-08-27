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


#include "PVFinder.cuh"
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <cstdlib>  // For getenv



INSTANTIATE_ALGORITHM(PVFinder::PVFinder_t)

namespace PVFinder {

// Function to save A-E parameters to a CSV file
void save_extended_params_to_csv(
    const std::string& file_name,
    const std::vector<float>& ellipsoid_params_data,
    const std::vector<float>& major_axes_data,
    const std::vector<float>& minor_axes_data,
    const std::vector<float>& poca_data,
    unsigned num_events,
    unsigned max_tracks) 
{
    std::ofstream outfile(file_name);
    if (!outfile.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_name);
    }

    // Write the header
    outfile << "Event,Track,A,B,C,D,E,F,Major_X,Major_Y,Major_Z,Minor1_X,Minor1_Y,Minor1_Z,Minor2_X,Minor2_Y,Minor2_Z,POCA_X,POCA_Y,POCA_Z" << std::endl;

    // Write the parameters
    for (unsigned event = 0; event < num_events; ++event) {
        for (unsigned track = 0; track < max_tracks; ++track) {
            unsigned idx = (event * max_tracks + track);
            outfile << event << "," << track;

            // Ellipsoid parameters (A-F)
            for (unsigned j = 0; j < 6; ++j) {
                outfile << "," << ellipsoid_params_data[idx * 6 + j];
            }

            // Major axis
            for (unsigned j = 0; j < 3; ++j) {
                outfile << "," << major_axes_data[idx * 3 + j];
            }

            // Minor axes (2 minor axes, 3 components each)
            for (unsigned j = 0; j < 6; ++j) {
                outfile << "," << minor_axes_data[idx * 6 + j];
            }

            // POCA coordinates
            for (unsigned j = 0; j < 3; ++j) {
                outfile << "," << poca_data[idx * 3 + j];
            }

            outfile << std::endl;
        }
    }

    outfile.close();
}

std::vector<float> read_txt_file(const std::string& file_path) {
    std::vector<float> data;
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }
    
    float value;
    while (file >> value) {
        data.push_back(value);
    }
    
    return data;
}

void PVFinder_t::set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const {
    set_size<dev_pv_positions_t>(arguments, first<host_number_of_events_t>(arguments) * 100);
    set_size<dev_pv_number_t>(arguments, first<host_number_of_events_t>(arguments));
    set_size<dev_output_histogram_t>(arguments, first<host_number_of_events_t>(arguments) * 100);
    set_size<dev_layer_weights_t>(arguments, MAX_LAYERS * MAX_LAYER_SIZE * MAX_LAYER_SIZE);
    set_size<dev_layer_biases_t>(arguments, MAX_LAYERS * MAX_LAYER_SIZE);
    set_size<dev_layer_sizes_t>(arguments, MAX_LAYERS + 1);
    set_size<dev_input_mean_t>(arguments, 9);
    set_size<dev_input_std_t>(arguments, 9);
    set_size<dev_ellipsoid_params_t>(arguments, first<host_number_of_events_t>(arguments) * 256 * 6);  // Assuming max 256 tracks per event
}

void PVFinder_t::operator()(
    const ArgumentReferences<Parameters>& arguments,
    const RuntimeOptions& runtime_options,
    const Constants& constants,
    const Allen::Context& context) const
{
    std::cout << "PVFinder: Starting operation" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    auto [total_weights_size, total_biases_size] = load_model_parameters(arguments);
    
    auto load_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_time = load_end - start;
    std::cout << "PVFinder: Model parameters loaded. Time taken: " << load_time.count() << " seconds" << std::endl;
    
    // Initialize input mean and std
    std::vector<float> input_mean = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> input_std = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    // Allocate memory for outputs
    set_size<dev_layer_weights_t>(arguments, total_weights_size);
    set_size<dev_layer_biases_t>(arguments, total_biases_size);
    set_size<dev_layer_sizes_t>(arguments, layer_sizes.size());
    set_size<dev_input_mean_t>(arguments, 9);
    set_size<dev_input_std_t>(arguments, 9);

    Allen::memcpy(
        const_cast<float*>(data<dev_input_mean_t>(arguments)),
        input_mean.data(),
        input_mean.size() * sizeof(float),
        Allen::memcpyHostToDevice);

    Allen::memcpy(
        const_cast<float*>(data<dev_input_std_t>(arguments)),
        input_std.data(),
        input_std.size() * sizeof(float),
        Allen::memcpyHostToDevice);

    global_function(pv_finder_kernel)(
        dim3(first<host_number_of_events_t>(arguments)), dim3(256), context)(arguments);
    
    // Copy parameters back to host
    unsigned num_events = first<host_number_of_events_t>(arguments);
    unsigned max_tracks = 256;  // Assuming max 256 tracks per event

    std::vector<float> ellipsoid_params(num_events * max_tracks * 6);
    std::vector<float> major_axes(num_events * max_tracks * 3);
    std::vector<float> minor_axes(num_events * max_tracks * 6);
    std::vector<float> poca_coords(num_events * max_tracks * 3);

    Allen::memcpy(ellipsoid_params.data(), data<dev_ellipsoid_params_t>(arguments), 
                  ellipsoid_params.size() * sizeof(float), Allen::memcpyDeviceToHost);
    Allen::memcpy(major_axes.data(), data<dev_major_axes_t>(arguments), 
                  major_axes.size() * sizeof(float), Allen::memcpyDeviceToHost);
    Allen::memcpy(minor_axes.data(), data<dev_minor_axes_t>(arguments), 
                  minor_axes.size() * sizeof(float), Allen::memcpyDeviceToHost);
    Allen::memcpy(poca_coords.data(), data<dev_poca_coords_t>(arguments), 
                  poca_coords.size() * sizeof(float), Allen::memcpyDeviceToHost);

    // Save extended parameters to CSV
    save_extended_params_to_csv("extended_params.csv", ellipsoid_params, major_axes, minor_axes, poca_coords, num_events, max_tracks);

    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = kernel_end - load_end;
    std::cout << "PVFinder: Kernel execution complete. Time taken: " << kernel_time.count() << " seconds" << std::endl;
    
    // Debug output
    std::vector<float> pv_positions(num_events * 100);
    std::vector<unsigned> pv_numbers(num_events);
    std::vector<float> output_histogram(num_events * 100);
    
    Allen::memcpy(pv_positions.data(), data<dev_pv_positions_t>(arguments), pv_positions.size() * sizeof(float), Allen::memcpyDeviceToHost);
    Allen::memcpy(pv_numbers.data(), data<dev_pv_number_t>(arguments), pv_numbers.size() * sizeof(unsigned), Allen::memcpyDeviceToHost);
    Allen::memcpy(output_histogram.data(), data<dev_output_histogram_t>(arguments), output_histogram.size() * sizeof(float), Allen::memcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cout << "PVFinder: Total execution time: " << total_time.count() << " seconds" << std::endl;
}

__device__ float3 normalize(float3 v) {
    float mag = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (mag > 0.0f) {
        return make_float3(v.x / mag, v.y / mag, v.z / mag);
    } else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
}

__device__ float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ bool state_poca(
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

__device__ void get_reco_resolution(
    float* pv_locations,
    int num_pvs,
    float* output_histogram,
    float* resolutions)
{
    for (int i = 0; i < num_pvs; ++i) {
        int bin = (int)pv_locations[i];
        float max_value = output_histogram[bin];
        float half_max = max_value / 2.0f;
        
        int left_bin = bin;
        while (left_bin > 0 && output_histogram[left_bin] > half_max) {
            left_bin--;
        }
        
        int right_bin = bin;
        while (right_bin < 99 && output_histogram[right_bin] > half_max) {
            right_bin++;
        }
        
        float fwhm = (float)(right_bin - left_bin);
        resolutions[i] = fwhm / 2.355f; // Convert FWHM to standard deviation
    }
}


std::pair<size_t, size_t> PVFinder_t::load_model_parameters(const ArgumentReferences<Parameters>& arguments) const {
    const char* home = getenv("HOME");
    if (!home) {
        throw std::runtime_error("Unable to find HOME environment variable.");
    }

    const std::string base_path = std::string(home) + "/allen_work/Allen/device/pvfinder/include/model_parameters";
    std::vector<float> all_weights;
    std::vector<float> all_biases;

    // Initialize total weights and biases size
    size_t total_weights_size = 0;
    size_t total_biases_size = 0;

    for (int i = 1; i <= 6; ++i) {
        std::string weight_file = base_path + "/layer" + std::to_string(i) + ".weight.txt";
        std::string bias_file = base_path + "/layer" + std::to_string(i) + ".bias.txt";

        std::vector<float> weights = read_txt_file(weight_file);
        std::vector<float> biases = read_txt_file(bias_file);

        // Verify the dimensions
        unsigned expected_weight_size = layer_sizes[i-1] * layer_sizes[i];
        unsigned expected_bias_size = layer_sizes[i];

        if (weights.size() != expected_weight_size || biases.size() != expected_bias_size) {
            throw std::runtime_error("Unexpected dimensions for layer " + std::to_string(i) +
                                     ": weights size = " + std::to_string(weights.size()) +
                                     ", expected " + std::to_string(expected_weight_size) +
                                     ", biases size = " + std::to_string(biases.size()) +
                                     ", expected " + std::to_string(expected_bias_size));
        }

        all_weights.insert(all_weights.end(), weights.begin(), weights.end());
        all_biases.insert(all_biases.end(), biases.begin(), biases.end());

        total_weights_size += weights.size();
        total_biases_size += biases.size();
    }

    // Copy weights to device
    Allen::memcpy(
        const_cast<float*>(data<dev_layer_weights_t>(arguments)),
        all_weights.data(),
        all_weights.size() * sizeof(float),
        Allen::memcpyHostToDevice);

    // Copy biases to device
    Allen::memcpy(
        const_cast<float*>(data<dev_layer_biases_t>(arguments)),
        all_biases.data(),
        all_biases.size() * sizeof(float),
        Allen::memcpyHostToDevice);

    // Copy layer sizes to device
    Allen::memcpy(
        const_cast<unsigned*>(data<dev_layer_sizes_t>(arguments)),
        layer_sizes.data(),
        layer_sizes.size() * sizeof(unsigned),
        Allen::memcpyHostToDevice);

    return {total_weights_size, total_biases_size};
}

__device__ void assign_intervals(float z_poca, int* intervals, int* num_intervals) {
    // Shift z_poca to [0, 400] range for easier interval calculation
    z_poca += 100.0f;

    // Handle edge cases: z_poca outside [0, 400] range
    if (z_poca < 0.0f || z_poca > 400.0f) {
        *num_intervals = 1;
        intervals[0] = z_poca < 0.0f ? 0 : 39;
        return;
    }

    // Calculate base interval
    float interval_float = z_poca / 10.0f;
    int base_interval = floorf(interval_float);

    *num_intervals = 0;

    // Check if z_poca is within OVERLAP_WIDTH of lower boundary
    if (interval_float - base_interval <= OVERLAP_WIDTH && base_interval > 0) {
        intervals[(*num_intervals)++] = base_interval - 1;
    }

    // Always add the base interval
    intervals[(*num_intervals)++] = base_interval;

    // Check if z_poca is within OVERLAP_WIDTH of upper boundary
    if (base_interval + 1 - interval_float <= OVERLAP_WIDTH && base_interval < 39) {
        intervals[(*num_intervals)++] = base_interval + 1;
    }

    // Ensure all intervals are within [0, 39]
    for (int i = 0; i < *num_intervals; ++i) {
        intervals[i] = min(39, max(0, intervals[i]));
    }
}

__device__ void normalize_input(float* input, const float* mean, const float* std) {
    for (int i = 0; i < 9; ++i) {
        input[i] = (input[i] - mean[i]) / std[i];
    }
}

__device__ void calculate_ellipsoid_params(
    const Allen::Views::Physics::KalmanState& state,
    float* ellipsoid_params,
    float* major_axis,
    float* minor_axes,
    float& poca_x, float& poca_y, float& poca_z,
    unsigned event_number,
    unsigned track_index)
{
    BeamLine beam_line;

    bool poca_success = state_poca(state, beam_line, poca_x, poca_y, poca_z);
    if (!poca_success) {
        poca_x = poca_y = poca_z = 0.0f;
        for (int i = 0; i < 6; ++i) ellipsoid_params[i] = 0.0f;
        for (int i = 0; i < 3; ++i) major_axis[i] = 0.0f;
        for (int i = 0; i < 6; ++i) minor_axes[i] = 0.0f;
        return;
    }

    float3 center = make_float3(poca_x, poca_y, poca_z);
    float3 track_dir = normalize(make_float3(state.tx(), state.ty(), 1.0f));

    float3 zhat = normalize(center);
    float3 xhat = track_dir;
    float3 yhat = normalize(cross(zhat, xhat));

    float road_error = sqrtf(state.c00());

    float3 u1 = make_float3(road_error * zhat.x, road_error * zhat.y, road_error * zhat.z);
    float3 u2 = make_float3(road_error * yhat.x, road_error * yhat.y, road_error * yhat.z);

    float arg = dot(xhat, track_dir);
    arg = fminf(arg, 0.9999f);
    float u3_scale = (road_error * arg) / sqrtf(1.0f - arg * arg);
    float3 u3 = make_float3(u3_scale * xhat.x, u3_scale * xhat.y, u3_scale * xhat.z);

    // Store ellipsoid parameters (A-F)
    ellipsoid_params[0] = u1.x * u1.x + u2.x * u2.x + u3.x * u3.x;
    ellipsoid_params[1] = u1.y * u1.y + u2.y * u2.y + u3.y * u3.y;
    ellipsoid_params[2] = u1.z * u1.z + u2.z * u2.z + u3.z * u3.z;
    ellipsoid_params[3] = u1.x * u1.y + u2.x * u2.y + u3.x * u3.y;
    ellipsoid_params[4] = u1.x * u1.z + u2.x * u2.z + u3.x * u3.z;
    ellipsoid_params[5] = u1.y * u1.z + u2.y * u2.z + u3.y * u3.z;

    // Store major axis (u3)
    major_axis[0] = u3.x;
    major_axis[1] = u3.y;
    major_axis[2] = u3.z;

    // Store minor axes (u1 and u2)
    minor_axes[0] = u1.x;
    minor_axes[1] = u1.y;
    minor_axes[2] = u1.z;
    minor_axes[3] = u2.x;
    minor_axes[4] = u2.y;
    minor_axes[5] = u2.z;

    // Debug output
    if (event_number == 0 && track_index < 3) {
        printf("Track %d: u1 = (%.6f, %.6f, %.6f)\n", track_index, u1.x, u1.y, u1.z);
        printf("Track %d: u2 = (%.6f, %.6f, %.6f)\n", track_index, u2.x, u2.y, u2.z);
        printf("Track %d: u3 = (%.6f, %.6f, %.6f)\n", track_index, u3.x, u3.y, u3.z);
    }
}
__device__ float leaky_relu(float x) {
    return x > 0 ? x : 0.01f * x;
}

__device__ float softplus(float x) {
    const float THRESHOLD = 20.0f;
    if (x > THRESHOLD) {
        return x;
    } else if (x < -THRESHOLD) {
        return expf(x);
    } else {
        return logf(1.0f + expf(x));
    }
}

__device__ void pv_locations_updated_res(
    float* targets,
    float threshold,
    float integral_threshold,
    int min_width,
    float* pv_locations,
    int* num_pvs)
{
    int state = 0;
    float integral = 0.0f;
    float sum_weights_locs = 0.0f;
    bool peak_passed = false;
    float local_peak_value = 0.0f;
    int local_peak_index = 0;
    *num_pvs = 0;

    for (int i = 0; i < 100; ++i) {
        if (targets[i] >= threshold) {
            state += 1;
            integral += targets[i];
            sum_weights_locs += i * targets[i];

            if (targets[i] > local_peak_value) {
                local_peak_value = targets[i];
                local_peak_index = i;
            }

            if ((i > 0 && targets[i-1] > targets[i] + 0.05f && targets[i-1] > 1.1f * targets[i])) {
                peak_passed = true;
            }
        }

        if ((targets[i] < threshold || i == 99 || (i > 0 && targets[i-1] < targets[i] && peak_passed)) && state > 0) {
            if (state >= min_width && integral >= integral_threshold) {
                pv_locations[*num_pvs] = (sum_weights_locs / integral) + 0.5f;
                (*num_pvs)++;
            }

            state = 0;
            integral = 0.0f;
            sum_weights_locs = 0.0f;
            peak_passed = false;
            local_peak_value = 0.0f;
        }
    }
}

__device__ void neural_network_forward(const float* input, float* output, const float* weights, const float* biases, const unsigned* layer_sizes) {
    float layer_input[20];  // Max size of any hidden layer
    float layer_output[100];  // Size of the output layer

    // Copy input to layer_input
    for (int i = 0; i < 9; ++i) {
        layer_input[i] = input[i];
    }

    int weight_offset = 0;
    int bias_offset = 0;

    // Hidden layers (5 layers)
    for (int layer = 0; layer < 5; ++layer) {
        int input_size = (layer == 0) ? 9 : 20;
        for (int j = 0; j < 20; ++j) {
            float sum = biases[bias_offset + j];
            for (int k = 0; k < input_size; ++k) {
                sum += weights[weight_offset + j * input_size + k] * layer_input[k];
            }
            layer_output[j] = leaky_relu(sum);
        }

        // Copy output to input for next layer
        for (int j = 0; j < 20; ++j) {
            layer_input[j] = layer_output[j];
        }

        weight_offset += input_size * 20;
        bias_offset += 20;
    }

    // Output layer
    for (int j = 0; j < 100; ++j) {
        float sum = biases[bias_offset + j];
        for (int k = 0; k < 20; ++k) {
            sum += weights[weight_offset + j * 20 + k] * layer_input[k];
        }
        output[j] = softplus(sum);
    }
}

__global__ void pv_finder_kernel(Parameters parameters) {
    const unsigned event_number = blockIdx.x;
    const unsigned thread_id = threadIdx.x;

    if (event_number >= parameters.dev_number_of_events[0]) return;

    // Get views for VELO tracks and states for the current event
    const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];
    const auto velo_states_view = parameters.dev_velo_states_view[event_number];
    const unsigned num_tracks = velo_tracks_view.size();

    // Shared memory for intermediate results
    __shared__ float s_interval_data[256 * 9];  // Assuming max 256 threads per block

    // Base pointers for various parameters in global memory
    float* ellipsoid_params_base = parameters.dev_ellipsoid_params + event_number * 256 * 6;
    float* major_axes_base = parameters.dev_major_axes + event_number * 256 * 3;
    float* minor_axes_base = parameters.dev_minor_axes + event_number * 256 * 6;
    float* poca_coords_base = parameters.dev_poca_coords + event_number * 256 * 3;


    // Step 1: Indexing and preprocessing
    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        const auto track = velo_tracks_view.track(i);
        const auto state = velo_states_view.state(track.track_index());

        float ellipsoid_params[6];
        float major_axis[3];
        float minor_axes[6];
        float poca_x, poca_y, poca_z;

        calculate_ellipsoid_params(
            state,
            ellipsoid_params,
            major_axis,
            minor_axes,
            poca_x, poca_y, poca_z,
            event_number,
            i);
        // Debug output
        if (event_number == 0 && i < 10) {
            printf("Track %d: A-F = %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n", 
                   i, ellipsoid_params[0], ellipsoid_params[1], ellipsoid_params[2],
                   ellipsoid_params[3], ellipsoid_params[4], ellipsoid_params[5]);
            printf("Track %d: Major axis = %.6f, %.6f, %.6f\n", 
                   i, major_axis[0], major_axis[1], major_axis[2]);
            printf("Track %d: Minor axes = %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n", 
                   i, minor_axes[0], minor_axes[1], minor_axes[2],
                   minor_axes[3], minor_axes[4], minor_axes[5]);
        }            
        // Store preprocessed data in shared memory
        s_interval_data[thread_id * 9 + 0] = poca_x;
        s_interval_data[thread_id * 9 + 1] = poca_y;
        s_interval_data[thread_id * 9 + 2] = poca_z;
        for (int j = 0; j < 6; ++j) {
            s_interval_data[thread_id * 9 + 3 + j] = ellipsoid_params[j];
        }

        // Store data in global memory
        for (int j = 0; j < 6; ++j) {
            ellipsoid_params_base[i * 6 + j] = ellipsoid_params[j];
        }
        for (int j = 0; j < 3; ++j) {
            major_axes_base[i * 3 + j] = major_axis[j];
        }
        for (int j = 0; j < 6; ++j) {
            minor_axes_base[i * 6 + j] = minor_axes[j];
        }
        poca_coords_base[i * 3 + 0] = poca_x;
        poca_coords_base[i * 3 + 1] = poca_y;
        poca_coords_base[i * 3 + 2] = poca_z;        // Assign intervals based on z_poca

        int intervals[3];
        int num_intervals;
        assign_intervals(poca_z, intervals, &num_intervals);

        // Store interval information (you may need to adjust this based on your data structures)
        parameters.dev_interval_counts[event_number * num_tracks + i] = num_intervals;
        for (int j = 0; j < num_intervals; ++j) {
            parameters.dev_intervals[(event_number * num_tracks + i) * 3 + j] = intervals[j];
        }
    }

    __syncthreads();

    // Step 2: FCNN forward pass
    float input[9];
    float output[100];

    // Initialize input with preprocessed data
    for (int i = 0; i < 9; ++i) {
        input[i] = s_interval_data[thread_id * 9 + i];
    }

    // Normalize input
    normalize_input(input, parameters.dev_input_mean, parameters.dev_input_std);

    // Forward pass through the network
    neural_network_forward(input, output, parameters.dev_layer_weights, parameters.dev_layer_biases, parameters.dev_layer_sizes);

    // Step 3: Accumulate results and find PV positions
    float* output_histogram = parameters.dev_output_histogram + event_number * 100;
    for (int i = 0; i < 100; ++i) {
        atomicAdd(&output_histogram[i], 0.001f * output[i]);
    }

    __syncthreads();

    if (thread_id == 0) {
        float pv_locations[100];
        int num_pvs;
        float resolutions[100];

        pv_locations_updated_res(output_histogram, 0.01f, 0.2f, 2, pv_locations, &num_pvs);
        get_reco_resolution(pv_locations, num_pvs, output_histogram, resolutions);

        // Store results
        float* dev_pv_positions = parameters.dev_pv_positions + event_number * 100;
        unsigned* dev_pv_number = parameters.dev_pv_number + event_number;

        for (int i = 0; i < num_pvs; ++i) {
            dev_pv_positions[i] = pv_locations[i] * 0.1f - 100.0f;  // Convert bin to z position
        }
        *dev_pv_number = num_pvs;

        printf("PVFinder Kernel: Event %d processed. Found %d PVs\n", event_number, num_pvs);
    }
}

} // namespace PVFinder
