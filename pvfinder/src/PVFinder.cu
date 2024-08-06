#include "PVFinder.cuh"
#include <algorithm>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <chrono>

INSTANTIATE_ALGORITHM(PVFinder::PVFinder_t)

namespace PVFinder {

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
    
    std::cout << "First few values from " << file_path << ":" << std::endl;
    for (size_t i = 0; i < std::min(data.size(), size_t(5)); ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
    return data;
}

void PVFinder_t::set_arguments_size(ArgumentReferences<Parameters> arguments, const RuntimeOptions&, const Constants&) const
{
    set_size<dev_pv_positions_t>(arguments, first<host_number_of_events_t>(arguments) * 100);
    set_size<dev_pv_number_t>(arguments, first<host_number_of_events_t>(arguments));
    set_size<dev_output_histogram_t>(arguments, first<host_number_of_events_t>(arguments) * 100);
    set_size<dev_layer_weights_t>(arguments, MAX_LAYERS * MAX_LAYER_SIZE * MAX_LAYER_SIZE);
    set_size<dev_layer_biases_t>(arguments, MAX_LAYERS * MAX_LAYER_SIZE);
    set_size<dev_layer_sizes_t>(arguments, MAX_LAYERS + 1);
    set_size<dev_input_mean_t>(arguments, 9);
    set_size<dev_input_std_t>(arguments, 9);
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


    load_model_parameters(arguments);
    
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
    
    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_time = kernel_end - load_end;
    std::cout << "PVFinder: Kernel execution complete. Time taken: " << kernel_time.count() << " seconds" << std::endl;
    
    // Debug output
    unsigned num_events = first<host_number_of_events_t>(arguments);
    std::cout << "PVFinder: Processed " << num_events << " events" << std::endl;
    
    // Transfer data from device to host
    std::vector<float> pv_positions(num_events * 100);
    std::vector<unsigned> pv_numbers(num_events);
    std::vector<float> output_histogram(num_events * 100);
    
    Allen::memcpy(pv_positions.data(), data<dev_pv_positions_t>(arguments), pv_positions.size() * sizeof(float), Allen::memcpyDeviceToHost);
    Allen::memcpy(pv_numbers.data(), data<dev_pv_number_t>(arguments), pv_numbers.size() * sizeof(unsigned), Allen::memcpyDeviceToHost);
    Allen::memcpy(output_histogram.data(), data<dev_output_histogram_t>(arguments), output_histogram.size() * sizeof(float), Allen::memcpyDeviceToHost);
    
    // // Output the results
    // std::ofstream outfile("pv_finder_output.txt");
    
    // for (unsigned i = 0; i < std::min(num_events, 5u); ++i) {
    //     unsigned num_pvs = pv_numbers[i];
    //     outfile << "Event " << i << ": " << num_pvs << " PVs found" << std::endl;
    //     for (unsigned j = 0; j < std::min(num_pvs, 5u); ++j) {
    //         outfile << "  PV " << j << ": " << pv_positions[i*100 + j] << std::endl;
    //     }
    //     outfile << "  Histogram:" << std::endl;
    //     for (unsigned j = 0; j < 100; ++j) {
    //         outfile << "    Bin " << j << ": " << output_histogram[i*100 + j] << std::endl;
    //     }
    //     outfile << std::endl;
    // }
    
    // outfile.close();
    
    // std::cout << "PVFinder: Output written to pv_finder_output.txt" << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end - start;
    std::cout << "PVFinder: Total execution time: " << total_time.count() << " seconds" << std::endl;
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
    const std::string base_path = "/data/home/melashri/iris/debug-allen/non-Allen/closure_test/model_parameters/model_parameters_txt/";
    std::vector<float> all_weights;
    std::vector<float> all_biases;
    
    // Initialize total weights and biases size
    size_t total_weights_size = 0;
    size_t total_biases_size = 0;

    for (int i = 1; i <= 6; ++i) {
        std::string weight_file = base_path + "layer" + std::to_string(i) + ".weight.txt";
        std::string bias_file = base_path + "layer" + std::to_string(i) + ".bias.txt";

        std::vector<float> weights = read_txt_file(weight_file);
        std::vector<float> biases = read_txt_file(bias_file);

        // Verify the dimensions
        unsigned expected_weight_size = layer_sizes[i-1] * layer_sizes[i];
        unsigned expected_bias_size = layer_sizes[i];
        
        std::cout << "Layer " << i << ":" << std::endl;
        std::cout << "  Expected weight size: " << expected_weight_size << std::endl;
        std::cout << "  Actual weight size: " << weights.size() << std::endl;
        std::cout << "  Expected bias size: " << expected_bias_size << std::endl;
        std::cout << "  Actual bias size: " << biases.size() << std::endl;

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

    std::cout << "PVFinder: Loaded " << all_weights.size() << " weights and " << all_biases.size() << " biases" << std::endl;
    for (int i = 0; i < 7; ++i) {
        std::cout << "Layer " << i << ": " << layer_sizes[i] << " neurons" << std::endl;
    }

    std::cout << "Total weights size: " << total_weights_size << std::endl;
    std::cout << "Total biases size: " << total_biases_size << std::endl;

    return {total_weights_size, total_biases_size};
}

__device__ void assign_intervals(float z_poca, int* intervals, int& num_intervals)
{
    // Shift z_poca to [0, 400] range for easier interval calculation
    z_poca += 100.0f;

    // Handle edge cases: z_poca outside [0, 400] range
    if (z_poca < 0.0f || z_poca > 400.0f) {
        num_intervals = 1;
        intervals[0] = z_poca < 0.0f ? 0 : 39;
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

__device__ void normalize_input(float* input, const float* mean, const float* std)
{
    for (int i = 0; i < 9; ++i) {
        input[i] = (input[i] - mean[i]) / std[i];
    }
}



  __device__ void calculate_ellipsoid_params(
    float x, float y, float z, float tx, float ty,
    float c00, float c20, float c22, float c11, float c31, float c33,
    float* ellipsoid_params)
  {
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
  }

__device__ float leaky_relu(float x)
{
    return x > 0 ? x : 0.01f * x;
}

__device__ float softplus(float x)
{
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

__device__ void neural_network_forward(const float* input, float* output, const float* weights, const float* biases, const unsigned* layer_sizes)
{
    float layer_input[20];  // Max size of any hidden layer
    float layer_output[100];  // Size of the output layer

    // Copy input to layer_input
    for (int i = 0; i < 9; ++i) {
        layer_input[i] = input[i];
        // Debug print
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            printf("Input %d: %f\n", i, input[i]);
        }
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


__global__ void pv_finder_kernel(Parameters parameters)
{
    const unsigned event_number = blockIdx.x;
    const unsigned thread_id = threadIdx.x;

    if (event_number >= parameters.dev_number_of_events[0]) return;

    // Get views for VELO tracks and states for the current event
    const auto velo_tracks_view = parameters.dev_velo_tracks_view[event_number];
    const auto velo_states_view = parameters.dev_velo_states_view[event_number];
    const unsigned num_tracks = velo_tracks_view.size();

    // Shared memory for intermediate results
    __shared__ float s_interval_data[256 * 9];  // Assuming max 256 threads per block

    // Step 1: Indexing and preprocessing
    for (unsigned i = thread_id; i < num_tracks; i += blockDim.x) {
        const auto track = velo_tracks_view.track(i);
        const auto state = velo_states_view.state(track.track_index());

        float ellipsoid_params[6];
        calculate_ellipsoid_params(
            state.x(), state.y(), state.z(), state.tx(), state.ty(),
            state.c00(), state.c20(), state.c22(), state.c11(), state.c31(), state.c33(),
            ellipsoid_params);

        // Store preprocessed data in shared memory
        s_interval_data[thread_id * 9 + 0] = state.x();
        s_interval_data[thread_id * 9 + 1] = state.y();
        s_interval_data[thread_id * 9 + 2] = state.z();
        for (int j = 0; j < 6; ++j) {
            s_interval_data[thread_id * 9 + 3 + j] = ellipsoid_params[j];
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
