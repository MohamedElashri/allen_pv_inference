// pv-finder.cpp

#include <torch/torch.h>
#include <torch/script.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

struct LayerParams {
    Eigen::MatrixXf weights;
    Eigen::VectorXf biases;
};

class Model {
private:
    std::vector<LayerParams> layers;
    
    std::vector<LayerParams> readPytFile(const std::string& filename) {
        std::vector<LayerParams> layers;
        
        torch::jit::script::Module module;
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            module = torch::jit::load(filename);
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model\n";
            std::cerr << e.msg() << std::endl;
            return layers;
        }

        for (const auto& p : module.named_parameters()) {
            if (p.name.find(".weight") != std::string::npos) {
                at::Tensor tensor = p.value;
                Eigen::MatrixXf weight = Eigen::Map<Eigen::MatrixXf>(
                    tensor.data_ptr<float>(), tensor.size(0), tensor.size(1));
                layers.push_back({weight, Eigen::VectorXf()});
            } else if (p.name.find(".bias") != std::string::npos) {
                at::Tensor tensor = p.value;
                Eigen::VectorXf bias = Eigen::Map<Eigen::VectorXf>(
                    tensor.data_ptr<float>(), tensor.size(0));
                layers.back().biases = bias;
            }
        }
        
        return layers;
    }

public:
    Model(const std::string& filename) {
        layers = readPytFile(filename);
    }
    
    Eigen::MatrixXf forward(const Eigen::MatrixXf& input) {
        if (input.array().isNaN().any() || !input.array().isFinite().all()) {
            std::cerr << "Input contains NaN or infinite values!" << std::endl;
            // Return a matrix of zeros or throw an exception
            return Eigen::MatrixXf::Zero(input.rows(), input.cols());
        }

        Eigen::MatrixXf x = input;
        std::cout << "Input shape: " << x.rows() << "x" << x.cols() << std::endl;
        std::cout << "Input (first 5 rows, all columns):\n" << x.topRows(5) << std::endl;

        for (size_t i = 0; i < layers.size() - 1; ++i) {
            std::cout << "Layer " << i + 1 << " weights shape: " << layers[i].weights.rows() << "x" << layers[i].weights.cols() << std::endl;
            std::cout << "Layer " << i + 1 << " biases shape: " << layers[i].biases.size() << std::endl;

            x = (layers[i].weights * x.transpose()).transpose() + layers[i].biases.replicate(1, x.rows()).transpose();
            std::cout << "After layer " << i + 1 << " (before activation):\n" << x.topRows(5) << std::endl;
            
            x = x.array().max(0.01f * x.array()); // LeakyReLU
            std::cout << "After layer " << i + 1 << " (after activation):\n" << x.topRows(5) << std::endl;
        }
        
        // Last layer (Softplus)
        std::cout << "Last layer weights shape: " << layers.back().weights.rows() << "x" << layers.back().weights.cols() << std::endl;
        std::cout << "Last layer biases shape: " << layers.back().biases.size() << std::endl;

        x = (layers.back().weights * x.transpose()).transpose() + layers.back().biases.replicate(1, x.rows()).transpose();
        std::cout << "After last layer (before activation):\n" << x.topRows(5) << std::endl;
        
        // Numerically stable Softplus
        x = x.unaryExpr([](float v) {
            if (v <= -50.0f) {
                return std::exp(v);
            } else if (v >= 50.0f) {
                return v;
            } else {
                return std::log1p(std::exp(v));
            }
        });
        std::cout << "After last layer (after activation):\n" << x.topRows(5) << std::endl;
        
        return x;
    }


};

Eigen::Matrix3f constructCovarianceMatrix(float c00, float c20, float c22, float c11, float c31, float c33) {
    Eigen::Matrix3f cov;
    cov << c00, c20, c31,
           c20, c22, c11,
           c31, c11, c33;
    return cov;
}

Eigen::VectorXf calculateABCDEF(float c00, float c20, float c22, float c11, float c31, float c33) {
    Eigen::Matrix3f cov = constructCovarianceMatrix(c00, c20, c22, c11, c31, c33);
    Eigen::Matrix3f invCov = cov.inverse();
    
    Eigen::VectorXf abcdef(6);
    abcdef << invCov(0,0), invCov(1,1), invCov(2,2), 
              invCov(0,1), invCov(0,2), invCov(1,2);
    
    return abcdef;
}

std::vector<Eigen::VectorXf> readInputCSV(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    std::vector<Eigen::VectorXf> inputs;

    // Skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<float> values;
        
        while (std::getline(iss, token, ',')) {
            values.push_back(std::stof(token));
        }
        
        Eigen::VectorXf input(9);
        input << values[2], values[3], values[4], // x, y, z
                 calculateABCDEF(values[8], values[9], values[10], values[11], values[12], values[13]);
        
        inputs.push_back(input);
    }

    return inputs;
}

void saveOutput(const Eigen::MatrixXf& output, const std::string& filename) {
    std::ofstream file(filename);
    file << output.format(Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n"));
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_file.pyt> <input_file.csv> <output_file.csv>" << std::endl;
        return 1;
    }

    std::string model_file = argv[1];
    std::string input_file = argv[2];
    std::string output_file = argv[3];

    try {
        // Load the model
        Model model(model_file);
        
        // Read input data
        std::vector<Eigen::VectorXf> inputs = readInputCSV(input_file);
        std::cout << "Number of input samples: " << inputs.size() << std::endl; // debugging statement

        
        // Prepare input matrix
        Eigen::MatrixXf inputMatrix(inputs.size(), 9);
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputMatrix.row(i) = inputs[i];
        }

        std::cout << "Input matrix shape: " << inputMatrix.rows() << "x" << inputMatrix.cols() << std::endl; // debugging statement
        std::cout << "First 5 rows of input matrix:\n" << inputMatrix.topRows(5) << std::endl; // debugging statement
        
        // Perform inference
        Eigen::MatrixXf output = model.forward(inputMatrix);

        std::cout << "Output matrix shape: " << output.rows() << "x" << output.cols() << std::endl; // debugging statement
        std::cout << "First 5 rows of output matrix:\n" << output.topRows(5) << std::endl; // debugging statement

        // Save results
        saveOutput(output, output_file);
        
        std::cout << "Inference completed successfully. Results saved to " << output_file << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
