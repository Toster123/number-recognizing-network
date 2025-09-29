#include "network.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <array>

namespace {
    constexpr size_t kWeightsSize = 361578;
}

SequentialNetwork::SequentialNetwork() {
    layers_.reserve(10);
    std::vector<double> weights(kWeightsSize);
    
    if (LoadWeightsFromFile("weights.txt", weights)) {
        InitWeights(weights);
    } else {
        InitWeights();
    }
}

bool SequentialNetwork::LoadWeightsFromFile(const std::string& filename, std::vector<double>& weights) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error loading weights" << std::endl;
        return false;
    }
    
    std::string line;
    double weight;
    size_t i = 0;
    for (; i < weights.size(); ++i) {
        try {
            if (!std::getline(file, line)) {
                break;
            }

            weight = std::stod(line);
            weights[i] = weight;
        } catch (const std::exception& e) {
            std::cerr << "Error getting weight" << std::endl;
        }
    }
    
    file.close();

    if (i < weights.size()) {
        std::cerr << "Not enough weights in file" << std::endl;
        return false;
    }
    
    return true;
}

void SequentialNetwork::InitWeights(const std::vector<double>& weights) {
    if (!layers_.empty()) {
        layers_.clear();
    }
    
    if (weights.size() < kWeightsSize) {
        layers_.push_back(std::make_unique<Convolution2DLayer>(std::array<size_t, 3>{1, 28, 28}, std::array<size_t, 2>{3, 3}, 32));
        layers_.push_back(std::make_unique<Convolution2DLayer>(std::array<size_t, 3>{32, 26, 26}, std::array<size_t, 2>{3, 3}, 32));
        layers_.push_back(std::make_unique<MaxPooling2DLayer>(std::array<size_t, 3>{32, 24, 24}, std::array<size_t, 2>{2, 2}));
        layers_.push_back(std::make_unique<Convolution2DLayer>(std::array<size_t, 3>{32, 12, 12}, std::array<size_t, 2>{3, 3}, 64));
        layers_.push_back(std::make_unique<Convolution2DLayer>(std::array<size_t, 3>{64, 10, 10}, std::array<size_t, 2>{3, 3}, 64));
        layers_.push_back(std::make_unique<MaxPooling2DLayer>(std::array<size_t, 3>{64, 8, 8}, std::array<size_t, 2>{2, 2}));
        layers_.push_back(std::make_unique<FlattenLayer>(std::array<size_t, 3>{64, 4, 4}));
        layers_.push_back(std::make_unique<DenseLayer>(1024, 256, layers::Activation::kActivationReLU));
        layers_.push_back(std::make_unique<DenseLayer>(256, 128, layers::Activation::kActivationReLU));
        layers_.push_back(std::make_unique<DenseLayer>(128, 10, layers::Activation::kActivationSoftmax));
        return;
    }
    
    size_t weights_index = 0;
    
    Matrix4D kernels1(32, Matrix3D(1, Matrix2D(3, std::vector<double>(3))));
    for (size_t i = 0; i < 32; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                kernels1[i][0][j][k] = weights[weights_index++];
            }
        }
    }
    std::vector<double> shifts1(weights.begin() + weights_index, weights.begin() + weights_index + 32);
    weights_index += 32;
    layers_.push_back(std::make_unique<Convolution2DLayer>(std::array<size_t, 3>{1, 28, 28}, std::array<size_t, 2>{3, 3}, 32, layers::Activation::kActivationReLU, kernels1, shifts1));
    
    Matrix4D kernels2(32, Matrix3D(32, Matrix2D(3, std::vector<double>(3))));
    for (size_t i = 0; i < 32; ++i) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                for (size_t l = 0; l < 3; ++l) {
                    kernels2[i][j][k][l] = weights[weights_index++];
                }
            }
        }
    }
    std::vector<double> shifts2(weights.begin() + weights_index, weights.begin() + weights_index + 32);
    weights_index += 32;
    layers_.push_back(std::make_unique<Convolution2DLayer>(std::array<size_t, 3>{32, 26, 26}, std::array<size_t, 2>{3, 3}, 32, layers::Activation::kActivationReLU, kernels2, shifts2));
    
    layers_.push_back(std::make_unique<MaxPooling2DLayer>(std::array<size_t, 3>{32, 24, 24}, std::array<size_t, 2>{2, 2}));
    
    Matrix4D kernels3(64, Matrix3D(32, Matrix2D(3, std::vector<double>(3))));
    for (size_t i = 0; i < 64; ++i) {
        for (size_t j = 0; j < 32; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                for (size_t l = 0; l < 3; ++l) {
                    kernels3[i][j][k][l] = weights[weights_index++];
                }
            }
        }
    }
    std::vector<double> shifts3(weights.begin() + weights_index, weights.begin() + weights_index + 64);
    weights_index += 64;
    layers_.push_back(std::make_unique<Convolution2DLayer>(std::array<size_t, 3>{32, 12, 12}, std::array<size_t, 2>{3, 3}, 64, layers::Activation::kActivationReLU, kernels3, shifts3));
    
    Matrix4D kernels4(64, Matrix3D(64, Matrix2D(3, std::vector<double>(3))));
    for (size_t i = 0; i < 64; ++i) {
        for (size_t j = 0; j < 64; ++j) {
            for (size_t k = 0; k < 3; ++k) {
                for (size_t l = 0; l < 3; ++l) {
                    kernels4[i][j][k][l] = weights[weights_index++];
                }
            }
        }
    }
    std::vector<double> shifts4(weights.begin() + weights_index, weights.begin() + weights_index + 64);
    weights_index += 64;
    layers_.push_back(std::make_unique<Convolution2DLayer>(std::array<size_t, 3>{64, 10, 10}, std::array<size_t, 2>{3, 3}, 64, layers::Activation::kActivationReLU, kernels4, shifts4));    
    
    layers_.push_back(std::make_unique<MaxPooling2DLayer>(std::array<size_t, 3>{64, 8, 8}, std::array<size_t, 2>{2, 2}));
    
    layers_.push_back(std::make_unique<FlattenLayer>(std::array<size_t, 3>{64, 4, 4}));
    
    Matrix2D weights5(256, std::vector<double>(1024));
    for (size_t i = 0; i < 256; ++i) {
        for (size_t j = 0; j < 1024; ++j) {
            weights5[i][j] = weights[weights_index++];
        }
    }
    std::vector<double> shifts5(weights.begin() + weights_index, weights.begin() + weights_index + 256);
    weights_index += 256;
    layers_.push_back(std::make_unique<DenseLayer>(1024, 256, layers::Activation::kActivationReLU, weights5, shifts5));
    
    Matrix2D weights6(128, std::vector<double>(256));
    for (size_t i = 0; i < 128; ++i) {
        for (size_t j = 0; j < 256; ++j) {
            weights6[i][j] = weights[weights_index++];
        }
    }
    std::vector<double> shifts6(weights.begin() + weights_index, weights.begin() + weights_index + 128);
    weights_index += 128;
    layers_.push_back(std::make_unique<DenseLayer>(256, 128, layers::Activation::kActivationReLU, weights6, shifts6));
    
    Matrix2D weights7(10, std::vector<double>(128));
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 128; ++j) {
            weights7[i][j] = weights[weights_index++];
        }
    }
    std::vector<double> shifts7(weights.begin() + weights_index, weights.begin() + weights_index + 10);
    weights_index += 10;
    layers_.push_back(std::make_unique<DenseLayer>(128, 10, layers::Activation::kActivationSoftmax, weights7, shifts7));
    
    std::cout << "loaded " << weights_index << " weights" << std::endl;
}

std::vector<double> SequentialNetwork::Feedforward(const Matrix3D& input) {
    return layers_[9]->FeedforwardDense(layers_[8]->FeedforwardDense(layers_[7]->FeedforwardDense(layers_[6]->FeedforwardFlat(layers_[5]->Feedforward(layers_[4]->Feedforward(layers_[3]->Feedforward(layers_[2]->Feedforward(layers_[1]->Feedforward(layers_[0]->Feedforward(input))))))))));
}