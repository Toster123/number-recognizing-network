#include "layers.hpp"
#include <iostream>

namespace {
    double ReLU(double x) {
        return std::max(0.0, x);
    }
    
    std::vector<double> Softmax(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        double max_value = *std::max_element(x.begin(), x.end());
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_value);
            sum += result[i];
        }
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] /= sum;
        }
        
        return result;
    };
};

Convolution2DLayer::Convolution2DLayer(const std::array<size_t, 3>& input_size, const std::array<size_t, 2>& kernel_size, size_t filters_count, const std::string& activation,
                                      const Matrix4D& kernels, const std::vector<double>& shifts)
    : filters_count_(filters_count), input_size_(input_size), kernel_size_(kernel_size) {
    
    output_size_ = {filters_count_, 1 + input_size_[1] - kernel_size_[0], 1 + input_size_[2] - kernel_size_[1]};
    
    if (kernels.empty()) {
        std::fill_n(std::back_inserter(kernels_), filters_count_, Matrix3D(input_size_[0], Matrix2D(kernel_size_[0], std::vector<double>(kernel_size_[1]))));
    } else {
        kernels_ = kernels;
    }
    
    if (shifts.empty()) {
        std::fill_n(std::back_inserter(shifts_), filters_count_, 0.0);
    } else {
        shifts_ = shifts;
    }
}

Matrix3D Convolution2DLayer::Feedforward(const Matrix3D& input) const {
    Matrix3D output(output_size_[0], Matrix2D(output_size_[1], std::vector<double>(output_size_[2])));
    for (int f = 0; f < output_size_[0]; ++f) {
        for (int h = 0; h < output_size_[1]; ++h) {
            for (int w = 0; w < output_size_[2]; ++w) {
                double sum = 0.0;
                for (int c = 0; c < input_size_[0]; ++c) {
                    for (int kh = 0; kh < kernel_size_[0]; ++kh) {
                        for (int kw = 0; kw < kernel_size_[1]; ++kw) {
                            sum += kernels_[f][c][kh][kw] * input[c][h + kh][w + kw];
                        }
                    }
                }
                output[f][h][w] = ReLU(sum + shifts_[f]);
            }
        }
    }
    return output;
}

MaxPooling2DLayer::MaxPooling2DLayer(const std::array<size_t, 3>& input_size, const std::array<size_t, 2>& kernel_size)
    : input_size_(input_size), kernel_size_(kernel_size) {
    
    output_size_ = {input_size_[0], input_size_[1] / kernel_size_[0], input_size_[2] / kernel_size_[1]};
}

Matrix3D MaxPooling2DLayer::Feedforward(const Matrix3D& input) const {
    Matrix3D output(output_size_[0], Matrix2D(output_size_[1], std::vector<double>(output_size_[2])));
    for (int c = 0; c < input_size_[0]; ++c) {
        for (int h = 0; h < output_size_[1]; ++h) {
            for (int w = 0; w < output_size_[2]; ++w) {
                double value = input[c][kernel_size_[0] * h][kernel_size_[1] * w];
                for (int kh = 0; kh < kernel_size_[0]; ++kh) {
                    for (int kw = 0; kw < kernel_size_[1]; ++kw) {
                        value = std::max(value, input[c][kernel_size_[0] * h + kh][kernel_size_[1] * w + kw]);
                    }
                }
                output[c][h][w] = value;
            }
        }
    }
    return output;
}

FlattenLayer::FlattenLayer(const std::array<size_t, 3>& input_size)
    : input_size_(input_size) {
    
    output_size_ = input_size_[0] * input_size_[1] * input_size_[2];
}

std::vector<double> FlattenLayer::FeedforwardFlat(const Matrix3D& input) const {
    std::vector<double> output(output_size_);
    int i = 0;

    // (height, width, channels)
    for (int h = 0; h < input_size_[1]; ++h) {
        for (int w = 0; w < input_size_[2]; ++w) {
            for (int c = 0; c < input_size_[0]; ++c) {
                output[i++] = input[c][h][w];
            }
        }
    }

    return output;
}

DenseLayer::DenseLayer(size_t input_size, size_t output_size, const std::string& activation,
                      const Matrix2D& weights, const std::vector<double>& shifts)
    : input_size_(input_size), output_size_(output_size), activation_(activation) {
    
    if (weights.empty()) {
        std::fill_n(std::back_inserter(weights_), output_size_, std::vector<double>(input_size_));
    } else {
        weights_ = weights;
    }
    
    if (shifts.empty()) {
        std::fill_n(std::back_inserter(shifts_), output_size_, 0.0);
    } else {
        shifts_ = shifts;
    }
}

std::vector<double> DenseLayer::FeedforwardDense(const std::vector<double>& input) const {
    std::vector<double> output(output_size_);

    for (int j = 0; j < output_size_; ++j) {
        for (int i = 0; i < input_size_; ++i) {
            output[j] += input[i] * weights_[j][i];
        }
        
        if (activation_ == kActivationReLU) {
            output[j] = ReLU(output[j] + shifts_[j]);
        } else if (activation_ == kActivationSoftmax) {
            output[j] += shifts_[j];
        }
    }
    
    if (activation_ == kActivationSoftmax) {
        output = Softmax(output);
    }
    
    return output;
}
