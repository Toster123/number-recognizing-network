#include "layers.hpp"

namespace {
    double ReLU(double x) {
        return std::max(0.0, x);
    }
    
    std::vector<double> Softmax(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        double max_val = *std::max_element(x.begin(), x.end());//MARK: tst
        
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_val);
            sum += result[i];
        }
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] /= sum;
        }
        
        return result;
    };
};

Convolution2DLayer::Convolution2DLayer(std::tuple<size_t, size_t, size_t>& input_size, std::tuple<size_t, size_t>& kernel_size, size_t filters_count, const std::string& activation,
                                      const Matrix4D& kernels, const std::vector<double>& shifts)
    : filters_count_(filters_count), input_size_(input_size), kernel_size_(kernel_size) {
    
    output_size_ = {filters_count_, 1 + std::get<1>(input_size_) - std::get<0>(kernel_size_), 1 + std::get<2>(input_size_) - std::get<1>(kernel_size_)};
    
    if (kernels.empty()) {
        std::fill_n(std::back_inserter(kernels_), filters_count_, Matrix3D(std::get<0>(input_size_), Matrix2D(std::get<0>(kernel_size), std::vector<double>(std::get<1>(kernel_size)))));
        
        // std::cout << kernels_.size() << " " << kernels_[0].size() << " " << kernels_[0][0].size() << " " << kernels_[0][0][0].size() << std::endl;
        // std::cout << kernels_[4][0][0][0] << std::endl;
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
    Matrix3D output(std::get<0>(output_size_), Matrix2D(std::get<1>(output_size_), std::vector<double>(std::get<2>(output_size_))));
    for (int f = 0; f < std::get<0>(output_size_); ++f) {
        for (int h = 0; h < std::get<1>(output_size_); ++h) {
            for (int w = 0; w < std::get<2>(output_size_); ++w) {
                double sum = 0.0;
                for (int c = 0; c < std::get<0>(input_size_); ++c) {
                    for (int kh = 0; kh < std::get<0>(kernel_size_); ++kh) {
                        for (int kw = 0; kw < std::get<1>(kernel_size_); ++kw) {
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

MaxPooling2DLayer::MaxPooling2DLayer(std::tuple<size_t, size_t, size_t>& input_size, std::tuple<size_t, size_t>& kernel_size)
    : input_size_(input_size), kernel_size_(kernel_size) {
    
    output_size_ = {std::get<0>(input_size_), std::get<1>(input_size_) / std::get<0>(kernel_size_), std::get<2>(input_size_) / std::get<1>(kernel_size_)};
}

Matrix3D MaxPooling2DLayer::Feedforward(const Matrix3D& input) const {
    Matrix3D output(std::get<0>(output_size_), Matrix2D(std::get<1>(output_size_), std::vector<double>(std::get<2>(output_size_))));
    for (int c = 0; c < std::get<0>(input_size_); ++c) {
        for (int h = 0; h < std::get<1>(output_size_); ++h) {
            for (int w = 0; w < std::get<2>(output_size_); ++w) {
                double value = input[c][std::get<0>(kernel_size_) * h][std::get<1>(kernel_size_) * w];
                for (int kh = 0; kh < std::get<0>(kernel_size_); ++kh) {
                    for (int kw = 0; kw < std::get<1>(kernel_size_); ++kw) {
                        value = std::max(value, input[c][std::get<0>(kernel_size_) * h + kh][std::get<1>(kernel_size_) * w + kw]);
                    }
                }
                output[c][h][w] = value;
            }
        }
    }
    return output;
}

FlattenLayer::FlattenLayer(std::tuple<size_t, size_t, size_t>& input_size)
    : input_size_(input_size) {
    
    output_size_ = std::get<0>(input_size_) * std::get<1>(input_size_) * std::get<2>(input_size_);
}

std::vector<double> FlattenLayer::FeedforwardFlat(const Matrix3D& input) const {
    std::vector<double> output(output_size_);
    int i = 0;

    // (height, width, channels)
    for (int h = 0; h < std::get<1>(input_size_); ++h) {
        for (int w = 0; w < std::get<2>(input_size_); ++w) {
            for (int c = 0; c < std::get<0>(input_size_); ++c) {
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
        output[j] = 0.0; //MARK: tst
        for (int i = 0; i < input_size_; ++i) {
            output[j] += input[i] * weights_[j][i];
        }
        
        if (activation_ == layers::kActivationReLU) {
            output[j] = ReLU(output[j] + shifts_[j]);
        } else if (activation_ == layers::kActivationSoftmax) {
            output[j] += shifts_[j];
        }
    }
    
    if (activation_ == layers::kActivationSoftmax) {
        output = Softmax(output);
    }
    
    return output;
}
