#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <string>

namespace layers {
    const std::string kActivationReLU = "relu";
    const std::string kActivationSoftmax = "softmax";
};

using Matrix2D = std::vector<std::vector<double>>;
using Matrix3D = std::vector<std::vector<std::vector<double>>>;
using Matrix4D = std::vector<std::vector<std::vector<std::vector<double>>>>;

class Layer {
public:
    virtual ~Layer() = default;
    virtual std::vector<double> FeedforwardDense(const std::vector<double>& input) const { return {}; }
    virtual Matrix3D Feedforward(const Matrix3D& input) const { return {}; }
    virtual std::vector<double> FeedforwardFlat(const Matrix3D& input) const { return {}; }
};

class Convolution2DLayer : public Layer {
private:
    size_t filters_count_;
    std::tuple<size_t, size_t, size_t> input_size_;
    std::tuple<size_t, size_t> kernel_size_;
    std::tuple<size_t, size_t, size_t> output_size_;
    Matrix4D kernels_;
    std::vector<double> shifts_;

public:
    Convolution2DLayer(std::tuple<size_t, size_t, size_t>& input_size, std::tuple<size_t, size_t>& kernel_size, size_t filters_count, const std::string& activation = layers::kActivationReLU,
                      const Matrix4D& kernels = {}, const std::vector<double>& shifts = {});
    
    Matrix3D Feedforward(const Matrix3D& input) const override;
};

class MaxPooling2DLayer : public Layer {
private:
    std::tuple<size_t, size_t, size_t> input_size_;
    std::tuple<size_t, size_t> kernel_size_;
    std::tuple<size_t, size_t, size_t> output_size_;

public:
    MaxPooling2DLayer(std::tuple<size_t, size_t, size_t>& input_size, std::tuple<size_t, size_t>& kernel_size);
    
    Matrix3D Feedforward(const Matrix3D& input) const override;
};

class FlattenLayer : public Layer {
private:
    std::tuple<size_t, size_t, size_t> input_size_;
    size_t output_size_;

public:
    FlattenLayer(std::tuple<size_t, size_t, size_t>& input_size);
    
    std::vector<double> FeedforwardFlat(const Matrix3D& input) const override;
};

class DenseLayer : public Layer {
private:
    size_t input_size_, output_size_;
    Matrix2D weights_;
    std::vector<double> shifts_;
    std::string activation_;

public:
    DenseLayer(size_t input_size, size_t output_size, const std::string& activation = layers::kActivationReLU,
              const Matrix2D& weights = {}, const std::vector<double>& shifts = {});
    
    std::vector<double> FeedforwardDense(const std::vector<double>& input) const override;
};
