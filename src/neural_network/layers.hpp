#pragma once

#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <string>
#include <array>

namespace layers {
    enum Activation {
        kActivationReLU,
        kActivationSoftmax
    };
};

using namespace layers;
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
public:
    Convolution2DLayer(const std::array<size_t, 3>& input_size, const std::array<size_t, 2>& kernel_size, size_t filters_count, const Activation& activation = kActivationReLU,
                      const Matrix4D& kernels = {}, const std::vector<double>& shifts = {});
    
    Matrix3D Feedforward(const Matrix3D& input) const override;

private:
    size_t filters_count_;
    std::array<size_t, 3> input_size_;
    std::array<size_t, 2> kernel_size_;
    std::array<size_t, 3> output_size_;
    Matrix4D kernels_;
    std::vector<double> shifts_;
    Activation activation_;
};

class MaxPooling2DLayer : public Layer {
public:
    MaxPooling2DLayer(const std::array<size_t, 3>& input_size, const std::array<size_t, 2>& kernel_size);
    
    Matrix3D Feedforward(const Matrix3D& input) const override;

private:
    std::array<size_t, 3> input_size_;
    std::array<size_t, 2> kernel_size_;
    std::array<size_t, 3> output_size_;
};

class FlattenLayer : public Layer {
public:
    FlattenLayer(const std::array<size_t, 3>& input_size);
    
    std::vector<double> FeedforwardFlat(const Matrix3D& input) const override;

private:
    std::array<size_t, 3> input_size_;
    size_t output_size_;
};

class DenseLayer : public Layer {
public:
    DenseLayer(size_t input_size, size_t output_size, const Activation& activation = kActivationReLU,
              const Matrix2D& weights = {}, const std::vector<double>& shifts = {});
    
    std::vector<double> FeedforwardDense(const std::vector<double>& input) const override;

private:
    size_t input_size_, output_size_;
    Matrix2D weights_;
    std::vector<double> shifts_;
    Activation activation_;
};
