#pragma once

#include "layers.hpp"
#include <vector>
#include <memory>
#include <string>

using namespace layers;

class SequentialNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers_;
    
    bool LoadWeightsFromFile(const std::string& filename, std::vector<double>& weights);
    void InitWeights(const std::vector<double>& weights = {});

public:
    SequentialNetwork();
    ~SequentialNetwork() = default;
    
    std::vector<double> Feedforward(const Matrix3D& input);
};
