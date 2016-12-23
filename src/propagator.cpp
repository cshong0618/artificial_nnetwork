#include "../include/propagator.h"

void ann::Propagator::SetActivationFunction(std::function<double(const double&)> f)
{
    activation_function = f;
}

double ann::Propagator::ActivationFunction(const double& net) const
{
    return activation_function(net);
}

void ann::Propagator::SetForwardPropogateFunction(std::function<double(int, int, std::vector<double>)> f)
{
    this->forward_propagate = f;
}

double ann::Propagator::ForwardPropagate(int layer, int node, std::vector<double> input) const
{
    return forward_propagate(layer, node, input);
}

void ann::Propagator::SetNNetwork(ann::ANN& nnetwork)
{
    this->nnetwork = nnetwork;
}
