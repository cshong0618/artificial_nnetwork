#include "../include/propagator.h"

void ann::Propagator::SetActivationFunction(std::function<double(const double&)> f)
{
    activation_function = f;
}

void ann::Propagator::ResetActivationFunction()
{
    this->activation_function = [](const double& net)
    {
        return (1 / (1 + exp(-net)));
    };
}

double ann::Propagator::ActivationFunction(const double& net) const
{
    return activation_function(net);
}

void ann::Propagator::SetForwardPropagateFunction(std::function<double(int, int, std::vector<double>)> f)
{
    this->forward_propagate = f;
}

void ann::Propagator::ResetForwardPropagateFunction()
{
    this->forward_propagate = [&](int layer, int node, std::vector<double> inputs)
    {
        double net = 0.0;

        try
        {
            for(size_t i = 0; i < inputs.size(); i++)
            {
                net += (this->nnetwork.GetWeight(layer, node, i) * inputs.at(i));
            }
        }
        catch (std::exception e)
        {
            throw e;
        }
        return net;
    };
}

double ann::Propagator::ForwardPropagate(int layer, int node, std::vector<double> input) const
{
    return forward_propagate(layer, node, input);
}

std::vector<double> ann::Propagator::AutoForwardPropagate(std::vector<double> input)
{
    std::vector<double> temp;
    for(int i = 0; i < nnetwork.GetLayerCount(); i++)
    {
        temp.clear();
        for(int j = 0; j <nnetwork.GetLayer(i).size(); j++)
        {
            temp.push_back(nnetwork.ForwardPropagate(i, j, input));
        }
        input = temp;
    }

    return temp;
}

void ann::Propagator::SetSmallChangeFunction(std::function<double(const double&, const double&)> f)
{
    this->s_change = f;
}

void ann::Propagator::ResetSmallChangeFunction()
{
    this->s_change = [&](const double& target, const double& actual)
    {
        double a = activation_function(actual);
        return ((target - a) * a * (1 - a));
    };
}

double ann::Propagator::SmallChangeFunction(const double& target, const double& actual) const
{
    return s_change(target, actual);
}

void ann::Propagator::SetNNetwork(ann::ANN& nnetwork)
{
    this->nnetwork = nnetwork;
}
