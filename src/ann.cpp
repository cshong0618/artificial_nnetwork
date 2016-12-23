#include "../include/ann.h"

/*
    AddLayer Method.
*/
void ann::ANN::AddLayer()
{
    layer l;
    nnetwork.push_back(l);
}


/*
    AddNode Method
*/
bool ann::ANN::AddNode(int layer)
{
    if(layer >= 0 && layer < (int)nnetwork.size())
    {
        node n;
        nnetwork.at(layer).push_back(n);
        return true;
    }

    return false;
}


/*
    AddWeight Method.

    Used for controlled iterated initialization helper.
*/
bool ann::ANN::AddWeight(int layer, int node, const double& weight)
{
    if(layer >= 0 && layer < (int)nnetwork.size())
    {
        if(node >= 0 && node < (int)nnetwork.at(layer).size())
        {
            nnetwork.at(layer).at(node).push_back(weight);
            return true;
        }
    }

    return false;
}

/*
    SetWeight Method.

    Used to append weight
*/
bool ann::ANN::SetWeight(int layer, int node, int n, const double& weight)
{
    if(layer >= 0 && layer < (int)nnetwork.size())
    {
        if(node >= 0 && node < (int)nnetwork.at(layer).size())
        {
            if(n >= 0 && n < (int)nnetwork.at(layer).at(node).size())
            {
                nnetwork.at(layer).at(node).at(n) = weight;
                return true;
            }
        }
    }

    return false;
}

void ann::ANN::AddTrainingSet(const std::vector<double>& input,
                              const std::vector<double>& output)
{
    this->training_set[input] = output;
}

/*
    SetSigmoidFunction

    Used to set custom sigmoid function
*/
void ann::ANN::SetActivationFunction(std::function<double(const double& net)>f)
{
    this->sigmoid_function = f;
}

double ann::ANN::ActivationValue(const double& net) const
{
    return sigmoid_function(net);
}

void ann::ANN::SetErrorMargin(const double& error_margin)
{
    this->error_margin = error_margin;
}


void ann::ANN::SetLearningRate(const double& learning_rate)
{
    this->learning_rate = learning_rate;
}

double ann::ANN::ForwardPropagate(int layer, int node, std::vector<double> inputs)
{
    double net = 0.0;
    try
    {
        for(size_t i = 0; i < inputs.size(); i++)
        {
            net += nnetwork.at(layer).at(node).at(i) * inputs.at(i);
        }
    }
    catch (std::exception e)
    {
        throw e;
    }

    return net;
}

/*
    FinalBackwardPropagate Method.

    Calculates delta(W)kj
    net = net_j
    actual = a_k
    target = t_k

    User should plug in values rather using it automatically.
    Algorithm to run it is up to the user.

*/
double ann::ANN::FinalBackwardPropagate(double net, double actual, double target)
{
    double delta = 0.0;

    try
    {
        delta = learning_rate *
                ((target - actual) * ActivationValue(target) * (1 - ActivationValue(target))) *
                ActivationValue(net);
    }
    catch (std::exception e)
    {
        throw e;
    }

    return delta;
}
