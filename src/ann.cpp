#include "../include/ann.h"

/*
    AddLayer Method.
*/
void ann::ANN::AddLayer()
{
    t_layer l;
    nnetwork.push_back(l);
}


/*
    AddNode Method
*/
bool ann::ANN::AddNode(int layer)
{
    if(layer >= 0 && layer < (int)nnetwork.size())
    {
        t_weight n;
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
                this->nnetwork.at(layer).at(node).at(n) = weight;
                return true;
            }
        }
    }

    return false;
}

void ann::ANN::SetRawNNetwork(const ann::network& nnetwork)
{
    this->nnetwork = nnetwork;
}

void ann::ANN::SetRawNode(const std::vector<struct node>& input)
{
    this->raw_node = input;
}

void ann::ANN::AddTrainingSet(const std::vector<double>& input,
                              const std::vector<double>& output)
{
    this->training_set[input] = output;
}

void ann::ANN::AddTrainingSet(const std::vector<node>& input,
                              const std::vector<double>& output)
{
    std::vector<double> temp;
    for(auto n : input)
    {
        temp.push_back(n.val);
    }

    this->AddTrainingSet(temp, output);
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
