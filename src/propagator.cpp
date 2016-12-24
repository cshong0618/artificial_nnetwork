#include "../include/propagator.h"

void ann::Propagator::SetActivationFunction(std::function<double(const double&)> f)
{
    activation_function = f;
}

void ann::Propagator::ResetActivationFunction()
{
    this->activation_function = [](const double& net)
    {
        //std::cout << "\tnet: " << net << std::endl;
        double ac_f = (1 / (1 + exp(-net)));
        //std::cout << "\tac_f: " << ac_f << std::endl;
        return ac_f;
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

        for(size_t i = 0; i < inputs.size(); i++)
        {
            net += (this->nnetwork.GetWeight(layer, node, i) * inputs.at(i));
            std::cout << "\tnet at " << layer << " " << node << " " << i << ": " << net << "\n";
        }


        if(net == 0.0)
        {
            net = 1;
        }



        std::cout << "\t\tfinal net: " << net << "\n";

        return net;
    };
}

double ann::Propagator::ForwardPropagate(int layer, int node, std::vector<double> input) const
{
    return forward_propagate(layer, node, input);
}

ann::node_network ann::Propagator::AutoForwardPropagate(std::vector<double> input)
{
    ann::node_network temp;
    temp.push_back(ann::node_layer());
    temp.at(0) = input;
    for(int i = 0; i < nnetwork.GetLayerCount(); i++)
    {
        temp.push_back(ann::node_layer());
        for(int j = 0; j < nnetwork.GetLayer(i).size(); j++)
        {
            temp.at(i + 1).push_back(this->ForwardPropagate(i, j, input));
            std::cout << "temp.at(" << i + 1 << ").at(" << j << "): " << temp.at(i+1).at(j) << std::endl;
        }
        input = temp.at(i + 1);
    }

    return temp;
}

void ann::Propagator::SetBackwardPropagateFunction(std::function<double (const double&, const double &)> backward_propagate)
{
    this->backward_propagate = backward_propagate;
}

void ann::Propagator::ResetBackwardPropagateFunction()
{
    this->backward_propagate = [&](const double& s_change,
                                   const double& net)
    {
        return this->nnetwork.GetLearningRate() * s_change * ActivationFunction(net);
    };
}

double ann::Propagator::BackwardPropagate(const double& s_change,const double &net) const
{
    return this->backward_propagate(s_change, net);
}

ann::network ann::Propagator::AutoBackwardPropagate(const ann::node_network& nets,
                                                    std::vector<double> target)
{
    /*
        Here, we might want to work with the raw neural network...
    */
    ann::ANN initial_network = this->GetNNetwork();
    ann::network delta = initial_network.GetRawNNetwork();
    ann::network small_changes = delta;
    //std::cout << "Learning rate: " << this->nnetwork.GetLearningRate() << std::endl;
    /*
        Straight run preparation
    */

    int last = initial_network.GetLayerCount() - 1;
    for(int i = 0; i < initial_network.GetLayer(last).size(); i++)
    {
        for(int j = 0; j < initial_network.GetNode(last, i).size(); j++)
        {
            //std::cout << "lbp " << j << "\n";
            //std::cout << "target.at(i): " << target.at(i) << std::endl;
            //std::cout << "initial_network.GetWeight(last, i, j): " << initial_network.GetWeight(last, i, j) <<std::endl;
            double sc = this->SmallChangeFunction(target.at(i), nets.at(last - 1).at(i));
            small_changes.at(last).at(i).at(j) = sc;
            //std::cout << "sc: " << sc << std::endl;
            // std::cout << "nets.at(last-1).at(i): " << nets.at(last).at(j) << std::endl;
            // std::cin.ignore();
            double val = this->BackwardPropagate(sc, nets.at(last).at(j));
            //std::cout << "val: " << val << std::endl;
            delta.at(last).at(i).at(j) = val;
        }
    }

    for(int i = last - 1; i >= 0; --i)
    {
        //std::cout << "Layer count: " << initial_network.GetLayerCount() << std::endl;
        for(int j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            //std::cout << "GetLayer size: " << initial_network.GetLayer(i).size() << "\n";
            for(int k = 0; k < initial_network.GetNode(i, j).size(); k++)
            {
                //std::cout << "GetNode size: " << initial_network.GetNode(i, j).size() << "\n";
                //std::cout << "i: " << i << " j: " << j << " k: " << k << "\n";
                double sc;
                std::vector<double> prev_small_change;
                std::vector<double> prev_weight;
                for(int _i = 0; _i < delta.at(i + 1).size(); _i++)
                {
                    prev_small_change.push_back(small_changes.at(i + 1).at(_i).at(j));
                    prev_weight.push_back(initial_network.GetWeight(i + 1, _i, j));
                }

                sc = this->HiddenSmallChangeFunction(prev_small_change, prev_weight, nets.at(i).at(j));
                // std::cout << "sc: " << sc << std::endl;

                small_changes.at(i).at(j).at(k) = sc;
                // std::cout << "nets.at...: " << nets.at(i).at(k) << std::endl;
                // std::cin.ignore();
                double val = this->BackwardPropagate(sc, nets.at(i).at(k));
                // std::cout << "val: " << val << std::endl;
                delta.at(i).at(j).at(k) = val;
            }
        }
    }

    std::cout << initial_network << std::endl;

    for(int i = 0; i < initial_network.GetLayerCount(); i++)
    {
        for(int j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            for(int k = 0; k < initial_network.GetNode(i, j).size(); k++)
            {
                // std::cout << small_changes.at(i).at(j).at(k) << std::endl;
                double new_weight = 0.0;

                // Hack for bias where it do not have previous weights
                if(initial_network.GetWeight(i, j, k) != 0)
                {
                    new_weight = initial_network.GetWeight(i, j, k) + delta.at(i).at(j).at(k);
                }

                initial_network.SetWeight(i, j, k, new_weight);
            }
        }
    }

    std::cout << "Small changes:\n";
    for(int i = 0; i < initial_network.GetLayerCount(); i++)
    {
        for(int j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            for(int k = 0; k < initial_network.GetNode(i, j).size(); k++)
            {
                std::cout << small_changes.at(i).at(j).at(k) << std::endl;
            }
        }
    }
    std::cout << "\nDelta:\n";
    for(int i = 0; i < initial_network.GetLayerCount(); i++)
    {
        for(int j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            for(int k = 0; k < initial_network.GetNode(i, j).size(); k++)
            {
                std::cout << delta.at(i).at(j).at(k) << std::endl;
            }
        }
    }

    std::cout << initial_network << std::endl;

    return initial_network.GetRawNNetwork();
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
        double small_change = (double)((target - a) * a * (1 - a));
        return small_change;
    };
}

double ann::Propagator::SmallChangeFunction(const double& target, const double& actual) const
{
    return s_change(target, actual);
}

void ann::Propagator::SetHiddenSmallChangeFunction(std::function<double(std::vector<double>, std::vector<double>, const double&)> hidden_s_change)
{
    this->hidden_s_change = hidden_s_change;
}

void ann::Propagator::ResetHiddenSmallChangeFunction()
{
    this->hidden_s_change = [&](std::vector<double> prev_small_change, std::vector<double> weight, const double& actual)
    {
        double change = 0.0;
        for(int i = 0; i < (int)prev_small_change.size(); i++)
        {
            change += (prev_small_change.at(i) * weight.at(i));
        }
        double a = activation_function(actual);
        change *= (a * (1 - a));
        return change;
    };
}

double ann::Propagator::HiddenSmallChangeFunction(std::vector<double> prev_small_change, std::vector<double> weight, const double& actual) const
{
    return this->hidden_s_change(prev_small_change, weight, actual);
}

void ann::Propagator::SetNNetwork(ann::ANN& nnetwork)
{
    this->nnetwork = nnetwork;
}
