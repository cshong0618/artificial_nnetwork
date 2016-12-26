#include "../include/propagator.h"

void ann::Propagator::SetActivationFunction(std::function<double(const double&)> f)
{
    activation_function = f;
}

void ann::Propagator::SetRawActivationFunction(std::function<double(const struct ann::node&)> f)
{
    raw_activation_function = f;
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

void ann::Propagator::ResetRawActivationFunction()
{
    this->raw_activation_function = [](const struct ann::node& net)
    {
        if(net.type == 'b')
        {
            return 1.0;
        }

        //std::cout << "\tnet: " << net << std::endl;
        double ac_f = (1 / (1 + exp(-net.val)));
        //std::cout << "\tac_f: " << ac_f << std::endl;
        return ac_f;
    };
}

double ann::Propagator::ActivationFunction(const double& net) const
{
    return activation_function(net);
}

double ann::Propagator::RawActivationFunction(const struct ann::node& net) const
{
    return raw_activation_function(net);
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
            std::cout << "Weight: " << this->nnetwork.GetWeight(layer, node, i) << std::endl;
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

void ann::Propagator::SetRawForwardPropagateFunction(std::function<struct ann::node(int, int, std::vector<double>)> f)
{
    this->r_forward_propagate = f;
}

void ann::Propagator::ResetRawForwardPropagateFunction()
{
    this->r_forward_propagate = [&](int layer, int node, std::vector<double> inputs)
    {
        double net = 0.0;
        struct ann::node _node;
        for(size_t i = 0; i < inputs.size(); i++)
        {
            // std::cout << "Weight: " << this->nnetwork.GetWeight(layer, node, i) << std::endl;
            net += (this->nnetwork.GetWeight(layer, node, i) * inputs.at(i));
        }

        _node.val = net;
        _node.type = 'n';
        if(net == 0.0)
        {
            net = 1;
            _node.val = 1;
            _node.type = 'b';
        }



        std::cout << "\t\tfinal net: " << net << "\n";

        return _node;
    };
}

struct ann::node ann::Propagator::RawForwardPropagate(int layer, int node, std::vector<double> input) const
{
    return this->r_forward_propagate(layer, node, input);
}

ann::node_network ann::Propagator::AutoForwardPropagate(std::vector<double> input)
{
    ann::node_network temp;
    ann::raw_node_network r_temp;
    temp.push_back(ann::node_layer());
    r_temp.push_back(ann::raw_node_layer());
    temp.at(0) = input;
    r_temp.at(0) = this->GetNNetwork().GetRawNode();
    for(size_t i = 0; i < nnetwork.GetLayerCount(); i++)
    {
        temp.push_back(ann::node_layer());
        r_temp.push_back(ann::raw_node_layer());
        for(size_t j = 0; j < nnetwork.GetLayer(i).size(); j++)
        {
            double new_net = this->ForwardPropagate(i, j, input);
            temp.at(i + 1).push_back(new_net);
            r_temp.at(i + 1).push_back(this->RawForwardPropagate(i, j, input));
            std::cout << "temp.at(" << i + 1 << ").at(" << j << "): " << temp.at(i+1).at(j) << std::endl;
        }
        input = temp.at(i + 1);
        for(auto v : input)
        {
            std::cout << "new input at " << i << " " << v << std::endl;
        }
    }

    return temp;
}

ann::raw_node_network ann::Propagator::RawAutoForwardPropagate(std::vector<double> input)
{
    ann::raw_node_network r_temp;
    r_temp.push_back(ann::raw_node_layer());
    r_temp.at(0) = this->GetNNetwork().GetRawNode();
    for(size_t i = 0; i < nnetwork.GetLayerCount(); i++)
    {
        r_temp.push_back(ann::raw_node_layer());
        for(size_t j = 0; j < nnetwork.GetLayer(i).size(); j++)
        {
            r_temp.at(i + 1).push_back(this->RawForwardPropagate(i, j, input));
        }

        std::vector<double> dtemp;
        for(auto n : r_temp.at(i + 1))
        {
            dtemp.push_back(n.val);
        }

        input = dtemp;
    }

    return r_temp;
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
        std::cout << "bp: " << net << std::endl;
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
    for(size_t i = 0; i < initial_network.GetLayer(last).size(); i++)
    {
        for(size_t j = 0; j < initial_network.GetNode(last, i).size(); j++)
        {
            //std::cout << "lbp " << j << "\n";
            //std::cout << "target.at(i): " << target.at(i) << std::endl;
            //std::cout << "initial_network.GetWeight(last, i, j): " << initial_network.GetWeight(last, i, j) <<std::endl;
            double sc = this->SmallChangeFunction(target.at(i), nets.at(last + 1).at(i));
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
        for(size_t j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            //std::cout << "GetLayer size: " << initial_network.GetLayer(i).size() << "\n";
            for(size_t k = 0; k < initial_network.GetNode(i, j).size(); k++)
            {
                //std::cout << "GetNode size: " << initial_network.GetNode(i, j).size() << "\n";
                std::cout << "i: " << i << " j: " << j << " k: " << k << "\n";
                double sc;
                std::vector<double> prev_small_change;
                std::vector<double> prev_weight;
                double prev_net;

                std::exception_ptr eptr;
                for(size_t _i = 0; _i < delta.at(i + 1).size(); _i++)
                {
                    std::cout << "prev_small_change: " << small_changes.at(i + 1).at(_i).at(j) << std::endl;
                    prev_small_change.push_back(small_changes.at(i + 1).at(_i).at(j));

                    std::cout << "prev_weight: " << initial_network.GetWeight(i + 1, _i, j) << std::endl;
                    prev_weight.push_back(initial_network.GetWeight(i + 1, _i, j));

                }


                if(k < nets.at(j).size())
                {
                    prev_net = nets.at(i + 1).at(j);
                }
                else
                {
                    prev_net = 0.0;
                }

                sc = this->HiddenSmallChangeFunction(prev_small_change, prev_weight, prev_net);
                // std::cout << "sc: " << sc << std::endl;

                small_changes.at(i).at(j).at(k) = sc;
                std::cout << "nets.at...: " << nets.at(i).at(k) << std::endl;
                // std::cin.ignore();
                double val = this->BackwardPropagate(sc, nets.at(i).at(k));
                // std::cout << "val: " << val << std::endl;
                delta.at(i).at(j).at(k) = val;
            }
        }
    }

    std::cout << initial_network << std::endl;

    for(size_t i = 0; i < initial_network.GetLayerCount(); i++)
    {
        for(size_t j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            for(size_t k = 0; k < initial_network.GetNode(i, j).size(); k++)
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
    for(size_t i = 0; i < initial_network.GetLayerCount(); i++)
    {
        for(size_t j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            for(size_t k = 0; k < initial_network.GetNode(i, j).size(); k++)
            {
                std::cout << small_changes.at(i).at(j).at(k) << std::endl;
            }
        }
    }
    std::cout << "\nDelta:\n";
    for(size_t i = 0; i < initial_network.GetLayerCount(); i++)
    {
        for(size_t j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            for(size_t k = 0; k < initial_network.GetNode(i, j).size(); k++)
            {
                std::cout << delta.at(i).at(j).at(k) << std::endl;
            }
        }
    }

    std::cout << initial_network << std::endl;

    return initial_network.GetRawNNetwork();
}

void ann::Propagator::SetRawBackwardPropagateFunction(std::function<double (const double&, const struct ann::node &)> f)
{
    this->raw_backward_propagate = f;
}

void ann::Propagator::ResetRawBackwardPropagateFunction()
{
    this->raw_backward_propagate = [&](const double& s_change,
                                       const struct ann::node& net)
    {
        // std::cout << "bp: " << net << std::endl;
        return this->nnetwork.GetLearningRate() * s_change * this->RawActivationFunction(net);
    };
}

double ann::Propagator::RawBackwardPropagate(const double& s_change,const struct ann::node &net) const
{
    return this->raw_backward_propagate(s_change, net);
}


ann::network ann::Propagator::RawAutoBackwardPropagate(const ann::raw_node_network& nets,
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
    for(size_t i = 0; i < initial_network.GetLayer(last).size(); i++)
    {
        for(size_t j = 0; j < initial_network.GetNode(last, i).size(); j++)
        {
            //std::cout << "lbp " << j << "\n";
            //std::cout << "target.at(i): " << target.at(i) << std::endl;
            //std::cout << "initial_network.GetWeight(last, i, j): " << initial_network.GetWeight(last, i, j) <<std::endl;
            double sc = this->RawSmallChangeFunction(target.at(i), nets.at(last + 1).at(i));
            small_changes.at(last).at(i).at(j) = sc;
            //std::cout << "sc: " << sc << std::endl;
            // std::cout << "nets.at(last-1).at(i): " << nets.at(last).at(j) << std::endl;
            // std::cin.ignore();
            double val = this->RawBackwardPropagate(sc, nets.at(last).at(j));
            //std::cout << "val: " << val << std::endl;
            delta.at(last).at(i).at(j) = val;
        }
    }

    for(int i = last - 1; i >= 0; --i)
    {
        //std::cout << "Layer count: " << initial_network.GetLayerCount() << std::endl;
        for(size_t j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            //std::cout << "GetLayer size: " << initial_network.GetLayer(i).size() << "\n";
            for(size_t k = 0; k < initial_network.GetNode(i, j).size(); k++)
            {
                //std::cout << "GetNode size: " << initial_network.GetNode(i, j).size() << "\n";
                // std::cout << "i: " << i << " j: " << j << " k: " << k << "\n";
                double sc;
                std::vector<double> prev_small_change;
                std::vector<double> prev_weight;
                ann::node prev_net;

                std::exception_ptr eptr;
                for(size_t _i = 0; _i < delta.at(i + 1).size(); _i++)
                {
                    // std::cout << "prev_small_change: " << small_changes.at(i + 1).at(_i).at(j) << std::endl;
                    prev_small_change.push_back(small_changes.at(i + 1).at(_i).at(j));

                    // std::cout << "prev_weight: " << initial_network.GetWeight(i + 1, _i, j) << std::endl;
                    prev_weight.push_back(initial_network.GetWeight(i + 1, _i, j));

                }


                if(k < nets.at(j).size())
                {
                    prev_net = nets.at(i + 1).at(j);
                }
                else
                {
                    prev_net = {0.0, 'n'};
                }

                sc = this->RawHiddenSmallChangeFunction(prev_small_change, prev_weight, prev_net);
                // std::cout << "sc: " << sc << std::endl;

                small_changes.at(i).at(j).at(k) = sc;
                // std::cout << "nets.at...: " << nets.at(i).at(k) << std::endl;
                // std::cin.ignore();
                double val = this->RawBackwardPropagate(sc, nets.at(i).at(k));
                // std::cout << "val: " << val << std::endl;
                delta.at(i).at(j).at(k) = val;
            }
        }
    }

    std::cout << initial_network << std::endl;

    for(size_t i = 0; i < initial_network.GetLayerCount(); i++)
    {
        for(size_t j = 0; j < initial_network.GetLayer(i).size(); j++)
        {
            for(size_t k = 0; k < initial_network.GetNode(i, j).size(); k++)
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

    // std::cout << "Small changes:\n";
    // for(size_t i = 0; i < initial_network.GetLayerCount(); i++)
    // {
    //     for(size_t j = 0; j < initial_network.GetLayer(i).size(); j++)
    //     {
    //         for(size_t k = 0; k < initial_network.GetNode(i, j).size(); k++)
    //         {
    //             std::cout << small_changes.at(i).at(j).at(k) << std::endl;
    //         }
    //     }
    // }
    // std::cout << "\nDelta:\n";
    // for(size_t i = 0; i < initial_network.GetLayerCount(); i++)
    // {
    //     for(size_t j = 0; j < initial_network.GetLayer(i).size(); j++)
    //     {
    //         for(size_t k = 0; k < initial_network.GetNode(i, j).size(); k++)
    //         {
    //             std::cout << delta.at(i).at(j).at(k) << std::endl;
    //         }
    //     }
    // }

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
        std::cout << "target: " << target << std::endl;
        std::cout << "last small change: " <<  actual << std::endl;
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
        std::cout << "actual: " <<  actual << std::endl;
        double change = 0.0;
        for(size_t i = 0; i < prev_small_change.size(); i++)
        {
            // std::cout << "i: " << i << " || prev_small_change: " << prev_small_change.at(i) << std::endl;
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


void ann::Propagator::SetRawSmallChangeFunction(std::function<double(const double&, const struct ann::node&)> f)
{
    this->raw_s_change = f;
}

void ann::Propagator::ResetRawSmallChangeFunction()
{
    this->raw_s_change = [&](const double& target, const struct ann::node& actual)
    {
        // std::cout << "target: " << target << std::endl;
        // std::cout << "last small change: " <<  actual << std::endl;
        double a = raw_activation_function(actual);
        double small_change = (double)((target - a) * a * (1 - a));
        return small_change;
    };
}

double ann::Propagator::RawSmallChangeFunction(const double& target, const struct ann::node& actual) const
{
    return raw_s_change(target, actual);
}

void ann::Propagator::SetRawHiddenSmallChangeFunction(std::function<double(std::vector<double>, std::vector<double>, const struct ann::node&)> hidden_s_change)
{
    this->raw_hidden_s_change = hidden_s_change;
}

void ann::Propagator::ResetRawHiddenSmallChangeFunction()
{
    this->raw_hidden_s_change = [&](std::vector<double> prev_small_change, std::vector<double> weight, const struct ann::node& actual)
    {
        // std::cout << "actual: " <<  actual << std::endl;
        double change = 0.0;
        for(size_t i = 0; i < prev_small_change.size(); i++)
        {
            // std::cout << "i: " << i << " || prev_small_change: " << prev_small_change.at(i) << std::endl;
            change += (prev_small_change.at(i) * weight.at(i));
        }
        double a = raw_activation_function(actual);
        change *= (a * (1 - a));
        return change;
    };
}

double ann::Propagator::RawHiddenSmallChangeFunction(std::vector<double> prev_small_change, std::vector<double> weight, const ann::node& actual) const
{
    return this->raw_hidden_s_change(prev_small_change, weight, actual);
}


void ann::Propagator::SetNNetwork(ann::ANN& nnetwork)
{
    this->nnetwork = nnetwork;
}
