#include <iostream>
#include <vector>

#include "../include/ann.h"
#include "../include/propagator.h"

int main()
{
    ann::ANN nnetwork;
    nnetwork.AddNode(0);
    nnetwork.AddWeight(0, 0, 0.5);
    nnetwork.AddWeight(0, 0, 0.5);
    nnetwork.AddNode(0);
    nnetwork.AddWeight(0, 1, 0.2);
    nnetwork.AddWeight(0, 1, 0.2);

    std::cout << nnetwork;

    nnetwork.SetLearningRate(0.5);
    std::cout << "Learning rate: " << nnetwork.GetLearningRate() << "\n";

    nnetwork.SetErrorMargin(0.0001);
    std::cout << "Error margin: " << nnetwork.GetErrorMargin() << "\n";

    std::vector<double> in = {20, 60};
    std::vector<double> out = {1};
    nnetwork.AddTrainingSet(in, out);

    std::vector<double> test = nnetwork.GetTrainingSet(in);
    std::cout << "Key: ";
    for(auto i : in)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    std::cout << "Val: ";
    for(auto i : test)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    try
    {
    std::cout << "Forward propagate 1: " << nnetwork.ForwardPropagate(0, 0, in) << "\n";
    std::cout << "Forward propagate 2: " << nnetwork.ForwardPropagate(0, 1, in) << "\n";
    }
    catch (std::exception e)
    {
        std::cout << e.what() << std::endl;
    }
    ann::Propagator propagator(nnetwork);
    propagator.GetNNetwork().SetWeight(0, 0, 0, 100);
    propagator.SetForwardPropagateFunction([&](int layer, int node, std::vector<double> inputs)
    {
        double net = 0.0;
        try
        {
            for(size_t i = 0; i < inputs.size(); i++)
            {
                net += (propagator.GetNNetwork().GetWeight(layer, node, i) * inputs.at(i));
            }
        }
        catch (std::exception e)
        {
            throw e;
        }

        return net;
    });

    std::cout << "Propagator forward propagate: " << propagator.ForwardPropagate(0, 0, in) << "\n";
    std::cout << "Propagator forward propagate: " << propagator.ForwardPropagate(0, 1, in) << "\n";

    ann::ANN nn_test;
    nn_test.AddNode(0);
    nn_test.AddWeight(0, 0, 0.5);
    nn_test.AddWeight(0, 0, 0.3);
    nn_test.AddNode(0);
    nn_test.AddWeight(0, 1, 0.2);
    nn_test.AddWeight(0, 1, 0.6);
    nn_test.AddLayer();
    nn_test.AddNode(1);
    nn_test.AddWeight(1, 0, 0.8);
    nn_test.AddWeight(1, 0, 0.1);

    std::vector<double> input = {0.5, 0.2};
    ann::Propagator test_p(nn_test);
    ann::node_network test_run = test_p.AutoForwardPropagate(input);

    for(int i = 0; i < (int)test_run.size(); i++)
    {
        std::cout << "Layer " << i << ":\n";
        for(int j = 0; j < (int)test_run.at(i).size(); j++)
        {
            std::cout << "--Node " << j << ": " << test_run.at(i).at(j) << std::endl;
        }

    }

    return 0;
}
