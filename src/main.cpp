#include <iostream>
#include <vector>

#include "../include/ann.h"
#include "../include/propagator.h"

int main()
{
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
    nn_test.SetLearningRate(1);
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
    test_p.AutoBackwardPropagate(test_run, {1.0});


    return 0;
}
