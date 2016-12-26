#include <iostream>
#include <vector>

#include "../include/ann.h"
#include "../include/propagator.h"

int main()
{
    ann::ANN nn_test;
    nn_test.AddNode(0);
    nn_test.AddWeight(0, 0, 0.15);
    nn_test.AddWeight(0, 0, 0.25);
    nn_test.AddWeight(0, 0, 0.35);
    nn_test.AddNode(0);
    nn_test.AddWeight(0, 1, 0.10);
    nn_test.AddWeight(0, 1, 0.30);
    nn_test.AddWeight(0, 1, 0.35);
    nn_test.AddNode(0);
    nn_test.AddWeight(0, 2, 0);
    nn_test.AddWeight(0, 2, 0);
    nn_test.AddWeight(0, 2, 0);


    nn_test.AddLayer();
    nn_test.AddNode(1);
    nn_test.AddWeight(1, 0, 0.40);
    nn_test.AddWeight(1, 0, 0.50);
    nn_test.AddWeight(1, 0, 0.60);
    nn_test.AddNode(1);
    nn_test.AddWeight(1, 1, 0.45);
    nn_test.AddWeight(1, 1, 0.55);
    nn_test.AddWeight(1, 1, 0.60);
    nn_test.AddNode(1);
    nn_test.AddWeight(1, 2, 0);
    nn_test.AddWeight(1, 2, 0);
    nn_test.AddWeight(1, 2, 0);

    nn_test.AddLayer();
    nn_test.AddNode(2);
    nn_test.AddWeight(2, 0, 0.40);
    nn_test.AddWeight(2, 0, 0.50);
    nn_test.AddWeight(2, 0, 0.60);
    nn_test.AddNode(2);
    nn_test.AddWeight(2, 1, 0.45);
    nn_test.AddWeight(2, 1, 0.55);
    nn_test.AddWeight(2, 1, 0.60);

    nn_test.AddLayer();
    nn_test.AddNode(3);
    nn_test.AddWeight(3, 0, 0.20);
    nn_test.AddWeight(3, 0, 0.20);
    nn_test.AddNode(3);
    nn_test.AddWeight(3, 1, 0.25);
    nn_test.AddWeight(3, 1, 0.25);

    nn_test.SetLearningRate(0.1);
    nn_test.SetErrorMargin(0.000001);

    std::vector<ann::node> nodes = {{0.5, 'n'}, {0.1,'n'}, {1, 'b'}};
    std::vector<double> input = {0.05, 0.1, 1};
    std::vector<double> target = {0.01, 0.99};

    // nn_test.AddNode(0);
    // nn_test.AddWeight(0,0, 1);
    // nn_test.AddWeight(0,0, 0);
    // nn_test.AddNode(0);
    // nn_test.AddWeight(0, 1, 1);
    // nn_test.AddWeight(0, 1, 0);
    // nn_test.AddLayer();
    // nn_test.AddNode(1);
    // nn_test.AddWeight(1, 0, 0.8);
    // nn_test.AddWeight(1, 0, 0.2);
    // nn_test.SetLearningRate(0.1);
    // nn_test.SetErrorMargin(0.01);
    // std::vector<ann::node> nodes = {{1, 'n'}, {0, 'n'}};
    // std::vector<double> input = {1, 0};
    // std::vector<double> target = {1};

    nn_test.AddTrainingSet(input, target);
    nn_test.SetRawNode(nodes);
    ann::Propagator test_p(nn_test);
    //ann::node_network test_run = test_p.AutoForwardPropagate(input);
    ann::node_network test_run;

    std::cout << std::endl;

    std::cout << nn_test << std::endl;
    std::vector<double> v;

    unsigned int counter = 0;
    do
    {
        std::cout << "Run: " << counter << std::endl;
        v.clear();
        ann::raw_node_network test_run = test_p.RawAutoForwardPropagate(input);

        // for(size_t i = 0; i < test_run.size(); i++)
        // {
        //     std::cout << "Layer " << i << ":\n";
        //     for(size_t j = 0; j < test_run.at(i).size(); j++)
        //     {
        //         std::cout << "\tNode " << j << ": " << test_run.at(i).at(j).val << std::endl;
        //     }
        // }

        test_p.GetNNetwork().SetRawNNetwork(test_p.RawAutoBackwardPropagate(test_run, target));
        // std::cout << test_p.GetNNetwork() << std::endl;
        // v = test_run.back();
        for(auto n : test_run.back())
        {
            // std::cout << n.val << std::endl;
            v.push_back(n.val);
        }

        counter++;
    }while(nn_test.ErrorInMargin(input, v) ? [&](int a){counter++;return true;}(counter) : [&](int a){return false;}(counter));

    std::cout << "====END====" << std::endl;
    std::cout << counter << " epoch" << (counter > 1 ? "s" : "") << std::endl;
    nn_test.SetRawNNetwork(test_p.GetNNetwork().GetRawNNetwork());
    std::cout << nn_test;
    return 0;
}
