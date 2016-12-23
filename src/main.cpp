#include <iostream>
#include <vector>

#include "../include/ann.h"

int main()
{
    ann::ANN nnetwork;
    nnetwork.AddNode(0);
    nnetwork.AddWeight(0, 0, 0.5);
    nnetwork.AddWeight(0, 0, 0.5);

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
    return 0;
}
