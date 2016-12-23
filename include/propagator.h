#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include "ann.h"

#include <cmath>
#include <functional>
#include <vector>


namespace ann
{
    class Propagator
    {
    public:
        Propagator(ann::ANN nnetwork)
        {
            this->nnetwork = nnetwork;
            this->activation_function = [](const double& net)
            {
                return (1 / (1 + exp(-net)));
            };

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

        void SetActivationFunction(std::function<double(const double&)> f);
        double ActivationFunction(const double& net) const;

        void SetForwardPropogateFunction(std::function<double(int, int, std::vector<double>)> f);
        double ForwardPropagate(int layer, int node, std::vector<double> input) const;

        void SetNNetwork(ann::ANN& nnetwork);
        inline ann::ANN GetNNetwork() const
        {
            return nnetwork;
        }

    private:
        ann::ANN nnetwork;
        std::function<double(const double&)> activation_function;
        std::function<double(int, int, std::vector<double>)> forward_propagate;
    };
}

#endif
