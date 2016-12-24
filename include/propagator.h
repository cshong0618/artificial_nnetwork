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

            ResetActivationFunction();
            ResetForwardPropagateFunction();
            ResetSmallChangeFunction();
            ResetBackwardPropagateFunction();
            ResetHiddenSmallChangeFunction();
        }

        void SetActivationFunction(std::function<double(const double&)> f);
        void ResetActivationFunction();
        double ActivationFunction(const double& net) const;

        void SetForwardPropagateFunction(std::function<double(int, int, std::vector<double>)> f);
        void ResetForwardPropagateFunction();
        double ForwardPropagate(int layer, int node, std::vector<double> input) const;

        ann::node_network AutoForwardPropagate(std::vector<double> input);

        void SetBackwardPropagateFunction(std::function<double(const double&, const double&)> backward_propagate);
        void ResetBackwardPropagateFunction();
        double BackwardPropagate(const double& s_change, const double& net) const;
        ann::network AutoBackwardPropagate(const ann::node_network& nets, std::vector<double> target);


        void SetSmallChangeFunction(std::function<double(const double&, const double&)> f);
        void ResetSmallChangeFunction();
        double SmallChangeFunction(const double& target, const double& actual) const;

        void SetHiddenSmallChangeFunction(std::function<double(std::vector<double>, std::vector<double>, const double&)> hidden_s_change);
        void ResetHiddenSmallChangeFunction();
        double HiddenSmallChangeFunction(std::vector<double> prev_small_change, std::vector<double> weights, const double& actual) const;

        void SetNNetwork(ann::ANN& nnetwork);
        inline ann::ANN& GetNNetwork()
        {
            return nnetwork;
        }

    private:
        ann::ANN nnetwork;
        std::function<double(const double&)> activation_function;
        std::function<double(int, int, std::vector<double>)> forward_propagate;
        std::function<double(const double&, const double&)> backward_propagate;
        std::function<double(const double& target, const double& actual)> s_change;
        std::function<double(std::vector<double>, std::vector<double>, const double&)> hidden_s_change;

    };
}

#endif
