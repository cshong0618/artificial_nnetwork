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
        }

        void SetActivationFunction(std::function<double(const double&)> f);
        void ResetActivationFunction();
        double ActivationFunction(const double& net) const;

        void SetForwardPropagateFunction(std::function<double(int, int, std::vector<double>)> f);
        void ResetForwardPropagateFunction();
        double ForwardPropagate(int layer, int node, std::vector<double> input) const;

        ann::node_network AutoForwardPropagate(std::vector<double> input);

        void SetBackwardPropagateFunction(std::function<double(std::function<double(const double&, const double&)>, const double&, const double&, const double&)> backward_propagate);
        void ResetBackwardPropagateFunction();
        double BackwardPropagate(std::function<double(const double&, const double&)> s_change,
                                 const double& _param_1,
                                 const double& _param_2,
                                 const double& net) const;

        void SetSmallChangeFunction(std::function<double(const double&, const double&)> f);
        void ResetSmallChangeFunction();
        double SmallChangeFunction(const double& target, const double& actual) const;

        void SetNNetwork(ann::ANN& nnetwork);
        inline ann::ANN& GetNNetwork()
        {
            return nnetwork;
        }

    private:
        ann::ANN nnetwork;
        std::function<double(const double&)> activation_function;
        std::function<double(int, int, std::vector<double>)> forward_propagate;
        std::function<double(std::function<double(const double&, const double&)>, const double&, const double&, const double&)> backward_propagate;
        std::function<double(const double& target, const double& actual)> s_change;

    };
}

#endif
