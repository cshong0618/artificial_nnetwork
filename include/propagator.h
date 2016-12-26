#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include "ann.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <exception>
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

            ResetRawActivationFunction();
            ResetRawForwardPropagateFunction();
            ResetRawSmallChangeFunction();
            ResetRawBackwardPropagateFunction();
            ResetRawHiddenSmallChangeFunction();
        }

        void SetActivationFunction(std::function<double(const double&)> f);
        void SetRawActivationFunction(std::function<double(const struct ann::node&)> f);
        void ResetActivationFunction();
        void ResetRawActivationFunction();
        double ActivationFunction(const double& net) const;
        double RawActivationFunction(const struct ann::node& net) const;

        void SetForwardPropagateFunction(std::function<double(int, int, std::vector<double>)> f);
        void ResetForwardPropagateFunction();
        double ForwardPropagate(int layer, int node, std::vector<double> input) const;

        void SetRawForwardPropagateFunction(std::function<struct ann::node(int, int, std::vector<double>)> f);
        void ResetRawForwardPropagateFunction();
        struct node RawForwardPropagate(int layer, int node, std::vector<double> input) const;

        ann::node_network AutoForwardPropagate(std::vector<double> input);
        ann::raw_node_network RawAutoForwardPropagate(std::vector<double> input);

        void SetBackwardPropagateFunction(std::function<double(const double&, const double&)> backward_propagate);
        void ResetBackwardPropagateFunction();
        double BackwardPropagate(const double& s_change, const double& net) const;
        ann::network AutoBackwardPropagate(const ann::node_network& nets, std::vector<double> target);

        void SetRawBackwardPropagateFunction(std::function<double(const double&, const struct ann::node&)> f);
        void ResetRawBackwardPropagateFunction();
        double RawBackwardPropagate(const double& s_change, const struct ann::node& net) const;
        ann::network RawAutoBackwardPropagate(const ann::raw_node_network& nets, std::vector<double> target);


        void SetSmallChangeFunction(std::function<double(const double&, const double&)> f);
        void ResetSmallChangeFunction();
        double SmallChangeFunction(const double& target, const double& actual) const;

        void SetRawSmallChangeFunction(std::function<double(const double&, const struct ann::node&)> f);
        void ResetRawSmallChangeFunction();
        double RawSmallChangeFunction(const double& target, const struct ann::node& actual) const;

        void SetHiddenSmallChangeFunction(std::function<double(std::vector<double>, std::vector<double>, const double&)> hidden_s_change);
        void ResetHiddenSmallChangeFunction();
        double HiddenSmallChangeFunction(std::vector<double> prev_small_change, std::vector<double> weights, const double& actual) const;

        void SetRawHiddenSmallChangeFunction(std::function<double(std::vector<double>, std::vector<double>, const struct ann::node&)> hidden_s_change);
        void ResetRawHiddenSmallChangeFunction();
        double RawHiddenSmallChangeFunction(std::vector<double> prev_small_change, std::vector<double> weights, const ann::node& actual) const;

        void SetNNetwork(ann::ANN& nnetwork);
        inline ann::ANN& GetNNetwork()
        {
            return nnetwork;
        }

    private:
        ann::ANN nnetwork;
        std::function<double(const double&)> activation_function;
        std::function<double(const struct ann::node&)> raw_activation_function;

        std::function<double(int, int, std::vector<double>)> forward_propagate;
        std::function<struct node(int, int, std::vector<double>)> r_forward_propagate;

        std::function<double(const double&, const double&)> backward_propagate;
        std::function<double(const double& target, const double& actual)> s_change;
        std::function<double(std::vector<double>, std::vector<double>, const double&)> hidden_s_change;

        std::function<double(const double&, const struct ann::node&)> raw_backward_propagate;
        std::function<double(const double& target, const struct ann::node& actual)> raw_s_change;
        std::function<double(std::vector<double>, std::vector<double>, const struct ann::node&)> raw_hidden_s_change;
    };
}

#endif
