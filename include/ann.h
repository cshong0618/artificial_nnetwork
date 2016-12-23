#ifndef ANN_H
#define ANN_H

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ann
{
    class ANN
    {
    private:
        typedef std::vector<double> node;
        typedef std::vector<node> layer;
        typedef std::vector<layer> network;

    public:
// Constructors
        ANN(int layers)
        {
            if(layers >= 1)
            {
                for(int i = 0; i < layers; i++)
                {
                    AddLayer();
                }
            }
            else
            {
                // At least one input and one output

                // Perceptron
                for(int i = 0; i < 1; i++)
                {
                    AddLayer();
                }
            }

            sigmoid_function = [](const double& net)
            {
                return (1 / 1 + (exp(-net)));
            };
        }

        ANN() : ANN(1){}
        ANN(int layers, const double& learning_rate) : ANN(layers)
        {
            this->learning_rate = learning_rate;
        }

        ANN(int layers, std::string weight_filename) : ANN(layers)
        {
            this->weight_filename = weight_filename;
        }

        ANN(std::string weight_filename) : ANN(1, weight_filename){}

// Methods
        void AddLayer();
        bool AddNode(int layer);
        bool AddWeight(int layer, int node, const double& weight);
        bool SetWeight(int layer, int node, int n, const double& weight);

        inline std::vector<node> GetLayer(int layer) const
        {
            if (layer >= 0 && layer < (int)nnetwork.size())
            {
                return this->nnetwork.at(layer);
            }
            else
            {
                throw std::range_error("Out of range");
            }
        }

        inline std::vector<double> GetNode(int layer, int node)
        {
            if(layer >= 0 && layer < (int)nnetwork.size())
            {
                if(node >= 0 && node < (int)nnetwork.at(layer).size())
                {
                    return nnetwork.at(layer).at(node);
                }
            }
            else
            {
                throw std::range_error("Out of range");
            }
        }

        void AddTrainingSet(const std::vector<double>& input, const std::vector<double>& output);
        inline std::vector<double> GetTrainingSet(const std::vector<double>&key) const
        {
            if(training_set.find(key) != training_set.end())
            {
                return training_set.at(key);
            }
            else
            {
                throw std::invalid_argument("Key not found");
            }
        }

        void SetActivationFunction(std::function<double(const double& net)> f);
        double ActivationValue(const double& net) const;


        void SetErrorMargin(const double& error_margin);
        inline double GetErrorMargin() const {return this->error_margin;}

        inline bool ErrorInMargin(const std::vector<double>& input,
                                  const std::vector<double>& output) const
        {
            std::vector<double> actual = GetTrainingSet(input);
            for(int i = 0; i < (int)actual.size(); i++)
            {
                if(fabs(actual.at(i) - output.at(i)) > error_margin)
                {
                    return false;
                }
            }

            return true;
        }

        void SetLearningRate(const double& learning_rate);
        inline double GetLearningRate() const {return this->learning_rate;}

        double ForwardPropagate(int layer, int node, std::vector<double> inputs);
        double FinalBackwardPropagate(double net, double actual, double target);
        void BackwardPropagate();

// Operators
        friend std::ostream& operator<<(std::ostream &os, const ANN& obj)
        {
            os << std::fixed << std::setprecision(4) << "Network weights\n";

            for(size_t i = 0; i < obj.nnetwork.size(); i++)
            {
                os << "Layer " << i << ": \n";
                for(size_t j = 0; j < obj.nnetwork.at(i).size(); j++)
                {
                    os << "--Node " << j << ": ";
                    for(auto node : obj.nnetwork.at(i).at(j))
                        os << node << " ";
                    os << "\n"; 
                }
                os << "\n";
            }

            return os;
        }


    private:
        // Network data
        network nnetwork;
        std::map<std::vector<double>, std::vector<double>> training_set;

        // Calculation data
        double learning_rate;
        double error_margin;

        // Calculation functions
        std::function<double(const double& net)> sigmoid_function;
        std::function<double(const double& weight)> output_update_function;
        std::function<double(const double& weight)> hidden_update_function;

        // Others
        std::string weight_filename;
    };
}

#endif
