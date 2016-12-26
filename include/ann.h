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
    struct node
    {
        double val;
        char type;
    };

    typedef std::vector<double> t_weight;
    typedef std::vector<t_weight> t_layer;
    typedef std::vector<t_layer> network;

    typedef std::vector<double> node_layer;
    typedef std::vector<node_layer> node_network;

    typedef std::vector<struct node> raw_node_layer;
    typedef std::vector<raw_node_layer> raw_node_network;

    class ANN
    {
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
                return (1 / (1 + (exp(-net))));
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
        void SetRawNNetwork(const ann::network& nnetwork);

        inline t_layer& GetLayer(int layer)
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

        inline std::vector<double>& GetNode(int layer, int node)
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

        inline double GetWeight(int layer, int node, int n) const
        {
            if(layer >= 0 && layer < (int)nnetwork.size())
            {
                if(node >= 0 && node < (int)nnetwork.at(layer).size())
                {
                    if(n >= 0 && n < (int)nnetwork.at(layer).at(node).size())
                    {
                        return nnetwork.at(layer).at(node).at(n);
                    }
                }
            }
            else
            {
                throw std::range_error("Out of range");
            }
        }

        inline size_t GetLayerCount() const
        {
            return nnetwork.size();
        }


        inline ann::network GetRawNNetwork() const
        {
            return nnetwork;
        }

        void SetRawNode(const std::vector<struct node>& input);
        inline std::vector<struct node> GetRawNode() const
        {
            return this->raw_node;
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

        void AddTrainingSet(const std::vector<node>& input, const std::vector<double>& output);
        inline std::vector<double> GetTrainingSet(const std::vector<node>&key) const
        {
            std::vector<double> temp;
            for(auto n : key)
            {
                temp.push_back(n.val);
            }

            return this->GetTrainingSet(temp);
        }

        void SetActivationFunction(std::function<double(const double& net)> f);
        double ActivationValue(const double& net) const;


        void SetErrorMargin(const double& error_margin);
        inline double GetErrorMargin() const {return this->error_margin;}

        inline bool ErrorInMargin(const std::vector<double>& input,
                                  const std::vector<double> output) const
        {
            std::vector<double> actual = this->GetTrainingSet(input);
            std::cout << "--------------------------------------" << std::endl;
            std::cout << "\tEntered error margin check"<<std::endl;

            double output_total = 0;
            double actual_total = 0;
            double total_error_margin = this->error_margin * actual.size();
            total_error_margin = this->error_margin;
            // for(int i = 0; i < (int)actual.size(); i++)
            // {
            //     actual_total += actual.at(i);
            //     output_total += output.at(i);
            // }
            //
            for(size_t i = 0; i < actual.size(); i++)
            {
                std::cout << "actual[" << i << "]: " << actual[i]
                          << " output[" << i << "]: " << output[i] << std::endl;
            }
            // double diff = fabs(actual_total - output_total);
            // double error_percentage = diff / actual_total;
            double error_percentage = 1.0;
            for(size_t i = 0; i < actual.size(); i++)
            {
                error_percentage *= (1 / fabs(actual.at(i) - output.at(i)) / actual.at(i));
            }


            std::cout << "error_percentage: " << error_percentage << " total_error_margin: " << total_error_margin << std::endl;
            // std::cin.ignore();
            std::cout << "--------------------------------------" << std::endl;

            if(error_percentage > total_error_margin)
            {
                return true;
            }
            else
            {
                return false;
            }
        }

        void SetLearningRate(const double& learning_rate);
        inline double GetLearningRate() const {return this->learning_rate;}

        double ForwardPropagate(int layer, int node, std::vector<double> inputs);
        double FinalBackwardPropagate(double net, double actual, double target);
        void BackwardPropagate();
        double BackwardPropagate(int layer, int node, double p_net, double actual);
// Operators
        friend std::ostream& operator<<(std::ostream &os, const ANN& obj)
        {
            os << std::fixed << std::setprecision(8) << "Network weights\n";

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
        std::map<std::vector<node>, std::vector<double>> _training_set;
        std::vector<struct node> raw_node;
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
