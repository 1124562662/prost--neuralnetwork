#ifndef NN_H
#define NN_H

#include "utils/eigen/Eigen/Dense"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace Eigen;
using namespace std;

class NN {
public:
    NN() {
        this->ReadParameterFromFile = true;
        std::vector<int> topolog;
        topolog.push_back(11);
        topolog.push_back(8);
        topolog.push_back(6);
        topolog.push_back(5);
        this->topology = topolog;
        this->learningRate = 0.05;
        this->regularization = 0.005;

        if (ReadParameterFromFile) {
            ReadParameters("parameters.txt");
        } else {
            for (unsigned int i = 0; i < topology.size() - 1; ++i) {
                Eigen::MatrixXd m =
                    Eigen::MatrixXd::Random(topology[i + 1], topology[i] + 1);
                weights.push_back(m);
            };
        };
    }

        NN(std::vector<int> topology, double learningRate,
       bool ReadParameterFromFile, double regularization) {
        this->ReadParameterFromFile = ReadParameterFromFile;
        this->topology = topology;
        this->learningRate = learningRate;
        this->regularization = regularization;

        if (ReadParameterFromFile) {
            ReadParameters("parameters.txt");
        } else {
            for (unsigned int i = 0; i < topology.size() - 1; ++i) {
                Eigen::MatrixXd m =
                    Eigen::MatrixXd::Random(topology[i + 1], topology[i] + 1);
                weights.push_back(m);
            };
        };
    };
    ~NN(){};

    void train(vector<vector<double>> data, vector<vector<double>> label,
               int epoch);

    vector<double> predict(vector<double> data_point);

    void ReadParameters(string name);

    void SaveParameters(string name);

private:
    MatrixXd checkNan(MatrixXd m);
    MatrixXd activation(MatrixXd in);
    MatrixXd softmax(MatrixXd in);
    MatrixXd feedForward(MatrixXd tmp);
    double loss(MatrixXd output, vector<vector<double>> label);
    void BackPropagation(MatrixXd output, vector<vector<double>> label);

    std::vector<int> topology;
    double regularization;
    double learningRate = 0.05;

    bool ReadParameterFromFile;
    vector<MatrixXd> weights;
    vector<MatrixXd> layer_input;
    vector<MatrixXd> layer_output;
};

#endif
