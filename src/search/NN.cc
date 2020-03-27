#include "NN.h"
#include "utils/eigen/Eigen/Dense"
#include "utils/math_utils.h"
#include "utils/string_utils.h"
#include "utils/system_utils.h"
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

using namespace Eigen;
using namespace std;

void NN::train(vector<vector<double>> data, vector<vector<double>> label,
               int epoch) {
    MatrixXd m = MatrixXd::Zero(data[0].size(), data.size());

    for (unsigned int i = 0; i < data.size(); ++i) {
        for (unsigned int j = 0; j < data[0].size(); ++j) {
            m(j, i) = data[i][j];
        }
    }

    for (unsigned int i = 0; i < epoch; ++i) {
        this->layer_input.clear();
        this->layer_output.clear();
        if (epoch % 800 == 0) {
            this->learningRate = this->learningRate * 0.99;
        }
        MatrixXd outp;
        outp = this->feedForward(m);
        double long currentLoss = this->loss(outp, label);
        cout << "loss:    " << currentLoss << endl;
        this->BackPropagation(outp, label);
    }
}

MatrixXd NN::feedForward(MatrixXd in) {
    in.conservativeResize(in.rows() + 1, in.cols());
    in.row(in.rows() - 1) = MatrixXd::Ones(1, in.cols());

    layer_output.push_back(in);
    layer_input.push_back(in);
    for (unsigned int i = 0; i < topology.size() - 1; ++i) {
        MatrixXd tmp = weights[i] * in;
        layer_input.push_back(tmp);

        if (i != topology.size() - 2) {
            in = activation(tmp);
            in.conservativeResize(in.rows() + 1, in.cols());
            in.row(in.rows() - 1) = MatrixXd::Ones(1, in.cols());
            layer_output.push_back(in);
        } else {
            in = softmax(tmp);
        }
    }
    return checkNan(in);
}

double NN::loss(MatrixXd output, vector<vector<double>> label) {
    double long res = 0;
    for (unsigned int i = 0; i < label.size(); ++i) {
        for (unsigned int j = 0; j < label[0].size(); ++j) {
            double long tmp = log((double long)output(j, i));
            if (isnan(tmp)) {
                tmp = 0;
            }
            res -= (double long)label[i][j] * tmp;
        }
    }
    if (isnan(res)) {
        res = 0;
    }
    return res;
}

void NN::BackPropagation(MatrixXd output, vector<vector<double>> label) {
    MatrixXd err = output;
    for (unsigned int j = 0; j < output.cols(); ++j) {
        for (unsigned int i = 0; i < output.rows(); ++i) {
            err(i, j) = err(i, j) - label[j][i];
        }
    }
    err = err / label.size();
    for (int i = topology.size() - 2; i > -1; --i) {
        MatrixXd old_weight = weights[i];
        weights[i] -= learningRate * (err * (layer_output[i].transpose())) +
                      regularization * old_weight;
        weights[i] = checkNan(weights[i]);
        err = old_weight.transpose() * err;
        err.conservativeResize(err.rows() - 1, err.cols());

        err = checkNan(err);
        for (unsigned int k = 0; k < err.cols(); ++k) {
            for (unsigned int j = 0; j < err.rows(); ++j) {
                if (layer_input[i](j, k) < 0) {
                    err(j, k) = 0;
                }
            }
        }
    }
}

// Only for binary state-fluent now
std::vector<double> NN::predict(std::vector<double> data_point) {
    MatrixXd m = MatrixXd::Zero(data_point.size(), 1);
    for (unsigned int i = 0; i < data_point.size(); ++i) {
        if(m(i,0)>0){  m(i, 0) = data_point[i];}
        else{ m(i, 0) = -1.0;}
      
    };
    data_point.clear();
    m.conservativeResize(m.rows() + 1, m.cols());
    m.row(m.rows() - 1) = MatrixXd::Ones(1, m.cols());
    for (unsigned int i = 0; i < topology.size() - 1; ++i) {
        MatrixXd tmp = weights[i] * m;
        if (i != topology.size() - 2) {
            m = activation(tmp);
            m.conservativeResize(m.rows() + 1, m.cols());
            m.row(m.rows() - 1) = MatrixXd::Ones(1, m.cols());
        } else {
            m = softmax(tmp);
        }
    }
    vector<double> outv;
    for (unsigned int i = 0; i < m.rows(); ++i) {
        outv.push_back(m(i, 0));
    };
    return outv;
}

MatrixXd NN::activation(MatrixXd in) {
    for (unsigned int j = 0; j < in.cols(); ++j) {
        for (unsigned int i = 0; i < in.rows(); ++i) {
            if (in(i, j) < 0) {
                in(i, j) = 0;
            }
        }
    }
    return in;
}

MatrixXd NN::softmax(MatrixXd in) {
    for (unsigned int j = 0; j < in.cols(); ++j) {
        double long col_sum = 0;
        for (unsigned int i = 0; i < in.rows(); ++i) {
            double long tmp = exp(in(i, j));
            col_sum += tmp;
            in(i, j) = tmp;
        }
        for (unsigned int i = 0; i < in.rows(); ++i) {
            in(i, j) = in(i, j) / col_sum;
        }
    }
    return in;
}

void NN::ReadParameters(string name) {
    ifstream inFile;
    string line;
    weights.clear();
    for (unsigned int i = 0; i < topology.size() - 1; ++i) {
        MatrixXd m = MatrixXd::Zero(topology[i + 1], topology[i] + 1);
        weights.push_back(m);
    };

    inFile.open(name);
    if (inFile.is_open()) {
        double num;
        int ind = 0;
        int double_count = 0;
        int i = 0;
        int j = 0;

        while (inFile >> num) {
            int w_size = (topology[ind] + 1) * topology[ind + 1];
            if (double_count == w_size - 1) {
                weights[ind](i, j) = num;
                double_count = 0;
                ind += 1;
                i = 0;
                j = 0;

            } else {
                weights[ind](i, j) = num;
                double_count += 1;
                j += 1;
                if (j == topology[ind] + 1) {
                    i += 1;
                    j = 0;
                }
            }
        }
    }
    inFile.close();
}

void NN::SaveParameters(string name) {
    std::ofstream ofs;
    ofs.open(name, std::ofstream::app);

    string write = "topology:";
    for (unsigned int index = 0; index < topology.size(); ++index) {
        write += to_string(topology[index]);
        write += " # ";
    }
    // ofs << write << "\n";

    for (unsigned int index = 0; index < weights.size(); ++index) {
        string write = "size:" + to_string(weights[index].rows()) + " : " +
                       to_string(weights[index].cols());

        // ofs << write << "\n";

        for (unsigned int i = 0; i < weights[index].rows(); ++i) {
            string write = "";
            for (unsigned int j = 0; j < weights[index].cols(); ++j) {
                write += to_string(weights[index](i, j));
                write += " | ";
                ofs << weights[index](i, j);
            }
            // ofs << write << "\n";
        }
    }

    ofs.close();
}

MatrixXd outer(MatrixXd m1, MatrixXd m2) {
    MatrixXd res = MatrixXd::Zero(m1.rows(), m2.cols());
    for (unsigned int i = 0; i < m1.rows(); ++i) {
        for (unsigned int j = 0; j < m2.cols(); ++j) {
            res(i, j) = m1(i, 0) * m2(0, j);
        }
    }
    return res;
}

MatrixXd NN::checkNan(MatrixXd m) {
    for (unsigned int i = 0; i < m.rows(); ++i) {
        for (unsigned int j = 0; j < m.cols(); ++j) {
            if (isnan((double)m(i, j))) {
                m(i, j) = 0;
            }
        }
    }
    return m;
}