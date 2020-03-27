#include "NN.cc"
#include "NN.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;


int main() {
    static int input_size = 11;
    static int out_size = 5;

    int i = 0;
    string line;
    vector<vector<double>> data;
    vector<vector<double>> label;

    ifstream myfile("data.txt");
    if (myfile.is_open()) {
        while (getline(myfile, line)) {
            if (i % 2 == 0) {
                vector<double> data_point(input_size, 0);
                for (unsigned int i = 0; i < line.length(); i++) {
                    char c = line[i];
                    int tmp = (int)c - 48;
                    if (tmp > 0) {
                        data_point[i] = 1.0;
                    } else {
                        data_point[i] = -1.0;
                    }
                }
                data.push_back(data_point);
                data_point.clear();

            } else {
                vector<double> label_point(out_size, 0);

                for (unsigned int i = 0; i < line.length(); i++) {
                    char c = line[i];
                    int tmp = (int)c - 48;
                    label_point[tmp] = 1.0;
                }
                label.push_back(label_point);
                label_point.clear();
            }
            i = i + 1;
        }

        myfile.close();
    }
    std::vector<int> topology;
    topology.push_back(input_size);
    topology.push_back(out_size + 3);
    topology.push_back(out_size);

    NN mynn(topology, 0.08, false, 0.003);
    mynn.train(data, label, 900);

    // std::vector<double> rres = mynn.predict(data[5]);
    // for (unsigned int i = 0; i < rres.size(); i++) {
    //     cout << rres[i] << "   ";
    // }
    // cout << "" << endl;


    mynn.SaveParameters("parameters.txt");
    //mynn.ReadParameters("parameters.txt");
    data.clear();
    label.clear();

    return 0;
}