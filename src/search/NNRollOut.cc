
#include "NNRollOut.h"
#include "NN.h"
#include "prost_planner.h"
#include "utils/math_utils.h"
#include "utils/system_utils.h"
#include <algorithm>
#include <iostream>
#include <numeric>
using namespace std;

/******************************************************************
                     Search Engine Creation
******************************************************************/

NNRollOut::NNRollOut()
    : ProbabilisticSearchEngine("NNRollOut"), numberOfIterations(1) {
    std::vector<int> topology;
    topology.push_back(11);
    topology.push_back(8);
    topology.push_back(6);
    topology.push_back(5);
    NN knn(topology, 0.08, true, 0.05);
    this->rollout_NN = knn;
}

bool NNRollOut::setValueFromString(std::string& param, std::string& value) {
    if (param == "-it") {
        setNumberOfIterations(atoi(value.c_str()));
        return true;
    }

    return SearchEngine::setValueFromString(param, value);
}

/******************************************************************
                       Main Search Functions
******************************************************************/

void NNRollOut::estimateQValue(State const& state, int actionIndex,
                               double& qValue) {
    assert(state.stepsToGo() > 0);
    PDState current(state);
    performNNWalks(current, actionIndex, qValue);
}

void NNRollOut::estimateQValues(State const& state,
                                std::vector<int> const& actionsToExpand,
                                std::vector<double>& qValues) {
    assert(state.stepsToGo() > 0);
    PDState current(state);
    for (size_t index = 0; index < qValues.size(); ++index) {
        if (actionsToExpand[index] == index) {
            performNNWalks(current, index, qValues[index]);
        }
    }
}

void NNRollOut::performNNWalks(PDState const& root, int firstActionIndex,
                               double& result) {
    result = 0.0;
    double reward = 0.0;
    PDState next;

    for (unsigned int i = 0; i < numberOfIterations; ++i) {
        PDState current(root.stepsToGo() - 1);
        sampleSuccessorState(root, firstActionIndex, current, reward);
        result += reward;

        while (current.stepsToGo() > 0) {
            std::vector<int> applicable =
                getIndicesOfApplicableActions(current);
            std::vector<double> predicted =
                this->rollout_NN.predict(current.ReadState2Vector());
            int ActionIndex = 0;
            double maxi = 0;
            for (unsigned int j = 0; j < predicted.size(); ++j) {
                if (predicted[j] >= maxi) {
                    for (unsigned int jj = 0; jj < applicable.size(); ++jj) {
                        if (j == applicable[jj]) {
                            ActionIndex = j;
                        }
                    }
                }
            }
            next.reset(current.stepsToGo() - 1);
            sampleSuccessorState(current, ActionIndex, next, reward);
            result += reward;
            current = next;
        }
    }

    result /= (double)numberOfIterations;

    // cout<<result<<endl;
}

void NNRollOut::sampleSuccessorState(PDState const& current,
                                     int const& actionIndex, PDState& next,
                                     double& reward) const {
    calcReward(current, actionIndex, reward);
    calcSuccessorState(current, actionIndex, next);
    for (unsigned int varIndex = 0;
         varIndex < State::numberOfProbabilisticStateFluents; ++varIndex) {
        next.sample(varIndex);
    }
    State::calcStateFluentHashKeys(next);
    State::calcStateHashKey(next);
}
