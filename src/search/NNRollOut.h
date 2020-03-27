#ifndef NNRollOut_H
#define NNRollOut_H

#include "NN.h"
#include "search_engine.h"
#include "states.h"
#include "utils/stopwatch.h"

// Evaluates all actions by simulating a run that starts with that action
// followed by a  NN until a terminal state is reached

class NNRollOut : public ProbabilisticSearchEngine {
public:
    NNRollOut();

    // Set parameters from command line
    bool setValueFromString(std::string& param, std::string& value) override;

    // Start the search engine to estimate the Q-value of a single action
    void estimateQValue(State const& state, int actionIndex,
                        double& qValue) override;

    // Start the search engine to estimate the Q-values of all applicable
    // actions
    void estimateQValues(State const& state,
                         std::vector<int> const& actionsToExpand,
                         std::vector<double>& qValues) override;

    // Parameter Setter
    virtual void setNumberOfIterations(int _numberOfIterations) {
        numberOfIterations = _numberOfIterations;
    }

private:
    void performNNWalks(PDState const& root, int firstActionIndex,
                        double& result);
    void sampleSuccessorState(PDState const& current, int const& actionIndex,
                              PDState& next, double& reward) const;

    // Parameter
    int numberOfIterations;
    NN rollout_NN;
};

#endif
