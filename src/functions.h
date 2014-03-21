#ifndef EVALUATABLE_H
#define EVALUATABLE_H

#include "logical_expressions.h"

class Evaluatable {
public:
    Evaluatable(std::string _name, LogicalExpression* _formula) :
        name(_name),
        formula(_formula),
        isProb(false),
        hasArithmeticFunction(false),
        hashIndex(-1),
        cachingType(NONE),
        kleeneCachingType(NONE) {}

    Evaluatable(Evaluatable const& other, LogicalExpression* _formula) :
        name(other.name),
        formula(_formula),
        isProb(other.isProb),
        hasArithmeticFunction(other.hasArithmeticFunction),
        hashIndex(other.hashIndex),
        cachingType(other.cachingType),
        evaluationCacheVector(other.pdEvaluationCacheVector.size(), -std::numeric_limits<double>::max()),
        pdEvaluationCacheVector(other.pdEvaluationCacheVector.size(), DiscretePD()),
        kleeneCachingType(other.kleeneCachingType),
        kleeneEvaluationCacheVector(other.kleeneEvaluationCacheVector.size()),
        actionHashKeyMap(other.actionHashKeyMap) {}

    //TODO: This is very very ugly, but cpfs, planning tasks and states are very
    //tightly coupled. Nevertheless, there must be a way to get rid of this,
    //even if it takes some work!
    friend class PlanningTask;
    friend class Preprocessor;

    // This function is called in state transitions if and only if this
    // Evaluatable is deterministic
    void evaluate(double& res, State const& current, ActionState const& actions) {
        assert(!isProbabilistic());

        switch(cachingType) {
        case NONE:
            formula->evaluate(res, current, actions);
            break;
        case MAP:
            stateHashKey = current.stateFluentHashKey(hashIndex) + actionHashKeyMap[actions.index];
            assert((current.stateFluentHashKey(hashIndex) >= 0) && (actionHashKeyMap[actions.index] >= 0) && (stateHashKey >= 0));

            if(evaluationCacheMap.find(stateHashKey) != evaluationCacheMap.end()) {
                res = evaluationCacheMap[stateHashKey];
            } else {
                formula->evaluate(res, current, actions);
                evaluationCacheMap[stateHashKey] = res;
            }
            break;
        case DISABLED_MAP:
            stateHashKey = current.stateFluentHashKey(hashIndex) + actionHashKeyMap[actions.index];
            assert((current.stateFluentHashKey(hashIndex) >= 0) && (actionHashKeyMap[actions.index] >= 0) && (stateHashKey >= 0));

            if(evaluationCacheMap.find(stateHashKey) != evaluationCacheMap.end()) {
                res = evaluationCacheMap[stateHashKey];
            } else {
                formula->evaluate(res, current, actions);
            }

            break;
        case VECTOR:
            stateHashKey = current.stateFluentHashKey(hashIndex) + actionHashKeyMap[actions.index];
            assert((current.stateFluentHashKey(hashIndex) >= 0) && (actionHashKeyMap[actions.index] >= 0) && (stateHashKey >= 0));
            assert(stateHashKey < evaluationCacheVector.size());

            if(MathUtils::doubleIsMinusInfinity(evaluationCacheVector[stateHashKey])) {
                formula->evaluate(res, current, actions);
                evaluationCacheVector[stateHashKey] = res;
            } else {
                res = evaluationCacheVector[stateHashKey];
            }
            break;
        }
    }

    // This function is called in state transitions if this Evaluatable is
    // probabilistic or if it is deterministic but part of a probabilistic
    // planning state
    void evaluateToPD(DiscretePD& res, State const& current, ActionState const& actions) {
        assert(res.isUndefined());

        if(isProbabilistic()) {
            switch(cachingType) {
            case NONE:
                formula->evaluateToPD(res, current, actions);
                break;
            case MAP:
                stateHashKey = current.stateFluentHashKey(hashIndex) + actionHashKeyMap[actions.index];
                assert((current.stateFluentHashKey(hashIndex) >= 0) && (actionHashKeyMap[actions.index] >= 0) && (stateHashKey >= 0));

                if(pdEvaluationCacheMap.find(stateHashKey) != pdEvaluationCacheMap.end()) {
                    res = pdEvaluationCacheMap[stateHashKey];
                } else {
                    formula->evaluateToPD(res, current, actions);
                    pdEvaluationCacheMap[stateHashKey] = res;
                }
                break;
            case DISABLED_MAP:
                stateHashKey = current.stateFluentHashKey(hashIndex) + actionHashKeyMap[actions.index];
                assert((current.stateFluentHashKey(hashIndex) >= 0) && (actionHashKeyMap[actions.index] >= 0) && (stateHashKey >= 0));

                if(pdEvaluationCacheMap.find(stateHashKey) != pdEvaluationCacheMap.end()) {
                    res = pdEvaluationCacheMap[stateHashKey];
                } else {
                    formula->evaluateToPD(res, current, actions);
                }
                break;
            case VECTOR:
                stateHashKey = current.stateFluentHashKey(hashIndex) + actionHashKeyMap[actions.index];
                assert((current.stateFluentHashKey(hashIndex) >= 0) && (actionHashKeyMap[actions.index] >= 0) && (stateHashKey >= 0));
                assert(stateHashKey < pdEvaluationCacheVector.size());

                if(pdEvaluationCacheVector[stateHashKey].isUndefined()) {
                    formula->evaluateToPD(res, current, actions);
                    pdEvaluationCacheVector[stateHashKey] = res;
                } else {
                    res = pdEvaluationCacheVector[stateHashKey];
                }
                break;
            }
        } else {
            double tmp;
            evaluate(tmp, current, actions);
            res.assignDiracDelta(tmp);
        }
    }

    // This function is called for state transitions with KleeneStates
    // (regardless of deterministic / probabilistic)
    void evaluateToKleene(std::set<double>& res, KleeneState const& current, ActionState const& actions) {
        assert(res.empty());
        switch(kleeneCachingType) {
        case NONE:
            formula->evaluateToKleene(res, current, actions);
            break;
        case MAP:
            stateHashKey = current.stateFluentHashKey(hashIndex) + actionHashKeyMap[actions.index];
            assert((current.stateFluentHashKey(hashIndex) >= 0) && (actionHashKeyMap[actions.index] >= 0) && (stateHashKey >= 0));

            if(kleeneEvaluationCacheMap.find(stateHashKey) != kleeneEvaluationCacheMap.end()) {
                res = kleeneEvaluationCacheMap[stateHashKey];
            } else {
                formula->evaluateToKleene(res, current, actions);
                kleeneEvaluationCacheMap[stateHashKey] = res;
            }
            break;
        case DISABLED_MAP:
            stateHashKey = current.stateFluentHashKey(hashIndex) + actionHashKeyMap[actions.index];
            assert((current.stateFluentHashKey(hashIndex) >= 0) && (actionHashKeyMap[actions.index] >= 0) && (stateHashKey >= 0));

            if(kleeneEvaluationCacheMap.find(stateHashKey) != kleeneEvaluationCacheMap.end()) {
                res = kleeneEvaluationCacheMap[stateHashKey];
            } else {
                formula->evaluateToKleene(res, current, actions);
            }

            break;
        case VECTOR:
            stateHashKey = current.stateFluentHashKey(hashIndex) + actionHashKeyMap[actions.index];
            assert((current.stateFluentHashKey(hashIndex) >= 0) && (actionHashKeyMap[actions.index] >= 0) && (stateHashKey >= 0));
            assert(stateHashKey < kleeneEvaluationCacheVector.size());

            if(kleeneEvaluationCacheVector[stateHashKey].empty()) {
                formula->evaluateToKleene(res, current, actions);
                kleeneEvaluationCacheVector[stateHashKey] = res;
            } else {
                res = kleeneEvaluationCacheVector[stateHashKey];
            }
            break;
        }
    }

    // Initialization
    void initialize();
    void initializeHashKeys(int _hashIndex, std::vector<ActionState> const& actionStates,
                            std::vector<ConditionalProbabilityFunction*> const& CPFs,
                            std::vector<std::vector<std::pair<int,long> > >& indexToStateFluentHashKeyMap,
                            std::vector<std::vector<std::pair<int,long> > >& indexToKleeneStateFluentHashKeyMap);

    // Disable caching
    void disableCaching();

    // Properties
    bool const& isProbabilistic() const {
        return isProb;
    }

    // SAC only
    bool containsArithmeticFunction() const {
        return hasArithmeticFunction;
    }

    // SAC only
    bool containsStateFluent() const {
        return !dependentStateFluents.empty();
    }

    // SAC only
    bool containsActionFluent() const {
        return !(positiveActionDependencies.empty() && negativeActionDependencies.empty());
    }

    // SAC & Reward
    std::set<ActionFluent*> const& getPositiveDependentActionFluents() {
        return positiveActionDependencies;
    }

    // SAC only
    std::set<ActionFluent*>const& getNegativeDependentActionFluents() {
        return negativeActionDependencies;
    }

    // Reward only (apart from print in PlanningTask)
    bool isActionIndependent() const {
        return (positiveActionDependencies.empty() && negativeActionDependencies.empty());
    }

protected:
    // A unique string that describes this (only used for print)
    std::string name;

    // The formula that is evaluatable
    LogicalExpression* formula;

    // Properties of this Evaluatable
    std::set<StateFluent*> dependentStateFluents;
    std::set<ActionFluent*> positiveActionDependencies;
    std::set<ActionFluent*> negativeActionDependencies;
    bool isProb;
    bool hasArithmeticFunction;

    // hashIndex is the index in the stateFluentHashKey vector of a state where
    // the state fluent hash key of this Evaluatable is stored
    int hashIndex;

    enum CachingType {
        NONE, // too many variables influence formula
        MAP, // many variables influence formula, but it's still possible with a map
        DISABLED_MAP, // as MAP, but after disableCaching() has been called
        VECTOR // only few variables influence formula, so we use a vector for caching
    };

    // CachingType describes which of the two (if any) datastructures is used to
    // cache computed values. If this Evaluatable is probabilistic, the
    // datastructures that start with 'pd' are used and the other ones if the
    // Evaluatable is deterministic
    CachingType cachingType;
    std::map<long, double> evaluationCacheMap;
    std::map<long, DiscretePD> pdEvaluationCacheMap;
    std::vector<double> evaluationCacheVector;
    std::vector<DiscretePD> pdEvaluationCacheVector;

    // KleeneCachingType describes which of the two (if any) datastructures is
    // used to cache computed values on Kleene states
    CachingType kleeneCachingType;
    std::map<long, std::set<double> > kleeneEvaluationCacheMap;
    std::vector<std::set<double> > kleeneEvaluationCacheVector;

    // ActionHashKeyMap contains the hash keys of the actions that influence
    // this Evaluatable (these are added to the state fluent hash keys of a
    // state)
    std::vector<long> actionHashKeyMap;

private:
    // These function are used to calculate the two parts of state fluent hash
    // keys: the action part (that is stored in the actionHashKeyMap of
    // Evaluatable), and the state fluent part (that is stored in PlanningTask
    // and computed within states).
    long initializeActionHashKeys(std::vector<ActionState> const& actionStates);
    void calculateActionHashKey(std::vector<ActionState> const& actionStates, ActionState const& action, long& nextKey);
    long getActionHashKey(std::vector<ActionState> const& actionStates, std::vector<ActionFluent*>& scheduledActions);

    void initializeStateFluentHashKeys(std::vector<ConditionalProbabilityFunction*> const& CPFs,
                                       std::vector<std::vector<std::pair<int,long> > >& indexToStateFluentHashKeyMap,
                                       long const& firstStateFluentHashKeyBase);
    void initializeKleeneStateFluentHashKeys(std::vector<ConditionalProbabilityFunction*> const& CPFs,
                                             std::vector<std::vector<std::pair<int,long> > >& indexToKleeneStateFluentHashKeyMap,
                                             long const& firstStateFluentHashKeyBase);
    bool dependsOnActionFluent(ActionFluent* fluent) {
        return (positiveActionDependencies.find(fluent) != positiveActionDependencies.end() ||
                negativeActionDependencies.find(fluent) != negativeActionDependencies.end());
    }

    // This is a temporary variable used in evaluate. It is a member var as
    // initializing variables in case statements is only possible if additional
    // parentheses are used
    long stateHashKey;
};

class RewardFunction : public Evaluatable {
public:
    RewardFunction(LogicalExpression* _formula) :
        Evaluatable("Reward" , _formula) {}

    void setDomain(std::set<double> _domain) {
        domain = _domain;
    }

    double const& getMinVal() const {
        assert(!domain.empty());
        return *domain.begin();
    }

    double const& getMaxVal() const {
        assert(!domain.empty());
        return *domain.rbegin();
    }

private:
    std::set<double> domain;
};

class ConditionalProbabilityFunction : public Evaluatable {
public:
    //TODO: This is very very ugly, but cpfs, planning tasks and
    //states are very tightly coupled. Nevertheless, there must be a
    //way to get rid of this, even if it takes some work!
    friend class PlanningTask;

    // This is used to sort transition functions by their name to ensure
    // deterministic behaviour
    struct TransitionFunctionSort {
        bool operator() (ConditionalProbabilityFunction* const& lhs, ConditionalProbabilityFunction* const& rhs) const {
            return lhs->name < rhs->name;
        }
    };

    ConditionalProbabilityFunction(StateFluent* _head, LogicalExpression* _formula) :
        Evaluatable(_head->fullName, _formula),
        head(_head),
        kleeneDomainSize(0) {}

    ConditionalProbabilityFunction(ConditionalProbabilityFunction const& other, LogicalExpression* _formula) :
        Evaluatable(other, _formula),
        head(other.head),
        domain(other.domain),
        kleeneDomainSize(other.kleeneDomainSize) {}

    ConditionalProbabilityFunction* determinizeMostLikely(NumericConstant* randomNumberReplacement);

    StateFluent* getHead() const {
        assert(head);
        return head;
    }

    void setDomain(std::set<double> _domain) {
        domain = _domain;
    }

    int getDomainSize() const {
        return domain.size();
    }

    bool hasFiniteDomain() const {
        return (!domain.empty());
    }

    void setKleeneDomainSize(long _kleeneDomainSize) {
        kleeneDomainSize = _kleeneDomainSize;
    }

    long getKleeneDomainSize() const {
        return kleeneDomainSize;
    }

    bool hasFiniteKleeneDomain() const {
        return (kleeneDomainSize > 0);
    }

    void setIndex(int _index) {
        head->index = _index;
    }

    double getInitialValue() const {
        return head->initialValue;
    }

private:
    StateFluent* head;

    // The values this CPF can take
    std::set<double> domain;

    // Hashing of KleeneStates
    long kleeneDomainSize;
};


#endif