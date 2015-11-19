#include <cassert>
#include <iostream>
#include <cmath>
#include <cfloat>

/**
 * Viterbi algorithm to get state sequence given a set of observations
 *
 * Output state transitions can be compared with observed state transition to see of there is any state
 * tranition.
 *
 * email: flair.rajesh@gmail.com
 * 
 */

using namespace std;

class HMM {
public:
    int _states, _events;
    double **_stateTransitions, **_emissionMatrix;
    double *_state_prob;

    HMM(int states, int events, double **state_transitions, double **emissionMat, double *state_prob_) {

        // Asserts for input params
        assert(states > 0);
        assert(events > 0);
        assert(state_transitions != NULL);
        assert(emissionMat != NULL);
        assert(state_prob_ != NULL);

        // Set number of State Transitions and Emission Matrix
        _states = states;
        _events = events;

        // Initialize State Transitions
        _stateTransitions = new double*[states];
        for (int i = 0; i < states; i++) {
            _stateTransitions[i] = new double[states];
            for (int j = 0; j < states; j++) _stateTransitions[i][j] = log(state_transitions[i][j]);
        }

        // Initialize Emission Matrix
        _emissionMatrix = new double*[states];
        for (int i = 0; i < states; i++) {
            _emissionMatrix[i] = new double[events];
            for (int j = 0; j < events; j++) _emissionMatrix[i][j] = log(emissionMat[i][j]);
        }

        // Initialize Starting State Probabilities
        _state_prob = new double[states];
        for (int i = 0; i < states; i++) _state_prob[i] = log(state_prob_[i]);
    }
    ~HMM() {
        delete[] _state_prob;
        for (int i = 0; i < _states; i++) {
            delete[] _stateTransitions[i];
            delete[] _emissionMatrix[i];
        }
        delete[] _stateTransitions;
        delete[] _emissionMatrix;
    }

};

int viterbi(HMM const& hmm, const int observed[], const int n) {
    // Assert for Input Observations
    assert(n > 0);
    assert(observed != NULL);

    // Output sequence initialization
    int *seq = new int[n];
    for (int i = 0; i < n; i++) seq[i] = 0;

    //Prob and Previous state prob initialization
    double **prob = new double*[n];
    int **prevs = new int*[n];
    for (int i = 0; i < n; i++) {
        prob[i] = new double[hmm._states];
        prevs[i] = new int[hmm._states];
        for (int j = 0; j < hmm._states; j++) { // not required actually
            prob[i][j] = 0;
            prevs[i][j] = 0;
        }
    }

    // Get Initial Prob state
    for (int i = 0; i < hmm._states; i++) {
        prob[0][i] = hmm._state_prob[i] * hmm._emissionMatrix[i][ observed[0] ];
    }

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < hmm._states; j++) {
            double maxProb = -DBL_MAX;
            double eachProb;
            int maxIndex;
            // Get the maxProb that best maps from the previous sequence
            for (int k = 0; k < hmm._states; k++) {
                eachProb = prob[i-1][k] * hmm._stateTransitions[k][j];
                if (eachProb > maxProb) {
                    maxProb = eachProb;
                    maxIndex = k;
                }
            }
            prob[i][j] = maxProb * hmm._emissionMatrix[j][ observed[i] ];
            prevs[i-1][j] = maxIndex;
        }
    }

    double pmax = -DBL_MAX;
    int maxIndex;
    for (int i = 0; i < hmm._states; i++) {
        if (prob[n-1][i] > pmax) {
            pmax = prob[n-1][i];
            maxIndex = i;
        }
    }
    seq[n-1] = maxIndex;

    // Back Prop Of states
    for (int i = n-2; i >= 0; i--) {
        seq[i] = prevs[i][ seq[i+1] ];
    }

    /**
      * Debug statements here..
      */
    for (int i = 0; i < n; i++) {
        cout << "t = " << i << endl;
        for (int j = 0; j < hmm._states; j++) {
            cout << '[' << j << ']' << prob[i][j] << ' ';
        }
        cout << "\n\n";
    }

    for (int i = 0; i < n; i++) {
        cout << "t = " << i << endl;
        for (int j = 0; j < hmm._states; j++) {
            cout << '[' << j << ']' << prevs[i][j] << ' ';
        }
        cout << "\n\n";
    }

    cout << endl;
    cout << "Printing Sequence: " <<endl;
    for (int i = 0; i < n; i++) cout << '[' << i << ']' << seq[i] << ' ';
    cout << endl;
    /** End Debug **/

    // Clearing memory
    for (int i = 0; i < n; i++) {
        delete[] prob[i];
        delete[] prevs[i];
    }
    delete[] prob;
    delete[] prevs;

    return 0;
}

int main() {
    double A0[2] = {0.1, 0.9};
    double A1[2] = {0.1, 0.9};
    double *stateTransitions[2] = {A0, A1};
    double B0[4] = {0.6, 0.1, 0.2, 0.1};
    double B1[4] = {0.1, 0.7, 0.1, 0.1};
    double *stateObservations[2]= {B0, B1};
    double state_prob[2] = {0.2, 0.8};
    int observed[] = {0, 1, 1, 0, 1};

    HMM hmm(2, 4, stateTransitions, stateObservations, state_prob);
    viterbi(hmm, observed, 5);

    return 0;

}
