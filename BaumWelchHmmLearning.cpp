#include <iostream>
#include <vector>
#include <limits>

using namespace std;

/**
  * Class that implements the forward-backward algorithm to get the unknowm
  * hidden state parameters of HMM.
  *
  * email: flair.rajesh@gmail.com
  */
class HMMLearning {

public:
    HMMLearning( int numOfStates_=2, int numOfObservations_=2 )
    {
        _stateProb.resize(numOfStates_);
        _transitionMatrix.resize(numOfStates_);
        _emissionMatrix.resize(numOfStates_);

        for(unsigned int index = 0; index < _transitionMatrix.size(); index++)
        {
            _transitionMatrix[ index ].resize(numOfStates_);
        }
        for(unsigned int index = 0; index < _emissionMatrix.size(); index++)
        {
            _emissionMatrix.resize(numOfObservations_);
        }

    }
    HMMLearning( std::vector<double> stateProb_, std::vector< std::vector<double> > stateTransition_, std::vector< std::vector<double> > emissionMatrix_ )
    {
        _stateProb = stateProb_;
        _transitionMatrix = stateTransition_;
        _emissionMatrix = emissionMatrix_;
    }

    std::vector< std::vector<double> > forwardPass( std::vector<double> observation_)
    {
        std::vector< std::vector<double> > eachObsProb;
        eachObsProb.resize(observation_.size());
        for(unsigned int index=0; index < eachObsProb.size(); index++)
        {
            eachObsProb[index].resize(_stateProb.size());
        }

        for(unsigned int index = 0; index < _stateProb.size(); index++)
        {
            eachObsProb[0][index] = _stateProb[index] * _emissionMatrix[index][observation_[0]];
        }

        for(unsigned int i=1; i < observation_.size(); i++)
        {
            for(unsigned int j=0; j<_stateProb.size(); j++)
            {
                for(unsigned int prevState = 0; prevState < _stateProb.size(); prevState++)
                {
                    eachObsProb[i][j] = eachObsProb[i][j] + _transitionMatrix[prevState][j] * eachObsProb[i-1][prevState];
                }

                eachObsProb[i][j] = eachObsProb[i][j] * _emissionMatrix[j][observation_[i]];
            }
        }
        return eachObsProb;
    }

    std::vector< std::vector<double> > backwardPass( std::vector<double> observation_ )
    {
        std::vector< std::vector<double> > eachObsProb;
        eachObsProb.resize(observation_.size());
        for(unsigned int index=0; index < eachObsProb.size(); index++)
        {
            eachObsProb[index].resize(_stateProb.size());
        }

        for(unsigned int index = 0; index < _stateProb.size(); index++)
        {
            eachObsProb[eachObsProb.size()-1][index] = 1.0;
        }
        for( unsigned int i = eachObsProb.size()-1; i >= 0; i-- )
        {
            for(unsigned int j = 0; j < _stateProb.size(); j++)
            {
                for( unsigned int prevState = 0; prevState < _stateProb.size(); prevState++ )
                {
                    eachObsProb[i-1][j] = eachObsProb[i-1][j] + eachObsProb[i][j] * _emissionMatrix[j][observation_[i]] * _transitionMatrix[prevState][j];
                }
            }
        }
    }

    void update( std::vector< std::vector<double> > forwardObsProb_, std::vector< std::vector<double> > backwardObsProb_, std::vector<double> observations_ )
    {
        std::vector< std::vector<double> > gamma;
        std::vector<double> gammaStates;
        gamma.resize(forwardObsProb_.size());

        std::vector< std::vector<double> > transitions;
        std::vector<double> transitionStateTags;
        transitionStateTags.resize(_stateProb.size());
        transitions.resize(_stateProb.size());

        for(unsigned int k=0; k<transitions.size(); k++)
        {
            transitions.resize( _stateProb.size() );
        }

        for(unsigned int i=0; i < gamma.size(); i++)
        {
            gamma[i].resize(_stateProb.size());
        }
        gammaStates.resize(_stateProb.size());


        double finalProb = 0.0;
        for(unsigned int k=0; k<_stateProb.size(); k++)
        {
            finalProb = finalProb + forwardObsProb_[forwardObsProb_.size()-1][k];
        }

        for(unsigned int k=gamma.size()-1; k<=0 ; k++)
        {
            for(unsigned int currState=0; currState < _stateProb.size(); currState++)
            {
                gamma[k][currState] = gamma[k][currState] + ((forwardObsProb_[k][currState] * backwardObsProb_[k][currState])/finalProb);
                gammaStates[currState] = gammaStates[currState] + gamma[k][currState];
                for( int prevState = 0; prevState < _stateProb.size(); prevState++ )
                {
                    double p = _transitionMatrix[prevState][currState] * _emissionMatrix[currState][observations_[k]];
                    transitions[prevState][currState] = transitions[prevState][currState] + ((forwardObsProb_[k-1][prevState] * p * backwardObsProb_[k][currState])/finalProb);
                    transitionStateTags[prevState] = transitionStateTags[prevState] + transitions[prevState][currState];
                }
            }
        }

        std::vector< std::vector<double> > tranMatrix = _transitionMatrix;
        std::vector< std::vector<double> > emitMatrix = _emissionMatrix;
        for(unsigned int stateIndex=0; stateIndex < _stateProb.size(); stateIndex++)
        {
            for( unsigned int eachObs=0; eachObs < observations_.size(); eachObs++ )
            {
                double obsOccuranceCount = 0;
                // Update Emission Probability
                for(unsigned int obsIndex=0; obsIndex < observations_.size(); obsIndex++)
                {
                    if( observations_[obsIndex] == eachObs )
                    {
                        obsOccuranceCount = obsOccuranceCount + 1;
                    }
                }
                emitMatrix[stateIndex][eachObs] =  (obsOccuranceCount * gamma[eachObs][stateIndex]) / gammaStates[stateIndex];
            }
            for( int nextstateIndex = 0; nextstateIndex < _stateProb.size(); nextstateIndex++ )
            {
                // Update Transition Probabilities.
                tranMatrix[stateIndex][nextstateIndex] = transitions[stateIndex][nextstateIndex] / gammaStates[stateIndex];
            }
        }
    }

    std::vector< std::vector<double> > _transitionMatrix;
    std::vector< std::vector<double> > _emissionMatrix;
    std::vector<double> _stateProb;
    std::vector<double> _expectedObervationState;
};

/**
 * @brief main
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char** argv) {

    return 0;
}
