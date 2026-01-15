//holds array of likelihoods

#include "markov.hpp"
#include <fstream>

#pragma once

using namespace std;

class Curve {

public:

    Curve() {
        //set up velocity likelihoods
        for(int i = 0; i < MAX_VELOCITY *2; i++) {
            double v = (double)(i+1)/2;
            double c = (double)i*0.15;
            velocity[i] = new Likelihood(v, true, i);
            change[i] = new Likelihood(c, false, i);
            frequency[0][i] = 0;
            frequency[1][i] = 0;
        }
    }

    void print_velocity() {
        for(int i = 0; i < MAX_VELOCITY * 2; i++) {
            velocity[i]->print_likelihood();
        }
    }

    //assumes that the given file is valid
    void process_likelihoods(istream &input) {
        double l;
        for(int i =0; i < MAX_VELOCITY *2; i++) {
            input >> l;
            velocity[i]->add_likelihood(true, l);
        }
        for(int i =0; i < MAX_VELOCITY *2; i++) {
            input >> l;
            velocity[i]->add_likelihood(false, l);
        }

    }

    void process_training(istream &input) {
        double cur = 0;
        double prev = 0;
        double diff;
        int intdiff;
        string dummy;
        //bird then plane
        for(int t = 0; t < 2; t++) {
            int total = 0;
            for (int i =0; i < NUMBER_OF_SAMPLES; i++) {
                //reset
                cur = 0;
                prev = 0;
                //tracks number of seconds since last data point was found
                int sec = 0;

                for(int j = 0; j < TRACK_DURATION; j++) {
                    //catch NaN cases
                    input >> dummy;
                    sec++;
                    if(dummy == "NaN") {
                        continue;
                    }
                    cur = stod(dummy);
                    //check if a prev has already been set
                    if(prev == 0) {
                        prev = cur;
                        sec = 0;
                        continue;
                    }
                    total++;
                    diff = abs((prev - cur)/sec);
                    //convert to (rounded) index equivalent
                    intdiff = diff/0.15;
                    frequency[t][intdiff] = frequency[t][intdiff] + 1;
                    //reset
                    sec = 0;
                }      
            }
             
            for(int i = 0; i < MAX_VELOCITY*2; i++) {
                //convert frequency into likelihood    
                change[i]->add_likelihood((t+1)%2, (double)frequency[t][i]/(double)total);
            } 
        }
    }

    //calculate test using markov chaining
    void bayes(Markov *m, istream &input) {
        double v;   
        double c;
        string dummy;
        for(int i = 0; i < NUMBER_OF_TESTS; i++) {
            double vprev = 0;
            int sec = 0;
            for(int j = 0; j < TRACK_DURATION; j++) {
                input >> dummy;
                sec++;
                //catch NaN cases
                if(dummy == "NaN") {
                    continue;
                }
                v = stod(dummy);
                //convert v to index 
                int rounder = v *2;
                if(vprev == 0) {
                    m[i].init(velocity[rounder-1]);
                    vprev = v;
                    sec = 0;
                    continue;
                }
                c = abs((v - vprev)/sec);
                int c_round = c/0.15;
                m[i].update(velocity[rounder-1], change[c_round]);

                vprev = v;
                sec= 0;
            
            }
            //print result
            cout << (i+1) << ": ";
            m[i].print();
        }

    }


private:

    Likelihood *velocity[MAX_VELOCITY*2];

    //new element, this time change in velocity over time (1 second)
    //also contains 400 elements, each 0.15 kt so it makes out at a change of 60 knots
    //need to fill it in with data and then fill in gaps so there aren't 0 probability places in the middle
    Likelihood *change[MAX_VELOCITY*2];

    //2nd dimension, index 0 = change of 0 in velocity, increments by 0.15
    //1st dimension, 0 = bird, 1 = plane
    int frequency[2][MAX_VELOCITY*2];

    
};
