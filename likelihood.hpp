#include "globals.hpp"
#include <iostream>


#pragma once

//tracks the likelihood that a bird or plane would travel at a given velocity
//or would experience a given change in velocity over 1 second
  
class Likelihood {

public:

    //null initializer
    Likelihood() {
        
    }

    Likelihood(double v, bool type, int i) {
        value = v;
        isVelocity = type;
        index = i;
    }

    double get_bird() {
        return bird;
    }

    double get_plane() {
        return plane;
    }

    void add_likelihood(bool isBird, double l) {
        if(isBird) {
            bird = l;
        } else {
            plane = l;
        }
    }


    void print_likelihood() {
        if(isVelocity) {
            std::cout << "The likelihood for a velocity of " << value << " knots is ";
        } else {
            std::cout << "The likelihood for a change in velocity of " << value << " knots over 1 second is ";
        }
        std::cout << bird << " for birds and " 
            << plane << " for airplanes" << std::endl;
    }


private:

//velocities are discretely valued in 0.5 kt increments
//change in velocity (v') will probably be modeled in 0.15 kt increments
//if a given v' lacks a data point, use weighted average of two closest ones
double value;

//likelihood of given value for birds and airplanes
double bird;
double plane;

//identfies this particular likelihood as
//either for velocity or change in velocity over time
bool isVelocity;

//help find given likelihood in array
int index;
//closest likelihoods that have been attributed values
//only potentially needed for v' but not velocity!
int next;
int prev;

};
