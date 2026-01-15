include "likelihood.hpp"
#include <cmath>
#include <math.h>
#include <fstream>

#pragma once

//for a given test, keeps track of current and previous probablity that
//it's a bird or plane as data gets processed 

class Markov {

public:

    Markov() {
        bird = 0;
        plane = 0;
        total = 0;
        bird_prev = 0;
        plane_prev = 0;
    }

    //initialize beginning probabilities
    void init(Likelihood *v) {
            bird = 0.5 * v->get_bird();
            plane = 0.5 * v->get_plane();
            //normalize
            total = bird + plane;
            bird = bird/total;
            plane = plane/total;

    }

    void update(Likelihood *v, Likelihood *c) {

            //markov chain calculation
            bird_prev = bird;
            plane_prev = plane;
            double bird_trans = 0.9 * bird_prev + 0.1 * plane_prev;
            double plane_trans = 0.9 * plane_prev + 0.1 * bird_prev;
            //uses logarithm to account for underflow
            bird = get_log(v->get_bird()) + get_log(c->get_bird()) + get_log(bird_trans);
            plane = get_log(v->get_plane()) + get_log(c->get_plane()) + get_log(plane_trans);
            //normalize
            bird = pow(10, bird);
            plane = pow(10, plane);
            total = bird + plane;
            bird = bird/total;
            plane = plane/total;

    }


    void print() {
        if(bird > plane) {
            std::cout << "This sample has been determined to be a bird. " << std::endl;
        } else {
            std::cout << "This sample has been determined to be a plane. " << std::endl;
        }
        std::cout << "It has a " << bird << " probability of being a bird and a " << plane
        << " probability of being a plane." << std::endl;
        
    }

private:

    double bird;
    double plane;
    double total;

    double bird_prev;
    double plane_prev;

    double get_log(double num) {
        if(num == 0) {
            //give small value
            return -16; 
        } else {
            return log10(num);
        }
    }

};
