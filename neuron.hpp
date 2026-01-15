#include "data.hpp"
#include <vector>
#include <math.h>

#pragma once

class Neuron {
public:

    Neuron() {
        num = -1;
    }

    Neuron(int i) {
        num = i;
        numChild = 0;
        numParent = 0;
        isOutput = false;
        isInput = false;
        output = 0;
        error = 0;
        if(i <= 6) {
            //first 7 neurons are for input only
            isInput = true;
        } else if(i >= NUM_NEURONS - 3) {
            //final 3 neurons are output
            isOutput = true;
        }
    }

    //propagation functions

    //only for input neurons!
    void set_value(double d) {
        output = d;
    }

    double get_output() {
        return output;
    }

    double get_error() {
        return error;
    }

    //calculate forward propagation output
    //takes array of weights as parameter to make calculations
    void forward(double **weight) {
        if(isInput) {
            return;
        }
        double sum = 0;
        //make sure to clear the previous test
        output = 0;
        for(int i = 0; i < numParent; i++) {
            Neuron *n = parents.at(i);
            sum += weight[n->get_num()][num]*n->get_output();
        }
        //sigmoid activation function
        output = (double)1/(1+exp(-sum));
    }

    /* d represents desired output
     * for all neurons that aren't input: calculate error
     * for all neurons that aren't output: calculate delta weight
     * for each edge leading to its children
     */
    void backward(double **weight, double **prev_weight, int *d) {
        //error calculation
        if(isOutput) {
            //output specific equation
            int desired_value = d[num-NUM_NEURONS+3];
            error = (double)output*(1-output)*(desired_value-output);
            return;
        }
        if(!isInput) {
            //hidden neuron specific equation
            double sum = 0;
            for(int i = 0; i < numChild; i++) {
                Neuron *n = children.at(i);
                sum += (double)weight[num][n->get_num()]*n->get_error();
            }
            error = (double)output*(1-output)*sum;
        }

        //change weights
        //also implements momentum
        for(int i =0; i < numChild; i++) {
            Neuron *n = children.at(i);
            double change = (double)LEARNING_RATE*output*n->get_error();
            //momentum calc
            double m = MOMENTUM*prev_weight[num][n->get_num()];
            //update previous weight change
            prev_weight[num][n->get_num()] = change;
            weight[num][n->get_num()] = weight[num][n->get_num()] + change + m;
        }
    }

    //setter and getter functions
    void set_child(Neuron *n) {
        children.push_back(n);
        numChild++;
        n->set_parent(this);
    }

    void set_parent(Neuron *n) {
        parents.push_back(n);
        numParent++;
    }

    int get_num() {
        return num;
    }

    int num_parents() {
        return numParent;
    }

    //i assumed to be valid
    Neuron *get_parent(int i) {
        return parents.at(i);
    }

    int num_childen() {
        return numChild;
    }

    //i assumed to be valid
    Neuron *get_child(int i) {
        return children.at(i);
    }

    //debug function, shows specific details of neuron
    void print() {
        std::cout << "Neuron " << num << ":" << std::endl;
        if(numChild == 0) {
            std::cout << "    No children" << std::endl;
        } else {
            std::cout << "    Children: ";
            for(int i = 0; i < numChild; i++) {
                std::cout << children.at(i)->get_num() << " ";
            }
            std::cout << std::endl;
        }
        if(numParent == 0) {
            std::cout << "    No parents" << std::endl;
        } else {
            std::cout << "    Parents: ";
            for(int i = 0; i < numParent; i++) {
                std::cout << parents.at(i)->get_num() << " ";
            }
            std::cout << std::endl;
        }  
        std::cout << "    Output: " << output << std::endl;
        std::cout << "    Error: " << error << std::endl;
    }


//derivative O'(p) = O(p)[1-O(p)]

//BACK PROPAGATION
//error signal = O'(t)[d(t)-O(t)]


private:

    //each neuron is labeled from 0 to (n-1)
    //numerical value is directly associated to type of neuron
    int num;

    //keeps track of relationships btwn neurons
    int numChild;
    int numParent;

    //nullptr if output
    std::vector<Neuron*> children;

    //nullptr if input
    std::vector<Neuron*> parents;

    bool isOutput;
    bool isInput;

    //math
    double output;
    double error;

};
