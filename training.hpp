//holds dataset and runs NN iterations

#include "data.hpp"
#include "neuron.hpp"
 
#include <algorithm>
#include <random>
#include <fstream>
#include <cmath>

template<class RandomIt, class URBG> void shuffle(RandomIt first, RandomIt last, URBG&& g);

#pragma once

class Network {
public:

    //constructs ANN based on globals.hpp parameters
    Network() {
        weight = new double*[NUM_NEURONS];
        prev_weight = new double*[NUM_NEURONS];
        neurons = new Neuron[NUM_NEURONS];
        for(int i = 0; i < NUM_NEURONS; i++) {
            neurons[i] = Neuron(i);
            weight[i] = new double[NUM_NEURONS];
            prev_weight[i] = new double[NUM_NEURONS];
            for(int j = 0; j < NUM_NEURONS; j++) {
                //initially set all weights as 0
                //manually instantiate first weights in another function
                weight[i][j] = 0;
                prev_weight[i][j] = 0;
            }
        }
        set_relationships();
        set_weights();
    }

    //takes user inputs and returns closest flower classification
    void query() {
        bool done = false;
        double sl;
        double sw;
        double pl;
        double pw;
        std::cout << "Welcome to the iris classifier!" << std::endl;
        while(!done) {
            std::cout << "Enter sepal length: ";
            std::cin >> sl;
            std::cout << "Enter sepal width: ";
            std::cin >> sw;
            std::cout << "Enter petal length: ";
            std::cin >> pl;
            std::cout << "Enter petal width: ";
            std::cin >> pw;

            Data q = Data();
            q.process(sl, sw, pl, pw);

            classify(&q);

            std::cout << "Would you like to continue? (y/n) ";
            string ans;
            while(ans != "y" && ans != "n") {
                std::cin >> ans;
            }
            if(ans == "n") {
                done = true;
            }
        }
        std::cout << "Thank you!" << std::endl;
        return;
    }

    bool train() {
        //reset weights each time
        set_weights();
        double max_accuracy = 0;
        int max_accuracy_index = 0;
        for(int i = 0; i < TRAINING_ITERATIONS; i++) {
            double accuracy =  iterate();
            if (accuracy > max_accuracy) {
                max_accuracy = accuracy;
                max_accuracy_index = i;
                if(max_accuracy > 0.95) {
                    //training can end prematurely if it's good enough
                    break; 
                }
            }
            if(i%10 == 0) {
                //debug statement
                //std::cout << "The accuracy of this iteration is " << accuracy << std::endl; 
            }
        }
        
        if(max_accuracy > 0.95) {
            std::cout << "The greatest accuracy during this training is " << max_accuracy << std::endl;
            return true;
        }
        return false;
    }

    //single round of training
    double iterate() {
        //goes through randomized dataset
        randomize();
        //train
        for(int i = 0; i < NUM_TRAINING; i++) {
            propagate(&shuffle_d[i]);
        }
        //test
        int correct = 0;
        for(int i = NUM_TRAINING; i < 3*NUM_INSTANCE; i++) {
            if(forward(&shuffle_d[i])) correct++;
        }
        return (double)correct/NUM_TESTING;         
    }

    //for queries, similar to forward
    //but can't verify if the answer is correct
    void classify(Data *curr) {

        //instantiate input
        for(int i =0; i < 3; i++) {
            neurons[i].set_value(0);
        }
        for(int i = 0; i < 4; i++) {
            neurons[i+3].set_value(curr->get_val(i));
        }

        for(int i = 7; i < NUM_NEURONS; i++) {
            neurons[i].forward(weight);
        }

        std::string id = SETOSA;
        double max = neurons[NUM_NEURONS-3].get_output();
        if(neurons[NUM_NEURONS-2].get_output() > max) {
            max = neurons[NUM_NEURONS-2].get_output();
            id = VERSICOLOR;
        }
        if(neurons[NUM_NEURONS-1].get_output() > max) {
            max = neurons[NUM_NEURONS-1].get_output();
            id = VIRGINICA;
        }

        std::cout << "The flower had been identified as the " << id << std::endl;
        std::cout << "Its encoding was calculated as: ";
        for(int i = 0; i < 3; i++) {
            std::cout << neurons[NUM_NEURONS-3+i].get_output() << " ";
        }
        std::cout << std::endl;
    }

    //for testing
    //similar to propagate, but without the additional
    //backwards propagation
    //returns the accuracy of test
    bool forward(Data *curr) {

        //instantiate input
        int *name_encoding = curr->get_name();
        for(int i =0; i < 3; i++) {
            neurons[i].set_value(0);
        }
       
        for(int i = 0; i < 4; i++) {
            neurons[i+3].set_value(curr->get_val(i));
        }

        for(int i = 7; i < NUM_NEURONS; i++) {
            neurons[i].forward(weight);
        } 

        //test accuracy
        std::string id = SETOSA;
        double max = neurons[NUM_NEURONS-3].get_output();
        if(neurons[NUM_NEURONS-2].get_output() > max) {
            max = neurons[NUM_NEURONS-2].get_output();
            id = VERSICOLOR;
        }
        if(neurons[NUM_NEURONS-1].get_output() > max) {
            max = neurons[NUM_NEURONS-1].get_output();
            id = VIRGINICA;
        }

        if(id == curr->get_name_string()) {
            return true;
        }
        return false;
    }

    //for training
    void propagate(Data *curr) {

        //instantiate input
        int *name_encoding = curr->get_name();
        for(int i =0; i < 3; i++) {
            neurons[i].set_value(name_encoding[i]);
        }
        for(int i = 0; i < 4; i++) {
            neurons[i+3].set_value(curr->get_val(i));
        }
        //forward propagate
        for(int i = 7; i < NUM_NEURONS; i++) {
            neurons[i].forward(weight);
        }
        //backward propagate
        for(int i = NUM_NEURONS-1; i >= 0; i--) {
            neurons[i].backward(weight, prev_weight, name_encoding);
        }

        double *output = new double[3];

    }

    //processes initial dataset
    void process(istream &input) {
        for(int i = 0; i < 3 * NUM_INSTANCE; i++) {
            std::string s;
            input >> s;
            d[i].process(s);
            shuffle_d[i] = d[i];
        }
    }

    void print_neurons() {
        for(int i = 0; i < NUM_NEURONS; i++) {
            neurons[i].print();
        }
    }

    //debug functions
    //can also be used to see how the ANN develops over time
    void print_edges() {
        for(int i = 0; i < NUM_NEURONS; i++) {
            for(int j = 0; j < NUM_NEURONS; j++) {
                if(weight[i][j] != 0) {
                    std::cout << i << " to " << j << ": " << weight[i][j] << std::endl;
                }
            }
        }
    }

    //debug print functions
    //checks that data is being processed and shuffled properly
    void print_data() {
        for(int i = 0; i < 3 * NUM_INSTANCE; i++) {
            d[i].print();
        }
    }

    void print_data_rand() {
        randomize();
        for(int i = 0; i < 3 * NUM_INSTANCE; i++) {
            shuffle_d[i].print();
        }
    }

    //shuffles dataset to randomly partition into training vs. testing data
    void randomize() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(&shuffle_d[0], &shuffle_d[3*NUM_INSTANCE], gen);
    }

private:

    Data *d  = new Data[3*NUM_INSTANCE];

    //first 80% are for training
    //last 20% are for testing
    Data *shuffle_d = new Data[3*NUM_INSTANCE];

    /*array of neurons, begin by using 10 total
     *first 3 are the desired output neurons
     *3-6 are the input neurons, in order of:
     *      3: sepal lenth
     *      4: sepal width
     *      5: petal length
     *      6: petal width
     *7-(n-4) are the hidden neurons
     *last 3 are the output neuron
     */
    Neuron *neurons;

    //contains weights between neurons
    //in form of [i][j], where i is the parent
    //BE CAREFUL!! [j][i] will still be 0
    double **weight;

    //used for momentum calcs
    double **prev_weight;

    //helper function for construction of NN
    //manually initialize how neurons are connected and their starting weights
    void set_relationships() {
        for(int i = 7; i < NUM_NEURONS; i++) { 
            //training output neurons are connected to all hidden neurons and output
            for(int j = 0; j < 3; j++) {
                neurons[j].set_child(&neurons[i]);
            }
        }

        int num_layers = (NUM_NEURONS-6)/4;
        //fully connected layers
        for(int i = 0; i < 4; i++) {
            for(int j = 0; j < 4; j++) {
                for(int k = 0; k < num_layers; k++) {
                    neurons[i+3+num_layers*k].set_child(&neurons[j+7+num_layers*k]);
                }             
            }
            for(int j = 0; j < 3; j++) {
                neurons[i+NUM_NEURONS-7].set_child(&neurons[j+NUM_NEURONS-3]);
            }
        }

     
    }

    //initialize starting weights randomly as
    // 1/+-(#parents)
    //all prev weights are 0
    void set_weights() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(1, 2);

        for(int i = 0; i < NUM_NEURONS; i++) {
            int p = neurons[i].num_parents();
            for(int j = 0; j < p; j++) {
                int r = dis(gen);
                int num = neurons[i].get_parent(j)->get_num();
                if(r == 2) {
                    weight[num][i] = (double)1/p;
                } else { 
                    weight[num][i] = (double)-1/p;
                }
                prev_weight[num][i] = 0;
            }
        }
    }

};
