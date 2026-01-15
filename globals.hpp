#include <string>

#pragma once

//almost all major parameters of the neural network can be modified here:

//string vars to identify different iris names in data
const std::string SETOSA = "Iris-setosa";
const std::string VERSICOLOR = "Iris-versicolor";
const std::string VIRGINICA = "Iris-virginica";

//number of data instances per class
const int NUM_INSTANCE = 50;
//set ratio for training:validation:testing data
const int NUM_TRAINING = 3*NUM_INSTANCE*8/10;
//const int NUM_VALIDATION = 3*NUM_INSTANCE*2/10;
const int NUM_TESTING = 3*NUM_INSTANCE*2/10;

//number of times the NN is trained with (random) dataset
const int TRAINING_ITERATIONS = 200;

//number of neurons in use
//represented by a fully connected neural network
//can be manipulated to test if ANN accuracy changes
//must be 4*n+6, where n >= 2
const int NUM_NEURONS = 14;

//gradient descent algorithm parameters
const double MOMENTUM = 0.40;
const double LEARNING_RATE = 0.7;

