#include "training.hpp"
#include <fstream>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) 
{

    bool fileCommands = false;

    if(argc != 2) {
        cout << "Usage: ./Neural_Network [data.txt]" << endl;
        exit(EXIT_FAILURE);
    }

    ifstream instream;
    instream.open(argv[1]);

    if(!instream.is_open()) {
        cout << "Couldn't open file" << endl;
        exit(EXIT_FAILURE);
    }

    
    Network n;
    n.process(instream);

    //debug statements
    /*
    for(int k = 0; k < 10; k++) {
        for(int j = 0; j < 20; j++) {
            n.randomize();
        }
    }*/

    //continually retrains neural network until a satisfactory level of accuracy
    //is reached, shouldn't take many iterations if any
    while(!n.train()) {
        //repeat
    }

    n.query();
    
    
    instream.close();  
}
