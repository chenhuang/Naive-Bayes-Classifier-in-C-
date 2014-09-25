#include <iostream>
#include "NB.h"

using namespace std;

void usage() {
    cout << "Usage: ./learn examples.train model_file\n"
        << "\t-examples.train: train file\n"
        << "\t-model_file: output model file"
        << endl;
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        usage();
        return 1;
    }

    char *trainFile = argv[1];
    char *output = argv[2];

    NaiveBayesClassifier NB(trainFile, output); 
    NB.load();
    NB.dump();

    return 0;
}
