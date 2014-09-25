#include <iostream>
#include "NB.h"
#include "extract_features.h"

using namespace std;

void usage() {
    cout << "./classify.o examples.test model_file output_file\n"
        << "\t -examples.test: test file\n"
        << "\t -model_file: model file\n"
        << "\t -output_file: output file" << endl; 
}

int main(int argc, char *args[]) {
    if (argc != 4) {
        usage();
        return 0;
    }

    NaiveBayesClassifier NBC;
    NBC.loadFromModel(args[2]);
    NBC.test(args[1], args[3]);

    return 1;
}

