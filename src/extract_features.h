#include <iostream>
#include <vector>
#include <queue>
#include <string.h>

using namespace std;

struct TermChi {
    int featureID;
    double value;
    TermChi():featureID(0), value(0){};
    TermChi(int f, int v):featureID(f), value(v) {};
};

struct classData {
    int len;
    int *classes;

    classData(int l);
    classData();
    ~classData();
    void init(int l);
    unsigned int getSum();
    int getClassDF(int);
};

int classData::getClassDF(int i) {
    return classes[i];
}

classData::classData():len(0) {
    classes = NULL;
}

classData::~classData() {
    delete[] classes;
}

classData::classData(int l):len(l) {
    classes = new int[l];

    for (int i = 0; i < l; i++) {
        classes[i] = 0;
    }
}

void classData::init(int l) {
    len = l;
    classes = new int[l];
    // Never use this, this is unstable!
    // bzero(classes, l);
    // May use this: memset(classes, 0, l*4);
    
    for (int i = 0; i < l; i++) {
        classes[i] = 0;
    }
}

unsigned int classData::getSum() {
    unsigned int sum = 0;

    for (int i = 0; i < len; i++) {
        sum += classes[i];
    }

    return sum;
}

int atoi(const char *input) {
    int result = 0;
    for(int i = 0; input[i] != '\0'; i++) {
        result *= 10;
        result += input[i] - '0';
    }

    return result;
}

enum Mode {
    TEST,
    TRAIN
};

double computeChiSquare(double, double,double,double);
void getTrainMatrices();
void getTestMatrices();
void getFeatures(int TermDF[], int *TermDCF, Mode mode);
void initialize(char *train, char *test, char *voc, char *output, int m1, char* featureFile);
void initialize(char *train, char *test, char *voc, char *output);

void reportError() { 
    cout << "Format Error" << endl; 
}

void usage();


// This is wrong, because it can not identify char * and string! 
/*
   int atoi(string &input) {
   int result = 0;
   size_t counter = 0;
   while(counter != string::npos) {
   result *= 10;
   result += input[counter++] - '0';
   }

   return result;
   }
   */
