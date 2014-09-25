#include <string>
#include <map>
#include <vector>

#ifndef _NB_H_  
#define _NB_H_  

using namespace std;
class NaiveBayesClassifier {
    public: 
        NaiveBayesClassifier();
        NaiveBayesClassifier(char *input, char *output);
        void loadFromModel(string input);
        void load();
        void train();
        void dump();
        void test(string input, string output);

    private:
        string input;
        string output;
        map<int, int>classCount;
        int docSize;
        int vocSize;
        map<int, vector<double> >termClassCount;
        map<int, bool> vocabulary;
        vector<double>classTotTerm;
        vector<double>classWeight;
        map<int, vector<double> >termClassWeight;
};

#endif
