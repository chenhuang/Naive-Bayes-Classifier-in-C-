#include <iostream>
#include <fstream>
#include <math.h>
#include "NB.h"
#include <stdlib.h>
#define MIN -100000

using namespace std;

/*
 * Train (Classes, Documents) {
 *      V = extractVocabulary(Documents);
 *      N = CountDocs(D);
 *
 *      foreach class(Classes) {
 *              Nc = countDocsInClass(class, Documents);
 *              Prior(c) = Nc/N;
 *
 *              foreach doc(Documents, C) {
 *                      foreach term(extractTerms(doc)) {
 *                              T[c][term]++;
 *                      }
 *              }
 *
 *              cond[c][t] = ...
 *      }
 * }
 */

// Here I used the "variant of the multinomial model"
// See http://nlp.stanford.edu/IR-book/html/htmledition/a-variant-of-the-multinomial-model-1.html 
// for more.
void NaiveBayesClassifier::test(string testFile, string output) {
    ifstream fin;
    ofstream fout;
    fin.open(testFile.c_str());
    fout.open(output.c_str());

    int rightDocs = 0;
    int wrongDocs = 0;

    string line;
    while(getline(fin, line)) {
        // if the sentence is empty, ignore it.
        size_t found = line.find(' ');
//        if (line.find_first_of(' ', found+1) == string::npos)
//            continue;

        // get the class label first
        int classLabel = atoi(line.substr(0, found).c_str());

        vector<int> featureVector;
        vector<double> featureValues;

        // then get the document vector model and get the MAP class
        size_t found1 = line.find_first_of(' ', found);
        size_t found2 = line.find_first_of(' ', found1+1);
        while (line.substr(found1+1, 1).compare("#") != 0) {
            string token = line.substr(found1+1, found2-found1-1);

            // split the token, get featureID and value
            size_t foundColon = token.find(':');
            int featureID = atoi(token.substr(0, foundColon).c_str());
            double value = atof(token.substr(foundColon+1, token.size()-foundColon).c_str());
            featureVector.push_back(featureID);
            featureValues.push_back(value);

            found1 = found2;
            found2 = line.find_first_of(' ', found1+1);
        }

        int maxClass = 1;
        double maxValue = MIN;

        // get the MAP class
        for (int i = 1; i < classWeight.size(); i++) {
            double map = 0;
            map += classWeight[i];
            for (int j = 0; j < featureVector.size(); j++) {
                //cout << "Feature: " << featureVector[j] << " Class: " << i << endl;
                // WHen the word has not seem in the class, skip
                // I'm not sure if this is correct, but it looks not
                if (termClassWeight[featureVector[j]].size() < i+1) { 
                    map += 0;
                } else 
                    map += featureValues[j] * termClassWeight[featureVector[j]][i];
            }
            // cout << "Doc: "<< line.substr(found1+3) << "class:" << i << ", weight:" << map << endl;
            if (map > maxValue) {
                maxValue = map;
                maxClass = i;
            }
        }


        if (classLabel == maxClass) {
            fout << line.substr(found1+3) << " " << maxValue << " " << maxClass << " " << classLabel << " y" << endl; 
            rightDocs++;
        } else {
            fout << line.substr(found1+3) << " " << maxValue << " " << maxClass << " " << classLabel << " n" << endl; 
            wrongDocs++;
        }
    }

    fin.close();
    cout << "Accuracy = " << 100 * (double)rightDocs/(double)(rightDocs+wrongDocs) << endl;
}

void NaiveBayesClassifier::loadFromModel(string input) {
    ifstream fin;
    fin.open(input.c_str());

    string line;

    getline(fin, line);
    int clsSize = atoi(line.c_str());
    getline(fin, line);
    this->vocSize = atoi(line.c_str());

    /*
     * log(Nci / N)
     */
    for (int i = 1; i < clsSize; i++) {
        getline(fin, line);
        size_t found = line.find("\t");
        int clsLabel = atoi(line.substr(0, found).c_str());
        double count = atof(line.substr(found+1).c_str());

        while (classWeight.size() < clsLabel+1) { 
            classWeight.push_back(0);
        }

        classWeight[clsLabel] = count;
    }

    /* 
     * get term class log values
     */
    while(getline(fin, line)) {
        size_t found = line.find("\t");
        int featureID = atoi(line.substr(0, found).c_str());
        double count = atof(line.substr(found+1).c_str());

        size_t found1 = found;
        size_t found2 = line.find_first_of("\t", found1+1);
        while(found1 != string::npos) {
            string token = line.substr(found1+1, found2-found1-1);

            int clsLabel = atoi(token.substr(0, token.find(':')).c_str());

            double classValue = atof(token.substr(token.find(':')+1).c_str());

            int tcwSize = termClassWeight[featureID].size();

            while (tcwSize < clsLabel+1) {
                termClassWeight[featureID].push_back(0);
                tcwSize++;
            }

            termClassWeight[featureID][clsLabel] = classValue;
            // cout << "FeatureID: " << featureID << ", clsLabel: "<< clsLabel << ", weight: " << termClassWeight[featureID][clsLabel] << endl;

            found1 = found2;
            found2 = line.find_first_of("\t", found1+1);
        }
    }

    fin.close();
}

// Load the feature file, and build the following tables/data:
// 1. Vocabulary Size
// 2. Document Size
// 2.5 Class Size
// 3. classDocument table
// 4. Term Class count table
// 5. class Term total count
void NaiveBayesClassifier::load() {
    ifstream fin;
    fin.open(input.c_str());

    string line;
    while(getline(fin, line)) {
        // if the sentence is empty, ignore it.
        size_t found = line.find(' ');
        if (line.find_first_of(' ', found+1) == string::npos)
            continue;

        // get class labelfirst 
        int classLabel = atoi(line.substr(0, found).c_str());

        // update Nc, N
        classCount[classLabel]++;
        docSize++;

        // get each token and update T(Ci, Ti);
        size_t found1 = line.find_first_of(' ', found);
        size_t found2 = line.find_first_of(' ', found1+1);
        while (found1 != string::npos) {
            string token = line.substr(found1+1, found2-found1-1);

            // split the token, get featureID and value
            size_t foundColon = token.find(':');
            int featureID = atoi(token.substr(0, foundColon).c_str());
            double value = atof(token.substr(foundColon+1, token.size()-foundColon).c_str());
            // cout << featureID << "\t" << value << endl;
            // count vocabulary size
            if (vocabulary[featureID] == false) {
                vocSize++;
                vocabulary[featureID] = true;
            }

            // update the term class count table

            int tcSize = termClassCount[featureID].size();

            while (tcSize < classLabel+1) {
                termClassCount[featureID].push_back(0);
                tcSize++;
            }

            termClassCount[featureID][classLabel] += value;

            // update class total term count table
            int ctSize = classTotTerm.size();
            while (ctSize < classLabel+1) {
                classTotTerm.push_back(0);
                ctSize++;
            }

            classTotTerm[classLabel] += value;

            found1 = found2;
            found2 = line.find_first_of(' ', found1+1);
        }
    }

    fin.close();
}

void NaiveBayesClassifier::dump() {
    ofstream out;
    out.open(output.c_str());
    // format of the model:
    // class size
    // vocSize
    // class DocumentCount (divied by document size and then log scale)
    // Term Class Count (log scale)

    out << classTotTerm.size() << endl;
    out << vocSize << endl;

    for (map<int, int>::iterator it = classCount.begin(); it != classCount.end(); it++) {
        out << (*it).first << "\t" << log((double)(*it).second/(double)docSize) << endl;
    }

    for (map<int, vector<double> >::iterator it = termClassCount.begin(); it != termClassCount.end(); it++) {
        out << (*it).first << "\t";

        for (int i = 1; i < classTotTerm.size(); i++) {
            out << i << ":" << log(((*it).second[i]+1)/(vocSize+classTotTerm[i]))<< "\t";
        }

        out << endl;
    }
    out.close();
}

NaiveBayesClassifier::NaiveBayesClassifier():docSize(0), vocSize(0), classWeight(0), classTotTerm(0) {
    ;
}

NaiveBayesClassifier::NaiveBayesClassifier(char *input, char *output): classTotTerm(0), classWeight(0) {
    this->input.assign(input);
    this->output.assign(output);
    docSize = 0;
    vocSize = 0;
}

