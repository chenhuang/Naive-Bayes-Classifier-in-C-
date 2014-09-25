#include <iostream>
#include "extract_features.h"
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <queue>
#include <string.h>
#include <map>

using namespace std;

string trainFile;
string testFile;
string output;
string vocabulary;
int m = 0;
string featureFile;
int classes = 0;
int docs = 0;
int vocSize = 0;
const short shortargc = 5;
const short longargc = 7;
int *classDF;
map<int, string>id2term;
bool *listOfFeatures;

int main(int argc, char* argv[]) {
    if (argc != shortargc and argc != longargc) {
        usage(); 
        return 1;
    } 

    if (argc == shortargc) {
        initialize(argv[1], argv[2], argv[3], argv[4]);
    }

    if (argc == longargc) {
        initialize(argv[1], argv[2], argv[3], argv[4], atoi(argv[5]), argv[6]);
    }

    getTrainMatrices();
    getTestMatrices();
}

void getTestMatrices() {

    ifstream fin;
    fin.open(testFile.c_str());

    string line;

    // get # of classes and # of documents
    if (getline(fin, line)) {
        if (line.find('#') != string::npos) {
            size_t found_class = line.find("class:");
            size_t found_doc = line.find("docs:");

            if (found_class != string::npos && found_doc != string::npos) {
                classes = atoi(line.substr(found_class+6, found_doc - found_class - 7).c_str()); 
                docs = atoi(line.substr(found_doc + 5).c_str());
            } else {
                reportError();
            }
        } else {
            reportError();
        }
    }

    // Now get the data and build:
    // 1. TermID -> DF
    // 2. TermID -> C_1: DCF (Document Class Frequency)
    //           -> C_2: DCF
    //           ...
    //
    // Then get:
    // 3. C1 -> Term1 -> Chi_square1
    //       -> Term2 -> Chi_square2
    //       ...
    int TermDF[vocSize];
//    vector<classData> TermDCF(vocSize);
    int TermDCF[vocSize][classes];

    for (int i = 0; i < vocSize; i++) {
        TermDF[i] = 0;
        for (int j = 0; j < classes; j++) {
            TermDCF[i][j] = 0;
        }
    }

    while (getline(fin, line)) {

        // get the class number first
        size_t classLabelPos = line.find_first_of(' ');
        int cls = 0;
        if (classLabelPos != string::npos) {
            cls = atoi(line.substr(0, classLabelPos).c_str());
        }

        // get features and TF
        size_t found1 = classLabelPos;
        size_t found2 = line.find_first_of(' ', found1+1);

        while(line.substr(found1+1,1).compare("#") != 0) {
            string token = line.substr(found1+1, found2-found1);
            size_t foundColon = token.find(':');
            int featureID = atoi(token.substr(0, foundColon).c_str());
            TermDF[featureID]++;

            TermDCF[featureID][cls-1]++;

            found1 = found2;
            found2 = line.find_first_of(' ', found2+1);
        }
    }

    fin.close();

    /*
     * Test case, verify our data
     for (int i = 0; i < vocSize; i++) {
     cout <<i << ":" <<TermDF[i] << endl;
     cout << i << ":" << TermDCF[i].getSum() << endl;
     }
     */

    // Now get the chi-square features
    getFeatures(TermDF, &TermDCF[0][0], TEST);
}

void getTrainMatrices() {

    ifstream fin;
    fin.open(trainFile.c_str());

    string line;

    // get # of classes and # of documents
    if (getline(fin, line)) {
        if (line.find('#') != string::npos) {
            size_t found_class = line.find("class:");
            size_t found_doc = line.find("docs:");

            if (found_class != string::npos && found_doc != string::npos) {
                classes = atoi(line.substr(found_class+6, found_doc - found_class - 7).c_str()); 
                docs = atoi(line.substr(found_doc + 5).c_str());
            } else {
                reportError();
            }
        } else {
            reportError();
        }
    }

    // intialize classDF array
    classDF = new int[classes];
    for (int i = 0; i < classes; i++)
        classDF[i] = 0;
    //bzero(classDF, classes);


    // Now get the data and build:
    // 1. TermID -> DF
    // 2. TermID -> C_1: DCF (Document Class Frequency)
    //           -> C_2: DCF
    //           ...
    //
    // Then get:
    // 3. C1 -> Term1 -> Chi_square1
    //       -> Term2 -> Chi_square2
    //       ...
    int TermDF[vocSize];
    int TermDCF[vocSize][classes];

    for (int i = 0; i < vocSize; i++) {
        TermDF[i] = 0;
        for (int j = 0; j < classes; j++) {
            TermDCF[i][j] = 0;
        }
    }

    while (getline(fin, line)) {

        // get the class number first
        size_t classLabelPos = line.find_first_of(' ');
        int cls = 0;
        if (classLabelPos != string::npos) {
            cls = atoi(line.substr(0, classLabelPos).c_str());
        }

        // update classDF
        classDF[cls-1]++;

        // get features and TF
        size_t found1 = classLabelPos;
        size_t found2 = line.find_first_of(' ', found1+1);

        while(found1 != string::npos) {
            string token = line.substr(found1+1, found2-found1);
            size_t foundColon = token.find(':');
            int featureID = atoi(token.substr(0, foundColon).c_str());
            TermDF[featureID]++;

            //            cout << "TermDCF size: " << TermDCF.size() << endl;
            //            cout << " class vector size: " << TermDCF[featureID].size() << endl;

            TermDCF[featureID][cls-1]++;
            //            cout << "class: " << cls - 1 << "featureID: " << featureID << ":" <<TermDCF[featureID][cls-1] << endl;

            found1 = found2;
            found2 = line.find_first_of(' ', found2+1);
        }
    }

    fin.close();

    /*
     * Test case, verify our data
     for (int i = 0; i < vocSize; i++) {
     cout <<i << ":" <<TermDF[i] << endl;
     cout << i << ":" << TermDCF[i].getSum() << endl;
     }
     */

    // Now get the features
    getFeatures(TermDF, &TermDCF[0][0], TRAIN);
}

void getFeatures(int TermDF[], int *TermDCF, Mode mode) {

    // Open file handler to input/output feature files.
    // read the data again, compute TF*IDF, output feature files
    ifstream fin;
    ofstream fout;

    if (mode == TRAIN) {
        fin.open(trainFile.c_str());
        string out = output + ".train";
        fout.open(out.c_str());
    } else if (mode == TEST) {
        fin.open(testFile.c_str());
        string out = output + ".test";
        fout.open(out.c_str());
    } else {
        cout << "ERROR! No such option" << endl;
    }

    // Get the number of classes and number of documents first
    string line;
    if (getline(fin, line)) {
        if (line.find('#') != string::npos) {
            size_t found_class = line.find("class:");
            size_t found_doc = line.find("docs:");

            if (found_class != string::npos && found_doc != string::npos) {
                classes = atoi(line.substr(found_class+6, found_doc - found_class - 7).c_str()); 
                docs = atoi(line.substr(found_doc + 5).c_str());
            } else {
                reportError();
            }
        } else {
            reportError();
        }
    }

    // if we only want to get features using TF*IDF
    if (m == 0) {
        while (getline(fin, line)) {

            // get the class number first
            size_t classLabelPos = line.find_first_of(' ');
            int cls = 0;
            if (classLabelPos != string::npos) {
                cls = atoi(line.substr(0, classLabelPos).c_str());
            }

            // output class number
            fout << cls << " ";

            // get features and TF
            size_t found1 = classLabelPos;
            size_t found2 = line.find_first_of(' ', found1+1);

            if (mode == TEST) {
                while(line.substr(found1+1,1).compare("#") != 0) {
                    string token = line.substr(found1+1, found2-found1 - 1);
                    size_t foundColon = token.find(':');
                    int featureID = atoi(token.substr(0, foundColon).c_str());
                    int TF = atoi(token.substr(foundColon+1).c_str());
                    //cout << featureID << ":" << TF << " ";

                    // compute IDF and print out TF*IDF
                    int DF = TermDF[featureID];
                    double IDF = log(double(docs)/double(DF+1));
                    fout << featureID << ":" << double(TF) * IDF << " ";
                    // fout << featureID << ":" << double(TF) << " ";

                    found1 = found2;
                    found2 = line.find_first_of(' ', found2+1);
                }
                fout << "# " << line.substr(found1+3) << endl;
            } else if (mode == TRAIN) {
                while(found1 != string::npos) {
                    string token = line.substr(found1+1, found2-found1 - 1);
                    size_t foundColon = token.find(':');
                    int featureID = atoi(token.substr(0, foundColon).c_str());
                    int TF = atoi(token.substr(foundColon+1).c_str());
                    //cout << featureID << ":" << TF << " ";

                    // compute IDF and print out TF*IDF
                    int DF = TermDF[featureID];
                    double IDF = log(double(docs)/double(DF+1));
                    fout << featureID << ":" << double(TF) * IDF << " ";
                    // fout << featureID << ":" << double(TF) << " ";

                    found1 = found2;
                    found2 = line.find_first_of(' ', found2+1);
                }
                fout << endl;
            }
        }
    } else {
        // if we also wanted to get features using Chi-square
        // Chi-square is computed using:
        // http://nlp.stanford.edu/IR-book/html/htmledition/chi-square-feature-selection-1.html#eqn:chisquare

        // Class - Top m Terms, stored in a heap
        //priority_queue<TermChi> *CTT = new priority_queue<TermChi>[m];

        if (mode == TRAIN) {
            priority_queue<double> CTHeap[classes];
            map<double, vector<int> > chi2featureID[classes];

            // go through all terms and compute Chi-square, then filter out the top M features with largest Chi-square values.
            for (int j = 0; j < classes; j++) {

                for (int i = 1; i < vocSize; i++) {
                    if (TermDF[i] == 0) 
                        continue;
                    int N11 = TermDCF[i * classes + j];
                    int N10 = TermDF[i] - TermDCF[i* classes + j];
                    int N01 = classDF[j] - TermDCF[i* classes + j];
                    int N00 = docs - N01 - N10 - N11;
                    double chiSquare = computeChiSquare((double)N00, (double)N01, (double)N10, (double)N11);
                    // cout <<"class: " << j << " i:" << id2term[i] << ": "<< N00 << ", " << N01 << ", " << N10 << ", " << N11 << ":"<< chiSquare <<endl;

                    if (CTHeap[j].size() < (m/classes)) {
                        CTHeap[j].push(chiSquare);
                        chi2featureID[j][chiSquare].push_back(i);
                    } else {
                        // cout << CTHeap[j].top() << ", " << chiSquare << endl;
                        if (CTHeap[j].top() > chiSquare) {
                            double oldTop = CTHeap[j].top();

                            CTHeap[j].pop();
                            CTHeap[j].push(chiSquare);
                            chi2featureID[j][chiSquare].push_back(i);

                            int oldFeatureID = chi2featureID[j][oldTop].back();
                            chi2featureID[j][oldTop].pop_back();
                            // cout << i << ", " << chiSquare << endl;
                        }
                    }
                }
            }

            // output feature file
            ofstream fout;
            fout.open(featureFile.c_str());

            for (int i = 0; i < classes; i++) {
                while (CTHeap[i].size() != 0) {
                    double oldTop = CTHeap[i].top();
                    CTHeap[i].pop();
                    int id = chi2featureID[i][oldTop].back();
                    fout << id2term[id]  << "\t" << -1 * oldTop << endl;
                    chi2featureID[i][oldTop].pop_back();
                    listOfFeatures[id] = true;
                }
            }

            fout.close();

            /*
               for (int i = 1;i < vocSize; i++) {
               if (listOfFeatures[i])
               cout <<i <<" " <<listOfFeatures[i] << endl;
               }
               */
        }


        if (mode == TRAIN) {
            // output the feature file
            while (getline(fin, line)) {
                // get the class number first
                size_t classLabelPos = line.find_first_of(' ');
                int cls = 0;
                if (classLabelPos != string::npos) {
                    cls = atoi(line.substr(0, classLabelPos).c_str());
                }

                // output class number
                fout << cls << " ";

                // get features and TF
                size_t found1 = classLabelPos;
                size_t found2 = line.find_first_of(' ', found1+1);

                while(found1 != string::npos) {
                    string token = line.substr(found1+1, found2-found1 - 1);
                    size_t foundColon = token.find(':');
                    int featureID = atoi(token.substr(0, foundColon).c_str());
                    int TF = atoi(token.substr(foundColon+1).c_str());


                    // find if the ID exists in the listOfFeatures;
                    if (listOfFeatures[featureID]) {
                        // compute IDF and print out TF*IDF
                        int DF = TermDF[featureID];
                        double IDF = log(double(docs)/double(DF+1));
                        fout << featureID << ":" << double(TF) * IDF << " ";
                        // fout << featureID << ":" << double(TF) << " ";
                    }

                    found1 = found2;
                    found2 = line.find_first_of(' ', found2+1);
                }
                fout << endl;
                // TODO: edit here
            }
        } else if (mode == TEST) {
            // output the feature file
            while (getline(fin, line)) {
                // get the class number first
                size_t classLabelPos = line.find_first_of(' ');
                int cls = 0;
                if (classLabelPos != string::npos) {
                    cls = atoi(line.substr(0, classLabelPos).c_str());
                }

                // output class number
                fout << cls << " ";

                // get features and TF
                size_t found1 = classLabelPos;
                size_t found2 = line.find_first_of(' ', found1+1);

                while(line.substr(found1+1,1).compare("#") != 0) {
                    string token = line.substr(found1+1, found2-found1 - 1);
                    size_t foundColon = token.find(':');
                    int featureID = atoi(token.substr(0, foundColon).c_str());
                    int TF = atoi(token.substr(foundColon+1).c_str());

                    // find if the ID exists in the listOfFeatures;
                    // if (listOfFeatures[featureID]) {
                        // compute IDF and print out TF*IDF
                        int DF = TermDF[featureID];
                        double IDF = log(double(docs)/double(DF+1));
                        fout << featureID << ":" << double(TF) * IDF << " ";
                        // fout << featureID << ":" << double(TF) << " ";
                    // }

                    found1 = found2;
                    found2 = line.find_first_of(' ', found2+1);
                }

                fout << line.substr(found1+1) << endl;
            }
        }
    }

    fin.close(); 
    fout.close();

}

// This computes NEGATIVE chi-square in order to use the priority-queue container.
double computeChiSquare(double N00, double N01, double N10, double N11) {
    N00++;
    N01++;
    N10++;
    N11++;
    return -1 * ((N11+N10+N01+N00) * (N11*N00 - N10*N01) * (N11*N00 - N10*N01))/((N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00));
}

void initialize(char *train,  char *test, char *voc, char *output, int m1, char* featureFile) {
    trainFile = train; 
    testFile = test;
    ::output = output;
    vocabulary = voc;

    m = m1;
    if (featureFile == NULL) {
        ::featureFile = "";
    } else 
        ::featureFile = featureFile;

    // get vocabulary size
    ifstream fin1;
    fin1.open(vocabulary.c_str());

    string tmp;
    while(getline(fin1, tmp)) {
        vocSize++;
        size_t found = tmp.find(' ');
        int id = atoi(tmp.substr(0, found).c_str());
        string term = tmp.substr(found+1);

        id2term[id] = term;
    }

    vocSize++;

    listOfFeatures = new bool[vocSize];

    for (int i = 0; i < vocSize; i++)
        listOfFeatures[i] = false;
}

void initialize(char *train, char *test, char *voc, char *output) {
    initialize(train, test, voc, output, 0, NULL);
}

void usage() {
    cout << "Usage: ./extract_features.o pre_processed_training_file pre_processed_test_file output_dir [M feature_file]\n"\
        << "\t-pre_processed_training_file: feature file outputed by extract_features.pl\n" \
        << "\t-pre_processed_test_file: feature file outputed by extract_features.pl\n" \
        << "\t-pre_processed_vocabulary: vocabulary outputed by extract_features.pl\n" \
        << "\t-outpur: output prefix" << endl; 
}
