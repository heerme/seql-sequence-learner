/*
 * Author: Georgiana Ifrim (georgiana.ifrim@gmail.com)
 *
 * This library uses a model stored in a trie
 * for fast classification of a given document set.
 *
 * The classification threshold is tuned on the given set in order
 * to maximize accuracy. This code is used for tuning the classification
 * threshold on the training set.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
*/
#include <limits>
#include <vector>
#include <string>
#include <map>
#include "mmap.h"
#include <algorithm>
#include <cstdio>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <cmath>
#include "common_string_symbol.h"
#include "darts.h"
#include "sys/time.h"

static inline char *read_ptr (char **ptr, size_t size)
{
  char *r = *ptr;
  *ptr += size;
  return r;
}

template <class T> static inline void read_static (char **ptr, T& value)
{
  char *r = read_ptr (ptr, sizeof (T));
  memcpy (&value, r, sizeof (T));
}

template <typename T1, typename T2>
struct pair_2nd_cmp: public std::binary_function<bool, T1, T2> {
   bool operator () (const std::pair <T1, T2>& x1, const std::pair<T1, T2> &x2)
   {
      return x1.second > x2.second;
   }
};

class SEQLClassifier
{
private:

	MeCab::Mmap<char> mmap;
	double *alpha;
	Darts::DoubleArray da;
	std::vector <int>  result;
	std::vector <stx::string_symbol> doc;
	std::map <std::string, double> rules;
	bool userule;

	// Recursive traversal of strings starting at pos.
	// prefix: current prefix, pos: current pos in the document
	void project (std::string prefix, unsigned int pos, size_t trie_pos, size_t str_pos, bool token_type) {

	  		if (pos == doc.size() - 1) return;

	  		// Check traversal with both the next actual unigram in the doc and the wildcard *.
	  		string next_unigrams[2];
	  		next_unigrams[0] = doc[pos + 1].key();
	  		next_unigrams[1] = "*";

	  		for (int i = 0; i < 2; ++i) {

	  			string next_unigram = next_unigrams[i];
				std::string item;
				if (!token_type) { //word-level token
		  		   	item = prefix + " " + next_unigram;
				} else { // char-level token
					item = prefix + next_unigram;
				}

	  		   	//cout << "\nitem: " << item.c_str();
	  		   	size_t new_trie_pos = trie_pos;
	  		   	size_t new_str_pos  = str_pos;
	  		   	int id = da.traverse (item.c_str(), new_trie_pos, new_str_pos);
	  		   	//cout <<"\nid: " << id;

	  		   	//if (id == -2) return;
	  		    	if (id == -2) {
	  		    		if (i == 0) continue;
	  		    		else return;
				}
	  		   	if (id >= 0) {
				  if (userule) {
					//cout << "\nnew rule: " << item;
					rules.insert (std::make_pair <std::string, double> (item, alpha[id]));
				  }
				  result.push_back (id);
				}
				project (item, pos + 1, new_trie_pos, new_str_pos, token_type);
	  		}
	  }


public:
	double bias;

	SEQLClassifier(): userule(false) {};

  	void setRule(bool t) {
		userule = t;
  	}

  	bool open (const char *file)   	{

    	if (! mmap.open (file)) return false;

	    char *ptr = mmap.begin ();
	    unsigned int size = 0;
	    read_static<unsigned int>(&ptr, size);
	    da.set_array (ptr);
	    ptr += size;
	    read_static<double>(&ptr, bias);
	    alpha = (double *)ptr;

	    return true;
	}

	// Compute the area under the ROC curve.
	double calcROC( std::vector< std::pair<double, int> >& forROC ) {

	    //std::sort( forROC.begin(), forROC.end() );
	    double area = 0;
	    double x=0, xbreak=0;
	    double y=0, ybreak=0;
	    double prevscore = - numeric_limits<double>::infinity();
	    for( vector< pair<double, int> >::reverse_iterator ritr=forROC.rbegin(); ritr!=forROC.rend(); ritr++ )
	    {
	        double score = ritr->first;
	        int label = ritr->second;
	        if( score != prevscore ) {
	            area += (x-xbreak)*(y+ybreak)/2.0;
	            xbreak = x;
	            ybreak = y;
	            prevscore = score;
	        }
	        if( label > 0)  y ++;
	        else     x ++;
	    }
	    area += (x-xbreak)*(y+ybreak)/2.0; //the last bin
	    if( 0==y || x==0 )   area = 0.0;   // degenerate case
	    else        area = 100.0 * area /( x*y );
	    return area;
	}

	// Classify with several classif threshold provided.
	// Return classification results in vector<double> results_tuned_threshold.
	void classify (const char *line, double* predicted_score, bool token_type) {

	    result.clear ();
	    doc.clear ();
	    rules.clear ();

	// Prepare instance as a vector of string_symbol, where sting symbol is a word or a character depending on tokenization type
	    str2node (line, doc, token_type);

	    for (unsigned int i = 0; i < doc.size(); ++i) {
	       std::string item = doc[i].key();
	       int id;
         da.exactMatchSearch (item.c_str(), id);
	       //cout << "\ndoc[i]: " << doc[i].key();
	       //cout << "\nid: " << id;
	       if (id == -2) continue;
	       if (id >= 0) {
			 if (userule) {
			  	//cout << "\nnew rule: " << doc[i].key();
			   	rules.insert (std::make_pair <std::string, double> (doc[i].key(), alpha[id]));
			 }
			 result.push_back (id);
	       }
	       project (doc[i].key(), i, 0, 0, token_type);
	    }

		std::sort (result.begin(), result.end());
	    // Binary frequencies, erase the duplicate feature ids, features count only once.
		result.erase (std::unique (result.begin(), result.end()), result.end());

	    for (unsigned int i = 0; i < result.size(); ++i) {
	    	(*predicted_score) += alpha[result[i]];
	    }
	}

	std::ostream &printRules (std::ostream &os) {

	    std::vector <std::pair <std::string, double> > tmp;

	    for (std::map <std::string, double>::iterator it = rules.begin(); it != rules.end(); ++it)
	    	tmp.push_back (std::make_pair <std::string, double> (it->first,  it->second));

	    std::sort (tmp.begin(), tmp.end(), pair_2nd_cmp<std::string, double>());

	    //os << "rule: " << bias << " __DFAULT__" << std::endl;

	    for (std::vector <std::pair <std::string, double> >::iterator it = tmp.begin(); it != tmp.end(); ++it)
	    	os << "rule: " << it->second << " " << it->first << std::endl;

	    return os;
	}
};

#define OPT " [-v verbose] [-n token_type: 0 word tokens, 1 char tokens] test_file binary_model_file"

int main (int argc, char **argv) {

 	std::istream *is = 0;
  	unsigned int verbose = 0;
	// By default char token.
	bool token_type = 1;
	// Profiling variables.
	struct timeval t;
	struct timeval t_origin;

	gettimeofday(&t_origin, NULL);

	int opt;
  	while ((opt = getopt(argc, argv, "n:v:")) != -1) {
    	switch(opt) {
    		case 'n':
  			token_type = atoi(optarg);
  			break;
    		case 'v':
		       	verbose = atoi (optarg);
		       	break;
		default:
		      	std::cout << "Usage: " << argv[0] << OPT << std::endl;
		      	return -1;
    	}
  	}

  	if (argc < 3) {
    	std::cout << "Usage: " << argv[0] << OPT << std::endl;
    	return -1;
  	}

	if (! strcmp (argv[argc - 2], "-")) {
		is = &std::cin;
	} else {
		is = new std::ifstream (argv[argc - 2]);
	    if (! *is) {
	      std::cerr << argv[0] << " " << argv[argc-2] << " No such file or directory" << std::endl;
	      return -1;
	    }
	  }

	SEQLClassifier seql;

	if (verbose >= 3) seql.setRule (true);

	if (! seql.open (argv[argc-1])) {
    	std::cerr << argv[0] << " " << argv[argc-1] << " No such file or directory" << std::endl;
    	return -1;
  	}

  	std::string line;
  	char *column[4];

	// Predicted score for a single document.
  	double predicted_score = 0;
  	// Predicted and true scores for all docs.
  	vector<pair<double, int> > scores;
  	// Total number of true positives.
  	unsigned int num_positives = 0;
  	// Total number of docs.
  	unsigned int all = 0;

  	cout << "\nreading training file for classif_tune_threshold...\n\n";
  	// Gather the predicted scores for all docs.
  	while (std::getline (*is, line)) {

    	if (line[0] == '\0' || line[0] == ';') continue;
    	if (line[line.size() - 1] == '\r') {
    		line[line.size() - 1] = '\0';
    	}

    	if (2 != tokenize ((char *)line.c_str(), "\t ", column, 2)) {
      		std::cerr << "Format Error: " << line.c_str() << std::endl;
      		return -1;
    	}

//    	cout <<"\ncolumn[0]:*" << column[0] << "*";
//    	cout <<"\ncolumn[1]:*" << column[1] << "*";
//    	cout.flush();

	    int y = atoi (column[0]);
	    predicted_score = 0;
	    seql.classify (column[1], &predicted_score, token_type);
	    // Keep predicted and true score.
		scores.push_back(pair<double, int>(predicted_score, y));

		// Transform the predicted_score which is a real number, into a probability,
	    // using the logistic transformation: exp^{predicted_score} / 1 + exp^{predicted_score} = 1 / 2+ e^{-predicted_score}.
	    double predicted_prob;
	    if (predicted_score < -8000) {
	    	predicted_prob = 0;
	    } else {
	    	predicted_prob = 1.0 / (1.0 + exp(-predicted_score));
	    }
	    if (verbose == 1) {
	      		std::cout << y << " " << predicted_score << " " << predicted_prob << std::endl;
	    } else if (verbose == 2) {
	      			std::cout << y << " " << predicted_score << " " << predicted_prob <<  " " << column[1] << std::endl;
	    } else if (verbose >= 3) {
	      	std::cout << "<instance>" << std::endl;
	      		std::cout << y << " " << predicted_score << " " << predicted_prob << " " << column[1] << std::endl;
	      seql.printRules (std::cout);
	      std::cout << "</instance>" << std::endl;
	    }

	    ++all;
		if (y > 0) {
			++num_positives;
		}
  	}

  	// Sort the scores ascendingly by the predicted score.
  	sort(scores.begin(), scores.end());
	double AUC = seql.calcROC(scores);

    std::printf ("AUC:		 %.5f%%\n", AUC);
   	std::printf ("(1 - AUC): %.5f%%\n", (100 - AUC));
	// Choose the threshold that minimized the errors on training data.
	// Same as Madigan et al BBR.

	// Start by retrieving all, e.g. predict all as positives.
	// Compute the error as FP + FN.
	unsigned int TP = num_positives;
  	unsigned int FP = all - num_positives;
  	unsigned int FN = 0;
  	unsigned int TN = 0;

	unsigned int min_error = FP + FN;
	unsigned int current_error = 0;
	double best_threshold = -numeric_limits<double>::max();

	for (unsigned int i = 0; i < all; ++i) {
		// Take only 1st in a string of equal values
		if (i != 0 && scores[i].first > scores[i-1].first) {
            current_error = FP + FN; // sum of errors, e.g # training errors
            if (current_error < min_error) {
                min_error = current_error;
                best_threshold = (scores[i-1].first + scores[i].first) / 2;
                //cout << "\nThreshold: " << best_threshold;
                //cout << "\n# errors (FP + FN): " << min_error;
                //std::printf ("\nAccuracy: %.5f%% (%d/%d)\n", 100.0 * (TP + TN) / all, TP + TN, all);
            }
        }
        if (scores[i].second > 0) {
            FN++; TP--;
        }else{
            FP--; TN++;
        }
    }

	// Finally, check the "retrieve none" situation
    current_error = FP + FN;
    if (current_error < min_error) {
    	min_error = current_error;
        best_threshold = scores[all-1].first + 1;
        //cout << "\nThreshold (retrieve none): " << best_threshold;
        //cout << "\n# errors (FP + FN): " << min_error;
        //std::printf ("\nAccuracy: %.5f%% (%d/%d)\n", 100.0 * (TP + TN) / all, TP + TN, all);
    }

	// This procedure finds best_threshold such as if(predicted_score > best_threshold) classify pos;
	// Our seql_classify code uses predicted_score + bias > 0, thus we need to take -threshold.

	gettimeofday(&t, NULL);
    cout << "end classification( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )\n";
    cout.flush();

//    cout << "\nBest Threshold: " << best_threshold;
    cout << "\n# errors (FP + FN): " << min_error;
    std::printf ("\nAccuracy: %.5f%% (%d/%d)\n", 100.0 * (all - min_error) / all, all - min_error, all);

//   	std::cout << "\nBias (-best_threshold):" << -best_threshold << std::endl;
    cout << "\nBest threshold: " << best_threshold;
   	if (is != &std::cin) delete is;

   return 0;
}
