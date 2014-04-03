/*
 * Author: Georgiana Ifrim (georgiana.ifrim@gmail.com)
 *
 * This library uses a model stored in a trie
 * for fast classification of a given test set.
 *
 * A customized (tuned) classification threshold can be provided as input to the classifier.
 * The program simply applies a suffix tree model to the test documents for predicting classification labels.
 * Prec, Recall, F1 and Accuracy are reported.
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
  double bias;
  Darts::DoubleArray da;
  std::vector <int>  result;
  std::vector <stx::string_symbol> doc;
  std::map <std::string, double> rules;
  std::map <std::string, int> rules_and_ids;

  bool userule;
  int oov_docs;

  void project (std::string prefix,
		unsigned int pos,
		size_t trie_pos,
		size_t str_pos,
		bool token_type)
  {
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
					rules_and_ids.insert (std::make_pair <std::string, int> (item, id));
				}
			  result.push_back (id);
			}
			project (item, pos + 1, new_trie_pos, new_str_pos, token_type);
  		}
  }

public:

  SEQLClassifier(): userule(false), oov_docs(0) {};

  double getBias() {
  	return bias;
  }

  int getOOVDocs() {
  	return oov_docs;
  }

  void setRule(bool t)
  {
    userule = t;
  }

  bool open (const char *file, double threshold)
  {
    if (! mmap.open (file)) return false;

    char *ptr = mmap.begin ();
    unsigned int size = 0;
    read_static<unsigned int>(&ptr, size);
    da.setArray (ptr);
    ptr += size;
    read_static<double>(&ptr, bias);    // this bias from the model file is not used for classif; it is automatically obtained by summing 
					// up the features of the model and it is used for info only
    bias = -threshold;  //set bias to minus user-provided-thereshold
    
    alpha = (double *)ptr;

    return true;
  }

  // Compute the area under the ROC curve.
  double calcROC( std::vector< std::pair<double, int> >& forROC )
  {
    //std::sort( forROC.begin(), forROC.end() );
    double area = 0;
    double x=0, xbreak=0;
    double y=0, ybreak=0;
    double prevscore = - numeric_limits<double>::infinity();
    for( vector< pair<double, int> >::reverse_iterator ritr=forROC.rbegin(); ritr!=forROC.rend(); ritr++ )
    {
        double score = ritr->first;
        int label = ritr->second;
        //cout << "\nscore: " << score << " label: " << label;
        if( score != prevscore ) {
        	//cout << "\nx: " << x << " xbreak: " << xbreak << " y: " << y << " ybreak: " << ybreak;
        	area += (x-xbreak)*(y+ybreak)/2.0;
        	//cout << "\narea: " << area;
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
    //cout << "\narea: " << area;
    return area;
  }

  // Compute the area under the ROC50 curve.
  // Fixes the number of negatives to 50.
  // Stop computing curve after seeing 50 negatives.
  double calcROC50( std::vector< std::pair<double, int> >& forROC )
  {
    //std::sort( forROC.begin(), forROC.end() );
    double area50 = 0;
    double x=0, xbreak=0;
    double y=0, ybreak=0;
    double prevscore = - numeric_limits<double>::infinity();
    for( vector< pair<double, int> >::reverse_iterator ritr=forROC.rbegin(); ritr!=forROC.rend(); ritr++ )
    {
    	double score = ritr->first;
        int label = ritr->second;
        
        if( score != prevscore && x < 50) {
            area50 += (x-xbreak)*(y+ybreak)/2.0;
            xbreak = x;
            ybreak = y;
            prevscore = score;
        }
        if( label > 0)  y ++;
        else if (x < 50) x ++;
    }
    area50 += (x-xbreak)*(y+ybreak)/2.0; //the last bin
    if( 0==y || x==0 )   area50 = 0.0;   // degenerate case
    else        area50 = 100.0 * area50 /( 50*y );
    return area50;
  }
  
  double classify (const char *line, bool token_type)
  {
    result.clear ();
    doc.clear ();
    rules.clear ();
    double r = bias;

	// Prepare instance as a vector of string_symbol
    str2node (line, doc, token_type);

    for (unsigned int i = 0; i < doc.size(); ++i) {
    	std::string item = doc[i].key();
	    int id = da.exactMatchSearch (item.c_str());
    	//int id = da.exactMatchSearch (doc[i].key().c_str());
       	if (id == -2) continue;
       	if (id >= 0) {
			if (userule) {
		   		rules.insert (std::make_pair <std::string, double> (doc[i].key(), alpha[id]));
				rules_and_ids.insert (std::make_pair <std::string, int> (doc[i].key(), id));			
			}
		 	result.push_back (id);
       	}
       	project (doc[i].key(), i, 0, 0, token_type);
    }

    std::sort (result.begin(), result.end());

    // Binary frequencies, erase the duplicate feature ids, features count only once.
  	result.erase (std::unique (result.begin(), result.end()), result.end());

    if (result.size() == 0) {
    	if (userule)
    		cout << "\n Test doc out of vocabulary\n";
    	oov_docs++;
    }
    for (unsigned int i = 0; i < result.size(); ++i) r += alpha[result[i]];

    return r;
  }

  std::ostream &printRules (std::ostream &os)
  {
    std::vector <std::pair <std::string, double> > tmp;

    for (std::map <std::string, double>::iterator it = rules.begin();
	 it != rules.end(); ++it)
      tmp.push_back (std::make_pair <std::string, double> (it->first,  it->second));

    std::sort (tmp.begin(), tmp.end(), pair_2nd_cmp<std::string, double>());
	  os << "\nrule: " << bias << " __DFAULT__" << std::endl;

//    for (std::vector <std::pair <std::string, double> >::iterator it = tmp.begin();
//	 it != tmp.end(); ++it)
    for (std::map <std::string, double>::iterator it = rules.begin();
	 it != rules.end(); ++it)
	//os << "rule: " << rules_and_ids[it->first] << " " << it->second << " " << it->first << std::endl;
	os << "rule: " << it->first << " " << it->second << std::endl;

    return os;
  }

std::ostream &printIds (std::ostream &os) {
	for (std::map <std::string, int>::iterator it = rules_and_ids.begin(); it != rules_and_ids.end(); ++it)
		os << (it->second + 1) << ":1.0 ";
	os << "\n";

	return os;
}
};

#define OPT " [-n token_type: 0 word tokens, 1 char tokens] [-t classif_threshold] [-v verbose] test_file binary_model_file"

int main (int argc, char **argv)
{
  std::istream *is = 0;
  unsigned int verbose = 0;
  double threshold = 0; // By default zero threshold = zero bias.
  // By default char tokens.
  bool token_type = 1;
  // Profiling variables.
  struct timeval t;
  struct timeval t_origin;

	gettimeofday(&t_origin, NULL);

  int opt;
  while ((opt = getopt(argc, argv, "n:t:v:")) != -1) {
    switch(opt) {
    	case 'n':
  			token_type = atoi(optarg);
  			break;
    	case 't':
  			threshold = atof(optarg);
  			break;
     	case 'v':
       		verbose = atoi(optarg);
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

  if (! seql.open (argv[argc-1], threshold)) {
    std::cerr << argv[0] << " " << argv[argc-1] << " No such file or directory" << std::endl;
    return -1;
  }

  std::string line;
  char *column[4];
  // Predicted and true scores for all docs.
  vector<pair<double, int> > scores;
    
  unsigned int all = 0;
  unsigned int correct = 0;
  unsigned int res_a = 0;
  unsigned int res_b = 0;
  unsigned int res_c = 0;
  unsigned int res_d = 0;

  //cout << "\n\nreading test data...\n";
  while (std::getline (*is, line)) {

    	if (line[0] == '\0' || line[0] == ';') continue;
    	if (line[line.size() - 1] == '\r') {
    		line[line.size() - 1] = '\0';
    	}
    	//cout << "\nline:*" << aux.c_str() << "*";

    	if (2 != tokenize ((char *)line.c_str(), "\t ", column, 2)) {
      		std::cerr << "Format Error: " << line.c_str() << std::endl;
      		return -1;
    	}

    	//cout <<"\ncolumn[0]:*" << column[0] << "*";
    	//cout <<"\ncolumn[1]:*" << column[1] << "*";
    	//cout.flush();

	    int y = atoi (column[0]);
	    //cout << "\ny: " << y;
	    double predicted_score = seql.classify (column[1], token_type);

	    // Keep predicted and true score.
		scores.push_back(pair<double, int>(predicted_score, y));

		// Transform the predicted_score which is a real number, into a probability,
	    // using the logistic transformation: exp^{predicted_score} / 1 + exp^{predicted_score} = 1 / 1 + e^{-predicted_score}.
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
	    } else if (verbose == 4) {
	      	std::cout << "<instance>" << std::endl;
      		std::cout << y << " " << predicted_score << " " << predicted_prob << " " << column[1] << std::endl;
		seql.printRules (std::cout);
		std::cout << "</instance>" << std::endl;
	    } else if (verbose == 5) {
			std::cout << y << " ";
			seql.printIds (std::cout);
		}

	    all++;
	    if (predicted_score > 0) {
	      if(y > 0) correct++;
	      if(y > 0) res_a++; else res_b++;
	    } else {
	      if(y < 0) correct++;
	      if(y > 0) res_c++; else res_d++;
	    }
  }

   double prec = 1.0 * res_a/(res_a + res_b);
   if (res_a + res_b == 0) prec = 0;
   double rec  = 1.0 * res_a/(res_a + res_c);
   if (res_a + res_c == 0) rec = 0;
   double f1 =  2 * rec * prec / (prec+rec);
   if (prec + rec == 0) f1 = 0;

   double specificity = 1.0 * res_d/(res_d + res_b);
   if (res_d + res_b == 0) specificity = 0;
   // sensitivity = recall	
   double sensitivity  = 1.0 * res_a/(res_a + res_c);
   if (res_a + res_c == 0) sensitivity = 0;
   double fss =  2 * specificity * sensitivity / (specificity + sensitivity);
   if (specificity + sensitivity == 0) fss = 0;

   // Sort the scores ascendingly by the predicted score.
   sort(scores.begin(), scores.end());
   double AUC = seql.calcROC(scores);
   double AUC50 = seql.calcROC50(scores);
   double balanced_error = 0.5 * ((1.0 * res_c / (res_a + res_c)) + (1.0 * res_b / (res_b + res_d)));
   
	//if (verbose >= 3) {
		std::printf ("Classif Threshold:   %.5f\n", -seql.getBias());
	   	std::printf ("Accuracy:   %.5f%% (%d/%d)\n", 100.0 * correct / all , correct, all);
	   	std::printf ("Error:	  %.5f%% (%d/%d)\n", 100.0 - 100.0 * correct / all, all - correct, all);
	   	std::printf ("Balanced Error:	  %.5f%%\n", 100.0 * balanced_error);
	   	std::printf ("AUC:		  %.5f%%\n", AUC);
	   	//std::printf ("(1 - AUC):	 %.5f%%\n", 100 - AUC);
	   	std::printf ("AUC50:	  %.5f%%\n", AUC50);
	   	std::printf ("Precision:  %.5f%% (%d/%d)\n", 100.0 * prec,  res_a, res_a + res_b);
	   	std::printf ("Recall:     %.5f%% (%d/%d)\n", 100.0 * rec, res_a, res_a + res_c);
	   	std::printf ("F1:         %.5f%%\n",         100.0 * f1);
		std::printf ("Specificity:  %.5f%% (%d/%d)\n", 100.0 * specificity,  res_d, res_d + res_b);
	   	std::printf ("Sensitivity:     %.5f%% (%d/%d)\n", 100.0 * sensitivity, res_a, res_a + res_c);
	   	std::printf ("FSS:         %.5f%%\n",         100.0 * fss);

	   	std::printf ("System/Answer p/p p/n n/p n/n: %d %d %d %d\n", res_a,res_b,res_c,res_d);
	   	std::printf ("OOV docs:   %d\n", seql.getOOVDocs());

   		gettimeofday(&t, NULL);
   		cout << "end classification( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )\n";
   		cout.flush();
	//}
   if (is != &std::cin) delete is;

   return 0;
}
