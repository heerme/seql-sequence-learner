/*
 * Author: Georgiana Ifrim (georgiana.ifrim@gmail.com)
 * SEQL: Sequence Learner
 * This library trains ElasticNet-regularized Logistic Regression and L2-loss (squared-hinge-loss) SVM for Classifying Sequences in the feature space of all possible        
 * subsequences in the given training set.
 * Elastic Net regularizer: alpha * L1 + (1 - alpha) * L2, which combines L1 and L2 penalty effects. L1 influences the sparsity of the model, L2 corrects potentially high      
 * coeficients resulting due to feature correlation (see Regularization Paths for Generalized Linear Models via Coordinate Descent, by Friedman et al, 2010).
 *
 * The user can influence the outcome classification model by specifying the following parameters:
 * [-o objective] (objective function; choice between logistic regression and squared-hinge-svm. By default: logistic regression.)
 * [-T maxitr] (number of optimization iterations; by default this is set using a convergence threshold on the aggregated change in score predictions.)
 * [-l minpat] (constraint on the min length of any feature)
 * [-L maxpat] (constraint on the max length of any feature)
 * [-m minsup] (constraint on the min support of any feature, i.e. number of sequences containing the feature)
 * [-g maxgap] (number of consecutive gaps or wildcards allowed in a feature, e.g. a**b, is a feature of size 4 with any 2 characters in the middle)
 * [-n token_type] (word or character-level token to allow sequences such as 'ab cd ab' and 'abcdab')
 * [-C regularizer_value] value of the regularization parameter, the higher value means more regularization
 * [-a alpha] (weight on L1 vs L2 regularizer, alpha=0.5 means equal weight for l1 and l2)
 * [-r traversal_strategy] (BFS or DFS traversal of the search space), BFS by default
 * [-c convergence_threshold] (stopping threshold for optimisation iterations based on change in aggregated score predictions)
 * [-v verbosity] (amount of printed detail about the model)
 *
 * License:
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
 */

/* The obj fct is: loss(x, y, beta) + C * ElasticNetReg(alpha, beta).
*/

#include <cfloat>
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>
#include <strstream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <map>
#include <set>
#include <iterator>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include "common.h"
#include "sys/time.h"
#include <list>

using namespace std;

class SeqLearner {

private:

// Best ngram rule.
struct rule_t {
	// Gradient value for this ngram.
	double gradient;
	// Length of ngram.
	unsigned int size;
	// Ngram label.
	std::string  ngram;
	// Ngram support, e.g. docids where it occurs in the collection.
	std::vector <unsigned int> loc;
	friend bool operator < (const rule_t &r1, const rule_t &r2)
	{
	return r1.ngram < r2.ngram;
	}
};

// The search space rooted at this ngram.
struct space_t {
	// Last docid.
	int last_docid;
	// Pointer to previous ngram.
	space_t *prev;
	// Label of node where extension is done.
	string ne;
	// Total list of occurrences in the entire collection for an ngram. A sort of expanded inverted index.
	std::vector <int>       loc;
	// Vector of ngrams which are extensions of current ngram.
	std::vector <space_t *> next;

	// Add a doc_id and position of occurrence to the total list of occurrences,
	// for this ngram.
	// Encode the doc_id in the vector of locations.
	// Negative entry means new doc_id.
	void add (unsigned int docid, int pos) {
		if (last_docid != (int)docid) loc.push_back (-(int)(docid+1));
		loc.push_back (pos);
		last_docid = (int)docid;

	//cout << "\ndocid:" << docid << " pos:" << pos;
	}

	// Shrink the list of total occurrences to contain just support doc_ids, instead of doc_ids and occurences.
	void shrink () {
		std::vector<int> tmp;
		for (unsigned int i = 0; i < loc.size(); ++i) {
			if (loc[i] < 0) tmp.push_back (loc[i]);
		}
      		loc = tmp;
		std::vector<int>(loc).swap (loc); // shrink
		last_docid = -1;
	}

	// Return the support of current ngram.
	// Simply count the negative loc as doc_ids.
	unsigned int support () {
		unsigned int result = 0;
		for (unsigned int i = 0; i < loc.size(); ++i) {
			if (loc[i] < 0) ++result;
		}
		return result;
	}

	// Constructor for space_t.
	space_t(): last_docid(-1), prev(0) {};
};

	// Entire collection of documents, each doc represented as a string.
	// The collection is a vector of strings.
  	std::vector < string > transaction;
  	// True classes.
  	std::vector < int > y;
  	// The fraction: 1 / 1 + exp^(yi*beta^t*xi) in the gradient computation.
  	std::vector < long double > exp_fraction;
  	// Per document, sum of best beta weights, beta^t * xi = sum_{j best beta coord} gradient_j
  	std::vector < double > sum_best_beta;
  	// The scalar product obtained with the optimal beta according to the line search for best step size.
  	std::vector < double > sum_best_beta_opt;
  	// Regularized loss function: loss + C * elasticnet_reg
	// SLR loss function: log(1+exp(-yi*beta^t*xi))
	// Squared Hinge SVM loss function: sum_{i|1-yi*beta^t*xi > 0} (1 - yi*beta^t*xi)^2
  	long double loss;
  	long double old_loss; //keep loss in previous iteration for checking convergence

 	std::map <string, double> features_cache;
	map<string, double>::iterator features_it;

	// Objective function. For now choice between logistic regression and l2 (Squared Hinge Loss).
	unsigned int objective;
	// Regularizer value.
  	double C;
	// Weight on L1 vs L2 regularizer.
	double alpha;

  	// The sum of squared values of all non-zero beta_j.
  	double sum_squared_betas;

  	// The sum of abs values of all non-zero beta_j.
  	double sum_abs_betas;

  	std::set <string> single_node_minsup_cache;

  	// Current rule.
	rule_t       rule;
	// Current suboptimal gradient.
	double       tau;
	// Max length for an ngram.
	unsigned int maxpat;
	// Min length for an ngram.
	unsigned int minpat;
	// Min supoort for an ngram.
	unsigned int minsup;

	// Allowe features with gaps. Treat the gap as an additional unigram.
	// Max # of consecutive gaps allowed in a feature.
	unsigned int maxgap;

	// Total number of times the pruning condition is checked
	unsigned int total;
	// Total number of times the pruning condition is satisfied.
	unsigned int pruned;
	// Total number of times the best rule is updated.
	unsigned int rewritten;

	// Convergence threshold on aggregated change in score predictions.
	// Used to automatically set the number of optimisation iterations.
	double convergence_threshold;

	// Verbosity level: 0 - print no information,
	// 					1 - print profiling information,
	//					2 - print statistics on model and obj-fct-value per iteration
	//					> 2 - print details about search for best n-gram and pruning process
	int verbosity;

	// Type of token
	bool token_type;

	// Traversal strategy: BFS or DFS.
	bool traversal_strategy;

	// Profiling variables.
	struct timeval t;
	struct timeval t_origin;
	struct timeval t_start_iter;

	//long double LDBL_MAX = numeric_limits<long double>::max();

// Read the input training documents, "true_class document" per line.
// A line in the training file can be: "+1 a b c"
bool read (const char *filename) {
  	// Set the max line/document size to (10Mb).
  	const int kMaxLineSize = 10000000;
	char *line = new char[kMaxLineSize];
	char *column[5];
  	unsigned int num_pos = 0;
  	unsigned int num_neg = 0;
  	string doc;

  std::istream *ifs = 0;
  if (!strcmp(filename, "-")) {
    ifs = &std::cin;
  } else {
    ifs = new std::ifstream(filename);
    if (!*ifs) {
      std::cerr << "seql_learn" << " " << filename << " No such file or directory" << std::endl;
      return -1;
    }
  }

	if (! ifs) return false;
	cout << "\n\nread() input data....";

	while (ifs->getline (line, kMaxLineSize)) {
		if (line[0] == '\0' || line[0] == ';') continue;
	if (line[strlen(line) - 1 ] == '\r')
	  	line[strlen(line) - 1 ] = '\0';
	//cout << "\nline size: " << strlen(line);
	//cout.flush();

	  if (2 != tokenize (line, "\t ", column, 2)) {
	 	std::cerr << "FATAL: Format Error: " << line << std::endl;
	 	return false;
	}
	// Prepare class. _y is +1/-1.
	int _y = atoi (column[0]);
	y.push_back (_y);

	if (_y > 0) num_pos++;
	else num_neg++;

	// Prepare doc. Assumes column[1] is the original text, e.g. no bracketing of original doc.
	doc.assign(column[1]);
	transaction.push_back(doc);

	//cout << "\nscanning docid: " << transaction.size() - 1 << " class y:" << _y << " doc:" << doc <<"*";
	cout.flush();
	}
	cout << "\n# positive samples: " << num_pos;
	cout << "\n# negative samples: " << num_neg;
	cout << "\nend read() input data....";

	delete [] line;

	return true;
}

// For current ngram, compute the gradient value and check prunning conditions.
// Update the current optimal ngram.
bool can_prune (space_t *space, unsigned int size) {
	//struct timeval t_start;
  	//gettimeofday(&t_start, NULL);

    	++total;

	// Upper bound for the positive class.
	double upos = 0;
    	// Upper bound for the negative class.
    	double uneg = 0;
    	// Gradient value at current ngram.
    	double gradient = 0;
    	// Support of current ngram.
    	unsigned int support = 0;
    	//string reversed_ngram;	
    	string ngram;	
    	std::vector <int>& loc = space->loc;

	// Compute the gradient and the upper bound on gradient of extensions.
	for (unsigned int i = 0; i < loc.size(); ++i) {
		if (loc[i] >= 0) continue;
      		++support;
      		unsigned int j = (unsigned int)(-loc[i]) - 1;
	
		// Choose objective function. 0: SLR, 2: l2-SVM.
		if (objective == 0) { //SLR
			// From differentiation we get a - in front of the sum_i_to_N
			gradient -= y[j] * exp_fraction[j];

			if (y[j] > 0) {
				upos -= y[j] * exp_fraction[j];
			} else {
				uneg -= y[j] * exp_fraction[j];
			}
		} //end SLR (logistic regression)

		/* if (objective == 1) { //L1-SVM (Hinge loss)
			if (1 - y[j] * sum_best_beta[j] > 0) {
    		  		gradient -= y[j];

				if (y[j] > 0) {
					upos -= y[j];
				} else {
					uneg -= y[j];
				}
      			}
		} //end l1-SVM */

		if (objective == 2) { //L2-SVM
			if (1 - y[j] * sum_best_beta[j] > 0) {
    		  		gradient -= 2 * y[j] * (1 - y[j] * sum_best_beta[j]);

				if (y[j] > 0) {
					upos -= 2 * y[j] * (1 - y[j] * sum_best_beta[j]);
				} else {
					uneg -= 2 * y[j] * (1 - y[j] * sum_best_beta[j]);
				}
      			}
		} //end l2-SVM

	}
    

    //************ Debug info *****************
	if (C != 0) {    
		// Retrieve the current ngram
		if (!token_type) { // If word-level token: a bb cd a bb
		
			for (space_t *t = space; t; t = t->prev) {
				ngram = " " + t->ne + ngram;
			}
			// skip the space in front of the ngram
			ngram.assign(ngram.substr(1));

		} else { //char-level tokens: abbcdabb
			for (space_t *t = space; t; t = t->prev) {
				ngram = t->ne + ngram;
			}
		}
		if (verbosity > 3) {
			cout << "\ncurrent ngram rule: " << ngram;
			cout << "\nlocation size: " << space->loc.size() << "\n";
		 	//for (unsigned int i = 0; i < space->loc.size(); ++i)
		 	//	cout <<  space->loc[i] << " ";
		 	cout << "\ngradient (before regularizer): " << gradient;
			cout << "\nupos (before regularizer): " << upos;
			cout << "\nuneg (before regularizer): " << uneg;
			cout << "\ntau: " << tau;
		}
	} // end if (C != 0)
	//*****************************************************

	// Assume we can prune until bound sais otherwise.
	bool flag_can_prune = 1;
	
	// In case C != 0 we need to update the gradient and check the exact bound
	// for non-zero features.
	if ( C != 0 ) {
		double current_upos = 0;
		double current_uneg = 0;

		// Retrieve the beta_ij coeficient of this feature. If beta_ij is non-zero, 
		// update the gradient: gradient += C * [alpha*sign(beta_j) + (1-alpha)*beta_j];
		// Fct lower_bound return an iterator to the key >= given key.
		features_it = features_cache.lower_bound(ngram);
		// If there are keys starting with this prefix (this ngram starts at pos 0 in existing feature).
		if (features_it != features_cache.end() && features_it->first.find(ngram) == 0) {
			// If found an exact match for the key.
			// add regularizer to gradient.
			if (features_it->first.compare(ngram) == 0) {
				int sign = abs(features_it->second)/features_it->second;
				gradient += C * (alpha * sign + (1-alpha) * features_it->second);
			}
			
			if (verbosity > 3) {
				cout << "\ngradient after regularizer: " << gradient;
			}
			// Check if current feature s_j is a prefix of any non-zero features s_j'. 
			// Check exact bound for every such non-zero feature.
			while (features_it != features_cache.end() && features_it->first.find(ngram) == 0) {
				int sign = abs(features_it->second)/features_it->second;
				current_upos = upos + C * (alpha * sign + (1-alpha) * features_it->second);
				current_uneg = uneg + C * (alpha * sign + (1-alpha) * features_it->second);
				
				if (verbosity > 3) {
					cout << "\nexisting feature starting with current ngram rule prefix: "
					 << features_it->first << ", " << features_it->second << ",  sign: " << sign;
					
					cout << "\ncurrent_upos: " << current_upos;
					cout << "\ncurrent_uneg: " << current_uneg;
					cout << "\ntau: " << tau;
				}
				// Check bound. If any non-zero feature starting with current ngram as a prefix
				// can still qualify for selection in the model,
				// we cannot prune the search space.
				if (std::max (abs(current_upos), abs(current_uneg)) > tau ) {
					flag_can_prune = 0;
					break;
				}
				++features_it;
			}
		} else {
			// If there are no features in the model starting with this prefix, check regular bound.
			if (std::max (abs(upos), abs(uneg)) > tau ) {
				flag_can_prune = 0;
			}
		}
	} // If C = 0, check regular bound.
	else {
		if (std::max (abs(upos), abs(uneg)) > tau ) {
			flag_can_prune = 0;
		}
	}
	
    if (support < minsup || flag_can_prune) {
      ++pruned;
      if (verbosity > 3) {
      	cout << "\n\nminsup || upper bound: pruned!\n";
      }
      return true;
    }

    double g = std::abs (gradient);

	// If current ngram better than previous best ngram, update optimal ngram.
	 // Check min length requirement.
    if (g > tau && size >= minpat) {
    // || g == tau && size < rule.size) {

      ++rewritten;

      tau = g;
      rule.gradient = gradient;
      rule.size = size;

	if (C == 0) { // Retrieve the best ngram. If C != 0 this is already done.
		if (!token_type) { // If word-level token: a bb cd a bb
			// traverse ngram going from the end to the front		
			for (space_t *t = space; t; t = t->prev) {
				ngram = " " + t->ne + ngram;
			}
			// skip the space in front of the ngram
			ngram.assign(ngram.substr(1));

		} else { //char-level tokens: abbcdabb
			for (space_t *t = space; t; t = t->prev) {
				ngram = t->ne + ngram;
			}
		}
	} //end C==0. 
     	rule.ngram = ngram;

	if (verbosity >= 3) {
		cout << "\n\ncurrent best ngram rule: " << rule.ngram;
	  	cout << "\ngradient: " << gradient << "\n";
	}

      rule.loc.clear ();
      for (unsigned int i = 0; i < space->loc.size(); ++i) {
		// Keep the doc ids where the best ngram appears.
		if (space->loc[i] < 0) rule.loc.push_back ((unsigned int)(-space->loc[i]) - 1);
      }
    }
    //cout << "\nend can_prune...";
    //gettimeofday(&t, NULL);
    //cout << " ( " << t.tv_sec - t_start.tv_sec << " seconds; " << (t.tv_sec - t_start.tv_sec) / 60.0 << " minutes )";

    return false;
  }

// Try to grow the ngram to next level, and prune the appropriate extensions.
// The growth is done breadth-first, e.g. grow all unigrams to bi-grams, than all bi-grams to tri-grams, etc.
void span_bfs (space_t *space, std::vector<space_t *>& new_space, unsigned int size) {
	//struct timeval t_start;
  	//gettimeofday(&t_start, NULL);
  	//cout << "\nspan next level....";
  	//cout.flush();
	std::vector <space_t *>& next = space->next;

	// If working with gaps.
	// Check if number of consecutive gaps exceeds the max allowed.
	if (maxgap) {
		unsigned int num_consec_gaps = 0;
		for (space_t *t = space; t; t = t->prev) {
			if (t->ne.compare("*") == 0)
				num_consec_gaps ++;
			else break;
		}
		if (num_consec_gaps > maxgap) return;
	}

	if (next.size() == 1 && next[0] == 0) {
		return;
    	}
    	else {
		// If there are candidates for extension, iterate through them, and try to prune some.
		if (! next.empty()) {
			if (verbosity > 4)
				cout << "\n !next.empty()";
		    	for (std::vector<space_t*>::iterator it = next.begin(); it != next.end(); ++it) {
	    		// If the last token is a gap, skip checking gradient and pruning bound, since this is the same as for the prev ngram without the gap token.
	    		// E.g., if we checked the gradient and bounds for "a" and didnt prune it, then the gradient and bounds for "a*" will be the same,
	    		// so we can safely skip recomputing the gradient and bounds.
	    		if ((*it)->ne.compare("*") == 0) {
	    			if (verbosity > 4)
	    				cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next dfs level.";
	    			new_space.push_back ((*it));
	    		} else if (! can_prune (*it, size)) {
					new_space.push_back ((*it));
				}
	    	}
	    } else {
	    	//cout << "\nnext.empty";
	    	// Prepare possible extensions.
	    	unsigned int docid = 0;
	      	// Candidates obtained by extension.
	      	std::map<string, space_t> candidates;
	      	std::vector <int>& loc = space->loc;

	      	// Iterate through the inverted index of the current feature.
		for (unsigned int i = 0; i < loc.size(); ++i) {
			if (loc[i] < 0) {
				  docid = (unsigned int)(-loc[i]) - 1;
				  //cout << "\n\ndocid: " << docid;
				  continue;
			}

			// the start position of this unigram in the document
			unsigned int pos = loc[i];

			// current unigram where extension is done
			string unigram = space->ne;
			if (verbosity > 4) {	
				cout << "\ncurrent pos and start char: " <<  pos << " " << transaction[docid][pos];
				cout << "\ncurrent unigram to be extended (space->ne):" << unigram; 
			}
			string next_unigram;
			// If not re-initialized, it should fail.
			unsigned int pos_start_next_unigram = transaction[docid].size();

			//cout << "\npos: " << pos;
			if (pos + unigram.size() < transaction[docid].size()) { //unigram is not in the end of doc, thus it can be extended.
				if (verbosity > 4) {				
					cout << "\npotential extension...";
				}
				if (!token_type) { // Word level token.

					// Find the first space after current unigram position.
					unsigned int pos_space = pos + unigram.size();
					// Skip over consecutive spaces. 
					while ( (pos_space < transaction[docid].size()) && isspace(transaction[docid][pos_space]) ) {
						pos_space++;
					}
					// Check if doc ends in spaces, rather than a unigram.
					if (pos_space == transaction[docid].size()) {
						//cout <<"\ndocument with docid" << docid << " ends in (consecutive) space(s)...move to next doc";
						//std::exit(-1);
						continue;
					} else {
						pos_start_next_unigram = pos_space; //stoped at first non-space after consec spaces
						size_t pos_next_space = transaction[docid].find(' ', pos_start_next_unigram + 1);

						// Case1: the next unigram is in the end of the doc, no second space found.
						if (pos_next_space == string::npos) {
							next_unigram.assign(transaction[docid].substr(pos_start_next_unigram));
						} else { //Case2: the next unigram is inside the doc.
							next_unigram.assign(transaction[docid].substr(pos_start_next_unigram, pos_next_space - pos_start_next_unigram));
						}
					}
				} else { // Char level token. Skip spaces.
					if (!isspace(transaction[docid][pos + 1])) {
						//cout << "\nnext char is not space";
						next_unigram = transaction[docid][pos + 1]; //next unigram is next non-space char
						pos_start_next_unigram = pos + 1;
					} else { // If next char is space.
						unsigned int pos_space = pos + 1;
						// Skip consecutive spaces.
						while ((pos_space < transaction[docid].size()) && isspace(transaction[docid][pos_space])) {
							pos_space++;
						}
						// Check if doc ends in space, rather than a unigram.
						if (pos_space == transaction[docid].size()) {
							//cout <<"\ndocument with docid" << docid << " ends in (consecutive) space(s)...move to next doc";
							//std::exit(-1);
							continue;
						} 
						/* //disallow using char-tokenization for space separated tokens.
						else {
							pos_start_next_unigram = pos_space;
							//cout << "\nnext next char is not space";
							next_unigram = transaction[docid][pos_start_next_unigram];
						} */
					}
				} //end char level token

				if (next_unigram.empty()) {
					cout <<"\nFATAL...in span_bfs() next_unigram for extension of current unigram" << unigram << " is empty...exit";
					std::exit(-1);
				} else {
					if (verbosity > 4) {	
						cout << "\nnext_unigram for extension:" << next_unigram; 
						cout << "\npos: " <<  pos_start_next_unigram << " " << transaction[docid][pos_start_next_unigram];

					}
				}
				
				if (minsup == 1 || single_node_minsup_cache.find (next_unigram) != single_node_minsup_cache.end()) {
					//cout << "\nunigram: " << unigram <<"  extension: " << next_unigram;
					candidates[next_unigram].add (docid, pos_start_next_unigram);
				}

				if (maxgap) {
					// If we allow gaps, we treat a gap as an additional unigram "*".
					// Its inverted index will simply be a copy pf the inverted index of the original features.
					// Example, for original feature "a", we extend to "a *", where inverted index of "a *" is simply
					// a copy of the inverted index of "a", except for positions where "a" is the last char in the doc.
					candidates["*"].add (docid, pos_start_next_unigram);
				}
			} //end generating candidates for the current pos

		} //end iteration through inverted index (docids iand pos) to find candidates

		// Keep only doc_ids for occurrences of current ngram.
		space->shrink ();

		//cout << "\nfinal candidates: ";
		//cout.flush();
		// Prepare the candidate extensions.
		for (std::map <string, space_t >::iterator it = candidates.begin();
		     it != candidates.end(); ++it) {

			space_t* c = new space_t;
			c->loc = it->second.loc;
			std::vector<int>(c->loc).swap(c->loc);
			c->ne    = it->first;
			c->prev   = space;
			c->next.clear ();

			// Keep all the extensions of the current feature for future iterations.
			// If we need to save memory we could sacrifice this storage.
			next.push_back (c);

			//cout << "\nnode extended: " << space->ne;
			//cout << "\nnode for extention: " << it->first;

			// If the last token is a gap, skip checking gradient and pruning bound, since this is the same as for ngram without last gap token.
			// E.g., if we checked the gradient and bounds for "a" and didnt prune it, then the gradient and bounds for "a*" will be the same,
			// so we can safely skip recomputing the gradient and bounds.
			if (c->ne.compare("*") == 0) {
			    if (verbosity >= 3)
			    	cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next bfs level.";
			    new_space.push_back (c);
			} else if (! can_prune (c, size)) {
				new_space.push_back (c);
			}
		}

		if (next.empty()) {
			next.push_back (0);
		}
		std::vector<space_t *>(next).swap (next);
	    } //end generation of candidates when they weren't stored already.
    }
    //cout << "\nend span next level....";
    //gettimeofday(&t, NULL);
    //cout << " ( " << t.tv_sec - t_start.tv_sec << " seconds; " << (t.tv_sec - t_start.tv_sec) / 60.0 << " minutes )";

    return;
  }

// Try to grow the ngram to next level, and prune the appropriate extensions.
// The growth is done deapth-first rather than breadth-first, e.g. grow each candidate to its longest unpruned sequence

void span_dfs (space_t *space, unsigned int size) {
    //struct timeval t_start;
    //gettimeofday(&t_start, NULL);
    //cout << "\nspan next level....";
    //cout.flush();
    std::vector <space_t *>& next = space->next;

    // Check if ngram larger than maxsize allowed.
    if (size > maxpat) return;

    // If using gaps.
	// Check if number of consecutive gaps exceeds the max allowed.
    if (maxgap) {
		unsigned int num_consec_gaps = 0;
		for (space_t *t = space; t; t = t->prev) {
			if (t->ne.compare("*") == 0)
				num_consec_gaps ++;
			else break;
		}
		if (num_consec_gaps > maxgap) return;
    }

    if (next.size() == 1 && next[0] == 0) {
    	return;
    }
    else {
    	//{
    	// If the extensions are already computed, iterate through them and check pruning condition.
    	if (! next.empty()) {
        	if (verbosity >= 3)
        		cout << "\n !next.empty()";
    	    for (std::vector<space_t*>::iterator it = next.begin(); it != next.end(); ++it) {
    	    	if ((*it)->ne.compare("*") == 0) {
    	    		if (verbosity >= 3)
    	    	    	cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next dfs level.";
    	    		// Expand each candidate DFS wise.
    	    		span_dfs(*it, size + 1);
    	    	} else if (! can_prune (*it, size)) {
    				// Expand each candidate DFS wise.
    				span_dfs(*it, size + 1);
    			}
    	    }
    	} else {
    		// Extend the ngram to its next level.
    	    //cout << "\nnext.empty";
    	    // Prepare possible extensions.
    	    unsigned int docid = 0;
    	    // Candidates obtained by extension.
    	    std::map<string, space_t> candidates;
    	    std::vector <int>& loc = space->loc;

    	    // Iterate through the inverted index of the current feature.
		for (unsigned int i = 0; i < loc.size(); ++i) {
			if (loc[i] < 0) {
				  docid = (unsigned int)(-loc[i]) - 1;
				  //cout << "\n\ndocid: " << docid;
				  continue;
			}

			// the start position of this unigram in the document
			unsigned int pos = loc[i];

			// current unigram where extension is done
			string unigram = space->ne;
			if (verbosity > 4) {	
				cout << "\ncurrent pos and start char: " <<  pos << " " << transaction[docid][pos];
				cout << "\ncurrent unigram to be extended (space->ne):" << unigram; 
			}
			string next_unigram;
			// If not re-initialized, it should fail.
			unsigned int pos_start_next_unigram = transaction[docid].size();

			//cout << "\npos: " << pos;
			if (pos + unigram.size() < transaction[docid].size()) { //unigram is not in the end of doc, thus it can be extended.
				if (verbosity > 4) {				
					cout << "\npotential extension...";
				}
				if (!token_type) { // Word level token.

					// Find the first space after current unigram position.
					unsigned int pos_space = pos + unigram.size();
					// Skip over consecutive spaces. 
					while ( (pos_space < transaction[docid].size()) && isspace(transaction[docid][pos_space]) ) {
						pos_space++;
					}
					// Check if doc ends in spaces, rather than a unigram.
					if (pos_space == transaction[docid].size()) {
						//cout <<"\ndocument with docid" << docid << " ends in (consecutive) space(s)...move to next doc";
						//std::exit(-1);
						continue;
					} else {
						pos_start_next_unigram = pos_space; //stoped at first non-space after consec spaces
						size_t pos_next_space = transaction[docid].find(' ', pos_start_next_unigram + 1);

						// Case1: the next unigram is in the end of the doc, no second space found.
						if (pos_next_space == string::npos) {
							next_unigram.assign(transaction[docid].substr(pos_start_next_unigram));
						} else { //Case2: the next unigram is inside the doc.
							next_unigram.assign(transaction[docid].substr(pos_start_next_unigram, pos_next_space - pos_start_next_unigram));
						}
					}
				} else { // Char level token. Skip spaces.
					if (!isspace(transaction[docid][pos + 1])) {
						//cout << "\nnext char is not space";
						next_unigram = transaction[docid][pos + 1]; //next unigram is next non-space char
						pos_start_next_unigram = pos + 1;
					} else { // If next char is space.
						unsigned int pos_space = pos + 1;
						// Skip consecutive spaces.
						while ((pos_space < transaction[docid].size()) && isspace(transaction[docid][pos_space])) {
							pos_space++;
						}
						// Check if doc ends in space, rather than a unigram.
						if (pos_space == transaction[docid].size()) {
							//cout <<"\ndocument with docid" << docid << " ends in (consecutive) space(s)...move to next doc";
							//std::exit(-1);
							continue;
						} 
						/* //disallow using char-tokenization for space separated tokens.
						else {
							pos_start_next_unigram = pos_space;
							//cout << "\nnext next char is not space";
							next_unigram = transaction[docid][pos_start_next_unigram];
						} */
					}
				} //end char level token

				if (next_unigram.empty()) {
					cout <<"\nFATAL...in span_dfs() next_unigram for extension of current unigram" << unigram << " is empty...exit";
					std::exit(-1);
				} else {
					if (verbosity > 4) {	
						cout << "\nnext_unigram for extension:" << next_unigram; 
						cout << "\npos: " <<  pos_start_next_unigram << " " << transaction[docid][pos_start_next_unigram];

					}
				}
				
				if (minsup == 1 || single_node_minsup_cache.find (next_unigram) != single_node_minsup_cache.end()) {
					//cout << "\nunigram: " << unigram <<"  extension: " << next_unigram;
					candidates[next_unigram].add (docid, pos_start_next_unigram);
				}

				if (maxgap) {
					// If we allow gaps, we treat a gap as an additional unigram "*".
					// Its inverted index will simply be a copy pf the inverted index of the original features.
					// Example, for original feature "a", we extend to "a *", where inverted index of "a *" is simply
					// a copy of the inverted index of "a", except for positions where "a" is the last char in the doc.
					candidates["*"].add (docid, pos_start_next_unigram);
				}
			} //end generating candidates for the current pos

		} //end iteration through inverted index (docids iand pos) to find candidates

    		// Keep only doc_ids for occurrences of current ngram.
    	    space->shrink ();

    	    // If no candidate extensions were found, return.
//    	    if (candidates.empty()) {
//    	    	if (verbosity >= 3)
//    	    		cout << "\n candidates is empty()!... return";
//    	    	return;
//    	    }

    		//cout << "\nfinal candidates: ";
    		//cout.flush();
    		// Prepare the candidate extensions.
    		for (std::map <string, space_t >::iterator it = candidates.begin();
    			     it != candidates.end(); ++it) {

    			space_t* c = new space_t;
    			c->loc = it->second.loc;
    			std::vector<int>(c->loc).swap(c->loc);
    			c->ne    = it->first;
    			c->prev   = space;
    			c->next.clear ();

    			// Keep all the extensions of the current feature for future iterations.
    			// If we need to save memory we could sacrifice this storage.
    			next.push_back (c);

				//cout << "\nnode extended: " << space->ne;
				//cout << "\nnode for extention: " << it->first;

    			// If the last token is a gap, skip checking gradient and pruning bound, since this is the same as for ngram without last gap token.
    			// E.g., if we checked the gradient and bounds for "a" and didnt prune it, then the gradient and bounds for "a*" will be the same,
    			// so we can safely skip recomputing the gradient and bounds.
    			if (c->ne.compare("*") == 0) {
    				if (verbosity >= 3)
    					cout << "\nFeature ends in *, skipping gradient and bound computation. Extending to next dfs level.";
    				span_dfs(c, size + 1);
    			} else // If this doesn't end in *, then check gradient and pruning condition.
    				if (! can_prune (c, size)) {
					span_dfs(c, size + 1);
				}
			}

		    if (next.empty()) {
		    	next.push_back (0);
		    }
		    std::vector<space_t *>(next).swap (next);
		}
    }
    //cout << "\nend span next level....";
    //gettimeofday(&t, NULL);
    //cout << " ( " << t.tv_sec - t_start.tv_sec << " seconds; " << (t.tv_sec - t_start.tv_sec) / 60.0 << " minutes )";

    return;
}

// Line search method. Search for step size that minimizes loss.
// Compute loss in middle point of range, beta_n1, and
// for mid of both ranges beta_n0, beta_n1 and bet_n1, beta_n2
// Compare the loss for the 3 points, and choose range of 3 points
// which contains the minimum. Repeat until the range spanned by the 3 points is small enough,
// e.g. the range approximates well the vector where the loss function is minimized.
// Return the middle point of the best range.
void find_best_range(vector<double>& sum_best_beta_n0, vector<double>& sum_best_beta_n1, vector<double>& sum_best_beta_n2,
		vector<double>& sum_best_beta_mid_n0_n1, vector<double>& sum_best_beta_mid_n1_n2,
		rule_t& rule, vector<double>* sum_best_beta_opt) {

    	//struct timeval t, t_start;
    	//gettimeofday(&t_start, NULL);
	//	if (verbosity > 3)
	//		cout << "\n In find_best_range()!";
    	//cout.flush();

    	double min_range_size = 1e-3;
    	double current_range_size = 0;
    	int current_interpolation_iter = 0;

	long double loss_mid_n0_n1 = 0;
	long double loss_mid_n1_n2 = 0;
	long double loss_n1 = 0;

    	for (unsigned int i = 0; i < transaction.size();  ++i) {
    		if (verbosity > 4) {
			cout << "\nsum_best_beta_n0[i]: " << sum_best_beta_n0[i];
    			cout << "\nsum_best_beta_n1[i]: " << sum_best_beta_n1[i];
    			cout << "\nsum_best_beta_n2[i]: " << sum_best_beta_n2[i];
    		}
    		current_range_size += abs(sum_best_beta_n2[i] - sum_best_beta_n0[i]);
    	}
    	if (verbosity > 3)
    		cout << "\ncurrent range size: " << current_range_size;

    	double beta_coef_n1 = 0;
    	double beta_coef_mid_n0_n1 = 0;
    	double beta_coef_mid_n1_n2 = 0;
	
    	if (C != 0 && sum_squared_betas != 0) {
	    	features_it = features_cache.find(rule.ngram);
	}
	// Start interpolation loop.
	while (current_range_size > min_range_size) {
    		if (verbosity > 3)
    			cout << "\ncurrent interpolation iteration: " << current_interpolation_iter;

    		for (unsigned int i = 0; i < transaction.size();  ++i) { //loop through training samples
    			sum_best_beta_mid_n0_n1[i] = (sum_best_beta_n0[i] + sum_best_beta_n1[i]) / 2;
    			sum_best_beta_mid_n1_n2[i] = (sum_best_beta_n1[i] + sum_best_beta_n2[i]) / 2;

			if ( C != 0) {    			
				beta_coef_n1 = sum_best_beta_n1[rule.loc[0]] - sum_best_beta[rule.loc[0]];
	    			beta_coef_mid_n0_n1 = sum_best_beta_mid_n0_n1[rule.loc[0]] - sum_best_beta[rule.loc[0]];
	    			beta_coef_mid_n1_n2 = sum_best_beta_mid_n1_n2[rule.loc[0]] - sum_best_beta[rule.loc[0]];
			}

        		if (verbosity > 4) {
        			cout << "\nsum_best_beta_mid_n0_n1[i]: " << sum_best_beta_mid_n0_n1[i];
					cout << "\nsum_best_beta_mid_n1_n2[i]: " << sum_best_beta_mid_n1_n2[i];
        		}

			if (objective == 0 ) { //SLR
				if (-y[i] * sum_best_beta_n1[i] > 8000) {
					loss_n1 += log(LDBL_MAX);
				} else {
					loss_n1 += log(1 + exp(-y[i] * sum_best_beta_n1[i]));
				}

				if (-y[i] * sum_best_beta_mid_n0_n1[i] > 8000) {
					loss_mid_n0_n1 += log(LDBL_MAX);
				} else {
					loss_mid_n0_n1 += log(1 + exp(-y[i] * sum_best_beta_mid_n0_n1[i]));
				}

				if (-y[i] * sum_best_beta_mid_n1_n2[i] > 8000) {
					loss_mid_n1_n2 += log(LDBL_MAX);
				} else {
					loss_mid_n1_n2 += log(1 + exp(-y[i] * sum_best_beta_mid_n1_n2[i]));
				}
			} //end SLR

			/*
			if (objective == 1) { //L1-SVM
				if (1 - y[i] * sum_best_beta_n1[i] > 0)
    					loss_n1 += 1 - y[i] * sum_best_beta_n1[i];

				if (1 - y[i] * sum_best_beta_mid_n0_n1[i] > 0)
					loss_mid_n0_n1 += 1 - y[i] * sum_best_beta_mid_n0_n1[i];

				if (1 - y[i] * sum_best_beta_mid_n1_n2[i] > 0)
					loss_mid_n1_n2 += 1 - y[i] * sum_best_beta_mid_n1_n2[i];			
			} //end l1-SVM
			*/

			if (objective == 2) { //l2-SVM
				if (1 - y[i] * sum_best_beta_n1[i] > 0)
    					loss_n1 += pow(1 - y[i] * sum_best_beta_n1[i], 2);

				if (1 - y[i] * sum_best_beta_mid_n0_n1[i] > 0)
					loss_mid_n0_n1 += pow(1 - y[i] * sum_best_beta_mid_n0_n1[i], 2);

				if (1 - y[i] * sum_best_beta_mid_n1_n2[i] > 0)
					loss_mid_n1_n2 += pow(1 - y[i] * sum_best_beta_mid_n1_n2[i], 2);			
			} //end l2-SVM

   		} //end loop through training samples.

		if ( C != 0 ) {
	    		// Add the Elastic Net regularization term.
	    		// If this is the first ngram selected.
	    		if (sum_squared_betas == 0) {
	    			loss_n1 = loss_n1 + C * (alpha * abs(beta_coef_n1) + (1-alpha) * 0.5 * pow(beta_coef_n1, 2));
	    			loss_mid_n0_n1 = loss_mid_n0_n1 + C * (alpha * abs(beta_coef_mid_n0_n1) + (1-alpha) * 0.5 * pow(beta_coef_mid_n0_n1, 2));
	    			loss_mid_n1_n2 = loss_mid_n1_n2 + C * (alpha * abs(beta_coef_mid_n1_n2) + (1-alpha) * 0.5 * pow(beta_coef_mid_n1_n2, 2));
	    		} else {
				// If this feature was not selected before.
	    			if (features_it == features_cache.end()) {
	    				loss_n1 = loss_n1 + C * (alpha * (sum_abs_betas + abs(beta_coef_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coef_n1, 2)));
	    				loss_mid_n0_n1 = loss_mid_n0_n1 + C * (alpha * (sum_abs_betas + abs(beta_coef_mid_n0_n1)) + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coef_mid_n0_n1, 2)));
	    				loss_mid_n1_n2 = loss_mid_n1_n2 + C * (alpha * (sum_abs_betas + abs(beta_coef_mid_n1_n2)) + (1 - alpha) * 0.5 * (sum_squared_betas + pow(beta_coef_mid_n1_n2, 2)));
	    			} else {
	    				double new_beta_coef_n1 = features_it->second + beta_coef_n1;
	    				double new_beta_coef_mid_n0_n1 = features_it->second + beta_coef_mid_n0_n1;
	    				double new_beta_coef_mid_n1_n2 = features_it->second + beta_coef_mid_n1_n2;
	    				loss_n1 = loss_n1  + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coef_n1)) + (1 - alpha) * 0.5 * 									(sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coef_n1, 2)));
	    				loss_mid_n0_n1 = loss_mid_n0_n1  + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coef_mid_n0_n1)) + (1 - alpha) 								* 0.5 * (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coef_mid_n0_n1, 2)));
	    				loss_mid_n1_n2 = loss_mid_n1_n2  + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coef_mid_n1_n2)) + (1 - alpha) 								* 0.5 * (sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coef_mid_n1_n2, 2)));
	    				}
	    			}
		}// end check C != 0.

    		// Focus on the range that contains the minimum of the loss function.
    		// Compare the 3 points beta_n, and mid_beta_n-1_n and mid_beta_n_n+1.
    		if (loss_n1 <= loss_mid_n0_n1 && loss_n1 <= loss_mid_n1_n2) {
    			// Min is in beta_n1.
	    	    	if (verbosity > 4) {
				cout << "\nmin is sum_best_beta_n1";
				cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
				cout << "\nloss_n1: " << loss_n1;
				cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
			}
	    		// Make the beta_n0 be the beta_mid_n0_n1.
	    		sum_best_beta_n0.assign(sum_best_beta_mid_n0_n1.begin(), sum_best_beta_mid_n0_n1.end());
	    		// Make the beta_n2 be the beta_mid_n1_n2.
	    		sum_best_beta_n2.assign(sum_best_beta_mid_n1_n2.begin(), sum_best_beta_mid_n1_n2.end());
    		}
    		else {
			if (loss_mid_n0_n1 <= loss_n1 && loss_mid_n0_n1 <= loss_mid_n1_n2) {
				// Min is beta_mid_n0_n1.
    		    		if (verbosity > 4) {
    		    			cout << "\nmin is sum_best_beta_mid_n0_n1";
    		    			cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
					cout << "\nloss_n1: " << loss_n1;
					cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
    		    		}
    				// Make the beta_n2 be the beta_n1.
    				sum_best_beta_n2.assign(sum_best_beta_n1.begin(), sum_best_beta_n1.end());
    				// Make the beta_n1 be the beta_mid_n0_n1.
    				sum_best_beta_n1.assign(sum_best_beta_mid_n0_n1.begin(), sum_best_beta_mid_n0_n1.end());
    			} else {
    				// Min is beta_mid_n1_n2.
	    		    	if (verbosity > 4) {
	    		    		cout << "\nmin is sum_best_beta_mid_n1_n2";
	    		    		cout << "\nloss_mid_n0_n1: " << loss_mid_n0_n1;
	    		    		cout << "\nloss_n1: " << loss_n1;
	    		    		cout << "\nloss_mid_n1_n2: " << loss_mid_n1_n2;
	    		    	}
    				// Make the beta_n0 be the beta_n1.
    				sum_best_beta_n0.assign(sum_best_beta_n1.begin(), sum_best_beta_n1.end());
    				// Make the beta_n1 be the beta_mid_n1_n2
    				sum_best_beta_n1.assign(sum_best_beta_mid_n1_n2.begin(), sum_best_beta_mid_n1_n2.end());
    				}
    		}

	    	++current_interpolation_iter;
	    	loss_mid_n0_n1 = 0;
	    	loss_mid_n1_n2 = 0;
	    	loss_n1 = 0;
	    	current_range_size = 0;

	    	for (unsigned int i = 0; i < transaction.size();  ++i) {
	       		if (verbosity > 4) {
	   			cout << "\nsum_best_beta_n0[i]: " << sum_best_beta_n0[i];
	    			cout << "\nsum_best_beta_n1[i]: " << sum_best_beta_n1[i];
	    			cout << "\nsum_best_beta_n2[i]: " << sum_best_beta_n2[i];
	    		}
			current_range_size += abs(sum_best_beta_n2[i] - sum_best_beta_n0[i]);
		}
		if (verbosity > 4) {
			cout << "\ncurrent range size: " << current_range_size;
		}	
	} // end while loop.	

	// Keep the middle point of the best range.
	for (unsigned int i = 0; i < transaction.size();  ++i) {
		sum_best_beta_opt->push_back(sum_best_beta_n1[i]);
		// Trust this step only by a fraction.
		//sum_best_beta_opt->push_back(0.5 * sum_best_beta[i] + 0.5 * sum_best_beta_n1[i]);
	}
	//cout << "\n end find_best_range()!";
	//gettimeofday(&t, NULL);
	//cout << " ( " << t.tv_sec - t_start.tv_sec << " seconds; " << (t.tv_sec - t_start.tv_sec) / 60.0 << " minutes )";
} // end find_best_range().

// Line search method. Binary search for optimal step size. Calls find_best_range(...).
// sum_best_beta keeps track of the scalar product beta_best^t*xi for each doc xi.
// Instead of working with the new weight vector beta_n+1 obtained as beta_n - epsilon * gradient(beta_n)
// we work directly with the scalar product.
// We output the sum_best_beta_opt which contains the scalar poduct of the optimal beta found, by searching for the optimal
// epsilon, e.g. beta_n+1 = beta_n - epsilon_opt * gradient(beta_n)
// epsilon is the starting value
// rule contains info about the gradient at the current iteration
void binary_line_search(rule_t& rule, vector<double>* sum_best_beta_opt) {

	//struct timeval t, t_start;
	//gettimeofday(&t_start, NULL);
	//if (verbosity > 3)
	//	cout << "\nbinary_line_search()!";
	//cout.flush();

	// Starting value for parameter in step size search.
	// Set the initial epsilon value small enough to guaranteee
     	// log-lik increases in the first steps.
	double exponent = ceil(log10(abs(rule.gradient)));
	double epsilon = min(1e-3, pow(10, -exponent));

    	if (verbosity > 3) {
		cout << "\nrule.ngram: " << rule.ngram;
		cout << "\nrule.gradient: " << rule.gradient;
		cout << "\nexponent of epsilon: " << -exponent;
		cout << "\nepsilon: " << epsilon;
    	}

    	// Keep track of scalar product at points beta_n-1, beta_n and beta_n+1.
    	// They are denoted with beta_n0, beta_n1, beta_n2.
    	vector<double> sum_best_beta_n0(sum_best_beta.size());
    	vector<double> sum_best_beta_n1(sum_best_beta);
    	vector<double> sum_best_beta_n2(sum_best_beta);

    	// Keep track of loss at the three points, n0, n1, n2.
    	long double loss_n0 = 0;
    	long double loss_n1 = 0;
    	long double loss_n2 = 0;

    	// Binary search for epsilon. Similar to bracketing phase in which
    	// we search for some range with promising epsilon.
    	// The second stage finds the epsilon or corresponding weight vector with smallest l2-loss value.

        // **************************************************************************/
    	// As long as the l2-loss decreases, double the epsilon.
    	// Keep track of the last three values of beta, or correspondingly
    	// the last 3 values for the scalar product of beta and xi.
    	int n = 0;
    	
    	if ( C != 0 && sum_squared_betas != 0) {
    	    	features_it = features_cache.find(rule.ngram);
	}

  	double beta_coeficient_update = 0;
    	do {
        	if (verbosity > 3)
        		cout << "\nn: " << n;

		// For each location (e.g. docid), update the score of the documents containing best rule.
    	      	// E.g. update beta^t * xi.
    	      	beta_coeficient_update -= pow(2, n * 1.0) * epsilon * rule.gradient;
    	      	for (unsigned int i = 0; i < rule.loc.size(); ++i) {
    	      		sum_best_beta_n0[rule.loc[i]] = sum_best_beta_n1[rule.loc[i]];
    	      		sum_best_beta_n1[rule.loc[i]] = sum_best_beta_n2[rule.loc[i]];
    	      		sum_best_beta_n2[rule.loc[i]] = sum_best_beta_n1[rule.loc[i]] - pow(2, n * 1.0) * epsilon * rule.gradient;

    	      		if (verbosity > 4 && i == 0) {
    	      			cout << "\nsum_best_beta_n0[rule.loc[i]: " << sum_best_beta_n0[rule.loc[i]];
    	      			cout << "\nsum_best_beta_n1[rule.loc[i]: " << sum_best_beta_n1[rule.loc[i]];
    	      			cout << "\nsum_best_beta_n2[rule.loc[i]: " << sum_best_beta_n2[rule.loc[i]];
    	      		}
    	      	}

    	      	// Compute loss for all 3 values: n-1, n, n+1
    		// In the first iteration compute necessary loss.
    		if (n == 0) {
    			loss_n0 = loss;
    			loss_n1 = loss;    			
    		} else {
			// Update just loss_n2.
			// The loss_n0 and loss_n1 are already computed.
			loss_n0 = loss_n1;
			loss_n1 = loss_n2;
		}
		loss_n2 = 0;
		for (unsigned int i = 0; i < transaction.size();  ++i) { //loop through docs
			if (objective == 0) { // SLR
				// Compute loss_n2.
				if (-y[i] * sum_best_beta_n2[i] > 8000) {
	    				loss_n2 += log(LDBL_MAX);
	    			} else {
	    				 loss_n2 += log(1 + exp(-y[i] * sum_best_beta_n2[i]));
	    			}
			} //end SLR

			/*
			if (objective == 1) { //L1-SVM loss
				// Compute loss_n2.
    				 if (1 - y[i] * sum_best_beta_n2[i] > 0)
    					 loss_n2 += 1 - y[i] * sum_best_beta_n2[i];				
			} //end L1-SVM
			*/

			if (objective == 2) { //L2-SVM loss
				// Compute loss_n2.
    				 if (1 - y[i] * sum_best_beta_n2[i] > 0)
    					 loss_n2 += pow(1 - y[i] * sum_best_beta_n2[i], 2);				
			} //end L2-SVM	

    		} //end loop through docs

    		//cout << "\nloss_n2: " << loss_n2;
		if (verbosity > 4) {
			cout << "\nloss_n2 before adding regularizer: " << loss_n2;
        	}
    		// If C != 0, add the L2 regularizer term to the l2-loss.
    		// If this is the first ngram selected.
		if ( C != 0 ) {    			
			if (sum_squared_betas == 0) {
    				loss_n2 = loss_n2 + C * (alpha * abs(beta_coeficient_update) + (1 - alpha) * 0.5 * pow(beta_coeficient_update, 2));
				
				if (verbosity > 4) {				
					cout << "\nregularizer: " << C * (alpha * abs(beta_coeficient_update) + (1 - alpha) * 0.5 * pow(beta_coeficient_update, 2));
				}
			} else {
    			 	// If this feature was not selected before.
    				if (features_it == features_cache.end()) {
    					loss_n2 = loss_n2 + C * (alpha * (sum_abs_betas + abs(beta_coeficient_update)) + (1 - alpha) * 0.5 * (sum_squared_betas + 
								pow(beta_coeficient_update, 2)));
    				} else {
    					double new_beta_coeficient = features_it->second + beta_coeficient_update;
    					loss_n2 = loss_n2 + C * (alpha * (sum_abs_betas - abs(features_it->second) + abs(new_beta_coeficient)) + (1 - alpha) * 0.5 * 									(sum_squared_betas - pow(features_it->second, 2) + pow(new_beta_coeficient, 2)));
    				}
    			}
		} // end C != 0.        	
		if (verbosity > 4) {
			cout << "\nloss_n0: " << loss_n0;
			cout << "\nloss_n1: " << loss_n1;
			cout << "\nloss_n2: " << loss_n2;
        	}
		++n;
    	} while (loss_n2 < loss_n1);
    	// **************************************************************************/

    	if (verbosity > 3)
    		cout << "\nFinished doubling epsilon! The monotonicity loss_n+1 < loss_n is broken!";

    	// Search for the beta in the range beta_n-1, beta_mid_n-1_n, beta_n, beta_mid_n_n+1, beta_n+1
    	// that minimizes the objective function. It suffices to compare the 3 points beta_mid_n-1_n, beta_n, beta_mid_n_n+1,
    	// as the min cannot be achieved at the extrem points of the range.
    	// Take the 3 point range containing the point that achieves minimum loss.
    	// Repeat until the 3 point range is too small, or a fixed number of iterations is achieved.

    	// **************************************************************************/
    	vector<double> sum_best_beta_mid_n0_n1(sum_best_beta.size());
    	vector<double> sum_best_beta_mid_n1_n2(sum_best_beta.size());

    	find_best_range(sum_best_beta_n0, sum_best_beta_n1, sum_best_beta_n2,
    			sum_best_beta_mid_n0_n1, sum_best_beta_mid_n1_n2,
    			rule, sum_best_beta_opt);
	// **************************************************************************/
	//cout << "\nend binary_line_search()!";
	//gettimeofday(&t, NULL);
	//cout << " ( " << t.tv_sec - t_start.tv_sec << " seconds; " << (t.tv_sec - t_start.tv_sec) / 60.0 << " minutes )";
} // end binary_line)search().

public:

bool run (const char *in,
	const char *out,
	unsigned int _objective,
	unsigned int _maxpat,
	unsigned int _minpat,
	unsigned int maxitr,
	unsigned int _minsup,
	unsigned int _maxgap,
	bool _token_type,
	bool _traversal_strategy,
	double _convergence_threshold,
	double _regularizer_value,
	double _l1vsl2_regularizer,
	int _verbosity) {

	objective = _objective;
	maxpat = _maxpat;
	minpat = _minpat;
	minsup = _minsup;
	maxgap = _maxgap;
	token_type = _token_type;
	traversal_strategy = _traversal_strategy;
	convergence_threshold = _convergence_threshold;
	C = _regularizer_value;
	alpha = _l1vsl2_regularizer;
	verbosity = _verbosity;

	gettimeofday(&t_origin, NULL);
	//cout << "prepare input data...";

	if (! read (in)) {
	      std::cerr << "FATAL: Cannot open input file: " << in << std::endl;
	      return false;
	}
	gettimeofday(&t, NULL);
	cout << "( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )";
	cout.flush();

	std::ofstream os (out);
	if (! os) {
		std::cerr << "FATAL: Cannot open output file: " << out << std::endl;
	      return false;
	}

	std::cout.setf(std::ios::fixed,std::ios::floatfield);
	std::cout.precision(8);

	os.setf(std::ios::fixed,std::ios::floatfield);
	os.precision(12);

	unsigned int l   = transaction.size();
	tau              = 0.0;
	// All beta coeficients are zero when starting.
	sum_squared_betas = 0;
	sum_abs_betas = 0;

	sum_best_beta.resize(l);
	// The starting point is beta = (0, 0, 0, 0, .....).
	std::fill(sum_best_beta.begin(), sum_best_beta.end(), 0.0);
	exp_fraction.resize (l);
	std::fill (exp_fraction.begin(), exp_fraction.end(), 1.0 /2.0);

	rule.ngram = "";
	rule.loc.clear ();
	rule.size = 0;

	// A map from unigrams to search_space.
    	std::map <string, space_t> seed;
    	string unigram;
    	bool at_space = false;

	// Prepare the locations for unigrams.
	if (verbosity >= 1) {
		cout << "\nprepare inverted index for unigrams";
	}

	for (unsigned int docid = 0; docid < l; ++docid) {
		at_space = false;
	    	//cout << "\nscanning docid: " << docid << ", class y: " << y[docid] << "\n";
	    	for (unsigned int pos = 0; pos < transaction[docid].size(); ++pos) {
	    		// Skip white spaces. They are not considered as unigrams.
	    		if (isspace(transaction[docid][pos])) {
	    			at_space = true;
	    			continue;
	    		}
	    		// If word level tokens.
	    		if (!token_type) {
		    		if (at_space) {
		    			at_space = false;
		    			if (!unigram.empty()) {
		    				space_t & tmp = seed[unigram];
						tmp.add (docid,pos - unigram.size() - 1);
						tmp.next.clear ();
						tmp.ne = unigram;
						tmp.prev = 0;
//						if (unigram.size() >= 1)
    						//cout << "\ndocid: " << docid << ", pos: " << pos - unigram.size() - 1; 
						//cout << "\nunigram:#" << unigram << "#";
						unigram.clear();
		    			}
		    			unigram.push_back(transaction[docid][pos]);
		    		} else {
		    			unigram.push_back(transaction[docid][pos]);
		    		}
	    		} else {
				if (at_space) { 
					//Previous char was a space. 
					//Disallow using char-tokenization for space-separated tokens. It confuses the features, whether to add " " or not.
					cout << "\nFATAL...found space in docid: " << docid << ", at position: " << pos-1;
					cout << "\nFATAL...char-tokenization assumes contiguous tokens (i.e., tokens are not separated by space).";
					cout << "\nFor space separated tokens please use word-tokenization or remove spaces to get valid input for char-tokenization.";
					cout << "\n...Exiting.....\n";
					std::exit(-1);
				}
	    			// Char (i.e. byte) level token.
	    			unigram = transaction[docid][pos];
	    			space_t & tmp = seed[unigram];
				tmp.add (docid,pos);
				tmp.next.clear ();
				tmp.ne = unigram;
				tmp.prev = 0;
	//			if (unigram.size() >= 1)
				//cout << "\ndocid: " << docid << ", pos: " << pos; 				
				//cout << "\nunigram:#" << unigram << "#";				
				unigram.clear();
		   	}
		} //end for transaction.
		// For word-tokens take care of last word of doc.
		if (!token_type) {
		    	if (!unigram.empty()) {
		    		space_t & tmp = seed[unigram];
				tmp.add (docid, transaction[docid].size() - unigram.size());
				tmp.next.clear ();
				tmp.ne = unigram;
				tmp.prev = 0;
	//			if (unigram.size() >= 1)
	//    			cout << "\nunigram:" << unigram << "*";
				//cout << "\ndocid: " << docid << ", pos: " << transaction[docid].size() - unigram.size(); 
				//cout << "\nunigram:#" << unigram << "#";
				unigram.clear();
			}
		} 
	} //end for docid.

	gettimeofday(&t, NULL);
	if (verbosity >= 1) {
		cout << " ( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )";
		cout.flush();
	}
	// Keep only unigrams above minsup threshold.
	for (std::map <string, space_t>::iterator it = seed.begin (); it != seed.end(); ++it) {
		if (it->second.support() < minsup) {
	       		seed.erase (it++);
       		} else {
	       		single_node_minsup_cache.insert (it->second.ne);
			if (verbosity >= 1) {       	
				cout << "\ndistinct unigram:" << it->first;
			}
       		}
	}

	gettimeofday(&t, NULL);
	if (verbosity >= 1) {
	    	cout << "\n# distinct unigrams: " << single_node_minsup_cache.size();
		cout << " ( " << (t.tv_sec - t_origin.tv_sec) << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes )";
		cout.flush();
	}
	std::vector <space_t*> old_space;
	std::vector <space_t*> new_space;

	// The optimal step length.
	double step_length_opt;
	// Set the convergence threshold as in paper by Madigan et al on BBR.
	//double convergence_threshold = 0.005;
	double convergence_rate;
	//struct timeval t_start;

 	double sum_abs_scalar_prod_diff;
   	double sum_abs_scalar_prod;
 	loss = 0.0;
 	// Compute loss with start beta vector.
 	for (unsigned int i = 0; i < transaction.size();  ++i) {
		//cout << "\n y[i]: " << y[i];
		//cout << "\nsum_best_beta[i]: " << sum_best_beta[i];

		if (objective == 0) { //SLR
	 		if (-y[i] * sum_best_beta[i] > 8000) {
	 			loss += log(LDBL_MAX);
	 		} else {
	 			loss += log(1 + exp(-y[i] * sum_best_beta[i]));
	 		}
		} //end SLR
		/*
		if (objective == 1 ) { //L1-SVM (Hinge loss)
			if (1 - y[i] * sum_best_beta[i] > 0)
	 			loss += 1 - y[i] * sum_best_beta[i];

		} //end L1-SVM
		*/

		if (objective == 2 ) { //L2-SVM loss
			if (1 - y[i] * sum_best_beta[i] > 0)
	 			loss += pow(1 - y[i] * sum_best_beta[i], 2);

		} //end L2-SVM
		//cout << "\nstart loss: " << loss;
 	}

	if (verbosity >= 1) {
		cout << "\nstart loss: " << loss;
	}

	//cout << "\nstart iterations... ";
	// Loop for number of given optimization iterations.
	for (unsigned int itr = 0; itr < maxitr; ++itr) {
	    	gettimeofday(&t_start_iter, NULL);
    		pruned = total = rewritten = 0;
      		old_space.clear ();
		new_space.clear ();

		// Iterate through unigrams.
		for (std::map <string, space_t>::iterator it = seed.begin (); it != seed.end(); ++it) {
			if (!can_prune (&(it->second), 1)) {
				// Check BFS vs DFS traversal.				
				if (!traversal_strategy) {
					old_space.push_back (&(it->second));
				  } else {
					  // Traversal is DFS.
					  span_dfs (&(it->second), 2);
				  }
			  }
		}

		// If BFS traversal.
		if (!traversal_strategy) {
			// Search for best n-gram. Try to extend in a bfs fashion,
			// level per level, e.g., first extend unigrams to bigrams, then bigrams to trigrams, etc.
			//*****************************************************/
			for (unsigned int size = 2; size <= maxpat; ++size) {
				for (unsigned int i = 0; i < old_space.size(); ++i) {
					span_bfs (old_space[i], new_space, size);
				  }
				if (new_space.empty()) {
					break;
				}
				old_space = new_space;
				new_space.clear ();
			} // End search for best n-gram.
		} // end check BFS traversal.

		// Keep best ngram rule.

		//rule_cache.insert (rule);
		if (verbosity >= 2) {
			cout << "\nfound best ngram! ";
			cout << "\nrule.gradient: " << rule.gradient;
			gettimeofday(&t, NULL);
			cout << " (per iter: " << t.tv_sec - t_start_iter.tv_sec << " seconds; " << (t.tv_sec - t_start_iter.tv_sec) / 60.0 << " minutes; total time:" 
				<< (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes)";
		}

		// Check convergence based on gradient magnitude. If this is small enough, we are close enough to optimal solution.
		/*		
		if (abs(rule.gradient) <= 1e-8) {
			if (verbosity >= 1) {
		    		cout << "\n\nFinish iterations due to grad ~ 0 (close enough to optimum)!";
		    		cout << "\n# iterations: " << itr + 1;
		    	}
			break;
		}*/
	      // Use line search to detect the best step_length/learning_rate.
	      // The method does binary search for the step length, by using a start parameter epsilon.
	      // It works by first bracketing a promising value, followed by focusing on the smallest interval containing this value.
	      sum_best_beta_opt.clear();
	      binary_line_search(rule, &sum_best_beta_opt);

	      // The optimal step length as obtained from the line search.
	      step_length_opt = sum_best_beta_opt[rule.loc[0]] - sum_best_beta[rule.loc[0]];
	      //cout << "\nOptimal step length: " << step_length_opt;

	      // Update the weight of the best n-gram.
	      // Insert or update new feature.
		if ( C != 0 ) {
		      map<string, double>::iterator features_it = features_cache.find(rule.ngram);
		      if (features_it == features_cache.end()) {
			// If feature not there, insert it.
		    	  features_cache[rule.ngram] = step_length_opt;
		      } else {// Adjust coeficient and the sums of coeficients.
		    	  sum_squared_betas = sum_squared_betas - pow(features_it->second, 2);
		    	  sum_abs_betas = sum_abs_betas - abs(features_it->second);
		      	  features_it->second += step_length_opt;
		      }
		      sum_squared_betas += pow(features_cache[rule.ngram], 2);
                      sum_abs_betas += abs(features_cache[rule.ngram]);

		}
		sum_abs_scalar_prod_diff = 0;
		sum_abs_scalar_prod = 0;
		// Remember the loss from prev iteration.
		old_loss = loss;		
		loss = 0;
		for (unsigned int i = 0; i < transaction.size();  ++i) {
		      	// Compute loss.
			if (objective == 0) { //SLR
				if (y[i] * sum_best_beta_opt[i] > 8000) {
					exp_fraction[i] = 0;
				  } else {
					  exp_fraction[i] = 1 / (1 + exp(y[i] * sum_best_beta_opt[i]));
				  }

				  if (-y[i] * sum_best_beta_opt[i] > 8000) {
				  	loss += log(LDBL_MAX);
				  } else {
					loss += log(1 + exp(-y[i] * sum_best_beta_opt[i]));
				  }
			} //end SLR
			/*
			if (objective == 1) { //L1-SVM (Hinge loss)
				if (1 - y[i] * sum_best_beta_opt[i] > 0)
					loss += 1 - y[i] * sum_best_beta_opt[i];
			} //end L1-SVM
			*/

			if (objective == 2) { //L2-SVM
				if (1 - y[i] * sum_best_beta_opt[i] > 0)
					loss += pow(1 - y[i] * sum_best_beta_opt[i], 2);
			} //end L2-SVM

			//cout << "\n y[i]: " << y[i];
			//cout << "\nsum_best_beta_opt[i]: " << sum_best_beta_opt[i];
			//cout << "\nloss: " << loss;

			// Compute the sum of per document difference between the scalar product at 2 consecutive iterations.
		  	sum_abs_scalar_prod_diff += abs(sum_best_beta_opt[i] - sum_best_beta[i]);
		  	// Compute the sum of per document scalar product at current iteration.
		  	sum_abs_scalar_prod += abs(sum_best_beta_opt[i]);
		}

		if (verbosity >= 2) {
			cout << "\nloss: " << loss;
			if ( C != 0 ) {		
				cout << "\npenalty_term: " << C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
			}
		}
		// Update the log-likelihood with the regularizer term.
		if ( C != 0 ) {		
			loss = loss + C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
		}

		//stop if loss doesn't improve; a failsafe check on top of conv_rate (based on predicted score residuals) reaching conv_threshold
		if (old_loss - loss == 0) {		
			if (verbosity >= 1) {
		    		cout << "\n\nFinish iterations due to: no change in loss value!";
				cout << "\nloss + penalty term: " << loss;
		    		cout << "\n# iterations: " << itr + 1;
		    	}
			break;
		} //else cout << "\nold_loss-new_loss: " << old_loss - loss;

		//cout << "time now: " << gettimeofday(t, NULL);

	      	// The optimal step length as obtained from the line search. 
		// Stop the alg if weight of best grad feature is below a given threshold.
		// Inspired by paper of Liblinear people that use a thereshold on the value of the gradient to stop close to optimal solution.
		if (abs(step_length_opt) > 1e-8)	      	
			os << step_length_opt << ' ' << rule.ngram << std::endl;
		else {
			if (verbosity >= 1) {
		    		cout << "\n\nFinish iterations due to: step_length_opt <= 1e-8 (due to numerical precision loss doesn't improve for such small weights)!";
		    		cout << "\n# iterations: " << itr + 1;
		    	}
		    	break;
		}		
		
		if (verbosity >= 2) {
		      	std::cout <<  "\n" << itr << " " << features_cache.size () << " " << rewritten << "/" << pruned << "/" << total << " "
					<< step_length_opt << " " << rule.ngram;

			cout << "\nloss + penalty term: " << loss;
			cout.flush();
		}

		tau = 0;
		rule.ngram = "";
		rule.gradient = 0;
		rule.loc.clear ();
		rule.size = 0;

		// Set the convergence rate as in paper by Madigan et al on BBR.
		convergence_rate = sum_abs_scalar_prod_diff / (1 + sum_abs_scalar_prod);
		//cout << "\nconvergence rate: " << convergence_rate;

		if (convergence_rate <= convergence_threshold) {
			if (verbosity >= 1) {
				cout << "\nconvergence rate: " << convergence_rate;
		    		cout << "\n\nFinish iterations due to: convergence test (convergence_thereshold=" << convergence_threshold << ")!";
		    		cout << "\n# iterations: " << itr + 1;
		    	}
		    	break;
		} // Otherwise, loop up to the user provided # iter or convergence threshold.

		//sum_best_beta is the optimum found using line search
		sum_best_beta.assign(sum_best_beta_opt.begin(), sum_best_beta_opt.end());

		//gettimeofday(&t, NULL);
		//cout << "\ntime 1 boosting iter: " << t.tv_sec - t_start.tv_sec << " seconds; " << (t.tv_sec - t_start.tv_sec) / 60.0 << " minutes ";
	} //end optimization iterations.

	gettimeofday(&t, NULL);
	if (verbosity >= 1) {
		if ( C != 0 ) {		
			cout << "\nend penalty_term: " << C * (alpha * sum_abs_betas + (1 - alpha) * 0.5 * sum_squared_betas);
		}
		cout << "\nend loss + penalty_term: " << loss;
		cout << "\n\ntotal time: " << t.tv_sec - t_origin.tv_sec << " seconds; " << (t.tv_sec - t_origin.tv_sec) / 60.0 << " minutes\n ";
	}
	return true;
  } //end run().
};

#define OPT " [-o objective_function] [-m minsup] [-l minpat] [-L maxpat] [-g gap] [-r traversal_strategy ] [-T #round] [-n token_type] [-c convergence_threshold] [-C regularizer_value] [-a l1_vs_l2_regularizer_weight] [-v verbosity] training_file model_file"

int main (int argc, char **argv)
{
	extern char *optarg;
	// By default the objective fct is logistic regression. 
	// objective=0: SLR 
	// objective=1: for Hinge loss aka l1-loss-SVM (disabled for now)
	// objective=2: for squared-hinge-loss aka l2-loss-SVM
	unsigned int objective = 0;
  	unsigned int maxpat = 0xffffffff;
  	unsigned int minpat = 1;
	unsigned int maxitr = 5000;
  	unsigned int minsup = 1;
  	// Max # of consec gaps allowed in a feature.
  	unsigned int maxgap = 0;
  	// Word or character token type. By default char token.
  	bool token_type = 1;
  	// BFS vs DFS traversal. By default BFS.
  	bool traversal_strategy = 0;

  	// The C regularizer parameter in regularized loss formulation. It constraints the weights of features.
  	// C = 0 no constraints (standard SLR), the larger the C, the more the weights are shrinked towards each other (using L2) or towards 0 (using L1)
  	double C = 1;
	// The alpha parameter decides weight on L1 vs L2 regularizer: alpha * L1 + (1 - alpha) * L2. By default we use an L2 regularizer.
	double alpha = 0.2; 

  	double convergence_threshold = 0.005;
  	int verbosity = 1;

  	if (argc < 2) {
    	std::cerr << "Usage: " << argv[0] << OPT << std::endl;
    	return -1;
  	}

  	int opt;
	while ((opt = getopt(argc, argv, "o:T:L:l:m:g:n:r:c:C:a:v:")) != -1) {
		switch(opt) {
			case 'o':
			      objective = atoi (optarg);
			      break;	
		    case 'T':
		      maxitr = atoi (optarg);
		      break;
		    case 'L':
		      maxpat = atoi (optarg);
		      break;
		    case 'l':
		      minpat = atoi (optarg);
		      break;
		    case 'm':
		      minsup = atoi (optarg);
		      break;
		    case 'g':
		    	maxgap = atoi (optarg);
		    	break;
		    case 'n':
		      token_type = atoi (optarg);
		      break;
		    case 'r':
		      	traversal_strategy = atoi (optarg);
		      	break;
		    case 'c':
		      convergence_threshold = atof (optarg);
		      break;
		    case 'C':
		    	C = atof (optarg);
		    	break;
		    case 'a':
		    	alpha = atof (optarg);
		    	break;
		    case 'v':
		      verbosity = atoi (optarg);
		      break;
		    default:
		      std::cout << "Usage: " << argv[0] << OPT << std::endl;
		      return -1;
	    }
  	}

	if (verbosity >= 1) {
  	  cout << "\nParameters used: " << "obective fct: " << objective << " T: " << maxitr << " minpat: " << minpat << " maxpat: " << maxpat << " minsup: " << minsup
  		<< " maxgap: " << maxgap << " token_type: " << token_type << " traversal_strategy: " << traversal_strategy
  		<< " convergence_threshold: " 	<< convergence_threshold << " C (regularizer value): " << C << " alpha (weight on l1_vs_l2_regularizer): " 
		<< alpha  << " verbosity: " << verbosity;
    }

  	SeqLearner seql_learner;
  	seql_learner.run (argv[argc-2], argv[argc-1], objective, maxpat, minpat, maxitr,
  			minsup, maxgap, token_type, traversal_strategy, convergence_threshold, C, alpha, verbosity);

  	return 0;
}
