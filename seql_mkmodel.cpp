/*
 * Author: Georgiana Ifrim (georgiana.ifrim@gmail.com)
 *
 * This library takes as input the classification model provided by seql_learn.cpp (with potential repetitions of the same feature),
 * prepares the final model by aggregating the weights of repeated (identical) features
 * and builds a trie from the resulting (unique) features for fast classification (as done in seql_classify.cpp).
 *
 * The library uses parts of Taku Kudo's
 * open source code for BACT, available from: http://chasen.org/~taku/software/bact/
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation.
 *
*/


#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>
#include "unistd.h"
#include "darts.h"
#include "common.h"

#define OPT " [-i model_file] [-o binary_model_file] [-O predictors_file]"

template <typename T1, typename T2>
struct pair_2nd_cmp: public std::binary_function<bool, T1, T2> {
   bool operator () (const std::pair <T1, T2>& x1, const std::pair<T1, T2> &x2)
   {
      return x1.second > x2.second;
   }
};

int main (int argc, char **argv)
{
  std::string file  = "";
  std::string index = "";
  std::string ofile = "";
  extern char *optarg;

  int opt;
  while ((opt = getopt(argc, argv, "i:o:O:")) != -1) {
    switch(opt) {
    case 'i':
      file = std::string (optarg);
      break;
    case 'o':
      index = std::string (optarg);
      break;
    case 'O':
      ofile = std::string (optarg);
       break;
     default:
      std::cout << "Usage: " << argv[0] << OPT << std::endl;
      return -1;
    }
  }

  if (file.empty () || index.empty ()) {
    std::cout << "Usage: " << argv[0] << OPT << std::endl;
    return -1;
  }

  std::istream *is;
  if (file == "-")  is = &std::cin;
  else              is = new std::ifstream (file.c_str());

  if (! *is) {
    std::cerr << "Cannot Open: " << file << std::endl;
    return -1;
  }

  std::vector <Darts::DoubleArray::key_type *> ary;
  std::vector <std::pair<const char *, double> > ary2;
  std::vector <double> alpha;
  std::map<std::string, double> rules;

  char buf[8192];
  char *column[2];
  double bias = 0.0;
  double alpha_sum = 0.0;
  double l1_norm = 0.0;
  double l2_norm = 0.0;

	while (is->getline (buf, 8192)) {
	  	if (buf[strlen(buf) - 1] == '\r') {
	  		buf[strlen(buf) - 1] = '\0';
	  	}

	  	//cout << "\nline:" << no_cr_line;
      	//cout.flush();
    	if (2 != tokenize (buf, "\t ", column, 2)) {
	 		std::cerr << "FATAL: Format Error: " << buf << std::endl;
	 		return -1;
      	}
    	// Ignore rules containing only 1 character.
    	//if (strlen(column[1]) <= 1) continue;

		double a = atof (column[0]);
    	bias -= a;
    	alpha_sum += std::abs (a);
    	rules[column[1]] += 2 * a;
  	}

  bias /= alpha_sum;
  //bias = 0;
 l1_norm = alpha_sum;

  for (std::map<std::string, double>::iterator it = rules.begin(); it != rules.end(); ++it) {
    double a = it->second / alpha_sum;
    l2_norm += 	pow(it->second, 2);

    ary2.push_back (std::make_pair <const char*, double>(it->first.c_str(), a));
    ary.push_back  ((Darts::DoubleArray::key_type *)it->first.c_str());
    alpha.push_back (a);
  }

  l2_norm = pow(l2_norm, 0.5);

  std::cout << "Total: " << alpha.size() << " rule(s)" << std::endl;
  std::cout << "l1_norm: " << l1_norm << ", l2_norm: " << l2_norm << std::endl;

  if (ary.empty()) {
    std::cerr << "FATAL: no feature is added" << std::endl;
    return -1;
  }

  if (file != "-") delete is;

  Darts::DoubleArray da;

  if (da.build (ary.size(), &ary[0], 0, 0, 0) != 0) {
    std::cerr << "Error: cannot build double array  " << file << std::endl;
    return -1;
  }

  std::ofstream ofs (index.c_str(), std::ios::binary|std::ios::out);

  if (!ofs) {
    std::cerr << "Error: cannot open " << index << std::endl;
    return -1;
  }

  unsigned int s = da.size() * da.unit_size();
  ofs.write ((char *)&s, sizeof (unsigned));
  ofs.write ((char *)da.array (), s);
  ofs.write ((char *)&bias, sizeof (double));
  ofs.write ((char *)&alpha[0], sizeof (double) * alpha.size());
  ofs.close ();

  if (! ary2.empty() && ! ofile.empty()) {
    std::ofstream ofs2 (ofile.c_str());
    if (! ofs2) {
       std::cerr << "Cannot Open: " << ofile << std::endl;
       return -1;
    }
    ofs2.precision (24);
    ofs2 << bias << std::endl;
    std::sort (ary2.begin(), ary2.end(), pair_2nd_cmp <const char*, double>());
    for (unsigned int i = 0; i < ary2.size (); ++i) ofs2 << ary2[i].second << " " << ary2[i].first << std::endl;
  }

  return 0;
}
