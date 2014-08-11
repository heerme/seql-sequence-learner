Description
————————————
This page describes the usage of the SEQL (SEQuence Learner) software and gives links to open source code and sequence data. 
This tool has been used for text and biological sequence classification (see References section, PlosOne14, KDD11, KDD08), but can be applied to any string classification task.

SEQL is an implementation of a greedy coordinate-wise gradient descent technique for efficiently learning sequence classifiers.

The theoretical framework is based on discriminative sequence classification where linear classifiers work directly in the high dimensional predictor space of all subsequences in the training set (as opposed to string-kernel-induced feature spaces). This is computationally challenging, but made feasible by employing a greedy coordinate-descent algorithm coupled with bounding the magnitude of the gradient for efficiently selecting discriminative subsequences. Logistic regression (binomial log-likelihood loss) and support vector machines (squared hinge loss) are currently implemented. 


Installation
---------------
    * Requirements
          o C++ compiler (gcc 3.4 or higher)
          o POSIX getopt library.
    *

      To install SEQL download seqlv2.0.tar.gz and unpack it with

      tar -zxvf seql-v2.0.tar.gz
        Execute
          o cd seql-v2.0/
          o make or make all

      which compiles the code and creates the three executables

          seql_learn (learning module)    
          seql_mkmodel (prepare the model)
           seql_classify (classification module)

    * For running the code on a toy example
          o make test_char (or make test_word)


How to Use
-----------
A simple toy training/test example can be found in: data/toy.char.train data/toy.char.test.
The data is represented in the file as "class_label example_sequence" per line.
Please be aware of setting the parameter [-n: 0 or 1] for the token representation desired, i.e., word-level ([-n 0]), where word-tokens are separated by space, or char-level ([-n 0]), where the input is a continuous string of characters, without spaces. This parameter needs to be set for "./seql_learn", "./seql_classify" and 
"./seql_classify_tune_threshold_min_errors". To see an example of running the code for word-tokens and char-tokens, please also have a look at the Makefile.

1. Train using ./seql_learn
  Usage:
    ./seql_learn    [-o objective_function] [-m minsup] [-l minpat] [-L maxpat] [-g maxgap] [-r traversal_strategy ]
                    [-T #round] [-n token_type] [-c convergence_threshold] [-C regularizer_value] [-a l1_vs_l2_regularizer_weight]
                    [-v verbosity] train_file model_file

  Default values for parameters:
    [-o objective: 0 or 2] Objective function. Choice between logistic regression (-o 0) and squared-hinge support vector ma-
     chines (-o 2). By default set to logistic regression.
    [-g maxgap >= 0] Maximum number of consecutive gaps or wildcards allowed in a feature, e.g., a**b,
     is a feature of size 4 with any 2 characters from the input alphabet in the middle. By default
     set to 0.
    [-C regularizer value > 0] Value of the regularization parameter. By default set to 1.
    [-a alpha in [0,1]] Weight of l1 vs l2 regularizer for the elastic-net penalty. By default set to 0.2, i.e., 0.8*l1 + 0.2*l2 regularization.
    [-l minpat >= 1] Threshold on the minimum length of any feature. By default set to 1.
    [-L maxpat] Threshold on the maximum length of any feature. By default the maximum length
     is unrestricted, i.e., at most as long as the longest sequence in the training set.
    [-m minsup >= 1] Threshold on the minimum support of features, i.e., number of sequences containing
     a given feature. By default set to 1.
    [-n token type: 0 or 1] Word or character-level token. Words are delimited by white spaces. By default
     set to 1, character-level tokens.
    [-r traversal strategy: 0 or 1] Breadth First Search or Depth First Search traversal of the search tree.
     By default set to BFS.
    [-c convergence threshold >= 0] Stopping threshold based on change in aggregated score predictions.
     By default set to 0.005.
    [-T maxitr] Number of optimization iterations. By default set to the maximum between 5,000
     and the number of iterations resulting by using a convergence threshold on the aggregated
     change in score predictions.
    [-v verbosity: 1 to 5] Amount of printed detail about the training of the classifier. By default set to 1
     (light profiling information).

      
    Example call for char-token representation: (all other parameters set to their default values):
    ./seql_learn -n 1 -v 2 data/toy.char.train toy.seql.char.model

2. Prepare the final model using ./seql_mkmodel (this builds a trie on the features of the model for fast classification).
    Usage: ./seql_mkmodel [-i model_file] [-o binary_model_file] [-O predictors_file]
   
    Example call:
  ./seql_mkmodel -i toy.seql.char.model -o toy.seql.char.model.bin -O toy.seql.char.model.predictors

3. Classify using ./seql_classify (apply the learned model on new examples).
    Usage:  ./seql_classify [-n token_type: 0 word tokens, 1 char tokens; by default set to 1] [-t classif_threshold: default 0] [-v verbosity level: default 0] test_file binary_model_file

    Example call:
    ./seql_classify -n 1 -v 2 data/toy.char.test toy.seql.char.model.bin

Optionally one can tune the classification threshold on the training set, to minimize the number of training errors:
  ./seql_classify_tune_threshold_min_errors -n 1 -v 2 data/toy.char.train toy.seql.char.model.bin

    Best threshold:0.0746284

and use the best theshold for classifying the test set:
  ./seql_classify -n 1 -t 0.0746284 -v 2 data/toy.char.test toy.seql.char.model.bin


Disclaimer
----------
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.


Acknowledgements
————————————————
Many thanks to Dan Søndergaard for his patch of darts.h and related files.


References
----------

B. P. Pedersen, G. Ifrim, P. Liboriussen, K. B. Axelsen, M. G. Palmgren, P. Nissen, C. Wiuf, C. N. S. Pedersen
“Large scale identification and categorization of protein sequences using Structured Logistic Regression” (PlosOne14)
 
G. Ifrim, C. Wiuf
“Bounded Coordinate-Descent for Biological Sequence Classification in High Dimensional Predictor Space” (KDD 2011)

G. Ifrim: 
“Sequence Classification in High Dimensional Predictor Space", Cork Constraint Computation Centre, Seminar Talk, 2011.


G. Ifrim, G. Bakir, G. Weikum: "Fast Logistic Regression for Text Categorization with Variable-Length N-grams", (KDD 2008)
