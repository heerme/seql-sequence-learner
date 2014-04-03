#include <vector>
#include <string>
#include <iostream>
#include "common_string_symbol.h"

void str2node (const char *str, std::vector <stx::string_symbol>& doc, int token_type)
{
    unsigned int len = strlen (str);
  	bool at_space = false;
	std::string unigram = "";
	
    for (unsigned int pos = 0; pos < len; ++pos) {
    	// Skip white spaces. They are not considered as unigrams.
    	if (isspace(str[pos])) {
    		at_space = true;
    		continue;
    	}
    	// If word level tokens.
    	if (!token_type) {
	    	if (at_space || pos == 0) {
	    		at_space = false;
	    		
	    		if (!unigram.empty()) {
					doc.push_back(unigram);
					unigram.clear();
	    		}
	    		unigram += str[pos];
	    	} else {
	    			unigram += str[pos];
	    		}
    	} else {
    			// Char (i.e. byte) level token.
    			unigram = str[pos];
    			doc.push_back(unigram);
				unigram.clear();
	   			}    			
	}
	
	if (!token_type) {
	   	if (!unigram.empty()) {
	   		doc.push_back(unigram);
			unigram.clear();
	   	}
	} 
}

//void str2node (const char *str, std::vector <stx::string_symbol>& doc, int token_type)
//{
//   try {
//   	
//    //unsigned int size = 0;
//    unsigned int len = strlen (str);
//    std::string buf = "";
//	//char prev_char;
//	
//    for (unsigned int i = 0; i < len; i++) {
////		if (i > 0) {
////    		prev_char = str[i - 1];
////    	} else {
////    		prev_char = str[i];
////    	}
//		if (str[i] == '(' || str[i] == ')') {
//	 		if (! buf.empty()) {
//	    		doc.push_back (buf);
//	    		//std::cout << doc[doc.size() - 1] << " ";
//	    		buf = "";
//	    		//++size;
//	  		}
//	  	}
//		else {
//			if (str[i] == '\t' || str[i] == ' ') { 	  // do nothing
//			// if (str[i] == '\t') { 	  // do nothing
//			} else {
//				//if (prev_char == ' ') { //do nothing
//				//} else {
//	  				buf += str[i];
//				}
//			}		
//    }
//    //std::cout << "\n";
//	//std::cout << "bf size: " << buf.size();
//    if (!buf.empty()) {// && !isspace(buf[buf.size() - 1])) {
//    	throw 2;
//    }
//    
//    return;
//    } catch (const int) {
//      std::cerr << "Fatal: parse error << [" << str << "]\n";
//      std::exit (-1);
//    }
//}
