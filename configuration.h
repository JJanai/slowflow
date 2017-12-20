/*
 * configuration.h
 *
 *  Created on: Jul 15, 2017
 *      Author: jjanai
 */

#ifndef CONFIGURATION_H_
#define CONFIGURATION_H_

#include <iostream>
#include <string>

#define IDENT(x) x
#define XSTR(x) #x
#define STR(x) XSTR(x)
#define CONCAT(x,y) STR(IDENT(x)IDENT(y))

//############################# PLEASE SPECIFY #############################
const std::string DEEPMATCHING_PATH = "[SPECIFY PATH TO DEEP MATCHING]";					// DeepMatching
#define MIDDLEBURY_PATH(file) 	CONCAT([SPECIFY PATH TO MIDDLEBURY DEVKIT], file)			// Middlebury Devkit
#define GCO_PATH(file) 			CONCAT([SPECIFY PATH TO GRAPH CUT], file)	    			// Graph cut library
#define TRWS_PATH(file) 		CONCAT([SPECIFY PATH TO TRWS], file)						// Tree-Reweigthed Message Passing
//#define DMGUNTURK
//#define DMGUNTURK_PATH(file) 	CONCAT([SPECIFY PATH TO HAMILTON ADAMS DEMOSAICING], file)	// Gunturk-Altunbasak-Mersereau Alternating Projections Image Demosaicking
//############################# PLEASE SPECIFY #############################

// source path
const std::string SOURCE_FILE = __FILE__;
const std::string SOURCE_PATH = SOURCE_FILE.substr(0, SOURCE_FILE.rfind("/") + 1);

#endif /* CONFIGURATION_H_ */
