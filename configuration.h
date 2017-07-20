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

// include Hamilton-Adams demosaicing
extern "C"
{
//	#include "[SPECIFY PATH TO HAMILTON ADAMS DEMOSAICING]/dmha.h"
}

// include flowcode (middlebury devkit)
#include "[SPECIFY PATH TO MIDDLEBURY DEVKIT]/cpp/colorcode.h"
#include "[SPECIFY PATH TO MIDDLEBURY DEVKIT]/cpp/flowIO.h"

// include TRWS
#include "[SPECIFY PATH TO TRWS]/MRFEnergy.h"

// path to deepmatching
const string DEEPMATCHING_PATH = "[SPECIFY PATH TO DEEPMATCHING]/deepmatching/";

// source path
const string SOURCE_FILE = __FILE__;
const string SOURCE_PATH = SOURCE_FILE.substr(0, SOURCE_FILE.rfind("/") + 1);

void HADemosaicing(float *Output, const float *Input, int Width, int Height, int RedX, int RedY) {
//	HamiltonAdamsDemosaic(Output, Input, Width, Height, RedX, RedY); // Hamilton-Adams implemented by Pascal Getreuer
}




#endif /* CONFIGURATION_H_ */
