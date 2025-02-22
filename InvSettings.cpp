/*
Copyright © 2025 Suzuki Atsushi <mk.pn14951011 at gmail.com>
For Non-commercial uses, this work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
For commercial uses, when you use this work until April 1, 2025, you must open the original or derived source codes you use.
This term for commercial uses is prioritized the other terms for this work.
The other conditions are same as terms of the Do What The Fuck You Want To Public License, Version 2.
After April 1, 2025, this term is expired and this work comply terms of the Do What The Fuck You Want To Public License, Version 2.
*/
#pragma once 
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include "InvSettings.h"
#include <fstream>
#include "optim.hpp"
#include <ostream>
InvSettings::InvSettings::InvSettings(int numOfLambda) {
	objFuncChangeThresholdVector.resize(numOfLambda);
	thresholdRelativeResistivityChangeVector.resize(numOfLambda);
}
std::vector<std::string> InvSettings::InvSettings::split(std::string str) {
	std::vector<std::string> result;
	std::string subStr;
	std::vector<char> del;
	del.push_back(' ');
	del.push_back('\t');
	del.push_back('\n');
	for (const char c : str) {
		bool delFlag = false;
		for (auto itr = del.begin(); itr != del.end(); ++itr) {
			if (c == *itr) {
				delFlag = true;
			}
		}
		//if (c == del) {
		if (delFlag) {
			if (!subStr.empty()) {
				result.push_back(subStr);
				subStr.clear();
			}
		}
		else {
			subStr += c;
		}
	}

	if (!subStr.empty()) {
		result.push_back(subStr);
	}
	return result;
}

void InvSettings::InvSettings::ReadManualSettingData(optim::algo_settings_t *settings) {
	if (manualSettingFile == "None") {
		return;
	}
	double tmpStepSize = settings->gd_settings.par_step_size;
	bool tmpIsFinishOptimize = settings->isFinishOptimize;
	
	try {
		FILE* fp;
		fopen_s(&fp,manualSettingFile.c_str(), "r");
		if (fp == NULL) {
			std::cout << "Warning::Manual Inversion Setting File Does Not Exist!" << std::endl;
			return;
		}
		std::ifstream f(manualSettingFile);
		while (!f.eof()) {
			std::vector<std::string> line = readNext(&f);
			if (line.size() < 2) {
				return;
			}
			if (_stricmp(line[0].c_str(), "stepSize") == 0) {
				if (std::stod(line[1]) > 0) {
					settings->gd_settings.par_step_size = std::stod(line[1]);
				}
				else {
					cout << "Warning::Step Size In Manual Inversion Setting File is Wrong. " << endl;
				}
			}
			else if (_stricmp(line[0].c_str(), "GoingNext") == 0) {
				if (_stricmp(line[1].c_str(), "True") == 0) {
					settings->isFinishOptimize = true;
					cout << "Going Next Lambda." << endl;
				}
				else {
					cout << "Continue This Lambda." << endl;
				}
			}
		}
		f.close();
		std::ofstream wf;
		wf.open(manualSettingFile, std::ios::trunc); //Delete the contents.
		wf.close();
	}
	catch (...) {
		settings->gd_settings.par_step_size = tmpStepSize;
		settings->isFinishOptimize = tmpIsFinishOptimize;
		std::cout << "Warning:: Could not Read Manual Inversion Setting File. No Change has occurred." << std::endl;
		std::cout << "Manual Inversion Setting File Is:" << manualSettingFile << std::endl;
		return;
	}
}

std::vector<std::string> InvSettings::InvSettings::readNext(std::ifstream* f) {
	while (true) {
		if (f->eof()) {
			std::vector<std::string> line;
			return line;
		}
		std::string tmpLine;
		std::getline(*f, tmpLine);
		std::vector<std::string> line = split(tmpLine);
		if (line.size() == 0) {
			return line;
		}
		auto itr = line.begin();
		std::string word = *itr;
		const char* iChar = word.c_str();

		std::string compare1{ word[0] };
		std::string compare2{ "#" };

		if (compare1 == compare2) {
			continue;
		}
		else {
			return line;
		}
	}
}
