/*
Copyright c 2025 Suzuki Atsushi <mk.pn14951011 at gmail.com>
For Non-commercial uses, this work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
For commercial uses, when you use this work until April 1, 2025, you must publish the original or derived source works you use as the terms of the Do What The Fuck You Want To Public License, Version 2.
This term for commercial uses is prioritized the other terms for this work.
The other conditions are same as terms of the Do What The Fuck You Want To Public License, Version 2.
After April 1, 2025, this term is expired and this work comply terms of the Do What The Fuck You Want To Public License, Version 2.
*/
#pragma once
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "pch.h"
#include <iostream>
#include <fstream>
#include "ReadData.h"
#include "Property.h"
#include "InvSettings.h"
#include <string.h>
#include <sys/stat.h>
#define _CRT_NONSTDC_NO_DEPRECATE
#define _CRT_SECURE_NO_WARNINGS

ReadData::ReadData::ReadData() {
	invSettings = new InvSettings::InvSettings(0);
	output = new Output::Output();
}
#include <iostream>
#include <string>
#include <vector>

std::vector<std::string> ReadData::ReadData::split(std::string str) {
	std::vector<std::string> result;
	std::string subStr;
	std::vector<char> del;
	del.push_back(' ');
	del.push_back ('\t');
	del.push_back ('\n');
	for (const char c : str) {
		bool delFlag = false;
		for (auto itr = del.begin(); itr != del.end(); ++itr) {
			if (c == *itr) {
				delFlag = true;
			}
		}
		//if (c == del) {
		if (delFlag){
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
//
//std::vector<std::string> ReadData::ReadData::split(std::string* line) {
//
//	//std::string str;
//	std::vector<std::string> returnLine;
//	std::vector<char> del;
//	del.push_back(' ');
//	del.push_back('\t');
//	del.push_back('\n');
//	std::vector<int> splitPoint;
//	for (int i = 0; i < line->length(); i++) {
//		//std::cout << "test" << std::endl;
//		std::string tmpLine = *line;
//		for (auto iDel = del.begin(); iDel != del.end(); iDel++) {
//			if (tmpLine[i] == *iDel) {
//				splitPoint.push_back(i);
//			}
//		}
//	}
//	std::sort(splitPoint.begin(), splitPoint.end());//昇順ソート
//	int startPoint = 0;
//	//std::vector<std::string> str;
//	for (int i = 0; i < splitPoint.size(); i++) {
//		returnLine.push_back("");
//		if (i == 0) {
//			if (splitPoint[i] > 0) {
//				for (int j = 0; j < splitPoint[i]; j++) {
//					returnLine[i] += line[j];
//				}
//			}
//		}
//		else if (i == splitPoint.size() - 1) {
//			if (splitPoint[i] < line->length() - 1) {
//				for (int j = splitPoint[i]+1; j < line->length(); j++) {
//					returnLine[i] += line[j];
//				}
//			}
//		}
//		else {
//			if (splitPoint[i] != splitPoint[i + 1] - 1) {
//				for (int j = splitPoint[i] + 1; j < splitPoint[i + 1]; j++){
//					returnLine[i] += line[j];
//				}
//			}
//		}
//
//		//returnLine.push_back(str);
//	}
//	//for (int i = 0; i < line.length(); i++) {
//	//	if (line[i] == ' ' || line[i] == '\t' || line[i] == '\n') {
//	//		if (!str.empty()) {
//	//			returnLine.push_back(str);
//	//			str.clear();
//	//		}
//	//	}
//	//	else {
//	//		str += line[i];
//	//		
//	//		
//	//	}
//	//}
//	
//	//returnLine.push_back(str);
//	//std::cout << returnLine.size()<< std::endl;
//	return returnLine;
//}
std::string ReadData::ReadData::AnalysisTag(std::string tmpLine) {
	std::vector<std::string> line;
	line= split(tmpLine);
	std::string tag;
	for (auto itr = line.begin(); itr != line.end(); ++itr) {
		std::string word = *itr;
		const char* tmpWord = word.c_str();

		std::string compare1{ word[0] };
		std::string compare2{ "#" }; //For judgement if Comment or not.
		if (_stricmp("Elements", tmpWord) == 0) { //大文字小文字区別せずタグがElementsの場合
			tag = "ELEMENTS";
			break;
		}
		else if (_stricmp("PROPERTIES", tmpWord) == 0) {  //大文字小文字区別せずタグがPropertiesの場合
			tag = "PROPERTIES";
			break;
		}
		else if (_stricmp("BOUNDARY", tmpWord) == 0) { // 大文字小文字区別せずタグがCONDITIONSの場合
			tag = "BOUNDARY";
			break;
		}
		else if (_stricmp("OBSDATAFILE", tmpWord) == 0) { // 大文字小文字区別せずタグがCONDITIONSの場合
			tag = "OBSDATAFILE";
			break;
		}
		else if (_stricmp("INVSETTINGS", tmpWord) == 0) { // 大文字小文字区別せずタグがINVSETTINGSの場合
			tag = "INVSETTINGS";
			break;
		}
		//else if (_stricmp("#", &tmpWord[0]) == 0) {
		else if(compare1==compare2){
			tag = "COMMENT";
			break;
		}
		else {
			std::cout << "No Match Tag In Your Data" << std::endl;
			exit(1);
		}
	}


	return tag;
}



void ReadData::ReadData::ReadFile(std::string modelFileName, bool forwardCalc) {

	std::ifstream f(modelFileName);

	std::string line;
	bool endOfFile = false;
	std::getline(f, line);

	while (!f.eof()) {
		std::string tag = AnalysisTag(line);
		const char* tmpTag = tag.c_str();
		if (_stricmp("ELEMENTS", tmpTag) == 0) {
			AnalysisElements(&f);
			std::getline(f, line);
			continue;
		}
		if (_stricmp("PROPERTIES", tmpTag) == 0) {
			AnalysisProperties(&f);
			std::getline(f, line);
			continue;
		}
		if (_stricmp("BOUNDARY", tmpTag) == 0) {
			AnalysisBoundary(&f);
			std::getline(f, line);
			continue;
		}
		if (_stricmp("INVSETTINGS", tmpTag) == 0) {
			AnalysisInvSettings(&f);
			std::getline(f, line);
			continue;
		}
		if (_stricmp("COMMENT", tmpTag) == 0) {
			std::getline(f, line);
			continue;
		}
		std::getline(f, line);
	}
	f.close();
	if (forwardCalc == false) {
		ReadImpedanceObsData(invSettings->impedanceFile);
		ReadTipperObsData(invSettings->tipperFile);
	}
}
std::vector<std::string> ReadData::ReadData::readNext(std::ifstream* f) {
	while (true) {
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

		if (compare1==compare2) {
			continue;
		}
		else {
			return line;
		}
	}
}



void ReadData::ReadData::AnalysisElements(std::ifstream* f) {
	std::vector<std::string> line = readNext(f);
	while (true) {
		if (_stricmp(line[0].c_str(), "END") == 0) {
			if (_stricmp(line[1].c_str(), "ELEMENTS") == 0) {
				break;
			}
		}
		if (line.size() != 9) {
			std::cout << "Wrong Elements Data" << std::endl;
			exit(1);
		}
		std::string ID = line[0];
		double rootCoord1 = stod(line[1]);
		double rootCoord2 = stod(line[2]);
		double rootCoord3 = stod(line[3]);
		double dx = stod(line[4]);
		double dy = stod(line[5]);
		double dz = stod(line[6]);
		int propID = stoi(line[7]);
		bool isParent = false;
		if (!_stricmp("true",line[8].c_str())) {
			isParent = true;
		}
		Element::Element* element = new Element::Element();
		element->ID = ID;
		element->rootCoord.coeffRef(0) = rootCoord1;
		element->rootCoord.coeffRef(1) = rootCoord2;
		element->rootCoord.coeffRef(2) = rootCoord3;
		element->dx = dx;
		element->dy = dy;
		element->dz = dz;
		element->centerCoord.coeffRef(0) = rootCoord1 + dx / 2;
		element->centerCoord.coeffRef(1) = rootCoord2 + dy / 2;
		element->centerCoord.coeffRef(2) = rootCoord3 + dz / 2;
		element->isParent = isParent;
		element->propID = propID;
		elements[ID] = element;
		elementsVector.push_back(element);
		line = readNext(f);
		if (f->eof()) {
			std::cout << "No END ELEMENTS In Data" << std::endl;
			exit(1);
		}
	}
}
void ReadData::ReadData::AnalysisProperties(std::ifstream* f) {
	std::vector<std::string> line = readNext(f);
	while (true) {
		if (_stricmp(line[0].c_str(), "END") == 0) {
			if (_stricmp(line[1].c_str(), "PROPERTIES") == 0) {
				break;
			}
		}
		if (_stricmp(line[0].c_str(), "PROPERTY") == 0) {
			Property::Property* property = new Property::Property();
			property->ID = stoi(line[1]);

			line = readNext(f);
			while (true) {
				if (_stricmp(line[0].c_str(), "END") == 0) {
					if (_stricmp(line[1].c_str(), "PROPERTY") == 0) {
						break;
					}
				}
				if (_stricmp(line[0].c_str(), "Resistivity") == 0) {
						property->resistivity = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "Type") == 0) {
					if (stod(line[1]) == Property::Property::types::NORMAL) {
						property->type = Property::Property::types::NORMAL;
					}
					else if (stod(line[1]) == Property::Property::types::AIR) {
						property->type = Property::Property::types::AIR;
					}
					else if (stod(line[1]) == Property::Property::types::FIXED) {
						property->type = Property::Property::types::FIXED;
					}
					else {
						std::cout << "Wrong Data in Property Type" << std::endl;
						exit(1);
					}
					line = readNext(f);
				}
				else {
					std::cout << "Wrong Data in Property" << std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END Property In Data" << std::endl;
					exit(1);
				}
			}
			properties[property->ID] = property;
			propertiesVector.push_back(property);
		}
		else {
			std::cout << "Wrong Data in Properties" << std::endl;
			exit(1);
		}
		line = readNext(f);
		if (f->eof()) {
			std::cout << "No END Properties In Data" << std::endl;
			exit(1);
		}
	}
}
void ReadData::ReadData::AnalysisBoundary(std::ifstream* f) {
	boundary = new Boundary::Boundary();
	std::vector<std::string> line = readNext(f);
	while (true) {
		if (_stricmp(line[0].c_str(), "END") == 0) {
			if (_stricmp(line[1].c_str(), "BOUNDARY") == 0) {
				break;
			}
		}
		if (_stricmp(line[0].c_str(), "omega") == 0) {
			int i = 0;
			line = readNext(f);
			while (true) {
				if (_stricmp(line[0].c_str(), "END") == 0) {
					if (_stricmp(line[1].c_str(), "omega") == 0) {
						break;
					}
				}
				boundary->omega.push_back( stod(line[0]));
				i++;
				line = readNext(f);
				if (f->eof()) {
					std::cout << "No END omega In Data" << std::endl;
					exit(1);
				}
			}
		}
		else {
			std::cout << "Wrong Data in BOUNDARY" << std::endl;
			exit(1);
		}


		line = readNext(f);
		if (f->eof()) {
			std::cout << "No END BOUNDARY In Data" << std::endl;
			exit(1);
		}
	}
}

void ReadData::ReadData::ReadImpedanceObsData(string obsFileName) {
	//Impedance
	struct stat st;
	const char* file = obsFileName.c_str();
	int ret = stat(file, &st);
	if (obsFileName == "") {
		return;
	}
	else if (0 != ret) {
		std::cout << "Impedance File does not exist." << std::endl;
		exit(1);
	}

	std::ifstream f(obsFileName);

	std::vector<std::string> line;
	line = readNext(&f);

	int iID = lastObsDataID;
	while (!f.eof()) {
		
		if (line.size()!=2) {
			std::cout << "Zobs Coordinate Setting is wrong." << std::endl;
			exit(1);
		}
		Eigen::Vector2d coord;
		double x = stod(line[0]);
		double y = stod(line[1]);
		coord.coeffRef(0) = x;
		coord.coeffRef(1) = y;
		ObsData::ObsData* tmpObsData = new ObsData::ObsData();
		tmpObsData->isImpedanceData = true;
		tmpObsData->coord = coord;
		for (int i = 0; i < boundary->omega.size(); i++) {
			line = readNext(&f);
			if (line.size() != 8) {
				std::cout << "Number Of Zobs is wrong." << std::endl;
				exit(1);
			}
			Eigen::Matrix2cd tmpZobs;
			tmpZobs.coeffRef(0, 0).real(stod(line[0]));
			tmpZobs.coeffRef(0, 0).imag(stod(line[1]));
			tmpZobs.coeffRef(0, 1).real(stod(line[2]));
			tmpZobs.coeffRef(0, 1).imag(stod(line[3]));
			tmpZobs.coeffRef(1, 0).real(stod(line[4]));
			tmpZobs.coeffRef(1, 0).imag(stod(line[5]));
			tmpZobs.coeffRef(1, 1).real(stod(line[6]));
			tmpZobs.coeffRef(1, 1).imag(stod(line[7]));
			tmpObsData->ZobsVector.push_back(tmpZobs);
			line = readNext(&f);
			Eigen::Matrix2d tmpWeightReal;
			Eigen::Matrix2d tmpWeightImag;
			tmpWeightReal.coeffRef(0, 0)=stod(line[0]);
			tmpWeightImag.coeffRef(0, 0) = stod(line[1]);
			tmpWeightReal.coeffRef(0, 1) = stod(line[2]);
			tmpWeightImag.coeffRef(0, 1) = stod(line[3]);
			tmpWeightReal.coeffRef(1, 0) = stod(line[4]);
			tmpWeightImag.coeffRef(1, 0) = stod(line[5]);
			tmpWeightReal.coeffRef(1, 1) = stod(line[6]);
			tmpWeightImag.coeffRef(1, 1) = stod(line[7]);
			tmpObsData->varianceZobsVectorReal.push_back(tmpWeightReal);
			tmpObsData->varianceZobsVectorImag.push_back(tmpWeightImag);
		}
		tmpObsData->ID = iID;
		iID++;
		lastObsDataID++;
		obsData.push_back(tmpObsData);
		line = readNext(&f);
	}
}

void ReadData::ReadData::ReadTipperObsData(string obsFileName) {
	//Impedance
	std::ifstream f(obsFileName);
	struct stat st;
	const char* file = obsFileName.c_str();
	int ret = stat(file, &st);
	if (obsFileName == "") {
		return;
	}
	else if (0 != ret) {
		std::cout << "Tipper File does not exist." << std::endl;
		exit(1);
	}
	std::vector<std::string> line;
	line = readNext(&f);

	int iID = lastObsDataID;
	while (!f.eof()) {

		if (line.size() != 2) {
			std::cout << "Tobs Coordinate Setting is wrong." << std::endl;
			exit(1);
		}
		Eigen::Vector2d coord;
		double x = stod(line[0]);
		double y = stod(line[1]);
		coord.coeffRef(0) = x;
		coord.coeffRef(1) = y;
		ObsData::ObsData* tmpObsData = new ObsData::ObsData();
		tmpObsData->isTipperData = true;
		tmpObsData->coord = coord;
		for (int i = 0; i < boundary->omega.size(); i++) {
			line = readNext(&f);
			if (line.size() != 4) {
				std::cout << "Number Of Tobs is wrong." << std::endl;
				exit(1);
			}
			Eigen::Vector2cd tmpTobs;
			tmpTobs.coeffRef(0).real(stod(line[0]));
			tmpTobs.coeffRef(0).imag(stod(line[1]));
			tmpTobs.coeffRef(1).real(stod(line[2]));
			tmpTobs.coeffRef(1).imag(stod(line[3]));
			tmpObsData->TobsVector.push_back(tmpTobs);
			line = readNext(&f);
			Eigen::Vector2d tmpWeightReal;
			Eigen::Vector2d tmpWeightImag;
			tmpWeightReal.coeffRef(0) = stod(line[0]);
			tmpWeightImag.coeffRef(0) = stod(line[1]);
			tmpWeightReal.coeffRef(1) = stod(line[2]);
			tmpWeightImag.coeffRef(1) = stod(line[3]);
			tmpObsData->varianceTobsVectorReal.push_back(tmpWeightReal);
			tmpObsData->varianceTobsVectorImag.push_back(tmpWeightImag);
		}
		tmpObsData->ID = iID;
		iID++;
		lastObsDataID++;
		obsData.push_back(tmpObsData);
		line = readNext(&f);
	}


}

void ReadData::ReadData::AnalysisInvSettings(std::ifstream* f) {
	std::vector<std::string> line = readNext(f);
	
	while (true) {
		if (_stricmp(line[0].c_str(), "END") == 0) {
			if (_stricmp(line[1].c_str(), "InvSettings") == 0) {
				break;
			}
		}
	
		if (_stricmp(line[0].c_str(), "Parameters") == 0) {
			line = readNext(f);
			while (true) {
				if (_stricmp(line[0].c_str(), "END") == 0) {
					if (_stricmp(line[1].c_str(), "Parameters") == 0) {
						break;
					}
				}
				if (_stricmp(line[0].c_str(), "StepSize") == 0) {
					invSettings->par_step_size = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "ParamN") == 0) {
					invSettings->paramLogNormalization = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "LambdaMax") == 0) {
					invSettings->modelConstraintMax = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "LambdaMin") == 0) {
					invSettings->modelConstraintMin = stod(line[1]);
					line = readNext(f);
				}			
				else if (_stricmp(line[0].c_str(), "NumOfLambda") == 0) {
					invSettings->numOfCalcModelConstraint = stoi(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "maxIteration") == 0) {
					invSettings->maxIterationPerModelConstraint = stoi(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "maxResis") == 0) {
					invSettings->maxResis = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "minResis") == 0) {
					invSettings->minResis = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "gradErrorTol") == 0) {
					invSettings->grad_err_tol = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "thresholdResistivityChange") == 0) {
					invSettings->thresholdResistivityChange = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "thresholdObjFunctionChange") == 0) {
					invSettings->objFuncChangeThresholdForNextmodelConstraint = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "ManualSettingFile") == 0) {
					invSettings->manualSettingFile = line[1];
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "CoeffForSearchStepSize") == 0) {
					invSettings->coeffForSearchStepSize = stod(line[1]);
					std::cout << "Warning!!!!!!!! \"CoeffForSearchStepSize\" is not used!!\n Please Use \"decreaseFactor\"!!!!!" << std::endl;
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "UseGD") == 0) {
					if (!_stricmp("true", line[1].c_str())) {
						invSettings->optMethod = "GD";
					}
					else {
						invSettings->optMethod = "LBFGS";
					}
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "UseDistanceInModelConstraint") == 0) {
					if (!_stricmp("true", line[1].c_str())) {
						invSettings->isUseDistanceInModelConstraint = true;
					}
					else {
						invSettings->isUseDistanceInModelConstraint = false;
					}
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "UseIterativeSolver") == 0) {
					if (!_stricmp("true", line[1].c_str())) {
						invSettings->isDirectSolver = false;
					}
					else {
						invSettings->isDirectSolver = true;
					}
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "ToleranceIterativeSolver") == 0) {
					if (stod(line[1]) < 0.0) {
						std::cout << "Value ToleranceIterativeSolver must be 0 or more than 0." << std::endl;
						exit(1);
					}
					invSettings->toleranceIterativeSolver = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "loosenFactor") == 0) {
					if (stod(line[1]) < 1.0) {
						std::cout << "Value loosenFactor must be 1 or more than 1." << std::endl;
						exit(1);
					}
					invSettings->loosenFactor = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "decreaseFactor") == 0) {
					if (stod(line[1]) > 1.0) {
						std::cout << "Value decreaseFactor must be 1 or less than 1." << std::endl;
						exit(1);
					}
					invSettings->decreaseFactor = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "minStep") == 0) {
					if (stod(line[1]) <= 0.0) {
						std::cout << "Value minStep must be more than zero." << std::endl;
						exit(1);
					}
					invSettings->minStep = stod(line[1]);
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "InheritPreviousSettingAdam") == 0) {
					if (!_stricmp("true", line[1].c_str())) {
						invSettings->inheritPreviousSettingAdam = true;
					}
					else {
						invSettings->inheritPreviousSettingAdam = false;
					}
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "SettingEachLambda") == 0) {
					line = readNext(f);
					invSettings->lambdaVector.resize(0);
					invSettings->grad_err_tolVector.resize(0);
					invSettings->par_step_sizeVector.resize(0);
					invSettings->maxIterationVector.resize(0);
					while (true) {
						if (_stricmp(line[0].c_str(), "END") == 0) {
							if (_stricmp(line[1].c_str(), "SettingEachLambda") == 0) {
								break;
							}
							else {
								std::cout << "No End SettingEachLambda In Data" << std::endl;
								exit(1);
							}
						}
						else if (_stricmp(line[0].c_str(), "data") == 0) {
							if (line.size() != 7) {
								std::cout <<"Data In SettingEachLambda Is Wrong." << std::endl;
								exit(1);
							}
							invSettings->lambdaVector.push_back(stod(line[1]));
							invSettings->par_step_sizeVector.push_back(stod(line[2]));
							invSettings->grad_err_tolVector.push_back(stod(line[3]));
							invSettings->maxIterationVector.push_back(stoi(line[4]));
							invSettings->objFuncChangeThresholdVector.push_back(stod(line[5]));
							invSettings->thresholdRelativeResistivityChangeVector.push_back(stod(line[6]));

							line = readNext(f);
						}
						else {
							std::cout << "Wrong Data in SettingEachLambda" << std::endl;
							exit(1);
						}
					}
					line = readNext(f);

				}
				else {
					std::cout << "Wrong Data in Parameters" << std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END Parameters In Data" << std::endl;
					exit(1);
				}
			}
		}
		else if (_stricmp(line[0].c_str(), "InitialSettings") == 0) {
			line = readNext(f);
			while (true) {
				if (_stricmp(line[0].c_str(), "END") == 0) {
					if (_stricmp(line[1].c_str(), "InitialSettings") == 0) {
						break;
					}
				}
				if (_stricmp(line[0].c_str(), "InitialResistivityFile") == 0) {
					ReadInitialResistivityData(line[1]);
					line = readNext(f);
				}

				else {
					std::cout << "Wrong Data in InitialSettings" << std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END InitialSettings In Data" << std::endl;
					exit(1);
				}
			}
		}
		else if (_stricmp(line[0].c_str(), "ObsFiles") == 0) {
			line = readNext(f);
			while (true) {
				if (_stricmp(line[0].c_str(), "END") == 0) {
					if (_stricmp(line[1].c_str(), "ObsFiles") == 0) {
						break;
					}
				}
				if (_stricmp(line[0].c_str(), "ImpedanceFile") == 0) {
					invSettings->impedanceFile = line[1];
					line = readNext(f);
				}
				else if (_stricmp(line[0].c_str(), "TipperFile") == 0) {
					invSettings->tipperFile = line[1];
					line = readNext(f);
				}
				else {
					std::cout << "Wrong Data in ObsFiles" << std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END ObsFiles In Data" << std::endl;
					exit(1);
				}
			}
		}
		else if (_stricmp(line[0].c_str(), "OutputSettings") == 0) {
			line = readNext(f);
			while (true) {
				if (_stricmp(line[0].c_str(), "END") == 0) {
					if (_stricmp(line[1].c_str(), "OutputSettings") == 0) {
						break;
					}
				}
				else if (_stricmp(line[0].c_str(), "iteration") == 0) {
					output->outputIteration = stoi(line[1]);
					line = readNext(f);
				}
				else {
					std::cout << "Wrong Data in OutputSettings" << std::endl;
					exit(1);
				}

				if (f->eof()) {
					std::cout << "No END OutputSettings In Data" << std::endl;
					exit(1);
				}
			}
		}
		else {
			std::cout << "Wrong Data in InvSettings" << std::endl;
			exit(1);
		}
		line = readNext(f);
		if (f->eof()) {
			std::cout << "No END InvSettings In Data" << std::endl;
			exit(1);
		}


	}
}


void ReadData::ReadData::ReadInitialResistivityData (string resisFile) {
	//Impedance
	std::ifstream f(resisFile);
	struct stat st;
	const char* file = resisFile.c_str();
	int ret = stat(file, &st);
	if (resisFile == "") {
		return;
	}
	else if (0 != ret) {
		std::cout << "Initial Resistivity File does not exist." << std::endl;
		exit(1);
	}
	std::vector<std::string> line;
	line = readNext(&f);

	int iID = 0;
	while (!f.eof()) {

		if (line.size() != 4) {
			std::cout << "Coordinate Setting In Initial Resistivity File is wrong." << std::endl;
			exit(1);
		}
		Eigen::Vector3d coord;
		double x = stod(line[0]);
		double y = stod(line[1]);
		double z = stod(line[2]);
		double resis = stod(line[3]);
		coord.coeffRef(0) = x;
		coord.coeffRef(1) = y;
		coord.coeffRef(2) = z;

		InitialData::InitialData* tmpInitialData = new InitialData::InitialData();
		tmpInitialData->ID = iID;
		tmpInitialData->coord = coord;
		tmpInitialData->resistivity = resis;

		initialData.push_back(tmpInitialData);

		iID++;
		line = readNext(&f);

	}


}