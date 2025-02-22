/*
Copyright © 2025 Suzuki Atsushi <mk.pn14951011 at gmail.com>
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
#include "Element.h"
#include "Property.h"
#include "Boundary.h"
#include "Process.h"
#include "Function.h"
#include "InvSettings.h"
#include "ObsData.h"
#include "InitialData.h"
#include "Output.h"
namespace ReadData
{
	class ReadData
	{

	public:
		ReadData();
		std::unordered_map<std::string, Element::Element*> elements;
		std::vector< Element::Element*> elementsVector;
		std::unordered_map<int, Property::Property*> properties;
		std::vector< Property::Property*> propertiesVector;
		Boundary::Boundary* boundary;
		InvSettings::InvSettings* invSettings;
		Output::Output* output;
		std::string obsFileName;
		std::vector<ObsData::ObsData*> obsData;
		std::vector<InitialData::InitialData*> initialData;
		double weightForModelConstraint;

		int lastObsDataID = 0;


		void ReadFile(std::string modelFileName,bool forwardCalc);
		std::string AnalysisTag(std::string line);
		std::vector<std::string> split( std::string line);
		std::vector<std::string> readNext(std::ifstream* f);
		void AnalysisElements(std::ifstream* f);
		void AnalysisProperties(std::ifstream* f);
		void AnalysisBoundary(std::ifstream* f);
		void AnalysisInvSettings(std::ifstream* f);
		void AnalysisObsDataFile(std::ifstream* f);
		void ReadImpedanceObsData(string filename);
		void ReadTipperObsData(string filename);
		void ReadInitialResistivityData(string resisFile);
	};
}