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
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/PardisoSupport>

using namespace std;
namespace InvSettings {
	class InvSettings {
	public:
		InvSettings(int numOfLambda);
		std::string impedanceFile="";
		std::string tipperFile="";
		double par_step_size = 0.1;
		double paramLogNormalization = 1;
		double limitOfparamLogNormalization = 10; //to prevent divergence
		double modelConstraintMax = 10;
		double modelConstraintMin = 0.001;
		int numOfCalcModelConstraint = 5;
		double objFuncChangeThresholdForNextmodelConstraint = 0.0;
		vector<double> objFuncChangeThresholdVector;
		int maxIterationPerModelConstraint = 40;
		std::string optMethod = "GD";


		double grad_err_tol = 0.0;
		double rel_sol_change_tol = 0.0;

		double minResis = 0.001; 
		double maxResis = 1e5; 

		double toleranceIterativeSolver = 1e-7;
		int maxIterationBiCGSTAB=20000;

		double thresholdResistivityChange = 0.0;
		vector<double> thresholdRelativeResistivityChangeVector;
		std::vector<double> lambdaVector;
		std::vector<double> grad_err_tolVector;
		std::vector<double> par_step_sizeVector;
		std::vector<int> maxIterationVector;

		std::string manualSettingFile = "None";

		double coeffForSearchStepSize = 0;

		bool inheritPreviousSettingAdam = true;

		bool isDirectSolver = true;

		double loosenFactor = 1.1;
		double decreaseFactor = 0.5;
		double minStep = 0.0001;

		bool isUseDistanceInModelConstraint = true;

		std::vector<std::string> split(std::string str);
		std::vector<std::string> readNext(std::ifstream* f);
		void ReadManualSettingData(optim::algo_settings_t *settings);
	};
}