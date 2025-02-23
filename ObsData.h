/*
Copyright © 2025 Suzuki Atsushi <mk.pn14951011 at gmail.com>
For Non-commercial uses, this work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
For commercial uses, when you use this work until April 1, 2025, you must publish the original or derived works you use as the terms of the Do What The Fuck You Want To Public License, Version 2.
This term for commercial uses is prioritized the other terms for this work.
The other conditions are same as terms of the Do What The Fuck You Want To Public License, Version 2.
After April 1, 2025, this term is expired and this work comply terms of the Do What The Fuck You Want To Public License, Version 2.
*/
#pragma once
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/PardisoSupport>
#include "pch.h"
using namespace std;
namespace ObsData {
	class ObsData {
	public:
		ObsData();
		int ID;
		vector<int> rowIDsJacobian;
		vector<int> rowIDEachOmega;
		Eigen::Vector2d coord;
		vector<Eigen::Matrix2cd> ZobsVector;
		bool isAlreadyFoundElementImpedance = false;
		vector<Eigen::Matrix2d> varianceZobsVectorReal;
		vector<Eigen::Matrix2d> varianceZobsVectorImag;
		bool isImpedanceData = false;

		vector<Eigen::Vector2cd> TobsVector;
		bool isAlreadyFoundElementTipper = false;
		vector<Eigen::Vector2d> varianceTobsVectorReal;
		vector<Eigen::Vector2d> varianceTobsVectorImag;
		bool isTipperData = false;
	};
}