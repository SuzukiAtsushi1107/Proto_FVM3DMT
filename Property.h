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
#include "pch.h"

namespace Function {
	class Function;
}


namespace Property {
	class Property {
	public:
		Property();
		int ID = -1;
		double resistivity;
		enum types {NORMAL,AIR,FIXED}; //type 0:normal 1:air 2:fixed
		types type;

		double density = -1.0;
		double specificHeat = -1.0;
		int conductivityFuncID = -1;
		Function::Function* conductivityFunction;
		int densityFuncID = -1;
		Function::Function* densityFunction;
		int specificHeatFuncID = -1;
		Function::Function* specificHeatFunction;
	};
}
