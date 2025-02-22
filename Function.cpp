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
#include "Function.h"
Function::Function::Function() {

}
double Function::Function::CalcValue(double variable) { //this function has a premise that variables order from small to large
	double val;
	if (variable <= variables[0]) {
		val = values[0]; // if small more than data, use the smallest one.
		return val;
	}
	else if (variables.size() == 1) {
		val = values[0]; // if data has only one data.
		return val;
	}
	for (int i = 1; i < variables.size(); i++) {
		double variablePre = variables[i - 1];
		double valuePre = values[i - 1];
		double variableCur = variables[i];
		double valueCur = values[i];
		if (variableCur >= variable) {
			double w1 = abs(variableCur - variable);
			double w2 = abs(variablePre - variable);
			double val = (w1*valuePre + w2 * valueCur) / (w1 + w2); //linear interpolation
			return val;
		}
		if (i == variables.size() - 1) {
			val = values[values.size() - 1]; // if large more than data, use the largest one.
			return val;
		}
	}
	std::cout << "variables or values setting is wrong in FUNCTION" << std::endl; //something wrong happend.
	exit(1);
}