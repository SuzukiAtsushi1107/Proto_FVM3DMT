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
#define OPTIM_ENABLE_EIGEN_WRAPPERS
#include "optim.hpp"
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "pch.h"
#include <iostream>
#include "ReadData.h"
#include <time.h>
#include "Analysis.h"
#include <kv/complex.hpp>
#include <boost/version.hpp>
//template <class T> ub::vector<T> f(double b, ub::vector<T>& x,double a) {
//	ub::vector<T> y(2);
//
//	y(0) = 2. * x(0) * x(0) * x(1) - 1.;
//	y(1) = x(0) + 0.5 * x(1) * x(1) - 2.;
//	y(0) = y(0)*a*b;
//	return y;
//}

inline double booth_fn(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
{
	double x_1 = vals_inp(0);
	double x_2 = vals_inp(1);

	double obj_val = std::pow(x_1 + 2 * x_2 - 7.0, 2) + std::pow(2 * x_1 + x_2 - 5.0, 2);
	//
	if (grad_out) {
		(*grad_out)(0) = 2 * (x_1 + 2 * x_2 - 7.0) + 2 * (2 * x_1 + x_2 - 5.0) * 2;
		(*grad_out)(1) = 2 * (x_1 + 2 * x_2 - 7.0) * 2 + 2 * (2 * x_1 + x_2 - 5.0);
	}
	//
	return obj_val;
}

int main(int args,char* argv[])
{

	
	const int test_dim = 2;
	Eigen::VectorXd xa = Eigen::VectorXd::Ones(test_dim)*2; // initial values (1,1,...,1)
	bool success = optim::cg(xa, booth_fn, nullptr);
	if (success) {
		std::cout << "cg: sphere test completed successfully." << "\n";
	}
	else {
		std::cout << "cg: sphere test completed unsuccessfully." << "\n";
	}
	std::cout << "cg: solution to sphere test:\n" << xa << std::endl;

	//ub::vector<kv::complex<double>> v1, v2;
	//ub::vector<kv::autodif<kv::complex<double>>> va1, va2;
	//ub::matrix<kv::complex<double>> m;

	//v1.resize(2);
	//v1(0) = 5.; v1(1) = 6.;

	//va1 = kv::autodif< kv::complex<double >>::init(v1);

	//va2 = f(2.0,va1,0.5);

	//kv::autodif<kv::complex<double>>::split(va2, v2, m);

	//std::cout << v2 << "\n"; // f(5, 6)
	//std::cout << m << "\n"; // Jacobian matrix
	bool forwardCalc = false;
	time_t start_t = time(NULL);
	if (argv[1] == NULL) {
		std::cout << "Calculation Data Must be Writen In argv!!" << std::endl;
		exit(1);
	}
	std::string modelFileName = argv[1];

	struct stat st;
	const char* file = modelFileName.c_str();
	int ret = stat(file, &st);
	if (0 != ret) {
		std::cout << "Calculation Data Does Not Exist!!" << std::endl;
		exit(1);
	}

	ReadData::ReadData* readData = new ReadData::ReadData();
	readData->ReadFile(modelFileName, forwardCalc); //読んだデータのクラスを作成
	Analysis::Analysis*  analysis=new Analysis::Analysis{ readData }; //実際に解析をするクラスを作成
	if (forwardCalc == true) {
		analysis->RunAnalysis(); //解析実行
	}
	else {
		analysis->RunOptimize();
	}

	time_t end_t = time(NULL);
	std::cout << "Total CalcTime:"<<(end_t-start_t)/60<<" minute" << std::endl;
}

