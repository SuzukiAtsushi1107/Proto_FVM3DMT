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
#include <fstream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace std;
namespace Element {
	class Element;
}
namespace Output {
	class Output {
	public:
		int outputIteration = 10;

		void DebugOutput(int loop, double omega, std::unordered_map<string, Element::Element*>* elements);
		void VTKFileOputput(double omega, std::unordered_map<string, Element::Element*>* elements, string type = "rho",string outputFile="None");
		void RhoOutput(std::unordered_map<string, Element::Element*>* elements, std::string filename = "Rho.vtk");
		void AppRhoOutputSurface(double omega, std::unordered_map<string, Element::Element*>* elements);
		void PhiOutputSurface(double omega, std::unordered_map<string, Element::Element*>* elements);
		void TipperOutputSurface(int iOmega,double omega, std::unordered_map<string, Element::Element*>* elements);
		void TxtOutputAppRho(double omega, std::unordered_map<string, Element::Element*>* elements);
		void TxtOutputResistivity(std::unordered_map<string, Element::Element*>* elements, std::string filename);
		void ImpedanceOutputSurface(vector<double> omegas, std::unordered_map<string, Element::Element*>* elements);
		void OutputObsCalcImpedance(vector<double> omegas, std::vector< Element::Element*>* obsPointsElements);
		void TipperOutputSurface(vector<double> omegas, std::unordered_map<string, Element::Element*>* elements);
	};
}
