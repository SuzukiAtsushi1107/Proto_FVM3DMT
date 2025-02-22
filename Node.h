/*
Copyright c 2025 Suzuki Atsushi <mk.pn14951011 at gmail.com>
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

namespace Element {
	class Element;
}
namespace Face {
	class Face;
}


namespace Node {
	class Node 
	{
	public:
		Node();
		Eigen::Vector3d x = Eigen::Vector3d::Zero();
		int	ID = -1;
		bool isAirGroundBoundary = false;
		std::map<int, Face::Face* > faces;
		std::vector<Face::Face* > facesVector;
		std::map<int, Element::Element*>	elements;
		std::vector<Element::Element*>	elementsVector;
		Eigen::SparseMatrix<double,Eigen::RowMajor> Hcoeff;
		Eigen::Matrix3d resistivity;
		Eigen::SparseMatrix<double> rhoJ;
		int numOfElementsBelongToThisNode;
		void CalcH(int numOfElements);
		void CalcResistivity(int numOfElements);
		std::vector<double> CalcWeight(std::vector<Eigen::Vector3d> xVector,Eigen::Vector3d x0);
		std::vector<double> CalcWeightByDistance(std::vector<Eigen::Vector3d> xVector, Eigen::Vector3d x0);
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW // Ç±ÇÃÉ}ÉNÉçÇí«â¡


	};
}

