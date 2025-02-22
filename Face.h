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
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "pch.h"

namespace Node {
	class Node;
}
namespace Element {
	class Element;
}


namespace Face {
	class Face
	{
	public:
		Face();
		int ID = -1;
		std::vector<int> nodesID;
		std::map<int,Node::Node*> nodes;
		std::map<int,Element::Element*> elements;
		std::vector< Node::Node*> nodesVector;
		std::vector<Element::Element*> elementsVector;
		double ds = 0.0;
		bool isBoundary = false;
		std::map<int,Eigen::SparseMatrix<double, Eigen::RowMajor >> rhoNCrossRotationHds;
		Eigen::Vector3d nVec = Eigen::Vector3d::Zero();
		Eigen::Vector3d eXVec = Eigen::Vector3d::Zero();
		Eigen::Vector3d eYVec = Eigen::Vector3d::Zero();
		Eigen::Matrix3d resistivity = Eigen::Matrix3d::Zero();
		Eigen::Vector3d centerCoord = Eigen::Vector3d::Zero();
		
		enum{NOT_BOUN,X_BOUN,Y_BOUN,Z_BOUN};
		int whichBoundary= NOT_BOUN;

		void MarkBoundary();
		void CalcArea();
		void CalcRhoNCrossRotationHds(int numOfElements,int loop); //numOfElements is the number of freedom.
		Eigen::SparseMatrix<double, Eigen::RowMajor> getRhoNCrossRotationHds(Element::Element* element);


		bool CalcRhoCrossH(Eigen::MatrixXcd* A, Eigen::MatrixXcd* W, Eigen::VectorXcd* b,Element::Element* element, Eigen::VectorXcd* H,int iRow);
		Eigen::Vector3cd CalcRhoNCrossHds(Element::Element* element);
	};
}