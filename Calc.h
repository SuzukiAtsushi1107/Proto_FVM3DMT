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
#include "Property.h"
#include "Process.h"
#include "Function.h"
#include "ReadData.h"
#include "Output.h"


namespace Element {
	class Element;
}

namespace Analysis {
	class Analysis {
	public:
		Analysis(ReadData::ReadData* readData);
		std::map<int, Node::Node*> nodes;
		std::vector<Node::Node*> nodesVector;
		std::map<int, Face::Face*> faces;
		std::vector<Face::Face*> facesVector;
		std::map<int, Element::Element*> elements;
		std::vector<Element::Element*> elementsVector;
		std::map<int, Property::Property*> properties;
		std::vector<Property::Property*> propertiesVector;
		Boundary::Boundary* boundary;
		std::map<int, Process::Process*> processes;
		std::map<int, Function::Function*> functions;
		Output::Output output;
		Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>* globalMatrix;
		Eigen::VectorXcd*  globalVector;
		Eigen::VectorXcd result;
		double omega;
		int Jpolarization = 0;
		void RunAnalysis();
		void Initialize();
		void Calculation();
		void AssociationBetweenNodesAndFaces();
		void AssociationBetweenFacesAndElements();
		void AssociationBetweenNodesAndElements();
		void MarkBoundaryFaces();
		void CalcAreaOfFaces();
		void CalcVolumeOfElements();
		void CalcHElements();
		void CalcWeightMatrixOfNodes();
		void CalcResistivityOfNodes();
		void CalcRhoNCrossRotationHdsFaces(bool isInternal);
		void AssociationPropertiesToElements();
		void MakeGlobalMatrix(int loop, double omega);
		void Solve();
		void CalcE();
		void CalcZ();
		void ClearEAndH();
		void MarkBoundaryElements();
		void CalcResistivityAtTransitionElements();
	};
}#pragma once
