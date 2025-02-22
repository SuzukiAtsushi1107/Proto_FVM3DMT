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
#include "Node.h"
#include "pch.h"
#include "Element.h"
#include "Face.h"


Node::Node::Node() {
	x = Eigen::Vector3d::Zero();
	ID = -1;
}


void Node::Node::CalcH(int numOfElements) {
	numOfElementsBelongToThisNode = elements.size();
	Hcoeff.resize(3, 3 * numOfElements); //num of freedom is correspond to the num of elements
	Hcoeff.reserve(Eigen::VectorXi::Constant(3,30));

	bool isAirElement = false;
	bool isNotAirElement = false;
	for (auto element = elements.begin(); element != elements.end(); ++element) {
		if (element->second->property->type == Property::Property::AIR) {
			isAirElement = true;
		}
		else {
			isNotAirElement = true;
		}
	}

	if (isAirElement == true && isNotAirElement == true) { //空気地面境界面に位置するノードにマーク
		isAirGroundBoundary = true;
	}

	//bool isAirElement = false;
	//bool isNotAirElement = false;
	//for (auto element = elements.begin(); element != elements.end(); ++element) {
	//	if (element->second->property->type == Property::Property::AIR) {
	//		isAirElement = true;
	//	}
	//	else {
	//		isNotAirElement = true;
	//	}
	//}

	//Eigen::VectorXd dl{ elements.size() };
	//dl.setZero();
	//int i = 0;
	//for (auto element = elements.begin(); element != elements.end(); ++element) {

	//	//if (isAirElement == true && isNotAirElement == true) {
	//	//	if (element->second->property->type != Property::Property::AIR) {
	//	//		continue;
	//	//	}
	//	//}
	//	
	//	Eigen::Vector3d distance = element->second->centerCoord - x;
	//	dl(i) = 1.0 / distance.norm();
	//	//reciprocal of distance from this node to the center of the element.
	//	i++;
	//}
	//double sumDist = dl.sum();

	//dl = dl / sumDist;

	//i = 0;

	//for (auto itr = elements.begin(); itr != elements.end(); ++itr) {
	//	Element::Element* element = itr->second;
	//	//if (isAirElement == true && isNotAirElement == true) {
	//	//	if (element->property->type != Property::Property::AIR) {
	//	//		continue;
	//	//	}
	//	//}
	//	int elemID = element->ID;

	//	
	//	Hcoeff.insert(0, 3 * elemID) = dl(i);

	//	Hcoeff.insert(1, 3 * elemID + 1) = dl(i);

	//	Hcoeff.insert(2, 3 * elemID + 2) = dl(i);
	//	i++;
	//}

	std::vector<Eigen::Vector3d> xVector;
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); ++itr) {
		Element::Element* element = *itr;
		xVector.push_back(element->centerCoord);
	}

	std::vector<double> weight = CalcWeight(xVector, x);

	int i = 0;
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); ++itr) {
		Element::Element* element = *itr;
		int elemID = element->ID;
		Hcoeff.insert(0, 3 * elemID ) = weight[i];
		Hcoeff.insert(1, 3 * elemID + 1) = weight[i];
		Hcoeff.insert(2, 3 * elemID + 2) = weight[i];
		i++;
	}
	Hcoeff.makeCompressed();
	Hcoeff.prune(1e-9);
	Hcoeff.data().squeeze();
}

void Node::Node::CalcResistivity(int numOfElements) {
	numOfElementsBelongToThisNode = elements.size();




	//bool isAirElement = false;
	//bool isNotAirElement = false;
	//for (auto element = elements.begin(); element != elements.end(); ++element) {
	//	if (element->second->property->type == Property::Property::AIR) {
	//		isAirElement = true;
	//	}
	//	else {
	//		isNotAirElement = true;
	//	}
	//}

	Eigen::VectorXd dl{ elements.size() };
	dl.setZero();
	int i = 0;
	for (auto element = elements.begin(); element != elements.end(); ++element) {

		//if (isAirElement == true && isNotAirElement == true) {
		//	if (element->second->property->type == Property::Property::AIR) {
		//		continue;
		//	}
		//}

		Eigen::Vector3d distance = element->second->centerCoord - x;
		dl(i) = 1.0 / distance.norm();
		//reciprocal of distance from this node to the center of the element.
		i++;
	}
	double sumDist = dl.sum();

	dl = dl / sumDist;

	i = 0;
	resistivity.setZero();
	for (auto itr = elements.begin(); itr != elements.end(); ++itr) {
		Element::Element* element = itr->second;
		//if (isAirElement == true && isNotAirElement == true) {
		//	if (element->property->type == Property::Property::AIR) {
		//		continue;
		//	}
		//}
		int elemID = element->ID;

		resistivity += dl.coeff(i)*element->resistivity;


		i++;
	}

}
std::vector<double> Node::Node::CalcWeight(std::vector<Eigen::Vector3d> xVector, Eigen::Vector3d x0) {

	double xFactor = x0.coeff(0);
	double yFactor = x0.coeff(1);
	double zFactor = x0.coeff(2);
	for (auto itr = xVector.begin(); itr != xVector.end(); itr++) {
		Eigen::Vector3d ix = *itr;
		ix.coeffRef(0) = ix.coeff(0) / xFactor;
		ix.coeffRef(1) = ix.coeff(1) / yFactor;
		ix.coeffRef(2) = ix.coeff(2) / zFactor;

	}
	x0.coeffRef(0) = 1;
	x0.coeffRef(1) = 1;
	x0.coeffRef(2) = 1;
	Eigen::Matrix3d I;
	I.setZero();
	Eigen::Vector3d R;
	R.setZero();


	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			for (int ii = 0; ii < xVector.size(); ii++) {
				I.coeffRef(i, j) += (xVector[ii].coeff(i) - x0.coeff(i))*(xVector[ii].coeff(j) - x0.coeff(j));
			}
			
			
		}
	}
	for (int i = 0; i < 3; i++) {
		for (int ii = 0; ii < xVector.size(); ii++) {
			R.coeffRef(i) += (xVector[ii].coeff(i) - x0.coeff(i));
		}
	}
	double detA = I.coeff(0, 0) * I.coeff(1, 1) * I.coeff(2, 2) + 2 * I.coeff(0, 1) * I.coeff(1, 2) * I.coeff(2, 0) - std::pow(I.coeff(0, 2), 2.0)*I.coeff(1, 1) - std::pow(I.coeff(0, 1), 2.0)*I.coeff(2, 2) - std::pow(I.coeff(1, 2), 2.0)*I.coeff(0, 0);

	Eigen::Vector3d lamda;
	lamda.setZero();
	int i = 0;
	int j = 1;
	int k = 2;
	lamda.coeffRef(i) = -2 / detA * ((I.coeff(j, j) * I.coeff(k, k) - std::pow(I.coeff(j, k), 2.0))*R.coeff(i) + (-I.coeff(i, j) * I.coeff(k, k) + I.coeff(i, k) * I.coeff(j, k))*R.coeff(j) + (I.coeff(i, j) * I.coeff(j, k) - I.coeff(i, k) * I.coeff(j, j))*R.coeff(k));

	i = 1;
	j = 2;
	k = 0;
	lamda.coeffRef(i) = -2 / detA * ((I.coeff(j, j) * I.coeff(k, k) - std::pow(I.coeff(j, k), 2.0))*R.coeff(i) + (-I.coeff(i, j) * I.coeff(k, k) + I.coeff(i, k) * I.coeff(j, k))*R.coeff(j) + (I.coeff(i, j) * I.coeff(j, k) - I.coeff(i, k) * I.coeff(j, j))*R.coeff(k));
	
	i = 2;
	j = 0;
	k = 1;
	lamda.coeffRef(i) = -2 / detA * ((I.coeff(j, j) * I.coeff(k, k) - std::pow(I.coeff(j, k), 2.0))*R.coeff(i) + (-I.coeff(i, j) * I.coeff(k, k) + I.coeff(i, k) * I.coeff(j, k))*R.coeff(j) + (I.coeff(i, j) * I.coeff(j, k) - I.coeff(i, k) * I.coeff(j, j))*R.coeff(k));

	std::vector<double> w;
	w.resize(xVector.size());
	double wSum = 0;
	for (i = 0; i < xVector.size(); i++) {
		w[i] = 1 + 0.5*(lamda.coeff(0) * (xVector[i].coeff(0) - x0.coeff(0)) + lamda.coeff(1) * (xVector[i].coeff(1) - x0.coeff(1)) + lamda.coeff(2) * (xVector[i].coeff(2) - x0.coeff(2)));
		wSum += w[i];
	}
	bool key = false;
	for (i = 0; i < xVector.size(); i++) {
		w[i] = w[i] / wSum;
		if (isnan(w[i])) {
			key = true;
		}
	}
	if (key) {
		w = CalcWeightByDistance(xVector, x0);
	}
	return w;
}
std::vector<double> Node::Node::CalcWeightByDistance(std::vector<Eigen::Vector3d> xVector, Eigen::Vector3d x0) {
	std::vector<double> dl;
	dl.resize(xVector.size());
	for (int i = 0; i<xVector.size(); i++) {

	
		Eigen::Vector3d distance = xVector[i] - x0;
		dl[i] = 1.0 / distance.norm();
		//reciprocal of distance from this node to the center of the element.

	}
	double sumDist = 0.0;
	for (int i = 0; i < xVector.size(); i++) {

		sumDist += dl[i];
	}
	for (int i = 0; i < xVector.size(); i++) {

		dl[i] = dl[i] / sumDist;
	}

	return dl;
}