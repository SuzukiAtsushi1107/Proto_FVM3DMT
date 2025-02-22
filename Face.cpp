/*
Copyright c 2025 Suzuki Atsushi <mk.pn14951011 at gmail.com>
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
#include "Element.h"
#include "Face.h"
#include "Node.h"
#include <omp.h>



Face::Face::Face() {
	resistivity(0, 0) = -1.0;
	resistivity(1, 1) = -1.0;
	resistivity(2, 2) = -1.0;
}

void Face::Face::MarkBoundary() {
	if (elements.size() == 1) {
		isBoundary = true;
		double tol = 1e-6;

		Element::Element* element = elements.begin()->second;
		Eigen::Vector3d nVecOrthogonalToSurface;
		Eigen::Vector3d elemCenterToFaceCenter = centerCoord - element->centerCoord;

		Eigen::Vector3d vec1 = nodes[nodesID[1]]->x - nodes[nodesID[0]]->x;
		Eigen::Vector3d vec2 = nodes[nodesID[3]]->x - nodes[nodesID[0]]->x;
		Eigen::Vector3d tmpNVec = vec1.cross(vec2);
		tmpNVec = tmpNVec / tmpNVec.norm();

		double val = tmpNVec.dot(elemCenterToFaceCenter);
		if (val >= 0.0) {//correspond
			nVecOrthogonalToSurface = tmpNVec; //セルの外向きになるように修正
		}
		else { //not correspond
			nVecOrthogonalToSurface = -tmpNVec;
		}

		Eigen::Vector3d tmpVec;
		tmpVec.coeffRef(0) = 1.;
		tmpVec.coeffRef(1) = 0.;
		tmpVec.coeffRef(2) = 0.; //X軸
		if (abs(nVecOrthogonalToSurface.dot(tmpVec)) < 1 + tol && abs(nVecOrthogonalToSurface.dot(tmpVec)) > 1 - tol) {
			whichBoundary = X_BOUN;
			return;
		}

		tmpVec.coeffRef(0) = 0.;
		tmpVec.coeffRef(1) = 1.;
		tmpVec.coeffRef(2) = 0.; //Y軸
		if (abs(nVecOrthogonalToSurface.dot(tmpVec)) < 1 + tol && abs(nVecOrthogonalToSurface.dot(tmpVec)) > 1 - tol) {
			whichBoundary = Y_BOUN;
			return;
		}

		tmpVec.coeffRef(0) = 0.;
		tmpVec.coeffRef(1) = 0.;
		tmpVec.coeffRef(2) = 1.; //Z軸
		if (abs(nVecOrthogonalToSurface.dot(tmpVec)) < 1 + tol && abs(nVecOrthogonalToSurface.dot(tmpVec)) > 1 - tol) {
			whichBoundary = Z_BOUN;
			return;
		}
		else {
			std::cout << "No Match Boundary, Your Data May have Boundaries not orthogonal x,y,z.";
			exit(1);
		}


	}
	else if (elements.size() < 1 || elements.size() > 2) {
		std::cout << ("The Face Data is wrong because a face has more than two elements or less than one") << std::endl;
		exit(1);
	}
	else {
		isBoundary = false;
	}
}
void Face::Face::CalcArea() {
	std::vector<Node::Node*> vectorNodes;
	for (auto itr = nodesID.begin(); itr != nodesID.end(); itr++) {
		vectorNodes.push_back(nodes[*itr]);
	}
	Eigen::Vector3d G = 0.25*(vectorNodes[0]->x + vectorNodes[1]->x + vectorNodes[2]->x + vectorNodes[3]->x);
	centerCoord = G; //define the center point.
	//calc by dividing quad to two triangles and using 0.5*sqrt(| vec1 | ^ 2 * | vec2 | ^ 2 - (vec1 dot vec2) ^ 2)
	Eigen::Vector3d vec1 = nodes[nodesID[0]]->x - nodes[nodesID[3]]->x;
	Eigen::Vector3d vec2 = nodes[nodesID[1]]->x - nodes[nodesID[3]]->x;
	double ds1 = std::pow(vec1.norm(), 2.0)*std::pow(vec2.norm(), 2.0);
	ds1 -= pow(vec1.dot(vec2), 2.0);
	ds1 = 0.5*std::sqrt(ds1);

	vec1 = nodes[nodesID[1]]->x - nodes[nodesID[3]]->x;
	vec2 = nodes[nodesID[2]]->x - nodes[nodesID[3]]->x;
	double ds2 = std::pow(vec1.norm(), 2.0)*std::pow(vec2.norm(), 2.0);
	ds2 -= pow(vec1.dot(vec2), 2.0);
	ds2 = 0.5*std::sqrt(ds2);

	ds = ds1 + ds2;
}







void Face::Face::CalcRhoNCrossRotationHds(int numOfElements, int Jpolarization) {

	int maxMatrixSize = 0;
	for (auto itr = nodes.begin(); itr != nodes.end(); itr++) {
		maxMatrixSize += itr->second->numOfElementsBelongToThisNode * 3 * 3; //取りうる最大の疎行列要素数　3*3なのは異方性があるため
	}

	std::vector<Element::Element*> vectorElems;
	for (auto itr = elements.begin(); itr != elements.end(); itr++) {
		vectorElems.push_back(itr->second);
	}

	std::vector<Node::Node*> vectorNodes;
	for (auto itr = nodesID.begin(); itr != nodesID.end(); itr++) {
		vectorNodes.push_back(nodes[*itr]);
	}

	//-------------幾何計算------------------------------
	Eigen::Vector3d	dXvec = 0.5* (vectorNodes[1]->x + vectorNodes[2]->x)
		- 0.5*(vectorNodes[0]->x + vectorNodes[3]->x);
	double dX = dXvec.norm();

	// it is needed to correct the direction Y because Xvec and Yvec are not always orthogonal,
	// center of Face : G
	// four vortex of faces : A, B, C, D(clockwise ore anticlockwise)
	// (AGvec-n*ABvec)dotXvec=0 is needed.
	// so nABvec dot Xvec = AGvec dot Xvec
	// so we can obtain "n", which can be weight.
	// DCvec is the same.
	//Eigen::Vector3d G = 0.25*(vectorNodes[0]->x + vectorNodes[1]->x + vectorNodes[2]->x + vectorNodes[3]->x);
	Eigen::Vector3d G =centerCoord;

	Eigen::Vector3d	ABvec = vectorNodes[1]->x - vectorNodes[0]->x;
	Eigen::Vector3d	AGvec = G - vectorNodes[0]->x;
	double n = AGvec.dot(dXvec) / ABvec.dot(dXvec);

	Eigen::Vector3d	DCvec = vectorNodes[2]->x - vectorNodes[3]->x;
	Eigen::Vector3d	DGvec = G - vectorNodes[3]->x;
	double m = DGvec.dot(dXvec) / DCvec.dot(dXvec);
	Eigen::Vector3d	dYvec = (m*vectorNodes[2]->x + (1 - m)*vectorNodes[3]->x) -
		((1 - n)*vectorNodes[0]->x + n * vectorNodes[1]->x);
	double dY = dYvec.norm();

	eXVec = dXvec / dXvec.norm();
	eYVec = dYvec / dYvec.norm();
	Eigen::Vector3d eNvec = eXVec.cross(eYVec);

	Eigen::Vector3d dlVec;
	double dl;
	if (isBoundary == false) {
		dlVec = vectorElems[0]->centerCoord - vectorElems[1]->centerCoord;
		dl = dlVec.norm();
	}
	else {
		dlVec = vectorElems[0]->centerCoord - centerCoord;
		dl = std::abs(dlVec.dot(eNvec)); //境界面では∂/∂l=∂/∂nになるようにする。
		dlVec = 2 * dl * eNvec;
	}

	Eigen::Vector3d eLvec = dlVec / dlVec.norm();


	nVec = eNvec;	//direction of nVec changes due to which element we consider,
			// so this nVec is used to check whether this gradient is correspond the direction of nVec of element we consider
			//when we assemble the global matrix.
			//gradientMatrixLocal = scipy.sparse.lil_matrix((3, numOfElements))
	//-----------------------------------------------

	//----Elem to Elemのローカル座標系での微分値を計算
	std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorElemsHLocal; //ローカル座標系の向きにHcoeffとRhoJに変換するために使用
	vectorElemsHLocal.reserve(2);
	Eigen::SparseMatrix<double, Eigen::RowMajor> localXdirection{ 1,3 * numOfElements };
	//localXdirection.reserve(Eigen::VectorXi::Constant(1,maxMatrixSize));
	localXdirection.reserve(maxMatrixSize);
	Eigen::SparseMatrix<double, Eigen::RowMajor> localYdirection{ 1,3 * numOfElements };
	//localYdirection.reserve(Eigen::VectorXi::Constant(1, maxMatrixSize));
	localYdirection.reserve(maxMatrixSize);
	Eigen::SparseMatrix<double, Eigen::RowMajor> localNdirection{ 1,3 * numOfElements };
	//localNdirection.reserve(Eigen::VectorXi::Constant(1, maxMatrixSize));
	localNdirection.reserve(maxMatrixSize);
	for (auto itr = elements.begin(); itr != elements.end(); itr++) {
		localXdirection.row(0) = (eXVec.coeff(0)*itr->second->Hcoeff.row(0) + eXVec.coeff(1)*itr->second->Hcoeff.row(1) + eXVec.coeff(2)*itr->second->Hcoeff.row(2));
		localYdirection.row(0) = (eYVec.coeff(0)*itr->second->Hcoeff.row(0) + eYVec.coeff(1)*itr->second->Hcoeff.row(1) + eYVec.coeff(2)*itr->second->Hcoeff.row(2));
		localNdirection.row(0) = (eNvec.coeff(0)*itr->second->Hcoeff.row(0) + eNvec.coeff(1)*itr->second->Hcoeff.row(1) + eNvec.coeff(2)*itr->second->Hcoeff.row(2));
		Eigen::SparseMatrix<double, Eigen::RowMajor> localH{ 3,3 * numOfElements };
		//localH.reserve((Eigen::VectorXi::Constant(3, maxMatrixSize)));
		localH.reserve(3*maxMatrixSize);
		localH.row(0) = localXdirection.row(0);
		localH.row(1) = localYdirection.row(0);
		localH.row(2) = localNdirection.row(0);
		localH.makeCompressed();
		localH.prune(1e-9);
		localH.data().squeeze();
		vectorElemsHLocal.push_back(localH);
	}
	localXdirection.setZero();
	localYdirection.setZero();
	localNdirection.setZero();
	Eigen::SparseMatrix<double, Eigen::RowMajor> gradientHElemToElem{ 3,3 * numOfElements };
	//gradientHElemToElem.reserve((Eigen::VectorXi::Constant(3, maxMatrixSize)));
	gradientHElemToElem.reserve(3*maxMatrixSize);
	//Eigen::SparseMatrix<double, Eigen::RowMajor> gradientRhoJElemToElem{ 3,3 * numOfElements };
	//gradientRhoJElemToElem.reserve((Eigen::VectorXi::Constant(3, maxMatrixSize)));


	//bool isAir = false;
	//bool isNotAir = false;
	//for (auto itr = elements.begin(); itr != elements.end(); itr++) {
	//	if (itr->second->property->type == Property::Property::AIR) {
	//		isAir = true;
	//	}
	//	else {
	//		isNotAir = true;
	//	}
	//}	//bool isAir = false;
	//bool isNotAir = false;
	//for (auto itr = elements.begin(); itr != elements.end(); itr++) {
	//	if (itr->second->property->type == Property::Property::AIR) {
	//		isAir = true;
	//	}
	//	else {
	//		isNotAir = true;
	//	}
	//}

	if (isBoundary == false) {
		//if (isAir == true && isNotAir == true) {
		//	if (vectorElems[0]->property->type == Property::Property::AIR) {
		//		dl = (vectorElems[1]->centerCoord - centerCoord).norm();
		//		//境界上ではAIRの値
		//	}
		//	else {
		//		dl = (vectorElems[0]->centerCoord - centerCoord).norm();
		//		//境界上ではAIRの値
		//	}
		//}
		gradientHElemToElem = (vectorElemsHLocal[0] - vectorElemsHLocal[1]) / dl;
	}
	else {
		//∂/∂n=0


		if (whichBoundary == X_BOUN) {
			if (Jpolarization == 0) { //X方向
				for (int i = 0; i < 3; i++) {
					if (i < 2) {
						gradientHElemToElem.row(i) = (vectorElemsHLocal[0].row(i)) / 2./dl; //平行成分は境界で0
																						   //範囲外に仮想セルを設けたとして考える。
					}
				}
			}
			else { //Y方向
				for (int i = 0; i < 3; i++) {
					if (i == 2) {
						gradientHElemToElem.row(i) = (vectorElemsHLocal[0].row(i))/2 / dl; //	垂直成分は境界で0
																						 //範囲外に仮想セルを設けたとして考える。

					}
				}
			}
		}
		else if (whichBoundary == Y_BOUN) {
			if (Jpolarization == 0) { //X方向
				for (int i = 0; i < 3; i++) {
					if (i == 2) {
						gradientHElemToElem.row(i) = (vectorElemsHLocal[0].row(i))/2 / dl; //垂直成分は境界で0
					}
				}
			}
			else { //Y方向
				for (int i = 0; i < 3; i++) {
					if (i < 2) {
						gradientHElemToElem.row(i) = (vectorElemsHLocal[0].row(i))/2 / dl; //	平行成分は境界で0
					}
				}
			}
		}
		else if (whichBoundary == Z_BOUN) {
			for (int i = 0; i < 3; i++) {
				//if (i < 2) {
				gradientHElemToElem.row(i) = (vectorElemsHLocal[0].row(i))/2 / dl; //全成分は境界で0
			}
		}
	}
	gradientHElemToElem.makeCompressed();
	gradientHElemToElem.prune(1e-9);
	gradientHElemToElem.data().squeeze();
	//------ローカル座標系でのX方向の微分値を計算-------
	std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorNodesHLocal; //ローカル座標系の向きにHcoeffとRhoJに変換するために使用
	for (auto itr = nodesID.begin(); itr != nodesID.end(); itr++) { //nodesIDは幾何的な順番（時計or反時計回り）になっているのでこれを利用
		Node::Node* node = nodes[*itr];
		localXdirection.row(0) = (eXVec.coeff(0)*node->Hcoeff.row(0) + eXVec.coeff(1)*node->Hcoeff.row(1) + eXVec.coeff(2)*node->Hcoeff.row(2));
		localYdirection.row(0) = (eYVec.coeff(0)*node->Hcoeff.row(0) + eYVec.coeff(1)*node->Hcoeff.row(1) + eYVec.coeff(2)*node->Hcoeff.row(2));
		localNdirection.row(0) = (eNvec.coeff(0)*node->Hcoeff.row(0) + eNvec.coeff(1)*node->Hcoeff.row(1) + eNvec.coeff(2)*node->Hcoeff.row(2));
		Eigen::SparseMatrix<double, Eigen::RowMajor> localH{ 3,3 * numOfElements };
		//localH.reserve(Eigen::VectorXi::Constant(3, maxMatrixSize));
		localH.reserve(3*maxMatrixSize);
		localH.row(0) = localXdirection;
		localH.row(1) = localYdirection;
		localH.row(2) = localNdirection;
		localH.makeCompressed();
		localH.prune(1e-9);
		localH.data().squeeze();
		vectorNodesHLocal.push_back(localH);

	}
	Eigen::SparseMatrix<double, Eigen::RowMajor> gradientHLocalX{ 3, 3 * numOfElements };
	//gradientHLocalX.reserve(Eigen::VectorXi::Constant(3, maxMatrixSize));
	gradientHLocalX.reserve(3*maxMatrixSize);
	if (isBoundary == false) {

		gradientHLocalX = 0.5*(vectorNodesHLocal[1] + vectorNodesHLocal[2]) / dX
			- 0.5*(vectorNodesHLocal[0] + vectorNodesHLocal[3]) / dX;
	}
	else {
		if (whichBoundary == X_BOUN) {
			if (Jpolarization == 0) { //X方向
				//垂直方向の成分のみ境界上で値を持つため、垂直方向成分のみ計算
				//それ以外は境界面で0より勾配は0
				for (int i = 0; i < 3; i++) {
					if (i == 2) {

						gradientHLocalX.row(i) = 0.5* (vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
							- 0.5*(vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;

					}
					else {
						gradientHLocalX.row(i) = 0.25* (vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
							- 0.25*(vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;

						//範囲外に仮想セル中心（必ずしもセルでなくてもよく、物理量を置く中心だけあればよい）を置く。
						//この点は境界セルから境界面に垂直におろし、かつ距離も等しくなる点に置く。
						//この置き方はelemToelemで計算した時も同様にしている。
						//すると、これら2点とそれぞれの境界面での節点は二等辺三角形を作るので、必ずこの節点までの距離は等しくなる。
						//また、この範囲外の仮想セル中心点での値は、ここで計算する成分は0となる。
						//よって境界面上の節点ではもともとの計算値を半分にすればよい。

					}
				}
			}
			else { //Y方向
				//平行方向の成分のみ境界上で値を持つため、平行方向成分のみ計算
				//それ以外は境界面で0より勾配は0
				for (int i = 0; i < 3; i++) {
					if (i < 2) {

						gradientHLocalX.row(i) = 0.5*(vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
							- 0.5* (vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;
					}
					else {
						gradientHLocalX.row(i) = 0.25*(vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
							- 0.25* (vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;

					}
				}
			}
		}
		if (whichBoundary == Y_BOUN) {
			if (Jpolarization == 0) { //X方向
				//平行方向の成分のみ境界上で値を持つため、平行方向成分のみ計算
				//それ以外は境界面で0より勾配は0
				for (int i = 0; i < 3; i++) {
					if (i < 2) {

						gradientHLocalX.row(i) = 0.5* (vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
							- 0.5*(vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;

					}
					else {

						gradientHLocalX.row(i) = 0.25* (vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
							- 0.25*(vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;

					}
				}
			}
			else { //Y方向
				//垂直方向の成分のみ境界上で値を持つため、垂直方向成分のみ計算
				//それ以外は境界面で0より勾配は0
				for (int i = 0; i < 3; i++) {
					if (i == 2) {

						gradientHLocalX.row(i) = 0.5* (vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
							- 0.5* (vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;

					}
					else {

						gradientHLocalX.row(i) = 0.25* (vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
							- 0.25* (vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;
					}
				}
			}
		}
		if (whichBoundary == Z_BOUN) {
			//全成分0
			for (int i = 0; i < 3; i++) {
				//if (i == 2) {

				//	gradientHLocalX.row(i) = 0.5* (vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
				//		- 0.5* (vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;

				//	gradientRhoJLocalX.row(i) = 0.25*(vectorNodesRhoJLocal[1].row(i) + vectorNodesRhoJLocal[2].row(i)) / dX
				//		- 0.25*(vectorNodesRhoJLocal[0].row(i) + vectorNodesRhoJLocal[3].row(i)) / dX;//Hcoeffの鉛直成分が連続→ρは0よりrhoJは0
				//}
				//else {
					gradientHLocalX.row(i) = 0.25* (vectorNodesHLocal[1].row(i) + vectorNodesHLocal[2].row(i)) / dX
						- 0.25* (vectorNodesHLocal[0].row(i) + vectorNodesHLocal[3].row(i)) / dX;
				//}

			}
		}

	}
	gradientHLocalX.makeCompressed();
	gradientHLocalX.prune(1e-9);
	gradientHLocalX.data().squeeze();


	//------ローカル座標系でのY方向の微分値を計算-------
	Eigen::SparseMatrix<double, Eigen::RowMajor> gradientHLocalY{ 3, 3 * numOfElements };
	//gradientHLocalY.reserve(Eigen::VectorXi::Constant(3, maxMatrixSize));
	gradientHLocalY.reserve(3*maxMatrixSize);
	if (isBoundary == false) {
		gradientHLocalY = (m*vectorNodesHLocal[2] + (1 - m)*vectorNodesHLocal[3]) / dY -
			((1 - n)*vectorNodesHLocal[0] + n * vectorNodesHLocal[1]) / dY;
	}
	else {
		//gradientHLocalY = (m*vectorNodesHLocal[2] + (1 - m)*vectorNodesHLocal[3]) / dY -
		//	((1 - n)*vectorNodesHLocal[0] + n * vectorNodesHLocal[1]) / dY;
		//gradientRhoJLocalY = (m*vectorNodesRhoJLocal[2] + (1 - m)*vectorNodesRhoJLocal[3]) / dY -
		//	((1 - n)*vectorNodesRhoJLocal[0] + n * vectorNodesRhoJLocal[1]) / dY;
		////∂/∂n=0


		if (whichBoundary == X_BOUN) {
			if (Jpolarization == 0) { //X方向
				//垂直方向の成分のみ境界上で値を持つため、垂直方向成分のみ計算
				//それ以外は境界面で0より勾配は0
				for (int i = 0; i < 3; i++) {
					if (i == 2) {


						gradientHLocalY.row(i) = (m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
							((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
					}
					else {
						gradientHLocalY.row(i) =0.5* (m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
							0.5*((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
					}
				}
			}
			else { //Y方向
				//平行方向の成分のみ境界上で値を持つため、平行方向成分のみ計算
				//それ以外は境界面で0より勾配は0
				for (int i = 0; i < 3; i++) {
					if (i < 2) {

						gradientHLocalY.row(i) = (m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
							((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
					}
					else {
						gradientHLocalY.row(i) =0.5* (m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
							0.5*((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
					}
				}
			}
		}
		else if (whichBoundary == Y_BOUN) {
			if (Jpolarization == 0) { //X方向
				//平行方向の成分のみ境界上で値を持つため、平行方向成分のみ計算
				//それ以外は境界面で0より勾配は0
				for (int i = 0; i < 3; i++) {
					if (i < 2) {

						gradientHLocalY.row(i) = (m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
							((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
					}
					else {
						gradientHLocalY.row(i) = 0.5*(m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
							0.5*((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
					}
				}
			}
			else { //Y方向
				//垂直方向の成分のみ境界上で値を持つため、垂直方向成分のみ計算
				//それ以外は境界面で0より勾配は0
				for (int i = 0; i < 3; i++) {
					if (i == 2) {
						gradientHLocalY.row(i) = (m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
							((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
					}
					else {
						gradientHLocalY.row(i) = 0.5*(m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
							0.5*((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
					}
				}
			}
		}
		else if (whichBoundary == Z_BOUN) {
			//全成分0
			for (int i = 0; i < 3; i++) {
			//	if (i == 2) {

			//		gradientHLocalY.row(i) = (m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
			//			((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
			//		gradientRhoJLocalX.row(i) = 0.5*(vectorNodesRhoJLocal[1].row(i) + vectorNodesRhoJLocal[2].row(i)) / dX
			//			- 0.5*(vectorNodesRhoJLocal[0].row(i) + vectorNodesRhoJLocal[3].row(i)) / dX;//Hcoeffの鉛直成分が連続→ρは0よりrhoJは0
			//	}
			//	else {
					gradientHLocalY.row(i) = 0.5*(m*vectorNodesHLocal[2].row(i) + (1 - m)*vectorNodesHLocal[3].row(i)) / dY -
						0.5*((1 - n)*vectorNodesHLocal[0].row(i) + n * vectorNodesHLocal[1].row(i)) / dY;
				//}
			}
		}
	}
	gradientHLocalY.makeCompressed();
	gradientHLocalY.prune(1e-9);
	gradientHLocalY.data().squeeze();

	Eigen::SparseMatrix<double, Eigen::RowMajor> gradientHLocalN{ 3, 3 * numOfElements };
	//gradientHLocalN.reserve(Eigen::VectorXi::Constant(3, maxMatrixSize));
	gradientHLocalN.reserve(3*maxMatrixSize);
	if (isBoundary == false) {
		gradientHLocalN = 1.0 / eNvec.dot(eLvec)*gradientHElemToElem -
			eXVec.dot(eLvec) / eNvec.dot(eLvec) *gradientHLocalX -
			eYVec.dot(eLvec) / eNvec.dot(eLvec)*gradientHLocalY;

	}
	else {
		gradientHLocalN = gradientHElemToElem;
	}
	gradientHLocalN.makeCompressed();
	gradientHLocalN.prune(1e-9);
	gradientHLocalN.data().squeeze();
	//-------------------------RhoNCrossRotationHds--------------------------------------

	Eigen::SparseMatrix<double, Eigen::RowMajor > rotationH{ 3, 3 * numOfElements };
	//rotationH.reserve(Eigen::VectorXi::Constant(3, maxMatrixSize));
	rotationH.reserve(3 * maxMatrixSize);
	Eigen::SparseMatrix<double, Eigen::RowMajor> tmpRhoNCrossRotationHdSGlobal{ 3, 3 * numOfElements };
	//tmpRhoNCrossRotationHdSGlobal.reserve(Eigen::VectorXi::Constant(3, maxMatrixSize));
	tmpRhoNCrossRotationHdSGlobal.reserve(3 * maxMatrixSize);

	Eigen::SparseMatrix<double, Eigen::RowMajor > nCrossRhoRotationH{ 3, 3 * numOfElements };
	//nCrossRhoRotationH.reserve(Eigen::VectorXi::Constant(3,maxMatrixSize));
	nCrossRhoRotationH.reserve(3 * maxMatrixSize);

	for (auto itr = elements.begin(); itr != elements.end(); itr++) {
		Element::Element* element = itr->second;

		
		Eigen::Vector3d elemCenterToFaceCenter = centerCoord - element->centerCoord;
		double val = nVec.dot(elemCenterToFaceCenter);
		Eigen::Vector3d nVecTmp;
		Eigen::Vector3d eXVecTmp;
		Eigen::Vector3d eYVecTmp;
		bool isNeedRepairGradient = false;
		if (val >= 0.0) {//correspond
			eXVecTmp = eXVec;
			eYVecTmp = eYVec;
			nVecTmp = nVec;
		}
		else { //not correspond
			eXVecTmp = -eXVec;
			eYVecTmp = eYVec; //Y方向は変わらないとする
			nVecTmp = -nVec;
			gradientHLocalX.row(1) = -gradientHLocalX.row(1);
			gradientHLocalY.row(0) = -gradientHLocalY.row(0);
			gradientHLocalY.row(2) = -gradientHLocalY.row(2);
			gradientHLocalN.row(1) = -gradientHLocalN.row(1);
			isNeedRepairGradient = true;
			//これらの符号がかわる。
		}

		//以下で比抵抗行列の基底を変換する。
		Eigen::Matrix3d P;
		std::vector<Eigen::Vector3d> tmp;
		tmp.push_back(eXVecTmp);
		tmp.push_back(eYVecTmp);
		tmp.push_back(nVecTmp); 
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				Eigen::Vector3d globalUnitVec;
				globalUnitVec.setZero();
				globalUnitVec.coeffRef(j) = 1.0;

				P.coeffRef(i, j) = tmp[i].dot(globalUnitVec);
			}
		}
		Eigen::Matrix3d resistivityAverage;
		resistivityAverage.setZero();

		//resistivityAverage = resistivityAverage / (double (nodes.size()));
		//std::cout << resistivityAverage.coeff(0,0) << std::endl;
		double w1;
		double w2;
		double wSum;
		if (vectorElems.size() > 1) {
			w1 = 1.0 / (vectorElems[0]->centerCoord - centerCoord).norm();
			w2 = 1.0 / (vectorElems[1]->centerCoord - centerCoord).norm();
			wSum = w1 + w2;
			w1 = w1 / wSum;
			w2 = w2 / wSum;
		}
		if (vectorElems.size() > 1) {
			//resistivityAverage = (w1*(vectorElems[0]->property->resistivity.inverse()) +
			//	w2*(vectorElems[1]->property->resistivity.inverse())).inverse();
			//resistivityAverage = w1 * (vectorElems[0]->resistivity) +
			//	w2 * (vectorElems[1]->resistivity);




			if (vectorElems[0]->property->type == Property::Property::AIR &&
				vectorElems[1]->property->type != Property::Property::AIR) {
				resistivityAverage = vectorElems[1]->resistivity;
			}
			else if (vectorElems[1]->property->type == Property::Property::AIR &&
				vectorElems[0]->property->type != Property::Property::AIR) {
				resistivityAverage = vectorElems[0]->resistivity;
			}
			else {
				resistivityAverage = w1 * (vectorElems[0]->resistivity) +
					w2 * (vectorElems[1]->resistivity);
				//w1 = 0.5; w2 = 0.5;
				//resistivityAverage = w1 * (vectorElems[0]->resistivity) +
				//	w2 * (vectorElems[1]->resistivity);
			}
		}
		else {
			resistivityAverage = vectorElems[0]->resistivity;
		}
		Eigen::Matrix3d resistivityLocal;
		resistivityLocal.setZero();
		resistivityLocal = P * resistivityAverage*P.transpose();


		rotationH.row(0) = gradientHLocalY.row(2) - gradientHLocalN.row(1);
		rotationH.row(1) = gradientHLocalN.row(0) - gradientHLocalX.row(2);
		rotationH.row(2) = gradientHLocalX.row(1) - gradientHLocalY.row(0);
		rotationH.makeCompressed();
		rotationH.prune(1e-9);
		rotationH.data().squeeze();

		Eigen::SparseMatrix<double, Eigen::RowMajor > rhoRotationH{ 3, 3 * numOfElements };
		//rhoRotationH.reserve(Eigen::VectorXi::Constant(3, maxMatrixSize));
		rhoRotationH.reserve(3 * maxMatrixSize);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				rhoRotationH.row(i) += ds * resistivityLocal.coeff(i, j)*rotationH.row(j);
			}
		}
		rhoRotationH.makeCompressed();
		rhoRotationH.prune(1e-9);
		rhoRotationH.data().squeeze();
		nCrossRhoRotationH.row(0) = - 1*rhoRotationH.row(1);
		nCrossRhoRotationH.row(1) = 1*rhoRotationH.row(0);
		//ローカル座標系でみているので、n=(0,0,1)になる。
		nCrossRhoRotationH.makeCompressed();
		nCrossRhoRotationH.prune(1e-9);
		nCrossRhoRotationH.data().squeeze();

		tmpRhoNCrossRotationHdSGlobal.row(0) = eXVecTmp.coeff(0)*nCrossRhoRotationH.row(0) +
			eYVecTmp.coeff(0)*nCrossRhoRotationH.row(1) +
			nVecTmp.coeff(0)*nCrossRhoRotationH.row(2);
		tmpRhoNCrossRotationHdSGlobal.row(1) = eXVecTmp.coeff(1)*nCrossRhoRotationH.row(0) +
			eYVecTmp.coeff(1)*nCrossRhoRotationH.row(1) +
			nVecTmp.coeff(1)*nCrossRhoRotationH.row(2);
		tmpRhoNCrossRotationHdSGlobal.row(2) = eXVecTmp.coeff(2)*nCrossRhoRotationH.row(0) +
			eYVecTmp.coeff(2)*nCrossRhoRotationH.row(1) +
			nVecTmp.coeff(2)*nCrossRhoRotationH.row(2); //全体座標系に変換
		tmpRhoNCrossRotationHdSGlobal.makeCompressed();
		tmpRhoNCrossRotationHdSGlobal.prune(1e-9);
		tmpRhoNCrossRotationHdSGlobal.data().squeeze();
		rhoNCrossRotationHds[element->ID].resize(3, 3 * numOfElements);
		//rhoNCrossRotationHds[element->ID].reserve(Eigen::VectorXi::Constant(3, maxMatrixSize));
		rhoNCrossRotationHds[element->ID].reserve(3*maxMatrixSize);
		rhoNCrossRotationHds[element->ID] = tmpRhoNCrossRotationHdSGlobal;
		rhoNCrossRotationHds[element->ID].makeCompressed();
		rhoNCrossRotationHds[element->ID].prune(1e-9);
		rhoNCrossRotationHds[element->ID].data().squeeze();

		if (isNeedRepairGradient){//not correspond
			gradientHLocalX.row(1) = -gradientHLocalX.row(1);
			gradientHLocalY.row(0) = -gradientHLocalY.row(0);
			gradientHLocalY.row(2) = -gradientHLocalY.row(2);
			gradientHLocalN.row(1) = -gradientHLocalN.row(1);
			//符号もとに戻す
		}

	}

}

Eigen::SparseMatrix<double, Eigen::RowMajor> Face::Face::getRhoNCrossRotationHds(Element::Element* element) {
	return rhoNCrossRotationHds[element->ID];
}

bool Face::Face::CalcRhoCrossH(Eigen::MatrixXcd* A, Eigen::MatrixXcd* W,Eigen::VectorXcd* b, Element::Element* element, Eigen::VectorXcd* H,int iRow) {

	//bool key1 = false;
	//bool key2 = false;
	//for (auto itr = nodes.begin(); itr != nodes.end(); itr++) {
	//	if (itr->second->isAirGroundBoundary == true) {
	//		key1 = true;
	//	}
	//	else if (itr->second->isAirGroundBoundary == false) {
	//		key2 = true;
	//	}
	//}
	//if (key1 == false && key2 == false) {
	//	return true; //地中を計算しない
	//}
	//bool isAir = false;
	//bool isNotAir = false;
	//for (auto itr = elements.begin(); itr != elements.end(); itr++) {
	//	if (itr->second->property->type == Property::Property::AIR) {
	//		isAir = true;
	//	}
	//	else {
	//		isNotAir = true;
	//	}
	//}
	//if (isAir == true && isNotAir == true) {
	//	return true;
	//}

	Eigen::Vector3d nVecOrthogonalToSurface;
	Eigen::Vector3d elemCenterToFaceCenter = centerCoord - element->centerCoord;
	double val = nVec.dot(elemCenterToFaceCenter);
	if (val >= 0.0) {//correspond
		nVecOrthogonalToSurface = nVec;
	}
	else { //not correspond
		nVecOrthogonalToSurface = -nVec;
	}

	std::map<int, Eigen::Vector3d> dlVec;
	double sum = 0.0;
	Eigen::Vector3cd E;
	E.setZero();
	for (auto itr = elements.begin(); itr != elements.end(); itr++) {
		dlVec[itr->second->ID] = centerCoord - itr->second->centerCoord;
		sum += 1.0 / dlVec[itr->second->ID].norm();
	}
	Eigen::SparseMatrix<double, Eigen::RowMajor > rhoNCrossRotationH;
	Eigen::Vector3cd nCrossEds;
	nCrossEds = getRhoNCrossRotationHds(element)* (*H);
	for (int i = 0; i < 3; i++) {
		b->coeffRef(iRow+i) = nCrossEds.coeff(i)/ds;
	}
	A->coeffRef(iRow, 1) = -nVecOrthogonalToSurface.coeff(2);
	A->coeffRef(iRow, 2) = -nVecOrthogonalToSurface.coeff(1);
	A->coeffRef(iRow+1, 0) = nVecOrthogonalToSurface.coeff(2);
	A->coeffRef(iRow + 1, 2) = -nVecOrthogonalToSurface.coeff(0);
	A->coeffRef(iRow + 2, 0) = -nVecOrthogonalToSurface.coeff(1);
	A->coeffRef(iRow + 2, 1) = nVecOrthogonalToSurface.coeff(0);
	
	//weight
	W->coeffRef(iRow, iRow) = 1;
	W->coeffRef(iRow+1, iRow+1) = 1;
	W->coeffRef(iRow+2, iRow+2) = 1;

	return false;
}
//
//Eigen::Vector3cd Face::Face::CalcRhoNCrossHds(Element::Element* element) {
//
//	std::vector<Element::Element*> vectorElems;
//	for (auto itr = elements.begin(); itr != elements.end(); itr++) {
//		vectorElems.push_back(itr->second);
//	}
//
//	Eigen::Matrix3d resistivityAverage;
//	resistivityAverage.setZero();
//
//	double w1;
//	double w2;
//	double wSum;
//	if (vectorElems.size() > 1) {
//		w1 = 1.0 / (vectorElems[0]->centerCoord - centerCoord).norm();
//		w2 = 1.0 / (vectorElems[1]->centerCoord - centerCoord).norm();
//		wSum = w1 + w2;
//		w1 = w1 / wSum;
//		w2 = w2 / wSum;
//	}
//	if (vectorElems.size() > 1) {
//		//resistivityAverage = (w1*(vectorElems[0]->property->resistivity.inverse()) +
//		//	w2*(vectorElems[1]->property->resistivity.inverse())).inverse();
//		//resistivityAverage = w1 * (vectorElems[0]->property->resistivity) +
//		//	w2 * (vectorElems[1]->property->resistivity);
//		if (vectorElems[0]->property->type == Property::Property::AIR &&
//			vectorElems[1]->property->type != Property::Property::AIR) {
//			resistivityAverage = vectorElems[1]->property->resistivity;
//		}
//		else if (vectorElems[1]->property->type == Property::Property::AIR &&
//			vectorElems[0]->property->type != Property::Property::AIR) {
//			resistivityAverage = vectorElems[0]->property->resistivity;
//		}
//		else {
//			resistivityAverage = w1 * (vectorElems[0]->property->resistivity) +
//				w2 * (vectorElems[1]->property->resistivity);
//		}
//	}
//	else {
//		resistivityAverage = vectorElems[0]->property->resistivity;
//	}
//
//	Eigen::Vector3d nVecOrthogonalToSurface;
//	Eigen::Vector3d elemCenterToFaceCenter = centerCoord - element->centerCoord;
//	double val = nVec.dot(elemCenterToFaceCenter);
//	if (val >= 0.0) {//correspond
//		nVecOrthogonalToSurface = nVec;
//	}
//	else { //not correspond
//		nVecOrthogonalToSurface = -nVec;
//	}
//	Eigen::Vector3cd Have;
//	if (vectorElems.size() > 1) {
//		Have = w1 * vectorElems[0]->H.back() + w2 * vectorElems[1]->H.back();
//	}
//	else {
//		Have = vectorElems[0]->H.back();
//	}
//	Eigen::Vector3cd rhoNCrossHds;
//	
//	rhoNCrossHds = resistivityAverage * (nVecOrthogonalToSurface.cross(Have))*ds;
//	return rhoNCrossHds;
//}