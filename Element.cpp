/*
Copyright c 2025 Suzuki Atsushi <mk.pn14951011 at gmail.com>
For Non-commercial uses, this work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
For commercial uses, when you use this work until April 1, 2025, you must publish the original or derived source works you use as the terms of the Do What The Fuck You Want To Public License, Version 2.
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
#include "Functions.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <kv/autodif.hpp>
#include <kv/complex.hpp>
namespace ub = boost::numeric::ublas;
using namespace std;
void Element::Element::InitializeHAndEAndZ(int nOmega) {
	H.resize(2);
	E.resize(2);
	for (int ii = 0; ii < 2; ii++) {
		H[ii].setZero();
		E[ii].setZero();
	}
	//Set Z Vector Size

	Z.resize(nOmega);
	dZdH.resize(nOmega);
	dZdRho.resize(nOmega);
	for (int ii = 0; ii < nOmega; ii++) {
		Z[ii].resize(2, 2);
		dZdH[ii].resize(2, 2);
		dZdRho[ii].resize(2, 2);

	}

	//Set Tipper Vector Size
	T.resize(nOmega);
	dTdH.resize(nOmega);
	for (int ii = 0; ii < nOmega; ii++) {
		T[ii].resize(2);
		dTdH[ii].resize(2);
	}
	
}

void Element::Element::ClearHAndE() {
	for (int i = 0; i < H.size(); i++) {
		H[i].setZero();
	}
	//H.clear();
	//H.shrink_to_fit();
	for (int i = 0; i < E.size(); i++) {
		E[i].setZero();
	}
	//E.clear();
	//E.shrink_to_fit();
}
void Element::Element::ClearZ() {
	for (int i = 0; i < Z.size(); i++) {
		Z[i].setZero();
	}
	//Z.clear();
}
void Element::Element::SetNeighborElements(unordered_map<string, Element*> *elements) {
	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {
			for (int k = -1; k < 2; k++) {
				Eigen::Vector3i tmp;
				tmp.coeffRef(0) = i;
				tmp.coeffRef(1) = j;
				tmp.coeffRef(2) = k;
				string neighborID=Functions::GetNeighborElement(elements, this, tmp,nx,ny,nz);
				int ipos = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
				if (ipos >= 0 && ipos < 27) {
					alreadyFoundNeighborID[ipos] = neighborID; //登録
					if ((*elements).count(neighborID) != 0) {
						neighborElements[ipos] = (*elements)[neighborID];
					}
					else {
						boundary = neighborID;
					}
				}
			}
		}
	}
}
void Element::Element::CalcInterpolatedRhoInElementCoeff(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map < string, Element* >* elements, int numCalcElements, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* resisCoeff) {
	//val方向の隣接セル内の値を補間する。
	vector<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> > relatedElementsRho;
	vector < Eigen::Vector3d> relatedElementsCenterCoord;
	Eigen::Vector3i pos;
	pos.setZero();
	unordered_map<string, Element*> nodeElementsDict;
	nodeElementsDict.reserve(100); //100は適当
	string neighborIDVirtual = Functions::GetVirturalNeighborElement(elements, ID, layer, val, nx, ny, nz);
	for (int i = 0; i < 4; i++) {
		pos.setZero();
		if (i == 0)     pos[0] = -1;
		else if (i == 1)pos[0] = 1;
		else if (i == 2)pos[1] = -1;
		else if (i == 3)pos[1] = 1;

		string neighborID = Functions::GetNeighborElement(elements, neighborIDVirtual, layer, pos, nx, ny, nz);
		if (nodeElementsDict.count(neighborID) == 0) {
			Element* element = (*elements)[neighborID];
			nodeElementsDict[neighborID] = element;
			relatedElementsCenterCoord.push_back(element->centerCoord);
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>  rhotmp{ 1,numCalcElements };
			rhotmp.makeCompressed();
			rhotmp.reserve(100);
			element->CalcCenterResistivityCoeff(elements, numCalcElements,&rhotmp,1.0);
			relatedElementsRho.push_back(rhotmp);

		}
	}

	vector<double> w = Functions::CalcWeight(relatedElementsCenterCoord, x0);
	for (int i = 0; i < relatedElementsCenterCoord.size(); i++) {
		*resisCoeff += w[i] * relatedElementsRho[i];
	}
	return;

}
void Element::Element::CalcCenterResistivityCoeff(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* resisCoeff,double coeff)
{
	if (isParent == true) {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					(*elements)[childID]->CalcCenterResistivityCoeff(elements, numOfCalcElements,resisCoeff,coeff/4.0);
				
			}
		}
	}
	else {
		resisCoeff->coeffRef(0, calcID) += 1.0*coeff;
	}
	return;
}
void Element::Element::CalcSurfaceRelatedResistivityCoeff(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* resisCoeff, double coeff, Eigen::Vector3i pos)
{
	if (isParent == true) {
		if (pos.coeff(0) == -1) {
			int i = 0;
			for (int j = 0; j < 2; j++) {
				string childID = ID + Functions::GetBinaryValue(i, j);
				(*elements)[childID]->CalcSurfaceRelatedResistivityCoeff(elements, numOfCalcElements, resisCoeff, coeff / 2.0,pos);
			}
		}
		else if (pos.coeff(0) == 1) {
			int i = 1;
			for (int j = 0; j < 2; j++) {
				string childID = ID + Functions::GetBinaryValue(i, j);
				(*elements)[childID]->CalcSurfaceRelatedResistivityCoeff(elements, numOfCalcElements, resisCoeff, coeff / 2.0, pos);
			}
		}
		else if (pos.coeff(1) == -1) {
			int j = 0;
			for (int i = 0; i < 2; i++) {
				string childID = ID + Functions::GetBinaryValue(i, j);
				(*elements)[childID]->CalcSurfaceRelatedResistivityCoeff(elements, numOfCalcElements, resisCoeff, coeff / 2.0, pos);
			}
		}
		else if (pos.coeff(1) == 1) {
			int j = 1;
			for (int i = 0; i < 2; i++) {
				string childID = ID + Functions::GetBinaryValue(i, j);
				(*elements)[childID]->CalcSurfaceRelatedResistivityCoeff(elements, numOfCalcElements, resisCoeff, coeff / 2.0, pos);
			}
		}
		else {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					(*elements)[childID]->CalcSurfaceRelatedResistivityCoeff(elements, numOfCalcElements, resisCoeff, coeff / 4.0, pos);
				}
			}
		}
		
	}
	else {
		resisCoeff->coeffRef(0, calcID) += 1.0*coeff;
	}
	return;
}
void Element::Element::CalcSurfaceResistivity(unordered_map<string, Element*>* elements, vector<Element*>* calcElementsVector, int numOfCalcElements) {
	if (isAlreadyCalcResisCoeff==false) {
		
		resistivitySurface = new vector<double>(6);
		//for (int j = 0; j < 6; j++) {
		//	(*resistivitySurface)[j] = 0.0;
		//}

		resistivitySurfaceCoeff.resize(6);
		diffResistivitySurfaceCoeff.resize(6);
		for (int j = 0; j < 6; j++) {
			resistivitySurfaceCoeff[j] = new Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor >{1, numOfCalcElements };
			resistivitySurfaceCoeff[j]->reserve(Eigen::VectorXi::Constant(1, 100));
			diffResistivitySurfaceCoeff[j] = new Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor >{ 1, numOfCalcElements };
			diffResistivitySurfaceCoeff[j]->reserve(Eigen::VectorXi::Constant(1, 100));
			

		}

	}
	if (boundary == "NOT_BOUNDARY") {
		double dv = dx * dy*dz;

		for (int i = 0; i < 6; i++) {
			(*resistivitySurface)[i] = 0.0;
			double dS = 0;
			if (i == 0 || i == 1)dS = dy * dz;
			else if (i == 2 || i == 3)dS = dx * dz;
			else dS = dx * dy;
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> rhoCoeff{ 1,numOfCalcElements };
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> diffRhoCoeff{ 1,numOfCalcElements };
			//rhoCoeff.makeCompressed();
			//rhoCoeff.reserve(100);
			//harmonicOperationRhoCoeff.makeCompressed();
			//harmonicOperationRhoCoeff.reserve(100);
			rhoCoeff.reserve(Eigen::VectorXi::Constant(1, 100));
			diffRhoCoeff.reserve(Eigen::VectorXi::Constant(1, 100));
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>rho1Coeff{ 1,numOfCalcElements };
			Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>rho2Coeff{ 1,numOfCalcElements };
			rho1Coeff.reserve(Eigen::VectorXi::Constant(1, 100));
			rho2Coeff.reserve(Eigen::VectorXi::Constant(1, 100));

			
			Eigen::Vector3i pos;
			pos[0] = 0;
			pos[1] = 0;
			pos[2] = 0;
			if (i == 0) pos[0] = -1;
			else if (i == 1) pos[0] = 1;
			else if (i == 2) pos[1] = -1;
			else if (i == 3) pos[1] = 1;
			else if (i == 4) pos[2] = -1;
			else if (i == 5) pos[2] = 1;
			int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
			string neighborID = alreadyFoundNeighborID[ipos];
			//string neighborID = Functions::GetNeighborElement(*elements, this, pos);
			if (isAlreadyCalcResisCoeff == false) {
				if (i == 0 || i == 1) {

					rho1Coeff.setZero();
					rho2Coeff.setZero();
					

					// Calc Rho For Foward Calc
					if (layer == neighborElements[ipos]->layer) {
						double w1 = dx; //重みこれでよい？要検討
						double w2 = neighborElements[ipos]->dx;
						if (neighborElements[ipos]->isParent == true) {
							w2 = w2 / 2.0; //assumed that the layer difference is one,not more than one.
						}
						
						double wSum = w1 + w2;
						double _w1 = w2 / wSum;
						double _w2 = w1 / wSum;


						//rho = resistivity;
							
						CalcCenterResistivityCoeff(elements, numOfCalcElements,&rho1Coeff,1.0);
						neighborElements[ipos]->CalcSurfaceRelatedResistivityCoeff(elements, numOfCalcElements,&rho2Coeff,1.0,-1*pos);
						Functions::SetAtoResultCoef1DotBPlusCoef2DotC(&rhoCoeff, &rho1Coeff, &rho2Coeff, _w1, _w2);
						//rhoCoeff = Functions::DotConst( _w1 , &rho1Coeff) + Functions::DotConst(_w2 , &rho2Coeff);
						//if (i == 0) {
						//	rhoCoeff = rho2Coeff; //minus surface
						//}
						//else {
						//	rhoCoeff = rho1Coeff; //plus surface
						//}

						//harmonicOperationRhoCoeff = (-rho1Coeff + rho2Coeff);// *dS / dv * (*modelParam)[0]; //followed by Usui san method
							
						//double dl = (w1 + w2) / 2;
						//harmonicOperationRhoCoeff = (-rho1Coeff + rho2Coeff)/dl  * (*modelParam)[0]; 
						

					}
					else if (layer > neighborElements[ipos]->layer) {
						int j;
						if (i == 0) { j = 1; }
						else { j = 0; }
						Functions::PlusEqual(&rhoCoeff, neighborElements[ipos]->resistivitySurfaceCoeff[j], 1.0);

						//Eigen::Vector3d x0 = centerCoord;
						//if (i == 0) { x0.coeffRef(0) += -dx; }
						//else { x0.coeffRef(0) += +dx; }
						//CalcCenterResistivityCoeff(elements, numOfCalcElements,&rho1Coeff,1.0);
						//CalcInterpolatedRhoInElementCoeff(pos, x0, elements, numOfCalcElements,&rho2Coeff);
						//Functions::SetAtoResultCoef1DotBPlusCoef2DotC(&rhoCoeff, &rho1Coeff, &rho2Coeff, 0.5, 0.5);


						
					}
					else {
						cout << "something wrong in calc of rho" << endl;
					}


				}
				if (i == 2 || i == 3) {
					rho1Coeff.setZero();
					rho2Coeff.setZero();

					//Calc Rho Coeff for Roughen Matrix
					 
					if (layer == neighborElements[ipos]->layer) {
						double w1 = dy;
						double w2 = neighborElements[ipos]->dy;
						if (neighborElements[ipos]->isParent == true) {
							w2 = w2 / 2.0; //assumed that the layer difference is one,not more than one.
						}
						double wSum = w1 + w2;
						double _w1 = w2 / wSum;
						double _w2 = w1 / wSum;

						CalcCenterResistivityCoeff(elements, numOfCalcElements,&rho1Coeff,1.0);
						neighborElements[ipos]->CalcSurfaceRelatedResistivityCoeff(elements, numOfCalcElements, &rho2Coeff, 1.0, -1 * pos);
						Functions::SetAtoResultCoef1DotBPlusCoef2DotC(&rhoCoeff, &rho1Coeff, &rho2Coeff, _w1,_w2);
						//rhoCoeff = Functions::DotConst(_w1, &rho1Coeff) + Functions::DotConst(_w2, &rho2Coeff);
						//rhoCoeff = _w1 * rho1Coeff + _w2 * rho2Coeff;
						//if (i == 2) {
						//	rhoCoeff = rho2Coeff; //minus surface
						//}
						//else {
						//	rhoCoeff = rho1Coeff; //plus surface
						//}
							
						//harmonicOperationRhoCoeff = (-rho1Coeff + rho2Coeff);// *dS / dv * (*modelParam)[1]; //followed by Usui san method
							
						//double dl = (w1 + w2) / 2;
						//harmonicOperationRhoCoeff = (-rho1Coeff + rho2Coeff) / dl * (*modelParam)[1];
						
					}
					else if (layer > neighborElements[ipos]->layer) {
						int j;
						if (i == 2) { j = 3; }
						else { j = 2; }
						Functions::PlusEqual(&rhoCoeff, neighborElements[ipos]->resistivitySurfaceCoeff[j], 1.0);

						//Eigen::Vector3d x0 = centerCoord;
						//if (i == 2) { x0.coeffRef(1) += -dy; }
						//else { x0.coeffRef(1) += +dy; }

						//CalcCenterResistivityCoeff(elements, numOfCalcElements,&rho1Coeff,1.0);
						//CalcInterpolatedRhoInElementCoeff(pos, x0, elements, numOfCalcElements,&rho2Coeff);
						//Functions::SetAtoResultCoef1DotBPlusCoef2DotC(&rhoCoeff, &rho1Coeff, &rho2Coeff, 0.5, 0.5);

						
					}
					else {
						cout << "something wrong in calc of rho" << endl;
					}
				}
				if (i == 4 || i == 5) {
					rho1Coeff.setZero();
					rho2Coeff.setZero();


					// Calc Rho For Foward Calc
					if (layer == neighborElements[ipos]->layer) {
						double w1 = dz;
						double w2 = neighborElements[ipos]->dz;
						double wSum = w1 + w2;
						double _w1 = w2 / wSum;
						double _w2 = w1 / wSum;
						//rho = resistivity;

							
						CalcCenterResistivityCoeff(elements, numOfCalcElements,&rho1Coeff,1.0);
						neighborElements[ipos]->CalcSurfaceRelatedResistivityCoeff(elements, numOfCalcElements, &rho2Coeff, 1.0, -1 * pos);
						Functions::SetAtoResultCoef1DotBPlusCoef2DotC(&rhoCoeff, &rho1Coeff, &rho2Coeff, _w1, _w2);
						//rhoCoeff = Functions::DotConst(_w1, &rho1Coeff) + Functions::DotConst(_w2, &rho2Coeff);
							

						
					}
					else if (layer > neighborElements[ipos]->layer) {
						int j;
						if (i == 4) { j = 5; }
						else { j = 4; }
						Functions::PlusEqual(&rhoCoeff, neighborElements[ipos]->resistivitySurfaceCoeff[j], 1.0);


						//Eigen::Vector3d x0 = centerCoord;
						//if (i == 4) { x0.coeffRef(2) += -dz; }
						//else { x0.coeffRef(2) += +dz; }
						//	
						//CalcCenterResistivityCoeff(elements, numOfCalcElements,&rho1Coeff,1.0);
						//CalcInterpolatedRhoInElementCoeff(pos, x0, elements, numOfCalcElements,&rho2Coeff);
						//Functions::SetAtoResultCoef1DotBPlusCoef2DotC(&rhoCoeff, &rho1Coeff, &rho2Coeff,0.5,0.5);
							
						
					}
					else {
						cout << "something wrong in calc of rho" << endl;
					}
				}
			}
			if (isAlreadyCalcResisCoeff == false) {
				resistivitySurfaceCoeff[i]->setZero();
				Functions::PlusEqual(resistivitySurfaceCoeff[i], &rhoCoeff,1.0);

				if (layer > neighborElements[ipos]->layer) {
					int j;
					if (i == 0) j = 1; //reverse
					else if (i == 1) j = 0;
					else if (i == 2) j = 3;
					else if (i == 3) j = 2;
					else if (i == 4) j = 5;
					else if (i == 5) j = 4;
					Functions::PlusEqual(diffResistivitySurfaceCoeff[i], neighborElements[ipos]->diffResistivitySurfaceCoeff[j], 1.0);
				}
				else {
					Functions::SetAtoResultCoef1DotBPlusCoef2DotC(diffResistivitySurfaceCoeff[i], &rho1Coeff, &rho2Coeff, 1.0, -1.0);
				}

				resistivitySurfaceCoeff[i]->makeCompressed();
				resistivitySurfaceCoeff[i]->data().squeeze();

				diffResistivitySurfaceCoeff[i]->makeCompressed();
				diffResistivitySurfaceCoeff[i]->data().squeeze();
			}
			(*resistivitySurface)[i] = 0.0;
			for (int j = 0; j < resistivitySurfaceCoeff[i]->outerSize(); ++j) {
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[i], j); it; ++it)
				{
					int iRow = it.row();
					int iCol = it.col();
					Element*  element = (*calcElementsVector)[iCol];
					(*resistivitySurface)[i] += resistivitySurfaceCoeff[i]->coeff(0, iCol).real()*element->resistivity;
				}
			}
			

			//Calc Max resis and its index for matrix standardization
			maxResistivityInSameRowsOfMatrix = 0;
			maxResisIndexInSameRowsOfMatrix = -1;
			for (int i = 0; i < 6; i++) {
				for (int j = 0; j < resistivitySurfaceCoeff[i]->outerSize(); ++j) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[i], j); it; ++it)
					{
						if ((*calcElementsVector)[it.col()]->resistivity > maxResistivityInSameRowsOfMatrix) {
							maxResistivityInSameRowsOfMatrix = (*calcElementsVector)[it.col()]->resistivity;
							maxResisIndexInSameRowsOfMatrix = it.col();
						}

					}
				}
			}
		}
		isAlreadyCalcResisCoeff = true;
	}
}
void Element::Element::CalcZ(unordered_map<string, Element*>* elements, int numOfCalcElements,int iOmega) {
	
	if (boundary != "NOT_BOUNDARY") {
		Eigen::Matrix2cd Ztmp;
		Ztmp.setZero();
		Z[iOmega]=Ztmp;
	}
	Eigen::Matrix2cd Emat;
	Eigen::Matrix2cd Hmat;
	Emat.coeffRef(0, 0) = E[0].coeff(0);
	Emat.coeffRef(1, 0) = E[0].coeff(1);
	Emat.coeffRef(0, 1) = E[1].coeff(0);
	Emat.coeffRef(1, 1) = E[1].coeff(1);
	Hmat.coeffRef(0, 0) = H[0].coeff(0);
	Hmat.coeffRef(1, 0) = H[0].coeff(1);
	Hmat.coeffRef(0, 1) = H[1].coeff(0);
	Hmat.coeffRef(1, 1) = H[1].coeff(1);

	//if (isInversionImpedance) {
	//	cout << "EMAT:" << Emat << endl;
	//	cout << "Hmat:" << Hmat << endl;
	//}
	Z[iOmega]=Emat * Hmat.inverse();
}
Eigen::Matrix2cd Element::Element::CalcZ(vector < Eigen::Vector3cd> E, vector < Eigen::Vector3cd>H) {
	if (boundary != "NOT_BOUNDARY") {
		Eigen::Matrix2cd Ztmp;
		Ztmp.setZero();
		return Ztmp;
	}
	Eigen::Matrix2cd Emat;
	Eigen::Matrix2cd Hmat;
	Emat.coeffRef(0, 0) = E[0].coeff(0);
	Emat.coeffRef(1, 0) = E[0].coeff(1);
	Emat.coeffRef(0, 1) = E[1].coeff(0);
	Emat.coeffRef(1, 1) = E[1].coeff(1);
	Hmat.coeffRef(0, 0) = H[0].coeff(0);
	Hmat.coeffRef(1, 0) = H[0].coeff(1);
	Hmat.coeffRef(0, 1) = H[1].coeff(0);
	Hmat.coeffRef(1, 1) = H[1].coeff(1);
	return Emat * Hmat.inverse();
}

Eigen::Vector3cd Element::Element::CalcE(Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor>* Hresult, unordered_map<string, Element*> *elements,int numOfCalcElements,int itr,bool isReturn) {
	if (boundary != "NOT_BOUNDARY") {
		Eigen::Vector3cd tmpE;
		tmpE.setZero();
		E[itr].setZero();
		return tmpE;
	}
	Eigen::VectorXcd result{ 3 * numOfCalcElements };
	result = Eigen::VectorXcd(Hresult->col(0));

	Eigen::Vector3cd tmpE;
	tmpE.setZero();
	double sumDs = 0;//test
	double sumWeight = 0;
	for(int i = 0; i < 4; i++) {
		double dS = 0;
		if (i == 0 || i == 1)dS = dy * dz;
		else if (i == 2 || i == 3)dS = dx * dz;
		else dS = dx * dy;
		double rho = 0.0;
		rho = (*resistivitySurface)[i];
		Eigen::Vector3cd tmp;
		tmp.setZero();
		tmp =1.0/dS * rho * rotHdS[i] * result;
		//tmp =  rho * rotHdS[i].cast<std::complex<double>>() * Hresult.col(0);
		sumDs += dS;
		if (i == 0 ||i ==1) {
			tmpE.coeffRef(1) += tmp.coeff(1) ;
			tmpE.coeffRef(2) += tmp.coeff(2) ;
			tmpE.eval();
			sumWeight += dx;
		}
		if (i == 2 || i == 3) {
			tmpE.coeffRef(0) += tmp.coeff(0) ;
			tmpE.coeffRef(2) += tmp.coeff(2) ;
			tmpE.eval();
			sumWeight += dy;
		}
		if (i == 4 || i == 5) {
			tmpE.coeffRef(1) += tmp.coeff(1) ;
			tmpE.coeffRef(0) += tmp.coeff(0) ;
			tmpE.eval();
			sumWeight +=  dz;
			
		}
		
	}
	tmpE.coeffRef(0) = 0.5 * tmpE.coeff(0);
	tmpE.coeffRef(1) = 0.5 * tmpE.coeff(1);
	tmpE.coeffRef(2) = 0.25 * tmpE.coeff(2);
	//tmpE = 0.25* tmpE;//すべての面合計で４回同一方向成分を持つため
	//tmpE = tmpE / sumWeight;

	//tmpE.eval();
	if (isReturn == false) {
		E[itr]=tmpE;
	}
	else {
		return tmpE;
	}
}
Eigen::Vector3cd Element::Element::CalcE(Eigen::VectorXcd* Hresult, unordered_map<string, Element*>* elements, int numOfCalcElements, int itr,bool isReturn) {
	if (boundary != "NOT_BOUNDARY") {
		Eigen::Vector3cd tmpE;
		tmpE.setZero();
		E[itr].setZero();
		return tmpE;
	}
	Eigen::Vector3cd tmpE;
	tmpE.setZero();
	double sumDs = 0;//test
	double sumWeight = 0;
	for (int i = 0; i < 4; i++) {
		//if (i !=4) {
		//	continue; //test
		//}
		double dS = 0;
		if (i == 0 || i == 1)dS = dy * dz;
		else if (i == 2 || i == 3)dS = dx * dz;
		else dS = dx * dy;
		double rho = 0.0;
		rho = (*resistivitySurface)[i];
		
	//}
	//cout << rotHdS[i].rows() << " " << rotHdS[i].cols() <<boundary  << endl;
		Eigen::Vector3cd tmp;
		tmp.setZero();
		tmp = 1.0 / dS * rho * (rotHdS[i] * (*Hresult));
		if (isInversionImpedance) {
			//cout << "rotHdS[i]:" << tmp << rotHdS[i];
		}
		//tmp =  rho * rotHdS[i].cast<std::complex<double>>() * Hresult.col(0);
		sumDs += dS;
		if (i == 0 || i == 1) {
			tmpE.coeffRef(1) += tmp.coeff(1);
			tmpE.coeffRef(2) += tmp.coeff(2);
			tmpE.eval();
			sumWeight += dx;
		}
		if (i == 2 || i == 3) {
			tmpE.coeffRef(0) += tmp.coeff(0);
			tmpE.coeffRef(2) += tmp.coeff(2);
			tmpE.eval();
			sumWeight += dy;
		}
		if (i == 4 || i == 5) {
			tmpE.coeffRef(1) += tmp.coeff(1);
			tmpE.coeffRef(0) += tmp.coeff(0);
			tmpE.eval();
			sumWeight += dz;

		}

	}
	tmpE.coeffRef(0) = 0.5 * tmpE.coeff(0);
	tmpE.coeffRef(1) = 0.5 * tmpE.coeff(1);
	tmpE.coeffRef(2) = 0.25 * tmpE.coeff(2);
	//tmpE = 0.25* tmpE;//すべての面合計で４回同一方向成分を持つため
	//tmpE = tmpE / sumWeight;

	//tmpE.eval();
	//if (isInversionImpedance) {
	//	cout << "tmpE:" << tmpE << endl;
	//}
	if (isReturn == false) {
		E[itr] = tmpE;
	}
	else {
		return tmpE;
	}
}

Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> Element::Element::GetSumNCrossRhoRotHdS(bool isNeededStandardizationVal) {
	if (isNeededStandardizationVal) {
		return *sumNCrossRhoRotHdSPerResistivityPerDxDyDz;
	}
	else {
		return *sumNCrossRhoRotHdS;
	}
}
Eigen::SparseMatrix<complex<double>,Eigen::RowMajor> Element::Element::CalcSumNCrossRhoRotHdS(unordered_map<string, Element*> *elements, int numOfCalcElements) {

	if (isAlreadyCalcRotHdS == false) {
		sumNCrossRhoRotHdS = new Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor >{ 3,3 * numOfCalcElements };
		sumNCrossRhoRotHdS->reserve(Eigen::VectorXi::Constant(3,81*6)); //3*3*3*3*3*6
		//sumNCrossRhoRotHdS->uncompress();
		sumNCrossRhoRotHdSPerResistivityPerDxDyDz = new Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor >{ 3,3 * numOfCalcElements };
		sumNCrossRhoRotHdSPerResistivityPerDxDyDz->reserve(Eigen::VectorXi::Constant(3, 81 * 6)); //3*3*3*3*3*6
		rotHdS.resize(6);
		nCrossRotHdS.resize(6);
		if (isAlreadyCalcRotHdS == false) {
			for (int i = 0; i < 6; i++) {
				rotHdS[i] = Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
				rotHdS[i].reserve(Eigen::VectorXi::Constant(3, 81)); //3*3*3*3*3
				rotHdS[i].uncompress();

				nCrossRotHdS[i] = Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
				nCrossRotHdS[i].reserve(Eigen::VectorXi::Constant(3, 81)); //3*3*3*3*3
				nCrossRotHdS[i].uncompress();
			}
		}
	}
	else {
		delete sumNCrossRhoRotHdS;
		sumNCrossRhoRotHdS = new Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor >{ 3,3 * numOfCalcElements };
		sumNCrossRhoRotHdS->reserve(Eigen::VectorXi::Constant(3, 81*6)); //3*3*3*3*3*6

		delete sumNCrossRhoRotHdSPerResistivityPerDxDyDz;
		sumNCrossRhoRotHdSPerResistivityPerDxDyDz = new Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor >{ 3,3 * numOfCalcElements };
		sumNCrossRhoRotHdSPerResistivityPerDxDyDz->reserve(Eigen::VectorXi::Constant(3, 81 * 6)); //3*3*3*3*3*6
	}


	if (boundary != "NOT_BOUNDARY") {
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
		tmp.makeCompressed();
		tmp.reserve(0);
		return tmp;
	}



	for (int i = 0; i < 6; i++) {
		double rho = (*resistivitySurface)[i];
		Eigen::Vector3i pos;
		pos[0] = 0;
		pos[1] = 0;
		pos[2] = 0;
		if (i == 0) pos[0] = -1;
		else if (i == 1) pos[0] = 1;
		else if (i == 2) pos[1] = -1;
		else if (i == 3) pos[1] = 1;
		else if (i == 4) pos[2] = -1;
		else if (i == 5) pos[2] = 1;


		if (isAlreadyCalcRotHdS == false) {
			rotHdS[i] = CalcRotHdS(elements, numOfCalcElements, pos);
			//rotHdS[i].makeCompressed();

			/*nCrossRotHdS[i].row(0) = (pos.coeff(1) * rotHdS[i].row(2) - pos.coeff(2) * rotHdS[i].row(1));
			nCrossRotHdS[i].row(1) = (pos.coeff(2) * rotHdS[i].row(0) - pos.coeff(0) * rotHdS[i].row(2));
			nCrossRotHdS[i].row(2) = (pos.coeff(0) * rotHdS[i].row(1) - pos.coeff(1) * rotHdS[i].row(0));*/
			if (i == 0) {
				//nCrossRotHdS[i].row(1) =   rotHdS[i].row(2);
				//nCrossRotHdS[i].row(2) = - rotHdS[i].row(1);
				int i1 = 1;
				int i2 = 2;
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i2); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i1, iCol) = rotHdS[i].coeff(i2, iCol);
				}
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i1); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i2, iCol) = -rotHdS[i].coeff(i1, iCol);
				}
			}
			if (i == 1) {
				//nCrossRotHdS[i].row(1) = - rotHdS[i].row(2);
				//nCrossRotHdS[i].row(2) =   rotHdS[i].row(1);
				int i1 = 2;
				int i2 = 1;
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i2); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i1, iCol) = rotHdS[i].coeff(i2, iCol);
				}
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i1); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i2, iCol) = -rotHdS[i].coeff(i1, iCol);
				}
			}
			if (i == 2) {
				//nCrossRotHdS[i].row(0) = - rotHdS[i].row(2);
				//nCrossRotHdS[i].row(2) =   rotHdS[i].row(0);
				int i1 = 2;
				int i2 = 0;
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i2); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i1, iCol) = rotHdS[i].coeff(i2, iCol);
				}
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i1); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i2, iCol) = -rotHdS[i].coeff(i1, iCol);
				}
			}
			if (i == 3) {
				//nCrossRotHdS[i].row(0) =   rotHdS[i].row(2);
				//nCrossRotHdS[i].row(2) = - rotHdS[i].row(0);
				int i1 = 0;
				int i2 = 2;
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i2); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i1, iCol) = rotHdS[i].coeff(i2, iCol);
				}
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i1); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i2, iCol) = -rotHdS[i].coeff(i1, iCol);
				}
			}
			if (i == 4) {
				//nCrossRotHdS[i].row(0) =   rotHdS[i].row(1);
				//nCrossRotHdS[i].row(1) = - rotHdS[i].row(0);
				int i1 = 0;
				int i2 = 1;
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i2); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i1, iCol) = rotHdS[i].coeff(i2, iCol);
				}
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i1); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i2, iCol) = -rotHdS[i].coeff(i1, iCol);
				}
			}
			if (i == 5) {
				//nCrossRotHdS[i].row(0) = - rotHdS[i].row(1);
				//nCrossRotHdS[i].row(1) =   rotHdS[i].row(0);
				int i1 = 1;
				int i2 = 0;
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i2); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i1, iCol) = rotHdS[i].coeff(i2, iCol);
				}
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], i1); it; ++it)
				{
					int iCol = it.col();
					int iRow = it.row();
					nCrossRotHdS[i].coeffRef(i2, iCol) = -rotHdS[i].coeff(i1, iCol);
				}
			}
			nCrossRotHdS[i].makeCompressed();
			nCrossRotHdS[i].data().squeeze();
			
			rotHdS[i].makeCompressed();
			rotHdS[i].data().squeeze();
		}

	}
	sumNCrossRhoRotHdS->setZero();	
	sumNCrossRhoRotHdSPerResistivityPerDxDyDz->setZero();


	for (int i = 0; i < 6; i++) {
		Functions::PlusEqual(sumNCrossRhoRotHdS, &nCrossRotHdS[i], (*resistivitySurface)[i]);
		Functions::PlusEqual(sumNCrossRhoRotHdSPerResistivityPerDxDyDz, &nCrossRotHdS[i], (*resistivitySurface)[i]/maxResistivityInSameRowsOfMatrix/dx/dy/dz);
	}
	//*sumNCrossRhoRotHdS =Functions::DotConst ((*resistivitySurface)[0],& nCrossRotHdS[0]) 
	//	+ Functions::DotConst((*resistivitySurface)[1] , &nCrossRotHdS[1])
	//	+ Functions::DotConst((*resistivitySurface)[2] , & nCrossRotHdS[2])
	//	+ Functions::DotConst((*resistivitySurface)[3] , & nCrossRotHdS[3])
	//	+ Functions::DotConst((*resistivitySurface)[4] , & nCrossRotHdS[4])
	//	+ Functions::DotConst((*resistivitySurface)[5] , & nCrossRotHdS[5]); //This writing way is For Speed
	isAlreadyCalcRotHdS = true;
	//cout << rotHdS[0].rows() << " " << rotHdS[0].cols() <<boundary<< " rotHds"  << endl;
	//cout << sumNCrossRhoRotHdS.rows() << " " << sumNCrossRhoRotHdS.cols() << " rotHds" << endl;
	
	sumNCrossRhoRotHdS->makeCompressed();
	sumNCrossRhoRotHdS->data().squeeze();

	sumNCrossRhoRotHdSPerResistivityPerDxDyDz->makeCompressed();
	sumNCrossRhoRotHdSPerResistivityPerDxDyDz->data().squeeze();

	return *sumNCrossRhoRotHdS;
}

Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> Element::Element::CalcRotHdS(unordered_map<string, Element*> *elements, int numOfCalcElements, Eigen::Vector3i pos)
{
	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);

	if (isParent == true) {
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmpRotHdS{ 3,3 * numOfCalcElements };
		tmpRotHdS.makeCompressed();
		tmpRotHdS.reserve(243);
		if (pos[0] == -1) {
			for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(0, j);
					tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);
				
			}
		}
		else if (pos[0] == 1) {
			for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(1, j);
					tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);
				
			}
		}
		else if (pos[1] == -1) {
			for (int i = 0; i < 2; i++) {
					string childID = ID + Functions::GetBinaryValue(i, 0);
					tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);
				
			}
		}
		else if (pos[1] == 1) {
			for (int i = 0; i < 2; i++) {
					string childID = ID + Functions::GetBinaryValue(i, 1);
					tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);
			}
		}
		else if (pos[2] == -1) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);
				}
			}
		}
		else if (pos[2] == 1) {
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					tmpRotHdS += (*elements)[childID]->CalcRotHdS(elements, numOfCalcElements, pos);
				}
			}
		}
		return tmpRotHdS;
	}
	string neighborID = alreadyFoundNeighborID[ipos];
	Element* neighborElement = neighborElements[ipos];

	if (neighborElement->isParent == true) {
		return neighborElement->CalcRotHdS(elements, numOfCalcElements, -1 * pos);
	}
	if (isAlreadyCalcRotHdS) {
		if (pos[0] == -1) { return rotHdS[0]; }
		if (pos[0] == +1) { return rotHdS[1]; }
		else if (pos[1] == -1) { return rotHdS[2]; }
		else if (pos[1] == +1) { return rotHdS[3]; }
		else if (pos[2] == -1) { return rotHdS[4]; }
		else if (pos[2] == +1) { return rotHdS[5]; }
	}
	if (neighborElement->isAlreadyCalcRotHdS) {
		if (pos[0] == -1) { return neighborElement->rotHdS[1]; }
		if (pos[0] == +1) { return neighborElement->rotHdS[0]; }
		else if (pos[1] == -1) { return neighborElement->rotHdS[3]; }
		else if (pos[1] == +1) { return neighborElement->rotHdS[2]; }
		else if (pos[2] == -1) { return neighborElement->rotHdS[5]; }
		else if (pos[2] == +1) { return neighborElement->rotHdS[4]; }
	}

	//============Calc Each Faces======================================
	vector < Eigen::SparseMatrix<double, Eigen::RowMajor> > dHdl;
	//vector <Eigen::MatrixXd> dHdl(3);
	for (int i = 0; i < 3; i++) {
		Eigen::SparseMatrix<double, Eigen::RowMajor>tmp{ 3, 3 * numOfCalcElements };
		tmp.makeCompressed();
		tmp.reserve(243);
		dHdl.push_back(tmp);
		dHdl[i].resize(3, 3 * numOfCalcElements);
	}
	double dS = 0.0;

	vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> edgeVal(4);
	//vector<Eigen::MatrixXd> edgeVal(4);
	for (int i = 0; i < 4; i++) {
		edgeVal[i]= Eigen::SparseMatrix<double, Eigen::RowMajor> { 3,3 * numOfCalcElements };
		edgeVal[i].reserve(Eigen::VectorXi::Constant(3, 243));
		//edgeVal[i].resize(3, 3 * numOfCalcElements);
	}
	if (pos[0] == -1) {
		for (int i = 0; i < 4; i++) {
			edgeVal[i].setZero();
		}
		
		for (int i = 0; i < 4; i++) {
			unordered_map<string, Element*>nodeElementsDict;

			nodeElementsDict.reserve(8);
			vector<Element*> nodeElements;
			Eigen::Vector3i tmp;
			
			for (int j = 0; j < 4; j++) {
				tmp.setZero();
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[2] = -1; }
					if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[1] = +1; }
					if (j == 3) tmp[1] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[2] = +1; }
					if (j == 3) tmp[2] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[0] = 0; //nothing to do
					if (j == 1)tmp[0] = -1;
					if (j == 2) { tmp[0] = -1; tmp[1] = -1; }
					if (j == 3) tmp[1] = -1;
				}
				int neighborVal;
				neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
				string neighborIDtmp;
				neighborIDtmp = alreadyFoundNeighborID[neighborVal];
				if (nodeElementsDict.count(neighborIDtmp) == 0) {
					nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
					nodeElements.push_back(neighborElements[neighborVal]);
				}
			}
			vector<double> weightList(nodeElements.size());
			vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
			//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
			Eigen::Vector3d x0;
			x0 = centerCoord;
			if (i == 0) { x0[0] -= dx / 2; x0[2] -= dz / 2; }
			else if (i == 1) { x0[0] -= dx / 2; x0[1] += dy / 2; }
			else if (i == 2) { x0[0] -= dx / 2; x0[2] += dz / 2; }
			else if (i == 3) { x0[0] -= dx / 2; x0[1] -= dy / 2; }
			for (int k = 0; k < nodeElements.size(); k++) {
				double tmpdistance;
				vectorList[k].resize(3, 3 * numOfCalcElements);
				nodeElements[k]->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements,&vectorList[k],&tmpdistance);
				vectorList[k].makeCompressed();
				weightList[k] = 1 / tmpdistance;
			}
			double wSum = 0;
			for (int k = 0; k < nodeElements.size(); k++) {
				wSum += weightList[k];
			}
			for (int k = 0; k < nodeElements.size(); k++) {
				//edgeVal[i] += weightList[k] / wSum * vectorList[k];
				Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
				//edgeVal[i] += weightList[k] / wSum * vectorList[k];
			}
			//edgeVal[i] = edgeValDense[i].sparseView();
		}
		Eigen::Vector3d x0;
		x0.coeffRef(0) = centerCoord.coeff(0) - dx;
		x0.coeffRef(1) = centerCoord.coeff(1);
		x0.coeffRef(2) = centerCoord.coeff(2);
		vector < Eigen::SparseMatrix<double, Eigen::RowMajor>>centerVal;
		//vector < Eigen::MatrixXd>centerVal(2);
		for (int i = 0; i < 2; i++) {
			Eigen::SparseMatrix<double, Eigen::RowMajor>tmp{ 3,3 * numOfCalcElements };
			tmp.makeCompressed();
			tmp.reserve(243);
			centerVal.push_back(tmp);
		}
		centerVal[0].coeffRef(0, 3 * calcID) = 1.0;
		centerVal[0].coeffRef(1, 3 * calcID + 1) = 1.0;
		centerVal[0].coeffRef(2, 3 * calcID + 2) = 1.0;
		//neighborID = Functions::GetNeighborElement(elements, this, pos);
		ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		neighborID = alreadyFoundNeighborID[ipos];
		if (layer > neighborElements[ipos]->layer) {
			centerVal[1] = CalcInterpolatedVectorInElement(pos, x0, elements, numOfCalcElements);
			dHdl[0] = (centerVal[0] - centerVal[1]) / dx;
		}
		else {
			centerVal[1].coeffRef(0, 3 * neighborElements[ipos]->calcID) = 1.0;
			centerVal[1].coeffRef(1, 3 * neighborElements[ipos]->calcID + 1) = 1.0;
			centerVal[1].coeffRef(2, 3 * neighborElements[ipos]->calcID + 2) = 1.0;
			dHdl[0] = (centerVal[0] - centerVal[1]) / (dx / 2 + neighborElements[ipos]->dx / 2);
		}
		dHdl[1] = (edgeVal[1] - edgeVal[3]) / dy;
		dHdl[2] = (edgeVal[2] - edgeVal[0]) / dz;
		dS = dy * dz;

	}
	else if (pos[0] == 1) {
		for (int i = 0; i < 4; i++) {
			edgeVal[i].setZero();
		}
		for (int i = 0; i < 4; i++) {
			unordered_map<string, Element*>nodeElementsDict;
			nodeElementsDict.reserve(8);
			vector<Element*> nodeElements;
			Eigen::Vector3i tmp;
			for (int j = 0; j < 4; j++) {
				tmp.setZero();
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[0] = 0; //nothing to do
					else if (j == 1)tmp[0] = +1;
					else if (j == 2) { tmp[0] = +1; tmp[2] = -1; }
					else if (j == 3) tmp[2] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[0] = 0; //nothing to do
					else if (j == 1)tmp[0] = +1;
					else if (j == 2) { tmp[0] = +1; tmp[1] = +1; }
					else if (j == 3) tmp[1] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[0] = 0; //nothing to do
					else if (j == 1)tmp[0] = +1;
					else if (j == 2) { tmp[0] = +1; tmp[2] = +1; }
					else if (j == 3) tmp[2] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[0] = 0; //nothing to do
					else if (j == 1)tmp[0] = +1;
					else if (j == 2) { tmp[0] = +1; tmp[1] = -1; }
					else if (j == 3) tmp[1] = -1;
				}
				int neighborVal;
				neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
				string neighborIDtmp;
				neighborIDtmp = alreadyFoundNeighborID[neighborVal];
				if (nodeElementsDict.count(neighborIDtmp) == 0) {
					nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
					nodeElements.push_back(neighborElements[neighborVal]);
				}
			}
			vector<double> weightList(nodeElements.size());
			vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
			//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
			Eigen::Vector3d x0;
			x0 = centerCoord;
			if (i == 0) { x0[0] += dx / 2; x0[2] -= dz / 2; }
			else if (i == 1) { x0[0] += dx / 2; x0[1] += dy / 2; }
			else if (i == 2) { x0[0] += dx / 2; x0[2] += dz / 2; }
			else if (i == 3) { x0[0] += dx / 2; x0[1] -= dy / 2; }
			for (int k = 0; k < nodeElements.size(); k++) {
				double tmpdistance;
				vectorList[k].resize(3, 3 * numOfCalcElements);
				nodeElements[k]->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance);
				vectorList[k].makeCompressed();
				weightList[k] = 1 / tmpdistance;
			}
			double wSum = 0;
			for (int k = 0; k < nodeElements.size(); k++) {
				wSum += weightList[k];
			}
			for (int k = 0; k < nodeElements.size(); k++) {
				Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
				//edgeVal[i] += weightList[k] / wSum * vectorList[k];
			}
		}
		Eigen::Vector3d x0;
		x0.coeffRef(0) = centerCoord.coeff(0) + dx;
		x0.coeffRef(1) = centerCoord.coeff(1);
		x0.coeffRef(2) = centerCoord.coeff(2);
		vector < Eigen::SparseMatrix<double, Eigen::RowMajor>>centerVal;
		//vector < Eigen::MatrixXd>centerVal(2);
		for (int i = 0; i < 2; i++) {
			Eigen::SparseMatrix<double, Eigen::RowMajor>tmp{ 3,3 * numOfCalcElements };
			tmp.makeCompressed();
			tmp.reserve(243);
			centerVal.push_back(tmp);
			//centerVal[i].resize(3, 3 * numOfCalcElements);
		}
		centerVal[0].coeffRef(0, 3 * calcID) = 1.0;
		centerVal[0].coeffRef(1, 3 * calcID + 1) = 1.0;
		centerVal[0].coeffRef(2, 3 * calcID + 2) = 1.0;
		//neighborID = Functions::GetNeighborElement(elements, this, pos);
		ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		neighborID = alreadyFoundNeighborID[ipos];
		if (layer > neighborElements[ipos]->layer) {
			centerVal[1] = CalcInterpolatedVectorInElement(pos, x0, elements, numOfCalcElements);
			dHdl[0] = (centerVal[1] - centerVal[0]) / dx;
		}
		else {
			centerVal[1].coeffRef(0, 3 * neighborElements[ipos]->calcID) = 1.0;
			centerVal[1].coeffRef(1, 3 * neighborElements[ipos]->calcID + 1) = 1.0;
			centerVal[1].coeffRef(2, 3 * neighborElements[ipos]->calcID + 2) = 1.0;
			dHdl[0] = (centerVal[1] - centerVal[0]) / (dx / 2 + neighborElements[ipos]->dx / 2);
		}
		dHdl[1] = (edgeVal[1] - edgeVal[3]) / dy;
		dHdl[2] = (edgeVal[2] - edgeVal[0]) / dz;
		dS = dy * dz;
	}
	else if (pos[1] == -1) {
		for (int i = 0; i < 4; i++) {
			edgeVal[i].setZero();
		}
		for (int i = 0; i < 4; i++) {
			unordered_map<string, Element*>nodeElementsDict;
			nodeElementsDict.reserve(8);
			vector<Element*> nodeElements;
			Eigen::Vector3i tmp;
			for (int j = 0; j < 4; j++) {
				tmp.setZero();
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[1] = 0; //nothing to do
					else if (j == 1)tmp[1] = -1;
					else if (j == 2) { tmp[1] = -1; tmp[0] = -1; }
					else if (j == 3) tmp[0] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[1] = 0; //nothing to do
					else if (j == 1)tmp[1] = -1;
					else if (j == 2) { tmp[1] = -1; tmp[2] = +1; }
					else if (j == 3) tmp[2] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[1] = 0; //nothing to do
					else if (j == 1)tmp[1] = -1;
					else if (j == 2) { tmp[1] = -1; tmp[0] = +1; }
					else if (j == 3) tmp[0] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[1] = 0; //nothing to do
					else if (j == 1)tmp[1] = -1;
					else if (j == 2) { tmp[1] = -1; tmp[2] = -1; }
					else if (j == 3) tmp[2] = -1;
				}
				int neighborVal;
				neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
				string neighborIDtmp;
				neighborIDtmp = alreadyFoundNeighborID[neighborVal];
				if (nodeElementsDict.count(neighborIDtmp) == 0) {
					nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
					nodeElements.push_back(neighborElements[neighborVal]);
				}
			}
			vector<double> weightList(nodeElements.size());
			vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
			//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
			Eigen::Vector3d x0;
			x0 = centerCoord;
			if (i == 0) { x0[1] -= dy / 2; x0[0] -= dx / 2; }
			else if (i == 1) { x0[1] -= dy / 2; x0[2] += dz / 2; }
			else if (i == 2) { x0[1] -= dy / 2; x0[0] += dx / 2; }
			else if (i == 3) { x0[1] -= dy / 2; x0[2] -= dz / 2; }
			for (int k = 0; k < nodeElements.size(); k++) {
				double tmpdistance;
				vectorList[k].resize(3, 3 * numOfCalcElements);
				nodeElements[k]->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance);
				vectorList[k].makeCompressed();
				weightList[k] = 1 / tmpdistance;
			}
			double wSum = 0;
			for (int k = 0; k < nodeElements.size(); k++) {
				wSum += weightList[k];
			}
			for (int k = 0; k < nodeElements.size(); k++) {
				Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
				//edgeVal[i] += weightList[k] / wSum * vectorList[k];
			}
		}
		Eigen::Vector3d x0;
		x0.coeffRef(0) = centerCoord.coeff(0);
		x0.coeffRef(1) = centerCoord.coeff(1) - dy;
		x0.coeffRef(2) = centerCoord.coeff(2);
		vector < Eigen::SparseMatrix<double, Eigen::RowMajor>>centerVal;
		//vector < Eigen::MatrixXd>centerVal(2);
		for (int i = 0; i < 2; i++) {
			Eigen::SparseMatrix<double, Eigen::RowMajor>tmp{ 3,3 * numOfCalcElements };
			tmp.makeCompressed();
			tmp.reserve(0);
			centerVal.push_back(tmp);
			//centerVal[i].resize(3, 3 * numOfCalcElements);
		}
		centerVal[0].coeffRef(0, 3 * calcID) = 1.0;
		centerVal[0].coeffRef(1, 3 * calcID + 1) = 1.0;
		centerVal[0].coeffRef(2, 3 * calcID + 2) = 1.0;
		//neighborID = Functions::GetNeighborElement(elements, this, pos);
		ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		neighborID = alreadyFoundNeighborID[ipos];
		if (layer > neighborElements[ipos]->layer) {
			centerVal[1] = CalcInterpolatedVectorInElement(pos, x0, elements, numOfCalcElements);
			//centerVal[1] = neighborElements[ipos]->CalcInterpolatedVectorInElement(x0, elements, numOfCalcElements);
			//centerVal[1] = neighborElements[ipos]->CalcCenterVector(elements, numOfCalcElements);
			dHdl[1] = (centerVal[0] - centerVal[1]) / dy;
			//Eigen::Vector3d el;
			//el = (neighborElements[ipos]->centerCoord - centerCoord) / (neighborElements[ipos]->centerCoord - centerCoord).norm();
			//dHdl[1] = (centerVal[1] - centerVal[0]) / (dy / 2 + neighborElements[ipos]->dy / 2);
			//dHdl[1] = dHdl[1] - el.coeff(0) / el.coeff(1)*dHdl[0] - el.coeff(2) / el.coeff(1)*dHdl[2];
		}
		else {
			centerVal[1].coeffRef(0, 3 * neighborElements[ipos]->calcID) = 1.0;
			centerVal[1].coeffRef(1, 3 * neighborElements[ipos]->calcID + 1) = 1.0;
			centerVal[1].coeffRef(2, 3 * neighborElements[ipos]->calcID + 2) = 1.0;
			dHdl[1] = (centerVal[0] - centerVal[1]) / (dy / 2 + neighborElements[ipos]->dy / 2);
		}
		dHdl[2] =  (edgeVal[1] - edgeVal[3]) / dz;
		dHdl[0] = (edgeVal[2] - edgeVal[0]) / dx;




		dS = dz * dx;
	}

	else if (pos[1] == 1) {
		for (int i = 0; i < 4; i++) {
			edgeVal[i].setZero();
		}
		for (int i = 0; i < 4; i++) {
			unordered_map<string, Element*>nodeElementsDict;
			nodeElementsDict.reserve(8);
			vector<Element*> nodeElements;
			Eigen::Vector3i tmp;
			for (int j = 0; j < 4; j++) {
				tmp.setZero();
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[1] = 0; //nothing to do
					else if (j == 1)tmp[1] = +1;
					else if (j == 2) { tmp[1] = +1; tmp[0] = -1; }
					else if (j == 3) tmp[0] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[1] = 0; //nothing to do
					else if (j == 1)tmp[1] = +1;
					else if (j == 2) { tmp[1] = +1; tmp[2] = +1; }
					else if (j == 3) tmp[2] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[1] = 0; //nothing to do
					else if (j == 1)tmp[1] = +1;
					else if (j == 2) { tmp[1] = +1; tmp[0] = +1; }
					else if (j == 3) tmp[0] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[1] = 0; //nothing to do
					else if (j == 1)tmp[1] = +1;
					else if (j == 2) { tmp[1] = +1; tmp[2] = -1; }
					else if (j == 3) tmp[2] = -1;
				}
				int neighborVal;
				neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
				string neighborIDtmp;
				neighborIDtmp = alreadyFoundNeighborID[neighborVal];
				if (nodeElementsDict.count(neighborIDtmp) == 0) {
					nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
					nodeElements.push_back(neighborElements[neighborVal]);
				}
			}
			vector<double> weightList(nodeElements.size());
			vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
			//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
			Eigen::Vector3d x0;
			x0 = centerCoord;
			if (i == 0) { x0[1] += dy / 2; x0[0] -= dx / 2; }
			else if (i == 1) { x0[1] += dy / 2; x0[2] += dz / 2; }
			else if (i == 2) { x0[1] += dy / 2; x0[0] += dx / 2; }
			else if (i == 3) { x0[1] += dy / 2; x0[2] -= dz / 2; }
			for (int k = 0; k < nodeElements.size(); k++) {
				double tmpdistance;
				vectorList[k].resize(3, 3 * numOfCalcElements);
				nodeElements[k]->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance);
				vectorList[k].makeCompressed();
				weightList[k] = 1 / tmpdistance;
			}
			double wSum = 0;
			for (int k = 0; k < nodeElements.size(); k++) {
				wSum += weightList[k];
			}
			for (int k = 0; k < nodeElements.size(); k++) {
				Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
				//edgeVal[i] += weightList[k] / wSum * vectorList[k];
			}
		}
		Eigen::Vector3d x0;
		x0.coeffRef(0) = centerCoord.coeff(0);
		x0.coeffRef(1) = centerCoord.coeff(1) + dy;
		x0.coeffRef(2) = centerCoord.coeff(2);
		vector < Eigen::SparseMatrix<double, Eigen::RowMajor>>centerVal;
		//vector < Eigen::MatrixXd>centerVal(2);
		for (int i = 0; i < 2; i++) {
			Eigen::SparseMatrix<double, Eigen::RowMajor>tmp{ 3,3 * numOfCalcElements };
			tmp.makeCompressed();
			tmp.reserve(0);
			centerVal.push_back(tmp);
			//centerVal[i].resize(3, 3 * numOfCalcElements);
		}
		centerVal[0].coeffRef(0, 3 * calcID) = 1.0;
		centerVal[0].coeffRef(1, 3 * calcID + 1) = 1.0;
		centerVal[0].coeffRef(2, 3 * calcID + 2) = 1.0;
		//neighborID = Functions::GetNeighborElement(elements, this, pos);
		ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		neighborID = alreadyFoundNeighborID[ipos];
		if (layer > neighborElements[ipos]->layer) {
			centerVal[1] = CalcInterpolatedVectorInElement(pos, x0, elements, numOfCalcElements);
			//centerVal[1] = neighborElements[ipos]->CalcInterpolatedVectorInElement(x0, elements, numOfCalcElements);
			//centerVal[1] = neighborElements[ipos]->CalcCenterVector(elements, numOfCalcElements);
			dHdl[1] = (centerVal[1] - centerVal[0]) / dy;
			//Eigen::Vector3d el;
			//el = (neighborElements[ipos]->centerCoord - centerCoord) / (neighborElements[ipos]->centerCoord - centerCoord).norm();
			//dHdl[1] = (centerVal[1] - centerVal[0]) / (dy / 2 + neighborElements[ipos]->dy / 2);
			//dHdl[1] = dHdl[1] - el.coeff(0) / el.coeff(1)*dHdl[0] - el.coeff(2) / el.coeff(1)*dHdl[2];
		}
		else {
			centerVal[1].coeffRef(0, 3 * neighborElements[ipos]->calcID) = 1.0;
			centerVal[1].coeffRef(1, 3 * neighborElements[ipos]->calcID + 1) = 1.0;
			centerVal[1].coeffRef(2, 3 * neighborElements[ipos]->calcID + 2) = 1.0;
			dHdl[1] = (centerVal[1] - centerVal[0]) / (dy / 2 + neighborElements[ipos]->dy / 2);
		}
		dHdl[2] = (edgeVal[1] - edgeVal[3]) / dz;
		dHdl[0] = (edgeVal[2] - edgeVal[0]) / dx;




		dS = dz * dx;
	}
	else if (pos[2] == -1) {
		for (int i = 0; i < 4; i++) {
			edgeVal[i].setZero();
		}
		for (int i = 0; i < 4; i++) {
			unordered_map<string, Element*>nodeElementsDict;
			nodeElementsDict.reserve(8);
			vector<Element*> nodeElements;
			Eigen::Vector3i tmp;
			for (int j = 0; j < 4; j++) {
				tmp.setZero();
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[2] = 0; //nothing to do
					else if (j == 1)tmp[2] = -1;
					else if (j == 2) { tmp[2] = -1; tmp[1] = -1; }
					else if (j == 3) tmp[1] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[2] = 0; //nothing to do
					else if (j == 1)tmp[2] = -1;
					else if (j == 2) { tmp[2] = -1; tmp[0] = +1; }
					else if (j == 3) tmp[0] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[2] = 0; //nothing to do
					else if (j == 1)tmp[2] = -1;
					else if (j == 2) { tmp[2] = -1; tmp[1] = +1; }
					else if (j == 3) tmp[1] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[2] = 0; //nothing to do
					else if (j == 1)tmp[2] = -1;
					else if (j == 2) { tmp[2] = -1; tmp[0] = -1; }
					else if (j == 3) tmp[0] = -1;
				}
				int neighborVal;
				neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
				string neighborIDtmp;
				neighborIDtmp = alreadyFoundNeighborID[neighborVal];
				if (nodeElementsDict.count(neighborIDtmp) == 0) {
					nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
					nodeElements.push_back(neighborElements[neighborVal]);
				}
			}
			vector<double> weightList(nodeElements.size());
			vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
			//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
			Eigen::Vector3d x0;
			x0 = centerCoord;
			if (i == 0) { x0[2] -= dz / 2; x0[1] -= dy / 2; }
			else if (i == 1) { x0[2] -= dz / 2; x0[0] += dx / 2; }
			else if (i == 2) { x0[2] -= dz / 2; x0[1] += dy / 2; }
			else if (i == 3) { x0[2] -= dz / 2; x0[0] -= dx / 2; }
			for (int k = 0; k < nodeElements.size(); k++) {
				double tmpdistance;
				vectorList[k].resize(3, 3 * numOfCalcElements);
				nodeElements[k]->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance);
				//vectorList[k].makeCompressed();
				weightList[k] = 1 / tmpdistance;
			}
			double wSum = 0;
			for (int k = 0; k < nodeElements.size(); k++) {
				wSum += weightList[k];
			}
			for (int k = 0; k < nodeElements.size(); k++) {
				Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
				//edgeVal[i] += weightList[k] / wSum * vectorList[k];
			}
		}
		Eigen::Vector3d x0;
		x0.coeffRef(0) = centerCoord.coeff(0);
		x0.coeffRef(1) = centerCoord.coeff(1);
		x0.coeffRef(2) = centerCoord.coeff(2) - dz;
		vector < Eigen::SparseMatrix<double, Eigen::RowMajor>>centerVal;
		//vector < Eigen::MatrixXd>centerVal(2);
		for (int i = 0; i < 2; i++) {
			Eigen::SparseMatrix<double, Eigen::RowMajor>tmp{ 3,3 * numOfCalcElements };
			tmp.makeCompressed();
			tmp.reserve(243);
			centerVal.push_back(tmp);
			//centerVal[i].resize(3, 3 * numOfCalcElements);
		}
		centerVal[0].coeffRef(0, 3 * calcID) = 1.0;
		centerVal[0].coeffRef(1, 3 * calcID + 1) = 1.0;
		centerVal[0].coeffRef(2, 3 * calcID + 2) = 1.0;
		//neighborID = Functions::GetNeighborElement(elements, this, pos);
		ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		neighborID = alreadyFoundNeighborID[ipos];
		if (layer > neighborElements[ipos]->layer) {
			centerVal[1] = CalcInterpolatedVectorInElement(pos, x0, elements, numOfCalcElements);
			//centerVal[1] = neighborElements[ipos]->CalcInterpolatedVectorInElement(x0,elements, numOfCalcElements);
			//centerVal[1] = neighborElements[ipos]->CalcCenterVector(elements, numOfCalcElements);

			dHdl[2] = (centerVal[0] - centerVal[1]) / dz;
			//Eigen::Vector3d el;
			//el = (centerCoord - neighborElements[ipos]->centerCoord) / (centerCoord - neighborElements[ipos]->centerCoord).norm();
			//dHdl[2] = (centerVal[0] - centerVal[1]) / (dz / 2 + neighborElements[ipos]->dz / 2);
			//dHdl[2] = dHdl[2] - el.coeff(0) / el.coeff(2)*dHdl[0] - el.coeff(1) / el.coeff(2)*dHdl[1];
		}
		else {
			centerVal[1].coeffRef(0, 3 * neighborElements[ipos]->calcID) = 1.0;
			centerVal[1].coeffRef(1, 3 * neighborElements[ipos]->calcID + 1) = 1.0;
			centerVal[1].coeffRef(2, 3 * neighborElements[ipos]->calcID + 2) = 1.0;
			dHdl[2] = (centerVal[0] - centerVal[1]) / (dz / 2 + neighborElements[ipos]->dz / 2);
		}
		dHdl[0] = (edgeVal[1] - edgeVal[3]) / dx;
		dHdl[1] = (edgeVal[2] - edgeVal[0]) / dy;



		dS = dy * dx;
	}
	else if (pos[2] == 1) {
		for (int i = 0; i < 4; i++) {
			edgeVal[i].setZero();
		}
		for (int i = 0; i < 4; i++) {
			unordered_map<string, Element*>nodeElementsDict;
			nodeElementsDict.reserve(8);
			vector<Element*> nodeElements;
			Eigen::Vector3i tmp;
			for (int j = 0; j < 4; j++) {
				tmp.setZero();
				if (i == 0) { //edgeNode No.1
					if (j == 0)tmp[2] = 0; //nothing to do
					else if (j == 1)tmp[2] = +1;
					else if (j == 2) { tmp[2] = +1; tmp[1] = -1; }
					else if (j == 3) tmp[1] = -1;
				}
				else if (i == 1) { //edgeNode No.2
					if (j == 0)tmp[2] = 0; //nothing to do
					else if (j == 1)tmp[2] = +1;
					else if (j == 2) { tmp[2] = +1; tmp[0] = +1; }
					else if (j == 3) tmp[0] = +1;
				}
				else if (i == 2) { //edgeNode No.3
					if (j == 0)tmp[2] = 0; //nothing to do
					else if (j == 1)tmp[2] = +1;
					else if (j == 2) { tmp[2] = +1; tmp[1] = +1; }
					else if (j == 3) tmp[1] = +1;
				}
				else if (i == 3) { //edgeNode No.4
					if (j == 0)tmp[2] = 0; //nothing to do
					else if (j == 1)tmp[2] = +1;
					else if (j == 2) { tmp[2] = +1; tmp[0] = -1; }
					else if (j == 3) tmp[0] = -1;
				}
				int neighborVal;
				neighborVal = (tmp.coeff(0) + 1) + 3 * (tmp.coeff(1) + 1) + 9 * (tmp.coeff(2) + 1);
				string neighborIDtmp;
				neighborIDtmp = alreadyFoundNeighborID[neighborVal];
				if (nodeElementsDict.count(neighborIDtmp) == 0) {
					nodeElementsDict[neighborIDtmp] = neighborElements[neighborVal];
					nodeElements.push_back(neighborElements[neighborVal]);
				}
			}
			vector<double> weightList(nodeElements.size());
			vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> vectorList(nodeElements.size());
			//vector< Eigen::MatrixXd> vectorList(nodeElements.size());
			Eigen::Vector3d x0;
			x0 = centerCoord;
			if (i == 0) { x0[2] += dz / 2; x0[1] -= dy / 2; }
			else if (i == 1) { x0[2] += dz / 2; x0[0] += dx / 2; }
			else if (i == 2) { x0[2] += dz / 2; x0[1] += dy / 2; }
			else if (i == 3) { x0[2] += dz / 2; x0[0] -= dx / 2; }
			for (int k = 0; k < nodeElements.size(); k++) {
				double tmpdistance;
				vectorList[k].resize(3, 3 * numOfCalcElements);
				nodeElements[k]->CalcNearestNeighborVectorEdge(x0, elements, numOfCalcElements, &vectorList[k], &tmpdistance);
				vectorList[k].makeCompressed();
				weightList[k] = 1 / tmpdistance;
			}
			double wSum = 0;
			for (int k = 0; k < nodeElements.size(); k++) {
				wSum += weightList[k];
			}
			for (int k = 0; k < nodeElements.size(); k++) {
				Functions::PlusEqual(&edgeVal[i], &vectorList[k], weightList[k] / wSum);
				//edgeVal[i] += weightList[k] / wSum * vectorList[k];
			}
		}
		Eigen::Vector3d x0;
		x0.coeffRef(0) = centerCoord.coeff(0);
		x0.coeffRef(1) = centerCoord.coeff(1);
		x0.coeffRef(2) = centerCoord.coeff(2) + dz;
		vector < Eigen::SparseMatrix<double, Eigen::RowMajor>>centerVal;
		//vector < Eigen::MatrixXd>centerVal(2);
		for (int i = 0; i < 2; i++) {
			Eigen::SparseMatrix<double, Eigen::RowMajor>tmp{ 3,3 * numOfCalcElements };
			tmp.makeCompressed();
			tmp.reserve(243);
			centerVal.push_back(tmp);
			centerVal[i].resize(3, 3 * numOfCalcElements);
		}
		centerVal[0].coeffRef(0, 3 * calcID) = 1.0;
		centerVal[0].coeffRef(1, 3 * calcID + 1) = 1.0;
		centerVal[0].coeffRef(2, 3 * calcID + 2) = 1.0;
		//neighborID = Functions::GetNeighborElement(elements, this, pos);
		ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		neighborID = alreadyFoundNeighborID[ipos];
		if (layer > neighborElements[ipos]->layer) {
			centerVal[1] = CalcInterpolatedVectorInElement(pos, x0, elements, numOfCalcElements);
			//centerVal[1] = neighborElements[ipos]->CalcInterpolatedVectorInElement(x0, elements, numOfCalcElements);
			//centerVal[1] = neighborElements[ipos]->CalcCenterVector( elements, numOfCalcElements);

			dHdl[2] = (centerVal[1] - centerVal[0]) / dz;
			//Eigen::Vector3d el;
			//el = (neighborElements[ipos]->centerCoord - centerCoord) / (neighborElements[ipos]->centerCoord - centerCoord).norm();
			//dHdl[2] = (centerVal[1] - centerVal[0]) / (dz / 2 + neighborElements[ipos]->dz / 2);
			//dHdl[2] = dHdl[2] - el.coeff(0) / el.coeff(2)*dHdl[0] - el.coeff(1) / el.coeff(2)*dHdl[1];

		}
		else {
			centerVal[1].coeffRef(0, 3 * neighborElements[ipos]->calcID) = 1.0;
			centerVal[1].coeffRef(1, 3 * neighborElements[ipos]->calcID + 1) = 1.0;
			centerVal[1].coeffRef(2, 3 * neighborElements[ipos]->calcID + 2) = 1.0;
			dHdl[2] = (centerVal[1] - centerVal[0]) / (dz / 2 + neighborElements[ipos]->dz / 2);

		}
		dHdl[0] =  (edgeVal[1] - edgeVal[3]) / dx;
		dHdl[1] = (edgeVal[2] - edgeVal[0]) / dy;
		

		dS = dy * dx;
	}
	else {
		cout << "No Surface Indicated By Pos Vector." << endl;
		exit(1);

	}
	//---------------calc rotH--------------------------
	Eigen::SparseMatrix<double, Eigen::RowMajor> rotH{ 3, 3 * numOfCalcElements };

	rotH.reserve(Eigen::VectorXi::Constant(3,243)); //3*3*3*3*3
	rotH.row(0) = dHdl[1].row(2) - dHdl[2].row(1);
	rotH.row(1) = dHdl[2].row(0) - dHdl[0].row(2);
	rotH.row(2) = dHdl[0].row(1) - dHdl[1].row(0);
	Functions::DotConstSelf(dS, &rotH);

	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> rotHSparse{ 3, 3 * numOfCalcElements };
	rotHSparse.reserve(Eigen::VectorXi::Constant(3, 243));
	for (int j = 0; j < rotH.outerSize(); ++j) {
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(rotH, j); it; ++it)
		{
			int iCol = it.col();
			int iRow = it.row();
			rotHSparse.coeffRef(iRow, iCol)=std::complex<double>(rotH.coeff(iRow, iCol),0.0);
		}
	}
	rotHSparse.makeCompressed();
	rotHSparse.data().squeeze();
	return rotHSparse;
}


Eigen::SparseMatrix<double, Eigen::RowMajor> Element::Element::CalcCenterVector(unordered_map<string, Element*> *elements, int numChildElements) {
	Eigen::SparseMatrix<double, Eigen::RowMajor> row{ 3,3 * numChildElements };
	//Eigen::MatrixXd row{ 3,3 * numChildElements };
	//row.makeCompressed();
	if (isParent == false) {
		row.makeCompressed();
		row.reserve(3);
		row.coeffRef(0, 3 * calcID) = 1.0;
		row.coeffRef(1, 3 * calcID + 1) = 1.0;
		row.coeffRef(2, 3 * calcID + 2) = 1.0;

	}
	else {
		double totalDist = 0;
		vector<double> distVector;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					double dist = ((*elements)[childID]->centerCoord - centerCoord).norm();
					distVector.push_back(dist);
					totalDist += dist;
				
			}
		}

		row.reserve(243);
		int ii = 0;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					row += (*elements)[childID]->CalcCenterVector(elements, numChildElements) *distVector[ii]/totalDist;
					ii++;			
			}
		}
	}
	return row;
}

void Element::Element::CalcNearestNeighborVectorEdge(Eigen::Vector3d x0, unordered_map<string, Element*> *elements, int numChildElements, Eigen::SparseMatrix<double, Eigen::RowMajor>* row, double* distance) {
	double eps = dx * 0.000001;
	if (isParent == false) {
		
		row->reserve(Eigen::VectorXi::Constant(3, 81));
		row->coeffRef(0, 3 * calcID) = 1.0;
		row->coeffRef(1, 3 * calcID + 1) = 1.0;
		row->coeffRef(2, 3 * calcID + 2) = 1.0;
		*distance=(centerCoord-x0).norm();
		return;
	}
	else {
		//row->reserve(Eigen::VectorXi::Constant(3,243));
		double minDistance = ((*elements)[ID + Functions::GetBinaryValue(0, 0)]->centerCoord-x0).norm();
		vector<string>IDVec;
		//vector<double> debug;
		IDVec.push_back( ID + Functions::GetBinaryValue(0, 0));
		//debug.push_back(minDistance);
		
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
					if (i == 0 && j == 0) {
						continue;
					}
					string childID = ID + Functions::GetBinaryValue(i, j);
					double tmpDistance= ((*elements)[childID]->centerCoord - x0).norm();
					if (std::abs(tmpDistance - minDistance)<eps) {
						minDistance = tmpDistance;
						IDVec.push_back(childID);
						//debug.push_back(minDistance);
					}
					else if (tmpDistance < minDistance) {
						minDistance = tmpDistance;
						IDVec.clear();
						IDVec.shrink_to_fit();
						IDVec.push_back(childID);
						//debug.clear();
						//debug.shrink_to_fit();
						//debug.push_back(minDistance);
					}
			}
		}
		//if (IDVec.size() == 2) {
		//	cout << IDVec.size() <<" "<< IDVec[0]<<" "<< IDVec[1] << endl;
		//	cout << debug[0] << " " << debug[1]  << endl;
		//}
		Eigen::Vector3d averageCenterPoint;
		averageCenterPoint.setZero();
		for (int i = 0; i < IDVec.size(); i++) {
			Eigen::SparseMatrix<double, Eigen::RowMajor>rowtmp{ 3,3 * numChildElements };
			//Eigen::MatrixXd rowtmp{ 3,3 * numChildElements };
			(*elements)[IDVec[i]]->CalcNearestNeighborVectorEdge(x0, elements, numChildElements, &rowtmp, distance);
			averageCenterPoint += (*elements)[IDVec[i]]->centerCoord;
			*row += rowtmp;
		}
		*row = *row / IDVec.size();
		averageCenterPoint= averageCenterPoint/ IDVec.size();
		*distance = (averageCenterPoint - x0).norm();
		return;
	}
	
}

std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, double,string> Element::Element::CalcNearestNeighborVectorNode(Eigen::Vector3d x0, unordered_map<string, Element*>* elements, int numChildElements) {
	Eigen::SparseMatrix<double, Eigen::RowMajor> row{ 3,3 * numChildElements };
	if (isParent == false) {
		Eigen::SparseMatrix<double, Eigen::RowMajor>row{ 3,3 * numChildElements };
		row.makeCompressed();
		row.reserve(3);
		row.coeffRef(0, 3 * calcID) = 1.0;
		row.coeffRef(1, 3 * calcID + 1) = 1.0;
		row.coeffRef(2, 3 * calcID + 2) = 1.0;
		double distance = (centerCoord - x0).norm();
		return { row,distance,ID };
	}
	else {

		double minDistance = ((*elements)[ID + Functions::GetBinaryValue(0, 0)]->centerCoord - x0).norm();
		string minID = ID + Functions::GetBinaryValue(0, 0);
		double distance;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					double tmpDistance = ((*elements)[childID]->centerCoord - x0).norm();
					if (tmpDistance <minDistance ) {
						minDistance = tmpDistance;
						minID = childID;
					}

				
			}
		}
		std::tie(row, distance, minID) = (*elements)[minID]->CalcNearestNeighborVectorNode(x0, elements, numChildElements);
		return { row,distance,minID };
	}

}


Eigen::SparseMatrix<double, Eigen::RowMajor> Element::Element::CalcInterpolatedVectorInElement(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map < string, Element* > *elements, int numChildElements) {
	//val方向の隣接セル内の値を補間する。

	//vector < Eigen::Vector3d> xNode;
	//vector<Eigen::SparseMatrix < double, Eigen::RowMajor >>nodeVal;

	//for (int i = 0; i < 8; i++) {
	//	Eigen::Vector3d tmp = Eigen::Vector3d::Zero();
	//	Eigen::SparseMatrix<double, Eigen::RowMajor> tmpM{ 3, 3 * numChildElements };
	//	tmpM.reserve(243*100);// 100はsafetyfactor
	//	nodeVal.push_back(tmpM);
	//	xNode.push_back(tmp);
	//}
	vector< Eigen::SparseMatrix<double, Eigen::RowMajor>> relatedElementsRow;
	//vector< Eigen::MatrixXd> relatedElementsRow;
	//vector<Element*> relatedElementsVector;
	vector < Eigen::Vector3d> relatedElementsCenterCoord;
	Eigen::Vector3i pos;
	pos.setZero();
	unordered_map<string, Element*> nodeElementsDict;
	nodeElementsDict.reserve(100); //100は適当
	string neighborIDVirtual = Functions::GetVirturalNeighborElement(elements, ID, layer, val, nx, ny, nz);
	for (int i = 0; i < 6; i++) {
		pos.setZero();
		if (i == 0)     pos[0] = -1 ;
		else if (i == 1)pos[0] = 1 ;
		else if (i == 2)pos[1] = -1 ;
		else if (i == 3)pos[1] = 1 ;
		else if (i == 4)pos[2] = -1 ;
		else if (i == 5)pos[2] = 1 ;
		string neighborID = Functions::GetNeighborElement(elements, neighborIDVirtual, layer, pos, nx, ny, nz);
		//if (neighborIDVirtual.length() != neighborID.length()) {
		//	cout << ID << " " << neighborIDVirtual << " " << neighborID << endl;
		//	cout << val[0] << " " << val[1] << " " << val[2] << endl;
		//	cout << pos[0] << " " << pos[1] << " " << pos[2] << endl;
		//}
		if (nodeElementsDict.count(neighborID) == 0) {
			Element* element = (*elements)[neighborID];
			nodeElementsDict[neighborID] = element;
			//cout <<boundary<< neighborID<<ID << endl;
			//cout << "pos" << pos[0]<< pos[1]<< pos[2]  << endl;
			//cout <<"val"<< val[0] << val[1] << val[2] << endl;
			relatedElementsCenterCoord.push_back(element->centerCoord);
			Eigen::SparseMatrix<double, Eigen::RowMajor> row{ 3, 3 * numChildElements };

			//Eigen::MatrixXd row{ 3, 3 * numChildElements };

			row.makeCompressed();
			row.reserve(243);
			row = element->CalcCenterVector(elements, numChildElements);
			relatedElementsRow.push_back(row);

		}
	}
	Eigen::SparseMatrix<double, Eigen::RowMajor> tmp{ 3,3 * numChildElements };
	//Eigen::MatrixXd tmp{ 3,3 * numChildElements };
	tmp.makeCompressed();
	tmp.reserve(243);
	vector<double> w = Functions::CalcWeight(relatedElementsCenterCoord, x0);
	for (int i = 0; i < relatedElementsCenterCoord.size(); i++) {
		tmp += w[i] * relatedElementsRow[i];
	}
	return tmp;
}

double Element::Element::CalcInterpolatedRhoInElement(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map < string, Element* >* elements, int numChildElements) {
	//val方向の隣接セル内の値を補間する。
	vector<double> relatedElementsRho;
	vector < Eigen::Vector3d> relatedElementsCenterCoord;
	Eigen::Vector3i pos;
	pos.setZero();
	unordered_map<string, Element*> nodeElementsDict;
	nodeElementsDict.reserve(100); //100は適当
	string neighborIDVirtual = Functions::GetVirturalNeighborElement(elements, ID, layer, val, nx, ny, nz);
	for (int i = 0; i < 6; i++) {
		pos.setZero();
		if (i == 0)     pos[0] = -1;
		else if (i == 1)pos[0] = 1;
		else if (i == 2)pos[1] = -1;
		else if (i == 3)pos[1] = 1;
		else if (i == 4)pos[2] = -1;
		else if (i == 5)pos[2] = 1;
		string neighborID = Functions::GetNeighborElement(elements, neighborIDVirtual, layer, pos, nx, ny, nz);
		if (nodeElementsDict.count(neighborID) == 0) {
			Element* element = (*elements)[neighborID];
			nodeElementsDict[neighborID] = element;
			relatedElementsCenterCoord.push_back(element->centerCoord);
			double rhotmp;
			rhotmp = element->CalcCenterResistivity(elements, numChildElements);
			relatedElementsRho.push_back(rhotmp);

		}
	}
	double tmp = 0;
	vector<double> w = Functions::CalcWeight(relatedElementsCenterCoord, x0);
	for (int i = 0; i < relatedElementsCenterCoord.size(); i++) {
		tmp += w[i] * relatedElementsRho[i];
	}
	return tmp;

}
double Element::Element::CalcCenterResistivity(unordered_map<string, Element*>* elements, int numChildElements) {
	double rho=0;

	if (isParent == false) {
		rho = resistivity;

	}
	else {
		double totalDist = 0;
		vector<double> distVector;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					double dist = ((*elements)[childID]->centerCoord - centerCoord).norm();
					distVector.push_back(dist);
					totalDist += dist;
				
			}
		}
		int ii = 0;
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					rho += (*elements)[childID]->CalcCenterResistivity(elements, numChildElements) * distVector[ii] / totalDist;
					ii++;
			}
		}
		
	}

	return rho;
}


void Element::Element::SetTransitionZone(unordered_map<string, Element*>* elements) {
	//if (boundary != "NOT_BOUNDARY") {
	//	return;
	//}

	if (property->type == Property::Property::types::AIR) {
		isAirGroundBoundaryCell = false;
	}
	else {
		//vector<double> resistivities;
		//resistivities.push_back(resistivity);
		isAirGroundBoundaryCell = false;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				for (int k = 0; k < 3; k++) {
					Element* element = neighborElements[i + 3 * j + 9 * k];
					
					if (element != NULL && element->property->type == Property::Property::types::AIR) {
						isAirGroundBoundaryCell = true;
						neighborElements[1 + 3 + 9 * 2]->isSecondCellOfAirGroundBoundary = true;
						neighborElements[1 + 3 + 9 * 2]->neighborElements[1 + 3 + 9 * 2]->isThirdCellOfAirGroundBoundary = true;
						//resistivities.push_back(element->resistivity);
						
					}
				}
			}

		}
		//if (resistivities.size() != 0) {
		//	resistivity = 0;
		//	for (auto itr = resistivities.begin(); itr != resistivities.end(); itr++) {
		//		double resis = *itr;
		//		resistivity += 1 / resis;
		//	}
		//	resistivity = resistivities.size() / resistivity;
		//}
		
	}

	//if (property->type != Property::Property::types::AIR) {
	//	isAirGroundBoundaryCell = false;
	//}
	//else {
	//	isAirGroundBoundaryCell = false;
	//	for (int i = 0; i < 6; i++) {
	//		
	//		Eigen::Vector3i val;
	//		val.setZero();
	//		if (i == 0) val.coeffRef(0) = -1;
	//		else if (i == 1) val.coeffRef(0) = 1;
	//		else if (i == 2) val.coeffRef(1) = -1;
	//		else if (i == 3) val.coeffRef(1) = 1;
	//		else if (i == 4) val.coeffRef(2) = -1;
	//		else if (i == 5) val.coeffRef(2) = 1;
	//		string neighborID = Functions::GetNeighborElement(elements, this, val);
	//		if ((*elements).count(neighborID)!=0&&(*elements)[neighborID]->property->type != Property::Property::types::AIR) {
	//			isAirGroundBoundaryCell = true;
	//			//resistivity = 2 / (1 / resistivity + 1 / (*elements)[neighborID]->resistivity);
	//			//resistivity = (*elements)[neighborID]->resistivity;
	//			return;
	//		}
	//		
	//	}
	//}
}

void Element::Element::SearchRelatedCalcElements(unordered_map<string, Element*>* elements) {
	for (int i = 0; i < neighborElements.size(); i++) {
		if (neighborElements[i] != nullptr) {
			neighborElements[i]->SearchChildrenElements(elements, &relatedNeighborCalcElementsMap);
		}
	}
}
void Element::Element::SearchChildrenElements(unordered_map<string, Element*>* elements,map<string, Element*>* elementsMap) {
	if (isParent == false) {
		(*elementsMap)[ID] = this;
	}
	else {
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
					string childID = ID + Functions::GetBinaryValue(i, j);
					if (childID.find("BOUNDARY") == string::npos) {
						(*elements)[childID]->SearchChildrenElements(elements, elementsMap);
					}
			}
		}
	}
}


Eigen::Vector3cd Element::Element::CalcDEDH(int derID, int numOfCalcElements, unordered_map<string, Element*>* elements) {
	if (boundary != "NOT_BOUNDARY") {
		Eigen::Vector3cd tmpE;
		return tmpE;
	}

	Eigen::VectorXcd dCoeffdH{ 3*numOfCalcElements };
	dCoeffdH.setZero();
	dCoeffdH.coeffRef(derID) = 1.0;

	Eigen::Vector3cd dEdH;
	dEdH.setZero();
	double sumDs = 0;//test
	double sumWeight = 0;
	for (int i = 0; i < 4; i++) {
		//if (i !=4) {
		//	continue; //test
		//}
		double dS = 0;
		if (i == 0 || i == 1)dS = dy * dz;
		else if (i == 2 || i == 3)dS = dx * dz;
		else dS = dx * dy;
		double rho = 0.0;
		rho = (*resistivitySurface)[i];
		Eigen::Vector3cd tmp;
		tmp.setZero();
		tmp = 1.0 / dS * rho * rotHdS[i] * dCoeffdH;
		//tmp =  rho * rotHdS[i].cast<std::complex<double>>() * Hresult.col(0);
		sumDs += dS;
		if (i == 0 || i == 1) {
			dEdH.coeffRef(1) += tmp.coeff(1);
			dEdH.coeffRef(2) += tmp.coeff(2);
			dEdH.eval();
			sumWeight += dx;
		}
		if (i == 2 || i == 3) {
			dEdH.coeffRef(0) += tmp.coeff(0);
			dEdH.coeffRef(2) += tmp.coeff(2);
			dEdH.eval();
			sumWeight += dy;
		}
		if (i == 4 || i == 5) {
			dEdH.coeffRef(1) += tmp.coeff(1);
			dEdH.coeffRef(0) += tmp.coeff(0);
			dEdH.eval();
			sumWeight += dz;

		}

	}
	dEdH.coeffRef(0) = 0.5 * dEdH.coeff(0);
	dEdH.coeffRef(1) = 0.5 * dEdH.coeff(1);
	dEdH.coeffRef(2) = 0.25 * dEdH.coeff(2);
	//tmpE = 0.25* tmpE;//すべての面合計で４回同一方向成分を持つため
	//tmpE = tmpE / sumWeight;

	//tmpE.eval();

	return dEdH;
	
}

void Element::Element::CalcDZDH(const ub::vector< kv::complex<double>>* HTwoItr, unordered_map<string, Element*>* elements, int numOfCalcElements,int iOmega) {
	int numOfCalcHComponentsPerOneItr = 3* numOfCalcElements;
	//密ベクトルでautodifを計算するとメモリを食いすぎるので、indexと値を保存
	ub::vector<kv::autodif<kv::complex<double>>>HtwoItrSparse;
	std::vector<int>nonZeroRowIndices;
	//count num of nonzero
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < rotHdS[i].outerSize(); ++j) {
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], j); it; ++it)
			{
				int iCol = it.col();
				bool alreadyUsedCol = false;
				for (int k = 0; k < nonZeroRowIndices.size(); k++) {
					if (nonZeroRowIndices[k] == iCol) {
						alreadyUsedCol = true;
						break;
					}
				}
				if (alreadyUsedCol == false) {
					nonZeroRowIndices.push_back(iCol);
					nonZeroRowIndices.push_back(numOfCalcHComponentsPerOneItr+iCol); 
				}
			}
		}
	}
	//initialize
	ub::vector<kv::complex<double>>HtwoItrSparseTmp(nonZeroRowIndices.size());
	for (int i = 0; i < nonZeroRowIndices.size(); i++) {
		HtwoItrSparseTmp[i] = (*HTwoItr)(nonZeroRowIndices[i]);
	}
	HtwoItrSparse = kv::autodif<kv::complex<double>>::init(HtwoItrSparseTmp);
	

	ub::matrix<kv::autodif<kv::complex<double>>> Etmp(3,2);
	Etmp = CalcDEDH(&HtwoItrSparse, nonZeroRowIndices, elements, numOfCalcElements);
	//cout << "Etmp(x,itr=1)" << Etmp(0,1).v << endl;
	//cout << "Etmp(y,itr=1)" << Etmp(1, 1).v << endl;
	//cout << "Etmp(z,itr=1)" << Etmp(2, 1).v << endl;
	//cout << "E[1]" << E[1] << endl;

	vector<vector<int>> indexThisElem(2);
	indexThisElem[0].resize(3);
	indexThisElem[1].resize(3);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 3; j++) {
			for (int k = 0; k < nonZeroRowIndices.size(); k++) {
				if (nonZeroRowIndices[k] == i * numOfCalcHComponentsPerOneItr + 3 * calcID + j) {
					indexThisElem[i][j] = k;
					break;
				}
			}
		}
	}
	
	ub::matrix<kv::autodif<kv::complex<double>>> Htmp(3,2);
	Htmp(0,0) = HtwoItrSparse(indexThisElem[0][0]); 
	Htmp(1,0) = HtwoItrSparse(indexThisElem[0][1]);
	Htmp(2, 0) = HtwoItrSparse(indexThisElem[0][2]);
	Htmp(0,1) = HtwoItrSparse(indexThisElem[1][0]);
	Htmp(1,1) = HtwoItrSparse(indexThisElem[1][1]);
	Htmp(2, 1) = HtwoItrSparse(indexThisElem[1][2]);

	kv::autodif<kv::complex<double>> detA = (Htmp(0,0)*Htmp(1,1) - Htmp(0,1)*Htmp(1,0));
	ub::matrix< kv::autodif<kv::complex<double>> > Ztmp(2,2);
	Ztmp(0, 0) = 1 / detA * (Etmp(0,0)*Htmp(1,1) - Etmp(0,1)*Htmp(1,0));
	Ztmp(0, 1) = 1 / detA * (-Etmp(0,0)*Htmp(0,1) + Etmp(0,1)*Htmp(0,0));
	Ztmp(1, 0) = 1 / detA * (Etmp(1,0)*Htmp(1,1) - Etmp(1,1)*Htmp(1,0));
	Ztmp(1, 1) = 1 / detA * (-Etmp(1,0)*Htmp(0,1) + Etmp(1,1)*Htmp(0,0));
	//cout << Ztmp(0,0).v <<" "<< Ztmp(0, 1).v<<" "<< Ztmp(1, 0).v<<" "<< Ztmp(1, 1).v<< endl;
	//cout << Z[0] << endl;
	//cout<< Ztmp(0, 0).d.size() << " " << Ztmp(0, 1).d.size() << " " << Ztmp(1, 0).d.size() << " " << Ztmp(1, 1).d.size() << endl;


	ub::matrix < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> dZdHtmp(2, 2);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			dZdHtmp(i, j).resize(1,2 * numOfCalcHComponentsPerOneItr);
			dZdHtmp(i,j).makeCompressed();
			dZdHtmp(i, j).reserve(nonZeroRowIndices.size());
			for (int k = 0; k < nonZeroRowIndices.size(); k++) {
				if (Ztmp(i, j).d(k).real() != 0.0 || Ztmp(i, j).d(k).imag() != 0.0) {
					dZdHtmp(i, j).coeffRef(0, nonZeroRowIndices[k]).real(Ztmp(i, j).d(k).real());
					dZdHtmp(i, j).coeffRef(0, nonZeroRowIndices[k]).imag(Ztmp(i, j).d(k).imag());
					//cout << k << " " << dZdHtmp(i, j).coeff(0, nonZeroRowIndices[k]) << endl;
					//cout << "Itr:" << int(nonZeroRowIndices[k] / 3 / numOfCalcElements) << endl;
					//cout << "elemCalcID:" << int(nonZeroRowIndices[k] / 3) << endl;
					//cout << "Hcomponent:" << nonZeroRowIndices[k] % 3 << endl;
					
					
				}
			}
			//cout << "dZdHtmp(i, j)" << Ztmp(i, j).d << endl;
			dZdH[iOmega](i, j) = dZdHtmp(i, j);
			dZdH[iOmega](i, j).makeCompressed();
			dZdH[iOmega](i, j).data().squeeze();
		}
	}
}
ub::matrix<kv::autodif<kv::complex<double>>> Element::Element::CalcDEDH(ub::vector<kv::autodif<kv::complex<double>>>* HresultTwoItr, std::vector<int>nonZeroRowIndices, unordered_map<string, Element*>* elements, int numOfCalcElements){
	if (boundary != "NOT_BOUNDARY") {
		ub::matrix< kv::autodif<kv::complex<double>>> tmpE(3,2);
		return tmpE;
	}

	ub::matrix< kv::autodif<kv::complex<double>>> tmpE(3,2);
	tmpE(0,0) = 0; tmpE(1,0) = 0; tmpE(2,0) = 0;
	tmpE(0, 1) = 0; tmpE(1, 1) = 0; tmpE(2, 1) = 0;
	double sumDs = 0;//test
	double sumWeight = 0;
	for (int i = 0; i < 4; i++) {
		double dS = 0;
		if (i == 0 || i == 1)dS = dy * dz;
		else if (i == 2 || i == 3)dS = dx * dz;
		else dS = dx * dy;
		double rho = 0.0;
		rho = (*resistivitySurface)[i];
	//}
		for (int itr = 0; itr < 2; itr++) {
			ub::vector<kv::autodif<kv::complex<double>>> tmp(3);
			tmp(0) = 0; tmp(1) = 0; tmp(2) = 0;
			for (int j = 0; j < rotHdS[i].outerSize(); ++j) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], j); it; ++it)
				{
					int iRow = it.row();
					int iCol = it.col();
					int index=-1;
					for (int k = 0; k < nonZeroRowIndices.size(); k++) {
						if (nonZeroRowIndices[k] == itr * 3 * numOfCalcElements + iCol) {
							index = k;
							break;
						}
					}
					if (index < 0) {
						cout << "Wrong In CalcEForAutoDiff" << endl;
						exit(-1);
					}
					
					tmp(iRow) += 1.0 / dS * rho*kv::complex<double>( rotHdS[i].coeff(iRow, iCol).real(), rotHdS[i].coeff(iRow, iCol).imag())*(*HresultTwoItr)(index);
					//std::cout << "(" << it.row() << ","; // row index
					//std::cout << it.col() << ")\t"; // col index (here it is equal to k)
				}
			}

			sumDs += dS;
			if (i == 0 || i == 1) {
				tmpE(1, itr) += tmp(1);
				tmpE(2, itr) += tmp(2);
				sumWeight += dx;
			}
			if (i == 2 || i == 3) {
				tmpE(0, itr) += tmp(0);
				tmpE(2, itr) += tmp(2);
				sumWeight += dy;
				if (i == 4 || i == 5) {
					tmpE(0, itr) += tmp(0);
					tmpE(1, itr) += tmp(1);
					sumWeight += dz;
				}
			}
		}
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			if (i != 2) {
				tmpE(i, j) = 0.5*tmpE(i, j);
			}
			else {
				tmpE(i, j) = 0.25*tmpE(i, j);
			}
		}
	}
	return tmpE;
}
void Element::Element::CalcDZDRho(const ub::vector<kv::complex<double>>* rhoVecUb, const ub::vector<kv::complex<double>>* HresultTwoItr,const vector<Element*>* calcElementsVector,const int numOfCalcElements,const int iOmega) {
	//密ベクトルでautodifを計算するとメモリを食いすぎるので、indexと値を保存
	ub::vector<kv::autodif<kv::complex<double>>>rhoVecSparse;
	std::vector<int>nonZeroRowIndices;
	//count num of nonzero
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < resistivitySurfaceCoeff[i]->outerSize(); ++j) {
			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[i], j); it; ++it)
			{
				int iCol = it.col();
				bool alreadyUsedCol = false;
				for (int k = 0; k < nonZeroRowIndices.size(); k++) {
					if (nonZeroRowIndices[k] == iCol) {
						alreadyUsedCol = true;
						break;
					}
				}
				if (alreadyUsedCol == false) {
					nonZeroRowIndices.push_back(iCol);
				}
			}
		}
	}
	//initialize
	ub::vector<kv::complex<double>>rhoVecSparseTmp(nonZeroRowIndices.size());
	for (int i = 0; i < nonZeroRowIndices.size(); i++) {
		rhoVecSparseTmp[i] = (*rhoVecUb)(nonZeroRowIndices[i]);
	}
	rhoVecSparse = kv::autodif<kv::complex<double>>::init(rhoVecSparseTmp);
	ub::vector<kv::complex<double>>HresultTwoItrUb(HresultTwoItr->size());
	for (int i = 0; i < HresultTwoItr->size(); i++) {
		HresultTwoItrUb(i) = kv::complex<double>((*HresultTwoItr)(i).real(), (*HresultTwoItr)(i).imag());
	}

	ub::matrix<kv::autodif<kv::complex<double>>>Etmp;
	Etmp = CalcDEDRho(&rhoVecSparse, &HresultTwoItrUb, nonZeroRowIndices, numOfCalcElements);

	ub::matrix<kv::complex<double>> Htmp(3, 2);
	Htmp(0, 0) = HresultTwoItrUb(3 * calcID);
	Htmp(1, 0) = HresultTwoItrUb(3 * calcID + 1);
	Htmp(2, 0) = HresultTwoItrUb(3 * calcID + 2);
	Htmp(0, 1) = HresultTwoItrUb(3 * calcID + 3 * numOfCalcElements);
	Htmp(1, 1) = HresultTwoItrUb(3 * calcID + 1 + 3 * numOfCalcElements);
	Htmp(2, 1) = HresultTwoItrUb(3 * calcID + 2 + 3 * numOfCalcElements);
	kv::complex<double> detA = (Htmp(0, 0)*Htmp(1, 1) - Htmp(0, 1)*Htmp(1, 0));
	ub::matrix< kv::autodif<kv::complex<double>> > Ztmp(2, 2);
	Ztmp(0, 0) = 1 / detA * (Etmp(0, 0)*Htmp(1, 1) - Etmp(0, 1)*Htmp(1, 0));
	Ztmp(0, 1) = 1 / detA * (-Etmp(0, 0)*Htmp(0, 1) + Etmp(0, 1)*Htmp(0, 0));
	Ztmp(1, 0) = 1 / detA * (Etmp(1, 0)*Htmp(1, 1) - Etmp(1, 1)*Htmp(1, 0));
	Ztmp(1, 1) = 1 / detA * (-Etmp(1, 0)*Htmp(0, 1) + Etmp(1, 1)*Htmp(0, 0));
	//cout << Ztmp(0,0).v <<" "<< Ztmp(0, 1).v<<" "<< Ztmp(1, 0).v<<" "<< Ztmp(1, 1).v<< endl;
	//cout << Z[0] << endl;
	//cout<< Ztmp(0, 0).d.size() << " " << Ztmp(0, 1).d.size() << " " << Ztmp(1, 0).d.size() << " " << Ztmp(1, 1).d.size() << endl;


	ub::matrix < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> dZdRhotmp(2, 2);
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			dZdRhotmp(i, j).resize(1, numOfCalcElements);
			dZdRhotmp(i,j).makeCompressed();
			dZdRhotmp(i, j).reserve(400);
			dZdRhotmp(i, j).setZero();
			for (int k = 0; k < nonZeroRowIndices.size(); k++) {
				if (Ztmp(i, j).d(k).real() != 0.0 || Ztmp(i, j).d(k).imag() != 0.0) {
					//if ((*calcElementsVector)[nonZeroRowIndices[k]]->masterResistivityElement != nullptr) { //if slave element, add coeff to the master 
					//	Element* slaveElement = (*calcElementsVector)[nonZeroRowIndices[k]];
					//	dZdRhotmp(i, j).coeffRef(0, slaveElement->masterResistivityElement->calcID).real(dZdRhotmp(i, j).coeff(0, slaveElement->masterResistivityElement->calcID).real() + Ztmp(i, j).d(k).real());
					//	dZdRhotmp(i, j).coeffRef(0, slaveElement->masterResistivityElement->calcID).imag(dZdRhotmp(i, j).coeff(0, slaveElement->masterResistivityElement->calcID).imag() + Ztmp(i, j).d(k).imag());
					//}
					//else {
						dZdRhotmp(i, j).coeffRef(0, nonZeroRowIndices[k]).real(Ztmp(i, j).d(k).real());
						dZdRhotmp(i, j).coeffRef(0, nonZeroRowIndices[k]).imag(Ztmp(i, j).d(k).imag());
					//}
				}
			}
			dZdRho[iOmega](i, j) = dZdRhotmp(i, j);
		}
	}
}

ub::matrix<kv::autodif<kv::complex<double>>> Element::Element::CalcDEDRho(ub::vector<kv::autodif<kv::complex<double>>>* rhoVec, ub::vector<kv::complex<double>>* HresultTwoItr, std::vector<int>nonZeroRowIndices, int numOfCalcElements) {
	if (boundary != "NOT_BOUNDARY") {
		ub::matrix< kv::autodif<kv::complex<double>>> tmpE(3, 2);
		return tmpE;
	}

	ub::matrix< kv::autodif<kv::complex<double>>> tmpE(3, 2);
	tmpE(0, 0) = 0; tmpE(1, 0) = 0; tmpE(2, 0) = 0;
	tmpE(0, 1) = 0; tmpE(1, 1) = 0; tmpE(2, 1) = 0;
	double sumDs = 0;//test
	double sumWeight = 0;
	for (int i = 0; i < 4; i++) {
		double dS = 0;
		if (i == 0 || i == 1)dS = dy * dz;
		else if (i == 2 || i == 3)dS = dx * dz;
		else dS = dx * dy;
		kv::autodif<kv::complex<double>>rho;
		rho = 0.0;
		for (int j = 0; j < resistivitySurfaceCoeff[i]->outerSize(); ++j) {
			for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[i], j); it; ++it)
			{
				int iCol = it.col();
				int iRow = it.row();
				for (int k = 0; k < nonZeroRowIndices.size(); k++) {
					if (nonZeroRowIndices[k] == iCol) {
						rho += resistivitySurfaceCoeff[i]->coeff(0, iCol).real()*(*rhoVec)(k);
					}
				}
			}
		}

		//}
		for (int itr = 0; itr < 2; itr++) {
			ub::vector<kv::autodif<kv::complex<double>>> tmp(3);
			tmp(0) = 0; tmp(1) = 0; tmp(2) = 0;
			for (int j = 0; j < rotHdS[i].outerSize(); ++j) {
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], j); it; ++it)
				{
					int iRow = it.row();
					int iCol = it.col();
					tmp(iRow) += 1.0 / dS *kv::complex<double>( rotHdS[i].coeff(iRow, iCol).real(),rotHdS[i].coeff(iRow, iCol).imag())*(*HresultTwoItr)(itr * 3 * numOfCalcElements + iCol)*rho;
					//std::cout << "(" << it.row() << ","; // row index
					//std::cout << it.col() << ")\t"; // col index (here it is equal to k)
				}
			}

			sumDs += dS;
			if (i == 0 || i == 1) {
				tmpE(1, itr) += tmp(1);
				tmpE(2, itr) += tmp(2);
				sumWeight += dx;
			}
			if (i == 2 || i == 3) {
				tmpE(0, itr) += tmp(0);
				tmpE(2, itr) += tmp(2);
				sumWeight += dy;
			}
			if (i == 4 || i == 5) {
				tmpE(0, itr) += tmp(0);
				tmpE(1, itr) += tmp(1);
				sumWeight += dz;
			}
		}
	}
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 2; j++) {
			if (i != 2) {
				tmpE(i, j) = 0.5*tmpE(i, j);
			}
			else {
				tmpE(i, j) = 0.25*tmpE(i, j);
			}
		}
	}
	return tmpE;
}
void Element::Element::CalcLambdaDSumNCrossRhoRotHdSDRho(std::unordered_map<std::string, Element*>* elements, const ub::vector<complex<double>>* rhoVec,
	const vector<Eigen::VectorXcd>* HresultTwoItr,const vector<Element*>* calcElementsVector,
	const int numOfCalcElements, const int numOfInvertedResisElem, const Eigen::VectorXcd* lambdaEachOmega, Eigen::VectorXcd* lambdaDRDRho) {
	//密ベクトルでautodifを計算するとメモリを食いすぎるので、indexと値を保存
	ub::vector<kv::autodif<complex<double>>>rhoVecSparse;
	
	if (boundary != "NOT_BOUNDARY") {
		return;
	}
	//count num of nonzero
	
	if (nonZeroRowIndices.size() == 0) {
		for (int i = 0; i < 6; i++) {
			for (int j = 0; j < resistivitySurfaceCoeff[i]->outerSize(); ++j) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[i], j); it; ++it)
				{
					int iCol = it.col();
					bool alreadyUsedCol = false;
					for (int k = 0; k < nonZeroRowIndices.size(); k++) {
						if (nonZeroRowIndices[k] == iCol) {
							alreadyUsedCol = true;
							break;
						}
					}
					if (alreadyUsedCol == false) {
						nonZeroRowIndices.push_back(iCol);

					}
					
				}
			}
		}
	}

	int maxResisIndex = -1;
	for (int i = 0; i < nonZeroRowIndices.size(); i++) {
		if (maxResisIndexInSameRowsOfMatrix == nonZeroRowIndices[i]) {
			maxResisIndex = i;
		}
	}

	
	//initialize
	ub::vector<complex<double>>rhoVecSparseTmp(nonZeroRowIndices.size());
	for (int i = 0; i < nonZeroRowIndices.size(); i++) {
		rhoVecSparseTmp[i] = (*rhoVec)(nonZeroRowIndices[i]);
	}
	rhoVecSparse = kv::autodif<complex<double>>::init(rhoVecSparseTmp);

	//Calc

	for (int itr = 0; itr < 2; itr++) {
		ub::vector<kv::autodif<complex<double>>> dSumNCrossRhoRotHdSRhoSparse{ 3 };

		for (int i = 0; i < 6; i++) {

			kv::autodif<complex<double>> rho;
			for (int j = 0; j < nonZeroRowIndices.size(); j++) {
				//rho += resistivitySurfaceCoeff[i]->coeff(0,nonZeroRowIndices[j]).real()*rhoVecSparse(j); 
				rho += resistivitySurfaceCoeff[i]->coeff(0, nonZeroRowIndices[j]).real()*rhoVecSparse(j);
			}
			//rho = rho / rhoVecSparse(maxResisIndex) / dx / dy / dz; //rhoVecSparse(indexMyself)/dx/dy/dz is for Standardization
															  //Connected With variable "unit" in MakeMatrix() in Analysis

			//for (int j = 0; j < resistivitySurfaceCoeff[i]->outerSize(); ++j) {
			//	for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(*resistivitySurfaceCoeff[i], j); it; ++it)
			//	{
			//		int iCol = it.col();
			//		for (int k = 0; k < nonZeroRowIndices.size(); k++) {
			//			if (nonZeroRowIndices[k] == iCol) {
			//				rho += resistivitySurfaceCoeff[i]->coeff(0, iCol).real()*rhoVecSparse(k);
			//			}
			//		}
			//	}
			//}
			Eigen::Vector3d pos;
			pos[0] = 0;
			pos[1] = 0;
			pos[2] = 0;
			if (i == 0) pos[0] = -1;
			else if (i == 1) pos[0] = 1;
			else if (i == 2) pos[1] = -1;
			else if (i == 3) pos[1] = 1;
			else if (i == 4) pos[2] = -1;
			else if (i == 5) pos[2] = 1;

			Eigen::Vector3cd rotHdSVal;
			rotHdSVal.setZero();
			for (int ii = 0; ii < 3; ii++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], ii); it; ++it) {
					rotHdSVal.coeffRef(ii) += rotHdS[i].coeff(ii, it.col())*(*HresultTwoItr)[itr].coeff(it.col());
				}
			}
			//rotHdSVal = rotHdS[i] * (*HresultTwoItr)[itr];

			dSumNCrossRhoRotHdSRhoSparse(1) += rho * (pos.coeff(2)*rotHdSVal.coeff(0));
			dSumNCrossRhoRotHdSRhoSparse(2) += -rho * (pos.coeff(1)*rotHdSVal.coeff(0));

			dSumNCrossRhoRotHdSRhoSparse(0) += -rho * (pos.coeff(2)*rotHdSVal.coeff(1));
			dSumNCrossRhoRotHdSRhoSparse(2) += rho * (pos.coeff(0)*rotHdSVal.coeff(1));

			dSumNCrossRhoRotHdSRhoSparse(0) += rho * (pos.coeff(1)*rotHdSVal.coeff(2));
			dSumNCrossRhoRotHdSRhoSparse(1) += -rho * (pos.coeff(0)*rotHdSVal.coeff(2));

			//for (int j = 0; j < rotHdS[i].outerSize(); ++j) {
			//	for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(rotHdS[i], j); it; ++it) {
			//		int iCol = it.col();
			//		int iRow = it.row();
			//		//n×ρ∇×HdS
			//		if (iRow == 0) {
			//				dSumNCrossRhoRotHdSRhoSparse(1) += rho * (pos.coeff(2)*kv::complex<double>(rotHdS[i].coeff(iRow, iCol).real(), rotHdS[i].coeff(iRow, iCol).imag())*(*HresultTwoItr)(itr * 3 * numOfCalcElements + iCol));
			//				dSumNCrossRhoRotHdSRhoSparse(2) += -rho * (pos.coeff(1)*kv::complex<double>(rotHdS[i].coeff(iRow, iCol).real(), rotHdS[i].coeff(iRow, iCol).imag())*(*HresultTwoItr)(itr * 3 * numOfCalcElements + iCol));
			//		}
			//		else if (iRow == 1) {
			//				dSumNCrossRhoRotHdSRhoSparse(0) += -rho * (pos.coeff(2)*kv::complex<double>(rotHdS[i].coeff(iRow, iCol).real(), rotHdS[i].coeff(iRow, iCol).imag())*(*HresultTwoItr)(itr * 3 * numOfCalcElements + iCol));
			//				dSumNCrossRhoRotHdSRhoSparse(2) += rho * (pos.coeff(0)*kv::complex<double>(rotHdS[i].coeff(iRow, iCol).real(), rotHdS[i].coeff(iRow, iCol).imag())*(*HresultTwoItr)(itr * 3 * numOfCalcElements + iCol));
			//		}
			//		else if (iRow == 2) {
			//				dSumNCrossRhoRotHdSRhoSparse(0) += rho * (pos.coeff(1)*kv::complex<double>(rotHdS[i].coeff(iRow, iCol).real(), rotHdS[i].coeff(iRow, iCol).imag())*(*HresultTwoItr)(itr * 3 * numOfCalcElements + iCol));
			//				dSumNCrossRhoRotHdSRhoSparse(1) += -rho * (pos.coeff(0)*kv::complex<double>(rotHdS[i].coeff(iRow, iCol).real(), rotHdS[i].coeff(iRow, iCol).imag())*(*HresultTwoItr)(itr * 3 * numOfCalcElements + iCol));
			//		}

			//	}
			//}
		}

		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> dSumNCrossRhoRotHdSRho{ 3,numOfInvertedResisElem };
		dSumNCrossRhoRotHdSRho.reserve(Eigen::VectorXi::Constant(3, 100));

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < nonZeroRowIndices.size(); j++) {
				int invertedRhoID = (*calcElementsVector)[nonZeroRowIndices[j]]->invertedRhoElementsID;
				if (invertedRhoID >= 0 && (dSumNCrossRhoRotHdSRhoSparse(i).d(j).real() != 0.0 || dSumNCrossRhoRotHdSRhoSparse(i).d(j).imag() != 0.0)) {
					dSumNCrossRhoRotHdSRho.coeffRef(i, invertedRhoID) += dSumNCrossRhoRotHdSRhoSparse(i).d(j);
					//dSumNCrossRhoRotHdSRho.coeffRef(i, invertedRhoID).real(dSumNCrossRhoRotHdSRho.coeff(i, invertedRhoID).real() + dSumNCrossRhoRotHdSRhoSparse(i).d(j).real());
					//dSumNCrossRhoRotHdSRho.coeffRef(i, invertedRhoID).imag(dSumNCrossRhoRotHdSRho.coeff(i, invertedRhoID).imag() + dSumNCrossRhoRotHdSRhoSparse(i).d(j).imag());
				}
			}
		}
		

		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < nonZeroRowIndices.size(); j++) {
				int invertedRhoID = (*calcElementsVector)[nonZeroRowIndices[j]]->invertedRhoElementsID;
				if (invertedRhoID >= 0 && (dSumNCrossRhoRotHdSRhoSparse(i).d(j).real() != 0.0 || dSumNCrossRhoRotHdSRhoSparse(i).d(j).imag() != 0.0)) {
					lambdaDRDRho->coeffRef(invertedRhoID) += std::conj(lambdaEachOmega->coeff(itr * 3 * numOfCalcElements + 3 * calcID + i))*dSumNCrossRhoRotHdSRho.coeff(i, invertedRhoID);
				}
			}
		}

	}

	return;
}


void Element::Element::CalcT(int iOmega) {
	Eigen::Matrix2cd H_mat;
	H_mat.coeffRef(0, 0) = H[0].coeff(0);
	H_mat.coeffRef(1, 0) = H[0].coeff(1);
	H_mat.coeffRef(0, 1) = H[1].coeff(0);
	H_mat.coeffRef(1, 1) = H[1].coeff(1);

	Eigen::MatrixXcd Hz{ 1,2 };
	Hz.coeffRef(0,0)= H[0].coeff(2);
	Hz.coeffRef(0,1) = H[1].coeff(2);

	Eigen::MatrixXcd tmpT{ 1,2 };

	tmpT = Hz * H_mat.inverse();

	T[iOmega].coeffRef(0) = tmpT.coeff(0, 0);
	T[iOmega].coeffRef(1) = tmpT.coeff(0, 1);
}


void Element::Element::CalcDTDH(int numOfCalcElements, int iOmega) {
	ub::vector<std::complex<double>> Htmp(6);
	Htmp(0) = H[0].coeff(0);
	Htmp(1) = H[0].coeff(1);
	Htmp(2) = H[0].coeff(2);
	Htmp(3) = H[1].coeff(0);
	Htmp(4) = H[1].coeff(1);
	Htmp(5) = H[1].coeff(2);
	ub::vector<kv::autodif<std::complex<double>>> Hvec;
	Hvec= kv::autodif<std::complex<double>>::init(Htmp);

	ub::vector< kv::autodif<std::complex<double>> > Ttmp(2);
	kv::autodif<std::complex<double>> detA = (Hvec(0)*Hvec(4) - Hvec(1)*Hvec(3));
	Ttmp(0) = 1.0 / detA * (Hvec(4)*Hvec(2) - Hvec(1)*Hvec(5));
	Ttmp(1) = 1.0 / detA * (-Hvec(3)*Hvec(2) + Hvec(0)*Hvec(5));
	
	ub::vector < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> dTdHtmp(2);
	int numOfCalcHComponentsPerOneItr = 3 * numOfCalcElements;
	for (int i = 0; i < 2; i++) {
		dTdHtmp(i).resize(1, 2 * numOfCalcHComponentsPerOneItr);
		dTdHtmp(i).makeCompressed();
		dTdHtmp(i).reserve(6);
		for (int k = 0; k < 6; k++) {
			if (Ttmp(i).d(k).real() != 0.0 || Ttmp(i).d(k).imag() != 0.0) {
				int rowIndex = (k / 3)*numOfCalcHComponentsPerOneItr + (3 * calcID + k % 3);

				dTdHtmp(i).coeffRef(0, rowIndex)=Ttmp(i).d(k);
			}
		}
		dTdH[iOmega](i) = dTdHtmp(i);
		dTdH[iOmega](i).makeCompressed();
		dTdH[iOmega](i).data().squeeze();
	}
}