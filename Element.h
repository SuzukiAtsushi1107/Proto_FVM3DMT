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
#include <unordered_map>    
#include "pch.h"
#include "Property.h"
#include "ConstantValues.h"
#include "ObsData.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <kv/autodif.hpp>
#include <kv/complex.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
namespace ub = boost::numeric::ublas;
using namespace std;

namespace Element {
	class Element 
	{
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		std::string ID = "-1";
		bool isAirGroundBoundaryCell;
		bool isSecondCellOfAirGroundBoundary = false;
		bool isThirdCellOfAirGroundBoundary = false;
		bool isInvertedElement = false;
		int propID = -1;
		Eigen::Vector3d rootCoord = Eigen::Vector3d::Zero();
		Eigen::Vector3d centerCoord =Eigen::Vector3d::Zero();
		double dx;
		double dy;
		double dz;
		int IDX=-1;
		int IDY=-1;
		int IDZ=-1;

		int nx = -1;
		int ny = -1;
		int nz = -1;

		double rhoXY = 0;
		double rhoYX = 0;
		double phiXY = 0;
		double phiYX = 0;
		int calcID; //全体の係数行列の行数に使う
		bool isParent;
		bool isObservationElement = false;
		ObsData::ObsData* impedanceObsData;
		ObsData::ObsData* tipperObsData;

		std::vector<Eigen::Vector3cd, Eigen::aligned_allocator<Eigen::Vector3cd>> E;
		std::vector<Eigen::Vector3cd, Eigen::aligned_allocator<Eigen::Vector3cd>> H;
		std::vector<Eigen::Matrix2cd, Eigen::aligned_allocator<Eigen::Matrix2cd>> Z;
		std::vector<Eigen::Vector2cd, Eigen::aligned_allocator<Eigen::Vector2cd>> T;
		Property::Property* property;
		double resistivity;
		double initialResistivity;
		std::vector<double>* resistivitySurface;
		bool isAlreadyCalcResisCoeff = false;
		std::vector<Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>*> resistivitySurfaceCoeff;
		std::vector<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>*> diffResistivitySurfaceCoeff;
		int layer;
		bool isAlreadyCalcRotHdS = false;
		std::string boundary = "NOT_BOUNDARY";
		vector<string> alreadyFoundNeighborID{27,"NOT_FOUND"};
		vector<Element*> neighborElements{ 27 };
		Eigen::SparseMatrix<double, Eigen::RowMajor> dKDRho;
		map<string,Element*> relatedNeighborCalcElementsMap;
		vector<Element*> relatedNeighborCalcElementsVector;
		int invertedRhoElementsID;
		vector<ub::matrix < Eigen::SparseMatrix<std::complex<double>,Eigen::RowMajor>>>dZdH;
		vector<ub::matrix < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>>dZdRho;

		vector<ub::vector < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>>dTdH;

		bool isInversionImpedance = false;
		bool isInversionTipper = false;
		Element* masterResistivityElement=nullptr;

		Eigen::Vector3cd lambda1 = Eigen::Vector3cd::Zero();
		Eigen::Vector3cd lambda2 = Eigen::Vector3cd::Zero();

		std::vector<int>nonZeroRowIndices;
		int maxResisIndexInSameRowsOfMatrix = -1;
		double maxResistivityInSameRowsOfMatrix = 0;

		double debug;

		double roughenMatrixUnit = 1.0;

		void SearchChildrenElements(unordered_map<string, Element*>* elements, map<string, Element*>* elementsMap);
		void SearchRelatedCalcElements(unordered_map<string, Element*>* elements);
		void ClearHAndE();
		void ClearZ();
		void InitializeHAndEAndZ(int nOmega);
		void SetNeighborElements(unordered_map<string,Element*> *elements);
		void CalcSurfaceResistivity(unordered_map<string, Element*>* elements, vector<Element*>* calcElementsVector,int numOfCalcElements);
		void CalcCenterResistivityCoeff(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* resisCoeff,double coeff);
		void CalcSurfaceRelatedResistivityCoeff(unordered_map<string, Element*>* elements, int numOfCalcElements, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* resisCoeff, double coeff,Eigen::Vector3i pos);
		void CalcInterpolatedRhoInElementCoeff(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map < string, Element* >* elements, int numChildElements, Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* resisCoeff);



		vector<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>> rhoRotHdS;
		vector<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>> rotHdS;
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* sumNCrossRhoRotHdS;
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>* sumNCrossRhoRotHdSPerResistivityPerDxDyDz;
		vector<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>> nCrossRotHdS;
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> GetSumNCrossRhoRotHdS(bool isNeededStandardizationVal = false);
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> CalcRotHdS(unordered_map < string, Element* >* elements, const int numOfCalcElements, Eigen::Vector3i pos); //must be calculated from elements of deeper layers.
		void SetTransitionZone(unordered_map < string, Element* > *elements);
		Eigen::SparseMatrix<double, Eigen::RowMajor> CalcInterpolatedVectorInElement(Eigen::Vector3i val,Eigen::Vector3d x0, unordered_map<string, Element*> *elements, int numChildElements);
		double CalcInterpolatedRhoInElement(Eigen::Vector3i val, Eigen::Vector3d x0, unordered_map < string, Element* >* elements, int numChildElements);

		double CalcCenterResistivity(unordered_map<string, Element*>* elements, int numChildElements);
		Eigen::SparseMatrix<double, Eigen::RowMajor> CalcCenterVector(unordered_map<string, Element*> *elements, int numChildElements);
		void CalcNearestNeighborVectorEdge(Eigen::Vector3d x0, unordered_map<string, Element*> *elements, int numChildElements, Eigen::SparseMatrix<double, Eigen::RowMajor>* row, double* distance);
		std::tuple<Eigen::SparseMatrix<double, Eigen::RowMajor>, double, string> CalcNearestNeighborVectorNode(Eigen::Vector3d x0, unordered_map<string, Element*>* elements, int numChildElements);
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>  CalcSumNCrossRhoRotHdS(unordered_map<string, Element*> *elements, int numOfCalcElements);

		Eigen::Vector3cd CalcE(Eigen:: SparseMatrix<std::complex< double >, Eigen::ColMajor>* Hresult, unordered_map<string, Element*>* elements, int numOfCalcElements, int itr, bool isReturn=false);
		Eigen::Vector3cd CalcE(Eigen::VectorXcd* Hresult, unordered_map<string, Element*>* elements, int numOfCalcElements,int itr, bool isReturn = false);
		ub::matrix<kv::autodif<kv::complex<double>>> CalcDEDH(ub::vector<kv::autodif<kv::complex<double>>>* HresultTwoItr, std::vector<int>nonZeroRowIndices,unordered_map<string, Element*>* elements, int numOfCalcElements);
		ub::matrix<kv::autodif<kv::complex<double>>> CalcDEDRho(ub::vector<kv::autodif<kv::complex<double>>>* rhoVec, ub::vector<kv::complex<double>>* Hresult , std::vector<int>nonZeroRowIndices,  int numOfCalcElements);

		void CalcZ(unordered_map<string, Element*>* elements, int numOfCalcElements, int iOmega);
		Eigen::Matrix2cd CalcZ(vector < Eigen::Vector3cd> E, vector < Eigen::Vector3cd>H);

		Eigen::Vector3cd CalcDEDH(int derID, int numOfCalcElements, unordered_map<string, Element*>* elements);
		void CalcDZDH(const ub::vector<kv::complex<double>>* HTwoItr, unordered_map<string, Element*>* elements, int numOfCalcElements,int iOmega);
		void CalcDZDRho(const ub::vector<kv::complex<double>>* rhoVecUb,const ub::vector<kv::complex<double>>* HresultTwoItr, const vector<Element*>* calcElementsVector,const int numOfCalcElements, const int iOmega);

		void CalcT(int iOmega);
		void CalcDTDH(int numOfCalcElements, int iOmega);


		void CalcLambdaDSumNCrossRhoRotHdSDRho(std::unordered_map<std::string, Element*>* elements,const ub::vector<complex<double>>* rhoVec,
			const vector<Eigen::VectorXcd>* HresultTwoItr,const vector<Element*>* calcElementsVector,
			const int numOfCalcElements,const int numOfInvertedResisElem,const Eigen::VectorXcd* lambdaEachOmega, Eigen::VectorXcd* lambdaDRDRho);

	};
}