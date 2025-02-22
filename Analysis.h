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
#define OPTIM_ENABLE_EIGEN_WRAPPERS
//#include "cg.hpp"
//#include "misc/optim_options.hpp"
#include "optim.hpp"
#include <iostream>
#include <vector>
#include <Eigen/SparseCore>
#include <stdio.h>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/PardisoSupport>
#include "Property.h"
#include "ReadData.h"
#include "Output.h"
#include "Element.h"
#include "InvSettings.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <kv/autodif.hpp>
#include <kv/complex.hpp>
#include <time.h>

namespace ub = boost::numeric::ublas;
namespace Functions {
	std::string GetNeighborElement(std::unordered_map<std::string, Element::Element*>* elements, Element::Element* element, Eigen::Vector3i val ,int nx,int ny,int nz);
	std::string GetBinaryValue(int i, int j);
}


namespace Analysis {
	class Analysis {
	public:
		int nx = -1;
		int ny = -1;
		int nz = -1;

		//double minResis = log10(0.1); //This should be given as input
		//double maxResis = log10(1e4); //This should be given as input
		double minResis = 0.001; //This should be given as input
		double maxResis = 1e4; //This should be given as input
		double thresholdRMS = 1.;
		double RMScur = 1e30;
		double weightRoughening;
		Analysis(ReadData::ReadData* readData);
		std::unordered_map<std::string, Element::Element*> elements;
		std::vector<Element::Element*> elementsVector;
		std::unordered_map<int, Property::Property*> properties;
		std::vector<Property::Property*> propertiesVector;
		Boundary::Boundary* boundary;
		InvSettings::InvSettings* invSettings;
		std::vector<ObsData::ObsData*> obsData;
		std::vector<Element::Element*> obsPointElements;
		int numOfObsPointElements;
		int numOfObsTensorComponents = 4; 
		std::vector<InitialData::InitialData*> initialData;
		Output::Output* output;
		Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>* globalMatrix;
		Eigen::SparseMatrix<std::complex< double >,Eigen::RowMajor>* globalMatrix1;
		Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>* globalMatrix2;
		//vector< Eigen::PardisoLU<Eigen::SparseMatrix<std::complex<double>>>*>solverVector;
		//vector< Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>> globalMatrixVector;
		//Eigen::VectorXcd*  globalVector;
		Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor>* globalVector;
		Eigen::SparseMatrix<complex<double> , Eigen::RowMajor>* dRdRho;
		Eigen::VectorXd dUdRho;
		Eigen::SparseMatrix<double , Eigen::RowMajor>* modelWeightMatrix;
		Eigen::VectorXd dRhoDParam;

		Eigen::MatrixXd* jacobian;

		vector<int> numOfObsDataPerOmega;

		double initObjVal=0;
		double dataMisfit;
		double mWTWm;
		vector<double> modelNormalizationCoeff{ 3 };
		int numOfObsData;
		bool isBelowRMSThreshold;
		double RMSpre = 1e30;
		double paramLogNormalization = 0.5;
		double limitOfparamLogNormalization = 10; //to prevent divergence
		double modelConstraintMax = 10;
		double modelConstraintMin = 0.001;
		int numOfCalcModelConstraint = 5;
		double objFuncChangeThresholdForNextmodelConstraint = 0.01;
		int maxIterationPerModelConstraint = 40;
		std::string optMethod = "GD";
		Eigen::VectorXd dDataMisfitDRho;
		Eigen::VectorXd dJdRho;
		vector<Eigen::MatrixXcd, Eigen::aligned_allocator<Eigen::MatrixXcd>> result;
		vector<Eigen::MatrixXcd, Eigen::aligned_allocator<Eigen::MatrixXcd>> result_pre;
		vector<Eigen::MatrixXcd, Eigen::aligned_allocator<Eigen::MatrixXcd>> result_adjoint_pre;
		vector<Eigen::MatrixXcd, Eigen::aligned_allocator<Eigen::MatrixXcd>> resultVector;
		Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>* solver1;
		Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>* solver2;
		Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>* solver1Adjoint;
		Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>* solver2Adjoint;
		Eigen::VectorXcd lambdaEachOmega;
		bool isInitializedSolver1 = false;
		bool isInitializedSolver2 = false;
		double omega;
		int numOfCalcElements;
		int numOfInvertedResistivityElements;
		int numOfDirichletConditionCells;
		int numOfIndependentInvertedResisElements;
		std::unordered_map<std::string, Element::Element*> calcElements;
		std::vector<Element::Element*> calcElementsVector;
		std::unordered_map<int, Element::Element*> invertedRhoIDToElementMap;
		std::vector<Element::Element*> invertedRhoIDToElementVector;
		Eigen::SparseMatrix<double,Eigen::RowMajor>* rougheningMatrix;
		optim::algo_settings_t settings;
		double obj_valPre=1e30;
		std::vector<Element::Element*> notBoundaryElements;
		Eigen::VectorXcd lambdaDRDRho;
		time_t startCalc_t = time(NULL);
		struct CalcLambdaDRDRhoParameters {
			vector<int>startIDVector;
			vector<int>endIDVector;
			vector<Eigen::VectorXcd> lambdaDRDRhoEachThread;
			bool isInitialized = false;
			vector<vector<int>>threadIDGroup;
			vector < vector < Element::Element* >> sameLayerElementsVector;
			int maxLayer;
		};
		Eigen::VectorXd preParams;
		
		std::ofstream infofile;

		
		bool isDirectSolver = true;


		

		bool inheritPreviousObjVal = false;
		int numOfSameModelWeightCalc = 1;
		bool isFirstLoopInheritPreviousObjVal = false;

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

		//std::vector< Eigen::SparseMatrix < std::complex< double >, Eigen::ColMajor>>jacobiH;
		//Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> dZdH;
		//Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> dZdRho;
		int Hpolarization = 0;
		void ClearHAndE();
		void ClearZ();
		void RunAnalysis();
		void Initialize();
		void MakeMatrix(bool isRebuildMatrix = true);
		void Solve(int iOmega,int itr);
		void CalcE(int itr);
		void CalcZ(int iOmega);
		void CalcT(int iOmega);
		void AssociationPropertiesToElements();
		void SetLayerOfElements();
		void SetNeighborElements();
		void SetNotBoundaryElements();
		void SetNumOfCalcElementsAndCalcElementsAndElementsVector();
		void CalcSumNCrossRhoRotHdSElements();
		void SetTransitionZoneElements();
		void CalcNumOfDirichletConditionCells();
		void CalcSurfaceResistivityElements();
		void SetObsDataToElement();
		void CalcLambda(int iOmega);
		void SearchRelatedCalcElements();
		void CalcLambdaDRDRho(const ub::vector<complex<double>>* rhoVec, const vector<Eigen::VectorXcd>* HresultItr);
		void SetInvertedElements();
		void CalcDDataMisfitDRho();
		void CalcDUDRho();
		double CalcDataMisfit();
		void SetDKDRhoElements();
		void CalcDKDRhoElements();
		void CalcForward(bool isCalcInversionValues = true, bool isCalcJacobiMatrix=false);
		void CalcDiffSmoothing();
		void CalcDZDHElements(const ub::vector<kv::complex<double>>* HVecUb,int iOmega);
		void CalcDZDRhoElements(const ub::vector<kv::complex<double>>* rhoVecUb , const ub::vector<kv::complex<double>>*, const int iOmega);
		void CalcDTDHElements(int iOmega);
		void CalcDJDRho();
		void CalcRougheningMatrix();
		double CalcRoughningMatrixPenalty();
		double Optimize(const Eigen::VectorXd& vals_inp,Eigen::VectorXd* grad_out, void* opt_data);
		Eigen::Vector2d  OptimizeUsingJacobian(const Eigen::VectorXd& vals_inp, Eigen::MatrixXd* jac_out);
		void RunOptimize();
		void CountObsData();
		void SetSameResistivityToBoundaryCell();
		bool CalcRhoFromParamAndDRhoDParam(Eigen::VectorXd paramVec);
		Eigen::VectorXd CalcParamFromRho();
		Element::Element* SearchMasterElement(Element::Element* slaveElement);
		void CountIndependentInvertedResisElements();
		void CalcJacobian(int iOmega);
		void SetInitialResistivityFromFile();

	};
}