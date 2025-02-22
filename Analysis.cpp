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
#define EIGEN_USE_MKL_ALL
#define OPTIM_ENABLE_EIGEN_WRAPPERS
//#define OPTIM_USE_OPENMP Comment out because openmp is used in each loop in the function Optimize()
#pragma once
#include "optim.hpp"
#include <vector>
#include <Eigen/PardisoSupport>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseLU> 
#include "pch.h"
#include <iostream>
#include "Analysis.h"
#include "ConstantValues.h"
#include "Output.h"
#include <omp.h>
#include <mkl.h>
#include <time.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/array.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <kv/autodif.hpp>
#include <kv/complex.hpp>
#include "Property.h"
#include "InvSettings.h"
#include <iomanip>

using namespace std;
using namespace ConstantValues;
Analysis::Analysis::Analysis(ReadData::ReadData* readData) {
	elementsVector = readData->elementsVector;
	elements = readData->elements;
	for (auto itr = readData->properties.begin(); itr != readData->properties.end(); itr++) {
		properties[itr->first] = itr->second;
	}
	propertiesVector = readData->propertiesVector;
	boundary = readData->boundary;
	obsData = readData->obsData;
	invSettings = readData->invSettings;
	output =readData->output;
	initialData = readData->initialData;

    //変数初期化
	maxResis = invSettings->maxResis;
	minResis = invSettings->minResis;
	paramLogNormalization = invSettings->paramLogNormalization;
	modelConstraintMax = invSettings->modelConstraintMax;
	modelConstraintMin = invSettings->modelConstraintMin;
}
void Analysis::Analysis::CalcForward(bool isCalcInversionValues, bool isCalcJacobiMatrix) {

	for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
		omega = boundary->omega[iOmega];
		solver1 = new Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>;
		solver2 = new Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>;
		result.clear();
		result.shrink_to_fit();
		for (int i = 0; i < 2; i++) {
			Hpolarization = i;
			cout << "Making Matrix.." << endl;
			time_t start_t = time(NULL);
			if (i == 0) {
				MakeMatrix(true);
			}
			else if (i == 1) {
				MakeMatrix(false);
			}
			if (isDirectSolver == false) {
				//Copy For CalcLambda
				//if (i == 0) {
				//	//delete globalMatrix1;
				//	//globalMatrix1 = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>;
				//	//globalMatrix1->resize(3 * numOfCalcElements, 3 * numOfCalcElements);
				//	//globalMatrix1->reserve(Eigen::VectorXi::Constant(3 * numOfCalcElements, 81));

				//	//for (int j = 0; j < 3 * numOfCalcElements; j++) {
				//	//	for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*globalMatrix, j); it; ++it)
				//	//	{
				//	//		globalMatrix1->coeffRef(j, it.col()) = globalMatrix->coeffRef(j, it.col());
				//	//	}
				//	//}
				//	*globalMatrix1 = *globalMatrix;
				//}
				//else {
				//	//delete globalMatrix2;
				//	//globalMatrix2 = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>;
				//	//globalMatrix2->resize(3 * numOfCalcElements, 3 * numOfCalcElements);
				//	//globalMatrix2->reserve(Eigen::VectorXi::Constant(3 * numOfCalcElements, 81));

				//	//for (int j = 0; j < 3 * numOfCalcElements; j++) {
				//	//	for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*globalMatrix, j); it; ++it)
				//	//	{
				//	//		globalMatrix2->coeffRef(j, it.col()) = globalMatrix->coeffRef(j, it.col());
				//	//	}
				//	//}
				//	*globalMatrix2 = *globalMatrix;
				//}
			}
			time_t end_t = time(NULL);
			std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
			cout << "End Making Matrix." << endl;
			Solve(iOmega, i);

			CalcE(i);
		}
		CalcZ(iOmega);
		CalcT(iOmega);

		//for (int i = 0; i < numOfObsPointElements; i++) {
		//	cout << obsPointElements[i]->boundary << endl;
		//	cout << "Z:" << obsPointElements[i]->Z[iOmega] << endl;
		//}
		if (isCalcInversionValues) {
			ub::vector < kv::autodif<kv::complex<double>>> HTwoItr;
			ub::vector<kv::complex<double>> HVecUb(3 * 2 * numOfCalcElements);
			vector<Eigen::VectorXcd> Hvec{ 2 };
			Hvec[0].resize(3 * numOfCalcElements);
			Hvec[1].resize(3 * numOfCalcElements);
			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 3 * numOfCalcElements; j++) {
					HVecUb(i * 3 * numOfCalcElements + j) = kv::complex<double>(result[i].coeff(j, 0).real(), result[i].coeff(j, 0).imag());
					Hvec[i].coeffRef(j) = result[i].coeff(j, 0);
				}
			}
			CalcDZDHElements(&HVecUb,iOmega);
			CalcDTDHElements(iOmega);
			if (isCalcJacobiMatrix == false) {
				std::cout << "Calculating Lambda for Sensitivity matrix." << std::endl;
				CalcLambda(iOmega);
				delete solver1; //to prevent large memory use 
				delete solver2;
				std::cout << "End of Calculating Lambda for Sensitivity matrix." << std::endl;
			}
			ub::vector< kv::complex<double>> rhoVecUb(numOfCalcElements);
			ub::vector< complex<double>> rhoVec(numOfCalcElements);
			for (int i = 0; i < numOfCalcElements; i++) {
				rhoVecUb(i) = calcElementsVector[i]->resistivity;
				rhoVec(i) = calcElementsVector[i]->resistivity;
			}

			CalcDZDRhoElements(&rhoVecUb, &HVecUb, iOmega);
			std::cout << "End of calculating dZdRho." << std::endl;
			if (isCalcJacobiMatrix == false) {
				CalcLambdaDRDRho(&rhoVec, &Hvec);
				std::cout << "End of calculating LambdaDRdRho." << std::endl;
			}
			if (isCalcJacobiMatrix == true) {
				CalcJacobian(iOmega);
			}
		}
		else {
			delete solver1;
			delete solver2;
		}
		std::cout << ("Output files..") << std::endl;
		//output->VTKFileOputput(omega, &elements);
		//output->VTKFileOputput(omega, &elements,"PHI");
		//output->VTKFileOputput(omega, &elements,"H");
		//output->VTKFileOputput(omega, &elements,"E");
		
		output->TxtOutputAppRho(omega, &calcElements);
		output->AppRhoOutputSurface(omega, &elements);
		output->PhiOutputSurface(omega, &elements);
		output->TipperOutputSurface(iOmega,omega, &elements);
		ClearHAndE();
	}


	output->ImpedanceOutputSurface(boundary->omega, &elements);
	output->TipperOutputSurface(boundary->omega, &elements);
}

void Analysis::Analysis::RunAnalysis() {

	std::cout << ("Initialize Data") << std::endl;
	Initialize();
	std::cout << ("Initialization End") << std::endl;
	
		


	CalcForward(false);
		

	ClearZ();


	
}
void Analysis::Analysis::Initialize() {
	SetNumOfCalcElementsAndCalcElementsAndElementsVector();
	cout << "Number of calculated elements: " << numOfCalcElements << endl;
	AssociationPropertiesToElements();
	SetLayerOfElements();
	
	SetNeighborElements();
	SetTransitionZoneElements();
	SetNotBoundaryElements();

	SetObsDataToElement();
	SetInitialResistivityFromFile();
	CountObsData();

	output->RhoOutput(&elements);
	output->TxtOutputResistivity(&elements, "InitialResis.txt");
	std::cout << "Calc Surface Resistivity.." << std::endl;;
	CalcSurfaceResistivityElements();
	std::cout << "End Calc Surface Resistivity.." << std::endl;
	
	std::cout << "Calc SumNCrossRhoRotHdSElements.." << std::endl;
	CalcSumNCrossRhoRotHdSElements();
	std::cout << "End Calc SumNCrossRhoRotHdSElements.." << std::endl;
	CalcNumOfDirichletConditionCells();

	//Initiialze globalMatrix
	globalMatrix = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>;
	globalMatrix->resize( 3 * numOfCalcElements, 3 * numOfCalcElements);

	/*globalMatrix->setZero();
	globalMatrix->makeCompressed();
	globalMatrix->reserve(3 * numOfCalcElements * 81);*/
	globalMatrix->reserve(Eigen::VectorXi::Constant(3 * numOfCalcElements, 81));
	//if (isDirectSolver == false) {
		globalMatrix1 = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>;
		globalMatrix1->resize(3 * numOfCalcElements, 3 * numOfCalcElements);
		//globalMatrix1->setZero();
		//globalMatrix1->makeCompressed();
		//globalMatrix1->reserve(3 * numOfCalcElements * 81);
		globalMatrix1->reserve(Eigen::VectorXi::Constant(3 * numOfCalcElements, 81));

		globalMatrix2 = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>;
		globalMatrix2->resize(3 * numOfCalcElements, 3 * numOfCalcElements);
		//globalMatrix2->setZero();
		//globalMatrix2->makeCompressed();
		//globalMatrix2->reserve(3 * numOfCalcElements * 81);
		globalMatrix2->reserve(Eigen::VectorXi::Constant(3 * numOfCalcElements, 81));
	//}

	globalVector = new Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor>;
	globalVector->resize( 3 * numOfCalcElements, 1 );
	globalVector->setZero();
	globalVector->makeCompressed();
	globalVector->reserve(numOfDirichletConditionCells);
	//Set resultVector;
	resultVector.resize(2*boundary->omega.size());
	for (int i = 0; i < 2*boundary->omega.size(); i++) {
		resultVector[i].resize(3 * numOfCalcElements,1);
	}
	//Set result_pre;
	result_pre.resize(2*boundary->omega.size());
	result_adjoint_pre.resize(2 * boundary->omega.size());
	for (int i = 0; i < 2*boundary->omega.size(); i++) {
		result_pre[i].resize( 3 * numOfCalcElements, 1);
		result_pre[i].setZero();
		result_adjoint_pre[i].resize(3 * numOfCalcElements, 1);
		result_adjoint_pre[i].setZero();
	}
	//Set H and E Vector Size
	for (int i = 0; i < numOfCalcElements; i++) {
		calcElementsVector[i]->InitializeHAndEAndZ(boundary->omega.size());
	}
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	calcElementsVector[i]->H.resize(2);
	//	calcElementsVector[i]->E.resize(2);
	//	for (int ii = 0; ii < 2; ii++) {
	//		calcElementsVector[i]->H[ii].resize(2);
	//		calcElementsVector[i]->E[ii].resize(2);
	//		calcElementsVector[i]->H[ii].setZero();
	//		calcElementsVector[i]->E[ii].setZero();
	//	}
	//}
	////Set Z Vector Size
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	calcElementsVector[i]->Z.resize(boundary->omega.size());
	//	calcElementsVector[i]->dZdH.resize(boundary->omega.size());
	//	calcElementsVector[i]->dZdRho.resize(boundary->omega.size());
	//	for (int ii = 0; ii < boundary->omega.size(); ii++) {
	//		calcElementsVector[i]->dZdH[ii].resize(2, 2);
	//		calcElementsVector[i]->dZdRho[ii].resize(2, 2);
	//	}
	//}

	//lambda.resize(boundary->omega.size() * 2 * 2* 3 * numOfCalcElements);
	lambdaEachOmega.resize(2 * 3 * numOfCalcElements);
	lambdaEachOmega.setZero();
	SearchRelatedCalcElements();
	SetInvertedElements();
	rougheningMatrix = new Eigen::SparseMatrix<double, Eigen::RowMajor>;
	rougheningMatrix->resize(6*numOfInvertedResistivityElements, numOfInvertedResistivityElements);
	//rougheningMatrix->resize(27*numOfInvertedResistivityElements, numOfInvertedResistivityElements);
	rougheningMatrix->setZero();
	CalcRougheningMatrix();
	//SetDKDRhoElements();
	//CalcDKDRhoElements();
	lambdaDRDRho =  Eigen::VectorXcd{ numOfInvertedResistivityElements };
	lambdaDRDRho.setZero();
	//dRdRho = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>;
	//dRdRho->resize( boundary->omega.size() * 2 * 3 * numOfCalcElements,numOfInvertedResistivityElements );
	//dRdRho->reserve(Eigen::VectorXi::Constant(boundary->omega.size() * 2 * 3 * numOfCalcElements,13));//this valuable is used as uncompressed mode for parallelization
	//dRdRho->reserve(243);
	dDataMisfitDRho.resize(numOfInvertedResistivityElements);
	dDataMisfitDRho.setZero();

	dJdRho.resize(numOfInvertedResistivityElements);
	dJdRho.setZero();

	//jacobian = new Eigen::MatrixXd;
	//jacobian->resize(numOfObsData, numOfInvertedResistivityElements);
	//jacobian->setZero();


	dRhoDParam.resize(numOfInvertedResistivityElements);
	dRhoDParam.setZero();
	modelWeightMatrix=new Eigen::SparseMatrix<double, Eigen::RowMajor>;
	modelWeightMatrix->makeCompressed();
	modelWeightMatrix->resize(numOfInvertedResistivityElements, numOfInvertedResistivityElements);
	modelWeightMatrix->reserve(numOfInvertedResistivityElements);
	
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		modelWeightMatrix->coeffRef(i, i) = 1.0; //this would be changed in the future, but at present this is set 1 to all elements.  
	}
	//CalcResistivityAtTransitionElements();
}
void Analysis::Analysis::CalcSurfaceResistivityElements() {
	int maxLayer = 0;
	for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = *itr;
		if (element->layer > maxLayer) {
			maxLayer = element->layer;
		}
	}
	vector < vector < Element::Element* >> sameLayerElementsVector;
	for (int iLayer = 0; iLayer <= maxLayer; iLayer++) {
		vector < Element::Element* > layerElementsVector;
		layerElementsVector.reserve(numOfCalcElements);
		for (int i = 0; i < calcElementsVector.size(); i++) {
			Element::Element* element = calcElementsVector[i];
			layerElementsVector.push_back(element);

		}
		sameLayerElementsVector.push_back(layerElementsVector);
	}

	for (int iLayer = 0; iLayer <= maxLayer; iLayer++) { //this is needed because larger layer elements need the result of lower one.
		vector < Element::Element* >tmpVector = sameLayerElementsVector[iLayer];
		for (int i = 0; i < tmpVector.size(); i++) {
			Element::Element* element = tmpVector[i];
			element->CalcSurfaceResistivity(&elements, &calcElementsVector, numOfCalcElements);
		}
	}

//#pragma omp parallel for
//	for (int i = 0; i < numOfCalcElements; i++) {
//		Element::Element* element = calcElementsVector[i];
//		element->CalcSurfaceResistivity(&elements,&calcElementsVector, numOfCalcElements, &modelNormalizationCoeff);
//		
//	}
}
void Analysis::Analysis::Solve(int iOmega,int itr) {
	Eigen::PardisoLU<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>* solver;
	
	//Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> iterativeSolver;
	//Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>, Eigen::IncompleteLUT<std::complex<double>>> iterativeSolver;
	//Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> iterativeSolver;
	//Eigen::LeastSquaresConjugateGradient < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> iterativeSolver;
	Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> iterativeSolver;
	iterativeSolver.setMaxIterations(invSettings->maxIterationBiCGSTAB);
	//iterativeSolver.preconditioner().setFillfactor(1);
	//iterativeSolver.preconditioner().setDroptol(1e-2);

	Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>* M;
	mkl_set_num_threads(omp_get_max_threads());
	if (itr == 0) {
		M = globalMatrix1;
	}
	else {
		M = globalMatrix2;
	}
	//if (itr == 0 && isInitializedSolver1==false) {
	//	isInitializedSolver1 = true;
	//	solver1 = new Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>;
	//}
	//else if (itr == 1 && isInitializedSolver2 == false) {
	//	isInitializedSolver2 = true;
	//	solver2 = new Eigen::PardisoLU < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>>;
	//}
	if (itr == 0) {
		solver = solver1;
	}
	else {
		solver = solver2;
	}

	std::cout << "Solve H.. #Omega: " << iOmega << " #Polarization:" << itr << endl;
	time_t start_t = time(NULL);
	//if ((itr == 0 && !isInitializedSolver1) || (itr == 1 && !isInitializedSolver2)) {
	//	solver->analyzePattern(globalMatrix);
	//	solverAdjoint->analyzePattern(globalMatrix.adjoint());
	//	if (itr == 0) { isInitializedSolver1 = true; }
	//	else { isInitializedSolver2 = true; }

	//}
	
	if (isDirectSolver == true) {
		solver->compute(*M);
		if (solver->info() != Eigen::Success) {
			std::cout << "decomposition failed" << std::endl;
			//exit(1);
		}
		else {
			Eigen::MatrixXcd tmp{ 3 * numOfCalcElements,1 };
			Eigen::VectorXcd res;
			res = solver->solve(*globalVector);
			for (int i = 0; i < 3 * numOfCalcElements; i++) {
				tmp.coeffRef(i, 0) = res.coeff(i);
				resultVector[2 * iOmega + itr].coeffRef(i, 0) = res.coeff(i);
			}
			result.push_back(res);
			if (isDirectSolver == false) {
				for (int i = 0; i < 3 * numOfCalcElements; i++) {
					result_pre[2*iOmega+itr].coeffRef(i) = res.coeff(i);
				}

			}
		}
	}
	else {
		Eigen::MatrixXcd tmp{ 3 * numOfCalcElements,1 };
		Eigen::VectorXcd res;
		iterativeSolver.compute(*M);
		iterativeSolver.setTolerance(invSettings->toleranceIterativeSolver);
		//res = iterativeSolver.solve(*globalVector);
		res = iterativeSolver.solveWithGuess(globalVector->toDense(),result_pre[2*iOmega+itr]);
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			resultVector[2 * iOmega + itr].coeffRef(i, 0) = res.coeff(i);
		}
		result.push_back(res);
		std::cout << "In BiCGSTAB #iterations:     " << iterativeSolver.iterations() << std::endl;
		std::cout << "In BiCGSTAB estimated error: " << iterativeSolver.error() << std::endl;
		std::cout << "In BICGSTAB Last Iteration, Relative Change of Solution:" << iterativeSolver.lastRelativeSolChange() << std::endl;

		//Update Pre Sol
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			result_pre[2 * iOmega + itr].coeffRef(i) = res.coeff(i);
		}
	}
	
	time_t end_t = time(NULL);
	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
	for (auto itrElem = calcElementsVector.begin(); itrElem != calcElementsVector.end(); itrElem++) {
		Element::Element* element = *itrElem;
		Eigen::Vector3cd tmp;
		tmp.coeffRef(0) = result.back().coeff(3 * element->calcID,0);
		tmp.coeffRef(1) = result.back().coeff(3 * element->calcID + 1,0);
		tmp.coeffRef(2) = result.back().coeff(3 * element->calcID + 2,0);
		element->H[itr].coeffRef(0) = tmp.coeff(0);
		element->H[itr].coeffRef(1) = tmp.coeff(1);
		element->H[itr].coeffRef(2) = tmp.coeff(2);
	}
//	start_t = time(NULL);
//	
//	for (int i = 0; i < numOfInvertedRhoElements;i++) {
//		Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor> tmp{ 3 * numOfCalcElements,1 };
//		tmp = invertedRhoIDToElementMap[i]->dKDRho.cast<std::complex<double>>() * result[itr];
//		tmp = (*solver).solve(tmp);
//		tmp = -tmp;
////#pragma omp parallel for
//		for (int j = 0; j < numOfObsPointElements; j++) {
//			Element::Element* element = obsPointElements[j];
//			for (int k = 0; k < element->relatedNeighborCalcElementsVector.size(); k++) {
//				int calcID = element->relatedNeighborCalcElementsVector[k]->calcID;
//				if (tmp.coeff(3 * calcID,0) != 0.0) {
//					dHdRho[iOmega][itr].coeffRef(3 * calcID, i) = tmp.coeff(3 * calcID, 0); //データ数×インバージョンする要素数のヤコビアンを作るのに必要最低限なHのみ保存（それ以外は0で問題ない）
//				}
//			}
//				
//		}
//	}
//	end_t = time(NULL);
//	std::cout << "Calc dH/dRho Time:" << (end_t - start_t) << " seconds" << std::endl;
	//solverVector.push_back(solver);

	//Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>> > solver;
	//globalMatrix->makeCompressed();
	//solver.compute(*globalMatrix);
	//if (solver.info() != Eigen::Success) {
	//	std::cout << "decomposition failed" << std::endl;
	//	exit(1);
	//}
	//else {
	//	Eigen::VectorXcd tmp;
	//	result.push_back(tmp);
	//	result.back().resize(3 * (int)calcElementsVector.size());
	//	result.back() = solver.solve(*globalVector);
	//}
	//std::cout << "#iterations:     " << solver.iterations() << std::endl;
	//std::cout << "estimated error: " << solver.error() << std::endl;

	//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
	//	Element::Element* element = *itr;
	//	Eigen::Vector3cd tmp;
	//	tmp.coeffRef(0) = result.back().coeff(3 * element->calcID);
	//	tmp.coeffRef(1) = result.back().coeff(3 * element->calcID + 1);
	//	tmp.coeffRef(2) = result.back().coeff(3 * element->calcID + 2);
	//	element->H.push_back(tmp);
	//}

}
void Analysis::Analysis::SetNotBoundaryElements() {
	
	int numElem = 0;
	for (int i = 0; i < numOfCalcElements; i++) {
		if (calcElementsVector[i]->boundary == "NOT_BOUNDARY") {
			numElem++;
		}
	}
	notBoundaryElements.resize(numElem);
	for (int i = 0; i < numOfCalcElements; i++) {
		if (calcElementsVector[i]->boundary == "NOT_BOUNDARY") {
			notBoundaryElements.push_back(calcElementsVector[i]);
		}
	}
}
void Analysis::Analysis::ClearHAndE() {
	for (int i = 0; i < calcElementsVector.size(); i++) {
		Element::Element* element = calcElementsVector[i];
		element->ClearHAndE(); 
	}
}
void Analysis::Analysis::ClearZ() {
	for (int i = 0; i < calcElementsVector.size(); i++) {
		Element::Element* element = calcElementsVector[i];
		element->ClearZ();
	}
}
void Analysis::Analysis::CalcE(int itr) {
	Eigen::VectorXcd Hresult{ 3 * numOfCalcElements };
	for (int i = 0; i < 3*calcElementsVector.size(); i++) {
		Hresult.coeffRef(i) = result.back().coeff(i, 0);
	}
	#pragma omp parallel for
	for (int i = 0; i < calcElementsVector.size();i++) {
	//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = calcElementsVector[i];
		element->CalcE(&Hresult,&elements,numOfCalcElements, itr); //本当はHの計算と整合性が取れるよう深いレイヤーから計算していき、浅いレイヤはその和にしないといけない？→rotHdSがそのようにして算出しているので、なってるはず
																  //if (Hpolarization == 0) {
		//	cout << element->E.back() <<" H:" << element->H.back() << " coord:" << element->rootCoord << endl;
		//}
	}

}
void Analysis::Analysis::CalcZ(int iOmega) {
	#pragma omp parallel for
	for (int i = 0; i < calcElementsVector.size(); i++) {
	//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = calcElementsVector[i];
		element->CalcZ(&elements, numOfCalcElements,iOmega);
		element->rhoXY = pow(std::sqrt(std::pow(element->Z[iOmega].coeff(0, 1).real(), 2.0) + std::pow(element->Z[iOmega].coeff(0, 1).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
		element->rhoYX = pow(std::sqrt(std::pow(element->Z[iOmega].coeff(1, 0).real(), 2.0) + std::pow(element->Z[iOmega].coeff(1, 0).imag(), 2.0)), 2.0) / omega / ConstantValues::mu;
		double tmp = 0;
		if (element->Z[iOmega].coeff(0, 1).real() != 0) {
			tmp = element->Z[iOmega].coeff(0, 1).imag() / element->Z[iOmega].coeff(0, 1).real();
			element->phiXY = std::atan(tmp) / ConstantValues::pi * 180;
		}
		else {
			element->phiXY = 0;
		}
		if (element->Z[iOmega].coeff(1, 0).real() != 0) {
			tmp = element->Z[iOmega].coeff(1, 0).imag() / element->Z[iOmega].coeff(1, 0).real();
			element->phiYX = std::atan(tmp) / ConstantValues::pi * 180;
		}
		else {
			element->phiYX = 0;
		}

		

	}
	
}
void Analysis::Analysis::CalcT(int iOmega) {
#pragma omp parallel for
	for (int i = 0; i < calcElementsVector.size(); i++) {
		//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = calcElementsVector[i];
		element->CalcT(iOmega);
	}

}
void Analysis::Analysis::SetNeighborElements() {
	//#pragma omp parallel for
	//for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
	for (int i = 0; i < elementsVector.size(); i++) {
		Element::Element* element = elementsVector[i];
		element->SetNeighborElements(&elements);
	}
}
void Analysis::Analysis::MakeMatrix(bool isRebuildMatrix) {
	if (isRebuildMatrix == true) {
		//if (globalMatrix) {
		//	delete globalMatrix;
		//	globalMatrix = nullptr;
		//}
		//if (globalVector) {
		//	delete globalVector;
		//	globalVector = nullptr;
		//}
		//globalMatrix = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>{ 3 * numOfCalcElements,3 * numOfCalcElements };
		////globalVector = new Eigen::VectorXcd{ 3 * numOfCalcElements };
		//globalVector = new Eigen::SparseMatrix < std::complex< double >, Eigen::ColMajor>{ 3 * numOfCalcElements,1 };
		////必要な配列要素数を確保
		//globalMatrix->reserve(3 * numOfCalcElements * 243);
		//globalVector->reserve(numOfDirichletConditionCells);

		//delete globalMatrix;
		//globalMatrix = new Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>;
		//globalMatrix->resize( 3 * numOfCalcElements, 3 * numOfCalcElements );
		////globalMatrix->setZero();
		////globalMatrix.makeCompressed();
		////globalMatrix.prune(1e-6);
		////globalMatrix->makeCompressed();
		////globalMatrix->reserve(3 * numOfCalcElements * 81);
		//globalMatrix->reserve(Eigen::VectorXi::Constant(3 * numOfCalcElements, 243));


		delete globalVector;
		globalVector = new Eigen::SparseMatrix<std::complex< double >, Eigen::ColMajor>;
		globalVector->resize( 3 * numOfCalcElements, 1 );
		//globalVector->setZero();
		//globalVector.makeCompressed();
		//globalVector.prune(1e-6);
		globalVector->makeCompressed();
		globalVector->reserve(numOfDirichletConditionCells);
	}
	std::vector<Eigen::SparseMatrix < std::complex<double >, Eigen::RowMajor >*> globalMatrixTmp(numOfCalcElements); //for parallel
	std::vector < Eigen::VectorXcd, Eigen::aligned_allocator<Eigen::VectorXcd>> globalVectorTmp(numOfCalcElements);
	for (int i = 0; i < numOfCalcElements; i++) {
		globalMatrixTmp[i] = new Eigen::SparseMatrix < std::complex<double >, Eigen::RowMajor>{ 3, 3 * numOfCalcElements };
		//globalMatrixTmp[i].resize(3, 3 * numOfCalcElements);
		globalMatrixTmp[i]->reserve(Eigen::VectorXi::Constant(3, 243));
		//globalMatrixTmp[i].makeCompressed();
		//globalMatrixTmp[i].reserve(243);
		globalVectorTmp[i].resize(3);
		globalVectorTmp[i].setZero();
	}

	//行列組み立て
	#pragma omp parallel for
	//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		double dv = element->dx * element->dy * element->dz;
		if (element->boundary == "NOT_BOUNDARY") {
			if (isRebuildMatrix == true) {

				//globalMatrixTmp[element->calcID] = element->GetSumNCrossRhoRotHdS(false);
				*(globalMatrixTmp[element->calcID]) = element->GetSumNCrossRhoRotHdS(false);
				
				//globalMatrixTmp[element->calcID].row(0) = element->GetSumNCrossRhoRotHdS().row(0);
				//globalMatrixTmp[element->calcID].row(1) = element->GetSumNCrossRhoRotHdS().row(1);
				//globalMatrixTmp[element->calcID].row(2) = element->GetSumNCrossRhoRotHdS().row(2);
				
				//double unit = element->resistivity*dv;//normalization
				double unit = 1.0;

				std::complex<double> term;
				term.imag(+omega * mu * dv);

				globalMatrixTmp[element->calcID]->coeffRef(0, 3 * element->calcID) += term / unit;
				globalMatrixTmp[element->calcID]->coeffRef(1, 3 * element->calcID + 1) += term / unit;
				globalMatrixTmp[element->calcID]->coeffRef(2, 3 * element->calcID + 2) += term / unit;

			}
			else {
				continue;
			}
		}
		else if (element->boundary == "-X_BOUNDARY") {
			//vector < Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>>centerVal;
			//for (int i = 0; i < 2; i++) {
			//	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//	tmp.makeCompressed();
			//	tmp.reserve(6);
			//	centerVal.push_back(tmp);
			//}
			//centerVal[0].coeffRef(0, 3 * element->calcID) = 1;
			//centerVal[0].coeffRef(1, 3 * element->calcID + 1) = 1;
			//centerVal[0].coeffRef(2, 3 * element->calcID + 2) = 1;
			//Eigen::Vector3i pos;
			//pos[0] = 1;
			//pos[1] = 0;
			//pos[2] = 0;
			//string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			//if (Hpolarization == 0) {
			//	centerVal[1].coeffRef(0, 3 * elements[neighborID]->calcID) = 1;
			//	centerVal[1].coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 0.0;
			//	centerVal[1].coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 0.0;
			//}
			//else {
			//	centerVal[1].coeffRef(0, 3 * elements[neighborID]->calcID) = 0.0;
			//	centerVal[1].coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 1;
			//	centerVal[1].coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 1;
			//}
			//Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//tmp.makeCompressed();
			//tmp.reserve(12);//6*2
			//tmp = centerVal[1] - centerVal[0];
			//globalMatrixTmp[element->calcID] = tmp;

			
			Eigen::Vector3i pos;
			pos[0] = 1;
			pos[1] = 0;
			pos[2] = 0;
			string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			globalMatrixTmp[element->calcID]->coeffRef(0, 3 * element->calcID) = -1.0;
			globalMatrixTmp[element->calcID]->coeffRef(1, 3 * element->calcID + 1) = -1.0;
			globalMatrixTmp[element->calcID]->coeffRef(2, 3 * element->calcID + 2) = -1.0;
			if (Hpolarization == 0) {
				globalMatrixTmp[element->calcID]->coeffRef(0, 3 * elements[neighborID]->calcID) = 1;
				globalMatrixTmp[element->calcID]->coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 0.0;
				globalMatrixTmp[element->calcID]->coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 0.0;
			}
			else {
				globalMatrixTmp[element->calcID]->coeffRef(0, 3 * elements[neighborID]->calcID) = 0.0;
				globalMatrixTmp[element->calcID]->coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 1;
				globalMatrixTmp[element->calcID]->coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 1;
			}
		}
		else if (element->boundary == "+X_BOUNDARY") {
			//vector < Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>>centerVal;
			//for (int i = 0; i < 2; i++) {
			//	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//	tmp.makeCompressed();
			//	tmp.reserve(6);
			//	centerVal.push_back(tmp);
			//}
			//centerVal[0].coeffRef(0, 3 * element->calcID) = 1;
			//centerVal[0].coeffRef(1, 3 * element->calcID + 1) = 1;
			//centerVal[0].coeffRef(2, 3 * element->calcID + 2) = 1;
			//Eigen::Vector3i pos;
			//pos[0] = -1;
			//pos[1] = 0;
			//pos[2] = 0;
			//string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			//if (Hpolarization == 0) {
			//	centerVal[1].coeffRef(0, 3 * elements[neighborID]->calcID) = 1;
			//	centerVal[1].coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 0.0;
			//	centerVal[1].coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 0.0;
			//}
			//else {
			//	centerVal[1].coeffRef(0, 3 * elements[neighborID]->calcID) = 0.0;
			//	centerVal[1].coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 1;
			//	centerVal[1].coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 1;
			//}
			//Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//tmp.makeCompressed();
			//tmp.reserve(12);//6*2
			//tmp = centerVal[0] - centerVal[1];
			//globalMatrixTmp[element->calcID] = tmp;

			Eigen::Vector3i pos;
			pos[0] = -1;
			pos[1] = 0;
			pos[2] = 0;
			string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			globalMatrixTmp[element->calcID]->coeffRef(0, 3 * element->calcID) = 1.0;
			globalMatrixTmp[element->calcID]->coeffRef(1, 3 * element->calcID + 1) = 1.0;
			globalMatrixTmp[element->calcID]->coeffRef(2, 3 * element->calcID + 2) = 1.0;
			if (Hpolarization == 0) {
				globalMatrixTmp[element->calcID]->coeffRef(0, 3 * elements[neighborID]->calcID) = -1.0;
				globalMatrixTmp[element->calcID]->coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 0.0;
				globalMatrixTmp[element->calcID]->coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 0.0;
			}
			else {
				globalMatrixTmp[element->calcID]->coeffRef(0, 3 * elements[neighborID]->calcID) = 0.0;
				globalMatrixTmp[element->calcID]->coeffRef(1, 3 * elements[neighborID]->calcID + 1) = -1.0;
				globalMatrixTmp[element->calcID]->coeffRef(2, 3 * elements[neighborID]->calcID + 2) = -1.0;
			}
		}
		else if (element->boundary == "-Y_BOUNDARY") {
			//vector < Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>>centerVal;
			//for (int i = 0; i < 2; i++) {
			//	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//	tmp.makeCompressed();
			//	tmp.reserve(6);
			//	centerVal.push_back(tmp);
			//}
			//centerVal[0].coeffRef(0, 3 * element->calcID) = 1;
			//centerVal[0].coeffRef(1, 3 * element->calcID + 1) = 1;
			//centerVal[0].coeffRef(2, 3 * element->calcID + 2) = 1;
			//Eigen::Vector3i pos;
			//pos[0] = 0;
			//pos[1] = 1;
			//pos[2] = 0;
			//string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			//if (Hpolarization == 0) {
			//	centerVal[1].coeffRef(0, 3 * elements[neighborID]->calcID) = 1;
			//	centerVal[1].coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 0.0;
			//	centerVal[1].coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 1;
			//}
			//else {
			//	centerVal[1].coeffRef(0, 3 * elements[neighborID]->calcID) = 0.0;
			//	centerVal[1].coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 1;
			//	centerVal[1].coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 0.0;
			//}
			//Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//tmp.makeCompressed();
			//tmp.reserve(12);//6*2
			//tmp = centerVal[1] - centerVal[0];
			//globalMatrixTmp[element->calcID] = tmp;

			Eigen::Vector3i pos;
			pos[0] = 0;
			pos[1] = 1;
			pos[2] = 0;
			string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			globalMatrixTmp[element->calcID]->coeffRef(0, 3 * element->calcID) = -1.0;
			globalMatrixTmp[element->calcID]->coeffRef(1, 3 * element->calcID + 1) = -1.0;
			globalMatrixTmp[element->calcID]->coeffRef(2, 3 * element->calcID + 2) = -1.0;
			if (Hpolarization == 0) {
				globalMatrixTmp[element->calcID]->coeffRef(0, 3 * elements[neighborID]->calcID) = 1.0;
				globalMatrixTmp[element->calcID]->coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 0.0;
				globalMatrixTmp[element->calcID]->coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 1.0;
			}
			else {
				globalMatrixTmp[element->calcID]->coeffRef(0, 3 * elements[neighborID]->calcID) = 0.0;
				globalMatrixTmp[element->calcID]->coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 1.0;
				globalMatrixTmp[element->calcID]->coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 0.0;
			}
		}
		else if (element->boundary == "+Y_BOUNDARY") {
			//vector < Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>>centerVal;
			//for (int i = 0; i < 2; i++) {
			//		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//		tmp.makeCompressed();
			//		tmp.reserve(6);
			//		centerVal.push_back(tmp);
			//	}
			//	centerVal[0].coeffRef(0, 3 * element->calcID) = 1;
			//	centerVal[0].coeffRef(1, 3 * element->calcID + 1) = 1;
			//	centerVal[0].coeffRef(2, 3 * element->calcID + 2) = 1;
			//	Eigen::Vector3i pos;
			//	pos[0] = 0;
			//	pos[1] = -1;
			//	pos[2] = 0;
			//	string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			//	if (Hpolarization == 0) {
			//		centerVal[1].coeffRef(0, 3 * elements[neighborID]->calcID) = 1;
			//		centerVal[1].coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 0.0;
			//		centerVal[1].coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 1;
			//	}
			//	else {
			//		centerVal[1].coeffRef(0, 3 * elements[neighborID]->calcID) = 0.0;
			//		centerVal[1].coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 1;
			//		centerVal[1].coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 0.0;
			//	}
			//	Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//	tmp.makeCompressed();
			//	tmp.reserve(12);//6*2
			//	tmp = centerVal[0] - centerVal[1];
			//	globalMatrixTmp[element->calcID] = tmp;

			Eigen::Vector3i pos;
			pos[0] = 0;
			pos[1] = -1;
			pos[2] = 0;
			string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			globalMatrixTmp[element->calcID]->coeffRef(0, 3 * element->calcID) = 1.0;
			globalMatrixTmp[element->calcID]->coeffRef(1, 3 * element->calcID + 1) = 1.0;
			globalMatrixTmp[element->calcID]->coeffRef(2, 3 * element->calcID + 2) = 1.0;
			if (Hpolarization == 0) {
				globalMatrixTmp[element->calcID]->coeffRef(0, 3 * elements[neighborID]->calcID) = -1.0;
				globalMatrixTmp[element->calcID]->coeffRef(1, 3 * elements[neighborID]->calcID + 1) = 0.0;
				globalMatrixTmp[element->calcID]->coeffRef(2, 3 * elements[neighborID]->calcID + 2) = -1.0;
			}
			else {
				globalMatrixTmp[element->calcID]->coeffRef(0, 3 * elements[neighborID]->calcID) = 0.0;
				globalMatrixTmp[element->calcID]->coeffRef(1, 3 * elements[neighborID]->calcID + 1) = -1.0;
				globalMatrixTmp[element->calcID]->coeffRef(2, 3 * elements[neighborID]->calcID + 2) = 0.0;
			}
		}
		else if (element->boundary == "+Z_BOUNDARY") {
			//vector < Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>>centerVal;
			//for (int i = 0; i < 2; i++) {
			//		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//		tmp.makeCompressed();
			//		tmp.reserve(6);
			//		centerVal.push_back(tmp);
			//	}
			//	centerVal[0].coeffRef(0, 3 * element->calcID) = 1;
			//	centerVal[0].coeffRef(1, 3 * element->calcID + 1) = 1;
			//	centerVal[0].coeffRef(2, 3 * element->calcID + 2) = 1;
			//	//All Components are zero
			//	globalMatrixTmp[element->calcID] = centerVal[0];


			globalMatrixTmp[element->calcID]->coeffRef(0, 3 * element->calcID) = 1.0;
			globalMatrixTmp[element->calcID]->coeffRef(1, 3 * element->calcID + 1) = 1.0;
			globalMatrixTmp[element->calcID]->coeffRef(2, 3 * element->calcID + 2) = 1.0;
			//	//All Components are zero
		}
		else if (element->boundary == "-Z_BOUNDARY") {
			//vector < Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>, Eigen::aligned_allocator<Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>>>centerVal;
			//for (int i = 0; i < 2; i++) {
			//		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> tmp{ 3,3 * numOfCalcElements };
			//		tmp.makeCompressed();
			//		tmp.reserve(6);
			//		centerVal.push_back(tmp);
			//	}
			//	centerVal[0].coeffRef(0, 3 * element->calcID) = 1;
			//	centerVal[0].coeffRef(1, 3 * element->calcID + 1) = 1;
			//	centerVal[0].coeffRef(2, 3 * element->calcID + 2) = 1;
			//	globalMatrixTmp[element->calcID] = centerVal[0];
			//	if (Hpolarization == 0) {
			//		globalVectorTmp[element->calcID].coeffRef(0) = 1;
			//	}
			//	else {
			//		globalVectorTmp[element->calcID].coeffRef(1) = 1;
			//	}
			Eigen::Vector3i pos;
			pos[0] = 0;
			pos[1] = -1;
			pos[2] = 0;
			string neighborID = Functions::GetNeighborElement(&elements, element, pos, nx, ny, nz);
			globalMatrixTmp[element->calcID]->coeffRef(0, 3 * element->calcID) = 1.0;
			globalMatrixTmp[element->calcID]->coeffRef(1, 3 * element->calcID + 1) = 1.0;
			globalMatrixTmp[element->calcID]->coeffRef(2, 3 * element->calcID + 2) = 1.0;
			if (Hpolarization == 0) {
				globalVectorTmp[element->calcID].coeffRef(0) = 1;
			}
			else {
				globalVectorTmp[element->calcID].coeffRef(1) = 1;
			}

		}

	}
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		//globalMatrixTmp[element->calcID]->makeCompressed(); //I don't know why this is needed.. But this is needed. Perhaps because Lazy Eval in MiddleRows and when evaluation globalMatrixTmp has been deleted.
		//globalMatrixTmp[element->calcID]->data().squeeze();
		if (isRebuildMatrix == true) {
			//globalMatrix->middleRows(3 * i, 3) = globalMatrixTmp[i];
			for (int j = 3 * i; j < 3 * i + 3; j++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*globalMatrixTmp[i], j - 3 * i); it; ++it)
				{
					globalMatrix1->coeffRef(j, it.col()) = globalMatrixTmp[i]->coeff(j - 3 * i, it.col());
					globalMatrix2->coeffRef(j, it.col()) = globalMatrixTmp[i]->coeff(j - 3 * i, it.col());

				}
			}
			//globalMatrix->row(3 * i) = globalMatrixTmp[i].row(0);
			//globalMatrix->row(3 * i + 1) = globalMatrixTmp[i].row(1);
			//globalMatrix->row(3 * i + 2) = globalMatrixTmp[i].row(2);
			if (element->boundary == "-Z_BOUNDARY") {
				globalVector->coeffRef(3 * i, 0) = globalVectorTmp[i].coeff(0);
				globalVector->coeffRef(3 * i + 1, 0) = globalVectorTmp[i].coeff(1);
				globalVector->coeffRef(3 * i + 2, 0) = globalVectorTmp[i].coeff(2);
			}
		}
		else {
			if (element->boundary != "NOT_BOUNDARY") {
				//globalMatrix->middleRows(3 * i, 3) = globalMatrixTmp[i];
				for (int j = 3 * i; j < 3 * i + 3; j++) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*globalMatrixTmp[i], j - 3 * i); it; ++it)
					{
						if (Hpolarization == 0)
						{
							globalMatrix1->coeffRef(j, it.col()) = globalMatrixTmp[i]->coeff(j - 3 * i, it.col());
						}
						else {
							globalMatrix2->coeffRef(j, it.col()) = globalMatrixTmp[i]->coeff(j - 3 * i, it.col());
						}
					}
				}
				//globalMatrix->row(3 * i) = globalMatrixTmp[i].row(0);
				//globalMatrix->row(3 * i + 1) = globalMatrixTmp[i].row(1);
				//globalMatrix->row(3 * i + 2) = globalMatrixTmp[i].row(2);
				if (element->boundary == "-Z_BOUNDARY") {
					globalVector->coeffRef(3 * i, 0) = globalVectorTmp[i].coeff(0);
					globalVector->coeffRef(3 * i + 1, 0) = globalVectorTmp[i].coeff(1);
					globalVector->coeffRef(3 * i + 2, 0) = globalVectorTmp[i].coeff(2);
				}
			}
		}
		
	}
	//globalMatrix->makeCompressed();


	//globalMatrix->eval();

	//globalMatrix->data().squeeze();

	for (int i = 0; i < numOfCalcElements; i++) {
		delete globalMatrixTmp[i];
	}
	//globalMatrixVector.push_back(*globalMatrix);
}

void Analysis::Analysis::CalcSumNCrossRhoRotHdSElements() {
	int maxLayer = 0;
	for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = *itr;
		if (element->layer > maxLayer) {
			
			maxLayer = element->layer;
		}
	}
	vector < vector < Element::Element* >> sameLayerElementsVector;
	for (int iLayer = 0; iLayer <= maxLayer; iLayer ++ ) {
		vector < Element::Element* > layerElementsVector;
		for (int i = 0; i < calcElementsVector.size(); i++) {
			Element::Element* element = calcElementsVector[i];
			if (element->layer == iLayer && element->boundary == "NOT_BOUNDARY") {
				layerElementsVector.push_back(element);
			}
		}
		sameLayerElementsVector.push_back(layerElementsVector);
	}
	cout <<"MaxLayer:"<< maxLayer << endl;

	for (int iLayer = maxLayer; iLayer >= 0; iLayer--) {
		vector < Element::Element* >tmpVector = sameLayerElementsVector[iLayer];
		//this would be faster by compiling layer by layer and parallelize
#pragma omp parallel for
		for (int i = 0; i < tmpVector.size(); i++) {
		//for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
			Element::Element* element = sameLayerElementsVector[iLayer][i];
			element->CalcSumNCrossRhoRotHdS(&elements, numOfCalcElements);
		}
	}
	//Eigen::setNbThreads(0);
	Eigen::initParallel();
}
void Analysis::Analysis::SetTransitionZoneElements() {
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		Element::Element* element = *itr;
		element->SetTransitionZone(&elements);
	}
}
void Analysis::Analysis::SetNumOfCalcElementsAndCalcElementsAndElementsVector() {
	numOfCalcElements = 0;
	double minDx = 10000000000;
	double minDy = 10000000000;
	double minDz = 10000000000;
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		
		Element::Element* element = *itr;
		if (element->isParent == false) {
			calcElementsVector.push_back(element);
			calcElements[element->ID] = element;
			element->calcID = numOfCalcElements;
			numOfCalcElements++;
			if (element->dx < minDx) {
				minDx = element->dx;
			}
			if (element->dy < minDy) {
				minDy = element->dy;
			}
			if (element->dz < minDz) {
				minDz = element->dz;
			}
		}
	}


	cout << "minimum dx:" << minDx << " minimum dy:" << minDy << " minimum dz:" << minDz << endl;
}

//
void Analysis::Analysis::AssociationPropertiesToElements() {
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		Element::Element* element = *itr;
		bool findProp = false;
		for (auto itr2 = propertiesVector.begin(); itr2 != propertiesVector.end(); itr2++) {
			Property::Property* property = *itr2;
			int propID = property->ID;
			if (element->propID == propID) {
				element->property = property;
				element->resistivity = property->resistivity;
				element->initialResistivity= property->resistivity;
				findProp = true;
			}
		}
		if (!findProp) {
			std::cout << "No Property ID which set to Element in Data" << std::endl;
			exit(1);
		}


	}
}
//
//
//
//void Analysis::Analysis::CalcResistivityAtTransitionElements() {
//	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
//		Element::Element* element=*itr;
//		bool isAirCellAtBoundaryAirGround = false;
//		if (element->property->type == Property::Property::AIR) {
//			Element::Element* groundElement;
//			for (auto itr2 = element->facesVector.begin(); itr2 != element->facesVector.end(); itr2++) {
//				Face::Face* face = *itr2;
//				for (auto itr3 = face->elementsVector.begin(); itr3 != face->elementsVector.end(); itr3++) {
//					Element::Element* tmpElement=*itr3;
//					if (tmpElement->property->type != Property::Property::AIR) {
//						isAirCellAtBoundaryAirGround = true;
//						groundElement = tmpElement;
//						break;
//					}
//				}
//				if (isAirCellAtBoundaryAirGround == true) {
//					break;
//				}
//
//			}
//
//			if (isAirCellAtBoundaryAirGround == true) {
//				//element->resistivity = (element->resistivity + groundElement->resistivity) / 2;
//				element->resistivity = ((0.5*element->resistivity.inverse()) +
//					(0.5*groundElement->resistivity.inverse())).inverse();
//				//element->resistivity = groundElement->resistivity;
//			}
//
//		}
//	}
//}
void Analysis::Analysis::SetLayerOfElements() {

	int maxNx = 0;
	int maxNy = 0;
	int maxNz = 0;
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		Element::Element* element = *itr;
		element->layer = (int)(element->ID.length()-9) / 2 - 1; //9 is nx,ny,nz place

		int nztmp = std::stoi(element->ID.substr(0, 3));
		int nytmp = std::stoi(element->ID.substr(3, 3));
		int nxtmp = std::stoi(element->ID.substr(6, 3));
		element->IDX = nxtmp;
		element->IDY = nytmp;
		element->IDZ = nztmp;
		if (maxNx < nxtmp) {
			maxNx = nxtmp;
		}
		if (maxNy < nytmp) {
			maxNy = nytmp;
		}
		if (maxNz < nztmp) {
			maxNz = nztmp;
		}
	}
	nx = maxNx + 1;
	ny = maxNy + 1;
	nz = maxNz + 1;
	for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
		Element::Element* element = *itr;
		element->nx = nx;
		element->ny = ny;
		element->nz = nz;
	}


	int maxLayer = 0;
	for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = *itr;
		if (element->layer > maxLayer) {
			maxLayer = element->layer;
		}
	}

	//set Roughen Matrix Normalization Unit Value

	for (int iLayer = 0; iLayer <= maxLayer; iLayer++) {
		double minDx = 10000000000;
		double minDy = 10000000000;
		double minDz = 10000000000;
		for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
			Element::Element* element= *itr;
			if (element->layer == iLayer) {
				if (element->dx < minDx) {
					minDx = element->dx;
				}
				if (element->dy < minDy) {
					minDy = element->dy;
				}
				if (element->dz < minDz) {
					minDz = element->dz;
				}
			}
		}
		for (auto itr = elementsVector.begin(); itr != elementsVector.end(); itr++) {
			Element::Element* element = *itr;
			if (element->layer == iLayer) {
				element->roughenMatrixUnit = std::max(std::max(minDx, minDy), minDz);
			}
		}
	}

}
void Analysis::Analysis::CalcNumOfDirichletConditionCells() {
	numOfDirichletConditionCells = 0;
	for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
		Element::Element* element = *itr;
		if (element->boundary == "-Z_BOUNDARY") {
			numOfDirichletConditionCells++;
		}
	}
}


void Analysis::Analysis::SetObsDataToElement() {
	numOfObsPointElements = 0;
	obsPointElements.resize(0);
	for (auto itr = elements.begin(); itr != elements.end(); itr++) {
		Element::Element* element = itr->second;
		bool isObsElement = false;
		if (element->property->type == Property::Property::AIR) {
			continue;
		}
		Eigen::Vector3i pos;
		//for (int i = 0; i < 6; i++) {
		pos.setZero();
		pos.coeffRef(2) = -1;
		int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
		if (!(element->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && element->isParent == false && elements[element->alreadyFoundNeighborID[ipos]]->isAirGroundBoundaryCell == true && element->property->type != Property::Property::AIR)) {
			continue;
		}

		Element::Element* tmpElement = element;
		Element::Element* obsElement = element;
		while (true) {
			bool isFoundElement = true;
			for (int i = 0; i < 3; i++) {
				for (int j = 0; j < 3; j++) {
					for (int k = 0; k < 3; k++) {
						ipos = i + 3 * j + 9 * k;
						if (tmpElement->alreadyFoundNeighborID[ipos].find("BOUNDARY") == string::npos && tmpElement->neighborElements[ipos]->property->type == Property::Property::AIR) {
							isFoundElement = false;
						}
					}
				}
			}
			if (isFoundElement) {
				obsElement = tmpElement;
				break;
			}
			else {
				tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 2]; //1つ深いセルへ
				//tmpElement = tmpElement->neighborElements[1 + 3 + 9 * 1]; 
			}
		}
		obsPointElements.reserve(obsData.size());// max size we should consider is  obsData.size().
		for (int i = 0; i < obsData.size(); i++) {
			if (obsData[i]->isImpedanceData == true) {
				double x = obsData[i]->coord.coeff(0);
				double y = obsData[i]->coord.coeff(1);
				//cout << obsElement->rootCoord.coeff(0) << " " << x << " " << obsElement->rootCoord.coeff(1) << " " << y << endl;
				if (obsElement->rootCoord.coeff(0) <= x && obsElement->rootCoord.coeff(0) + obsElement->dx > x && obsElement->rootCoord.coeff(1) <= y && obsElement->rootCoord.coeff(1) + obsElement->dy > y) {
					if (obsData[i]->isAlreadyFoundElementImpedance == false) {
						obsData[i]->isAlreadyFoundElementImpedance = true;
						obsElement->impedanceObsData = obsData[i];
						if (obsElement->isObservationElement == false) {
							obsElement->isObservationElement = true;
							obsPointElements.push_back(obsElement);
							numOfObsPointElements++;
						}
						obsElement->isInversionImpedance = true;
						if (obsElement->boundary != "NOT_BOUNDARY") {
							std::cout << "ERROR:Obs Data Location is in Boundary Cell." << std::endl;
							exit(1);
						}
					}
				}
			}
			if (obsData[i]->isTipperData == true) {
				double x = obsData[i]->coord.coeff(0);
				double y = obsData[i]->coord.coeff(1);
				//cout << obsElement->rootCoord.coeff(0) << " " << x << " " << obsElement->rootCoord.coeff(1) << " " << y << endl;
				if (obsElement->rootCoord.coeff(0) <= x && obsElement->rootCoord.coeff(0) + obsElement->dx > x && obsElement->rootCoord.coeff(1) <= y && obsElement->rootCoord.coeff(1) + obsElement->dy > y) {
					if (obsData[i]->isAlreadyFoundElementTipper == false) {
						obsData[i]->isAlreadyFoundElementTipper = true;
						obsElement->tipperObsData = obsData[i];
						if (obsElement->isObservationElement == false) {
							obsElement->isObservationElement = true;
							obsPointElements.push_back(obsElement);
							numOfObsPointElements++;
						}
						obsElement->isInversionTipper = true;
						if (obsElement->boundary != "NOT_BOUNDARY") {
							std::cout << "ERROR:Obs Data Location is in Boundary Cell." << std::endl;
							exit(1);
						}
					}
				}
			}

		}	
	}
	for (int i = 0; i < obsData.size(); i++) {
		if (obsData[i]->isImpedanceData==true && obsData[i]->isAlreadyFoundElementImpedance == false) {
			std::cout << "Zobs has data out of range." << std::endl;
			std::cout << std::fixed;
			std::cout << "X:"<<std::setprecision(5) << obsData[i]->coord.coeff(0) << endl;
			std::cout << "Y:"<< std::setprecision(5) << obsData[i]->coord.coeff(1) << endl;
			exit(1);
		}
		if (obsData[i]->isTipperData==true && obsData[i]->isAlreadyFoundElementTipper == false) {
			std::cout << "Tobs has data out of range." << std::endl;
			std::cout << std::fixed;
			std::cout << "X:" << std::setprecision(5) << obsData[i]->coord.coeff(0) << endl;
			std::cout << "Y:" << std::setprecision(5) << obsData[i]->coord.coeff(1) << endl;
			exit(1);
		}
	}

}

void Analysis::Analysis::CalcLambda(int iOmega) {
	lambdaEachOmega.setZero();

	//====Calc ∂J/∂H==========
	Eigen::VectorXcd dJdH{ 2 * 3 * numOfCalcElements }; //iteration*(real and imag part)*(Hx,Hy and Nz)*numOfCalcElements
	dJdH.setZero();

	//===Impedance=====
	//====Calc ∂/∂H_i　Σ_j (Zcalc_j-ZObs_j)**2

	//std::ofstream f2;
	//f2.open("debugZtmpdZdHtmp.txt", std::ios::trunc);
	for (int j = 0; j < numOfObsPointElements; j++) {
		Element::Element* element = obsPointElements[j];
		if (element->isInversionImpedance == true) {
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdH[iOmega](ii, jj), 0); it; ++it)
					{
						int iCol = it.col();
						/*cout << "in calcLambda" << ii << " " << jj << " " << element->dZdH[iOmega](ii, jj).coeff(0, 13981) << endl;*/
						std::complex<double> dZtmp = element->Z[iOmega].coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
						std::complex<double> dZdHtmp = element->dZdH[iOmega](ii, jj).coeff(0, iCol);
						//f2 << iCol << " " << ii << " " << jj << " " << dZtmp << " " << dZdHtmp << "before" << endl;
						double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
						double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));


						//Real Part
						if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0 && element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj).real() != 0) {
							dZtmp.real(dZtmp.real() / epsReal);
							dZdHtmp.real(dZdHtmp.real() / epsReal);
							//cout << "dZtmpReal:" << dZtmp.real() << endl;
							//cout << "dZdHtmpReal:" << dZdHtmp.real() << endl;
						}
						else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
							dZtmp.real(0.0);
						}
						else {
							//そのまま
						}
						if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0 && element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj).imag() != 0) {
							dZtmp.imag(dZtmp.imag() / epsImag);
							dZdHtmp.imag(dZdHtmp.imag() / epsImag);
							//cout << "dZtmpImag:" << dZtmp.imag() << endl;
							//cout << "dZdHtmpImag:" << dZdHtmp.imag() << endl;
						}
						else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
							dZtmp.imag(0.0);
						}
						else {
							//そのまま
						}
						dJdH.coeffRef(iCol).real(dJdH.coeffRef(iCol).real() + 2.0 * (std::conj(dZtmp)*dZdHtmp).real());
						//f2 << iCol << " " << dZtmp << " " << dZdHtmp << "middlle" << endl;

						//Imag Part
						dZdHtmp = element->dZdH[iOmega](ii, jj).coeff(0, iCol);
						if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0 && element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj).real() != 0) {
							dZdHtmp.real(dZdHtmp.real() / epsImag); //ひっくり返る
							//cout << "dZtmpReal:" << dZtmp.real() << endl;
							//cout << "dZdHtmpReal:" << dZdHtmp.real() << endl;
						}
						else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
							dZdHtmp.real(0.0);
						}
						else {
							//そのまま
						}
						if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0 && element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj).imag() != 0) {
							dZdHtmp.imag(dZdHtmp.imag() / epsReal); //ひっくり返る
								//cout << "dZtmpImag:" << dZtmp.imag() << endl;
								//cout << "dZdHtmpImag:" << dZdHtmp.imag() << endl;
						}
						else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
							dZdHtmp.imag(0.0);
						}
						else {
							//そのまま
						}
						dJdH.coeffRef(iCol).imag(dJdH.coeffRef(iCol).imag() + 2.0 * (std::conj(dZtmp)*dZdHtmp).imag());
					}

				}
			}

		}
	}

	//===Tipper=====
	//====Calc ∂/∂H_i　Σ_j (Tcalc_j-TObs_j)**2

	for (int j = 0; j < numOfObsPointElements; j++) {
		Element::Element* element = obsPointElements[j];
		if (element->isInversionTipper == true) {
			for (int ii = 0; ii < 2; ii++) {
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dTdH[iOmega](ii), 0); it; ++it)
				{
					int iCol = it.col();
					std::complex<double> dTtmp = element->T[iOmega].coeff(ii) - element->tipperObsData->TobsVector[iOmega].coeff(ii);
					std::complex<double> dTdHtmp = element->dTdH[iOmega](ii).coeff(0, iCol);
					double epsReal = std::abs(element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii));
					double epsImag = std::abs(element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii));


					//Real Part
					if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) > 0 && element->tipperObsData->TobsVector[iOmega].coeff(ii).real() != 0) {
						dTtmp.real(dTtmp.real() / epsReal);
						dTdHtmp.real(dTdHtmp.real() / epsReal);
					}
					else if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) <= 0) {
						dTtmp.real(0.0);
					}
					else {
						//そのまま
					}
					if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) > 0 && element->tipperObsData->TobsVector[iOmega].coeff(ii).imag() != 0) {
						dTtmp.imag(dTtmp.imag() / epsImag);
						dTdHtmp.imag(dTdHtmp.imag() / epsImag);

					}
					else if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) <= 0) {
						dTtmp.imag(0.0);
					}
					else {
						//そのまま
					}
					dJdH.coeffRef(iCol).real(dJdH.coeffRef(iCol).real() + 2.0 * (std::conj(dTtmp)*dTdHtmp).real());

					//Imag Part
					dTdHtmp = element->dTdH[iOmega](ii).coeff(0, iCol);
					if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) > 0 && element->tipperObsData->TobsVector[iOmega].coeff(ii).real() != 0) {
						dTdHtmp.real(dTdHtmp.real() / epsImag); //ひっくり返る
					}
					else if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) <= 0) {
						dTdHtmp.real(0.0);
					}
					else {
						//そのまま
					}
					if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) > 0 && element->tipperObsData->TobsVector[iOmega].coeff(ii).imag() != 0) {
						dTdHtmp.imag(dTdHtmp.imag() / epsReal); //ひっくり返る
					}
					else if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) <= 0) {
						dTdHtmp.imag(0.0);
					}
					else {
						//そのまま
					}
					dJdH.coeffRef(iCol).imag(dJdH.coeffRef(iCol).imag() + 2.0 * (std::conj(dTtmp)*dTdHtmp).imag());

					//debug
					//double preJ = CalcDataMisfit();
					//int itr = iCol / (3 * numOfCalcElements);
					//int elemNum = iCol % (3 * numOfCalcElements)/3;
					//int direcH = iCol % 3;
					//cout << "Tx or Ty:" << ii << endl;
					//cout << "eleNum:" << elemNum << " "
					//	<< "itr:" << itr << " "
					//	<< "direcH:" << direcH << endl;
					//cout <<"H1:"<< calcElementsVector[elemNum]->H[0] << endl;
					//cout << "H2:" << calcElementsVector[elemNum]->H[1] << endl;
					//cout << "dTxdHz1:" << calcElementsVector[elemNum]->dTdH[iOmega](0).coeffRef(0,3*elemNum+2) << endl;
					//cout << "dTxdHz2:" << calcElementsVector[elemNum]->dTdH[iOmega](0).coeffRef(0,3*numOfCalcElements + 3 * elemNum + 2) << endl;
					//cout << "dTydHz1:" << calcElementsVector[elemNum]->dTdH[iOmega](1).coeffRef(0, 3 * elemNum + 2) << endl;
					//cout << "dTydHz2:" << calcElementsVector[elemNum]->dTdH[iOmega](1).coeffRef(0, 3 * numOfCalcElements + 3 * elemNum + 2) << endl;

					//complex<double> dH = 0.001;
					//calcElementsVector[elemNum]->H[itr].coeffRef(direcH) += dH;
					//calcElementsVector[elemNum]->CalcT(iOmega);
					//double postJ = CalcDataMisfit();
					//cout<< "numerical Real:" << (postJ - preJ) / dH << endl;
					//calcElementsVector[elemNum]->H[itr].coeffRef(direcH) -= dH;
					//dH = complex < double>( 0 , 0.001);
					//calcElementsVector[elemNum]->H[itr].coeffRef(direcH) += dH;
					//calcElementsVector[elemNum]->CalcT(iOmega);
					//postJ = CalcDataMisfit();
					//cout <<"numerical Imag:" << (postJ - preJ) / dH << endl;
					//calcElementsVector[elemNum]->H[itr].coeffRef(direcH) -= dH;
					//calcElementsVector[elemNum]->CalcT(iOmega);
					//cout << "analysis:" << 2.0 * (std::conj(dTtmp)*dTdHtmp) << endl;

				}
			}
		}
	}

	// Todo::他のテンソル量を逆解析する場合はここに足す
	

	//====Calc adjoint(∂R/∂H) = adjoint(A of Ax=b) =======================
	//====Hiterごとに分けて計算できる(RがH1とH2で独立）ので分けて計算
	//====H1========
	{
		time_t start_t = time(NULL);
		/*Eigen::SparseMatrix<double, Eigen::RowMajor> M{ 2 * 2 * 3 * numOfCalcElements,2 * 2 * 3 * numOfCalcElements };
		
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> globalMatrixAdjointTmp;
		globalMatrixAdjointTmp = Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>{ 3 * numOfCalcElements, 3 * numOfCalcElements };
		globalMatrixAdjointTmp.setZero();
		globalMatrixAdjointTmp = globalMatrix1->adjoint();*/
		/*Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> solver;
		solver.compute(globalMatrixAdjointTmp);*/


		Eigen::VectorXcd vec{ 3 * numOfCalcElements };

		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			vec.coeffRef(i) = std::conj(dJdH.coeff(i));
		}



		Eigen::VectorXcd res;
		if (isDirectSolver == true) {
			solver1->pardisoParameterArray()(11) = 1; //set to adjoint mode
			res = solver1->solve(vec);
			solver1->pardisoParameterArray()(11) = 0; //reset to normal mode
			if (isDirectSolver == false) {
				for (int i = 0; i < 3 * numOfCalcElements; i++) {
					result_adjoint_pre[2 * iOmega].coeffRef(i) = res.coeff(i);
				}
			}
		}
		else {
			//Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>,
			//	Eigen::IncompleteLUT<std::complex<double>>> iterativeSolver;
			//Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>, Eigen::IncompleteLUT<std::complex<double>>> iterativeSolver;
			Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> iterativeSolver;
			//Eigen::LeastSquaresConjugateGradient < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> iterativeSolver;
			iterativeSolver.setMaxIterations(invSettings->maxIterationBiCGSTAB);
			//iterativeSolver.preconditioner().setFillfactor(1);
			//iterativeSolver.preconditioner().setDroptol(1e-2);

			Eigen::SparseMatrix < complex<double>, Eigen::RowMajor> Madjoint = globalMatrix1->adjoint();

			//for (int i = 0; i < Madjoint.outerSize(); ++i) {
			//	complex<double> factor = Madjoint.coeff(i,i);

			//	//factor = Madjoint.coeff(i, i);
			//	for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(Madjoint, i); it; ++it)
			//	{
			//		Madjoint.coeffRef(i, it.col()) /= factor;
			//	}
			//	vec.coeffRef(i) /= factor;
			//}

			//iterativeSolver.compute(globalMatrix1->adjoint());
			iterativeSolver.compute(Madjoint);
			iterativeSolver.setTolerance(invSettings->toleranceIterativeSolver);
			
			//res = iterativeSolver.solve(vec);
			res = iterativeSolver.solveWithGuess(vec, result_adjoint_pre[2 * iOmega]);
			
			std::cout << "In BiCGSTAB #iterations:     " << iterativeSolver.iterations() << std::endl;
			std::cout << "In BiCGSTAB estimated error: " << iterativeSolver.error() << std::endl;
			std::cout << "In BICGSTAB Last Iteration, Relative Change of Solution:" << iterativeSolver.lastRelativeSolChange() << std::endl;

			//Update Pre Sol
			for (int i = 0; i < 3 * numOfCalcElements; i++) {
				result_adjoint_pre[2 * iOmega].coeffRef(i) = res.coeff(i);
			}
		}
		time_t end_t = time(NULL);
		cout << "Calculate Lambda of H1. #Omega:" << iOmega << endl;
		std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
		//cout << "res1" << res << endl;
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			lambdaEachOmega.coeffRef(i) = res.coeff(i, 0);
			//std::cout << lambda.coeffRef(iOmega * 2 * 3 * numOfCalcElements + i) << std::endl;
		}
	}
	//====H2========
	{
		time_t start_t = time(NULL);

		/*Eigen::SparseMatrix<double, Eigen::RowMajor> M{ 2 * 2 * 3 * numOfCalcElements,2 * 2 * 3 * numOfCalcElements };
		
		Eigen::SparseMatrix<complex<double>, Eigen::RowMajor> globalMatrixAdjointTmp;
		globalMatrixAdjointTmp = Eigen::SparseMatrix<std::complex< double >, Eigen::RowMajor>{ 3 * numOfCalcElements, 3 * numOfCalcElements };
		globalMatrixAdjointTmp.setZero();
		globalMatrixAdjointTmp = globalMatrix2->adjoint();
		Eigen::SparseLU<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> solver;
		
		solver.compute(globalMatrixAdjointTmp);*/

		Eigen::VectorXcd vec{ 3 * numOfCalcElements };
		vec.setZero();
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			vec.coeffRef(i) = std::conj(dJdH.coeff(3 * numOfCalcElements + i));
		}
		//std::ofstream f;
		//f.open("debug2.txt", std::ios::trunc);
		//for (int i = 0; i < 3 * numOfCalcElements; i++) {
		//	f << 0 << " " << i << " " << vec.coeffRef(i).real() << " " << dJdH.coeff(3 * numOfCalcElements + i).real() << endl;
		//	f << 0 << " " << i << " " << vec.coeffRef(i).imag() << " " << dJdH.coeff(3 * numOfCalcElements + i).imag() << endl;
		//}
		Eigen::VectorXcd res;
		if (isDirectSolver == true) {
			solver2->pardisoParameterArray()(11) = 1; //set to adjoint mode
			res = solver2->solve(vec);
			solver2->pardisoParameterArray()(11) = 0; //reset to normal mode
			if (isDirectSolver==false) {
				for (int i = 0; i < 3 * numOfCalcElements; i++) {
					result_adjoint_pre[2 * iOmega + 1].coeffRef(i) = res.coeff(i);
				}
			}
		}
		else {
			//Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>,
			//	 Eigen::IncompleteLUT<std::complex<double>>> iterativeSolver;
			//Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>, Eigen::IncompleteLUT<std::complex<double>>> iterativeSolver;
			Eigen::BiCGSTAB<Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> iterativeSolver;
			//Eigen::LeastSquaresConjugateGradient < Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>> iterativeSolver;
			iterativeSolver.setMaxIterations(invSettings->maxIterationBiCGSTAB);
			//iterativeSolver.preconditioner().setFillfactor(1);
			//iterativeSolver.preconditioner().setDroptol(1e-2);

			Eigen::SparseMatrix < complex<double>, Eigen::RowMajor> Madjoint = globalMatrix2->adjoint();

			//for (int i = 0; i < Madjoint.outerSize(); ++i) {
			//	complex<double> factor = Madjoint.coeff(i, i);

			//	//factor = Madjoint.coeff(i, i);
			//	for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(Madjoint, i); it; ++it)
			//	{
			//		Madjoint.coeffRef(i, it.col()) /= factor;
			//	}
			//	vec.coeffRef(i) /= factor;
			//}

			iterativeSolver.compute(Madjoint);
			iterativeSolver.setTolerance(invSettings->toleranceIterativeSolver);

		
			//res = iterativeSolver.solve(vec);
			res = iterativeSolver.solveWithGuess(vec, result_adjoint_pre[2 * iOmega + 1]);
			std::cout << "In BiCGSTAB #iterations:     " << iterativeSolver.iterations() << std::endl;
			std::cout << "In BiCGSTAB estimated error: " << iterativeSolver.error() << std::endl;
			std::cout << "In BICGSTAB Last Iteration, Relative Change of Solution:" << iterativeSolver.lastRelativeSolChange()<< std::endl;
			//Update Pre Sol
			for (int i = 0; i < 3 * numOfCalcElements; i++) {
				result_adjoint_pre[2 * iOmega + 1].coeffRef(i) = res.coeff(i);
			}
		}
		time_t end_t = time(NULL);
		cout << "Calculate Lambda of H2. #Omega:" << iOmega << endl;
		std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
		//cout << "res2" << res << endl;
		for (int i = 0; i < 3 * numOfCalcElements; i++) {
			lambdaEachOmega.coeffRef( 3 * numOfCalcElements + i) = res.coeff(i, 0);
			//std::cout << lambda.coeffRef(iOmega * 2 * 3 * numOfCalcElements +  3 * numOfCalcElements + i) << std::endl;
		}
	}


	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	
	//	Eigen::Vector3cd tmp;
	//	tmp.setZero();

	//	//element->lambda[0] = tmp;
	//	//element->lambda[1] = tmp;
	//	
	//	tmp.coeffRef(0)=lambda.coeffRef(iOmega * 2 * 3 * numOfCalcElements +3*element->calcID+ 0);
	//	tmp.coeffRef(1) = lambda.coeffRef(iOmega * 2 * 3 * numOfCalcElements + 3 * element->calcID + 1);
	//	tmp.coeffRef(2) = lambda.coeffRef(iOmega * 2 * 3 * numOfCalcElements + 3 * element->calcID + 2);
	//	element->lambda1 = tmp;
	//	tmp.coeffRef(0) = lambda.coeffRef(iOmega * 2 * 3 * numOfCalcElements + 3 * numOfCalcElements + 3 * element->calcID + 0);
	//	tmp.coeffRef(1) = lambda.coeffRef(iOmega * 2 * 3 * numOfCalcElements + 3 * numOfCalcElements + 3 * element->calcID + 1);
	//	tmp.coeffRef(2) = lambda.coeffRef(iOmega * 2 * 3 * numOfCalcElements + 3 * numOfCalcElements + 3 * element->calcID + 2);
	//	element->lambda2 = tmp;
	//	//tmp.coeffRef(0) = dJdH.coeffRef(3 * element->calcID + 0);
	//	//tmp.coeffRef(1) = dJdH.coeffRef(3 * element->calcID + 1);
	//	//tmp.coeffRef(2) = dJdH.coeffRef(3 * element->calcID + 2);
	//	//element->lambda1 = tmp;
	//	//tmp.coeffRef(0) = dJdH.coeffRef(+ 3 * numOfCalcElements + 3 * element->calcID + 0);
	//	//tmp.coeffRef(1) = dJdH.coeffRef(+ 3 * numOfCalcElements + 3 * element->calcID + 1);
	//	//tmp.coeffRef(2) = dJdH.coeffRef(+ 3 * numOfCalcElements + 3 * element->calcID + 2);
	//	//element->lambda2 = tmp;

	//}
	//output->VTKFileOputput(omega, &elements, "Lambda");
	//std::ofstream f;
	//string fname= "debug" + std::to_string(iOmega) + ".txt";
	//const char* filename = fname.c_str();
	//f.open(filename, std::ios::trunc);
	//for (int i = 0; i < lambda.size(); i++) {
	//	f << lambda.coeff(i) << endl;;
	//}
	//f.close();
	////====Hiterごとに分けて計算できる(RがH1とH2で独立）ので分けて計算
	////====H1========
	//Eigen::PardisoLU<Eigen::SparseMatrix<double, Eigen::RowMajor>> solverAdjoint;
	//Eigen::SparseMatrix<double, Eigen::RowMajor> M{ 2 * 2 * 3 * numOfCalcElements,2 * 2 * 3 * numOfCalcElements };
	//for (int i = 0; i < globalMatrix1.outerSize(); ++i) {
	//	for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(globalMatrix1, i); it; ++it)
	//	{
	//		int iRow = it.row();
	//		int iCol = it.col();
	//		M.coeffRef(2 * iRow, 2 * iCol) = globalMatrix1.coeff(iRow, iCol).real();
	//		M.coeffRef(2 * iRow, 2 * iCol + 1) = -globalMatrix1.coeff(iRow, iCol).imag();
	//		M.coeffRef(2 * iRow + 1, 2 * iCol) = globalMatrix1.coeff(iRow, iCol).imag();
	//		M.coeffRef(2 * iRow + 1, 2 * iCol + 1) = globalMatrix1.coeff(iRow, iCol).real();
	//	}
	//}
	//solverAdjoint.compute(M.adjoint());

	//Eigen::VectorXd vec{2 * 3 * numOfCalcElements };
	//for (int i = 0;i <2 * 3 * numOfCalcElements;i++) {
	//	vec.coeffRef(i) = dJdH.coeff(i);
	//}

	//Eigen::VectorXd res;
	//res = solverAdjoint.solve(vec);

	//for (int i = 0; i <2 * 3 * numOfCalcElements; i++) {
	//	lambda.coeffRef(iOmega *2 * 2 * 3 * numOfCalcElements + i) = res.coeff(i, 0);
	//	std::cout << lambda.coeffRef(iOmega * 2 * 2 * 3 * numOfCalcElements + i) << std::endl;
	//}
	////====H2========
	//M.setZero();
	//for (int i = 0; i < globalMatrix2.outerSize(); ++i) {
	//	for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(globalMatrix2, i); it; ++it)
	//	{
	//		int iRow = it.row();
	//		int iCol = it.col();
	//		M.coeffRef(2 * iRow, 2 * iCol) = globalMatrix1.coeff(iRow, iCol).real();
	//		M.coeffRef(2 * iRow, 2 * iCol + 1) = -globalMatrix1.coeff(iRow, iCol).imag();
	//		M.coeffRef(2 * iRow + 1, 2 * iCol) = globalMatrix1.coeff(iRow, iCol).imag();
	//		M.coeffRef(2 * iRow + 1, 2 * iCol + 1) = globalMatrix1.coeff(iRow, iCol).real();
	//	}
	//}
	//solverAdjoint.compute(M.adjoint());

	//vec.setZero();
	//for (int i = 0; i < 2 * 3 * numOfCalcElements; i++) {
	//	vec.coeffRef(i) = dJdH.coeff(2 * 3 * numOfObsPointElements + i);
	//}

	//res = solverAdjoint.solve(vec);

	//for (int i = 0; i < 2 * 3 * numOfCalcElements; i++) {
	//	lambda.coeffRef(iOmega * 2 * 2 * 3 * numOfCalcElements + 2 * 3 * numOfCalcElements + i) = res.coeff(i, 0);
	//	std::cout << lambda.coeffRef(iOmega * 2 * 2 * 3 * numOfCalcElements + 2 * 3 * numOfCalcElements + i) << std::endl;
	//}
}

void Analysis::Analysis::SearchRelatedCalcElements() {
	for (int i = 0; i < calcElementsVector.size(); i++) {
		calcElementsVector[i]->SearchRelatedCalcElements(&elements);
		for (auto itr = calcElementsVector[i]->relatedNeighborCalcElementsMap.begin(); itr != calcElementsVector[i]->relatedNeighborCalcElementsMap.end(); itr++) {
			Element::Element* tmpElement = itr->second;
			calcElementsVector[i]->relatedNeighborCalcElementsVector.push_back(tmpElement);
		}
	}
}

void Analysis::Analysis::CalcLambdaDRDRho(const ub::vector<complex<double>>* rhoVec, const vector<Eigen::VectorXcd>* HresultItr) {
	
	int numThreads = omp_get_max_threads();

	//Analysis::Analysis::CalcLambdaDRDRhoParameters valForLambdaDRDRho;
	////if (valForLambdaDRDRho.isInitialized == false) {
	//	int maxLayer = 0;
	//	valForLambdaDRDRho.threadIDGroup.resize(numThreads);
	//	int numOfNotBoundaryElements = 0;
	//	for (auto itr = calcElementsVector.begin(); itr != calcElementsVector.end(); itr++) {
	//		Element::Element* element = *itr;
	//		if (element->layer > maxLayer) {

	//			maxLayer = element->layer;
	//		}
	//	}
	//	valForLambdaDRDRho.maxLayer = maxLayer;

	//	for (int iLayer = 0; iLayer <= maxLayer; iLayer++) {
	//		vector < Element::Element* > layerElementsVector;
	//		for (int i = 0; i < calcElementsVector.size(); i++) {
	//			Element::Element* element = calcElementsVector[i];
	//			if (element->layer == iLayer && element->boundary == "NOT_BOUNDARY") {
	//				layerElementsVector.push_back(element);
	//				numOfNotBoundaryElements++;
	//			}
	//		}
	//		valForLambdaDRDRho.sameLayerElementsVector.push_back(layerElementsVector);
	//	}

	//	for (int iLayer = maxLayer; iLayer >= 0; iLayer--) {
	//		vector < Element::Element* >tmpVector = valForLambdaDRDRho.sameLayerElementsVector[iLayer];
	//		for (int i = 0; i < tmpVector.size(); i++) {
	//			Element::Element* element = valForLambdaDRDRho.sameLayerElementsVector[iLayer][i];
	//			valForLambdaDRDRho.threadIDGroup[i%numThreads].push_back(element->calcID);
	//		}
	//	}
	//	valForLambdaDRDRho.isInitialized = true;
	////}


	//Multi Thread
	//vector<Eigen::VectorXcd> lambdaDRDRhoEachThread(numThreads);
	//for (int i = 0; i < numThreads; i++) {
	//	lambdaDRDRhoEachThread[i].resize(numOfInvertedResistivityElements);
	//	lambdaDRDRhoEachThread[i].setZero();
	//}

		
	//}


	time_t start_t = time(NULL);

	//vector<Eigen::VectorXcd> lambdaDRDRhoEachThread(numThreads);
	//for (int i = 0; i < numThreads; i++) {
	//	lambdaDRDRhoEachThread[i] = Eigen::VectorXcd(numOfInvertedResistivityElements);
	//	lambdaDRDRhoEachThread[i].setZero();
	//}

	
//	for (int iLayer = valForLambdaDRDRho.maxLayer; iLayer >= 0; iLayer--) {
//#pragma omp parallel for
//		for (int i = 0; i < numThreads; i++) {
//			for (int j = 0; j < valForLambdaDRDRho.threadIDGroup[i].size(); j++) {
//				Element::Element* element = calcElementsVector[valForLambdaDRDRho.threadIDGroup[i][j]];
//				if (element->layer == iLayer) {
//					element->CalcLambdaDSumNCrossRhoRotHdSDRho(&elements, rhoVecUb, HresultItr, calcElementsVector, numOfCalcElements, numOfInvertedResistivityElements, lambdaEachOmega, &lambdaDRDRhoEachThread[i]);
//				}
//			}
//		}
//	}

	vector<Eigen::VectorXcd> lambdaDRDRhoEachThread(numThreads);
	for (int i = 0; i < numThreads; i++) {
		lambdaDRDRhoEachThread[i] = Eigen::VectorXcd(numOfInvertedResistivityElements);
		lambdaDRDRhoEachThread[i].setZero();
	}

	Eigen::setNbThreads(1);
#pragma omp parallel for
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		element->CalcLambdaDSumNCrossRhoRotHdSDRho(&elements, rhoVec, HresultItr, &calcElementsVector, numOfCalcElements, numOfInvertedResistivityElements, &lambdaEachOmega, &lambdaDRDRhoEachThread[omp_get_thread_num()]);
		//element->CalcLambdaDSumNCrossRhoRotHdSDRho(&elements, rhoVec, HresultItr, calcElementsVector, numOfCalcElements, numOfInvertedResistivityElements, lambdaEachOmega, &lambdaDRDRho);

	}
	Eigen::setNbThreads(omp_get_max_threads());

	time_t end_t = time(NULL);
	std::cout << "Parallel Part Calculation Time:" << end_t - start_t << " Seconds." << endl;

	for (int i = 0; i < numThreads; i++) {
		lambdaDRDRho += lambdaDRDRhoEachThread[i];
	}


	//Single Thread
	//time_t start_t = time(NULL);
	//for (int iLayer = valForLambdaDRDRho.maxLayer; iLayer >= 0; iLayer--) {
	//	for (int j = 0; j < numOfCalcElements; j++) {
	//		Element::Element* element = calcElementsVector[j];
	//		if (element->layer == iLayer) {
	//			element->CalcLambdaDSumNCrossRhoRotHdSDRho(&elements, rhoVecUb, HresultItr, calcElementsVector, numOfCalcElements, numOfInvertedResistivityElements, lambdaEachOmega, lambdaDRDRho);
	//		}
	//	}
	//}

	end_t = time(NULL);
	std::cout << "Total Calc Lambda DRDRho Time:" << end_t - start_t << " Seconds." << endl;
}

void Analysis::Analysis::SetInvertedElements() {
	//テスト
	//for (int i = 0; i < numOfObsPointElements; i++) {
	//	obsPointElements[i]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[0 + 3 * 1 + 9 * 1]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[2 + 3 * 1 + 9 * 1]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[1 + 3 * 0 + 9 * 1]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[1 + 3 * 2 + 9 * 1]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[1 + 3 * 1 + 9 * 0]->property = propertiesVector[5];
	//	obsPointElements[i]->neighborElements[1 + 3 * 1 + 9 * 2]->property = propertiesVector[5];
	//	cout << propertiesVector[5]->type << endl;
	//}
	//テスト終わり
	//テスト
	/*for (int i = 0; i < numOfCalcElements; i++) {
		calcElementsVector[i]->property = propertiesVector[5];
	}
	for (int i = 0; i < numOfObsPointElements; i++) {
		obsPointElements[i]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[0 + 3 * 1 + 9 * 1]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[2 + 3 * 1 + 9 * 1]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[1 + 3 * 0 + 9 * 1]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[1 + 3 * 2 + 9 * 1]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[1 + 3 * 1 + 9 * 0]->property = propertiesVector[1];
		obsPointElements[i]->neighborElements[1 + 3 * 1 + 9 * 2]->property = propertiesVector[1];
	}*/
	//テスト終わり

	numOfInvertedResistivityElements = 0;
	invertedRhoIDToElementVector.clear();
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		if (element->property->type == Property::Property::NORMAL){// && element->boundary=="NOT_BOUNDARY" && element->isAirGroundBoundaryCell == false) {
			//elements of isAirGroundBoundaryCell and Boundary are inverted, but not independent.so here, they are included in inverted elements group.
			element->invertedRhoElementsID = numOfInvertedResistivityElements;
			invertedRhoIDToElementMap[numOfInvertedResistivityElements] = element;
			invertedRhoIDToElementVector.push_back(element);
			numOfInvertedResistivityElements++;
		}
		else {
			element->invertedRhoElementsID = -1;
		}
	}
}


void Analysis::Analysis::SetDKDRhoElements() {
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		Element::Element* element = invertedRhoIDToElementVector[i];
		Eigen::SparseMatrix<double, Eigen::RowMajor> tmpDKdRho{ 3 * numOfCalcElements, 3 * numOfCalcElements };
		element->dKDRho=tmpDKdRho;
		//Eigen::SparseMatrix<double, Eigen::RowMajor> tmp{ 1, 3 * numOfCalcElements };
		//tmp.setZero();
		//tmp.makeCompressed();
		//tmp.prune(1e-9);
		//tmp.data().squeeze();
		//tmp.reserve(0);
		//tmpDKdRho.row(j) = tmp;

		//calcElementsVector[i]->dKDRho.makeCompressed();
		//calcElementsVector[i]->dKDRho.prune(1e-9);
		//calcElementsVector[i]->dKDRho.data().squeeze();
		//calcElementsVector[i]->dKDRho.reserve(81);
		//calcElementsVector[i]->dKDRho.resize(3 * numOfCalcElements, 3 * numOfCalcElements);

	}
}
//void Analysis::Analysis::CalcDKDRhoElements() {
//	for (int i = 0; i < numOfInvertedRhoElements; i++) {
//		invertedRhoIDToElementVector[i]->CalcDKDRho(&elements, &invertedRhoIDToElementMap, numOfCalcElements, numOfInvertedRhoElements);
//	}
//}

double Analysis::Analysis::CalcDataMisfit() {
	dataMisfit = 0.0;
	//Impedance Tensor
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];
		if (element->isInversionImpedance == true) {
			for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
				for (int ii = 0; ii < 2; ii++) {
					for (int jj = 0; jj < 2; jj++) {
						std::complex<double> dZtmp = element->Z[iOmega].coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
						double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
						double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));

						if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0 && element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj).real()!=0) {
							dZtmp.real(dZtmp.real() / epsReal);
						}
						else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
							dZtmp.real(0.0);
						}
						else {
							//そのまま
						}
						if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0 && element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj).imag() != 0) {
							dZtmp.imag(dZtmp.imag() / epsImag);
						}
						else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
							dZtmp.imag(0.0);
						}
						else {
							//そのまま
						}



						dataMisfit += (dZtmp*std::conj(dZtmp)).real();
						
					}
				}
			}
		}
	}

	//Tipper 
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];
		if (element->isInversionTipper == true) {
			for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
				for (int ii = 0; ii < 2; ii++) {
					std::complex<double> dTtmp = element->T[iOmega].coeff(ii) - element->tipperObsData->TobsVector[iOmega].coeff(ii);
					double epsReal = std::abs(element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii));
					double epsImag = std::abs(element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii));

					if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) > 0 && element->tipperObsData->TobsVector[iOmega].coeff(ii).real() != 0) {
						dTtmp.real(dTtmp.real() / epsReal);
					}
					else if (element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii) <= 0) {
						dTtmp.real(0.0);
					}
					else {
						//そのまま
					}
					if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) > 0 && element->tipperObsData->TobsVector[iOmega].coeff(ii).imag() != 0) {
						dTtmp.imag(dTtmp.imag() / epsImag);
					}
					else if (element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii) <= 0) {
						dTtmp.imag(0.0);
					}
					else {
						//そのまま
					}
					
					dataMisfit += (dTtmp*std::conj(dTtmp)).real();


					//cout <<"Tcalc Tobs:"<< element->T[iOmega].coeff(ii) << " " << element->tipperObsData->TobsVector[iOmega].coeff(ii) << endl;
					//cout << "epsReal:" << epsReal << endl;
					//cout << "epsImag:" << epsImag << endl;
					//cout << "misfit" << (dTtmp*std::conj(dTtmp)).real()<< endl;


				}

			}
		}
	}
	// Todo::他のテンソル量を逆解析する場合はここに足す
	//dataMisfit /= numOfObsData;
	return dataMisfit;
}
double Analysis::Analysis::CalcRoughningMatrixPenalty() {
	Eigen::VectorXd rhoVec{ numOfInvertedResistivityElements };
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		//rhoVec.coeffRef(i) = log10(invertedRhoIDToElementVector[i]->resistivity);
		rhoVec.coeffRef(i) = log(invertedRhoIDToElementVector[i]->resistivity);
		//rhoVec.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
		//rhoVec.coeffRef(i) = log10(invertedRhoIDToElementVector[i]->resistivity) - log10(invertedRhoIDToElementVector[i]->initialResistivity);
	}

	mWTWm = 0.0;
	mWTWm =  rhoVec.transpose() * rougheningMatrix->transpose()*(*rougheningMatrix)*rhoVec;


	return mWTWm;
}

void Analysis::Analysis::CalcDZDHElements(const ub::vector<kv::complex<double>>* HVecUb, int iOmega) {
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];
		if (element->isInversionImpedance) {
			element->CalcDZDH(HVecUb, &elements, numOfCalcElements, iOmega);

			//cout << "element->Z[iOmega]3 * element->calcID" << endl;
			//cout << element->Z[iOmega](0, 0) <<
			//	element->Z[iOmega](0, 1) <<
			//	element->Z[iOmega](1, 0) <<
			//	element->Z[iOmega](1, 1) << endl;

			//cout << "element->dZdH[iOmega]3 * element->calcID" << endl;
			//cout << element->dZdH[iOmega](0, 0).coeff(0, 3 * element->calcID) <<
			//	element->dZdH[iOmega](0, 1).coeff(0, 3 * element->calcID) <<
			//	element->dZdH[iOmega](1, 0).coeff(0, 3 * element->calcID) <<
			//	element->dZdH[iOmega](1, 1).coeff(0, 3 * element->calcID) << endl;
			//cout << " " << endl;

			//test using numerical differential
			//{
				//std::ofstream f;
				//f.open("debugdZdH.txt", std::ios::trunc);
				//
				//for (int itr = 0; itr < 2; itr++) {
				//	for (int iComp = 0; iComp < 3; iComp++) {
				//		for (int ii = 0; ii < 3; ii++) {
				//			for (int jj = 0; jj < 3; jj++) {
				//				for (int kk = 0; kk < 3; kk++) {
				//					
				//					int ipos = (ii)+3 * (jj)+9 * (kk);
				//					element->CalcE(result[itr], &elements, numOfCalcElements, itr, false);
				//					element->CalcZ(&elements, numOfCalcElements, iOmega);
				//					Eigen::Matrix2cd tmpZ1 = element->Z[iOmega];
				//					Element::Element* neighbor = element->neighborElements[ipos];
				//					int elemID = element->neighborElements[ipos]->calcID;

				//					f << 3*itr * numOfCalcElements + 3 * elemID + iComp << endl;

				//					std::complex<double>dH = result[itr].coeffRef(3 * elemID + iComp, 0).real()*0.0000001;
				//					result[itr].coeffRef(3 * elemID + iComp, 0) += dH;
				//					element->CalcE(result[itr], &elements, numOfCalcElements, itr, false);
				//					neighbor->H[itr].coeffRef(iComp) += dH;
				//					element->CalcZ(&elements, numOfCalcElements, iOmega);
				//					Eigen::Matrix2cd tmpZ2 = element->Z[iOmega]; 
				//					f << "add real" << endl;
				//					f << tmpdZ << endl;
				//					result[itr].coeffRef(3 * elemID + iComp, 0) -= dH;
				//					neighbor->H[iOmega].coeffRef(iComp) -= dH;

				//					element->CalcE(result[itr], &elements, numOfCalcElements, itr, false);
				//					element->CalcZ(&elements, numOfCalcElements, iOmega);
				//					tmpZ1 = element->Z[iOmega];
				//					dH = result[itr].coeffRef(3 * elemID + iComp, 0).imag()* 0.0000001;
				//					result[itr].coeffRef(3 * elemID + iComp, 0) += dH * complex<double>(0, 1.0);
				//					element->CalcE(result[itr], &elements, numOfCalcElements, itr, false);
				//					neighbor->H[itr].coeffRef(iComp) += dH * complex<double>(0, 1.0);
				//					element->CalcZ(&elements, numOfCalcElements, iOmega);
				//					tmpZ2 = element->Z[iOmega];
				//					tmpdZ = (tmpZ2 - tmpZ1) / dH;
				//					f << "add complex" << endl;
				//					f << tmpdZ << endl;
				//					result[itr].coeffRef(3 * element->calcID + iComp, 0) -= dH * complex<double>(0, 1.0);
				//					neighbor->H[itr].coeffRef(iComp) -= dH * complex<double>(0, 1.0);
				//					f << "autodif" << endl;
				//					f << element->dZdH[iOmega](0, 0).coeff(0, itr* 3 * numOfCalcElements + 3 * elemID + iComp) <<
				//						element->dZdH[iOmega](0, 1).coeff(0, itr * 3 * numOfCalcElements + 3 * elemID + iComp) <<
				//						element->dZdH[iOmega](1, 0).coeff(0, itr * 3 * numOfCalcElements + 3 * elemID + iComp) <<
				//						element->dZdH[iOmega](1, 1).coeff(0, itr * 3 * numOfCalcElements + 3 * elemID + iComp) << endl;
				//					//cout << "in calddzdhelements(0,0)" << element->dZdH[iOmega](0,0).coeff(0, 13981) << endl;
				//					//cout << "in calddzdhelements(0,1)" << element->dZdH[iOmega](0, 1).coeff(0, 13981) << endl;
				//					//cout << "in calddzdhelements(1,0)" << element->dZdH[iOmega](1, 0).coeff(0, 13981) << endl;
				//					//cout << "in calddzdhelements(1,1)" << element->dZdH[iOmega](1, 1).coeff(0, 13981) << endl;
				//				}
				//			}
				//		}
				//	}
				//}
				//f.close();
			//}
		}
		
	}

}
void Analysis::Analysis::CalcDTDHElements(int iOmega) {
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];
		if (element->isInversionTipper) {
			element->CalcDTDH(numOfCalcElements, iOmega);
			//test
			//Eigen::Vector2cd preT = element->T[iOmega];
			//std::complex<double>dH = 0.001;
			//element->H[0].coeffRef(0) += dH;
			//element->CalcT(iOmega);
			//Eigen::Vector2cd postT = element->T[iOmega];
			//cout << "dT/dHx1=" << (postT - preT) / dH <<" "<<element->dTdH[iOmega](0).coeff(0,3*element->calcID)
			//	<< " " << element->dTdH[iOmega](1).coeff(0,3 * element->calcID) << endl;
			//element->H[0].coeffRef(0) -= dH;
			//element->CalcT(iOmega);

			//element->H[0].coeffRef(1) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHy1=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3 * element->calcID+1) 
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * element->calcID + 1) << endl;
			//element->H[0].coeffRef(1) -= dH;
			//element->CalcT(iOmega);

			//element->H[0].coeffRef(2) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHz1=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3 * element->calcID+2) 
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * element->calcID + 2) << endl;
			//element->H[0].coeffRef(2) -= dH;
			//element->CalcT(iOmega);

			//element->H[1].coeffRef(0) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHx2=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3*numOfCalcElements+ 3 * element->calcID) 
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * numOfCalcElements + 3 * element->calcID) << endl;
			//element->H[1].coeffRef(0) -= dH;
			//element->CalcT(iOmega);

			//element->H[1].coeffRef(1) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHy2=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3 * numOfCalcElements+3 * element->calcID + 1)
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * numOfCalcElements + 3 * element->calcID + 1) << endl;
			//element->H[1].coeffRef(1) -= dH;
			//element->CalcT(iOmega);

			//element->H[1].coeffRef(2) += dH;
			//element->CalcT(iOmega);
			//postT = element->T[iOmega];
			//cout << "dT/dHz2=" << (postT - preT) / dH << " " << element->dTdH[iOmega](0).coeff(0,3 * numOfCalcElements+3 * element->calcID + 2) 
			//	<< " " << element->dTdH[iOmega](1).coeff(0, 3 * numOfCalcElements + 3 * element->calcID + 2) << endl;
			//element->H[1].coeffRef(2) -= dH;
			//element->CalcT(iOmega);
		}

	}

}

void Analysis::Analysis::CalcDDataMisfitDRho() { 
	
	//dDataMisfitDRho.resize(numOfInvertedResistivityElements);
	//dDataMisfitDRho.setZero();

	//Impedance Tensor

	for (int j = 0; j < numOfObsPointElements; j++) {
		Element::Element* element = obsPointElements[j];
		if (element->isInversionImpedance == true) {
			for(int iOmega=0;iOmega< boundary->omega.size();iOmega++){
				//cout << element->neighborElements[0 + 3 * 1 + 9 * 1]->calcID << endl;
				//cout << element->neighborElements[2 + 3 * 1 + 9 * 1]->calcID << endl;
				//cout << element->neighborElements[1 + 3 * 0 + 9 * 1]->calcID << endl;
				//cout << element->neighborElements[1 + 3 * 2 + 9 * 1]->calcID << endl;
				//cout << element->neighborElements[1 + 3 * 1 + 9 * 0]->calcID << endl;
				//cout << element->neighborElements[1 + 3 * 1 + 9 * 2]->calcID << endl;
				//cout << element->neighborElements[1 + 3 * 1 + 9 * 1]->calcID << endl;
				for (int ii = 0; ii < 2; ii++) {
					for (int jj = 0; jj < 2; jj++) {
						for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdRho[iOmega](ii, jj), 0); it; ++it)
						{
							int iCol = it.col();
							int invertedID = calcElementsVector[it.col()]->invertedRhoElementsID; //element->dZdRhoはnumOfCalcID分微分値を保存している。
																									//一方dDataMisfitDRhoはnumOfInvertedRhoID分の微分値を保存する
							std::complex<double> dZtmp = element->Z[iOmega].coeff(ii, jj) - element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj);
							std::complex<double> dZdRhotmp = element->dZdRho[iOmega](ii, jj).coeff(0, iCol);
							//cout << dZtmp << " " << dZdRhotmp <<" "<<j<< " " << iOmega <<" "<<ii<<" "<<jj<<" "<<" "<< iCol << endl;
							
							double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
							double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));


							if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) > 0 && element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj).real() != 0) {
								dZtmp.real(dZtmp.real() / epsReal);
								dZdRhotmp.real(dZdRhotmp.real() / epsReal);
							}
							else if (element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj) <= 0) {
								dZtmp.real(0.0);
							}
							else {
								//そのまま
							}
							if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) > 0 && element->impedanceObsData->ZobsVector[iOmega].coeff(ii, jj).imag() != 0) {
								dZtmp.imag(dZtmp.imag() / epsImag);
								dZdRhotmp.imag(dZdRhotmp.imag() / epsImag);
							}
							else if (element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj) <= 0) {
								dZtmp.imag(0.0);
							}
							else {
								//そのまま
							}
							/*if (invertedRhoIDToElementVector[invertedID]->masterResistivityElement != nullptr) {
								dDataMisfitDRho.coeffRef(invertedRhoIDToElementVector[invertedID]->masterResistivityElement->invertedRhoElementsID) 
									+= 2.0*(std::conj(dZtmp)*dZdRhotmp).real();
							}
							else {*/
							if (invertedID >= 0) {
								dDataMisfitDRho.coeffRef(invertedID) += 2.0*(std::conj(dZtmp)*dZdRhotmp).real();
							}
							//}

							
						}
					}
				}
				//for (int i = 0; i < numOfInvertedResistivityElements; i++) {
				//	if (dDataMisfitDRho.coeff(i) != 0.0 && iOmega==0) {
				//		cout << "ID" << i << "iOmega " << iOmega << " dDataMisfitDRho" << dDataMisfitDRho.coeff(i) << endl;
				//	}
				//}
			}
		}
	}

	//Tipper Tensor, this is zero because tipper does not explicitly depend on Resistivity.
	// Todo::他のテンソル量を逆解析する場合はここに足す
	//dDataMisfitDRho /=  numOfObsData;
	
	
}

void Analysis::Analysis::CalcDZDRhoElements(const ub::vector<kv::complex<double>>* rhoVecUb,const ub::vector<kv::complex<double>>* HresultTwoItr,const int iOmega) {
	for (int i = 0; i < numOfObsPointElements; i++) {
		Element::Element* element = obsPointElements[i];
		if (element->isInversionImpedance == true) {
			element->CalcDZDRho(rhoVecUb, HresultTwoItr,&calcElementsVector, numOfCalcElements, iOmega);

			//test using numerical differential
			//Eigen::Matrix2cd tmpZ1 = element->Z[iOmega];
			//double dRho = element->resistivity*0.0000001;
			//element->resistivity += dRho;
			//element->CalcSurfaceResistivity(&elements, &calcElementsVector, numOfCalcElements);
			//element->CalcE(result[0], &elements, numOfCalcElements, 0, false);
			//element->CalcE(result[1], &elements, numOfCalcElements, 1, false);
			//element->CalcZ(&elements, numOfCalcElements, iOmega);
			//Eigen::Matrix2cd tmpZ2 = element->Z[iOmega];
			//Eigen::Matrix2cd tmpdZ = (tmpZ2 - tmpZ1) / dRho;
			//cout << tmpZ1 << endl;
			//cout << tmpZ2 << endl;
			//cout << tmpdZ << endl;
			//cout << element->dZdRho[iOmega](0, 0).coeff(0,  element->calcID) <<
			//	element->dZdRho[iOmega](0, 1).coeff(0,  element->calcID) <<
			//	element->dZdRho[iOmega](1, 0).coeff(0,  element->calcID) <<
			//	element->dZdRho[iOmega](1, 1).coeff(0,  element->calcID) << endl;
			//cout << " " << endl;
			//element->resistivity -= dRho;
			//element->CalcSurfaceResistivity(&elements, &calcElementsVector, numOfCalcElements);
		}
	}
}
void Analysis::Analysis::CalcDJDRho() {
	for (int i = 0; i < numOfCalcElements; i++) {
		calcElementsVector[i]->debug = 0.0; //debug
	}

	dJdRho.setZero();
	//if (RMScur <= thresholdRMS) {
	//	cout << "RMS is below threshold. Searching smoother solution.." << endl;
	//}
	//if (RMScur > thresholdRMS) { //test
		//term of data misfit
		
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		dJdRho.coeffRef(i) += dDataMisfitDRho.coeff(i);

		//test
		invertedRhoIDToElementVector[i]->debug = dDataMisfitDRho.coeff(i);

		bool flg = false;
		for (int j = 0; j < 3; j++) {
			int ipos = j + 3 * 1 + 9 * 1;
			if (invertedRhoIDToElementVector[i]->neighborElements[ipos] != NULL && invertedRhoIDToElementVector[i]->neighborElements[ipos]->isObservationElement) {
				flg = true;
			}

		}
		for (int j = 0; j < 3; j++) {
			int ipos = 1 + 3 * j + 9 * 1;
			if (invertedRhoIDToElementVector[i]->neighborElements[ipos] != NULL && invertedRhoIDToElementVector[i]->neighborElements[ipos]->isObservationElement) {
				flg = true;
			}

		}
		for (int j = 0; j < 3; j++) {
			int ipos = 1 + 3 * 1 + 9 * j;
			if (invertedRhoIDToElementVector[i]->neighborElements[ipos] != NULL && invertedRhoIDToElementVector[i]->neighborElements[ipos]->isObservationElement) {
				flg = true;
			}

		}
	}
	//}
	
	//term of roughning matrix
	Eigen::VectorXd logRhoVec{ numOfInvertedResistivityElements };
	Eigen::VectorXd rhoVec{ numOfInvertedResistivityElements };
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		//logRhoVec.coeffRef(i) = log10(invertedRhoIDToElementVector[i]->resistivity);
		logRhoVec.coeffRef(i) = log(invertedRhoIDToElementVector[i]->resistivity);
		//logRhoVec.coeffRef(i) = log10(invertedRhoIDToElementVector[i]->resistivity)- log10(invertedRhoIDToElementVector[i]->initialResistivity);
		
		//rhoVec.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
	}

	Eigen::VectorXd WTWm{ numOfInvertedResistivityElements };
	WTWm = rougheningMatrix->transpose()*(*rougheningMatrix)*logRhoVec;
	//WTWm = rougheningMatrix->transpose()*(*rougheningMatrix)*rhoVec;

	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		//double dmdRho = 1 / invertedRhoIDToElementVector[i]->resistivity*log10(exp(1.0));
		double dmdRho = 1 / invertedRhoIDToElementVector[i]->resistivity;
		//double dmdRho = 1.0;

		dJdRho.coeffRef(i) += weightRoughening * 2 * WTWm.coeff(i)*dmdRho;
		
	}
	//lambda term
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		dJdRho.coeffRef(i) -= lambdaDRDRho.coeff(i).real();
	}


	//set Slave Elem Terms to the master
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		if (invertedRhoIDToElementVector[i]->masterResistivityElement != nullptr) {
			

			dJdRho.coeffRef(invertedRhoIDToElementVector[i]->masterResistivityElement->invertedRhoElementsID) += dJdRho.coeff(i);
			dJdRho.coeffRef(i) = 0.0;
		}
		invertedRhoIDToElementVector[i]->debug = dJdRho.coeffRef(i);
	}

	//Convert dJ/dRho ->dJ/dParam
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		dJdRho.coeffRef(i) = dJdRho.coeff(i)*dRhoDParam.coeff(i);

		//test
		//if (invertedRhoIDToElementVector[i]->isAirGroundBoundaryCell == true) {
		//	dJdRho.coeffRef(i) = 0.0;
		//}
	}
}

void Analysis::Analysis::CalcRougheningMatrix() {
	rougheningMatrix->reserve(Eigen::VectorXi::Constant(6 * numOfInvertedResistivityElements, 6));
	//rougheningMatrix->reserve(Eigen::VectorXi::Constant(27*numOfInvertedResistivityElements, 27));
	//rougheningMatrix->setZero();
	
	for (int iInvElem = 0; iInvElem < numOfInvertedResistivityElements; iInvElem++) {
		Element::Element* element = invertedRhoIDToElementVector[iInvElem];
		for (int i = 0; i < 6; i++) {
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
			Element::Element* neighborElem = element->neighborElements[ipos];
			if (neighborElem == nullptr) {
				continue;
			}
			double unit;
			if (invSettings->isUseDistanceInModelConstraint == false) {
				unit = 1.0;
			}
			else {
				if (element->layer == neighborElem->layer) {
					unit = element->roughenMatrixUnit;
				}
				else {
					unit = std::max(element->roughenMatrixUnit, neighborElem->roughenMatrixUnit);
				}
				//unit = std::max(std::max(modelNormalizationCoeff[0], modelNormalizationCoeff[1]), modelNormalizationCoeff[2]);
			}

			double dl;
			if (neighborElem->layer == element->layer) {
				dl= (element->centerCoord - neighborElem->centerCoord).norm();
			}
			else {
				if (i == 0 || i == 1) dl = std::max(element->dx,neighborElem->dx);
				else if (i == 2 || i == 3) dl = std::max(element->dy, neighborElem->dy);
				else if (i == 4 || i == 5) dl = std::max(element->dz, neighborElem->dz);
			}
			if (invSettings->isUseDistanceInModelConstraint == false) {
				dl = 1.0;
			}
			
			for (int j = 0; j < element->diffResistivitySurfaceCoeff[i]->outerSize(); ++j) {
				for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it(*element->diffResistivitySurfaceCoeff[i], j); it; ++it)
				{
					int iCol = it.col();
					
					if (calcElementsVector[iCol]->invertedRhoElementsID >= 0) {
						rougheningMatrix->coeffRef(6 * iInvElem + i, calcElementsVector[iCol]->invertedRhoElementsID) =
							(element->diffResistivitySurfaceCoeff[i]->coeff(0, iCol)*1.0 / dl * unit).real();
					}
					else {
						for (Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor>::InnerIterator it2(*element->diffResistivitySurfaceCoeff[i], j); it2; ++it2)
						{
							int iCol2 = it2.col();
							if (calcElementsVector[iCol2]->invertedRhoElementsID >= 0) {
								rougheningMatrix->coeffRef(6 * iInvElem + i, calcElementsVector[iCol2]->invertedRhoElementsID) = 0.0; //reset settings
							}
						}
						break;
					}
				}
			}
		}
	}
	//debug
	std::ofstream f;
	f.open("debugWeightMatrix.txt", std::ios::trunc);
	for (int j = 0; j < rougheningMatrix->outerSize(); ++j) {
		for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(*rougheningMatrix, j); it; ++it)
		{
			f <<it.row()<<" "<<it.col()<<" "<< rougheningMatrix->coeff(j, it.col()) << endl;;
		}
	}
	f.close();
	//for (int iInvElem = 0; iInvElem < numOfInvertedResistivityElements; iInvElem++) {
	//	Element::Element* element = invertedRhoIDToElementVector[iInvElem];
	//	unordered_map<string, Element::Element*> alreadyCalcElem;
	//	for (int i = 0; i < 3; i++) {
	//		for (int j = 0; j < 3; j++) {
	//			for (int k = 0; k < 3; k++) {
	//				if (i == 1 && j == 1 && k == 1) {
	//					continue;
	//				}
	//				int ipos = i + 3 * j + 9 * k;
	//				
	//				Element::Element* neighborElem = element->neighborElements[ipos];
	//				if ( neighborElem != nullptr && neighborElem->isParent == false && neighborElem->invertedRhoElementsID >= 0 && alreadyCalcElem.count(neighborElem->ID) == 0){
	//					alreadyCalcElem[neighborElem->ID] = neighborElem;
	//					double dl = (element->centerCoord - neighborElem->centerCoord).norm();
	//					rougheningMatrix->insert(27 * iInvElem + ipos, element->invertedRhoElementsID) = 1.0/dl
	//						*std::pow(modelNormalizationCoeff[0]* modelNormalizationCoeff[0] + modelNormalizationCoeff[1]* modelNormalizationCoeff[1],0.5);
	//					rougheningMatrix->insert(27 * iInvElem + ipos, neighborElem->invertedRhoElementsID) = -1.0 / dl
	//						* std::pow(modelNormalizationCoeff[0] * modelNormalizationCoeff[0] + modelNormalizationCoeff[1] * modelNormalizationCoeff[1], 0.5);
	//			
	//					//rougheningMatrix->insert(27 * iInvElem + ipos, element->invertedRhoElementsID) = 1.0;
	//					//rougheningMatrix->insert(27 * iInvElem + ipos, neighborElem->invertedRhoElementsID) = -1.0;

	//				}
	//				else if (neighborElem != nullptr && neighborElem->isParent == true) {
	//					bool isAllChildInverted = true;
	//					for (int ii = 0; ii < 2; ii++) {
	//						for (int jj = 0; jj < 2; jj++) {
	//								string childID = neighborElem->ID + Functions::GetBinaryValue(ii, jj);
	//								if (elements[childID]->invertedRhoElementsID < 0) {
	//									isAllChildInverted = false;
	//								}
	//						}
	//					}
	//					if (isAllChildInverted) {
	//						double dl = (element->centerCoord - neighborElem->centerCoord).norm();
	//						rougheningMatrix->insert(27 * iInvElem + ipos, element->invertedRhoElementsID) = 1.0 / dl
	//							* std::pow(modelNormalizationCoeff[0] * modelNormalizationCoeff[0] + modelNormalizationCoeff[1] * modelNormalizationCoeff[1], 0.5);
	//						
	//						//rougheningMatrix->insert(27 * iInvElem + ipos, element->invertedRhoElementsID) = 1.0;

	//						for (int ii = 0; ii < 2; ii++) {
	//							for (int jj = 0; jj < 2; jj++) {
	//									string childID = neighborElem->ID + Functions::GetBinaryValue(ii, jj);
	//									Element::Element* childElem = elements[childID];
	//									if ( alreadyCalcElem.count(childElem->ID) == 0) {
	//										alreadyCalcElem[childElem->ID] = childElem;					
	//										rougheningMatrix->insert(27 * iInvElem + ipos, childElem->invertedRhoElementsID) = -1.0 / dl / 4.0
	//											* std::pow(modelNormalizationCoeff[0] * modelNormalizationCoeff[0] + modelNormalizationCoeff[1] * modelNormalizationCoeff[1], 0.5);

	//										//rougheningMatrix->insert(27 * iInvElem + ipos, childElem->invertedRhoElementsID) = -1.0 / 4.0;

	//									}

	//							}
	//						}
	//					}
	//					
	//				}

	//			}
	//		}
	//	}
	//}

	//follow femtic by usui-san
	//
	//rougheningMatrix->setZero();
	//rougheningMatrix->reserve(4 * 6 * numOfInvertedResistivityElements);


	//for (int iInvElem = 0; iInvElem < numOfInvertedResistivityElements; iInvElem++) {
	//	Element::Element* element = invertedRhoIDToElementVector[iInvElem];

	//	int invertedID = element->invertedRhoElementsID;
	//	int calcID = element->calcID;
	//	//vector<double> testVec(6);

	//	for (int i = 0; i < 6; i++) {
	//		for (int j = 0; j < element->diffOperationOfResistivitySurfaceCoeff[i]->outerSize(); ++j) {
	//			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(element->diffOperationOfResistivitySurfaceCoeff[i]), j); it; ++it)
	//			{
	//				
	//				int iCol = it.col();
	//				int anotherElemCalcID = iCol;
	//				Element::Element* anotherElement = calcElementsVector[anotherElemCalcID];
	//				//if (anotherElement->masterResistivityElement != nullptr) {
	//				//	//rougheningMatrix.coeffRef(invertedID, anotherElement->masterResistivityElement->invertedRhoElementsID) +=
	//				//	//	element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//				//	double val = element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//				//	rougheningMatrix.coeffRef(invertedID, anotherElement->masterResistivityElement->invertedRhoElementsID) +=
	//				//		val / abs(val);
	//				//}
	//				if (anotherElement->invertedRhoElementsID >= 0) {
	//					//rougheningMatrix.coeffRef(invertedID, anotherElement->invertedRhoElementsID) +=
	//					//	element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();

	//					double val=element->diffOperationOfResistivitySurfaceCoeff[i]->coeff(0, iCol).real();
	//					rougheningMatrix->coeffRef(6*invertedID+i, anotherElement->invertedRhoElementsID) += val;
	//					//cout<<i<<" " << calcID <<" "<< invertedID<< " " << iCol<<" "<< anotherElement->invertedRhoElementsID << " " << element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real() << endl;
	//				}
	//				else {
	//					/*rougheningMatrix.coeffRef(invertedID, invertedID) +=
	//						element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();*/
	//					double val= element->diffOperationOfResistivitySurfaceCoeff[i]->coeff(0, iCol).real();
	//					rougheningMatrix->coeffRef(6*invertedID+i, invertedID) += val;
	//					//	//if neighbor elements is not inverted, this constraint calculation should be 0.(once plus value is added above, and minus value is added here, so total is zero)
	//					//}
	//				}



	//			}
	//		}
	//	}

	//}


	

	//(*rougheningMatrix) *= modelNormalizationCoeff[0]; //rougheningMatrix has unit Of "/m", so convert non-dimensional.
	//constraint by ∂m
	//rougheningMatrix.resize(6*numOfInvertedResistivityElements, numOfInvertedResistivityElements);
	//rougheningMatrix.setZero();
	//rougheningMatrix.reserve(100 * numOfInvertedResistivityElements);//100は適当
	//for (int iInvElem = 0; iInvElem < numOfInvertedResistivityElements; iInvElem++) {
	//	Element::Element* element = invertedRhoIDToElementVector[iInvElem];
	//	int invertedID = element->invertedRhoElementsID;
	//	int calcID = element->calcID;
	//	//vector<double> testVec(6);

	//	for (int i = 0; i < 6; i++) {
	//		for (int j = 0; j < element->diffOperationOfResistivitySurfaceCoeff[i].outerSize(); ++j) {

	//			for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->diffOperationOfResistivitySurfaceCoeff[i], j); it; ++it)
	//			{
	//				int iCol = it.col();
	//				int anotherElemCalcID = iCol;
	//				Element::Element* anotherElement = calcElementsVector[anotherElemCalcID];
	//				//if (anotherElement->masterResistivityElement != nullptr) {
	//				//	//rougheningMatrix.coeffRef(invertedID, anotherElement->masterResistivityElement->invertedRhoElementsID) +=
	//				//	//	element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//				//	double val = element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//				//	rougheningMatrix.coeffRef(invertedID, anotherElement->masterResistivityElement->invertedRhoElementsID) +=
	//				//		val / abs(val);
	//				//}
	//				if (anotherElement->invertedRhoElementsID >= 0) {
	//					//rougheningMatrix.coeffRef(invertedID, anotherElement->invertedRhoElementsID) +=
	//					//	element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();

	//					double val = element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//					rougheningMatrix.coeffRef(6*invertedID+i, anotherElement->invertedRhoElementsID) += val;
	//					//cout<<i<<" " << calcID <<" "<< invertedID<< " " << iCol<<" "<< anotherElement->invertedRhoElementsID << " " << element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real() << endl;
	//				}
	//				else {
	//					/*rougheningMatrix.coeffRef(invertedID, invertedID) +=
	//						element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();*/
	//					double val = element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real();
	//					rougheningMatrix.coeffRef(6*invertedID+i, invertedID) += val;
	//					//	//if neighbor elements is not inverted, this constraint calculation should be 0.(once plus value is added above, and minus value is added here, so total is zero)
	//					//}
	//				}



	//			}
	//		}
	//	}

	//}

	//debug
	//for (int id = 0; id < numOfInvertedResistivityElements; id++) {
	//	cout << invertedRhoIDToElementVector[id]->boundary << endl;
	//	for (int ii = 0; ii < 3; ii++) {
	//		for (int jj = 0; jj < 3; jj++) {
	//			for (int kk = 0; kk < 3; kk++) {
	//				if (invertedRhoIDToElementVector[id]->neighborElements[ii + 3 * jj + 9*kk] != NULL) {
	//					cout << invertedRhoIDToElementVector[id]->neighborElements[ii + 3 * jj + 9*kk]->invertedRhoElementsID << " " << ii << " " << jj << " " << kk << endl;
	//				}
	//			}
	//		}
	//	}
	//	for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(rougheningMatrix, id); it; ++it)
	//	{
	//		int iCol = it.col();
	//		int iRow = it.row();
	//		cout << iRow << " " << iCol << " " << rougheningMatrix.coeff(iRow, iCol) << endl;
	//	}
	//}
	

	//rougheningMatrix *= modelNormalizationCoeff; //rougheningMatrix has unit Of "/m", so convert non-dimensional.

	//for (int i = 0; i < 6; i++) {
	//	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//		Element::Element* element = invertedRhoIDToElementVector[i];
	//		int invertedID = element->invertedRhoElementsID;
	//		int calcID = element->calcID;
	//		//vector<double> testVec(6);
	//		for (int i = 0; i < 6; i++) {
	//			//double test = 0;
	//			for (int j = 0; j < element->diffOperationOfResistivitySurfaceCoeff[i]->outerSize(); ++j) {

	//				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(*(element->diffOperationOfResistivitySurfaceCoeff[i]), j); it; ++it)
	//				{
	//					int iCol = it.col();
	//					int anotherElemCalcID = iCol;
	//					Element::Element* anotherElement = calcElementsVector[anotherElemCalcID];
	//					if (anotherElement->invertedRhoElementsID >= 0) {
	//						rougheningMatrix->coeffRef(6 * invertedID + i, anotherElement->invertedRhoElementsID) +=
	//							element->diffOperationOfResistivitySurfaceCoeff[i]->coeff(0, iCol).real();
	//						//cout<<i<<" " << calcID <<" "<< invertedID<< " " << iCol<<" "<< anotherElement->invertedRhoElementsID << " " << element->diffOperationOfResistivitySurfaceCoeff[i].coeff(0, iCol).real() << endl;
	//					}
	//					else {
	//						rougheningMatrix->coeffRef(6 * invertedID + i, invertedID) +=
	//							element->diffOperationOfResistivitySurfaceCoeff[i]->coeff(0, iCol).real();
	//						//if neighbor elements is not inverted, this constraint calculation should be 0.(once plus value is added above, and minus value is added here, so total is zero)
	//					}
	//				}

	//			}

	//			//cout << element->resistivitySurfaceCoeff[i] << endl;
	//			//testVec[i] = test;
	//		}

	//	}
	//}
	//rougheningMatrix *=  modelNormalizationCoeff; //rougheningMatrix has unit Of "/m", so convert non-dimensional.
	rougheningMatrix->makeCompressed();
	rougheningMatrix->data().squeeze();
}

inline double Analysis::Analysis::Optimize(const Eigen::VectorXd& vals_inp, Eigen::VectorXd* grad_out, void* opt_data)
{

	Eigen::VectorXd rhoVecPre{ numOfInvertedResistivityElements };
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		rhoVecPre.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
	}

	bool isChangeResis = false;
	isChangeResis = CalcRhoFromParamAndDRhoDParam(vals_inp);


	//debug
	std::ofstream f;
	f.open("debugParam.txt", std::ios::trunc);
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		Element::Element* element = invertedRhoIDToElementVector[i];
		f << element->IDX << " "<<element->IDY << " "<<element->IDZ << " "<<element->ID << " " << vals_inp[i] << endl;
	}
	f.close();


	SetSameResistivityToBoundaryCell();
	cout << "isChangeResis" << isChangeResis << endl;
	if (isChangeResis == true || initObjVal==0) {
		CalcSurfaceResistivityElements(); //Update Resistivity
		time_t start_t = time(NULL);
		cout << "Update SumNCrossRhoRotHdS.." << endl;
		CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix
		time_t end_t = time(NULL);
		std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
		cout << "End Update SumNCrossRhoRotHdS.." << endl;
		bool isNeededGradient = false;
		if (grad_out) {
			dDataMisfitDRho.setZero();
			lambdaDRDRho = Eigen::VectorXcd{ numOfInvertedResistivityElements };
			lambdaDRDRho.setZero();

			isNeededGradient = true;
		}


		CalcForward(isNeededGradient);

	}

	invSettings->ReadManualSettingData(&settings);

	double obj_val;
	obj_val = 0.0;
	obj_val += CalcDataMisfit();
	std::cout << "DataMisfit:" << obj_val << std::endl;

	double RMS = std::pow(obj_val/numOfObsData , 0.5);
	RMScur = RMS;

	double roughningMatrixPenaltyTerm = CalcRoughningMatrixPenalty();
	

	obj_val += weightRoughening * roughningMatrixPenaltyTerm;
	
	bool isFirstLoop = false;
	if (optMethod == "GD") {
		if (initObjVal == 0 && inheritPreviousObjVal == false) {
			initObjVal = obj_val;
			obj_valPre = 1.0;
			isFirstLoop = true;
		}
		else if (inheritPreviousObjVal == true && isFirstLoopInheritPreviousObjVal) {
			isFirstLoop = true;
			isFirstLoopInheritPreviousObjVal = false;
		}
	}
	else {
		if (initObjVal == 0) {
			initObjVal = obj_val;
			obj_valPre = 1.0;
			isFirstLoop = true;
		}
	}

	if (grad_out) {

		/*vector<int>debugElemID;
		debugElemID.push_back(invertedRhoIDToElementVector[10]->invertedRhoElementsID);
		debugElemID.push_back(invertedRhoIDToElementVector[100]->invertedRhoElementsID);
		debugElemID.push_back(obsPointElements[10]->invertedRhoElementsID);
		debugElemID.push_back(obsPointElements[20]->invertedRhoElementsID);*/
		if (isChangeResis == true || isFirstLoop==true) {
			CalcDDataMisfitDRho();
			CalcDJDRho();
			//CalcJacobian();
		}
		
		if (optMethod == "GD") {
			//(*grad_out) = dJdRho / obj_valPre;
			(*grad_out) = dJdRho / initObjVal;
			//(*jacobian) = (*jacobian) / initObjVal;
		}
		else {
			(*grad_out) = dJdRho / initObjVal;
			//(*jacobian) = (*jacobian) / initObjVal;
		}
	}
	if (numOfSameModelWeightCalc == 1) {
		std::string filename = "Rho_" + std::_Floating_to_string("%.4f", weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
		output->RhoOutput(&elements, filename);
		filename = "Rho_" + std::_Floating_to_string("%.4f", weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".txt";
		output->TxtOutputResistivity(&elements, filename);
	}
	else{
		std::string filename = "Rho_" + std::_Floating_to_string("%.4f", weightRoughening) + "_"+ std::to_string(numOfSameModelWeightCalc)+ "_" + std::to_string(settings.numOfIteration) + ".vtk";
		output->RhoOutput(&elements, filename);
		filename = "Rho_" + std::_Floating_to_string("%.4f", weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".txt";
		output->TxtOutputResistivity(&elements, filename);
	}
	output->OutputObsCalcImpedance(boundary->omega, &obsPointElements);

	
	

	Eigen::VectorXd rhoVecCur{ numOfInvertedResistivityElements };
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		rhoVecCur.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
	}
	Eigen::VectorXd diffRhoVec = rhoVecCur - rhoVecPre;
	double absMaxRhoDiff = 0;
	int absMaxRhoDiffIndex = 0;
	for (int i = 0; i < diffRhoVec.size(); i++) {
		if (absMaxRhoDiff < std::abs(diffRhoVec.coeff(i))) {
			absMaxRhoDiff = std::abs(diffRhoVec.coeff(i));
			absMaxRhoDiffIndex = i;
		}
	}
	//judge convergence
	//if (optMethod == "GD") {
	//	if (isFirstLoop == false && isChangeResis == true && absMaxRhoDiff <= invSettings->thresholdResistivityChange) {
	//		cout << "This model constraint Optimization has finished because the change of Parameters is below threshold." << endl;
	//		settings.isFinishOptimize = true;
	//		obj_val = 0.0;
	//		if (grad_out) {
	//			for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//				(*grad_out).coeffRef(i) = 0.0;
	//			}
	//		}
	//	}
	//}


	if (RMS < thresholdRMS) {
		//cout << "Optimization has finished because RMS is below threshold." << endl;
		//settings.isFinishOptimize = true;
		isBelowRMSThreshold = true;
		//obj_val = 0.0;
		//if (grad_out) {
		//	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		//		(*grad_out).coeffRef(i) = 0.0;
		//	}
		//}
	}
	else {
		isBelowRMSThreshold = false;
	}

	//else if (RMS > RMSpre) { //only for gradient discent
	//	obj_val = 0.0;
	//	if (grad_out) {
	//		for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//			(*grad_out).coeffRef(i) = 0.0;
	//			RMSpre = 1e30;
	//		}
	//	}
	//}
	//else {
	//	RMSpre = RMS;
	//}
	//if (optMethod == "GD") {
	//	if (settings.iteration > 0 &&
	//		std::abs(obj_val - obj_valPre)/obj_valPre < objFuncChangeThresholdForNextmodelConstraint) {
	//		settings.isFinishOptimize = true;
	//		obj_valPre = obj_val;
	//		obj_val = 0.0;
	//		RMSpre = 1e30;
	//	}
	//	else {
	//		obj_valPre = obj_val;
	//		RMSpre = RMS;
	//	}
	//}
	//else {
	
	//}
	//
	if (optMethod == "GD") {
		//double tmp = obj_val / obj_valPre; //nakayama eiji san method
		//obj_valPre = obj_val;
		//obj_val = tmp;
		obj_val = obj_val / initObjVal;
		
	}
	else {
		obj_val = obj_val / initObjVal;

	}
	
	//if (optMethod == "GD" && obj_valPre < obj_val) {
	//	//if (RMScur > thresholdRMS) {
	//		settings.gd_settings.isRestartAdam = true;
	//	//}
	//}
	//if (optMethod == "GD") {
	//	if (isFirstLoop == false && isChangeResis == true && obj_val!=0.0 && std::abs(obj_val - obj_valPre)/ obj_val < invSettings->objFuncChangeThresholdForNextmodelConstraint) {
	//		cout << "This model constraint Optimization has finished because the change of Objective Function is below threshold." << endl;
	//		cout << "obj_val:" << obj_val << endl;
	//		cout << "pre obj_val:" << obj_valPre << endl;
	//		settings.isFinishOptimize = true;
	//		obj_val = 0.0;
	//		if (grad_out) {
	//			for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//				(*grad_out).coeffRef(i) = 0.0;
	//			}
	//		}
	//	}
	//}

	std::cout << "RMS:" << RMS << endl;
	//std::cout << "DataMisfit:" << obj_val << std::endl;
	std::cout << "weightRoughening:" << weightRoughening << " PemaltyTerm:" << roughningMatrixPenaltyTerm << std::endl;
	std::cout << "Objective Function Value:" << obj_val << std::endl;
	std::cout << "Objective Function Change:" << obj_val - obj_valPre << std::endl;
	std::cout << "max Change of Resistivity:" << diffRhoVec.coeff(absMaxRhoDiffIndex)<<" Index:"<< absMaxRhoDiffIndex << endl;
	if (optMethod == "GD") {
		std::cout << "GD Step Size:" << settings.gd_settings.par_step_size << std::endl;
	}
	std::cout << "Total Calculation Time:" << time(NULL) - startCalc_t << " Seconds." << endl;

	infofile << "\nnumOfIteration:" << settings.numOfIteration<< endl;
	infofile << "weightRoughening:" << weightRoughening << endl;
	infofile << "RMS:" << RMS << endl;
	//std::cout << "DataMisfit:" << obj_val << std::endl;
	infofile << " PemaltyTerm:" << roughningMatrixPenaltyTerm << std::endl;
	infofile << "Objective Function Value:" << obj_val << std::endl;
	infofile << "Objective Function Change:" << obj_val - obj_valPre << std::endl;
	infofile << "max Change of Resistivity:" << diffRhoVec.coeff(absMaxRhoDiffIndex) << " Index:" << absMaxRhoDiffIndex << endl;
	if (optMethod == "GD") {
		infofile << "GD Step Size:" << settings.gd_settings.par_step_size << std::endl;
	}
	infofile << "Total Calculation Time:" << time(NULL) - startCalc_t << " Seconds." << endl;
	obj_valPre = obj_val;

	settings.objFuncVal = obj_val;
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		settings.resisVec.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
	}

	return obj_val;
}






inline Eigen::Vector2d Analysis::Analysis::OptimizeUsingJacobian(const Eigen::VectorXd& vals_inp, Eigen::MatrixXd* jac_out)
{
	bool isChangeResis = false;
	isChangeResis = CalcRhoFromParamAndDRhoDParam(vals_inp);

	
	SetSameResistivityToBoundaryCell();
	//isChangeResis = true; //test
	cout << "isChangeResis" << isChangeResis << endl;
	if (isChangeResis == true || initObjVal == 0) {
		CalcSurfaceResistivityElements(); //Update Resistivity
		CalcSumNCrossRhoRotHdSElements(); //Update coeffs of Matrix

		bool isNeededGradient = false;
		bool isNeededJacobi = true;
		if (jac_out) {
			jacobian->setZero();
			isNeededGradient = true;
		}

		CalcForward(isNeededGradient, isNeededJacobi);
	}
	double obj_val;
	obj_val = 0.0;
	obj_val += CalcDataMisfit();
	std::cout << "DataMisfit:" << obj_val << std::endl;

	double RMS = std::pow(obj_val / numOfObsData, 0.5);
	

	double roughningMatrixPenaltyTerm = CalcRoughningMatrixPenalty();


	obj_val += weightRoughening * roughningMatrixPenaltyTerm;

	bool isFirstLoop = false;
	if (initObjVal == 0) {
		initObjVal = obj_val;
		obj_valPre = obj_val;
		isFirstLoop = true;
	}

	if (jac_out) {
		*jac_out = (1.0/ initObjVal)*(*jacobian);

	}
	if (numOfSameModelWeightCalc == 1) {
		std::string filename = "Rho_" + std::_Floating_to_string("%.4f", weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
		output->RhoOutput(&elements, filename);
		filename = "Rho_" + std::_Floating_to_string("%.4f", weightRoughening) + "_" + std::to_string(settings.numOfIteration) + ".txt";
		output->TxtOutputResistivity(&elements, filename);
	}
	else {
		std::string filename = "Rho_" + std::_Floating_to_string("%.4f", weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".vtk";
		output->RhoOutput(&elements, filename);
		filename = "Rho_" + std::_Floating_to_string("%.4f", weightRoughening) + "_" + std::to_string(numOfSameModelWeightCalc) + "_" + std::to_string(settings.numOfIteration) + ".txt";
		output->TxtOutputResistivity(&elements, filename);
	}
	output->OutputObsCalcImpedance(boundary->omega, &obsPointElements);




	obj_val = obj_val / initObjVal;
	Eigen::Vector2d returnval;
	returnval.coeffRef(0) = obj_val;
	returnval.coeffRef(1) = RMS;
	return returnval;
}
void Analysis::Analysis::RunOptimize() {
	//mkl_set_num_threads(omp_get_max_threads());
	std::cout << "MKL_NUM_THREADS:" << mkl_get_max_threads() << std::endl;
	std::cout << "omp_get_max_threads:" << omp_get_max_threads() << std::endl;

	startCalc_t = time(NULL);
	std::cout << ("Initialize Data") << std::endl;
	Initialize();
	std::cout << ("Initialization End") << std::endl;

	//DEBUG
	//CalcForward(true);
	//CalcDDataMisfitDRho();
	//CalcDJDRho();



	output->VTKFileOputput(0.0, &elements, "ObsPoints");

	SetSameResistivityToBoundaryCell(); //set master/slave
	CountIndependentInvertedResisElements();

	//CalcForward(true);//need once calc to get DJDRho
	Eigen::VectorXd paramVec;
	paramVec = CalcParamFromRho();


	//double obj_val;
	//obj_val = 0.0;
	//obj_val += CalcDataMisfit() ;
	//std::cout << "DataMisfit:" << std::setprecision(20) << obj_val << std::endl;
	//double roughningMatrixPenaltyTerm = CalcRoughningMatrixPenalty();
	//std::cout << "weightRoughening:" << weightRoughening << " PemaltyTerm:" << roughningMatrixPenaltyTerm  << std::endl;
	//initObjVal = obj_val;
	//initObjVal = 1.0;

	

	//CalcDDataMisfitDRho();
	//CalcDJDRho();

	//Eigen::VectorXd rhoVec{ numOfInvertedResistivityElements };
	//for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//	rhoVec.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;
	//}
	


	std::function<double(const Eigen::VectorXd&, Eigen::VectorXd*, void*)> optFunc = std::bind(
		&Analysis::Optimize,
		this,
		std::placeholders::_1,
		std::placeholders::_2,
		std::placeholders::_3);
	void* opt_data;
	
	settings.print_level = 1;
	settings.gd_settings.method =6; ///adam
	settings.gd_settings.ada_max = true;
	settings.gd_settings.par_step_size = invSettings->par_step_size;
	settings.gd_settings.loosenFactor = invSettings->loosenFactor;
	settings.gd_settings.decreaseFactor = invSettings->decreaseFactor;
	settings.iter_max = invSettings->maxIterationPerModelConstraint;
	settings.lbfgs_settings.step =1;
	settings.lbfgs_settings.wolfe_cons_1 = 1e-3;
	settings.lbfgs_settings.wolfe_cons_2 = 0.9;
	settings.lbfgs_settings.par_M = 200;
	settings.lbfgs_settings.restart_M = 5;
	settings.lbfgs_settings.loosenFactor = invSettings->loosenFactor;
	settings.lbfgs_settings.decreaseFactor = invSettings->decreaseFactor;
	settings.lbfgs_settings.minStep = invSettings->minStep;

	settings.cg_settings.use_rel_sol_change_crit = true;
	settings.rel_objfn_change_tol = invSettings->objFuncChangeThresholdForNextmodelConstraint;
	settings.grad_err_tol = invSettings->grad_err_tol;
	settings.rel_sol_change_tol = invSettings->rel_sol_change_tol;
	opt_data = NULL;

	//Eigen::VectorXd lowerBounds{ numOfInvertedResistivityElements };
	//Eigen::VectorXd upperBounds{ numOfInvertedResistivityElements };
	//for (int i = 0; i < numOfInvertedResistivityElements; i++) {
	//	lowerBounds[i] = -limitOfparamLogNormalization;
	//	upperBounds[i] = +limitOfparamLogNormalization;
	//}
	//settings.lower_bounds = lowerBounds;
	//settings.upper_bounds = upperBounds;

	double adoptModelConstraint = 0.0;
	Eigen::VectorXd rhoVec{ numOfInvertedResistivityElements };
	isBelowRMSThreshold = false;

	optMethod = invSettings->optMethod;
	isDirectSolver = invSettings->isDirectSolver;

	cout << "Optimization Method is :" << optMethod << endl;
	if (isDirectSolver) {
		cout << "Direct Solver PARDISO is Used." << endl;
	}
	else {
		cout << "Iterative Solver BiCGSTAB is Used." << endl;
	}
	initObjVal = 0.0;

	infofile.open("optimizeInfo.txt", std::ios::out);
	
	vector<Eigen::VectorXd> resistivitiesEachLambda(invSettings->lambdaVector.size());
	vector<bool> isBelowRMS(invSettings->lambdaVector.size());
	int adoptedLambdaNumber = -1;

	settings.resisVec.resize(numOfInvertedResistivityElements);
	settings.resisVec_p.resize(numOfInvertedResistivityElements);
	settings.resisVec.setZero();
	settings.resisVec_p.setZero();

	if (invSettings->lambdaVector.size() == 0) {
		//for (int i = 0; i < numOfCalcModelConstraint;i++) {
		

		for (int i = numOfCalcModelConstraint - 1; i >= 0; i--) {
			settings.numOfIteration = -1;
			preParams.resize(numOfInvertedResistivityElements);
			preParams.setZero();
			//weightRoughening = modelConstraintMax - (modelConstraintMax - modelConstraintMin)*i / (numOfCalcModelConstraint - 1);
			weightRoughening = modelConstraintMin * std::pow(10.0, std::log10(modelConstraintMax / modelConstraintMin)*double(i) / double(numOfCalcModelConstraint - 1));
			bool success = false;
			if (optMethod == "GD") {
				
				settings.gd_settings.isRestartAdam = false;
				success = optim::gd(paramVec, optFunc, opt_data, settings);
			}
			else if (optMethod == "LBFGS") {
				//int itermax = settings.iter_max;
				//settings.iter_max = settings.lbfgs_settings.restart_M;
				//int iter = 0;
				//while (iter < itermax && success==false) { //restart every restart_M iteration
				initObjVal = 0.0;
				success = optim::lbfgs(paramVec, optFunc, opt_data, settings);
				//	iter += settings.lbfgs_settings.restart_M;
				//	cout << "#Iteration::" << iter << endl;
				//}
				//settings.iter_max = itermax;
			}
			else if (optMethod == "GD-LBFGS") {
				settings.grad_err_tol = settings.grad_err_tol * 10;
				success = optim::lbfgs(paramVec, optFunc, opt_data, settings);
				settings.grad_err_tol = settings.grad_err_tol / 10;
				//initObjVal = 0;
				success = optim::gd(paramVec, optFunc, opt_data, settings);
				//success = optim::lbfgs_gd(paramVec, optFunc, opt_data, settings);

			}
			else {
				success = optim::cg(paramVec, optFunc, opt_data, settings);
			}


			if (isBelowRMSThreshold == true) {
				cout << "Optimization has finished successfully." << endl;
				break;
			}
			//initObjVal = 0.0;

			//if (isBelowRMSThreshold == true) {
			//	for (int j = 0; j < numOfInvertedResistivityElements; j++) {
			//		rhoVec.coeffRef(j) = invertedRhoIDToElementVector[j]->resistivity;
			//	}
			//	adoptModelConstraint = weightRoughening;
			//}
			//else {
			//	if (i == 0) {
			//		adoptModelConstraint = weightRoughening;
			//	}
			//	else {
			//		for (int j = 0; j < numOfInvertedResistivityElements; j++) {
			//			invertedRhoIDToElementVector[j]->resistivity = rhoVec.coeff(j);
			//		}
			//	}
			//	break;
			//}

		}
	}
	else {
		bool isAscendingOrder = false;
		bool isDescendingOrder = false;
		for (int i = 1; i < invSettings->lambdaVector.size(); i++) {
			if (invSettings->lambdaVector[i - 1] > invSettings->lambdaVector[i]) {
				isDescendingOrder = true;
			}
			else if (invSettings->lambdaVector[i - 1] < invSettings->lambdaVector[i]) {
				isAscendingOrder = true;
			}
		}
		if (isAscendingOrder && isDescendingOrder) {
			cout << "Warning::Mixed Descending and Ascending Order in Model Weight Order is detected.\nContinue as  Ascending Order. " << endl;
		}
		else if (isAscendingOrder) {
			cout << "Model Weight Order is Ascending." << endl;
		}
		else if (isAscendingOrder==false){
			cout << "Model Weight Order is Descending." << endl;
		}
		
		for (int i =  0; i < invSettings->lambdaVector.size(); i++) {
		//for (int i = invSettings->lambdaVector.size() - 1; i >= 0 ; i--) {
			settings.rel_objfn_change_tol = invSettings->objFuncChangeThresholdVector[i];
			settings.rel_resis_change_tol = invSettings->thresholdRelativeResistivityChangeVector[i];
			//invSettings->objFuncChangeThresholdForNextmodelConstraint = invSettings->objFuncChangeThresholdVector[i];
			//invSettings->thresholdResistivityChange = invSettings->thresholdResistivityChangeVector[i];
			preParams.resize(numOfInvertedResistivityElements);
			preParams.setZero();
			isBelowRMS[i] = false;
			weightRoughening = invSettings->lambdaVector[i];
			
			settings.grad_err_tol = invSettings->grad_err_tolVector[i];
			settings.iter_max = invSettings->maxIterationVector[i];
			bool success = false;
			
			if (optMethod == "GD") {
				
				if (invSettings->inheritPreviousSettingAdam && i >= 1 && weightRoughening == invSettings->lambdaVector[i - 1]) {
					settings.gd_settings.inheritPreviousSettingAdam = true;
					settings.gd_settings.par_step_size *= invSettings->par_step_sizeVector[i];
					inheritPreviousObjVal = true;
					isFirstLoopInheritPreviousObjVal = true;
					numOfSameModelWeightCalc++;
				}
				else {
					inheritPreviousObjVal = false;
					initObjVal = 0.0;
					settings.gd_settings.par_step_size = invSettings->par_step_sizeVector[i];
					settings.gd_settings.inheritPreviousSettingAdam = false;
					numOfSameModelWeightCalc = 1;
				}
				success = optim::gd(paramVec, optFunc, opt_data, settings);
			}
			else if (optMethod == "LBFGS") {
				initObjVal = 0.0;
				success = optim::lbfgs(paramVec, optFunc, opt_data, settings);
			}
			else if (optMethod == "GD-LBFGS") {
				settings.grad_err_tol = settings.grad_err_tol * 10;
				success = optim::lbfgs(paramVec, optFunc, opt_data, settings);
				settings.grad_err_tol = settings.grad_err_tol / 10;
				success = optim::gd(paramVec, optFunc, opt_data, settings);

			}
			else {
				success = optim::cg(paramVec, optFunc, opt_data, settings);
			}

			isBelowRMS[i] = isBelowRMSThreshold;
			resistivitiesEachLambda[i].resize(numOfCalcElements);
			for (int j = 0; j < numOfCalcElements; j++) {
				resistivitiesEachLambda[i][j] = calcElementsVector[j]->resistivity;
			}
			if (isAscendingOrder == true) {
				if (i >0 && isBelowRMS[i] == false) {
					if (isBelowRMS[i - 1] == true) {
						adoptedLambdaNumber = i - 1;
						adoptModelConstraint = invSettings->lambdaVector[i - 1];
					}
					else {
						cout << "Optimization is NOT CONVERGED." << endl;
						adoptedLambdaNumber = 0;
						adoptModelConstraint = invSettings->lambdaVector[0];
					}
				}
				else if (i == 0 && isBelowRMS[i] == false) {
					cout << "Optimization is NOT CONVERGED." << endl;
					adoptedLambdaNumber = 0;
					adoptModelConstraint = invSettings->lambdaVector[0];
				}
				else if (i == invSettings->lambdaVector.size() - 1 && isBelowRMS[i] == true) {
					cout << "Optimization is CONVERGED at the Largest model Constraint." << endl;
					adoptedLambdaNumber = invSettings->lambdaVector.size() - 1;
					adoptModelConstraint = invSettings->lambdaVector[invSettings->lambdaVector.size() - 1];
				}
			}
			else if (isAscendingOrder == false) {
				if (isBelowRMSThreshold == true) {
					cout << "Optimization has finished successfully." << endl;
					adoptModelConstraint = invSettings->lambdaVector[i];
					adoptedLambdaNumber = i;
					break;
				}
				else if (i == invSettings->lambdaVector.size() - 1) {
					adoptModelConstraint = invSettings->lambdaVector[i];
					adoptedLambdaNumber = i;
				}
			}
		}
	}
	std::cout << "Adopted Model Constraint:" << adoptModelConstraint << endl;
	for (int j = 0; j < numOfCalcElements; j++) {
		calcElementsVector[j]->resistivity = resistivitiesEachLambda[adoptedLambdaNumber][j];
	}

	CalcForward(false);
	output->RhoOutput(&elements);
	output->TxtOutputResistivity(&elements, "FinalResistivity.txt");
	infofile.close();

	//if (success) {
	//	std::cout << "cg: sphere test completed successfully." << std::endl;
	//}
	//else {
	//	std::cout << "cg: sphere test completed unsuccessfully." << std::endl;
	//}
}
void Analysis::Analysis::CountObsData() {
	numOfObsDataPerOmega.resize(boundary->omega.size());
	for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
		numOfObsDataPerOmega[iOmega] = 0;
	}
	//Count Data
	numOfObsData = 0;
	
	vector<int> lastID;
	lastID.resize(boundary->omega.size());
	for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
		lastID[iOmega] = 0;
	}
	for (int j = 0; j < numOfObsPointElements; j++) {
		Element::Element* element = obsPointElements[j];
		if (element->isInversionImpedance == true) {
			element->impedanceObsData->rowIDsJacobian.resize(boundary->omega.size());
			element->impedanceObsData->rowIDEachOmega.resize(boundary->omega.size());
			for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {
				element->impedanceObsData->rowIDsJacobian[iOmega] = numOfObsData;
				numOfObsData += 4 * 2; //real and imag parts
				numOfObsDataPerOmega[iOmega] += 4 * 2;

				element->impedanceObsData->rowIDEachOmega[iOmega] = lastID[iOmega];
				lastID[iOmega] += 4 * 2;

			}
		}
	}
	
	for (int j = 0; j < numOfObsPointElements; j++) {
		Element::Element* element = obsPointElements[j];
		if (element->isInversionTipper == true) {
			element->tipperObsData->rowIDsJacobian.resize(boundary->omega.size());
			element->tipperObsData->rowIDEachOmega.resize(boundary->omega.size());

			for (int iOmega = 0; iOmega < boundary->omega.size(); iOmega++) {	
				element->tipperObsData->rowIDsJacobian[iOmega] = numOfObsData;
				numOfObsData = numOfObsData + 2 * 2; //real and imag parts
				numOfObsDataPerOmega[iOmega] += 2 * 2;
				element->tipperObsData->rowIDEachOmega[iOmega] = lastID[iOmega];
				lastID[iOmega] += 2 * 2;
			}
		}
		// Todo::他のテンソル量を逆解析する場合はここに足す
	}
	
}
void Analysis::Analysis::SetSameResistivityToBoundaryCell() {


	//First, decide resistivities of all inside elements.
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	Eigen::Vector3i pos;
	//	pos.setZero();
	//	if (element->property->type == Property::Property::NORMAL && element->isAirGroundBoundaryCell == true) {
	//	pos.coeffRef(2) = +1;
	//	}
	//	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	//	if (pos.coeff(0) == 0 && pos.coeff(1) == 0 && pos.coeff(2) == 0) {
	//		continue;
	//	}
	//	if (element->layer == element->neighborElements[ipos]->layer && element->neighborElements[ipos]->isParent == false) {
	//		element->masterResistivityElement = element->neighborElements[ipos];

	//		//the third subsurface layer 
	//		//if (element->neighborElements[ipos]->neighborElements[ipos]->layer == element->neighborElements[ipos]->layer && element->neighborElements[ipos]->neighborElements[ipos]->isParent == false) {
	//		//	element->neighborElements[ipos]->neighborElements[ipos]->masterResistivityElement = element->neighborElements[ipos];
	//		//}
	//		//else {
	//		//	cout << "The Third Subsurface Cell must be the same layer to the Second One" << endl;
	//		//	exit(-1);
	//		//}
	//		//

	//	}
	//	else {
	//		cout << "Boundary Cell must be the same layer to the neighbor" << endl;
	//		exit(-1);
	//	}
	//}

	//And then, decide boundary elements resistivities.
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	Eigen::Vector3i pos;
	//	pos.setZero();
	//	if (element->property->type == Property::Property::NORMAL && element->boundary == "-X_BOUNDARY") {
	//		pos.coeffRef(0) = 1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "+X_BOUNDARY") {
	//		pos.coeffRef(0) = -1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "-Y_BOUNDARY") {
	//		pos.coeffRef(1) = 1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "+Y_BOUNDARY") {
	//		pos.coeffRef(1) = -1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "-Z_BOUNDARY") {
	//		pos.coeffRef(2) = 1;
	//	}
	//	else if (element->property->type == Property::Property::NORMAL && element->boundary == "+Z_BOUNDARY") {
	//		pos.coeffRef(2) = -1;
	//	}
	//	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	//	if ((pos.coeff(0)==0 && pos.coeff(1)==0 && pos.coeff(2)==0) || element->isAirGroundBoundaryCell==true) {
	//		continue;
	//	}
	//	if (element->layer == element->neighborElements[ipos]->layer && element->neighborElements[ipos]->isParent==false) {
	//		element->masterResistivityElement = element->neighborElements[ipos];
	//	}
	//	else {
	//		cout << "Boundary Cell must be the same layer to the neighbor" << endl;
	//		exit(-1);
	//	}

	//}
	//Second ,set resistivity and foundamental parent elements(for example, master of mater)
	for (int i = 0; i < numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		element->masterResistivityElement = SearchMasterElement(element);
		if (element->masterResistivityElement != nullptr) {
			element->resistivity = element->masterResistivityElement->resistivity;
		}
	}

	//At last, set second layer from air ground
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	if (element->isSecondCellOfAirGroundBoundary ==true) {
	//		element->resistivity = (element->neighborElements[1 + 3 + 0]->resistivity + element->neighborElements[1 + 3 + 9 * 2]->resistivity)/2.0;
	//	}
	//}

	//test, not slave but same resis in sueface layer
	//for (int i = 0; i < numOfCalcElements; i++) {
	//	Element::Element* element = calcElementsVector[i];
	//	Eigen::Vector3i pos;
	//	pos.setZero();
	//	if (element->property->type == Property::Property::NORMAL && element->isAirGroundBoundaryCell == true) {
	//		pos.coeffRef(2) = +1;
	//	}
	//	int ipos = (pos.coeff(0) + 1) + 3 * (pos.coeff(1) + 1) + 9 * (pos.coeff(2) + 1);
	//	element->resistivity = element->neighborElements[ipos]->resistivity;
	//}
}
bool Analysis::Analysis::CalcRhoFromParamAndDRhoDParam(Eigen::VectorXd paramVec) {
	
	dRhoDParam.setZero();
	bool isChangeResis = false;
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		kv::autodif<double> x;
		//if (paramVec.coeff(i) > limitOfparamLogNormalization){
		//	x = kv::autodif<double>::init(limitOfparamLogNormalization);
		//	dRhoDParam.coeffRef(i) = 1e-3;
		//}
		//else if (paramVec.coeff(i) < -limitOfparamLogNormalization) {
		//	x = kv::autodif<double>::init(-limitOfparamLogNormalization);
		//	dRhoDParam.coeffRef(i) = 1e-3;
		//}
		//else {

		double logMaxResis = log(maxResis);
		double logMinResis = log(minResis);
		//double logMaxResis = maxResis;
		//double logMinResis = minResis;
		x = kv::autodif<double>::init(paramVec.coeff(i));
		kv::autodif<double> enx = exp(paramLogNormalization*x);
		//kv::autodif<double> m = (maxResis*enx + minResis) / (1 + enx);
		kv::autodif<double> m = (logMaxResis*enx + logMinResis) / (1 + enx);
		kv::autodif<double> resis = pow(std::exp(1.0), m);
		//kv::autodif<double> resis = m;
		if (std::isnan(resis.v) == true || std::isnan(resis.d(0))==true) {
			double resisv;
			if (paramVec.coeff(i) > 0) {
				resisv = maxResis;
			}
			else {
				resisv = minResis;
			}
			invertedRhoIDToElementVector[i]->resistivity = resisv;
			dRhoDParam.coeffRef(i) = 0.0;
		}
		else {
			if (preParams[i] != paramVec[i]) {
				invertedRhoIDToElementVector[i]->resistivity = resis.v;
				isChangeResis = true;
			}
			dRhoDParam.coeffRef(i) = resis.d(0);
		}
		
		//if (paramVec.coeff(i) >= maxResis) {
		//	invertedRhoIDToElementVector[i]->resistivity = maxResis;
		//}
		//else if (paramVec.coeff(i) <= minResis) {
		//	invertedRhoIDToElementVector[i]->resistivity = minResis;
		//}
		//else {
		//	invertedRhoIDToElementVector[i]->resistivity = paramVec.coeffRef(i) ;
		//	isChangeResis = true;
		//}
		//dRhoDParam.coeffRef(i) = 1.0;

	}
	
	preParams = paramVec;

	return isChangeResis;
}
Eigen::VectorXd Analysis::Analysis::CalcParamFromRho() {
	Eigen::VectorXd paramVec;
	paramVec.resize(numOfInvertedResistivityElements);
	paramVec.setZero();
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		double m = log(invertedRhoIDToElementVector[i]->resistivity);
		//double m = invertedRhoIDToElementVector[i]->resistivity;
		//double x = 1 / paramLogNormalization * log((m - minResis) / (maxResis - m));
		double logMaxResis = log(maxResis);
		double logMinResis = log(minResis);
		//double logMaxResis = maxResis;
		//double logMinResis = minResis;
		double x = 1 / paramLogNormalization * log((m - logMinResis) / (logMaxResis - m));
		paramVec.coeffRef(i) = x;


		//paramVec.coeffRef(i) = invertedRhoIDToElementVector[i]->resistivity;

	}
	return paramVec;
}
Element::Element* Analysis::Analysis::SearchMasterElement(Element::Element* slaveElement) {
	
	if (slaveElement->masterResistivityElement != nullptr) {
		Element::Element* masterElem = slaveElement->masterResistivityElement;
		while (true) {
			if (masterElem->masterResistivityElement != nullptr) {
				masterElem=masterElem->masterResistivityElement;
			}
			else {
				return masterElem;
			}
		}
	}
	else {
		return nullptr;
	}

	
}

void Analysis::Analysis::CountIndependentInvertedResisElements() {
	numOfIndependentInvertedResisElements = 0;
	for (int i = 0; i < numOfInvertedResistivityElements; i++) {
		if (invertedRhoIDToElementVector[i]->masterResistivityElement == nullptr) {
			numOfIndependentInvertedResisElements++;
		}
	}
}

void Analysis::Analysis::CalcJacobian(int iOmega) {
	//std::ofstream f2;
	//f2.open("debugZtmpdZdHtmp.txt", std::ios::trunc);
	Eigen::SparseMatrix < std::complex<double>, Eigen::RowMajor> dRdRhotmp{ 2 * 3 * numOfCalcElements,numOfInvertedResistivityElements };

	for (int i = 0; i < 2 * 3 * numOfCalcElements; i++) {
		dRdRhotmp.row(i) = dRdRho->row(iOmega * 2 * 3 * numOfCalcElements + i);
	}
	Eigen::SparseMatrix<std::complex<double>> M1{ 3 * numOfCalcElements,int(numOfObsDataPerOmega[iOmega] /2) }; //once real part of Z is solved, imag part can be calculated by using it. 
	Eigen::SparseMatrix<std::complex<double>> M2{ 3 * numOfCalcElements,int(numOfObsDataPerOmega[iOmega] /2) };

	cout << "Calculating dDatadH matrix.." << endl;
	for (int j = 0; j < numOfObsPointElements; j++) {
		Element::Element* element = obsPointElements[j];
		//===Impedance====
		if (element->isInversionImpedance == true) {
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdH[iOmega](ii, jj), 0); it; ++it)
					{
						int iCol = it.col();

						/*cout << "in calcLambda" << ii << " " << jj << " " << element->dZdH[iOmega](ii, jj).coeff(0, 13981) << endl;*/
						std::complex<double> dZdHtmp = element->dZdH[iOmega](ii, jj).coeff(0, iCol);
						//f2 << iCol << " " << ii << " " << jj << " " << dZtmp << " " << dZdHtmp << "before" << endl;
						double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
						double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));
						if (iCol < 3 * numOfCalcElements) {
							//real part
							M1.coeffRef(iCol, int(element->impedanceObsData->rowIDEachOmega[iOmega] / 2) + ii + 2 * jj).real(dZdHtmp.real() / epsReal);
							//imag part
							M1.coeffRef(iCol, int(element->impedanceObsData->rowIDEachOmega[iOmega] / 2) + ii + 2 * jj).imag(dZdHtmp.imag() / epsImag);
						}
						else {
							//real part
							M2.coeffRef(iCol - 3 * numOfCalcElements, int(element->impedanceObsData->rowIDEachOmega[iOmega] / 2) + ii + 2 * jj).real(dZdHtmp.real() / epsReal);
							//imag part
							M2.coeffRef(iCol - 3 * numOfCalcElements, int(element->impedanceObsData->rowIDEachOmega[iOmega] / 2) + ii + 2 * jj).imag(dZdHtmp.imag() / epsImag);
						}
					}
				}
			}
		}
		//===Tipper=====
		if (element->isInversionTipper == true) {
			for (int ii = 0; ii < 2; ii++) {
				//Real Part
				for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dTdH[iOmega](ii), 0); it; ++it)
				{
					int iCol = it.col();
					std::complex<double> dTdHtmp = element->dTdH[iOmega](ii).coeff(0, iCol);

					double epsReal = std::abs(element->tipperObsData->varianceTobsVectorReal[iOmega].coeff(ii));
					double epsImag = std::abs(element->tipperObsData->varianceTobsVectorImag[iOmega].coeff(ii));
					if (iCol < 3 * numOfCalcElements) {
						//real part
						M1.coeffRef(iCol, int(element->tipperObsData->rowIDEachOmega[iOmega] / 2) + ii).real(dTdHtmp.real() / epsReal);
						//imag part
						M1.coeffRef(iCol, int(element->tipperObsData->rowIDEachOmega[iOmega] / 2) + ii).imag(dTdHtmp.imag() / epsImag);
					}
					else {
						//real part
						M2.coeffRef(iCol - 3 * numOfCalcElements, int(element->tipperObsData->rowIDEachOmega[iOmega] / 2) + ii).real(dTdHtmp.real() / epsReal);
						//imag part
						M2.coeffRef(iCol - 3 * numOfCalcElements, int(element->tipperObsData->rowIDEachOmega[iOmega] / 2) + ii).imag(dTdHtmp.imag() / epsImag);
					}
				}

			}
		}
		// Todo::他のテンソル量を逆解析する場合はここに足す
	}

	cout << "Calculating Lambda.." << endl;
	cout << "  Calculating Lambda of H1.." << endl;
	time_t start_t = time(NULL);
	Eigen::MatrixXcd res1{ 3 * numOfCalcElements,int(numOfObsDataPerOmega[iOmega] / 2) };
	solver1->pardisoParameterArray()(11) = 1; //set to adjoint mode
	res1 = solver1->solve(M1.conjugate());
	solver1->pardisoParameterArray()(11) = 0; //reset to normal mode
	time_t end_t = time(NULL);
	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;

	cout << "  Calculating Lambda of H2.." << endl;
	start_t = time(NULL);
	Eigen::MatrixXcd res2{ 3 * numOfCalcElements,int(numOfObsDataPerOmega[iOmega] / 2) };
	solver2->pardisoParameterArray()(11) = 1; //set to adjoint mode
	res2 = solver2->solve(M2.conjugate());
	solver2->pardisoParameterArray()(11) = 0; //reset to normal mode
	end_t = time(NULL);
	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
	cout << "End Calculating Lambda.." << endl;


	//Set Lambda, this includes real and imag parts of Z
	Eigen::MatrixXcd lambdaTmp{2 * 3 * numOfCalcElements,numOfObsDataPerOmega[iOmega] };

	for (int j = 0; j<int(numOfObsDataPerOmega[iOmega] / 2); j++) {
		lambdaTmp.block(0, 2 * j, 3 * numOfCalcElements, 1) = res1.col(j);
		lambdaTmp.block(3 * numOfCalcElements, 2 * j, 3 * numOfCalcElements, 1) = std::complex<double>(0, 1)* res2.col(j);
		//set using Cauchy–Riemann equations
		lambdaTmp.block(0, 2 * j + 1, 3 * numOfCalcElements, 1) = res1.col(j);
		lambdaTmp.block(3 * numOfCalcElements, 2 * j + 1, 3 * numOfCalcElements, 1) = std::complex<double>(0, 1)* res2.col(j);
		cout << j << endl;
		//for (int i = 0; i < 3 * numOfCalcElements; i++) {
		//	lambdaTmp.coeffRef(i, 2 * j) = res1.coeffRef(i, j);
		//	lambdaTmp.coeffRef(3*numOfCalcElements + i, 2 * j) = res2.coeffRef(i, j);
		//	//set using Cauchy–Riemann equations
		//	lambdaTmp.coeffRef(i, 2 * j + 1) = std::complex<double>(0,1)* res1.coeffRef(i, j);
		//	lambdaTmp.coeffRef(3 * numOfCalcElements + i, 2 * j + 1) = std::complex<double>(0, 1)*res2.coeffRef(i, j);
		//	cout << i << " " << j << endl;
		//}
	}

	//calc jacobian
	Eigen::MatrixXd jac{ numOfObsDataPerOmega[iOmega],numOfInvertedResistivityElements };
	//∂Z/∂Rho
	start_t = time(NULL);
	cout << "Calculating Jacobian #Omega:" << iOmega << endl;
	for (int j = 0; j < numOfObsPointElements; j++) {
		Element::Element* element = obsPointElements[j];
		//===Impedance====
		if (element->isInversionImpedance == true) {
			for (int ii = 0; ii < 2; ii++) {
				for (int jj = 0; jj < 2; jj++) {
					for (Eigen::SparseMatrix<complex<double>, Eigen::RowMajor>::InnerIterator it(element->dZdRho[iOmega](ii, jj), 0); it; ++it)
					{
						int iCol = calcElementsVector[it.col()]->invertedRhoElementsID;
						if (iCol >= 0) {
							std::complex<double> dZdRhotmp = element->dZdRho[iOmega](ii, jj).coeff(0, iCol);
							double epsReal = std::abs(element->impedanceObsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
							double epsImag = std::abs(element->impedanceObsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));
							//d(ReZ)/dRho
							cout << numOfObsDataPerOmega[iOmega] << " " << element->impedanceObsData->rowIDEachOmega[iOmega] << " " << ii << " " << jj << " " << iCol << endl;
							jac.coeffRef(element->impedanceObsData->rowIDEachOmega[iOmega] + 2 * ii + 2 * 2 * jj, iCol) += dZdRhotmp.real() / epsReal;
							//d(ReZ)/dRho
							jac.coeffRef(element->impedanceObsData->rowIDEachOmega[iOmega] + 2 * ii + 2 * 2 * jj + 1, iCol) += dZdRhotmp.imag() / epsImag;
						}
					}
				}
			}
		}
	}
	//Tipper, ∂T/∂Rho=0
	//Eigen::setNbThreads(0);
	Eigen::initParallel();
	Eigen::MatrixXd lambdaDRDRho{ numOfObsDataPerOmega[iOmega],numOfInvertedResistivityElements };
	lambdaDRDRho = (lambdaTmp.adjoint()*dRdRhotmp).real();
	jac = jac - lambdaDRDRho;
	//Eigen::setNbThreads(1);
	Eigen::initParallel();
	//set jac to global jacobian
	int startID = 0;
	for (int i = 0; i < iOmega - 1; i++) {
		startID += numOfObsDataPerOmega[i];
	}

	cout << "startID" << startID << endl;
	for (int i = 0; i < numOfObsDataPerOmega[iOmega]; i++) {
		jacobian->row(startID + i) = jac.row(i);
	}

	std::cout << "Calculation Time:" << end_t - start_t << " Seconds." << endl;
	cout << "End Calculating Jacobian #Omega:" << iOmega << endl;
}
void Analysis::Analysis::SetInitialResistivityFromFile() { //Set Nearest Resistivity in InitialData
	if (initialData.size() == 0) {
		return;
	}
	for (int i = 0; i<numOfCalcElements; i++) {
		Element::Element* element = calcElementsVector[i];
		double nearestDistance = 1e30;
		int nearestID = -1;
		for (int j = 0; j < initialData.size(); j++) {
			double distance = (element->centerCoord - initialData[j]->coord).norm();
			if (distance < nearestDistance) {
				nearestDistance = distance;
				nearestID = initialData[j]->ID;
			}
		}
		if (nearestID == -1) {
			cout << "Error In SetInitialResistivityFromFile" << endl;
			exit(1);
		}
		element->resistivity = initialData[nearestID]->resistivity;
	}
}

