/*
Copyright © 2025 Suzuki Atsushi <mk.pn14951011 at gmail.com>
For Non-commercial uses, this work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
For commercial uses, when you use this work until April 1, 2025, you must publish the original or derived source works you use as the terms of the Do What The Fuck You Want To Public License, Version 2.
This term for commercial uses is prioritized the other terms for this work.
The other conditions are same as terms of the Do What The Fuck You Want To Public License, Version 2.
After April 1, 2025, this term is expired and this work comply terms of the Do What The Fuck You Want To Public License, Version 2.
*/
#pragma once
#include "Analysis.h"
#include "Element.h"
#include <unsupported/Eigen/NonLinearOptimization>
// tolerance for chekcing number of iterations
#define LM_EVAL_COUNT_TOL 4/3

#define LM_CHECK_N_ITERS(SOLVER,NFEV,NJEV) { \
            ++g_test_level; \
            VERIFY_IS_EQUAL(SOLVER.nfev, NFEV); \
            VERIFY_IS_EQUAL(SOLVER.njev, NJEV); \
            --g_test_level; \
            VERIFY(SOLVER.nfev <= NFEV * LM_EVAL_COUNT_TOL); \
            VERIFY(SOLVER.njev <= NJEV * LM_EVAL_COUNT_TOL); \
        }

using namespace Eigen;
namespace ObjectiveFunction {
	// Generic functor
	template<typename _Scalar, int NX = Dynamic, int NY = Dynamic>
	struct Functor
	{
		typedef _Scalar Scalar;
		enum {
			InputsAtCompileTime = NX,
			ValuesAtCompileTime = NY
		};
		typedef Matrix<Scalar, InputsAtCompileTime, 1> InputType;
		typedef Matrix<Scalar, ValuesAtCompileTime, 1> ValueType;
		typedef Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;

		const int m_inputs, m_values;

		Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
		Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

		int inputs() const { return m_inputs; }
		int values() const { return m_values; }

		// you should define that in the subclass :
	  //  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
	};

	struct ObjectiveFunction: Functor<double>
	{
		Analysis::Analysis* analysis;
		int paramSize;
		int termSize;
		ObjectiveFunction(Analysis::Analysis* a,int nx,int ny) : Functor<double>(nx, ny) {
			analysis = a;
			paramSize = nx;
			termSize = ny;
		}
		int operator()(const VectorXd &x, VectorXd &fvec)
		{
			bool isChangeResis = false;
			isChangeResis = analysis->CalcRhoFromParamAndDRhoDParam(x);
			analysis->SetSameResistivityToBoundaryCell();
			//isChangeResis = true; //test
			cout << "isChangeResis" << isChangeResis << endl;
			if (isChangeResis == true) {
				analysis->CalcForward(true);
			}
			double obj_val;
			obj_val = 0.0;
			obj_val += analysis->CalcDataMisfit();
			double RMS = std::pow(obj_val / analysis->numOfObsData, 0.5);
			std::cout << "RMS:" << RMS << endl;
			std::cout << "DataMisfit:" << obj_val << std::endl;
			double roughningMatrixPenaltyTerm = analysis->CalcRoughningMatrixPenalty();
			std::cout << "weightRoughening:" << analysis->weightRoughening << " PemaltyTerm:" << roughningMatrixPenaltyTerm << std::endl;

			obj_val += analysis->weightRoughening * roughningMatrixPenaltyTerm;

			analysis->output->RhoOutput(&analysis->elements);

			analysis->output->OutputObsCalcImpedance(analysis->boundary->omega, &analysis->obsPointElements);

			obj_val = obj_val / analysis->initObjVal;
			std::cout << "Objective Function Value:" << obj_val << std::endl;
			//judge convergence
			if (RMS < analysis->thresholdRMS) {
				//some process to judge
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

			//Impedance Data
			int iData = 0;
			for (int i = 0; i < analysis->numOfObsPointElements; i++) {
				Element::Element* element = analysis->obsPointElements[i];
				if (element->isInversionImpedance == true) {
					for (int iOmega = 0; iOmega < analysis->boundary->omega.size(); iOmega++) {
						for (int ii = 0; ii < 2; ii++) {
							for (int jj = 0; jj < 2; jj++) {
								double epsReal = std::abs(element->obsData->varianceZobsVectorReal[iOmega].coeff(ii, jj));
								double epsImag = std::abs(element->obsData->varianceZobsVectorImag[iOmega].coeff(ii, jj));

								fvec[iData] = (element->obsData->ZobsVector[iOmega].coeff(ii, jj).real()
									- element->Z[iOmega].coeff(ii, jj).real())/ epsReal;
								iData++;
								fvec[iData] = (element->obsData->ZobsVector[iOmega].coeff(ii, jj).imag()
									- element->Z[iOmega].coeff(ii, jj).imag())/ epsReal;
								iData++;
							}
						}
					}
				}
			}
			return 0;
		}
		int df(const VectorXd &x, VectorXd &jac_row, VectorXd::Index rownb)
		{
			analysis->dDataMisfitDRho.setZero();
			analysis->dRdRho.setZero();
			analysis->lambdaEachOmega.setZero();
			analysis->dJdRho.setZero();
			analysis->CalcDDataMisfitDRho();
			analysis->CalcDJDRho();
			return 0;
		}
	};
}