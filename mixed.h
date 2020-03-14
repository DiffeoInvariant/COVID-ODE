#ifndef MIXED_MODEL_H
#define MIXED_MODEL_H

#include <petscts.h>
#include <petsctao.h>
#include "obs.h"

typedef enum {
		     SEIS=0,
		     SEIR=1
} MixedModelType;

typedef enum {
	      VAR_IC=0,
	      VAR_PARAMS=1,
	      VAR_IC_PARAMS=2
} MixedModelOptimizationProblem;

typedef PetscErrorCode (*RHSJPFunc)(TS,PetscReal,Vec,Mat,void*);

typedef PetscErrorCode (*R0Func)(PetscScalar *, PetscReal *);

struct _mixed_model {
  MixedModelType type;
  TSRHSFunction  rhs;
  TSRHSJacobian  rhs_jac;
  RHSJPFunc      rhs_jacp;
  R0Func         basic_r0_func;
  PetscInt       states;
  PetscScalar    *params;
  PetscReal      population;
  TS             ts;
  Tao            tao;
  Mat            Jac, JacP;/*Jacobian for state and parameters*/
  Vec            X, X0Opt, F, *lambda, *mu;/*adjoint variables*/
  PetscBool      custom_cost_gradient, lambda_mu_allocated;
  EpidemicObs    obs;
  MixedModelOptimizationProblem problem_type;
};

typedef struct _mixed_model *MixedModel;

extern PetscErrorCode MixedModelCreate(MixedModel *model);

extern PetscErrorCode MixedModelDestroy(MixedModel model);

extern PetscErrorCode MixedModelSetParams(MixedModel model, PetscScalar *params);

extern PetscErrorCode MixedModelReproductionNumber(MixedModel model, PetscReal *R0);

extern PetscErrorCode MixedModelGetTotalDeaths(MixedModel model, PetscReal *TotalDeaths);

extern PetscErrorCode MixedModelGetInfectionDeaths(MixedModel model, PetscReal *InfectionDeaths);

extern PetscErrorCode MixedModelSetType(MixedModel model, MixedModelType type);

extern PetscErrorCode MixedModelSetTimeStep(MixedModel model, PetscReal dt);

extern PetscErrorCode MixedModelSetMaxTime(MixedModel model, PetscReal tmax);

extern PetscErrorCode MixedModelSetTSType(MixedModel model, TSType type);

extern PetscErrorCode MixedModelSetTaoType(MixedModel model, TaoType type);

extern PetscErrorCode MixedModelSetFromOptions(MixedModel model);

extern PetscErrorCode SEISCreate(MixedModel *seis);

extern PetscErrorCode SEIRCreate(MixedModel *seir);

extern PetscErrorCode MixedModelSolve(MixedModel model, Vec X0);

extern PetscErrorCode MixedModelSetCostGradients(MixedModel model, Vec *lambda, Vec *mu);

extern PetscErrorCode MixedModelAdjointSolve(MixedModel model);

extern PetscErrorCode MixedModelSetInitialCondition(MixedModel model, Vec X0);

extern PetscErrorCode MixedModelSetObservation(MixedModel model, EpidemicObs obs);

extern PetscErrorCode MixedModelSetProblemType(MixedModel model, MixedModelOptimizationProblem problem);

extern PetscErrorCode MixedModelOptimize(MixedModel model);

extern PetscErrorCode MixedModelOptimizeInitialCondition(MixedModel model);

extern PetscErrorCode MixedModelOptimizeParameters(MixedModel model);

extern PetscErrorCode MixedModelGenerateNoisyObs(MixedModel model, PetscReal mu, PetscReal sigma);

#endif
