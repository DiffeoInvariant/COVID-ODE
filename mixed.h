#ifndef MIXED_MODEL_H
#define MIXED_MODEL_H

#include <petscts.h>
#include <petsctao.h>

typedef enum {
		     SEIS=0,
		     SEIR=1
} MixedModelType;

typedef PetscErrorCode (*RHSJPFunc)(TS,PetscReal,Vec,Mat,void*);

struct _mixed_model {
  MixedModelType type;
  TSRHSFunction  rhs;
  TSRHSJacobian  rhs_jac;
  RHSJPFunc      rhs_jacp;
  PetscInt       states;
  PetscScalar    *params;
  TS             ts;
  Mat            Jac, JacP;/*Jacobian for state and parameters*/
  Vec            X, F, *lambda, *mu;/*adjoint variables*/
  PetscBool      custom_cost_gradient;
};

typedef struct _mixed_model *MixedModel;

extern PetscErrorCode MixedModelCreate(MixedModel *model);

extern PetscErrorCode MixedModelDestroy(MixedModel model);

extern PetscErrorCode MixedModelSetParams(MixedModel model, PetscScalar *params);

extern PetscErrorCode MixedModelSetType(MixedModel model, MixedModelType type);

extern PetscErrorCode MixedModelSetTimeStep(MixedModel model, PetscReal dt);

extern PetscErrorCode MixedModelSetMaxTime(MixedModel model, PetscReal tmax);

extern PetscErrorCode MixedModelSetTSType(MixedModel model, TSType type);

extern PetscErrorCode MixedModelSetFromOptions(MixedModel model);

extern PetscErrorCode SEISCreate(MixedModel *seis);

extern PetscErrorCode MixedModelSolve(MixedModel model, Vec X0);

extern PetscErrorCode MixedModelSetCostGradients(MixedModel model, Vec *lambda, Vec *mu);

extern PetscErrorCode MixedModelAdjointSolve(MixedModel model);

#endif
