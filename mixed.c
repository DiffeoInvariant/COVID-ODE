static char help[] = "Fully-mixed ODE epidemiology models. Currently only SEIS.\n\n";
#include "mixed.h"


const PetscInt _MMNumParams[] = {6, 6};
const PetscInt _MMNumStates[] = {3, 4};

PetscErrorCode SEISRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx);
PetscErrorCode SEISRHSJacobian(TS ts, PetscReal t, Vec X, Mat Jac, Mat Pre, void *ctx);
PetscErrorCode SEISRHSJacobianP(TS ts, PetscReal t, Vec X, Mat JacP, void *ctx);

PetscErrorCode SEIRRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx);
PetscErrorCode SEIRRHSJacobian(TS ts, PetscReal t, Vec X, Mat Jac, Mat Pre, void *ctx);
PetscErrorCode SEIRRHSJacobianP(TS ts, PetscReal t, Vec X, Mat JacP, void *ctx);

PetscErrorCode SEISR0Func(PetscScalar *params, PetscReal *R0);
PetscErrorCode SEIRR0Func(PetscScalar *params, PetscReal *R0);

const TSRHSFunction _MMRHS[] = {SEISRHSFunction, SEIRRHSFunction};
const TSRHSJacobian _MMRHS_JAC[] = {SEISRHSJacobian, SEIRRHSJacobian};
const RHSJPFunc     _MMRHS_JP[] = {SEISRHSJacobianP, SEIRRHSJacobianP};
const R0Func        _MMR0[] = {SEISR0Func, SEIRR0Func};



PetscErrorCode SEISR0Func(PetscScalar *params, PetscReal *R0)
{
  PetscScalar beta=params[1], eps=params[2], mu_b=params[3],
              mu_i=params[4], gamma=params[5];
  PetscFunctionBeginUser;
  *R0 = (eps / (eps + mu_b)) * (beta / (mu_b + mu_i + gamma));
  PetscFunctionReturn(0);
}

PetscErrorCode SEIRR0Func(PetscScalar *params, PetscReal *R0)
{
  PetscScalar beta=params[1], eps=params[2], mu_b=params[3],
              mu_i=params[4], gamma=params[5];
  PetscFunctionBeginUser;
  *R0 = (eps / (eps + mu_b)) * (beta / (mu_b + mu_i + gamma));
  PetscFunctionReturn(0);
}
  

#define _max(a,b) ((a) > (b) ? (a) : (b))

PetscErrorCode SEISRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  MixedModel        model = (MixedModel)ctx;
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x;
  PetscScalar       B=model->params[0], beta=model->params[1],
                    eps=model->params[2], mu_b=model->params[3],
                    mu_i=model->params[4], gamma=model->params[5];
  PetscReal         popchange;
  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);
  model->population = _max(x[0],0.0) + _max(x[1],0.0) + _max(x[2],0.0);
  if(model->population == 0.0){
    SETERRQ1(PETSC_COMM_WORLD, 1, "Error, population is zero at time %.4f.\n", t);
  }
  popchange = model->prev_population - model->population;
  if(popchange <= 0){
    model->dead += PetscAbsReal(popchange);
  }
  f[0] = B - beta * x[0] * x[2] / model->population - mu_b * x[0] + gamma * x[2];
  f[1] = beta * x[0] * x[2] / model->population - (eps + mu_b)*x[1];
  f[2] = eps * x[1] - (gamma + mu_i + mu_b)*x[2];
  /*PetscPrintf(PETSC_COMM_WORLD, "x[0] = %f, x[1] = %f, x[2] = %f.\n", x[0], x[1], x[2]);
    PetscPrintf(PETSC_COMM_WORLD, "f[0] = %f, f[1] = %f, f[2] = %f.\n\n", f[0], f[1], f[2]);*/
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  model->prev_population = model->population;
  PetscFunctionReturn(ierr);
}

PetscErrorCode SEISRHSJacobian(TS ts, PetscReal t, Vec X, Mat Jac, Mat Pre, void *ctx)
{
  PetscErrorCode    ierr;
  MixedModel        model=(MixedModel)ctx;
  PetscScalar       B=model->params[0], beta=model->params[1],
                    eps=model->params[2], mu_b=model->params[3],
                    mu_i=model->params[4], gamma=model->params[5];
  PetscInt          rows_and_cols[] = {0,1,2};
  PetscScalar       J[3][3];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  model->population = _max(x[0],0.0) + _max(x[1],0.0) + _max(x[2],0.0);
  J[0][0] = -beta * x[2] / model->population - mu_b;
  J[0][1] = 0.0;
  J[0][2] = -beta * x[0] / model->population + gamma;

  J[1][0] = beta * x[2] / model->population;
  J[1][1] = -(eps + mu_b);
  J[1][2] = beta * x[0] / model->population;

  J[2][0] = 0.0;
  J[2][1] = eps;
  J[2][2] = -(gamma + mu_i + mu_b);

  ierr = MatSetValues(Jac, 3, rows_and_cols, 3, rows_and_cols, &J[0][0], INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if(Pre != Jac){
    ierr = MatAssemblyBegin(Pre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Pre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SEISRHSJacobianP(TS ts, PetscReal t, Vec X, Mat JacP, void *ctx)
{
  PetscErrorCode    ierr;
  MixedModel        model=(MixedModel)ctx;
  PetscInt          rows[] = {0,1,2}, cols[]={0,1,2,3,4,5};
  PetscScalar       J[3][6];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  model->population = _max(x[0],0.0) + _max(x[1],0.0) + _max(x[2],0.0);
  J[0][0] = 1.0; J[0][1] = -x[0] * x[2] / model->population; J[0][2] = 0.0;
  J[0][3] = -x[0]; J[0][4] = 0.0; J[0][5] = x[2];

  J[1][0] = 0.0; J[1][1] = x[0] * x[2] / model->population; J[1][2] = -x[1];
  J[1][3] = -x[1]; J[1][4] = 0.0; J[1][5] = 0.0;

  J[2][0] = 0.0; J[2][1] = 0.0; J[2][2] = x[1];
  J[2][3] = -x[2]; J[2][4] = -x[2]; J[2][5] = -x[2];
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = MatSetValues(JacP, 3, rows, 6, cols, &J[0][0], INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JacP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}


PetscErrorCode SEIRRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  MixedModel        model = (MixedModel)ctx;
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x;
  PetscScalar       B=model->params[0], beta=model->params[1],
                    eps=model->params[2], mu_b=model->params[3],
                    mu_i=model->params[4], gamma=model->params[5];
  PetscReal         popchange;
  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);
  model->population = _max(x[0],0.0) + _max(x[1],0.0) + _max(x[2],0.0) + _max(x[3], 0.0);
  popchange = model->prev_population - model->population;
  if(popchange <= 0){
    model->dead += PetscAbsReal(popchange);
  }
  f[0] = B - mu_b * x[0] - beta * x[2] * x[0] / model->population;
  f[1] = beta * x[2] * x[0] / model->population - (mu_b + eps) * x[1];
  f[2] = eps * x[1] - (gamma + mu_i + mu_b) * x[2];
  f[3] = eps * x[2] - mu_b * x[3];
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  model->prev_population = model->population;
  PetscFunctionReturn(ierr);
}


PetscErrorCode SEIRRHSJacobian(TS ts, PetscReal t, Vec X, Mat Jac, Mat Pre, void *ctx)
{
  PetscErrorCode    ierr;
  MixedModel        model=(MixedModel)ctx;
  PetscScalar       B=model->params[0], beta=model->params[1],
                    eps=model->params[2], mu_b=model->params[3],
                    mu_i=model->params[4], gamma=model->params[5];
  PetscInt          rows_and_cols[] = {0,1,2,3};
  PetscScalar       J[4][4];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  model->population = _max(x[0],0.0) + _max(x[1],0.0) + _max(x[2],0.0) + _max(x[3], 0.0);
  J[0][0] = -mu_b - beta * x[2] / model->population;
  J[0][1] = 0.0;
  J[0][2] = -beta * x[0] / model->population;
  J[0][3] = 0.0;

  J[1][0] = beta * x[2] / model->population;
  J[1][1] = -(mu_b + eps);
  J[1][2] = beta * x[0] / model->population;
  J[1][3] = 0.0;

  J[2][0] = 0.0;
  J[2][1] = eps;
  J[2][2] = -(gamma + mu_i);
  J[2][3] = 0.0;

  J[3][0] = 0.0;
  J[3][1] = 0.0;
  J[3][2] = eps;
  J[3][3] = -mu_b;

  ierr = MatSetValues(Jac, 4, rows_and_cols, 4, rows_and_cols, &J[0][0], INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if(Pre != Jac){
    ierr = MatAssemblyBegin(Pre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Pre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SEIRRHSJacobianP(TS ts, PetscReal t, Vec X, Mat JacP, void *ctx)
{
  PetscErrorCode    ierr;
  MixedModel        model=(MixedModel)ctx;
  PetscInt          rows[] = {0,1,2,3}, cols[]={0,1,2,3,4,5};
  PetscScalar       J[4][6];
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  model->population = _max(x[0],0.0) + _max(x[1],0.0) + _max(x[2],0.0) + _max(x[3], 0.0);
  J[0][0] = 1.0; J[0][1] = x[2] * x[0] / model->population;
  J[0][2] = 0.0; J[0][3] = -x[0]; J[0][4] = 0.0;
  J[0][5] = 0.0;

  J[1][0] = 0.0; J[1][1] = x[0] * x[2] / model->population;
  J[1][2] = -x[1]; J[1][3] = -x[1]; J[1][4] = 0.0;
  J[1][5] = 0.0;

  J[2][0] = 0.0; J[2][1] = 0.0; J[2][2] = x[1];
  J[2][3] = -x[2]; J[2][4] = -x[2]; J[2][5] = -x[2];

  J[3][0] = 0.0; J[3][1] = 0.0; J[3][2] = 0.0;
  J[3][3] = -x[3]; J[3][4] = 0.0; J[3][5] = x[2];

  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = MatSetValues(JacP, 4, rows, 6, cols, &J[0][0], INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(JacP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}
  

  
PetscErrorCode MixedModelCreate(MixedModel *model)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscNew(model);CHKERRQ(ierr);
  /*ierr = TSCreate(PETSC_COMM_WORLD, &((*model)->ts));CHKERRQ(ierr);*/
  (*model)->rhs = NULL;
  (*model)->rhs_jac = NULL;
  (*model)->custom_cost_gradient = PETSC_FALSE;
  PetscFunctionReturn(ierr);
}

static PetscErrorCode destroy_adjoint_vecs(MixedModel model)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscFunctionBeginUser;
  for(i=0; i<model->states; ++i){
    if(model->lambda){
      ierr = VecDestroy(&model->lambda[i]);
    }
    if(model->mu){
      ierr = VecDestroy(&model->mu[i]);
    }
  }
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelDestroy(MixedModel model)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(model){
    if(model->params){
      ierr = PetscFree(model->params);CHKERRQ(ierr);
    }
    if(model->Jac){
      ierr = MatDestroy(&model->Jac);
    }
    if(model->JacP){
      ierr = MatDestroy(&model->JacP);
    }
    if(model->lambda){
      ierr = destroy_adjoint_vecs(model);
    }
    if(model->ts){
      ierr = TSDestroy(&model->ts);CHKERRQ(ierr);
    }
    /*if(model->X){
      ierr = VecDestroy(&model->X);CHKERRQ(ierr);
    }*/
     if(model->F){
      ierr = VecDestroy(&model->F);CHKERRQ(ierr);
     }
    
    model->rhs = NULL;
    model->rhs_jac = NULL;
    ierr = PetscFree(model);CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelReproductionNumber(MixedModel model, PetscReal *R0)
{
  PetscErrorCode ierr;
  ierr = model->basic_r0_func(model->params, R0);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelSetType(MixedModel model, MixedModelType type)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  model->type = type;
  ierr = PetscMalloc1(_MMNumParams[type], &model->params);CHKERRQ(ierr);
  model->states = _MMNumStates[type];
  ierr = PetscMalloc1(model->states, &model->lambda);CHKERRQ(ierr);
  ierr = PetscMalloc1(model->states, &model->mu);CHKERRQ(ierr);
  
  model->rhs = _MMRHS[type];
  model->rhs_jac = _MMRHS_JAC[type];
  model->rhs_jacp = _MMRHS_JP[type];
  model->basic_r0_func = _MMR0[type];
  ierr = TSCreate(PETSC_COMM_WORLD, &(model->ts));CHKERRQ(ierr);
  
  ierr = MatCreate(PETSC_COMM_WORLD, &model->Jac);
  ierr = MatSetSizes(model->Jac, PETSC_DECIDE, PETSC_DECIDE, model->states, model->states);CHKERRQ(ierr);
  ierr = MatSetFromOptions(model->Jac);CHKERRQ(ierr);
  ierr = MatSetUp(model->Jac);CHKERRQ(ierr);
  
  ierr = MatCreate(PETSC_COMM_WORLD, &model->JacP);
  ierr = MatSetSizes(model->JacP, PETSC_DECIDE, PETSC_DECIDE, model->states, _MMNumParams[type]);CHKERRQ(ierr);
  ierr = MatSetFromOptions(model->JacP);CHKERRQ(ierr);
  ierr = MatSetUp(model->JacP);CHKERRQ(ierr);
  
  ierr = MatCreateVecs(model->Jac, &model->X, NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(model->Jac, &model->F, NULL);CHKERRQ(ierr);
  
  ierr = TSSetRHSFunction(model->ts, model->F, model->rhs, model);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(model->ts, model->Jac, model->Jac, model->rhs_jac, model);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(model->ts, model->JacP, model->rhs_jacp, model);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelGetTotalDeaths(MixedModel model, PetscReal *TotalDeaths)
{
  PetscFunctionBeginUser;
  *TotalDeaths = model->dead;
  PetscFunctionReturn(0);
}

PetscErrorCode MixedModelSetTimeStep(MixedModel model, PetscReal dt)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = TSSetTimeStep(model->ts, dt);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelSetMaxTime(MixedModel model, PetscReal tmax)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = TSSetMaxTime(model->ts, tmax);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(model->ts,TS_EXACTFINALTIME_INTERPOLATE);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelSetTSType(MixedModel model, TSType type)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = TSSetType(model->ts, type);CHKERRQ(ierr);
  if(type == TSRK){
    ierr = TSRKSetType(model->ts, TSRK4);CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelSetFromOptions(MixedModel model)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = TSSetFromOptions(model->ts);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode SEISCreate(MixedModel *seis)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = MixedModelCreate(seis);CHKERRQ(ierr);
  ierr = MixedModelSetType(*seis, SEIS);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode SEIRCreate(MixedModel *seir)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = MixedModelCreate(seir);CHKERRQ(ierr);
  ierr = MixedModelSetType(*seir, SEIR);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelSetParams(MixedModel model, PetscScalar *params)
{
  PetscInt i;
  PetscFunctionBeginUser;
  for(i=0; i<_MMNumParams[model->type]; ++i){
    model->params[i] = params[i];
  }
  PetscFunctionReturn(0);
}
      
PetscErrorCode MixedModelSolve(MixedModel model, Vec X0)
{
  PetscErrorCode ierr;
  PetscScalar pop;
  PetscFunctionBeginUser;
  ierr = VecCopy(X0, model->X);CHKERRQ(ierr);
  ierr = VecSum(model->X, &pop);CHKERRQ(ierr);
  model->population = pop;
  model->prev_population = pop;
  model->dead = 0.0;
  ierr = TSSetSolution(model->ts, model->X);CHKERRQ(ierr);
  ierr = TSSetSaveTrajectory(model->ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(model->ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetUp(model->ts);CHKERRQ(ierr);
  ierr = TSSolve(model->ts, model->X);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

static PetscErrorCode allocate_identity_adjoint_vecs(MixedModel model)
{
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscInt       i, j;
  PetscFunctionBeginUser;
  for(i=0; i<model->states; ++i){
    ierr = MatCreateVecs(model->Jac, &model->lambda[i], NULL);CHKERRQ(ierr);
    ierr = VecGetArray(model->lambda[i], &x);CHKERRQ(ierr);
    for(j=0; j<model->states; ++j){
      if(j==i){
	x[j] = 1.0;
      } else {
	x[j] = 0.0;
      }
    }
    ierr = VecRestoreArray(model->lambda[i], &x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

static PetscErrorCode allocate_zero_param_adjoint_vecs(MixedModel model)
{
  PetscErrorCode ierr;
  PetscScalar    *x;
  PetscInt       i, j;
  PetscFunctionBeginUser;
  for(i=0; i<model->states; ++i){
    ierr = MatCreateVecs(model->JacP, &model->mu[i], NULL);CHKERRQ(ierr);
    ierr = VecGetArray(model->mu[i], &x);CHKERRQ(ierr);
    for(j=0; j<_MMNumParams[model->type]; ++j){
      x[j] = 0.0;
    }
    ierr = VecRestoreArray(model->mu[i], &x);CHKERRQ(ierr);
  }
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelSetCostGradients(MixedModel model, Vec *lambda, Vec *mu)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = TSSetCostGradients(model->ts, model->states, lambda, mu);CHKERRQ(ierr);
  model->custom_cost_gradient = PETSC_TRUE;
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelAdjointSolve(MixedModel model)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if(!(model->custom_cost_gradient)){
    ierr = allocate_identity_adjoint_vecs(model);CHKERRQ(ierr);
    ierr = allocate_zero_param_adjoint_vecs(model);CHKERRQ(ierr);
    ierr = TSSetCostGradients(model->ts, model->states, model->lambda, model->mu);CHKERRQ(ierr);
  }
  ierr = TSAdjointSolve(model->ts);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}
