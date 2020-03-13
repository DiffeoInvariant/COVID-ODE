static char help[] = "Fully-mixed ODE epidemiology models. Currently only SEIS.\n\n";
#include "mixed.h"


const PetscInt _MMNumParams[] = {6, 7};
const PetscInt _MMNumStates[] = {3, 4};

PetscErrorCode SEISRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx);
PetscErrorCode SEISRHSJacobian(TS ts, PetscReal t, Vec X, Mat Jac, Mat Pre, void *ctx);

const TSRHSFunction _MMRHS[] = {SEISRHSFunction};
const TSRHSJacobian _MMRHS_JAC[] = {SEISRHSJacobian};


  

PetscErrorCode SEISRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  MixedModel        model = (MixedModel)ctx;
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x;
  PetscScalar       B=model->params[0], beta=model->params[1],
                    eps=model->params[2], mu_b=model->params[3],
                    mu_i=model->params[4], gamma=model->params[5];
  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);
  f[0] = B - beta * x[0] * x[2] - mu_b * x[0] + gamma * x[2];
  f[1] = beta * x[0] * x[2] - (eps + mu_b)*x[1];
  f[2] = eps * x[1] - (gamma + mu_i)*x[2];
  /*PetscPrintf(PETSC_COMM_WORLD, "x[0] = %f, x[1] = %f, x[2] = %f.\n", x[0], x[1], x[2]);
    PetscPrintf(PETSC_COMM_WORLD, "f[0] = %f, f[1] = %f, f[2] = %f.\n\n", f[0], f[1], f[2]);*/
  
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
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
  J[0][0] = -beta * x[2] - mu_b;
  J[0][1] = 0.0;
  J[0][2] = -beta * x[0] + gamma;

  J[1][0] = beta * x[2];
  J[1][1] = -(eps + mu_b);
  J[1][2] = beta * x[0];

  J[2][0] = 0.0;
  J[2][1] = eps;
  J[2][2] = -(gamma + mu_i);

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
    if(model->lambda[i]){
      ierr = VecDestroy(&model->lambda[i]);
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

PetscErrorCode MixedModelSetType(MixedModel model, MixedModelType type)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  model->type = type;
  ierr = PetscMalloc1(_MMNumParams[type], &model->params);CHKERRQ(ierr);
  model->states = _MMNumStates[type];
  ierr = PetscMalloc1(model->states, &model->lambda);CHKERRQ(ierr);
  model->rhs = _MMRHS[type];
  model->rhs_jac = _MMRHS_JAC[type];
  ierr = TSCreate(PETSC_COMM_WORLD, &(model->ts));CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &model->Jac);
  ierr = MatSetSizes(model->Jac, PETSC_DECIDE, PETSC_DECIDE, model->states, model->states);CHKERRQ(ierr);
  ierr = MatSetFromOptions(model->Jac);CHKERRQ(ierr);
  ierr = MatSetUp(model->Jac);CHKERRQ(ierr);
  ierr = MatCreateVecs(model->Jac, &model->X, NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(model->Jac, &model->F, NULL);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(model->ts, model->F, model->rhs, model);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(model->ts, model->Jac, model->Jac, model->rhs_jac, model);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
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
  PetscFunctionBeginUser;
  ierr = VecCopy(X0, model->X);CHKERRQ(ierr);
  ierr = TSSetSolution(model->ts, model->X);CHKERRQ(ierr);
  ierr = TSSetSaveTrajectory(model->ts);CHKERRQ(ierr);
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

PetscErrorCode MixedModelSetCostGradients(MixedModel model, Vec *lambda)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = TSSetCostGradients(model->ts, model->states, lambda, NULL);CHKERRQ(ierr);
  model->custom_cost_gradient = PETSC_TRUE;
  PetscFunctionReturn(ierr);
}

PetscErrorCode MixedModelAdjointSolve(MixedModel model)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if(!(model->custom_cost_gradient)){
    ierr = allocate_identity_adjoint_vecs(model);CHKERRQ(ierr);
    ierr = TSSetCostGradients(model->ts, model->states, model->lambda, NULL);CHKERRQ(ierr);
  }
  ierr = TSAdjointSolve(model->ts);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}
