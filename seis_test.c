#include "mixed.h"

int main(int argc, char **argv)
{
  MixedModel  seis;
  Vec         X;
  PetscScalar *x0, params[6];
  PetscBool   flg;
  PetscInt    i, nsteps, nparam=6;
  PetscReal   dt=0.01, tf=1000.0;
  PetscErrorCode ierr;
  TSConvergedReason reason;
  TS ts;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  ierr = SEISCreate(&seis);CHKERRQ(ierr);
  ierr = PetscOptionsGetRealArray(NULL, NULL, "-params", params, &nparam, &flg);CHKERRQ(ierr);
  if(flg){
    if(nparam != 6){
      SETERRQ(PETSC_COMM_WORLD, 1, "MUST SUPPLY 6 PARAMS OR NOT PASS THE OPTIONS!");
    }
  } else {
    params[0] = 0.01;
    params[1] = 0.04;
    params[2] = 0.1;
    params[3] = 0.0001;
    params[4] = 0.02;
    params[5] = 0.04;
  }
  ierr = MixedModelSetParams(seis, params);CHKERRQ(ierr);

  ierr = MatCreateVecs(seis->Jac, &X, NULL);CHKERRQ(ierr);
  ierr = VecGetArray(X, &x0);CHKERRQ(ierr);
  x0[0] = 10.0;
  x0[1] = 1.0;
  x0[2] = 1.0;
  ierr = VecRestoreArray(X, &x0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial disease state (S, E, I):\n");CHKERRQ(ierr);
  ierr = VecView(X, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MixedModelSetMaxTime(seis, tf);CHKERRQ(ierr);
  ierr = MixedModelSetTimeStep(seis, dt);CHKERRQ(ierr);
  ierr = MixedModelSetTSType(seis, TSRK);CHKERRQ(ierr);
  ierr = MixedModelSetFromOptions(seis);CHKERRQ(ierr);
  ierr = MixedModelSolve(seis, X);CHKERRQ(ierr);
  TSGetConvergedReason(seis->ts,&reason);
  TSGetSolveTime(seis->ts,&tf);
  TSGetStepNumber(seis->ts,&nsteps);
  PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)tf,nsteps);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Final disease state (S, E, I):\n");CHKERRQ(ierr);
  ierr = VecView(seis->X, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MixedModelAdjointSolve(seis);CHKERRQ(ierr);

  /*ierr = TSGetSolution(seis->ts, &X);CHKERRQ(ierr);*/
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Adjoint variables:\n");CHKERRQ(ierr);
  for(i = 0; i < 3; ++i){
    ierr = VecView(seis->lambda[i], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = MixedModelDestroy(seis);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
