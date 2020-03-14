#include "mixed.h"

int main(int argc, char **argv)
{
  MixedModel  seir;
  Vec         X;
  PetscScalar *x0, params[6];
  PetscBool   flg;
  PetscInt    i, nsteps, nparam=6;
  PetscReal   R0, dead, dt=0.01, tf=1000.0;
  PetscErrorCode ierr;
  TSConvergedReason reason;
  TS ts;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);CHKERRQ(ierr);
  ierr = SEIRCreate(&seir);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL, NULL, "-tf", &tf, &flg);CHKERRQ(ierr);
  if(!flg){
    tf = 10000.0;
  }

  ierr = PetscOptionsGetReal(NULL, NULL, "-dt", &dt, &flg);CHKERRQ(ierr);
  if(!flg){
    dt = 0.01;
  }
  ierr = PetscOptionsGetRealArray(NULL, NULL, "-params", params, &nparam, &flg);CHKERRQ(ierr);
  if(flg){
    if(nparam != 6){
      SETERRQ(PETSC_COMM_WORLD, 1, "MUST SUPPLY 6 PARAMS OR NOT PASS THE OPTION!");
    }
  } else {
    params[0] = 0.1;
    params[1] = 3.5;
    params[2] = 0.5;
    params[3] = 0.0001;
    params[4] = 0.03;
    params[5] = 1.3;
  }
  ierr = MixedModelSetParams(seir, params);CHKERRQ(ierr);

  ierr = MatCreateVecs(seir->Jac, &X, NULL);CHKERRQ(ierr);
  ierr = VecGetArray(X, &x0);CHKERRQ(ierr);
  x0[0] = 900.0;
  x0[1] = 10.0;
  x0[2] = 2.0;
  x0[3] = 0.0;
  x0[4] = 0.0;
  x0[5] = 0.0;
  ierr = VecRestoreArray(X, &x0);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Initial disease state (S, E, I, R, Dead (non-infection), Dead (infection)):\n");CHKERRQ(ierr);
  ierr = MixedModelReproductionNumber(seir, &R0);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Basic reproduction number (R0): %.4f.\n", R0);
  ierr = VecView(X, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MixedModelSetMaxTime(seir, tf);CHKERRQ(ierr);
  ierr = MixedModelSetTimeStep(seir, dt);CHKERRQ(ierr);
  ierr = MixedModelSetTSType(seir, TSRK);CHKERRQ(ierr);
  ierr = MixedModelSetFromOptions(seir);CHKERRQ(ierr);
  ierr = MixedModelSolve(seir, X);CHKERRQ(ierr);
  TSGetConvergedReason(seir->ts,&reason);
  TSGetSolveTime(seir->ts,&tf);
  TSGetStepNumber(seir->ts,&nsteps);
  PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)tf,nsteps);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Final disease state (S, E, I, R, Dead (non-infection), Dead (infection)):\n");CHKERRQ(ierr);
  ierr = VecView(seir->X, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MixedModelGetTotalDeaths(seir, &dead);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Total dead: %.4f.\n", dead);
  ierr = MixedModelAdjointSolve(seir);CHKERRQ(ierr);

  /*ierr = TSGetSolution(seir->ts, &X);CHKERRQ(ierr);*/
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Adjoint variables w.r.t. state:\n");CHKERRQ(ierr);
  for(i = 0; i < 6; ++i){
    ierr = VecView(seir->lambda[i], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = PetscPrintf(PETSC_COMM_WORLD, "Adjoint variables w.r.t. parameters:\n");CHKERRQ(ierr);
  for(i = 0; i < 6; ++i){
    ierr = VecView(seir->mu[i], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /*ierr = VecDestroy(&X);CHKERRQ(ierr);*/
  ierr = MixedModelDestroy(seir);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
