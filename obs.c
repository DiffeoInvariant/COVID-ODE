#include "obs.h"

PetscErrorCode EpidemicObsCreate(EpidemicObs *obs)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = PetscNew(obs);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode EpidemicObsDestroy(EpidemicObs obs)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(obs->states){
    ierr = PetscFree(obs->states);CHKERRQ(ierr);
  }
  if(obs->values){
    ierr = PetscFree(obs->values);CHKERRQ(ierr);
  }
  ierr = PetscFree(obs);CHKERRQ(ierr);
  PetscFunctionReturn(ierr);
}

PetscErrorCode EpidemicObsSetTime(EpidemicObs obs, PetscReal t)
{
  PetscFunctionBeginUser;
  obs->t = t;
  PetscFunctionReturn(0);
}

PetscErrorCode EpidemicObsSetValues(EpidemicObs obs, PetscInt num_states, PetscInt *states, PetscScalar *values)
{
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  if(num_states <= 0){
    SETERRQ1(PETSC_COMM_WORLD, 1, "Error, must supply a positive number of states, not %d.\n", num_states);
  }
  obs->num_states = num_states;
  ierr = PetscCalloc2(num_states, &obs->states,
                      num_states, &obs->values);CHKERRQ(ierr);
  for(i=0; i<num_states; ++i){
    obs->states[i] = states[i];
    obs->values[i] = values[i];
  }
  PetscFunctionReturn(ierr);
}


    
  
  
