#ifndef MIXED_MODEL_OPT_H
#define MIXED_MODEL_OPT_H
#include <petscsys.h>

struct _epidemic_obs {
  PetscReal   t;
  PetscInt    num_states;
  PetscInt    *states;
  PetscScalar *values;
};

typedef struct _epidemic_obs *EpidemicObs;

extern PetscErrorCode EpidemicObsCreate(EpidemicObs *obs);

extern PetscErrorCode EpidemicObsDestroy(EpidemicObs obs);

extern PetscErrorCode EpidemicObsSetTime(EpidemicObs obs, PetscReal t);

extern PetscErrorCode EpidemicObsSetValues(EpidemicObs obs, PetscInt num_states, PetscInt *states, PetscScalar *values);




#endif
