from cython.cimports.mujoco import mjModel, mjData
from cython.cimports.gsl import gsl_rng

cdef struct MujocoEnv:
    int env_id
    mjModel* model
    mjData* data
    int state_size
    int action_size

cdef MujocoEnv create_env(int env_id, char* path) noexcept nogil
cdef void free_env(MujocoEnv env) noexcept nogil
cdef int get_state_size(MujocoEnv env) noexcept nogil
cdef int get_action_size(MujocoEnv env) noexcept nogil
cdef double* get_state(MujocoEnv env) noexcept nogil
#cdef void set_state(MujocoEnv env, double* state) noexcept nogil
cdef double* get_action(MujocoEnv env) noexcept nogil
cdef void set_action(MujocoEnv env, double* action) noexcept nogil
cdef void reset_env(MujocoEnv env, gsl_rng* rng) noexcept nogil
cdef double step(MujocoEnv env, double* action, int num_steps) noexcept nogil
cdef bint is_terminated(MujocoEnv env) noexcept nogil

cdef struct PolicyParams:
    int k
    int n
    double* w
    double* b

#cdef double* policy(PolicyParams params, double* state, MujocoEnv env) noexcept nogil