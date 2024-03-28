from cython.cimports.mujoco import mjModel, mjData, mjtNum
from cython.cimports.gsl import gsl_rng

cdef struct MujocoEnv:
    int env_id
    mjModel* model
    mjData* data
    int mj_state_size
    int state_size
    int action_size
    int num_steps   # Number of raw env steps to take with a particular action set
    int max_steps   # Max Transitions

cdef MujocoEnv create_env(int env_id, char* path, int num_steps, int max_steps) noexcept nogil
cdef void free_env(MujocoEnv env) noexcept nogil
cdef int get_mj_state_size(MujocoEnv env) noexcept nogil
cdef int get_state_size(MujocoEnv env) noexcept nogil
cdef int get_action_size(MujocoEnv env) noexcept nogil
cdef mjtNum* get_mj_state(MujocoEnv env) noexcept nogil
cdef void set_mj_state(MujocoEnv env, mjtNum* mj_state) noexcept nogil
cdef double* get_state(MujocoEnv env) noexcept nogil
#cdef void set_state(MujocoEnv env, double* state) noexcept nogil
cdef double* get_action(MujocoEnv env) noexcept nogil
cdef void set_action(MujocoEnv env, double* action) noexcept nogil
cdef void reset_env(MujocoEnv env, gsl_rng* rng) noexcept nogil
cdef double step(MujocoEnv env, double* action) noexcept nogil
cdef bint is_terminated(MujocoEnv env, int steps_taken) noexcept nogil

cdef struct PolicyParams:
    int k
    int n
    double* w
    double* b

cdef double* policy(PolicyParams params, double* state) noexcept nogil

cdef class MujocoPyEnv:
    cdef MujocoEnv env_struct
    cdef gsl_rng* rng