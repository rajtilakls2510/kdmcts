from cython.cimports.mujoco import mjData, mjtNum
from cython.cimports.openmp import omp_lock_t

cdef struct MKDNode:
    mjtNum* mj_state
    int current_step
    MKDNode* parent
    double parent_reward
    omp_lock_t access_lock
    bint terminal
    int num_kernels
    int action_dim
    bint params_initialized
    double* pi  # (num_kernels, )
    double* w   # (num_kernels, )
    int* n   # (num_kernels, )
    double* mu  # (num_kernels, action_dim)
    double* cov # (num_kernels, action_dim, action_dim)
    int iterations_left
    int max_iterations
    double init_cov
    double kernel_cov
    bint expanded
    MKDNode** children   # (num_kernels, )

