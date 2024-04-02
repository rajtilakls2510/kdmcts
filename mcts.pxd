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
    double* alt_pi  # (num_kernels, )
    double* alt_w   # (num_kernels, )
    int* alt_n  # (num_kernels, )
    double* alt_mu  # (num_kernels, action_dim)
    double* alt_cov # (num_kernels, action_dim, action_dim)
    double mean_w
    double std_w
    int iterations_left
    int replace_every
    int max_iterations
    double init_cov
    double kernel_cov
    bint expanded
    MKDNode** children   # (num_kernels, )

cdef struct MKDRolloutReturn:
    double rtn
    double backup_rtn
    double* action


cdef struct VGNode:
    mjtNum* mj_state
    int current_step
    VGNode* parent
    double parent_reward
    double* parent_action
    double* init_parent_action
    double parent_q_value
    omp_lock_t access_lock
    bint terminal
    double child_add_alpha
    int num_visitations
    int action_dim
    int num_children
    VGNode* children    # List: (num_children, )
    VGNode* next

cdef struct VGSAReturn:
    VGNode* next_node
    bint rollout

cdef struct VGSimReturn:
    int depth
    double rtn
    double** actions
    int last_action

cdef struct VGRolloutReturn:
    double** actions
    int last_action
    double rtn