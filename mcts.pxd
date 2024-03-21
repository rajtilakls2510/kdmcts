#from cython.cimports.
from cython.cimports.openmp import omp_lock_t

cdef struct MKDNode:
    double* state
    MKDNode* parent
    omp_lock_t access_lock
    bint terminal
    int num_kernels
    int action_dim
    double* pi  # (num_kernels, )
    double* w   # (num_kernels, )
    double* n   # (num_kernels, )
    double* mu  # (num_kernels, action_dim)
    double* cov # (num_kernels, action_dim, action_dim)
    int iterations
    int samples_per_iteration
    bint expanded
    MKDNode** children   # (num_kernels, )

