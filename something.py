import cython
from cython.cimports.mujoco import mj_loadXML, mjModel, mjData, mj_makeData, mj_deleteModel, \
    mj_deleteData, mj_resetData, mj_stateSize, mj_getState, mj_setState, mjSTATE_INTEGRATION, mjtNum
from cython.cimports.libc.stdlib import calloc, free
from cython.parallel import prange
from cython.cimports.gsl import gsl_rng_type, gsl_rng_default, gsl_rng, gsl_rng_alloc, gsl_rng_set, gsl_rng_uniform, gsl_rng_free, gsl_ran_gaussian
from cython.cimports.gsl import cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasTrans, cblas_daxpy
from cython.cimports.libc.math import pow, sqrt
import time
import numpy as np


@cython.cfunc
def cholesky_decomp(A: cython.pointer(cython.double), m: cython.int) -> cython.pointer(cython.double):
    L: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(m*m, cython.sizeof(cython.double)))

    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    k: cython.Py_ssize_t
    # into Lower Triangular
    for i in range(m):
        for j in range(i + 1):
            sum1: cython.double = 0.0
            # summation for diagonals
            if j == i:
                for k in range(j):
                    sum1 += pow(L[j*m+k], 2)
                L[j*m+j] = sqrt(A[j*m+j] - sum1)
            else:

                # Evaluating L(i, j)
                # using L(j, j)
                for k in range(j):
                    sum1 += (L[i*m+k] * L[j*m+k])
                if (L[j*m+j] > 0):
                    L[i*m+j] = (A[i*m+j] - sum1) /L[j*m+j]
    return L


@cython.cfunc
def sample_multivariate_gaussian(num_samples: cython.int, mu: cython.pointer(cython.double), cov: cython.pointer(cython.double), data_dim: cython.int, rng: cython.pointer(gsl_rng)) -> cython.pointer(cython.double):
    samples: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(num_samples * data_dim, cython.sizeof(cython.double)))

    i: cython.Py_ssize_t
    for i in range(num_samples * data_dim):
        samples[i] = gsl_ran_gaussian(rng, 1.0) # Mu: 0, Sigma: 1

    L_cov: cython.pointer(cython.double) = cholesky_decomp(cov, data_dim)
    output: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(num_samples * data_dim, cython.sizeof(cython.double)))
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_samples, data_dim, data_dim, 1.0, samples, data_dim, L_cov, data_dim, 0.0, output, data_dim)
    for i in range(num_samples):
        cblas_daxpy(data_dim, 1.0, mu, 1, cython.address(output[i*data_dim]), 1)
    return output


@cython.cfunc
def matrix_multiply( A: cython.pointer(cython.double),
                    B: cython.pointer(cython.double),
                    C: cython.pointer(cython.double), M: cython.int, N: cython.int, K: cython.int, alpha: cython.double = 1.0, beta: cython.double = 0.0, ):
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,  alpha, A, K, B, N,  beta, C, N)

@cython.cfunc
def generate_random_numbers(seed: cython.int, n: cython.int):
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, seed)

    random_numbers: list = []
    i: cython.Py_ssize_t
    for i in range(n):
        random_numbers.append(gsl_rng_uniform(rng))

    gsl_rng_free(rng)
    return random_numbers


# @cython.cfunc
# @cython.nogil
# @cython.exceptval(check=False)
# def rollout(n:cython.Py_ssize_t, n_steps: cython.int, model: cython.pointer(mjModel), data: cython.pointer(mjData)) -> cython.void:
#     for j in range(n_steps):
#         mj_step(model, data)

        # with cython.gil:
        #     state = np.zeros(shape=model.nq-2+model.nv)
        #     i: cython.Py_ssize_t
        #     for i in range(model.nq-2):
        #         state[i] = data.qpos[i+2]
        #     for i in range(model.nv):
        #         state[i+model.nq-2] = data.qvel[i]
        #
        #     print(n, state, state.shape)
        # with cython.gil:
        #     action = np.zeros(shape=model.nu)
        #     for i in range(model.nu):
        #         action[i] = data.ctrl[i]
        #     action = np.random.uniform(low=-1.0, high=1.0, size=model.nu)
        #     for i in range(model.nu):
        #         data.ctrl[i] = action[i]
        #
        #     print(n, action, action.shape)



@cython.cfunc
def some2(path: str):
    err: cython.char[300]
    model: cython.pointer(mjModel) = mj_loadXML(path.encode("UTF-8"), cython.NULL, err, 300)
    data: cython.pointer(mjData) = mj_makeData(model)
    mj_resetData(model, data)

    size: cython.int = mj_stateSize(model, mjSTATE_INTEGRATION)
    state: cython.pointer(mjtNum) = cython.cast(cython.pointer(mjtNum), calloc(size, cython.sizeof(mjtNum)))
    mj_getState(model, data, state, mjSTATE_INTEGRATION)

    print(f"State: {size}")
    i: cython.Py_ssize_t
    for i in range(size):
        print(f"{state[i]}", end=", ")
    print("")

    free(state)
    mj_deleteData(data)
    mj_deleteModel(model)


def some():
    some2("./env_xmls/ant.xml")
    # A: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(4, cython.sizeof(cython.double)))
    # B: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(4, cython.sizeof(cython.double)))
    # C: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(4, cython.sizeof(cython.double)))
    # A[0] = 1.0
    # A[1] = 2.0
    # A[2] = 3.0
    # A[3] = 4.0
    # B[0] = 5.0
    # B[1] = 6.0
    # B[2] = 7.0
    # B[3] = 8.0
    #
    # matrix_multiply(A, B, C, 2, 2, 2)
    # print(A[0], A[1], A[2], A[3])
    # print(B[0], B[1], B[2], B[3])
    # print(C[0], C[1], C[2], C[3])
    # free(A)
    # free(B)
    # free(C)
    # random_numbers = generate_random_numbers(1234, 5)
    # print(random_numbers)  # Should print a list of 5 random numbers

    # A: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(3*3, cython.sizeof(cython.double)))
    #
    # A[0*3+0] = 4
    # A[0*3+1] = 12
    # A[0*3+2] = -16
    # A[1*3+0] = 12
    # A[1*3+1] = 37
    # A[1*3+2] = -43
    # A[2*3+0] = -16
    # A[2*3+1] = -43
    # A[2*3+2] = 98
    #
    # L = cholesky_decomp(A, 3)
    # print(L[0*3+0],L[0*3+1],L[0*3+2],L[1*3+0],L[1*3+1],L[1*3+2],L[2*3+0],L[2*3+1],L[2*3+2])
    # free(A)
    # free(L)

    # mu: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(2, cython.sizeof(cython.double)))
    # mu[0] = 1.0
    # mu[1] = 10.0
    #
    # cov: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(2*2, cython.sizeof(cython.double)))
    # cov[0*2+0] = 0.9
    # cov[0*2+1] = 0.1
    # cov[1*2+0] = -0.1
    # cov[1*2+1] = 0.9
    #
    # T: cython.pointer(gsl_rng_type) = gsl_rng_default
    # rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    # gsl_rng_set(rng, 6)
    #
    # samples: cython.pointer(cython.double) = sample_multivariate_gaussian(100, mu, cov, 2, rng)
    #
    # i: cython.Py_ssize_t
    # j: cython.Py_ssize_t
    # for i in range(100):
    #     for j in range(2):
    #         print(samples[i*2+j], end=", ")
    #     print("")
    #
    # free(samples)
    # gsl_rng_free(rng)
    # free(cov)
    # free(mu)
