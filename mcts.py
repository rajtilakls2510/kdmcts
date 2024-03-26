# cython: cdivision = True

import cython
from cython.parallel import prange
from cython.cimports.mujoco_envs import MujocoEnv, PolicyParams, step, get_state, set_state, \
    is_terminated, policy, set_action, create_env, reset_env, free_env, get_mj_state, \
    set_mj_state
from cython.cimports.libc.stdlib import calloc, free
from cython.cimports.libc.math import pow, sqrt, exp, isnan, log
from cython.cimports.openmp import omp_lock_t, omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock
from cython.cimports.gsl import gsl_rng_uniform, gsl_rng_type, gsl_rng_default, \
    gsl_rng_alloc, gsl_rng_set, gsl_rng, gsl_rng_free, gsl_ran_flat, gsl_ran_gaussian
from cython.cimports.gsl import cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasTrans, cblas_daxpy, \
    cblas_ddot, cblas_dcopy, cblas_dscal, cblas_dger
from cython.cimports.mujoco import mjData, mj_copyData, mj_deleteData, mj_forward, mjModel, mjtNum
import time


# ================================== Sample from Multivariate Gaussian ==================================

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def cholesky_decomp(A: cython.pointer(cython.double), m: cython.int) -> cython.pointer(cython.double):
    L: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                   calloc(m * m, cython.sizeof(cython.double)))

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
                    sum1 += pow(L[j * m + k], 2)
                L[j * m + j] = sqrt(A[j * m + j] - sum1)
            else:

                # Evaluating L(i, j)
                # using L(j, j)
                for k in range(j):
                    sum1 += (L[i * m + k] * L[j * m + k])
                if (L[j * m + j] > 0):
                    L[i * m + j] = (A[i * m + j] - sum1) / L[j * m + j]
    return L


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def sample_multivariate_gaussian(num_samples: cython.int, mu: cython.pointer(cython.double),
                                 cov: cython.pointer(cython.double), data_dim: cython.int,
                                 rng: cython.pointer(gsl_rng)) -> cython.pointer(cython.double):
    samples: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                         calloc(num_samples * data_dim, cython.sizeof(cython.double)))

    i: cython.Py_ssize_t
    for i in range(num_samples * data_dim):
        samples[i] = gsl_ran_gaussian(rng, 1.0)  # Mu: 0, Sigma: 1

    L_cov: cython.pointer(cython.double) = cholesky_decomp(cov, data_dim)
    output: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                        calloc(num_samples * data_dim, cython.sizeof(cython.double)))
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, num_samples, data_dim, data_dim, 1.0, samples, data_dim, L_cov,
                data_dim, 0.0, output, data_dim)
    for i in range(num_samples):
        cblas_daxpy(data_dim, 1.0, mu, 1, cython.address(output[i * data_dim]), 1)
    free(L_cov)
    return output


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def softmax(a: cython.pointer(cython.double), size: cython.Py_ssize_t) -> cython.void:
    max: cython.double = a[0]
    for i in range(size):
        if a[i] > max:
            max = a[i]
    for i in range(size):
        a[i] -= max
        a[i] = exp(a[i])
    sum: cython.double = 0
    for i in range(size):
        sum += a[i]
    for i in range(size):
        a[i] /= sum


# =================================== MKD MCTS ==========================


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_create_tree_node(env: MujocoEnv, current_step: cython.int,
                         parent: cython.pointer(MKDNode), parent_reward: cython.double, terminal: cython.bint,
                         num_kernels: cython.int,
                         action_dim: cython.int, max_iterations: cython.int, init_cov: cython.double, kernel_cov: cython.double) -> cython.pointer(
    MKDNode):
    node: cython.pointer(MKDNode) = cython.cast(cython.pointer(MKDNode), calloc(1, cython.sizeof(MKDNode)))
    node.mj_state = get_mj_state(env)
    node.current_step = current_step
    node.parent = parent
    node.parent_reward = parent_reward
    omp_init_lock(cython.address(node.access_lock))
    node.terminal = terminal
    node.num_kernels = num_kernels
    node.action_dim = action_dim
    node.params_initialized = False
    node.iterations_left = max_iterations
    node.max_iterations = max_iterations
    node.init_cov = init_cov
    node.kernel_cov = kernel_cov
    if not terminal:
        node.pi = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
        i: cython.Py_ssize_t
        for i in range(num_kernels):
            node.pi[i] = 1.0 / num_kernels
        node.w = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
        node.n = cython.cast(cython.pointer(cython.int), calloc(num_kernels, cython.sizeof(cython.int)))
        node.mu = cython.cast(cython.pointer(cython.double),
                              calloc(num_kernels * action_dim, cython.sizeof(cython.double)))
        node.cov = cython.cast(cython.pointer(cython.double),
                               calloc(num_kernels * action_dim * action_dim, cython.sizeof(cython.double)))

    node.expanded = False
    node.children = cython.NULL
    return node


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_free_tree_node(node: cython.pointer(MKDNode)) -> cython.void:
    if node != cython.NULL:
        free(node.mj_state)
        omp_destroy_lock(cython.address(node.access_lock))
        node.parent = cython.NULL
        if not node.terminal:
            free(node.pi)
            free(node.w)
            free(node.n)
            free(node.mu)
            free(node.cov)
            if node.expanded:
                i: cython.Py_ssize_t
                for i in range(node.num_kernels):
                    mkd_free_tree_node(node.children[i])
                free(node.children)
        free(node)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_selection(node: cython.pointer(MKDNode), move_indices: cython.pointer(cython.int),
                  move_last: cython.pointer(cython.int), env: MujocoEnv,
                  rng: cython.pointer(gsl_rng)) -> cython.pointer(MKDNode):
    while node.expanded and not node.terminal:
        # Use node.pi to select an action

        selected_action_index: cython.Py_ssize_t = 0
        # # Sampling an action from pi
        # rm: cython.double = gsl_rng_uniform(rng)
        # sum: cython.double = 0
        # # omp_set_lock(cython.address(node.access_lock))
        # for selected_action_index in range(env.action_size):
        #     sum += node.pi[selected_action_index]
        #     if rm < sum:
        #         break
        # # omp_unset_lock(cython.address(node.access_lock))

        # Sampling an action from UCT
        scratch: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(node.num_kernels, cython.sizeof(cython.double)))
        i: cython.Py_ssize_t
        total_times: cython.int = 0
        for i in range(node.num_kernels):
            total_times += node.n[i]
        for i in range(node.num_kernels):
            scratch[i] = node.pi[i] + sqrt(total_times / node.n[i])
        max: cython.double = scratch[0]
        for i in range(node.num_kernels):
            if max < scratch[i]:
                selected_action_index = i
                max = scratch[i]
        free(scratch)

        # Note down the index of the action taken and continue selection to child
        move_indices[move_last[0]] = cython.cast(cython.int, selected_action_index)
        move_last[0] += 1
        # with cython.gil:
        #     print("W:")
        #     for i in range(node.num_kernels):
        #         print(f"{node.w[i]}, ", end="")
        #     print("")
        #     print("N:")
        #     for i in range(node.num_kernels):
        #         print(f"{node.n[i]}, ", end="")
        #     print("")
        #     print("P:")
        #     for i in range(node.num_kernels):
        #         print(f"{node.pi[i]}, ", end="")
        #     print("")
        node = node.children[selected_action_index]

    return node


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_expand_node(node: cython.pointer(MKDNode), env: MujocoEnv, rollout_params: PolicyParams,
                    rng: cython.pointer(gsl_rng)) -> cython.void:
    if not node.params_initialized:
        # Allocate pi, w, n, mu, cov and play some rollouts to initialize these parameters if not initialized

        node.params_initialized = True
        # if node is not terminal, initialize allocated parameters
        if not node.terminal:
            i: cython.Py_ssize_t
            j: cython.Py_ssize_t
            for i in range(node.num_kernels):
                for j in range(env.action_size):
                    node.mu[i * env.action_size + j] = gsl_ran_flat(rng, -1.0, 1.0)
                    node.cov[
                        i * env.action_size * env.action_size + j * env.action_size + j] = node.init_cov

            # Perform one rollout per kernel to initialize w and n values
            original_data: cython.pointer(mjData) = env.data
            # original_state: cython.pointer(mjtNum) = get_mj_state(env)
            for i in range(node.num_kernels):
                # Take an action and get next step
                env.data = mj_copyData(cython.NULL, env.model, original_data)

                set_mj_state(env, node.mj_state)    # Resetting the controller to node
                r: cython.double = step(env, cython.address(node.mu[i * env.action_size]))

                # Rollout from this next state
                rtn: cython.double = r + mkd_rollout(node.current_step + 1, env, rollout_params, rng)
                node.w[i] = rtn
                node.n[i] += 1
                mj_deleteData(env.data)
            env.data = original_data
            # set_mj_state(env, original_state)
            # free(original_state)
        else:
            node.expanded = True

    if not node.expanded:
        # Expansion is only performed when the leaf node has been visited a certain number of times
        if node.iterations_left == 0:
            node.children = cython.cast(cython.pointer(cython.pointer(MKDNode)),
                                        calloc(node.num_kernels, cython.sizeof(cython.pointer(MKDNode))))
            original_data: cython.pointer(mjData) = env.data
            # original_state: cython.pointer(mjtNum) = get_mj_state(env)
            i: cython.Py_ssize_t
            for i in range(node.num_kernels):
                env.data = mj_copyData(cython.NULL, env.model, original_data)
                set_mj_state(env, node.mj_state)    # Resetting the controller to node
                r: cython.double = step(env, cython.address(node.mu[cython.cast(cython.int, i) * env.action_size]))

                node.children[i] = mkd_create_tree_node(env, node.current_step + 1, node, r,
                                                        is_terminated(env, node.current_step + 1), node.num_kernels,
                                                        env.action_size, node.max_iterations, node.init_cov, node.kernel_cov)
                mj_deleteData(env.data)
            env.data = original_data  # Don't need to restore the original value because env is not a pointer but it is a good practice
            # set_mj_state(env, original_state)
            # free(original_state)
            node.expanded = True
        else:
            # Else, increase visitation count
            if node.iterations_left > 0:
                node.iterations_left -= 1
            else:
                node.iterations_left = 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_expansion(node: cython.pointer(MKDNode), move_indices: cython.pointer(cython.int),
                  move_last: cython.pointer(cython.int), env: MujocoEnv,
                  rollout_params: PolicyParams, rng: cython.pointer(gsl_rng)) -> cython.pointer(MKDNode):
    omp_set_lock(cython.address(node.access_lock))

    # If the node has not been expanded by another thread, expand it
    if not node.expanded:
        mkd_expand_node(node, env, rollout_params, rng)
        omp_unset_lock(cython.address(node.access_lock))
    else:
        # Else, release the lock immediately and resume from selection phase as another thread has already expanded this node
        omp_unset_lock(cython.address(node.access_lock))
        if not node.terminal:
            node = mkd_selection(node, move_indices, move_last, env, rng)
            node = mkd_expansion(node, move_indices, move_last, env, rollout_params, rng)
    return node


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_rollout(steps_taken: cython.int, env: MujocoEnv,
                rollout_params: PolicyParams, rng: cython.pointer(gsl_rng)) -> cython.double:
    total_reward: cython.double = 0.0
    steps: cython.Py_ssize_t = steps_taken
    state: cython.pointer(cython.double) = get_state(env)
    # Rollout till the end of the episode
    while (steps * env.num_steps < env.max_steps) and (not is_terminated(env, steps)):
        # Select actions according to rollout policy
        action: cython.pointer(cython.double) = policy(rollout_params, state)
        i: cython.Py_ssize_t
        for i in range(env.action_size):
            action[i] += gsl_ran_gaussian(rng, 0.1)  # Adding Exploration Noise
            if action[i] < -1.0:
                action[i] = -1.0
            elif action[i] > 1.0:
                action[i] = 1.0
        # # Random Policy
        # action: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size, cython.sizeof(cython.double)))
        # i: cython.Py_ssize_t
        # for i in range(env.action_size):
        #     action[i] = gsl_ran_flat(rng, -1.0, 1.0)

        total_reward += step(env, action)
        free(state)
        free(action)
        state = get_state(env)
        steps += 1
    free(state)
    if isnan(total_reward):
        total_reward = 0.0
    return total_reward


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_backup(node: cython.pointer(MKDNode), action: cython.pointer(cython.double), rtn: cython.double,
               move_indices: cython.pointer(cython.int),
               move_last: cython.pointer(cython.int), env: MujocoEnv) -> cython.int:
    depth: cython.int = move_last[0]
    if not node.terminal and not node.expanded:
        omp_set_lock(cython.address(node.access_lock))
        # Kernel of the action
        kmu: cython.pointer(cython.double) = action
        kcov: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                          calloc(env.action_size * env.action_size,
                                                                 cython.sizeof(cython.double)))
        i: cython.Py_ssize_t
        for i in range(env.action_size):
            kcov[i * env.action_size + i] = node.kernel_cov
        # print("Backup: Kernel Finding")
        # Finding the ideal kernel to merge with using Euclidean distance
        scratch: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                             calloc(env.action_size, cython.sizeof(cython.double)))
        min_kernel_index: cython.int = 0
        cblas_dcopy(env.action_size, cython.address(node.mu[min_kernel_index * env.action_size]), 1, scratch,
                    1)  # scratch = node.mu[0]
        cblas_daxpy(env.action_size, -1.0, kmu, 1, scratch, 1)  # scratch = scratch - kmu
        min_sum: cython.double = cblas_ddot(env.action_size, scratch, 1, scratch, 1)  # dot(scratch, scratch)
        for i in range(node.num_kernels):
            cblas_dcopy(env.action_size, cython.address(node.mu[cython.cast(cython.int, i) * env.action_size]), 1,
                        scratch,
                        1)  # scratch = node.mu[i]
            cblas_daxpy(env.action_size, -1.0, kmu, 1, scratch, 1)  # scratch = scratch - kmu
            sum: cython.double = cblas_ddot(env.action_size, scratch, 1, scratch, 1)  # dot(scratch, scratch)
            if sum < min_sum:
                min_sum = sum
                min_kernel_index = cython.cast(cython.int, i)
        # print(min_kernel_index)
        # Merging the kernels
        # print("Backup: Merging Kernel")
        # Z-score standardization of the kernel weights and merge
        mean: cython.double = 0
        for i in range(node.num_kernels):  # mean = sum(node.w / node.n) / node.num_kernels
            mean += node.w[i]
        mean /= node.num_kernels
        scratch2: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                              calloc(node.num_kernels, cython.sizeof(cython.double)))
        for i in range(node.num_kernels):  # scratch2 = (node.w / node.n - mean)
            scratch2[i] = (node.w[i] - mean)
        std: cython.double = cblas_ddot(node.num_kernels, scratch2, 1, scratch2,
                                        1) / node.num_kernels  # std = dot(scratch2, scratch2) / node.num_kernels
        free(scratch2)
        # print(mean, std)
        # print("Scratch:")
        # for i in range(env.action_size):
        #     print(f"{scratch[i]}", end=", ")
        # print("")
        # print("node.mu[min_kernel]")
        # for i in range(env.action_size):
        #     print(f"{node.mu[min_kernel_index * env.action_size + i]}", end=", ")
        # print("")
        # print("node.cov[min_kernel]")
        # for i in range(env.action_size):
        #     for j in range(env.action_size):
        #         print(f"{node.cov[min_kernel_index * env.action_size * env.action_size + i * env.action_size + j]}",
        #               end=", ")
        # print("")
        # print("kmu")
        # for i in range(env.action_size):
        #     print(f"{kmu[i]}", end=", ")
        # print("")
        # print("kcov")
        # for i in range(env.action_size):
        #     for j in range(env.action_size):
        #         print(f"{kcov[i * env.action_size + j]}",
        #               end=", ")
        # print("")

        # Merging Means, Cov, and Weights
        k_w_1: cython.double = (node.w[min_kernel_index] - mean) / std
        k_w_2: cython.double = (rtn - mean) / std
        k_w_1 = exp(k_w_1)
        k_w_2 = exp(k_w_2)
        k_w_1 /= (k_w_1 + k_w_2)
        k_w_2 /= (k_w_1 + k_w_2)
        # print(k_w_1, k_w_2)
        cblas_dcopy(env.action_size, cython.address(node.mu[min_kernel_index * env.action_size]), 1, scratch,
                    1)  # scratch = node.mu[min_kernel]
        cblas_dscal(env.action_size, k_w_1, scratch, 1)  # scratch = k_w_1 * scratch
        cblas_daxpy(env.action_size, k_w_2, kmu, 1, scratch, 1)  # scratch = scratch + k_w_2 * kmu
        # print("Scratch:")
        # for i in range(env.action_size):
        #     print(f"{scratch[i]}", end=", ")
        # print("")
        cblas_daxpy(env.action_size, -1.0, scratch, 1, cython.address(node.mu[min_kernel_index * env.action_size]),
                    1)  # node.mu[min_kernel] -= scratch
        cblas_dscal(env.action_size, -1.0, cython.address(node.mu[min_kernel_index * env.action_size]),
                    1)  # node.mu[min_kernel] = -node.mu[min_kernel]
        cblas_daxpy(env.action_size, -1.0, scratch, 1, kmu, 1)  # kmu -= scratch
        cblas_dscal(env.action_size, -1.0, kmu, 1)  # kmu = -kmu
        # print("Scratch:")
        # for i in range(env.action_size):
        #     print(f"{scratch[i]}", end=", ")
        # print("")

        # node.cov[min_kernel] += (node.mu[min_kernel] @ node.mu[min_kernel]^T)
        cblas_dger(CblasRowMajor, env.action_size, env.action_size, 1.0,
                   cython.address(node.mu[min_kernel_index * env.action_size]), 1,
                   cython.address(node.mu[min_kernel_index * env.action_size]), 1,
                   cython.address(node.cov[min_kernel_index * env.action_size * env.action_size]), env.action_size)

        # kcov += (kmu @ kmu^T)
        cblas_dger(CblasRowMajor, env.action_size, env.action_size, 1.0, kmu, 1, kmu, 1, kcov, env.action_size)

        cblas_dscal(env.action_size * env.action_size, k_w_1,
                    cython.address(node.cov[min_kernel_index * env.action_size * env.action_size]),
                    1)  # node.cov[min_kernel] = k_w_1 * node.cov[min_kernel]
        cblas_daxpy(env.action_size * env.action_size, k_w_2, kcov, 1,
                    cython.address(node.cov[min_kernel_index * env.action_size * env.action_size]),
                    1)  # node.cov[min_kernel] += k_w_2 * kcov
        # print("Scratch:")
        # for i in range(env.action_size):
        #     print(f"{scratch[i]}", end=", ")
        # print("")

        cblas_dcopy(env.action_size, scratch, 1, cython.address(node.mu[min_kernel_index * env.action_size]),
                    1)  # node.mu[min_kernel] = scratch
        # clip(node.mu[min_kernel], 1.0)
        for i in range(env.action_size):
            if node.mu[min_kernel_index * env.action_size + i] < -1.0:
                node.mu[min_kernel_index * env.action_size + i] = -1.0
            elif node.mu[min_kernel_index * env.action_size + i] > 1.0:
                node.mu[min_kernel_index * env.action_size + i] = 1.0

        # print("node.mu[min_kernel]")
        # for i in range(env.action_size):
        #     print(f"{node.mu[min_kernel_index * env.action_size + i]}", end=", ")
        # print("")
        # print("node.cov[min_kernel]")
        # for i in range(env.action_size):
        #     for j in range(env.action_size):
        #         print(f"{node.cov[min_kernel_index * env.action_size * env.action_size + i * env.action_size + j]}",
        #               end=", ")
        # print("")
        # print("kmu")
        # for i in range(env.action_size):
        #     print(f"{kmu[i]}", end=", ")
        # print("")
        # print("kcov")
        # for i in range(env.action_size):
        #     for j in range(env.action_size):
        #         print(f"{kcov[i * env.action_size + j]}",
        #               end=", ")
        # print("")
        node.n[min_kernel_index] += 1
        node.w[min_kernel_index] = node.w[min_kernel_index] + (1 / node.n[min_kernel_index]) * (
                rtn - node.w[min_kernel_index])
        mean: cython.double = 0
        for i in range(node.num_kernels):  # mean = sum(node.w / node.n) / node.num_kernels
            mean += node.w[i]
        mean /= node.num_kernels
        scratch2: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                              calloc(node.num_kernels, cython.sizeof(cython.double)))
        for i in range(node.num_kernels):  # scratch2 = (node.w / node.n - mean)
            scratch2[i] = (node.w[i] - mean)
        std: cython.double = cblas_ddot(node.num_kernels, scratch2, 1, scratch2,
                                        1) / node.num_kernels  # std = dot(scratch2, scratch2) / node.num_kernels
        free(scratch2)
        for i in range(node.num_kernels):
            node.pi[i] = (node.w[i] - mean) / std  # / node.n[i]
        softmax(node.pi, node.num_kernels)
        # print(f"Backup: Free scratch spaces")
        free(scratch)
        free(kmu)
        free(kcov)
        omp_unset_lock(cython.address(node.access_lock))

    # Update parent node statistics
    # print(f"Backup: Parent Updates {move_last[0]}")
    rtn += node.parent_reward
    node = node.parent
    while move_last[0] > 0:
        move_last[0] -= 1
        omp_set_lock(cython.address(node.access_lock))
        node.n[move_indices[move_last[0]]] += 1
        node.w[move_indices[move_last[0]]] = node.w[move_indices[move_last[0]]] + (
                1 / node.n[move_indices[move_last[0]]]) * (rtn - node.w[move_indices[move_last[0]]])
        mean: cython.double = 0
        i: cython.Py_ssize_t
        for i in range(node.num_kernels):  # mean = sum(node.w / node.n) / node.num_kernels
            mean += node.w[i]
        mean /= node.num_kernels
        scratch2: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                              calloc(node.num_kernels, cython.sizeof(cython.double)))
        for i in range(node.num_kernels):  # scratch2 = (node.w / node.n - mean)
            scratch2[i] = (node.w[i] - mean)
        std: cython.double = cblas_ddot(node.num_kernels, scratch2, 1, scratch2,
                                        1) / node.num_kernels  # std = dot(scratch2, scratch2) / node.num_kernels
        free(scratch2)
        i: cython.Py_ssize_t
        for i in range(node.num_kernels):
            node.pi[i] = (node.w[i])# - mean) / std  # / node.n[i]
        softmax(node.pi, node.num_kernels)
        rtn += node.parent_reward
        omp_unset_lock(cython.address(node.access_lock))
        node = node.parent
        # print(f"Backup: Parent Updates {move_last[0]}")

    # print("Backup: out")
    return depth


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_mcts_job(j: cython.Py_ssize_t, root: cython.pointer(MKDNode), max_depth: cython.int, env: MujocoEnv,
                 rollout_params: PolicyParams) -> cython.int:
    node: cython.pointer(MKDNode) = root
    # Allocating Scratch space to store moves selected in In-Tree phase
    move_indices: cython.pointer(cython.int) = cython.cast(cython.pointer(cython.int),
                                                           calloc(max_depth, cython.sizeof(cython.int)))
    move_last: cython.int = 0
    # with cython.gil:
    #     print(f"Starting: {j}")
    # Saving the original state of the environment and will be restored later. The MCTS routines change env.data to reset state to an arbitrary timestep

    original_data: cython.pointer(mjData) = env.data
    env.data = mj_copyData(cython.NULL, env.model, original_data)
    # env.data = data
    # print("Job: Selection")
    # Initializing a Random Number Generator
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, cython.cast(cython.int, j))

    # Selection
    node = mkd_selection(node, move_indices, cython.address(move_last), env, rng)
    # print("Job: Expansion")
    # Expansion
    node = mkd_expansion(node, move_indices, cython.address(move_last), env, rollout_params, rng)
    # with cython.gil:
    #     print(f"After expansion: {j} {node.expanded} {node.terminal}")
    # If tree-parallelized, lock might be held on the leaf node depending on whether it was expanded or not. Unlock it.

    # print("Job: Rollout")
    # Sample action according to current node params and begin rollout from next state
    if not node.terminal:
        # original_data: cython.pointer(mjData) = env.data
        # env.data = mj_copyData(cython.NULL, env.model, env.data)
        original_state: cython.pointer(mjtNum) = get_mj_state(env)
        set_mj_state(env, node.mj_state)    # Resetting the controller to node
        chosen_kernel: cython.int = 0
        sum: cython.double = 0.0
        rm: cython.double = gsl_rng_uniform(rng)
        i: cython.Py_ssize_t
        for i in range(node.num_kernels):
            sum += node.pi[i]
            if rm < sum:
                chosen_kernel = cython.cast(cython.int, i)
                break
        # with cython.gil:
        #     print(f"Chosen kernel: {chosen_kernel}")
        action: cython.pointer(cython.double) = sample_multivariate_gaussian(1, cython.address(
            node.mu[chosen_kernel * env.action_size]),
                                                                             cython.address(node.cov[
                                                                                                chosen_kernel * env.action_size * env.action_size]),
                                                                             env.action_size, rng)
        for i in range(env.action_size):
            if action[i] < -1.0:
                action[i] = -1.0
            elif action[i] > 1.0:
                action[i] = 1.0

        r: cython.double = step(env, action)
        rtn: cython.double = r + mkd_rollout(node.current_step + 1, env, rollout_params, rng)
        # mj_deleteData(env.data)
        # env.data = original_data
        set_mj_state(env, original_state)
        free(original_state)
    else:
        rtn: cython.double = 0
        action: cython.pointer(cython.double) = cython.NULL
    # with cython.gil:
    #     print(f"After rollout: {j}")

    # Backup
    depth: cython.int = mkd_backup(node, action, rtn, move_indices, cython.address(move_last), env)
    # with cython.gil:
    #     print(f"After backup: {j}")
    # Free allocated memory
    free(move_indices)
    gsl_rng_free(rng)
    mj_deleteData(env.data)
    # env.data = original_data  # Don't need to restore the original data because env is not a pointer but it is a good practice
    # mj_forward(env.model, env.data)
    return depth


@cython.cfunc
def mkd_mcts(num_simulations: cython.int, root: cython.pointer(MKDNode), max_depth: cython.int, env: MujocoEnv,
             rollout_params: PolicyParams) -> cython.int:
    i: cython.Py_ssize_t
    depths: cython.pointer(cython.int) = cython.cast(cython.pointer(cython.int),
                                                     calloc(num_simulations, cython.sizeof(cython.int)))
    # datas: cython.pointer(cython.pointer(mjData)) = cython.cast(cython.pointer(cython.pointer(mjData)), calloc(num_simulations, cython.sizeof(cython.pointer(mjData))))
    # for i in range(num_simulations):
    #     datas[i] = mj_copyData(cython.NULL, env.model, env.data)
    for i in prange(num_simulations, nogil=True):
        # print(f"Sim: {i}")
        depths[i] = mkd_mcts_job(i, root, max_depth, env, rollout_params)
        # print(f"Depth: {depths[i]}")
    max_depth: cython.int = 0
    for i in range(num_simulations):
        # mj_deleteData(datas[i])
        if depths[i] > max_depth:
            max_depth = depths[i]
    free(depths)
    return max_depth

# @cython.cfunc
# @cython.nogil
# @cython.exceptval(check=False)
# def prune_node(node: cython.pointer(MKDNode), keep_kernel: cython.Py_ssize_t):
#     i: cython.Py_ssize_t
#     for i in range(node.num_kernels):
#

def driver(env_name, weightT, bias):
    env_dict = {"ant": {"env_id": 0, "xml_path": "./env_xmls/ant.xml".encode(), "step_skip": 5, "max_steps": 5000}}
    env: MujocoEnv = create_env(env_dict[env_name]["env_id"], env_dict[env_name]["xml_path"],
                                env_dict[env_name]["step_skip"], env_dict[env_name]["max_steps"])
    print(env.env_id, env.state_size, env.action_size, env.mj_state_size)

    w: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                   calloc(weightT.shape[0] * weightT.shape[1],
                                                          cython.sizeof(cython.double)))
    b: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                   calloc(weightT.shape[1], cython.sizeof(cython.double)))

    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    for i in range(env.action_size):
        for j in range(env.state_size):
            w[i * env.state_size + j] = weightT[j][i]
    for j in range(env.action_size):
        b[j] = bias[j]
    # print("W:")
    # for i in range(env.action_size):
    #     for j in range(env.state_size):
    #         print(w[i * env.state_size + j], end=", ")
    #     print("")
    # print("Bias:")
    # for j in range(env.action_size):
    #     print(b[j], end=", ")
    # print("")

    params: PolicyParams = PolicyParams(k=env.action_size, n=env.state_size, w=w, b=b)
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, 2)
    reset_env(env, rng)

    i = 0
    num_kernels: cython.int = 10
    num_visitations: cython.int = 1000
    init_cov: cython.double = 1.0
    kernel_cov: cython.double = 0.1
    root: cython.pointer(MKDNode) = mkd_create_tree_node(env, i, cython.NULL, 0, False, num_kernels,
                                                         env.action_size, num_visitations, init_cov, kernel_cov)
    total_reward: cython.double = 0.0
    # print("env mj state:")
    # state: cython.pointer(cython.double) = get_mj_state(env)
    # for j in range(env.mj_state_size):
    #     print(f"{state[j]}", end=", ")
    # print("")
    # print("node mj state:")
    # state: cython.pointer(cython.double) = root.mj_state
    # for j in range(env.mj_state_size):
    #     print(f"{state[j]}", end=", ")
    # print("")

    while not is_terminated(env, i):
        print(f"Step: {i}")
        start = time.perf_counter_ns()
        print("Depth:", mkd_mcts(10000, root, 100, env, params))
        end = time.perf_counter_ns()
        print(f"Time: {(end - start) / 1e6} ms")
        j: cython.Py_ssize_t
        print("W:")
        for j in range(root.num_kernels):
            print(f"{round(root.w[j], 3)}, ", end="")
        print("")
        print("N:")
        for j in range(root.num_kernels):
            print(f"{root.n[j]}, ", end="")
        print("")
        print("P:")
        for j in range(root.num_kernels):
            print(f"{round(root.pi[j], 3)}, ", end="")
        print("")
        max: cython.double = root.w[0]
        selected_kernel: cython.Py_ssize_t = 0
        for j in range(root.num_kernels):
            if max < root.w[j]:
                max = root.w[j]
                selected_kernel = j
        print(f"Selected Kernel: {selected_kernel}")
        print("Action: ")
        for j in range(env.action_size):
            print(f"{round(root.mu[cython.cast(cython.int, selected_kernel) * env.action_size + j], 3)}", end=", ")
        print("")

        total_reward += step(env, cython.address(root.mu[cython.cast(cython.int, selected_kernel) * env.action_size]))
        if root.children == cython.NULL:
            root.iterations_left = 0
            mkd_expand_node(root, env, params, rng)
        node: cython.pointer(MKDNode) = root.children[selected_kernel]
        root.children[selected_kernel] = cython.NULL
        mkd_free_tree_node(root)
        root = node
        # print("env mj state:")
        # state: cython.pointer(cython.double) = get_mj_state(env)
        # for j in range(env.mj_state_size):
        #     print(f"{state[j]}", end=", ")
        # print("")
        # print("node mj state:")
        # state: cython.pointer(cython.double) = root.mj_state
        # for j in range(env.mj_state_size):
        #     print(f"{state[j]}", end=", ")
        # print("")
        state: cython.pointer(cython.double) = get_state(env)
        print("State:")
        for j in range(env.state_size):
            print(f"{round(state[j], 3)}", end=", ")
        print("")
        print(f"Reward Collected: {total_reward}")

        # if (i+1)%10 == 0:
        #     original_data: cython.pointer(mjData) = env.data
        action: cython.pointer(cython.double) = policy(params, state)
        free(state)
        print("Policy Action:")
        for j in range(env.action_size):
            print(f"{action[j]}", end=", ")
        print("")
        free(action)
        # for j in range(5):
        #     env.data = mj_copyData(cython.NULL, env.model, original_data)
        #     reward: cython.double = step(env, action)
        #     next_state: cython.pointer(cython.double) = get_state(env)
        #     print(f"State: {j} Reward: {reward}")
        #     k: cython.Py_ssize_t
        #     for k in range(env.state_size):
        #         print(f"{next_state[k]}", end=", ")
        #     print("")
        #
        #         mj_deleteData(env.data)
        #     env.data = original_data


        i += 1

    mkd_free_tree_node(root)
    free(params.w)
    free(params.b)
    free_env(env)
    gsl_rng_free(rng)
