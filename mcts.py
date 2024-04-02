# cython: cdivision = True

import cython
from cython.parallel import prange
from cython.cimports.mujoco_envs import MujocoEnv, PolicyParams, step, get_state, \
    is_terminated, policy, set_action, create_env, reset_env, free_env, get_mj_state, \
    set_mj_state
from cython.cimports.libc.stdlib import calloc, free
from cython.cimports.libc.string import memcpy
from cython.cimports.libc.math import pow, sqrt, exp, isnan, log
from cython.cimports.openmp import omp_lock_t, omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock
from cython.cimports.gsl import gsl_rng_uniform, gsl_rng_type, gsl_rng_default, \
    gsl_rng_alloc, gsl_rng_set, gsl_rng, gsl_rng_free, gsl_ran_flat, gsl_ran_gaussian
from cython.cimports.gsl import cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasTrans, cblas_daxpy, \
    cblas_ddot, cblas_dcopy, cblas_dscal, cblas_dger
from cython.cimports.mujoco import mjData, mj_copyData, mj_deleteData, mj_forward, mjModel, mjtNum
import time
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


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
                data_dim, 0.0, output, data_dim)  # output = samples @ L_cov
    for i in range(num_samples):
        cblas_daxpy(data_dim, 1.0, mu, 1, cython.address(output[i * data_dim]), 1)  # output[i] += mu
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

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def rollout(steps_taken: cython.int, env: MujocoEnv,
                rollout_params: PolicyParams, rng: cython.pointer(gsl_rng), noise: cython.bint) -> cython.double:
    total_reward: cython.double = 0.0
    steps: cython.Py_ssize_t = steps_taken
    state: cython.pointer(cython.double) = get_state(env)
    # Rollout till the end of the episode
    while (steps * env.num_steps < env.max_steps) and (not is_terminated(env, steps)):
        # Select actions according to rollout policy
        action: cython.pointer(cython.double) = policy(rollout_params, state)
        i: cython.Py_ssize_t
        if noise:
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
        #     action[i] = gsl_ran_gaussian(rng, 0.2)

        total_reward += step(env, action)
        free(state)
        free(action)
        state = get_state(env)
        steps += 1
    free(state)
    if isnan(total_reward):
        total_reward = 0.0
    return total_reward


# =================================== MKD MCTS ==========================


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_create_tree_node(env: MujocoEnv, current_step: cython.int,
                         parent: cython.pointer(MKDNode), parent_reward: cython.double, terminal: cython.bint,
                         num_kernels: cython.int,
                         action_dim: cython.int, replace_every_iterations: cython.int, max_iterations: cython.int,
                         init_cov: cython.double, kernel_cov: cython.double) -> cython.pointer(
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
    node.replace_every = replace_every_iterations
    node.max_iterations = max_iterations
    node.init_cov = init_cov
    node.kernel_cov = kernel_cov
    node.mean_w = 0.0
    node.std_w = 1.0
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
        node.alt_pi = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
        i: cython.Py_ssize_t
        for i in range(num_kernels):
            node.alt_pi[i] = 1.0 / num_kernels
        node.alt_w = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
        node.alt_n = cython.cast(cython.pointer(cython.int), calloc(num_kernels, cython.sizeof(cython.int)))
        node.alt_mu = cython.cast(cython.pointer(cython.double),
                                  calloc(num_kernels * action_dim, cython.sizeof(cython.double)))
        node.alt_cov = cython.cast(cython.pointer(cython.double),
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
            if node.alt_pi != cython.NULL:
                free(node.alt_pi)
                node.alt_pi = cython.NULL
            if node.alt_w != cython.NULL:
                free(node.alt_w)
                node.alt_w = cython.NULL
            if node.alt_n != cython.NULL:
                free(node.alt_n)
                node.alt_n = cython.NULL
            if node.alt_mu != cython.NULL:
                free(node.alt_mu)
                node.alt_mu = cython.NULL
            if node.alt_cov != cython.NULL:
                free(node.alt_cov)
                node.alt_cov = cython.NULL
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
        scratch: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                             calloc(node.num_kernels, cython.sizeof(cython.double)))
        i: cython.Py_ssize_t
        total_times: cython.int = 0
        for i in range(node.num_kernels):
            total_times += node.n[i]
        for i in range(node.num_kernels):
            scratch[i] = (node.w[i] - node.mean_w) / node.std_w + 100 * node.pi[i] * sqrt(total_times) / (1 + node.n[i])
        max: cython.double = scratch[0]
        for i in range(node.num_kernels):
            if max < scratch[i]:
                selected_action_index = i
                max = scratch[i]
        free(scratch)

        # Note down the index of the action taken and continue selection to child
        move_indices[move_last[0]] = cython.cast(cython.int, selected_action_index)
        move_last[0] += 1
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
            val: cython.double = 0.5
            for i in range(node.num_kernels):
                for j in range(env.action_size):
                    node.mu[i * env.action_size + j] = 0  # gsl_ran_flat(rng, -1.0, 1.0)
                    node.cov[i * env.action_size * env.action_size + j * env.action_size + j] = node.init_cov

            for i in range(node.num_kernels):
                if i != 0:
                    for j in range(env.action_size):
                        if j == i % env.action_size and j != 0:
                            val = -val
                        node.mu[i * env.action_size + j] = val
                        node.cov[
                            i * env.action_size * env.action_size + j * env.action_size + j] = node.init_cov
                    val = -val

            # Perform one rollout per kernel to initialize w and n values
            original_data: cython.pointer(mjData) = env.data
            for i in prange(node.num_kernels, nogil=True):
                # Take an action and get next step
                env2: MujocoEnv = env
                env2.data = mj_copyData(cython.NULL, env.model, original_data)

                set_mj_state(env2, node.mj_state)  # Resetting the controller to node
                r: cython.double = step(env2, cython.address(node.mu[i * env.action_size]))

                # Rollout from this next state
                rtn: cython.double = r + rollout(node.current_step + 1, env2, rollout_params, rng, True)
                node.w[i] = rtn
                node.n[i] += 1
                mj_deleteData(env2.data)
            # Copy new stats
            memcpy(node.alt_pi, node.pi, node.num_kernels * cython.sizeof(cython.double))
            memcpy(node.alt_w, node.w, node.num_kernels * cython.sizeof(cython.double))
            memcpy(node.alt_n, node.n, node.num_kernels * cython.sizeof(cython.int))
            memcpy(node.alt_mu, node.mu, node.num_kernels * node.action_dim * cython.sizeof(cython.double))
            memcpy(node.alt_cov, node.cov,
                   node.num_kernels * node.action_dim * node.action_dim * cython.sizeof(cython.double))
        else:
            node.expanded = True

    if not node.expanded:
        # Expansion is only performed when the leaf node has been visited a certain number of times
        if node.iterations_left == 0:
            node.children = cython.cast(cython.pointer(cython.pointer(MKDNode)),
                                        calloc(node.num_kernels, cython.sizeof(cython.pointer(MKDNode))))
            # Perform one rollout per kernel to finalize w  values
            original_data: cython.pointer(mjData) = env.data

            for i in prange(node.num_kernels, nogil=True):
                # Take an action and get next step
                env2: MujocoEnv = env
                env2.data = mj_copyData(cython.NULL, env2.model, original_data)
                set_mj_state(env2, node.mj_state)  # Resetting the controller to node

                r: cython.double = step(env2, cython.address(node.mu[i * env.action_size]))

                # Rollout from this next state
                rtn: cython.double = r + rollout(node.current_step + 1, env2, rollout_params, rng, True)
                node.w[i] = rtn
                mj_deleteData(env2.data)

            num_samples: cython.int = 100
            for i in prange(node.num_kernels, nogil=True):
                sampled_actions: cython.pointer(cython.double) = sample_multivariate_gaussian(num_samples,
                                                                                              cython.address(
                                                                                                  node.mu[
                                                                                                      i * env.action_size]),
                                                                                              cython.address(node.cov[
                                                                                                                 i * env.action_size * env.action_size]),
                                                                                              env.action_size, rng)
                for j in range(num_samples):
                    # Take an action and get next step
                    env2: MujocoEnv = env
                    env2.data = mj_copyData(cython.NULL, env2.model, original_data)
                    set_mj_state(env2, node.mj_state)  # Resetting the controller to node

                    r: cython.double = step(env2, cython.address(sampled_actions[j * env.action_size]))

                    # Rollout from this next state
                    rtn: cython.double = r + rollout(node.current_step + 1, env2, rollout_params, rng, True)
                    if rtn > node.w[i]:
                        node.w[i] = rtn
                        memcpy(cython.address(node.mu[i * env.action_size]),
                               cython.address(sampled_actions[j * env.action_size]),
                               env.action_size * cython.sizeof(cython.double))
                    mj_deleteData(env2.data)
            mean: cython.double = 0
            for i in range(node.num_kernels):  # mean = sum(node.w / node.n) / node.num_kernels
                mean += node.w[i]
            mean /= node.num_kernels
            scratch2: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                                  calloc(node.num_kernels,
                                                                         cython.sizeof(cython.double)))
            for i in range(node.num_kernels):  # scratch2 = (node.w / node.n - mean)
                scratch2[i] = (node.w[i] - mean)
            std: cython.double = sqrt(cblas_ddot(node.num_kernels, scratch2, 1, scratch2,
                                                 1) / node.num_kernels)  # std = sqrt(dot(scratch2, scratch2) / node.num_kernels)
            # if std == 0.0:
            std += 0.1
            free(scratch2)
            for i in range(node.num_kernels):
                node.pi[i] = (node.w[i] - mean) / std  # / node.n[i]
            softmax(node.pi, node.num_kernels)

            i: cython.Py_ssize_t
            for i in range(node.num_kernels):
                env.data = mj_copyData(cython.NULL, env.model, original_data)
                set_mj_state(env, node.mj_state)  # Resetting the controller to node
                # node.w[i] = gsl_ran_gaussian(rng, 1.0)   # Resetting Statistics for UCT
                node.n[i] = 1
                r: cython.double = step(env, cython.address(node.mu[cython.cast(cython.int, i) * env.action_size]))

                node.children[i] = mkd_create_tree_node(env, node.current_step + 1, node, r,
                                                        is_terminated(env, node.current_step + 1), node.num_kernels,
                                                        env.action_size, node.replace_every, node.max_iterations,
                                                        node.init_cov, node.kernel_cov)
                mj_deleteData(env.data)
            env.data = original_data  # Don't need to restore the original value because env is not a pointer but it is a good practice
            free(node.alt_pi)
            node.alt_pi = cython.NULL
            free(node.alt_w)
            node.alt_w = cython.NULL
            free(node.alt_n)
            node.alt_n = cython.NULL
            free(node.alt_mu)
            node.alt_mu = cython.NULL
            free(node.alt_cov)
            node.alt_cov = cython.NULL
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
def mkd_rollout(node: cython.pointer(MKDNode), env: MujocoEnv,
                rollout_params: PolicyParams, rng: cython.pointer(gsl_rng)) -> MKDRolloutReturn:
    # Sample action according to current node params and begin rollout from next state
    if not node.terminal:
        original_state: cython.pointer(mjtNum) = get_mj_state(env)
        set_mj_state(env, node.mj_state)  # Resetting the controller to node

        omp_set_lock(cython.address(node.access_lock))
        chosen_kernel: cython.int = 0
        sum: cython.double = 0.0
        rm: cython.double = gsl_rng_uniform(rng)
        i: cython.Py_ssize_t
        for i in range(node.num_kernels):
            sum += node.pi[i]
            if rm < sum:
                chosen_kernel = cython.cast(cython.int, i)
                break
        action: cython.pointer(cython.double) = sample_multivariate_gaussian(1, cython.address(
            node.mu[chosen_kernel * env.action_size]),
                                                                             cython.address(node.cov[
                                                                                                chosen_kernel * env.action_size * env.action_size]),
                                                                             env.action_size, rng)
        nan: cython.bint = False
        for i in range(env.action_size):
            nan = nan or isnan(action[i])
        if nan:
            with cython.gil:
                print("node.mu")
                for j in range(node.num_kernels):
                    print(f"{j} [", end=" ")
                    for k in range(env.action_size):
                        print(f"{round(node.mu[j * env.action_size + k], 3)}", end=", ")
                    print("] ")
                print("node.cov")
                l: cython.Py_ssize_t
                for l in range(node.num_kernels):
                    print(f"{l} [", end=" ")
                    for k in range(env.action_size):
                        for j in range(env.action_size):
                            print(
                                f"{round(node.cov[l * env.action_size * env.action_size + k * env.action_size + j], 3)}",
                                end=", ")
                        print("")
                    print("] ")
        omp_unset_lock(cython.address(node.access_lock))

        for i in range(env.action_size):
            if action[i] < -1.0:
                action[i] = -1.0
            elif action[i] > 1.0:
                action[i] = 1.0

        r: cython.double = step(env, action)
        rtn: cython.double = r + rollout(node.current_step + 1, env, rollout_params, rng, True)
        set_mj_state(env, original_state)

        bckup_rtn: cython.double = rollout(node.current_step, env, rollout_params, rng, True)

        set_mj_state(env, original_state)
        free(original_state)
    else:
        rtn: cython.double = 0.0
        bckup_rtn: cython.double = 0.0
        action: cython.pointer(cython.double) = cython.NULL
    return MKDRolloutReturn(rtn=rtn, backup_rtn=bckup_rtn, action=action)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_backup(node: cython.pointer(MKDNode), action: cython.pointer(cython.double), rtn: cython.double,
               bckup_rtn: cython.double,
               move_indices: cython.pointer(cython.int),
               move_last: cython.pointer(cython.int), env: MujocoEnv) -> cython.int:
    depth: cython.int = move_last[0]
    omp_set_lock(cython.address(node.access_lock))
    if not node.terminal and not node.expanded:
        if (node.max_iterations - node.iterations_left) % node.replace_every == 0:
            # Free Old Stats
            free(node.pi)
            free(node.w)
            free(node.n)
            free(node.mu)
            free(node.cov)
            # Re-reference new stats
            node.pi = node.alt_pi
            node.w = node.alt_w
            node.n = node.alt_n
            node.mu = node.alt_mu
            node.cov = node.alt_cov
            # Allocate new stats
            node.alt_pi = cython.cast(cython.pointer(cython.double),
                                      calloc(node.num_kernels, cython.sizeof(cython.double)))
            node.alt_w = cython.cast(cython.pointer(cython.double),
                                     calloc(node.num_kernels, cython.sizeof(cython.double)))
            node.alt_n = cython.cast(cython.pointer(cython.int), calloc(node.num_kernels, cython.sizeof(cython.int)))
            node.alt_mu = cython.cast(cython.pointer(cython.double),
                                      calloc(node.num_kernels * node.action_dim, cython.sizeof(cython.double)))
            node.alt_cov = cython.cast(cython.pointer(cython.double),
                                       calloc(node.num_kernels * node.action_dim * node.action_dim,
                                              cython.sizeof(cython.double)))
            # Copy new stats
            memcpy(node.alt_pi, node.pi,
                   node.num_kernels * cython.sizeof(cython.double))
            memcpy(node.alt_w, node.w, node.num_kernels * cython.sizeof(cython.double))
            memcpy(node.alt_n, node.n, node.num_kernels * cython.sizeof(cython.int))
            memcpy(node.alt_mu, node.mu,
                   node.num_kernels * node.action_dim * cython.sizeof(cython.double))
            memcpy(node.alt_cov, node.cov,
                   node.num_kernels * node.action_dim * node.action_dim * cython.sizeof(cython.double))

        # Kernel of the action
        kmu: cython.pointer(cython.double) = action
        kcov: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                          calloc(env.action_size * env.action_size,
                                                                 cython.sizeof(cython.double)))
        i: cython.Py_ssize_t
        for i in range(env.action_size):
            kcov[i * env.action_size + i] = node.kernel_cov
        # Finding the ideal kernel to merge with using Euclidean distance
        scratch: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                             calloc(env.action_size, cython.sizeof(cython.double)))
        min_kernel_index: cython.int = 0
        cblas_dcopy(env.action_size, cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1, scratch,
                    1)  # scratch = node.mu[0]
        cblas_daxpy(env.action_size, -1.0, kmu, 1, scratch, 1)  # scratch = scratch - kmu
        min_sum: cython.double = cblas_ddot(env.action_size, scratch, 1, scratch, 1)  # dot(scratch, scratch)
        for i in range(node.num_kernels):
            cblas_dcopy(env.action_size, cython.address(node.alt_mu[cython.cast(cython.int, i) * env.action_size]), 1,
                        scratch,
                        1)  # scratch = node.mu[i]
            cblas_daxpy(env.action_size, -1.0, kmu, 1, scratch, 1)  # scratch = scratch - kmu
            sum: cython.double = cblas_ddot(env.action_size, scratch, 1, scratch, 1)  # dot(scratch, scratch)
            if sum < min_sum:
                min_sum = sum
                min_kernel_index = cython.cast(cython.int, i)

        # Merging the kernels

        # Z-score standardization of the kernel weights and merge
        mean: cython.double = 0
        for i in range(node.num_kernels):  # mean = sum(node.w / node.n) / node.num_kernels
            mean += node.alt_w[i]
        mean /= node.num_kernels
        scratch2: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                              calloc(node.num_kernels, cython.sizeof(cython.double)))
        for i in range(node.num_kernels):  # scratch2 = (node.w / node.n - mean)
            scratch2[i] = (node.alt_w[i] - mean)
        std: cython.double = sqrt(cblas_ddot(node.num_kernels, scratch2, 1, scratch2,
                                             1) / node.num_kernels)  # std = sqrt(dot(scratch2, scratch2) / node.num_kernels)
        free(scratch2)
        std += 0.1
        # Merging Means, Cov, and Weights
        k_w_1_: cython.double = (node.alt_w[min_kernel_index] - mean) / std
        k_w_2_: cython.double = (rtn - mean) / std
        if k_w_1_ > k_w_2_:
            k_w_1_ -= k_w_1_
            k_w_2_ -= k_w_1_
        else:
            k_w_1_ -= k_w_2_
            k_w_2_ -= k_w_2_
        k_w_1_ = exp(k_w_1_)
        k_w_2_ = exp(k_w_2_)
        k_w_1: cython.double = k_w_1_ / (k_w_1_ + k_w_2_)
        k_w_2: cython.double = k_w_2_ / (k_w_1_ + k_w_2_)

        weight_update_rtn: cython.double = rtn
        if k_w_1 > k_w_2:
            cblas_dcopy(env.action_size, cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1, scratch,
                        1)  # scratch = node.mu[min_kernel]
            cblas_dscal(env.action_size, 2.0, scratch, 1)  # scratch = 2.0 * scratch
            cblas_daxpy(env.action_size, -1.0, kmu, 1, scratch, 1)  # scratch = scratch - kmu
            cblas_dcopy(env.action_size, scratch, 1, kmu, 1)  # kmu = scratch
            # weight_update_rtn = 2 * node.w[min_kernel_index] - rtn
            # k_w_1_ = (node.w[min_kernel_index] - mean) / std
            # k_w_2_ = (weight_update_rtn - mean) / std
            # k_w_1_ = exp(k_w_1_)
            # k_w_2_ = exp(k_w_2_)
            # k_w_1: cython.double = k_w_1_ / (k_w_1_ + k_w_2_)
            # k_w_2: cython.double = k_w_2_ / (k_w_1_ + k_w_2_)
        cblas_dcopy(env.action_size, cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1, scratch,
                    1)  # scratch = node.mu[min_kernel]
        cblas_dscal(env.action_size, k_w_1, scratch, 1)  # scratch = k_w_1 * scratch
        cblas_daxpy(env.action_size, k_w_2, kmu, 1, scratch, 1)  # scratch = scratch + k_w_2 * kmu

        cblas_daxpy(env.action_size, -1.0, scratch, 1, cython.address(node.alt_mu[min_kernel_index * env.action_size]),
                    1)  # node.mu[min_kernel] -= scratch
        cblas_dscal(env.action_size, -1.0, cython.address(node.alt_mu[min_kernel_index * env.action_size]),
                    1)  # node.mu[min_kernel] = -node.mu[min_kernel]
        cblas_daxpy(env.action_size, -1.0, scratch, 1, kmu, 1)  # kmu -= scratch
        cblas_dscal(env.action_size, -1.0, kmu, 1)  # kmu = -kmu

        # node.cov[min_kernel] += (node.mu[min_kernel] @ node.mu[min_kernel]^T)
        cblas_dger(CblasRowMajor, env.action_size, env.action_size, 1.0,
                   cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1,
                   cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1,
                   cython.address(node.alt_cov[min_kernel_index * env.action_size * env.action_size]), env.action_size)

        # kcov += (kmu @ kmu^T)
        cblas_dger(CblasRowMajor, env.action_size, env.action_size, 1.0, kmu, 1, kmu, 1, kcov, env.action_size)

        cblas_dscal(env.action_size * env.action_size, k_w_1,
                    cython.address(node.alt_cov[min_kernel_index * env.action_size * env.action_size]),
                    1)  # node.cov[min_kernel] = k_w_1 * node.cov[min_kernel]
        cblas_daxpy(env.action_size * env.action_size, k_w_2, kcov, 1,
                    cython.address(node.alt_cov[min_kernel_index * env.action_size * env.action_size]),
                    1)  # node.cov[min_kernel] += k_w_2 * kcov

        cblas_dcopy(env.action_size, scratch, 1, cython.address(node.alt_mu[min_kernel_index * env.action_size]),
                    1)  # node.mu[min_kernel] = scratch
        # clip(node.mu[min_kernel], 1.0)
        for i in range(env.action_size):
            if node.alt_mu[min_kernel_index * env.action_size + i] < -1.0:
                node.alt_mu[min_kernel_index * env.action_size + i] = -1.0
            elif node.alt_mu[min_kernel_index * env.action_size + i] > 1.0:
                node.alt_mu[min_kernel_index * env.action_size + i] = 1.0

        node.alt_n[min_kernel_index] += 1
        node.alt_w[min_kernel_index] = node.alt_w[min_kernel_index] + (1 / node.alt_n[min_kernel_index]) * (
                weight_update_rtn - node.alt_w[min_kernel_index])
        mean: cython.double = 0
        for i in range(node.num_kernels):  # mean = sum(node.w / node.n) / node.num_kernels
            mean += node.alt_w[i]
        mean /= node.num_kernels
        scratch2: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                              calloc(node.num_kernels, cython.sizeof(cython.double)))
        for i in range(node.num_kernels):  # scratch2 = (node.w / node.n - mean)
            scratch2[i] = (node.alt_w[i] - mean)
        std: cython.double = sqrt(cblas_ddot(node.num_kernels, scratch2, 1, scratch2,
                                             1) / node.num_kernels)  # std = sqrt(dot(scratch2, scratch2) / node.num_kernels)
        std += 0.1
        free(scratch2)
        for i in range(node.num_kernels):
            node.alt_pi[i] = (node.alt_w[i] - mean) / std  # / node.n[i]
        softmax(node.alt_pi, node.num_kernels)

        free(scratch)
        free(kmu)
        free(kcov)
    omp_unset_lock(cython.address(node.access_lock))

    # Update parent node statistics
    rtn = bckup_rtn
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
        std: cython.double = sqrt(cblas_ddot(node.num_kernels, scratch2, 1, scratch2,
                                             1) / node.num_kernels)  # std = sqrt(dot(scratch2, scratch2) / node.num_kernels)
        free(scratch2)
        node.mean_w = mean
        node.std_w = std + 0.1
        # i: cython.Py_ssize_t
        # for i in range(node.num_kernels):
        #     node.pi[i] = (node.w[i] - mean) / std  # / node.n[i]
        # softmax(node.pi, node.num_kernels)
        rtn += node.parent_reward
        omp_unset_lock(cython.address(node.access_lock))
        node = node.parent

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

    # Saving the original state of the environment and will be restored later. The MCTS routines change env.data to reset state to an arbitrary timestep

    original_data: cython.pointer(mjData) = env.data
    env.data = mj_copyData(cython.NULL, env.model, original_data)

    # Initializing a Random Number Generator
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, cython.cast(cython.int, j))

    # Selection
    node = mkd_selection(node, move_indices, cython.address(move_last), env, rng)
    # Expansion
    node = mkd_expansion(node, move_indices, cython.address(move_last), env, rollout_params, rng)

    # Rollout
    roll_return: MKDRolloutReturn = mkd_rollout(node, env, rollout_params, rng)

    # Backup
    depth: cython.int = mkd_backup(node, roll_return.action, roll_return.rtn, roll_return.backup_rtn, move_indices, cython.address(move_last), env)

    # Free allocated memory
    free(move_indices)
    gsl_rng_free(rng)
    mj_deleteData(env.data)
    return depth


@cython.cfunc
def mkd_mcts(num_simulations: cython.int, root: cython.pointer(MKDNode), max_depth: cython.int, env: MujocoEnv,
             rollout_params: PolicyParams) -> cython.int:
    i: cython.Py_ssize_t
    depths: cython.pointer(cython.int) = cython.cast(cython.pointer(cython.int),
                                                     calloc(num_simulations, cython.sizeof(cython.int)))
    for i in prange(num_simulations, nogil=True):
        # print(f"Sim: {i}")
        depths[i] = mkd_mcts_job(i, root, max_depth, env, rollout_params)
        # print(f"Depth: {depths[i]}")
    max_depth: cython.int = 0
    for i in range(num_simulations):
        if depths[i] > max_depth:
            max_depth = depths[i]
    free(depths)
    return max_depth


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def simple_rollout_job(j: cython.Py_ssize_t, node: cython.pointer(MKDNode), env: MujocoEnv,
                       rollout_params: PolicyParams) -> cython.void:
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, cython.cast(cython.int, j))

    original_data: cython.pointer(mjData) = env.data
    env.data = mj_copyData(cython.NULL, env.model, original_data)

    # Sample an action from kernels and perform rollout
    omp_set_lock(cython.address(node.access_lock))
    chosen_kernel: cython.int = 0
    sum: cython.double = 0.0
    rm: cython.double = gsl_rng_uniform(rng)
    i: cython.Py_ssize_t
    for i in range(node.num_kernels):
        sum += node.pi[i]
        if rm < sum:
            chosen_kernel = cython.cast(cython.int, i)
            break

    action: cython.pointer(cython.double) = sample_multivariate_gaussian(1, cython.address(
        node.mu[chosen_kernel * env.action_size]),
                                                                         cython.address(node.cov[
                                                                                            chosen_kernel * env.action_size * env.action_size]),
                                                                         env.action_size, rng)
    omp_unset_lock(cython.address(node.access_lock))
    nan: cython.bint = False
    for i in range(env.action_size):
        nan = nan or isnan(action[i])
    if nan:
        with cython.gil:
            print("node.mu")
            for j in range(node.num_kernels):
                print(f"{j} [", end=" ")
                for k in range(env.action_size):
                    print(f"{round(node.mu[j * env.action_size + k], 3)}", end=", ")
                print("] ")
            print("node.cov")
            l: cython.Py_ssize_t
            for l in range(node.num_kernels):
                print(f"{l} [", end=" ")
                for k in range(env.action_size):
                    for j in range(env.action_size):
                        print(f"{round(node.cov[l * env.action_size * env.action_size + k * env.action_size + j], 3)}",
                              end=", ")
                    print("")
                print("] ")

    for i in range(env.action_size):
        if action[i] < -1.0:
            action[i] = -1.0
        elif action[i] > 1.0:
            action[i] = 1.0
    r: cython.double = step(env, action)
    rtn: cython.double = r + rollout(node.current_step + 1, env, rollout_params, rng, True)

    # Update the kernels with the return value

    omp_set_lock(cython.address(node.access_lock))

    if (node.max_iterations - node.iterations_left) % node.replace_every == 0:
        # Free Old Stats
        free(node.pi)
        free(node.w)
        free(node.n)
        free(node.mu)
        free(node.cov)
        # Re-reference new stats
        node.pi = node.alt_pi
        node.w = node.alt_w
        node.n = node.alt_n
        node.mu = node.alt_mu
        node.cov = node.alt_cov
        # Allocate new stats
        node.alt_pi = cython.cast(cython.pointer(cython.double), calloc(node.num_kernels, cython.sizeof(cython.double)))
        node.alt_w = cython.cast(cython.pointer(cython.double), calloc(node.num_kernels, cython.sizeof(cython.double)))
        node.alt_n = cython.cast(cython.pointer(cython.int), calloc(node.num_kernels, cython.sizeof(cython.int)))
        node.alt_mu = cython.cast(cython.pointer(cython.double),
                                  calloc(node.num_kernels * node.action_dim, cython.sizeof(cython.double)))
        node.alt_cov = cython.cast(cython.pointer(cython.double),
                                   calloc(node.num_kernels * node.action_dim * node.action_dim,
                                          cython.sizeof(cython.double)))
        # Copy new stats
        memcpy(node.alt_pi, node.pi, node.num_kernels * cython.sizeof(cython.double))
        memcpy(node.alt_w, node.w, node.num_kernels * cython.sizeof(cython.double))
        memcpy(node.alt_n, node.n, node.num_kernels * cython.sizeof(cython.int))
        memcpy(node.alt_mu, node.mu,
               node.num_kernels * node.action_dim * cython.sizeof(cython.double))
        memcpy(node.alt_cov, node.cov,
               node.num_kernels * node.action_dim * node.action_dim * cython.sizeof(cython.double))

    # Kernel of the action
    kmu: cython.pointer(cython.double) = action
    kcov: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                      calloc(env.action_size * env.action_size,
                                                             cython.sizeof(cython.double)))
    i: cython.Py_ssize_t
    for i in range(env.action_size):
        kcov[i * env.action_size + i] = node.kernel_cov
    # Finding the ideal kernel to merge with using Euclidean distance
    scratch: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                         calloc(env.action_size, cython.sizeof(cython.double)))
    min_kernel_index: cython.int = 0
    cblas_dcopy(env.action_size, cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1, scratch,
                1)  # scratch = node.mu[0]
    cblas_daxpy(env.action_size, -1.0, kmu, 1, scratch, 1)  # scratch = scratch - kmu
    min_sum: cython.double = cblas_ddot(env.action_size, scratch, 1, scratch, 1)  # dot(scratch, scratch)
    for i in range(node.num_kernels):
        cblas_dcopy(env.action_size, cython.address(node.alt_mu[cython.cast(cython.int, i) * env.action_size]), 1,
                    scratch,
                    1)  # scratch = node.mu[i]
        cblas_daxpy(env.action_size, -1.0, kmu, 1, scratch, 1)  # scratch = scratch - kmu
        sum: cython.double = cblas_ddot(env.action_size, scratch, 1, scratch, 1)  # dot(scratch, scratch)
        if sum < min_sum:
            min_sum = sum
            min_kernel_index = cython.cast(cython.int, i)

    # Merging the kernels

    # Z-score standardization of the kernel weights and merge
    mean: cython.double = 0
    for i in range(node.num_kernels):  # mean = sum(node.w / node.n) / node.num_kernels
        mean += node.alt_w[i]
    mean /= node.num_kernels
    scratch2: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                          calloc(node.num_kernels, cython.sizeof(cython.double)))
    for i in range(node.num_kernels):  # scratch2 = (node.w / node.n - mean)
        scratch2[i] = (node.alt_w[i] - mean)
    std: cython.double = sqrt(cblas_ddot(node.num_kernels, scratch2, 1, scratch2,
                                         1) / node.num_kernels)  # std = sqrt(dot(scratch2, scratch2) / node.num_kernels)
    free(scratch2)
    # if std == 0.0:
    std += 0.1
    # with cython.gil:
    #     print(f"STD: {std}")
    # Merging Means, Cov, and Weights
    k_w_1_: cython.double = (node.alt_w[min_kernel_index] - mean) / std
    k_w_2_: cython.double = (rtn - mean) / std
    if k_w_1_ > k_w_2_:
        k_w_1_ -= k_w_1_
        k_w_2_ -= k_w_1_
    else:
        k_w_1_ -= k_w_2_
        k_w_2_ -= k_w_2_

    k_w_1_ = exp(k_w_1_)
    k_w_2_ = exp(k_w_2_)
    k_w_1: cython.double = k_w_1_ / (k_w_1_ + k_w_2_)
    k_w_2: cython.double = k_w_2_ / (k_w_1_ + k_w_2_)

    weight_update_rtn: cython.double = rtn
    if k_w_1 > k_w_2:
        cblas_dcopy(env.action_size, cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1, scratch,
                    1)  # scratch = node.mu[min_kernel]
        cblas_dscal(env.action_size, 2.0, scratch, 1)  # scratch = 2.0 * scratch
        cblas_daxpy(env.action_size, -1.0, kmu, 1, scratch, 1)  # scratch = scratch - kmu
        cblas_dcopy(env.action_size, scratch, 1, kmu, 1)  # kmu = scratch
        # weight_update_rtn = 2 * node.w[min_kernel_index] - rtn
        # k_w_1_ = (node.w[min_kernel_index] - mean) / std
        # k_w_2_ = (weight_update_rtn - mean) / std
        # k_w_1_ = exp(k_w_1_)
        # k_w_2_ = exp(k_w_2_)
        # k_w_1: cython.double = k_w_1_ / (k_w_1_ + k_w_2_)
        # k_w_2: cython.double = k_w_2_ / (k_w_1_ + k_w_2_)

    cblas_dcopy(env.action_size, cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1, scratch,
                1)  # scratch = node.mu[min_kernel]
    cblas_dscal(env.action_size, k_w_1, scratch, 1)  # scratch = k_w_1 * scratch
    cblas_daxpy(env.action_size, k_w_2, kmu, 1, scratch, 1)  # scratch = scratch + k_w_2 * kmu

    cblas_daxpy(env.action_size, -1.0, scratch, 1, cython.address(node.alt_mu[min_kernel_index * env.action_size]),
                1)  # node.mu[min_kernel] -= scratch
    cblas_dscal(env.action_size, -1.0, cython.address(node.alt_mu[min_kernel_index * env.action_size]),
                1)  # node.mu[min_kernel] = -node.mu[min_kernel]
    cblas_daxpy(env.action_size, -1.0, scratch, 1, kmu, 1)  # kmu -= scratch
    cblas_dscal(env.action_size, -1.0, kmu, 1)  # kmu = -kmu

    # node.cov[min_kernel] += (node.mu[min_kernel] @ node.mu[min_kernel]^T)
    cblas_dger(CblasRowMajor, env.action_size, env.action_size, 1.0,
               cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1,
               cython.address(node.alt_mu[min_kernel_index * env.action_size]), 1,
               cython.address(node.alt_cov[min_kernel_index * env.action_size * env.action_size]), env.action_size)

    # kcov += (kmu @ kmu^T)
    cblas_dger(CblasRowMajor, env.action_size, env.action_size, 1.0, kmu, 1, kmu, 1, kcov, env.action_size)

    cblas_dscal(env.action_size * env.action_size, k_w_1,
                cython.address(node.alt_cov[min_kernel_index * env.action_size * env.action_size]),
                1)  # node.cov[min_kernel] = k_w_1 * node.cov[min_kernel]
    cblas_daxpy(env.action_size * env.action_size, k_w_2, kcov, 1,
                cython.address(node.alt_cov[min_kernel_index * env.action_size * env.action_size]),
                1)  # node.cov[min_kernel] += k_w_2 * kcov

    cblas_dcopy(env.action_size, scratch, 1, cython.address(node.alt_mu[min_kernel_index * env.action_size]),
                1)  # node.mu[min_kernel] = scratch
    # clip(node.mu[min_kernel], 1.0)
    for i in range(env.action_size):
        if node.alt_mu[min_kernel_index * env.action_size + i] < -1.0:
            node.alt_mu[min_kernel_index * env.action_size + i] = -1.0
        elif node.alt_mu[min_kernel_index * env.action_size + i] > 1.0:
            node.alt_mu[min_kernel_index * env.action_size + i] = 1.0

    node.alt_n[min_kernel_index] += 1
    node.alt_w[min_kernel_index] = node.alt_w[min_kernel_index] + (1 / node.alt_n[min_kernel_index]) * (
            weight_update_rtn - node.alt_w[min_kernel_index])
    mean: cython.double = 0
    for i in range(node.num_kernels):  # mean = sum(node.w / node.n) / node.num_kernels
        mean += node.alt_w[i]
    mean /= node.num_kernels
    scratch2: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                          calloc(node.num_kernels, cython.sizeof(cython.double)))
    for i in range(node.num_kernels):  # scratch2 = (node.w / node.n - mean)
        scratch2[i] = (node.alt_w[i] - mean)
    std: cython.double = sqrt(cblas_ddot(node.num_kernels, scratch2, 1, scratch2,
                                         1) / node.num_kernels)  # std = sqrt(dot(scratch2, scratch2) / node.num_kernels)
    # if std == 0.0:
    std += 0.1
    free(scratch2)
    for i in range(node.num_kernels):
        node.alt_pi[i] = (node.alt_w[i] - mean) / std  # / node.n[i]
    softmax(node.alt_pi, node.num_kernels)

    free(scratch)
    free(kmu)
    free(kcov)
    node.iterations_left -= 1
    omp_unset_lock(cython.address(node.access_lock))

    mj_deleteData(env.data)
    gsl_rng_free(rng)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def plot_returns(i: cython.Py_ssize_t, env: MujocoEnv, action: cython.pointer(cython.double),
                 node: cython.pointer(MKDNode), rollout_params: PolicyParams, rtns: cython.pointer(cython.double),
                 rs: cython.pointer(cython.double)) -> cython.void:
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, cython.cast(cython.int, i))

    original_data: cython.pointer(mjData) = env.data
    env.data = mj_copyData(cython.NULL, env.model, original_data)

    rs[i] = step(env, action)
    rtns[i] = rs[i] + rollout(node.current_step + 1, env, rollout_params, rng, False)

    mj_deleteData(env.data)
    gsl_rng_free(rng)


def plot(X, Y, Z1, Z2, Z3):
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3)
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.PuRd,
    #                        linewidth=0, antialiased=False, alpha=0.5)
    surf = ax1.pcolormesh(X, Y, Z1, cmap=cm.PuRd)
    surf2 = ax2.pcolormesh(X, Y, Z2, cmap=cm.PuRd)
    surf3 = ax3.pcolormesh(X, Y, Z3, cmap=cm.PuRd)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # # A StrMethodFormatter is used automatically
    # ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, )  # shrink=0.1, aspect=5)
    fig.colorbar(surf2, )  # shrink=0.1, aspect=5)
    fig.colorbar(surf3, )  # shrink=0.1, aspect=5)
    plt.show()


def normal_pdf(X, mean, cov):
    # X: shape(N,D), mean: shape(K,D), cov: shape(K,D,D) out: shape(N,K)
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    x_u = np.expand_dims(X, axis=1) - np.expand_dims(mean, axis=0)
    x_u_t = np.expand_dims(x_u, axis=2)
    x_u = np.expand_dims(x_u, axis=-1)
    cov_inv = np.expand_dims(cov_inv, axis=0)
    cov_det = np.expand_dims(cov_det, axis=0)
    return np.squeeze(np.exp(-(x_u_t @ cov_inv @ x_u) / 2.0)) / np.sqrt(((2 * np.pi) ** mean.shape[1]) * cov_det)


@cython.cfunc
# @cython.nogil
@cython.exceptval(check=False)
def simple_rollout(num_rollouts: cython.int, node: cython.pointer(MKDNode), rollout_params: PolicyParams,
                   env: MujocoEnv, rng: cython.pointer(gsl_rng)) -> cython.void:
    # Initialize parameters
    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    # spacing: cython.double = 2 / node.num_kernels   # [1 - (-1)]/num_kernels
    # val: cython.double = -1
    # for i in range(node.num_kernels):
    #     for j in range(env.action_size):
    #         # node.mu[i * env.action_size + j] = gsl_ran_flat(rng, -1.0, 1.0)
    #         node.mu[i * env.action_size + j] = val
    #         node.cov[
    #             i * env.action_size * env.action_size + j * env.action_size + j] = node.init_cov
    #     val += spacing
    val: cython.double = 0.5
    for i in range(node.num_kernels):
        for j in range(env.action_size):
            node.mu[i * env.action_size + j] = 0
            node.cov[i * env.action_size * env.action_size + j * env.action_size + j] = node.init_cov

    for i in range(1, node.num_kernels):
        for j in range(env.action_size):
            if j == i % env.action_size and j != 0:
                val = -val
            node.mu[i * env.action_size + j] = val
            node.cov[
                i * env.action_size * env.action_size + j * env.action_size + j] = node.init_cov
        val = -val
    node.params_initialized = True

    # Perform one rollout per kernel to initialize w and n values
    original_data: cython.pointer(mjData) = env.data

    for i in prange(node.num_kernels, nogil=True):
        # Take an action and get next step
        env2: MujocoEnv = env
        env2.data = mj_copyData(cython.NULL, env2.model, original_data)

        r: cython.double = step(env2, cython.address(node.mu[i * env.action_size]))

        # Rollout from this next state
        rtn: cython.double = r + rollout(node.current_step + 1, env2, rollout_params, rng, True)
        node.w[i] = rtn
        node.n[i] += 1
        mj_deleteData(env2.data)
    env.data = original_data
    # Copy new stats
    memcpy(node.alt_pi, node.pi, node.num_kernels * cython.sizeof(cython.double))
    memcpy(node.alt_w, node.w, node.num_kernels * cython.sizeof(cython.double))
    memcpy(node.alt_n, node.n, node.num_kernels * cython.sizeof(cython.int))
    memcpy(node.alt_mu, node.mu,
           node.num_kernels * node.action_dim * cython.sizeof(cython.double))
    memcpy(node.alt_cov, node.cov,
           node.num_kernels * node.action_dim * node.action_dim * cython.sizeof(cython.double))

    # Perform rollouts
    for i in prange(num_rollouts, nogil=True):
        # print(f"Sim: {i}")
        simple_rollout_job(i, node, env, rollout_params)

    # Perform one rollout per kernel to finalize w  values
    original_data: cython.pointer(mjData) = env.data

    for i in prange(node.num_kernels, nogil=True):
        # Take an action and get next step
        env2: MujocoEnv = env
        env2.data = mj_copyData(cython.NULL, env2.model, original_data)

        r: cython.double = step(env2, cython.address(node.mu[i * env.action_size]))

        # Rollout from this next state
        rtn: cython.double = r + rollout(node.current_step + 1, env2, rollout_params, rng, True)
        node.w[i] = rtn
        mj_deleteData(env2.data)

    num_samples: cython.int = 100
    for i in prange(node.num_kernels, nogil=True):
        sampled_actions: cython.pointer(cython.double) = sample_multivariate_gaussian(num_samples, cython.address(
            node.mu[i * env.action_size]),
                                                                                      cython.address(node.cov[
                                                                                                         i * env.action_size * env.action_size]),
                                                                                      env.action_size, rng)
        for j in range(num_samples):
            # Take an action and get next step
            env2: MujocoEnv = env
            env2.data = mj_copyData(cython.NULL, env2.model, original_data)

            r: cython.double = step(env2, cython.address(sampled_actions[j * env.action_size]))

            # Rollout from this next state
            rtn: cython.double = r + rollout(node.current_step + 1, env2, rollout_params, rng, True)
            if rtn > node.w[i]:
                node.w[i] = rtn
                memcpy(cython.address(node.mu[i * env.action_size]),
                       cython.address(sampled_actions[j * env.action_size]),
                       env.action_size * cython.sizeof(cython.double))
            mj_deleteData(env2.data)
    env.data = original_data


@cython.cfunc
def plot_after_action(env: MujocoEnv, node: cython.pointer(MKDNode), rollout_params: PolicyParams) -> cython.void:
    print("Plotting...")
    X = np.arange(-1.0, 1.0, 0.01)
    Y = np.arange(-1.0, 1.0, 0.01)
    X, Y = np.meshgrid(X, Y)
    data = np.concatenate([X[:, :, np.newaxis], Y[:, :, np.newaxis]], axis=-1)
    data2 = data.reshape(-1, 2)

    rtns: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                      calloc(data2.shape[0], cython.sizeof(cython.double)))
    rs: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                    calloc(data2.shape[0], cython.sizeof(cython.double)))

    actions: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                         calloc(data2.shape[0] * 2, cython.sizeof(cython.double)))
    i: cython.Py_ssize_t
    for i in range(data2.shape[0]):
        actions[i * 2] = data2[i][0]
        actions[i * 2 + 1] = data2[i][0]
    size: cython.Py_ssize_t = data2.shape[0]
    for i in prange(size, nogil=True):
        plot_returns(i, env, cython.address(actions[i * 2]), node, rollout_params, rtns, rs)

    RTNS = np.zeros(shape=data2.shape[0])
    RS = np.zeros(shape=data2.shape[0])
    for i in range(data2.shape[0]):
        RTNS[i] = rtns[i]
        RS[i] = rs[i]

    RTNS = RTNS.reshape(*data.shape[:2])
    RS = RS.reshape(*data.shape[:2])

    kernel_weights = np.zeros(shape=node.num_kernels)
    kernel_means = np.zeros(shape=(node.num_kernels, 2))
    kernel_covs = np.zeros(shape=(node.num_kernels, 2, 2))
    for i in range(node.num_kernels):
        kernel_weights[i] = node.pi[i]
        for j in range(2):
            kernel_means[i][j] = node.mu[i * 2 + j]
            for k in range(2):
                kernel_covs[i][j][k] = node.cov[i * 2 * 2 + j * 2 + k]
    MIXTURE = np.sum(kernel_weights * normal_pdf(data.reshape(-1, 2), kernel_means, kernel_covs), axis=1).reshape(
        *data.shape[:2])
    print("Waiting plot...")
    plot(X, Y, RTNS, RS, MIXTURE)
    free(actions)
    free(rtns)
    free(rs)


# =================================== VG-UCT MCTS ==========================

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def vg_create_tree_node(env: MujocoEnv, current_step: cython.int, parent: cython.pointer(VGNode),
                        parent_reward: cython.double, parent_action: cython.pointer(cython.double), terminal: cython.bint, action_dim: cython.int, child_add_alpha: cython.double) -> cython.pointer(VGNode):
    node: cython.pointer(VGNode) = cython.cast(cython.pointer(VGNode), calloc(1, cython.sizeof(VGNode)))
    node.mj_state = get_mj_state(env)
    node.current_step = current_step
    node.parent = parent
    node.parent_reward = parent_reward
    node.parent_action = parent_action
    node.init_parent_action = parent_action
    node.parent_q_value = 0.0
    omp_init_lock(cython.address(node.access_lock))
    node.terminal = terminal
    node.child_add_alpha = child_add_alpha
    node.num_visitations = 0
    node.action_dim = action_dim
    node.num_children = 0
    node.children = cython.NULL
    node.next = cython.NULL
    return node


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def vg_free_tree_node(node: cython.pointer(VGNode)) -> cython.void:
    if node != cython.NULL:
        free(node.mj_state)
        omp_destroy_lock(cython.address(node.access_lock))
        node.parent = cython.NULL
        if node.parent_action != cython.NULL:
            if node.parent_action != node.init_parent_action:
                free(node.parent_action)
                free(node.init_parent_action)
            else:
                free(node.parent_action)
        child: cython.pointer(VGNode) = node.children
        i: cython.int
        for i in range(node.num_children):
            next_child: cython.pointer(VGNode) = child.next
            vg_free_tree_node(child)
            child = next_child
        free(node)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def vg_select_action(node: cython.pointer(VGNode), env: MujocoEnv, rng: cython.pointer(gsl_rng)) -> VGSAReturn:
    rollout: cython.bint = False
    next_node: cython.pointer(VGNode) = cython.NULL
    if pow(node.num_visitations, node.child_add_alpha) > node.num_children:
        omp_set_lock(cython.address(node.access_lock))
        # TODO: Add a child by selecting a random action
        omp_unset_lock(cython.address(node.access_lock))
        rollout = True
    else:
        # Select next child using UCT

        child: cython.pointer(VGNode) = node.children
        next_node = child
        i: cython.Py_ssize_t
        max_u: cython.double = 1.0 * sqrt(log(node.num_visitations) / child.num_visitations) + node.children.parent_q_value
        for i in range(node.num_children):
            if (1.0 * sqrt(log(node.num_visitations) / child.num_visitations) + node.children.parent_q_value) > max_u:
                max_u = 1.0 * sqrt(log(node.num_visitations) / child.num_visitations) + node.children.parent_q_value
                next_node = child
            child = child.next

    return VGSAReturn(next_node=next_node, rollout=rollout)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def vg_rollout(steps_taken: cython.int, max_steps: cython.int, env: MujocoEnv,
                rollout_params: PolicyParams, rng: cython.pointer(gsl_rng), noise: cython.bint) -> VGRolloutReturn:
    total_reward: cython.double = 0.0
    steps: cython.Py_ssize_t = steps_taken
    state: cython.pointer(cython.double) = get_state(env)
    actions: cython.pointer(cython.pointer(cython.double)) = cython.cast(cython.pointer(cython.pointer(cython.double)), calloc(max_steps, cython.sizeof(cython.pointer(cython.double))))
    last_action: cython.int = max_steps
    # Rollout till the end of the episode
    while (steps < max_steps) and (steps * env.num_steps < env.max_steps) and (not is_terminated(env, steps)):
        # Select actions according to rollout policy
        action: cython.pointer(cython.double) = policy(rollout_params, state)
        i: cython.Py_ssize_t
        if noise:
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
        #     action[i] = gsl_ran_gaussian(rng, 0.2)

        total_reward += step(env, action)
        free(state)
        # free(action)
        actions[last_action - 1] = action
        last_action -= 1
        state = get_state(env)
        steps += 1
    free(state)
    if isnan(total_reward):
        total_reward = 0.0

    return VGRolloutReturn(actions=actions, last_action=last_action, rtn=total_reward)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def vg_rtn(actions: cython.pointer(cython.pointer(cython.double)), last_action: cython.int, max_steps: cython.int, env: MujocoEnv) -> cython.double:
    total_reward: cython.double = 0.0
    i: cython.int = last_action
    while i < max_steps:
        total_reward += step(env, actions[i])
        i += 1
    if isnan(total_reward):
        total_reward = 0.0
    return total_reward

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def vg_simulate(node: cython.pointer(VGNode), env: MujocoEnv, rng: cython.pointer(gsl_rng), current_depth: cython.int, max_depth: cython.int, rollout_params: PolicyParams) -> VGSimReturn:
    # Select Action
    vgsareturn: VGSAReturn = vg_select_action(node, env, rng)

    rtn: cython.double = vgsareturn.next_node.parent_reward
    depth: cython.int = 1
    actions: cython.pointer(cython.pointer(cython.double)) = cython.NULL
    last_action: cython.int = 0
    if vgsareturn.rollout:
        # TODO: Reset controller to vgsareturn.next_node.mjstate
        vg_roll_return = vg_rollout(current_depth, max_depth, env, rollout_params, rng, False)
        rtn += vg_roll_return.rtn
        actions = vg_roll_return.actions
        last_action = vg_roll_return.last_action

    else:
        vg_sim_return: VGSimReturn = vg_simulate(vgsareturn.next_node, env, rng, current_depth+1, max_depth, rollout_params)
        depth += vg_sim_return.depth
        rtn += vg_sim_return.rtn
        actions = vg_sim_return.actions
        last_action = vg_sim_return.last_action

    omp_set_lock(cython.address(node.access_lock))
    node.num_visitations += 1
    if vgsareturn.rollout:
        vgsareturn.next_node.num_visitations += 1
    vgsareturn.next_node.parent_q_value += (rtn - vgsareturn.next_node.parent_q_value) / vgsareturn.next_node.num_visitations

    a_j: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size, cython.sizeof(cython.double)))
    g_j: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size, cython.sizeof(cython.double)))

    actions[last_action - 1] = a_j
    j: cython.Py_ssize_t
    for j in range(env.action_size):
        memcpy(a_j, vgsareturn.next_node.parent_action, cython.sizeof(cython.double) * env.action_size)
        a_j[j] += 1e-7
        # TODO: Reset controller to node.mjstate
        g_j[j] = (vg_rtn(actions, last_action - 1, max_depth, env) - rtn) / 1e-7

    cblas_daxpy(env.action_size, 0.01, g_j, 1, vgsareturn.next_node.parent_action, 1)

    # Clipping
    action_norm: cython.double = sqrt(cblas_ddot(env.action_size, vgsareturn.next_node.parent_action, 1, vgsareturn.next_node.parent_action, 1))
    if action_norm > 0.5:
        cblas_dscal(env.action_size, 0.5/action_norm, vgsareturn.next_node.parent_action, 1)

    omp_unset_lock(cython.address(node.access_lock))
    free(a_j)
    free(g_j)

    actions[last_action - 1] = vgsareturn.next_node.parent_action
    last_action -= 1

    return VGSimReturn(depth=depth, rtn=rtn, actions=actions, last_action=last_action)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def vg_mcts_job(j: cython.Py_ssize_t, node: cython.pointer(VGNode), max_depth: cython.int, env: MujocoEnv, rollout_params: PolicyParams) -> cython.int:


    # Saving the original state of the environment and will be restored later. The MCTS routines change env.data to reset state to an arbitrary timestep
    original_data: cython.pointer(mjData) = env.data
    env.data = mj_copyData(cython.NULL, env.model, original_data)

    # Initializing a Random Number Generator
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, cython.cast(cython.int, j))

    vg_sim_return: VGSimReturn = vg_simulate(node, env, rng, 0, max_depth, rollout_params)
    depth: cython.int = vg_sim_return.depth

    # Free allocated memory
    i: cython.int = vg_sim_return.last_action
    while i < max_depth:
        free(vg_sim_return.actions[i])
        i += 1
    free(vg_sim_return.actions)

    gsl_rng_free(rng)
    mj_deleteData(env.data)
    return depth


@cython.cfunc
def vg_mcts(num_simulations: cython.int, root: cython.pointer(VGNode), max_depth: cython.int, env: MujocoEnv, rollout_params: PolicyParams) -> cython.int:
    i: cython.Py_ssize_t
    depths: cython.pointer(cython.int) = cython.cast(cython.pointer(cython.int),
                                                     calloc(num_simulations, cython.sizeof(cython.int)))
    for i in range(num_simulations):#, nogil=True):
        # print(f"Sim: {i}")
        depths[i] = vg_mcts_job(i, root, max_depth, env, rollout_params)
        # print(f"Depth: {depths[i]}")
    max_depth: cython.int = 0
    for i in range(num_simulations):
        if depths[i] > max_depth:
            max_depth = depths[i]
    free(depths)
    return max_depth


# =================================== DRIVERS ==========================

def driver_simple_rollout(env_name, weightT, bias):
    env_dict = {"ant": {"env_id": 0, "xml_path": "./env_xmls/ant.xml".encode(), "step_skip": 5, "max_steps": 5000},
                "reacher": {"env_id": 1, "xml_path": "./env_xmls/reacher.xml".encode(), "step_skip": 2,
                            "max_steps": 100},
                "inverted_pendulum": {"env_id": 2, "xml_path": "./env_xmls/inverted_pendulum.xml".encode(),
                                      "step_skip": 2, "max_steps": 2000},
                "pusher": {"env_id": 3, "xml_path": "./env_xmls/pusher.xml".encode(), "step_skip": 5, "max_steps": 500}}
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

    params: PolicyParams = PolicyParams(k=env.action_size, n=env.state_size, w=w, b=b)
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, 3)
    reset_env(env, rng)

    i = 0
    num_kernels: cython.int = 36
    init_cov: cython.double = 3.0
    kernel_cov: cython.double = 0.005
    replace_every_iterations: cython.int = 20
    num_rollouts: cython.int = 200

    total_reward: cython.double = 0.0
    state: cython.pointer(cython.double) = get_state(env)
    print("State:")
    for j in range(env.state_size):
        print(f"{round(state[j], 3)}", end=", ")
    print("")
    while not is_terminated(env, i):
        print(f"Step: {i}")
        start = time.perf_counter_ns()
        node: cython.pointer(MKDNode) = mkd_create_tree_node(env, i, cython.NULL, 0.0, False, num_kernels,
                                                             env.action_size, replace_every_iterations, num_rollouts,
                                                             init_cov, kernel_cov)

        simple_rollout(num_rollouts, node, params, env, rng)

        end = time.perf_counter_ns()
        print(f"Time: {(end - start) / 1e6} ms")
        j: cython.Py_ssize_t
        print("W:")
        for j in range(node.num_kernels):
            print(f"{round(node.w[j], 3)}, ", end="")
        print("")
        print("N:")
        for j in range(node.num_kernels):
            print(f"{node.n[j]}, ", end="")
        print("")
        print("P:")
        for j in range(node.num_kernels):
            print(f"{round(node.pi[j], 3)}, ", end="")
        print("")

        max: cython.double = node.w[0]
        selected_kernel: cython.Py_ssize_t = 0
        for j in range(node.num_kernels):
            if max < node.w[j]:
                max = node.w[j]
                selected_kernel = j
        print(f"Selected Kernel: {selected_kernel}")
        print("Action: ")
        for j in range(env.action_size):
            print(f"{round(node.mu[cython.cast(cython.int, selected_kernel) * env.action_size + j], 3)}", end=", ")
        print("")
        # state: cython.pointer(cython.double) = get_state(env)
        action: cython.pointer(cython.double) = policy(params, state)
        free(state)
        print("Policy Action:")
        for j in range(env.action_size):
            print(f"{round(action[j], 3)}", end=", ")
        print("")

        # CAUTION: For Reacher only
        if (i + 1) % 10 == 0 and env_name == "reacher":
            plot_after_action(env, node, params)

        total_reward += step(env, cython.address(node.mu[cython.cast(cython.int, selected_kernel) * env.action_size]))
        free(action)
        state: cython.pointer(cython.double) = get_state(env)
        print("State:")
        for j in range(env.state_size):
            print(f"{round(state[j], 3)}", end=", ")
        print("")
        print(f"Reward Collected: {total_reward}")
        mkd_free_tree_node(node)
        i += 1
    free(params.w)
    free(params.b)
    free_env(env)
    gsl_rng_free(rng)


def driver(env_name, weightT, bias):
    env_dict = {"ant": {"env_id": 0, "xml_path": "./env_xmls/ant.xml".encode(), "step_skip": 5, "max_steps": 5000},
                "reacher": {"env_id": 1, "xml_path": "./env_xmls/reacher.xml".encode(), "step_skip": 2,
                            "max_steps": 100},
                "inverted_pendulum": {"env_id": 2, "xml_path": "./env_xmls/inverted_pendulum.xml".encode(),
                                      "step_skip": 2, "max_steps": 2000},
                "pusher": {"env_id": 3, "xml_path": "./env_xmls/pusher.xml".encode(), "step_skip": 5, "max_steps": 500}}
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

    params: PolicyParams = PolicyParams(k=env.action_size, n=env.state_size, w=w, b=b)
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, 3)
    reset_env(env, rng)

    i = 0
    num_kernels: cython.int = 56
    replace_every_iterations: cython.int = 50
    num_visitations: cython.int = 500
    init_cov: cython.double = 3.0
    kernel_cov: cython.double = 0.005
    root: cython.pointer(MKDNode) = mkd_create_tree_node(env, i, cython.NULL, 0, False, num_kernels,
                                                         env.action_size, replace_every_iterations, num_visitations,
                                                         init_cov, kernel_cov)
    total_reward: cython.double = 0.0

    state: cython.pointer(cython.double) = get_state(env)
    print("State:")
    for j in range(env.state_size):
        print(f"{round(state[j], 3)}", end=", ")
    print("")
    while not is_terminated(env, i):
        print(f"Step: {i}")
        start = time.perf_counter_ns()
        print("Depth:", mkd_mcts(5000, root, 100, env, params))
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
        # state: cython.pointer(cython.double) = get_state(env)
        action: cython.pointer(cython.double) = policy(params, state)
        free(state)
        print("Policy Action:")
        for j in range(env.action_size):
            print(f"{round(action[j], 3)}", end=", ")
        print("")
        free(action)

        # CAUTION: For Reacher only
        if (i + 1) % 10 == 0 and env_name == "reacher":
            plot_after_action(env, root, params)

        total_reward += step(env, cython.address(root.mu[cython.cast(cython.int, selected_kernel) * env.action_size]))
        if root.children == cython.NULL:
            root.iterations_left = 0
            mkd_expand_node(root, env, params, rng)
        node: cython.pointer(MKDNode) = root.children[selected_kernel]
        root.children[selected_kernel] = cython.NULL
        mkd_free_tree_node(root)
        root = node

        state: cython.pointer(cython.double) = get_state(env)
        print("State:")
        for j in range(env.state_size):
            print(f"{round(state[j], 3)}", end=", ")
        print("")
        print(f"Reward Collected: {total_reward}")

        i += 1

    mkd_free_tree_node(root)
    free(params.w)
    free(params.b)
    free_env(env)
    gsl_rng_free(rng)


def driver_vg(env_name, weightT, bias):
    env_dict = {"ant": {"env_id": 0, "xml_path": "./env_xmls/ant.xml".encode(), "step_skip": 5, "max_steps": 5000},
                "reacher": {"env_id": 1, "xml_path": "./env_xmls/reacher.xml".encode(), "step_skip": 2,
                            "max_steps": 100},
                "inverted_pendulum": {"env_id": 2, "xml_path": "./env_xmls/inverted_pendulum.xml".encode(),
                                      "step_skip": 2, "max_steps": 2000},
                "pusher": {"env_id": 3, "xml_path": "./env_xmls/pusher.xml".encode(), "step_skip": 5, "max_steps": 500}}
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

    params: PolicyParams = PolicyParams(k=env.action_size, n=env.state_size, w=w, b=b)
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, 3)
    reset_env(env, rng)

    i = 0
    total_reward: cython.double = 0.0

    state: cython.pointer(cython.double) = get_state(env)
    print("State:")
    for j in range(env.state_size):
        print(f"{round(state[j], 3)}", end=", ")
    print("")
    while not is_terminated(env, i):
        # Create Root Node
        root: cython.pointer(VGNode) = vg_create_tree_node(env, i, cython.NULL, 0, cython.NULL, False, env.action_size, 0.5)

        # MCTS Job
        print(f"Step: {i}")
        start = time.perf_counter_ns()
        print("Depth:", vg_mcts(10000, root, 100, env, params))
        end = time.perf_counter_ns()
        print(f"Time: {(end - start) / 1e6} ms")

        # Select and Take action
        j: cython.Py_ssize_t
        selected_action: cython.pointer(cython.double) = root.children.parent_action
        max_q: cython.double = root.children.parent_q_value
        child: cython.pointer(VGNode) = root.children
        for j in range(root.num_children):
            if child.parent_q_value > max_q:
                selected_action = child.parent_action
                max_q = child.parent_q_value
            child = child.next
        print("Action: ")
        for j in range(env.action_size):
            print(f"{round(selected_action[j], 3)}", end=", ")
        print("")
        # state: cython.pointer(cython.double) = get_state(env)
        action: cython.pointer(cython.double) = policy(params, state)
        free(state)
        print("Policy Action:")
        for j in range(env.action_size):
            print(f"{round(action[j], 3)}", end=", ")
        print("")
        free(action)

        total_reward += step(env, selected_action)

        # Free Root Node
        vg_free_tree_node(root)

        state: cython.pointer(cython.double) = get_state(env)
        print("State:")
        for j in range(env.state_size):
            print(f"{round(state[j], 3)}", end=", ")
        print("")
        print(f"Reward Collected: {total_reward}")

    free(params.w)
    free(params.b)
    free_env(env)
    gsl_rng_free(rng)