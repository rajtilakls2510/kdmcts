import cython
from cython.cimports.mujoco_envs import MujocoEnv, PolicyParams, step, get_state, set_state, \
    is_terminated, policy, set_action
from cython.cimports.libc.stdlib import calloc, free
from cython.cimports.libc.math import pow, sqrt, exp
from cython.cimports.openmp import omp_lock_t, omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock
from cython.cimports.gsl import gsl_rng_uniform, gsl_rng_type, gsl_rng_default, \
    gsl_rng_alloc, gsl_rng_set, gsl_rng, gsl_rng_free, gsl_ran_flat, gsl_ran_gaussian
from cython.cimports.gsl import cblas_dgemm, CblasRowMajor, CblasNoTrans, CblasTrans, cblas_daxpy, \
    cblas_ddot, cblas_dcopy, cblas_dscal, cblas_dger
from cython.cimports.mujoco import mjData, mj_copyData, mj_deleteData


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
def mkd_create_tree_node(state: cython.pointer(cython.double), current_step: cython.int,
                         parent: cython.pointer(MKDNode), parent_reward: cython.double, terminal: cython.bint,
                         num_kernels: cython.int,
                         action_dim: cython.int, iterations: cython.int,
                         samples_per_iteration: cython.int) -> cython.pointer(
    MKDNode):
    node: cython.pointer(MKDNode) = cython.cast(cython.pointer(MKDNode), calloc(1, cython.sizeof(MKDNode)))
    node.state = state
    node.current_step = current_step
    node.parent = parent
    node.parent_reward = parent_reward
    omp_init_lock(cython.address(node.access_lock))
    node.terminal = terminal
    node.num_kernels = num_kernels
    node.action_dim = action_dim
    node.params_initialized = False
    if not terminal:
        node.pi = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
        node.w = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
        node.n = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
        node.mu = cython.cast(cython.pointer(cython.double),
                              calloc(num_kernels * action_dim, cython.sizeof(cython.double)))
        node.cov = cython.cast(cython.pointer(cython.double),
                               calloc(num_kernels * action_dim * action_dim, cython.sizeof(cython.double)))
        node.iterations = iterations
        node.samples_per_iteration = samples_per_iteration
    node.expanded = False
    node.children = cython.NULL
    return node


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_free_tree_node(node: cython.pointer(MKDNode)) -> cython.void:
    if node != cython.NULL:
        free(node.state)
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

        # Sampling an action from pi
        rm: cython.double = gsl_rng_uniform(rng)
        selected_action_index: cython.Py_ssize_t = 0
        sum: cython.double = 0
        for selected_action_index in range(env.action_size):
            sum += node.pi[selected_action_index]
            if rm < sum:
                break

        # Note down the index of the action taken and continue selection to child
        move_indices[move_last[0]] = selected_action_index
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
            for i in range(node.num_kernels):
                for j in range(node.action_dim):
                    node.mu[i * node.action_dim + j] = gsl_ran_flat(rng, -1.0, 1.0)
                    node.cov[
                        i * node.action_dim * node.action_dim + j * node.action_dim + j] = 30.0  # TODO: Needs to be made a hyper-parameter

            # Perform one rollout per kernel to initialize w and n values
            original_data: cython.pointer(mjData) = env.data
            for i in range(node.num_kernels):
                # Take an action and get next step
                env.data = mj_copyData(cython.NULL, env.model, original_data)
                set_state(env, node.state)
                r: cython.double = step(env, cython.address(node.mu[i * env.action_size]))
                next_state: cython.pointer(cython.double) = get_state(env)
                # Rollout from this next state
                node.w[i] = r + mkd_rollout(next_state, node.current_step + 1, env, rollout_params, rng)
                node.n[i] += 1
                mj_deleteData(env.data)
            env.data = original_data
        else:
            node.expanded = True

    if not node.expanded:
        # Expansion is only performed when the leaf node has been visited a certain number of times
        if node.samples_per_iteration == 0 and node.iterations == 0:
            node.children = cython.cast(cython.pointer(cython.pointer(MKDNode)),
                                        calloc(node.num_kernels, cython.sizeof(cython.pointer(MKDNode))))
            original_data: cython.pointer(mjData) = env.data
            i: cython.Py_ssize_t
            for i in range(node.num_kernels):
                env.data = mj_copyData(cython.NULL, env.model, original_data)
                set_state(env, node.state)
                r: cython.double = step(env, cython.address(node.mu[i * env.action_size]))
                next_state: cython.pointer(cython.double) = get_state(env)
                node.children[i] = mkd_create_tree_node(next_state, node.current_step + 1, node, r,
                                                        is_terminated(env, node.current_step + 1), node.num_kernels,
                                                        env.action_size, 10, 1000)
                # TODO: Change iterations and samples per iteration to hyper-parameters
                mj_deleteData(env.data)
            env.data = original_data  # Don't need to restore the original value because env is not a pointer but it is a good practice
            node.expanded = True
        else:
            # Else, increase visitation count
            if node.samples_per_iteration == 0:
                node.iterations -= 1
                node.samples_per_iteration = 1000
            else:
                node.samples_per_iteration -= 1


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
        # LOCK IS NOT RELEASED UNTIL pi FOR node IS CALCULATED. THIS IS TO MAKE SURE OTHER THREADS WAIT FOR EXPANSION COMPLETION
    else:
        # Else, release the lock immediately and resume from selection phase as another thread has already expanded this node
        if not node.terminal:
            node = mkd_selection(node, move_indices, move_last, env, rng)
            node = mkd_expansion(node, move_indices, move_last, env, rollout_params, rng)
    return node


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_rollout(state: cython.pointer(cython.double), steps_taken: cython.int, env: MujocoEnv,
                rollout_params: PolicyParams, rng: cython.pointer(gsl_rng)) -> cython.double:
    # BEWARE: state will be freed by this function
    total_reward: cython.double = 0.0
    steps: cython.Py_ssize_t = steps_taken

    # Rollout till the end of the episode
    while (steps * env.num_steps < env.max_steps) and (not is_terminated(env, steps)):
        # Select actions according to rollout policy
        action: cython.pointer(cython.double) = policy(rollout_params, state, env)
        i: cython.Py_ssize_t
        for i in range(env.action_size):
            action[i] += gsl_ran_flat(rng, -0.1, 0.1)  # Adding Exploration Noise
        total_reward += step(env, action)
        free(state)
        free(action)
        state = get_state(env)
        steps += 1
    free(state)
    return total_reward


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_backup(node: cython.pointer(MKDNode), action: cython.pointer(cython.double), rtn: cython.double,
               move_indices: cython.pointer(cython.int),
               move_last: cython.pointer(cython.int), env: MujocoEnv) -> cython.int:
    depth: cython.int = move_last[0]
    omp_set_lock(cython.address(node.access_lock))
    # Kernel of the action
    kmu: cython.pointer(cython.double) = action
    kcov: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                      calloc(env.action_size * env.action_size,
                                                             cython.sizeof(cython.double)))
    i: cython.Py_ssize_t
    for i in range(env.action_size):
        kcov[i * env.action_size + i] = 0.005  # TODO: Make it a hyper-parameters

    # Finding the ideal kernel to merge with using Euclidean distance
    scratch: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                         calloc(env.action_size, cython.sizeof(cython.double)))
    min_kernel_index: cython.int = 0
    cblas_dcopy(env.action_size, cython.address(node.mu[min_kernel_index * env.action_size]), 1, scratch,
                1)  # scratch = node.mu[0]
    cblas_daxpy(env.action_size, -1.0, kmu, 1, scratch, 1)  # scratch = scratch - kmu
    min_sum: cython.double = cblas_ddot(env.action_size, scratch, 1, scratch, 1)  # dot(scratch, scratch)
    for i in range(node.num_kernels):
        cblas_dcopy(env.action_size, cython.address(node.mu[cython.cast(cython.int, i) * env.action_size]), 1, scratch,
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
        mean += (node.w[i] / node.n[i])
    mean /= node.num_kernels
    for i in range(node.num_kernels):  # scratch = (node.w / node.n - mean)
        scratch[i] = (node.w[i] / node.n[i] - mean)
    std: cython.double = cblas_ddot(node.num_kernels, scratch, 1, scratch,
                                    1) / node.num_kernels  # std = dot(scratch, scratch) / node.num_kernels

    # Merging Means, Cov, and Weights
    k_w_1: cython.double = (node.w[min_kernel_index] / node.n[min_kernel_index] - mean) / std
    k_w_2: cython.double = (rtn - mean) / std
    k_w_1 = exp(k_w_1)
    k_w_2 = exp(k_w_2)
    k_w_1 /= (k_w_1 + k_w_2)
    k_w_2 /= (k_w_1 + k_w_2)

    cblas_dcopy(env.action_size, cython.address(node.mu[min_kernel_index * env.action_size]), 1, scratch,
                1)  # scratch = node.mu[min_kernel]
    cblas_dscal(env.action_size, k_w_1, scratch, 1)  # scratch = k_w_1 * scratch
    cblas_daxpy(env.action_size, k_w_2, kmu, 1, scratch, 1)  # scratch = scratch + k_w_2 * kmu

    cblas_daxpy(env.action_size, -1.0, scratch, 1, cython.address(node.mu[min_kernel_index * env.action_size]),
                1)  # node.mu[min_kernel] -= scratch
    cblas_dscal(env.action_size, -1.0, cython.address(node.mu[min_kernel_index * env.action_size]),
                1)  # node.mu[min_kernel] = -node.mu[min_kernel]
    cblas_daxpy(env.action_size, -1.0, scratch, 1, kmu, 1)  # kmu -= scratch
    cblas_dscal(env.action_size, -1.0, kmu, 1)  # kmu = -kmu

    # node.cov[min_kernel] += (node.mu[min_kernel] @ node.mu[min_kernel]^T)
    cblas_dger(CblasRowMajor, env.action_size, env.action_size, 1.0, cython.address(node.mu[min_kernel_index * env.action_size]), 1,
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

    cblas_dcopy(env.action_size, scratch, 1, cython.address(node.mu[min_kernel_index * env.action_size]),
                1)  # node.mu[min_kernel] = scratch

    node.w[min_kernel_index] += rtn
    node.n[min_kernel_index] += 1
    for i in range(node.num_kernels):
        node.pi[i] = node.w[i] / node.n[i]
    softmax(node.pi, node.num_kernels)

    omp_unset_lock(cython.address(node.access_lock))

    # Update parent node statistics
    rtn += node.parent_reward
    node = node.parent
    while move_last[0] > 0:
        move_last[0] -= 1
        omp_set_lock(cython.address(node.access_lock))
        node.n[move_indices[move_last[0]]] += 1
        node.w[move_indices[move_last[0]]] += rtn
        for i in range(node.num_kernels):
            node.pi[i] = node.w[i] / node.n[i]
        softmax(node.pi, node.num_kernels)
        rtn += node.parent_reward
        omp_unset_lock(cython.address(node.access_lock))
        node = node.parent

    free(scratch)
    free(kcov)
    return depth


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_mcts_job(j: cython.Py_ssize_t, root: cython.pointer(MKDNode), max_depth: cython.int, env: MujocoEnv,
                 rollout_params: PolicyParams, seed: cython.int) -> cython.int:
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
    gsl_rng_set(rng, seed)

    # Selection
    node = mkd_selection(node, move_indices, cython.address(move_last), env, rng)

    # Expansion
    node = mkd_expansion(node, move_indices, cython.address(move_last), env, rollout_params, rng)

    # If tree-parallelized, lock might be held on the leaf node depending on whether it was expanded or not. Unlock it.
    omp_unset_lock(cython.address(node.access_lock))

    # Sample action according to current node params and begin rollout from next state
    set_state(env, node.state)
    chosen_kernel: cython.Py_ssize_t = 0
    sum: cython.double = 0.0
    rm: cython.double = gsl_rng_uniform(rng)
    for chosen_kernel in range(node.num_kernels):
        sum += node.pi[chosen_kernel]
        if rm < sum:
            break
    action: cython.pointer(cython.double) = sample_multivariate_gaussian(1, cython.address(
        node.mu[chosen_kernel * env.action_size]),
                                                                         cython.address(node.cov[
                                                                                            chosen_kernel * env.action_size * env.action_size]),
                                                                         env.action_size, rng)
    r: cython.double = step(env, action)
    next_state: cython.pointer(cython.double) = get_state(env)
    rtn: cython.double = r + mkd_rollout(next_state, node.current_step + 1, env, rollout_params, rng)

    # Backup
    depth: cython.int = mkd_backup(node, action, rtn, move_indices, cython.address(move_last), env)

    # Free allocated memory
    free(action)
    free(move_indices)
    gsl_rng_free(rng)
    mj_deleteData(env.data)
    env.data = original_data  # Don't need to restore the original data because env is not a pointer but it is a good practice
    return depth
