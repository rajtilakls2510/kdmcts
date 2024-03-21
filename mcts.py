import cython
from cython.cimports.mujoco_envs import MujocoEnv, PolicyParams
from cython.cimports.libc.stdlib import calloc, free
from cython.cimports.openmp import omp_lock_t, omp_init_lock, omp_destroy_lock, \
    omp_set_lock, omp_unset_lock


# =================================== MKD MCTS ==========================
# MKDNode = None

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_create_tree_node(state: cython.pointer(cython.double), parent: cython.pointer(MKDNode), num_kernels: cython.int,
                         action_dim: cython.int, terminal: cython.bint, iterations: cython.int,
                         samples_per_iteration: cython.int) -> cython.pointer(
    MKDNode):
    node: cython.pointer(MKDNode) = cython.cast(cython.pointer(MKDNode), calloc(1, cython.sizeof(MKDNode)))
    node.state = state
    node.parent = parent
    omp_init_lock(cython.address(node.access_lock))
    node.terminal = terminal
    node.num_kernels = num_kernels
    node.action_dim = action_dim
    node.pi = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
    node.w = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
    node.n = cython.cast(cython.pointer(cython.double), calloc(num_kernels, cython.sizeof(cython.double)))
    node.mu = cython.cast(cython.pointer(cython.double), calloc(num_kernels * action_dim, cython.sizeof(cython.double)))
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
                  move_last: cython.pointer(cython.int), env: MujocoEnv) -> cython.pointer(MKDNode):
    while node.expanded and not node.terminal:
        # TODO: Use node.pi to select an action using softmax
        pass
    return node


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_expand_node(node: cython.pointer(MKDNode), env: MujocoEnv, rollout_params: PolicyParams) -> cython.void:
    # TODO: Allocate pi, w, n, mu, cov and play some rollouts to initialize these parameters
    # TODO: Do not expand everytime
    pass


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_expansion(node: cython.pointer(MKDNode), move_indices: cython.pointer(cython.int),
                  move_last: cython.pointer(cython.int), env: MujocoEnv,
                  rollout_params: PolicyParams) -> cython.pointer(MKDNode):
    omp_set_lock(cython.address(node.access_lock))

    # If the node has not been expanded by another thread, expand it
    if not node.expanded:
        mkd_expand_node(node, env, rollout_params)
        # LOCK IS NOT RELEASED UNTIL pi FOR node IS CALCULATED. THIS IS TO MAKE SURE OTHER THREADS WAIT FOR EXPANSION COMPLETION
    else:
        # Else, release the lock immediately and resume from seleciton phase
        if not node.terminal:
            node = mkd_selection(node, move_indices, move_last, env)
            node = mkd_expansion(node, move_indices, move_last, env, rollout_params)
    return node


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_rollout(node: cython.pointer(MKDNode), env: MujocoEnv, rollout_params: PolicyParams) -> cython.double:
    # TODO: Rollout
    pass


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_backup(node: cython.pointer(MKDNode), rtn: cython.double, move_indices: cython.pointer(cython.int),
               move_last: cython.pointer(cython.int)) -> cython.int:
    depth: cython.int = move_last[0]

    # TODO: Backup

    return depth


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def mkd_mcts_job(j: cython.Py_ssize_t, root: cython.pointer(MKDNode), max_depth: cython.int, env: MujocoEnv,
                 rollout_params: PolicyParams) -> cython.int:
    node: cython.pointer(MKDNode) = root
    move_indices: cython.pointer(cython.int) = cython.cast(cython.pointer(cython.int),
                                                           calloc(max_depth, cython.sizeof(cython.int)))
    move_last: cython.int = 0
    node = mkd_selection(node, move_indices, cython.address(move_last), env)
    node = mkd_expansion(node, move_indices, cython.address(move_last), env, rollout_params)

    rtn: cython.double = mkd_rollout(node, env, rollout_params)

    omp_unset_lock(cython.address(node.access_lock))

    depth: cython.int = mkd_backup(node, rtn, move_indices, cython.address(move_last))

    free(move_indices)
    return depth
