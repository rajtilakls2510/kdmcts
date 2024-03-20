import cython
from cython.cimports.libc.stdlib import calloc, free
from cython.cimports.mujoco import mj_loadXML, mjModel, mj_printModel, \
    mjData, mj_makeData, mj_printData, mj_resetData, mj_step, mj_deleteData, \
    mj_deleteModel, mj_copyData
from cython.cimports.gsl import CblasRowMajor, CblasNoTrans, CblasTrans, \
    cblas_dgemv, cblas_dscal, cblas_dcopy, cblas_ddot, gsl_rng, gsl_ran_gaussian, \
    gsl_ran_flat, gsl_rng_type, gsl_rng_default, gsl_rng_alloc, gsl_rng_set, gsl_rng_free
from cython.cimports.libc.math import exp, isfinite


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def create_env(env_id: cython.int, path: cython.pointer(cython.char)) -> MujocoEnv:
    err: cython.char[300]
    model: cython.pointer(mjModel) = mj_loadXML(path, cython.NULL, err, 300)
    data: cython.pointer(mjData) = mj_makeData(model)
    env: MujocoEnv = MujocoEnv(env_id=env_id, model=model, data=data, state_size=0, action_size=0)
    env.state_size = get_state_size(env)
    env.action_size = get_action_size(env)
    return env


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def free_env(env: MujocoEnv) -> cython.void:
    mj_deleteData(env.data)
    mj_deleteModel(env.model)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_state_size(env: MujocoEnv) -> cython.int:
    if env.env_id == 0:
        return get_ant_state_size(env)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_action_size(env: MujocoEnv) -> cython.int:
    if env.env_id == 0:
        return get_ant_action_size(env)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_state(env: MujocoEnv) -> cython.pointer(cython.double):
    if env.env_id == 0:
        return get_ant_state(env)
    return cython.NULL


# @cython.cfunc
# @cython.nogil
# @cython.exceptval(check=False)
# def set_state(env: MujocoEnv, state: cython.pointer(cython.double)) -> cython.void:
#     if env.env_id == 0:
#         set_ant_state(env, state)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_action(env: MujocoEnv) -> cython.pointer(cython.double):
    if env.env_id == 0:
        return get_ant_action(env)
    return cython.NULL

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def set_action(env: MujocoEnv, action: cython.pointer(cython.double)) -> cython.void:
    if env.env_id == 0:
        set_ant_action(env, action)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def reset_env(env: MujocoEnv, rng: cython.pointer(gsl_rng)) -> cython.void:
    mj_resetData(env.model, env.data)
    if env.env_id == 0:
        ant_reset_env(env, rng)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def perform_steps(env: MujocoEnv, num_steps: cython.int) -> cython.void:
    i: cython.Py_ssize_t
    for i in range(num_steps):
        mj_step(env.model, env.data)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def step(env: MujocoEnv, action: cython.pointer(cython.double), num_steps: cython.int) -> cython.double:
    if env.env_id == 0:
        return ant_step(env, action, num_steps)
    return 0.0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def is_terminated(env: MujocoEnv) -> cython.bint:
    if env.env_id == 0:
        return ant_is_terminated(env)
    return True

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def policy(params: PolicyParams, state: cython.pointer(cython.double), env: MujocoEnv) -> cython.pointer(cython.double):
    action: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(params.k, cython.sizeof(cython.double)))

    # action = b
    i: cython.Py_ssize_t
    for i in range(params.k):
        action[i] = params.b[i]

    # action = W.T  @ state + action
    cblas_dgemv(CblasRowMajor, CblasNoTrans, params.k, params.n, 1.0, params.w, params.n, state, 1, 1.0, action, 1)

    # action = Tanh(action)
    cblas_dscal(env.action_size, 2.0, action, 1) # action = 2 * action
    scratch: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(params.k, cython.sizeof(cython.double)))
    cblas_dcopy(params.k, action, 1, scratch, 1) # scratch = action
    for i in range(params.k):
        action[i] = (exp(action[i]) - 1.0) / (exp(scratch[i]) + 1.0)
    free(scratch)
    return action


# ================================= ANT Env Handlers (EnvId: 0) ===============================


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_ant_state_size(env: MujocoEnv) -> cython.int:
    size: cython.int = env.model.nq - 2
    size += env.model.nv
    return size


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_ant_action_size(env: MujocoEnv) -> cython.int:
    return env.model.nu


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_ant_state(env: MujocoEnv) -> cython.pointer(cython.double):
    state: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                       calloc(env.state_size, cython.sizeof(cython.double)))
    i: cython.Py_ssize_t
    for i in range(env.model.nq - 2):
        state[i] = env.data.qpos[i + 2]
    for i in range(env.model.nv):
        state[i + env.model.nq - 2] = env.data.qvel[i]
    return state

#
# @cython.cfunc
# @cython.nogil
# @cython.exceptval(check=False)
# def set_ant_state(env: MujocoEnv, state: cython.pointer(cython.double)) -> cython.void:
#     # Not going to be used most probably
#     i: cython.Py_ssize_t
#     for i in range(env.model.nq - 2):
#         env.data.qpos[i + 2] = state[i]
#     for i in range(env.model.nv):
#         env.data.qvel[i] = state[i + env.model.nq - 2]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_ant_action(env: MujocoEnv) -> cython.pointer(cython.double):
    action: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size, cython.sizeof(cython.double)))

    i: cython.Py_ssize_t
    for i in range(env.model.nu):
        action[i] = env.data.ctrl[i]
    return action


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def set_ant_action(env: MujocoEnv, action: cython.pointer(cython.double)) -> cython.void:
    i: cython.Py_ssize_t
    for i in range(env.model.nu):
        env.data.ctrl[i] = action[i]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def ant_reset_env(env: MujocoEnv, rng: cython.pointer(gsl_rng)) -> cython.void:
    i: cython.Py_ssize_t
    for i in range(env.model.nq):
        env.data.qpos[i] += gsl_ran_flat(rng, -0.1, 0.1)
    for i in range(env.model.nv):
        env.data.qvel[i] = 0.1 * gsl_ran_gaussian(rng, 1.0)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def ant_is_terminated(env: MujocoEnv) -> cython.bint:
    healthy: cython.bint = True
    i: cython.Py_ssize_t
    for i in range(env.model.nq):
        healthy = healthy and isfinite(env.data.qpos[i])
    for i in range(env.model.nv):
        healthy = healthy and isfinite(env.data.qvel[i])

    return (0.2 <= env.data.qpos[0] <= 1.0) and healthy


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def ant_step(env: MujocoEnv, action: cython.pointer(cython.double), num_steps: cython.int) -> cython.double:
    set_action(env, action)
    body_n: cython.int = 1 # TORSO
    previous_x: cython.double = env.data.xpos[body_n * 3 + 0]

    perform_steps(env, num_steps)

    new_x: cython.double = env.data.xpos[body_n * 3 + 0]
    xvel: cython.double = (new_x - previous_x) / (env.model.opt.timestep * num_steps)

    ctrl_cost_weight: cython.double = 0.5
    ctrl_cost: cython.double = ctrl_cost_weight * cblas_ddot(env.action_size, action, 1, action, 1) # np.sum(action ** 2)

    healthy_reward: cython.double = 1.0
    healthy_reward = healthy_reward * ant_is_terminated(env)

    reward: cython.double = xvel + healthy_reward - ctrl_cost
    return reward


def driver(env_name, weightT, bias):
    env_dict = {"ant": {"env_id": 0, "xml_path": "./env_xmls/ant.xml".encode(), "step_skip": 5}}
    env: MujocoEnv = create_env(env_dict[env_name]["env_id"], env_dict[env_name]["xml_path"])
    print(env.env_id, env.state_size, env.action_size)

    w: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size*env.state_size, cython.sizeof(cython.double)))
    b: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size, cython.sizeof(cython.double)))

    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    for i in range(env.action_size):
        for j in range(env.state_size):
            w[i*env.state_size+j] = weightT[j][i]
    for j in range(env.action_size):
        b[j] = bias[j]
    print("W:")
    for i in range(env.action_size):
        for j in range(env.state_size):
            print(w[i * env.state_size + j], end=", ")
        print("")
    print("Bias:")
    for j in range(env.action_size):
        print(b[j], end=", ")
    print("")

    params: PolicyParams = PolicyParams(k=env.action_size, n=env.state_size, w=w, b=b)
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, 6)
    reset_env(env, rng)
    total_reward: cython.double = 0.0
    for j in range(1000):
        state: cython.pointer(cython.double) = get_state(env)
        print(f"{j} State: ")
        for i in range(env.state_size):
            print(state[i], end=", ")
        print("")
        action: cython.pointer(cython.double) = policy(params, state, env)
        print("Action: ")
        for i in range(env.action_size):
            print(action[i], end=", ")
        print("")
        reward: cython.double = step(env, action, env_dict[env_name]["step_skip"])
        terminated: cython.bint = is_terminated(env)
        total_reward += reward
        print("Reward: ", reward, "Terminated: ", terminated, "Total Reward: ", total_reward)
        free(action)
        free(state)
    free(params.w)
    free(params.b)
    free_env(env)
    gsl_rng_free(rng)