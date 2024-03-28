# cython: cdivision = True

import cython
from cython.cimports.libc.stdlib import calloc, free
from cython.cimports.mujoco import mj_loadXML, mjModel, mjData, mj_makeData, \
    mj_resetData, mj_step, mj_deleteData, mj_deleteModel, mj_copyData, mj_step1, \
    mj_step2, mj_stateSize, mjSTATE_INTEGRATION, mj_getState, mj_setState, mjtNum, \
    mj_id2name
from cython.cimports.gsl import CblasRowMajor, CblasNoTrans, CblasTrans, \
    cblas_dgemv, cblas_dscal, cblas_dcopy, cblas_ddot, gsl_rng, gsl_ran_gaussian, \
    gsl_ran_flat, gsl_rng_type, gsl_rng_default, gsl_rng_alloc, gsl_rng_set, gsl_rng_free
from cython.cimports.libc.math import exp, isfinite, isnan, sin, cos, sqrt
import time

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def create_env(env_id: cython.int, path: cython.pointer(cython.char), num_steps: cython.int, max_steps: cython.int) -> MujocoEnv:
    err: cython.char[300]
    model: cython.pointer(mjModel) = mj_loadXML(path, cython.NULL, err, 300)
    data: cython.pointer(mjData) = mj_makeData(model)
    # with cython.gil:
    #     for i in range(10):
    #         print(f"Name: {i} {mj_id2name(model, 1, i).decode()}")
    env: MujocoEnv = MujocoEnv(env_id=env_id, model=model, data=data)
    env.mj_state_size = get_mj_state_size(env)
    env.state_size = get_state_size(env)
    env.action_size = get_action_size(env)
    env.num_steps = num_steps
    env.max_steps = max_steps
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
def get_mj_state_size(env: MujocoEnv) -> cython.int:
    return mj_stateSize(env.model, mjSTATE_INTEGRATION)# + env.model.nv + env.model.na


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_mj_state(env: MujocoEnv) -> cython.pointer(mjtNum):
    mj_state: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.mj_state_size, cython.sizeof(cython.double)))
    mj_getState(env.model, env.data, mj_state, mjSTATE_INTEGRATION)
    return mj_state


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def set_mj_state(env: MujocoEnv, mj_state: cython.pointer(mjtNum)) -> cython.void:
    mj_setState(env.model, env.data, mj_state, mjSTATE_INTEGRATION)

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_state_size(env: MujocoEnv) -> cython.int:
    if env.env_id == 0:
        return get_ant_state_size(env)
    elif env.env_id == 1:
        return get_reacher_state_size(env)
    return 0

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_action_size(env: MujocoEnv) -> cython.int:
    if env.env_id == 0:
        return get_ant_action_size(env)
    elif env.env_id == 1:
        return get_reacher_action_size(env)
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_state(env: MujocoEnv) -> cython.pointer(cython.double):
    if env.env_id == 0:
        return get_ant_state(env)
    elif env.env_id == 1:
        return get_reacher_state(env)
    return cython.NULL


# @cython.cfunc
# @cython.nogil
# @cython.exceptval(check=False)
# def set_state(env: MujocoEnv, state: cython.pointer(cython.double)) -> cython.void:
#     if env.env_id == 0:
#         set_ant_state(env, state)
#     elif env.env_id == 1:
#         set_reacher_state(env, state)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_action(env: MujocoEnv) -> cython.pointer(cython.double):
    if env.env_id == 0:
        return get_ant_action(env)
    elif env.env_id == 1:
        return get_reacher_action(env)
    return cython.NULL

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def set_action(env: MujocoEnv, action: cython.pointer(cython.double)) -> cython.void:
    if env.env_id == 0:
        set_ant_action(env, action)
    elif env.env_id == 1:
        set_reacher_action(env, action)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def reset_env(env: MujocoEnv, rng: cython.pointer(gsl_rng)) -> cython.void:
    mj_resetData(env.model, env.data)
    if env.env_id == 0:
        ant_reset_env(env, rng)
    elif env.env_id == 1:
        reacher_reset_env(env, rng)


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
def step(env: MujocoEnv, action: cython.pointer(cython.double)) -> cython.double:
    if env.env_id == 0:
        return ant_step(env, action)
    elif env.env_id == 1:
        return reacher_step(env, action)
    return 0.0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def is_terminated(env: MujocoEnv, steps_taken: cython.int) -> cython.bint:
    if env.env_id == 0:
        return ant_is_terminated(env, steps_taken)
    elif env.env_id == 1:
        return reacher_is_terminated(env, steps_taken)
    return True

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def policy(params: PolicyParams, state: cython.pointer(cython.double)) -> cython.pointer(cython.double):
    action: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(params.k, cython.sizeof(cython.double)))

    # action = b
    i: cython.Py_ssize_t
    for i in range(params.k):
        action[i] = params.b[i]

    # action = W.T @ state + action
    cblas_dgemv(CblasRowMajor, CblasNoTrans, params.k, params.n, 1.0, params.w, params.n, state, 1, 1.0, action, 1)

    # action = Tanh(action)
    cblas_dscal(params.k, 2.0, action, 1) # action = 2 * action
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


# @cython.cfunc
# @cython.nogil
# @cython.exceptval(check=False)
# def set_ant_state(env: MujocoEnv, state: cython.pointer(cython.double)) -> cython.void:
#     # Used only for rollouts when the simulator is reset at a particular state
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
def ant_is_healthy(env: MujocoEnv) -> cython.bint:
    healthy: cython.bint = True
    i: cython.Py_ssize_t
    for i in range(env.model.nq):
        healthy = healthy and isfinite(env.data.qpos[i])
    for i in range(env.model.nv):
        healthy = healthy and isfinite(env.data.qvel[i])
    return (0.2 <= env.data.qpos[2] <= 1.0) and healthy

@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def ant_is_terminated(env: MujocoEnv, steps_taken: cython.int) -> cython.bint:
    return not ant_is_healthy(env) or (steps_taken * env.num_steps >= env.max_steps)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def ant_step(env: MujocoEnv, action: cython.pointer(cython.double)) -> cython.double:
    body_n: cython.int = 1 # TORSO
    previous_x: cython.double = env.data.xpos[body_n * 3 + 0]
    i: cython.Py_ssize_t
    action_aux: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size, cython.sizeof(cython.double)))
    for i in range(env.action_size):
        if action[i] < -1.0:
            action_aux[i] = -1.0
        elif action[i] > 1.0:
            action_aux[i] = 1.0
        else:
            action_aux[i] = action[i]
    mj_step1(env.model, env.data)
    set_action(env, action_aux)

    mj_step2(env.model, env.data)
    perform_steps(env, env.num_steps-1)

    new_x: cython.double = env.data.xpos[body_n * 3 + 0]
    xvel: cython.double = (new_x - previous_x) / (env.model.opt.timestep * env.num_steps)

    ctrl_cost_weight: cython.double = 0.5
    ctrl_cost: cython.double = ctrl_cost_weight * cblas_ddot(env.action_size, action_aux, 1, action_aux, 1) # np.sum(action ** 2)

    healthy_reward: cython.double = 1.0
    healthy_reward = healthy_reward * ant_is_healthy(env)

    reward: cython.double = xvel + healthy_reward - ctrl_cost
    free(action_aux)
    if isnan(reward):
        reward = 0.0
    return reward


# ================================= REACHER Env Handlers (EnvId: 1) ===============================


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_reacher_state_size(env: MujocoEnv) -> cython.int:
    size: cython.int = 2 + 2
    size += env.model.nq - 2
    size += env.model.nv - 2
    size += 2
    return size


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_reacher_action_size(env: MujocoEnv) -> cython.int:
    return env.model.nu


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_reacher_state(env: MujocoEnv) -> cython.pointer(cython.double):
    state: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double),
                                                       calloc(env.state_size, cython.sizeof(cython.double)))
    i: cython.Py_ssize_t
    for i in range(2):
        state[i] = cos(env.data.qpos[i])
    for i in range(2):
        state[i + 2] = sin(env.data.qpos[i])
    for i in range(env.model.nq - 2):
        state[i + 4] = env.data.qpos[i + 2]
    for i in range(env.model.nv - 2):
        state[i + env.model.nq - 2 + 4] = env.data.qvel[i]
    for i in range(2):
        state[i + env.model.nv - 2 + env.model.nq - 2 + 4] = env.data.xpos[3 * 3 + i] - env.data.xpos[4 * 3 + i]    # fingertip - target
    return state


# @cython.cfunc
# @cython.nogil
# @cython.exceptval(check=False)
# def set_reacher_state(env: MujocoEnv, state: cython.pointer(cython.double)) -> cython.void:
#     # Used only for rollouts when the simulator is reset at a particular state
#     i: cython.Py_ssize_t
#
#     for i in range(env.model.nq - 2):
#         env.data.qpos[i + 2] = state[i]
#     for i in range(env.model.nv):
#         env.data.qvel[i] = state[i + env.model.nq - 2]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def get_reacher_action(env: MujocoEnv) -> cython.pointer(cython.double):
    action: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size, cython.sizeof(cython.double)))
    i: cython.Py_ssize_t
    for i in range(env.model.nu):
        action[i] = env.data.ctrl[i]
    return action


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def set_reacher_action(env: MujocoEnv, action: cython.pointer(cython.double)) -> cython.void:
    i: cython.Py_ssize_t
    for i in range(env.model.nu):
        env.data.ctrl[i] = action[i]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def reacher_reset_env(env: MujocoEnv, rng: cython.pointer(gsl_rng)) -> cython.void:
    i: cython.Py_ssize_t
    for i in range(env.model.nq):
        env.data.qpos[i] += gsl_ran_flat(rng, -0.1, 0.1)
    while True:
        goal_0: cython.double = gsl_ran_flat(rng, -0.2, 0.2)
        goal_1: cython.double = gsl_ran_flat(rng, -0.2, 0.2)
        if sqrt(goal_0 * goal_0 + goal_1 * goal_1) < 0.2:
            break
    env.data.qpos[env.model.nq - 2] = goal_0
    env.data.qpos[env.model.nq - 1] = goal_1
    for i in range(env.model.nv - 2):
        env.data.qvel[i] = gsl_ran_flat(rng, -0.005, 0.005)
    env.data.qvel[env.model.nv - 2] = 0
    env.data.qvel[env.model.nv - 1] = 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def reacher_is_terminated(env: MujocoEnv, steps_taken: cython.int) -> cython.bint:
    return steps_taken * env.num_steps >= env.max_steps


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def reacher_step(env: MujocoEnv, action: cython.pointer(cython.double)) -> cython.double:

    i: cython.Py_ssize_t
    action_aux: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size, cython.sizeof(cython.double)))
    for i in range(env.action_size):
        if action[i] < -1.0:
            action_aux[i] = -1.0
        elif action[i] > 1.0:
            action_aux[i] = 1.0
        else:
            action_aux[i] = action[i]
    mj_step1(env.model, env.data)
    set_action(env, action_aux)

    mj_step2(env.model, env.data)
    perform_steps(env, env.num_steps-1)
    vec_sum: cython.double = 0.0
    for i in range(3):
        vec_sum += (env.data.xpos[3 * 3 + i] - env.data.xpos[4 * 3 + i])**2 # fingertip - target

    reward_dist_weight: cython.double = 1.0
    reward: cython.double = -sqrt(vec_sum) * reward_dist_weight

    reward_control_weight: cython.double = 1.0
    reward_ctrl: cython.double = 0.0
    for i in range(env.action_size):
        reward_ctrl += action_aux[i]**2
    reward -= reward_ctrl * reward_control_weight

    free(action_aux)
    if isnan(reward):
        reward = 0.0
    return reward


@cython.cclass
class MujocoPyEnv:
    env_struct: MujocoEnv
    rng: cython.pointer(gsl_rng)

    def __init__(self, env_name: str, seed: int = 5):
        self.env_dict = {"ant": {"env_id": 0, "xml_path": "./env_xmls/ant.xml".encode(), "step_skip": 5}}
        self.env_struct = create_env(self.env_dict[env_name]["env_id"], self.env_dict[env_name]["xml_path"], self.env_dict[env_name]["step_skip"], self.env_dict[env_name]["max_steps"])
        T: cython.pointer(gsl_rng_type) = gsl_rng_default
        self.rng = gsl_rng_alloc(T)
        gsl_rng_set(self.rng, seed)

    def __dealloc__(self):
        free_env(self.env_struct)
        gsl_rng_free(self.rng)

    #
    # def reset(self):
    #     reset_env(self.env_struct, self.rng)


def driver(env_name, weightT, bias):
    env_dict = {"ant": {"env_id": 0, "xml_path": "./env_xmls/ant.xml".encode(), "step_skip": 5, "max_steps": 5000},
                "reacher": {"env_id": 1, "xml_path": "./env_xmls/reacher.xml".encode(), "step_skip": 2, "max_steps": 100}}
    env: MujocoEnv = create_env(env_dict[env_name]["env_id"], env_dict[env_name]["xml_path"], env_dict[env_name]["step_skip"], env_dict[env_name]["max_steps"])
    print(weightT.shape, env.env_id, env.state_size, env.action_size, env.mj_state_size)

    w: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(weightT.shape[0] * weightT.shape[1], cython.sizeof(cython.double)))
    b: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(weightT.shape[1], cython.sizeof(cython.double)))

    i: cython.Py_ssize_t
    j: cython.Py_ssize_t
    for i in range(env.action_size):
        for j in range(weightT.shape[0]):
            w[i*weightT.shape[0]+j] = weightT[j][i]
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

    params: PolicyParams = PolicyParams(k=weightT.shape[1], n=weightT.shape[0], w=w, b=b)
    T: cython.pointer(gsl_rng_type) = gsl_rng_default
    rng: cython.pointer(gsl_rng) = gsl_rng_alloc(T)
    gsl_rng_set(rng, 2)
    reset_env(env, rng)
    total_reward: cython.double = 0.0
    start = time.perf_counter_ns()
    for j in range(50):
        state: cython.pointer(cython.double) = get_state(env)
        print(f"{j} State: ")
        for i in range(env.state_size):
            print(state[i], end=", ")
        print("")
        print("env mj state:")
        mj_state: cython.pointer(cython.double) = get_mj_state(env)
        for j in range(env.mj_state_size):
            print(f"{mj_state[j]}", end=", ")
        print("")
        free(mj_state)
        action: cython.pointer(cython.double) = policy(params, state)
        # action: cython.pointer(cython.double) = cython.cast(cython.pointer(cython.double), calloc(env.action_size, cython.sizeof(cython.double)))
        print("Action: ")
        for i in range(env.action_size):
            print(action[i], end=", ")
        print("")
        reward: cython.double = step(env, action)
        terminated: cython.bint = is_terminated(env, j)
        total_reward += reward
        print("Reward: ", reward, "Terminated: ", terminated, "Total Reward: ", total_reward)
        free(action)
        free(state)
    end = time.perf_counter_ns()
    print(f"Time: {(end - start) / 1e3}")
    # mj_deleteData(env.data)
    free(params.w)
    free(params.b)
    free_env(env)
    gsl_rng_free(rng)