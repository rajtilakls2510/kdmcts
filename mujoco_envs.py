import cython
from cython.cimports.libc.stdlib import calloc, free
from cython.cimports.mujoco import mj_loadXML, mjModel, mj_printModel, \
    mjData, mj_makeData, mj_printData, mj_resetData, mj_step, mj_deleteData, \
    mj_deleteModel, mj_copyData


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
def reset_env(env: MujocoEnv) -> cython.void:
    mj_resetData(env.model, env.data)


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def step(env: MujocoEnv, action: cython.pointer(cython.double), num_steps: cython.int) -> cython.void:
    set_action(env, action)
    i: cython.Py_ssize_t
    for i in range(num_steps):
        mj_step(env.model, env.data)



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
    # TODO: Add Clips and scaling as necessary
    i: cython.Py_ssize_t
    for i in range(env.model.nu):
        env.data.ctrl[i] = action[i]



def driver(env_name):
    names = {"ant": {"env_id": 0, "xml_path": "./env_xmls/ant.xml".encode()}}
    env: MujocoEnv = create_env(names[env_name]["env_id"], names[env_name]["xml_path"])
    print(env.env_id, env.state_size, env.action_size)
    reset_env(env)
    for j in range(5):
        state: cython.pointer(cython.double) = get_state(env)
        print("State: ")
        for i in range(env.state_size):
            print(state[i], end=", ")
        print("")
        free(state)

        action: cython.pointer(cython.double) = get_action(env)
        print("Action: ")
        for i in range(env.action_size):
            print(action[i], end=", ")
            action[i] = 1.0
        print("")
        step(env, action, 5)
        free(action)
    free_env(env)