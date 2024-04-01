# import something
import mujoco_envs
import mcts
import something
import tensorflow as tf

try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], enable=True)
except:
    pass


def tanh(a):
    a = tf.exp(2 * a)
    return (a - 1) / (a + 1)


def print_hi(name):
    # something.some()
    model = tf.keras.models.load_model("pusher_network")

    # state = tf.random.uniform(minval=0, maxval=1.0, shape=(4,27))
    # action = model(state)
    # print(action)
    # w, b = model.get_weights()
    # new_action = tanh(state @ w + b)
    # print(new_action)

    mcts.driver("pusher", *model.get_weights())
    # mujoco_envs.driver("pusher", *model.get_weights())


if __name__ == '__main__':
    print_hi('PyCharm')
