import something
import mujoco_envs
import tensorflow as tf

try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], enable=True)
except:
    pass

def print_hi(name):
    # something.some()


    mujoco_envs.driver("ant")

if __name__ == '__main__':
    print_hi('PyCharm')

