import sys, os

sys.argv = ["compile.py", "build_ext", "--inplace"]

from distutils.core import setup, Extension
from Cython.Build import cythonize
import glob

for ff in ("*.c", "*.html", "*.pyd", "*.so"):
    for f in glob.glob(ff):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass

ext_modules = [
    Extension("mujoco", sources=[], include_dirs=["/home/rajtilak/.mujoco/mujoco313/include"], libraries=["mujoco"], library_dirs=["/home/rajtilak/.mujoco/mujoco313/lib"]),
    Extension("gsl", sources=[], libraries=["gsl", "gslcblas"]),
    Extension("mcts", sources=["mcts.py"], include_dirs=["/home/rajtilak/.mujoco/mujoco313/include"], libraries=["mujoco", "gsl", "gslcblas"], library_dirs=["/home/rajtilak/.mujoco/mujoco313/lib"], extra_compile_args=["-fopenmp"], extra_link_args=["-fopenmp"]),
    Extension("mujoco_envs", sources=["mujoco_envs.py"], include_dirs=["/home/rajtilak/.mujoco/mujoco313/include"], libraries=["mujoco", "gsl", "gslcblas"], library_dirs=["/home/rajtilak/.mujoco/mujoco313/lib"]),
    Extension("something", sources=["something.py"],  include_dirs=["/home/rajtilak/.mujoco/mujoco313/include"], libraries=["mujoco", "gsl", "gslcblas"], library_dirs=["/home/rajtilak/.mujoco/mujoco313/lib"], ),
    # extra_compile_args=["/openmp"], extra_link_args=["/openmp"]),  # -fopenmp

]

setup(name="something",
      ext_modules=cythonize(ext_modules, annotate=True, compiler_directives={"language_level": "3"}, ))
