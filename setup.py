from cx_Freeze import setup, Executable
import sys
buildOptions = dict(packages=['cv2', 'exposal', 'numpy', 'time', 'PyQt5', 'os', 'random', 'sys'])
exe = [Executable('main.py')]
sys.setrecursionlimit(10000)

setup(
    name='BodySegmentation',
    version='1.0',
    options=dict(build_exe = buildOptions),
    executables=exe
)


