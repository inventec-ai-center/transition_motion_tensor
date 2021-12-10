import numpy as np
import sys
import random
from datetime import datetime
import subprocess

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

sys.path.append('../DeepMimic/')
sys.path.append('scripts/')

from carl_env import CarlEnv
from carl_rl_world import RLWorld
from util.arg_parser import ArgParser
from util.logger import Logger
import util.mpi_util as MPIUtil
import util.util as Util

# Dimensions of the window we are drawing into.
win_width = 800
win_height = int(win_width * 9.0 / 16.0)
reshaping = False

# anim
fps = 60
update_timestep = 1.0 / fps
display_anim_time = int(1000 * update_timestep)
animating = True

playback_speed = 1
playback_delta = 0.05

# FPS counter
prev_time = 0
updates_per_sec = 0

# Video capture
enable_recording = False
ffmpeg = None

args = []
world = None
arg_parser = None


def build_arg_parser(args):
    arg_parser = ArgParser()
    arg_parser.load_args(args)

    arg_file = arg_parser.parse_string('arg_file', '')
    if arg_file != '':
        succ = arg_parser.load_file(arg_file)
        assert succ, Logger.print('Failed to load args from: ' + arg_file)

    rand_seed_key = 'rand_seed'
    if arg_parser.has_key(rand_seed_key):
        rand_seed = arg_parser.parse_int(rand_seed_key)
        rand_seed += 1000 * MPIUtil.get_proc_rank()
        Util.set_global_seeds(rand_seed)

    return arg_parser

def launch_recording(output_filename=None):
    global ffmpeg, win_width, win_height
    if output_filename is None:
        output_filename = 'output_%s.mp4' % (datetime.now().strftime("%H%M%S"))
    cmds = 'ffmpeg -r %d -f rawvideo -pix_fmt rgba -s %dx%d -i - -threads 0 -preset fast -y -pix_fmt yuv420p -crf 18 -vf vflip %s' % (fps, win_width, win_height, output_filename)
    cmds = cmds.split(' ')
    ffmpeg = subprocess.Popen(cmds, stdin=subprocess.PIPE)
    return

def stop_recording():
    global ffmpeg
    ffmpeg.stdin.close()
    ffmpeg.wait()
    ffmpeg = None
    return

def toggle_recording():
    global enable_recording
    enable_recording = not enable_recording
    if enable_recording:
        launch_recording()
        print('Enable Recording')
    else:
        stop_recording()
        print('Disable Recording')
    return

def update_intermediate_buffer():
    if not reshaping:
        if (win_width != world.env.get_win_width() or win_height != world.env.get_win_height()):
            world.env.reshape(win_width, win_height)
    return

def update_world(world, time_elapsed):
    num_substeps = world.env.get_num_update_substeps()
    timestep = time_elapsed / num_substeps
    num_substeps = 1 if (time_elapsed == 0) else num_substeps

    for _ in range(num_substeps):
        world.update(timestep)

        valid_episode = world.env.check_valid_episode()
        if valid_episode:
            end_episode = world.env.is_episode_end()
            if end_episode:
                world.end_episode()
                world.reset()
                break
        else:
            world.reset()
            break
    return

def draw():
    global reshaping

    update_intermediate_buffer()
    world.env.draw()

    glutSwapBuffers()
    reshaping = False

    if enable_recording:
        buffer = glReadPixels(0, 0, win_width, win_height, GL_RGBA, GL_UNSIGNED_BYTE)
        ffmpeg.stdin.write(buffer)

    return

def reshape(w, h):
    global reshaping
    global win_width
    global win_height

    reshaping = True
    win_width = w
    win_height = h
    return

def step_anim(timestep):
    global animating
    global world

    update_world(world, timestep)
    animating = False
    glutPostRedisplay()
    return

def reload():
    global world
    global args

    world = build_world(args, enable_draw=True)
    return

def reset():
    world.reset()
    return

def get_num_timesteps():
    global playback_speed

    num_steps = int(playback_speed)
    if (num_steps == 0):
        num_steps = 1

    num_steps = np.abs(num_steps)
    return num_steps

def calc_display_anim_time(num_timestes):
    global display_anim_time
    global playback_speed

    anim_time = int(display_anim_time * num_timestes / playback_speed)
    anim_time = np.abs(anim_time)
    return anim_time

def shutdown():
    global world

    Logger.print('Shutting down...')
    world.shutdown()
    sys.exit(0)
    return

def get_curr_time():
    curr_time = glutGet(GLUT_ELAPSED_TIME)
    return curr_time

def init_time():
    global prev_time
    global updates_per_sec
    prev_time = get_curr_time()
    updates_per_sec = 0
    return

def animate(callback_val):
    global prev_time
    global updates_per_sec
    global world

    counter_decay = 0

    if animating:
        num_steps = get_num_timesteps()
        curr_time = get_curr_time()
        time_elapsed = curr_time - prev_time
        prev_time = curr_time

        timestep = -update_timestep if (playback_speed < 0) else update_timestep
        for i in range(num_steps):
            update_world(world, timestep)

        # FPS counting
        update_count = num_steps / (0.001 * time_elapsed)
        if (np.isfinite(update_count)):
            updates_per_sec = counter_decay * updates_per_sec + (1 - counter_decay) * update_count
            world.env.set_updates_per_sec(updates_per_sec)

        timer_step = calc_display_anim_time(num_steps)
        update_dur = get_curr_time() - curr_time
        timer_step -= update_dur
        timer_step = np.maximum(timer_step, 0)

        glutTimerFunc(int(timer_step), animate, 0)
        glutPostRedisplay()

    if (world.env.is_done()):
        shutdown()

    return

def toggle_animate():
    global animating

    animating = not animating
    if (animating):
        glutTimerFunc(display_anim_time, animate, 0)

    return

def change_playback_speed(delta):
    global playback_speed

    prev_playback = playback_speed
    playback_speed += delta
    world.env.set_playback_speed(playback_speed)

    if (np.abs(prev_playback) < 0.0001 and np.abs(playback_speed) > 0.0001):
        glutTimerFunc(display_anim_time, animate, 0)

    return

def keyboard(key, x, y):
    world.keyboard(key, x, y)

    if key == b'\x1b': # escape
        shutdown()
    elif key == b' ':
        toggle_animate()
    elif key == b'>':
        step_anim(update_timestep)
    elif key == b'<':
        step_anim(-update_timestep)
    elif key == b',':
        change_playback_speed(-playback_delta)
    elif key == b'.':
        change_playback_speed(playback_delta)
    elif key == b'/':
        change_playback_speed(-playback_speed + 1)
    elif key == b'l':
        reload()
    elif key == b'r':
        reset()
    elif key == b'g':
        toggle_recording()

    glutPostRedisplay()
    return

def mouse_click(button, state, x, y):
    world.env.mouse_click(button, state, x, y)
    glutPostRedisplay()
    return

def mouse_move(x, y):
    world.env.mouse_move(x, y)
    glutPostRedisplay()
    return

def init_draw():
    glutInit()

    glutInitContextVersion(3, 3)
    glutInitContextFlags(GLUT_FORWARD_COMPATIBLE)
    glutInitContextProfile(GLUT_COMPATIBILITY_PROFILE)

    if sys.platform != 'win32':
        glutSetOption(GLUT_MULTISAMPLE, 8)
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE)
        glEnable(GL_MULTISAMPLE)
    else:
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)

    glutInitWindowSize(win_width, win_height)
    glutCreateWindow(b'TMT')
    return

def setup_draw():
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse_click)
    glutMotionFunc(mouse_move)
    glutTimerFunc(display_anim_time, animate, 0)

    reshape(win_width, win_height)
    world.env.reshape(win_width, win_height)
    return

def build_world(args, enable_draw, playback_speed=1):
    arg_parser = build_arg_parser(args)
    env = CarlEnv(args, arg_parser, enable_draw)
    world = RLWorld(env, arg_parser)
    world.env.set_playback_speed(playback_speed)
    return world

def draw_main_loop():
    init_time()
    glutMainLoop()
    return

def main():
    global args

    # Command line arguments
    args = sys.argv[1:]

    init_draw()
    reload()
    setup_draw()
    draw_main_loop()
    return

if __name__ == '__main__':
    main()
