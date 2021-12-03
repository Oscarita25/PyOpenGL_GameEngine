# Author: Oscar Javurek
# Pygame with Opengl :D
# more Opengl than Pygame now!
import math
import os
import sys

import numpy as np
import pygame as pg

import OpenGL.GL as GL
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from numpy import array

import time
import logging
from logging.handlers import RotatingFileHandler

from datetime import datetime

# frame_time / time_passed / delta_time
# used to store the time between frames to speed up or slow down rendering
# for frame independent rendering speed.
delta_time = None

# TODO
#  make colors being used for something
# Colors:
WHITE = (255, 255, 255)

# config
config = None

version = "ALPHA 0.1"

# display settings:
window_mode = "full_screen"
display = pg.display.set_mode((0, 0), flags=pg.OPENGL | pg.HWSURFACE | pg.DOUBLEBUF | pg.FULLSCREEN)
resolution = display.get_size()
clock = pg.time.Clock()

# video settings
fov = 60
max_fps = 60

logger = logging.getLogger(__name__)
filename = "logs/{}.log".format(datetime.today().strftime('%Y-%m-%d--%H-%M-%S'))
# Logging
if os.path.exists("logs/"):
    file_handler = RotatingFileHandler(filename=filename, backupCount=2, encoding='utf-8', delay=False)
    # logging.basicConfig(filename=filename, level=logging.DEBUG)
else:
    os.makedirs("logs/")
    file_handler = RotatingFileHandler(filename=filename, backupCount=2, encoding='utf-8', delay=False)
    # logging.basicConfig(filename=filename, level=logging.DEBUG)
logger.setLevel(logging.DEBUG)

# export sysout to our log
sysout_handler = logging.StreamHandler(sys.stdout)
sysout_handler.setLevel(logging.DEBUG)

# format the logging
formatter = logging.Formatter("{} %(message)s".format(datetime.today().strftime('%Y-%m-%d--%H-%M-%S')))
#formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%I:%M:%S")
file_handler.setFormatter(formatter)
sysout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(sysout_handler)

logger.info("[- PROGRAM START -]")


# ####################  Base Scene ######################
class SceneBase:

    def __init__(self):
        # is the active Scene
        # change it for a Scene Change!
        # if set to None Gameloops stops
        # and the Game quits.
        self.next = self
        logger.info("Loading Scene: '{}'".format(str(self.__class__).strip("'<class ' __main__." + ">'")))

    def update(self):
        # so i don't have to write common events every time.
        for e in pg.event.get():
            if e.type == pg.QUIT:
                self.terminate()
            if e.type == pg.KEYDOWN:
                if e.key == pg.K_ESCAPE:
                    self.terminate()

    def render(self):
        raise Exception("does not overwrite base scene")

    def next_scene(self, next_scene):
        logger.info("switching Scene: '{}'".format(str(next_scene)))
        self.next = next_scene

    def terminate(self):
        logger.info("Terminating Scene: '{}'".format(str(self.__class__).strip("'<class ' __main__." + ">'")))
        self.next_scene(None)


# TODO
#   ####################  MainMenu Scene ######################
class SceneMainMenu(SceneBase):

    def __init__(self):
        SceneBase.__init__(self)
        self.shader_program = ShaderProgram("assets/shaders/vertex_shader.vsh", "assets/shaders/fragment_shader.fsh")
        self.shader_program.use()
        self.shader_program.setup_3D()
        self.videoship = Mesh("assets/models/cube.obj")
        # add uniforms that need to be read by everyone
        self.shader_program.set_uniform_4fv('vertex_color', 1, array([0, 1, 0, 1]))

        # TODO
        #  finish the init

        self.vbo = vbo.VBO(array([[0, 1, 0], [-1, -1, 0], [1, -1, 0],
                                  [2, -1, 0], [4, -1, 0], [4, 1, 0],
                                  [2, -1, 0], [4, 1, 0], [2, 1, 0]], 'f'))

        self.vbo = vbo.VBO(array([[-1,  1,  1],
                                 [-1, -1, -1],
                                 [-1, -1,  1],
                                 [-1,  1, -1],
                                 [ 1, -1, -1],
                                 [-1, -1, -1],
                                 [ 1,  1, -1],
                                 [ 1, -1,  1],
                                 [ 1, -1, -1],
                                 [ 1,  1,  1],
                                 [-1, -1,  1],
                                 [ 1, -1,  1],
                                 [ 1, -1, -1],
                                 [-1, -1,  1],
                                 [-1, -1, -1],
                                 [-1,  1, -1],
                                 [ 1,  1,  1],
                                 [ 1,  1, -1],
                                 [-1,  1,  1],
                                 [-1,  1, -1],
                                 [-1, -1, -1],
                                 [-1,  1, -1],
                                 [ 1,  1, -1],
                                 [ 1, -1, -1],
                                 [ 1,  1, -1],
                                 [ 1,  1,  1],
                                 [ 1, -1,  1],
                                 [ 1,  1,  1],
                                 [-1,  1,  1],
                                 [-1, -1,  1],
                                 [ 1, -1, -1],
                                 [ 1, -1,  1],
                                 [-1, -1,  1],
                                 [-1,  1, -1],
                                 [-1,  1,  1],
                                 [ 1,  1,  1]], "f"))

        # create a Vertex Buffer Object (for ex. a 3D Model)

        # reset Identity Matrix
        GL.glLoadIdentity()
        # Translate it (move it)
        GL.glTranslatef(0, 0, -10)
        # clear screen and set it to black
        GL.glClearColor(0, 0, 0, 1.0)

        # bind models of scene
        self.vbo.bind()
        #self.videoship.mesh.bind()

    def update(self):
        SceneBase.update(self)

    def render(self):
        clock.tick(max_fps)
        global delta_time
        delta_time = time.perf_counter() - delta_time
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # draw here:

        #GL.glTranslatef(0, 0, delta_time * -2)
        # print(delta_time)
        GL.glRotatef(delta_time * 500, delta_time * 1000, 0, 0)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointerf(self.vbo)

        self.shader_program.set_uniform_4fv('vertex_color', 1,
                                            array([0, 1, (math.sin(time.perf_counter()) / 4), 1.0]))
        # (math.sin(time.perf_counter()) / 2) <- color change
        # connects every 3 cordinates and fills it.
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 36)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

        self.shader_program.set_uniform_matrix4fv('modelview', 1, GL.GL_FALSE, GL.glGetFloat(GL.GL_MODELVIEW_MATRIX))
        delta_time = time.perf_counter()
        pg.display.flip()


# TODO Mesh-Object Class
#  OBJ; FBX; STL; support!
#
# ####################  Util  ######################
class Mesh(object):
    def __init__(self, file_path):
        self.mesh = None
        self.load_from_obj(file_path)

    def load_from_obj(self, file_path):
        # loads from obj file
        logger.debug("Loading Mesh : '{}'".format(file_path))
        if os.path.isfile(file_path):
            vertex_list = None
            for line in open(file_path, "r"):
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    if vertex_list is not None:
                        vertex = [float(values[1]), float(values[2]), float(values[3])]
                        vertex_list.append(vertex)
                        # debugging vertex list values
                        # logger.info("vertex_list: {}".format(vertex_list))
                    else:
                        vertex_list = [[float(values[1]), float(values[2]), float(values[3])]]

                if values[0] == 'f':
                    faces = list(map(int, values[1:4]))
                    if self.mesh is not None:
                        self.mesh = np.append(self.mesh, [vertex_list[faces[0]-1],
                                                          vertex_list[faces[1]-1],
                                                          vertex_list[faces[2]-1]], axis=0)

                    else:
                        self.mesh = array([vertex_list[faces[0]-1],vertex_list[faces[1]-1],vertex_list[faces[2]-1]], "f")

            print(self.mesh)

            self.mesh = vbo.VBO(self.mesh)
            # debugging mesh coordinates
            # logger.info("Mesh:\n {}".format(self.mesh))

        else:
            logger.error("Loading mesh failed : '{}'".format(file_path))
            del self


# ####################  Config  ######################
class Config:
    def __init__(self):
        global fov
        global max_fps
        global window_mode
        global resolution

        # TODO
        #  won't load fov/ make calculations with loaded value
        # Config Loading
        try:
            logger.info("Loading Config: 'config.yml'")
            with open("config.yml", mode='r') as self.config_file:
                for line in self.config_file:
                    if line.startswith("\t"):
                        if "fov:" in line:
                            s = line.strip("\tfov: ")
                            fov = int(s)
                        elif "max_fps:" in line:
                            s = line.strip("\tmax_fps: ")
                            max_fps = int(s)
                        elif "display_mode: " in line:
                            s = line.strip("\tdisplay_mode: ")
                            if s == "full_screen":
                                window_mode = s
                            elif s == "windowed":
                                window_mode = s
                            elif s == "borderless":
                                window_mode = s
                        elif "res: " in line:
                            s = line.replace("\tres: (", "")
                            s = s.replace(")", "")
                            resolution = tuple(map(int, s.split(", ")))

        # if config is not existent (throws OSError) -> Create Default Config
        except OSError:
            logger.info("Config does not exist creating a new one")
            with open("config.yml", mode='w') as self.config_file:
                self.config_file.write("version: {}\n".format(version))

                # default video settings
                self.config_file.write("video_settings:\n")
                self.config_file.write("\tfov: {}\n".format(fov))
                self.config_file.write("\tmax_fps: {}\n".format(max_fps))

                # default display settings
                self.config_file.write("display_settings:\n")
                self.config_file.write("\tdisplay_mode: \n".format(window_mode))
                self.config_file.write("\tres: {}\n".format(display.get_size()))

    # TODO
    #  ingame settings can save config
    def save_config(self):
        pass


# TODO
#  Shader programming
class ShaderProgram(object):

    def __init__(self, vertex_path, fragment_path):
        logger.debug("Loading Shader:\n"
                     "  vertex_shader: '{}'\n"
                     "fragment_shader: '{}'".format(vertex_path, fragment_path))
        self.ID = shaders.glCreateProgram()
        vertex_shader = shaders.glCreateShader(GL.GL_VERTEX_SHADER)
        fragment_shader = shaders.glCreateShader(GL.GL_FRAGMENT_SHADER)

        # 1. retrieve the vertex/fragment shader code from  file
        # could raise exception if file doesn't exist.
        try:
            with open(vertex_path) as file:
                # could raise exception if compiling is going wrong.
                try:
                    vertex_shader = shaders.compileShader(file.read(), GL.GL_VERTEX_SHADER)
                except shaders.ShaderCompilationError:
                    logger.exception("Couldn't Compile Shader '{}'".format(vertex_path))

        except IOError:
            logger.error("Couldn't open or write to file ({})".format(vertex_path))

        # could raise exception if file doesn't exist.
        try:
            with open(fragment_path) as file:

                # could raise exception if compiling is going wrong.
                try:
                    fragment_shader = shaders.compileShader(file.read(), GL.GL_FRAGMENT_SHADER)
                except shaders.ShaderCompilationError:
                    logger.exception("Couldn't Compile Shader '{}' ".format(fragment_path))

        except IOError:
            logger.error("Couldn't open or write to file ({})".format(fragment_path))

        # 2. with the loaded code compile the Programm
        try:
            self.ID = shaders.compileProgram(vertex_shader, fragment_shader)
        except shaders.ShaderValidationError:
            logger.exception("ShaderValidationError")

        # 3. get rid of the uncompiled files.
        GL.glDeleteShader(vertex_shader)
        GL.glDeleteShader(fragment_shader)

    def use(self):
        shaders.glUseProgram(self.ID)

    def setup_3D(self):

        # Fov, Aspect ratio, zNear, zFar
        # sets up Projection Matrix
        z_near = 0.01
        z_far = 10000.0
        aspect_ratio = float(display.get_size()[1]) / float(display.get_size()[0])
        # 0.00555555555 = 1/180° (so we don'T have to use division lol)
        fov_rad = 1 / math.tan(fov * 0.5 * math.pi * 0.00555555555)

        # projection matrix
        p_matrix = array([aspect_ratio * fov_rad, 0.0, 0.0, 0.0,
                          0.0, fov_rad, 0.0, 0.0,
                          0.0, 0.0, (z_far + z_near) / (z_far - z_near), -1.0,
                          0.0, 0.0, 2 * (z_far * z_near) / (z_far - z_near), 0.0], 'f')

        # create value 'pMatrixUniform' and pass it to our 'shader program'
        # set 'pMatrixUniform' to our 'pMatrix' (Projection Matrix)
        self.set_uniform_matrix4fv('projection', 1, GL.GL_FALSE, p_matrix)

        # get the default transformation matrix from Opengl
        self.set_uniform_matrix4fv('modelview', 1, GL.GL_FALSE, GL.glGetFloat(GL.GL_MODELVIEW_MATRIX))

    # setters for the shader methods (for the floating point operations)
    # get Location in the shader, write value to it
    def set_uniform_1f(self, location, value):
        GL.glUniform1f(GL.glGetUniformLocation(self.ID, location), value)

    def set_uniform_2f(self, location, value_0, value_1):
        GL.glUniform2f(GL.glGetUniformLocation(self.ID, location), value_0, value_1)

    def set_uniform_3f(self, location, value_0, value_1, value_2):
        GL.glUniform3f(GL.glGetUniformLocation(self.ID, location), value_0, value_1, value_2)

    def set_uniform_4f(self, location, value_0, value_1, value_2, value_3):
        GL.glUniform4f(GL.glGetUniformLocation(self.ID, location), value_0, value_1, value_2, value_3)

    # arrays / lists / vector
    def set_uniform_1fv(self, location, size, value):
        GL.glUniform1fv(GL.glGetUniformLocation(self.ID, location), size, value)

    def set_uniform_2fv(self, location, size, value):
        GL.glUniform2fv(GL.glGetUniformLocation(self.ID, location), size, value)

    def set_uniform_3fv(self, location, size, value):
        GL.glUniform3fv(GL.glGetUniformLocation(self.ID, location), size, value)

    def set_uniform_4fv(self, location, size, value):
        GL.glUniform4fv(GL.glGetUniformLocation(self.ID, location), size, value)

    def set_uniform_matrix4fv(self, location, size, transpose, value):
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(self.ID, location), size, transpose, value)


# The Main GameLoop
def main(game_name, start_scene):
    pg.init()
    pg.display.set_caption(game_name)

    # Load Config:
    global config
    config = Config()

    # set active scene
    active_scene = start_scene

    # use time.perf_counter()
    # because it is more accurate than time.time()
    global delta_time
    delta_time = time.perf_counter()

    while active_scene is not None:
        active_scene.update()
        active_scene.render()
        active_scene = active_scene.next


if __name__ == '__main__':
    main("Der Zerstörer", SceneMainMenu())
    logger.info("Closing Game")
    pg.quit()
    quit()
