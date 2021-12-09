# Author: Oscar Javurek
# Pygame with Opengl :D
# more Opengl than Pygame now!
import logging
import math
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

import OpenGL.GL as GL
import numpy as np
import pygame as pg
from OpenGL.GL import shaders
from OpenGL.arrays import vbo
from numpy import array

# frame_time / time_passed / delta_time
# used to store the time between frames to speed up or slow down rendering
# for frame independent rendering speed.
delta_time = None

# TODO - use colors
# Colors:
WHITE = (255, 255, 255)

# config
config = None

# display settings:
clock = pg.time.Clock()

logger = logging.getLogger(__name__)
filename = f"logs/{datetime.today().strftime('%Y-%m-%d--%H-%M-%S')}.log"
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
formatter = logging.Formatter(f"{datetime.today().strftime('%Y-%m-%d--%H-%M-%S')} %(message)s")
# formatter = logging.Formatter(fmt="%(asctime)s: %(message)s", datefmt="%I:%M:%S")
file_handler.setFormatter(formatter)
sysout_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(sysout_handler)

logger.info("[- PROGRAM START -]")


# ####################  Base Scene ######################
class SceneBase:

    def __init__(self):
        self.viewport_2D = ShaderProgram("assets/shaders/viewport_2D.vsh",
                                         "assets/shaders/viewport_2D.fsh")
        self.viewport_3D = ShaderProgram("assets/shaders/viewport_3D.vsh",
                                         "assets/shaders/viewport_3D.fsh")
        self.viewport_3D.use()
        self.viewport_3D.setup_3D()
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
        logger.info(f"switching Scene: '{str(next_scene)}'")
        self.next = next_scene

    def terminate(self):
        logger.info("Terminating Scene: '{}'".format(str(self.__class__).strip("'<class ' __main__." + ">'")))
        self.next_scene(None)


# TODO - make menu
class SceneMainMenu(SceneBase):

    def __init__(self):
        SceneBase.__init__(self)

        self.videoship = Mesh("assets/models/VideoShip.obj")
        self.cube = Mesh("assets/models/cube.obj")
        # TODO - replace this:
        self.viewport_3D.set_uniform_4fv('vertex_color', 1, array([0, 1, 0, 1]))

        self.viewport_3D.set_uniform_matrix4fv('modelview', 1, GL.GL_FALSE, GL.glGetFloat(GL.GL_MODELVIEW_MATRIX))
        # clear screen and set it to black
        GL.glClearColor(0, 0, 0, 1.0)
        self.videoship.set_pos(0, 0, -10)
        self.cube.set_pos(0, 10, -20)
        self.camera = Camera(self.viewport_3D)
        # bind models of scene

    def update(self):
        global delta_time
        SceneBase.update(self)
        keys = pg.key.get_pressed()
        mouse_pos = pg.mouse.get_pos()
        self.camera.move(keys, mouse_pos)

    def render(self):
        global delta_time
        clock.tick(config.max_fps)

        delta_time = time.perf_counter() - delta_time
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        # draw here:
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        # GL.glRotatef(delta_time*10, 0, 0, 1)
        # self.viewport_3D.set_uniform_matrix4fv('view', 1,
        #                                      GL.GL_FALSE,
        #                                     GL.glGetFloat(GL.GL_MODELVIEW_MATRIX))
        self.videoship.draw(self.viewport_3D)
        self.cube.draw(self.viewport_3D)
        self.camera.draw()
        # GL.glTranslatef(0, 0, delta_time * -10)
        # print(delta_time)
        delta_time = time.perf_counter()
        pg.display.flip()


# TODO - implement camera
class Camera(object):
    def __init__(self, viewport):
        self.view = array([1.0, 0.0, 0.0, 0.0,
                           0.0, 1.0, 0.0, 0.0,
                           0.0, 0.0, 1.0, 0.0,
                           0.0, 0.0, 0.0, 1.0], 'f')
        self.viewport = viewport

    def move(self, keys, mouse_rotation):
        if keys[pg.K_w]:
            self.add_pos(0, 0, 0.001)
        if keys[pg.K_a]:
            self.add_pos(0.001, 0, 0)
        if keys[pg.K_s]:
            self.add_pos(0, 0, -0.001)
        if keys[pg.K_d]:
            self.add_pos(-0.001, 0, 0)

        self.set_deg(mouse_rotation[0], 0, 0)

    def draw(self):
        self.viewport.set_uniform_matrix4fv('view', 1, GL.GL_FALSE, self.view)

    def set_pos(self, x, y, z):
        self.view[12] = x
        self.view[13] = y
        self.view[14] = z

    def add_pos(self, x, y, z):
        self.view[12] += delta_time * x
        self.view[13] += delta_time * y
        self.view[14] += delta_time * z
        pass

    def set_deg(self, x, y, z):
        self.view[0] = np.cos(x)
        self.view[4] = np.cos(y)
        self.view[8] = np.cos(z)

    def add_deg(self, x, y, z):
        self.view[0] += np.sin(delta_time * x)
        self.view[4] += np.sin(delta_time * y)
        self.view[8] += np.sin(delta_time * z)


# TODO - OBJ; FBX; STL; support!
class Mesh(object):
    def __init__(self, file_path):
        self.mesh = []
        self.x = 0
        self.y = 0
        self.z = 0
        self.x_deg = 0
        self.y_deg = 0
        self.z_deg = 0
        self.load_from_obj(file_path)

    def load_from_obj(self, file_path):
        # loads from obj file
        logger.debug(f"Loading Mesh : '{file_path}'")

        if os.path.isfile(file_path):

            vertex_list = []
            for line in open(file_path, "r"):
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    vertex = [float(values[1]),
                              float(values[2]),
                              float(values[3])]
                    vertex_list.append(vertex)

                if values[0] == 'f':
                    index = list(map(int, values[1:4]))
                    self.mesh.append([vertex_list[index[0] - 1],
                                      vertex_list[index[1] - 1],
                                      vertex_list[index[2] - 1]])

            self.mesh = vbo.VBO(array(self.mesh, "f"))

        else:
            logger.error(f"Loading mesh failed : '{file_path}'")
            del self

    def set_pos(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def add_pos(self, x, y, z):
        self.x += delta_time * x
        self.y += delta_time * y
        self.z += delta_time * z

    def set_deg(self, x, y, z):
        self.x_deg = x
        self.y_deg = y
        self.z_deg = z

    def add_deg(self, x, y, z):
        self.x_deg += delta_time * x
        self.y_deg += delta_time * y
        self.z_deg += delta_time * z

    def draw(self, viewport_shader):
        self.mesh.bind()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        GL.glPushMatrix()

        GL.glTranslatef(self.x, self.y, self.z)  # xyz position
        GL.glRotatef(self.x_deg, 1, 0, 0)  # x-rotation
        GL.glRotatef(self.y_deg, 0, 1, 0)  # y-rotation
        GL.glRotatef(self.z_deg, 0, 0, 1)  # z-rotation
        viewport_shader.set_uniform_matrix4fv('model', 1,
                                              GL.GL_FALSE,
                                              GL.glGetFloat(GL.GL_MODELVIEW_MATRIX))

        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glVertexPointerf(self.mesh)
        viewport_shader.set_uniform_4fv('vertex_color', 1,
                                        array([0, 0.5, (math.sin(time.perf_counter() * 3)), 1.0]))
        GL.glEnable(GL.GL_DEPTH_TEST)
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_LINE)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self.mesh) * 3)
        # GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)
        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)

        GL.glPopMatrix()


# ####################  Config  ######################
class Config:
    def __init__(self):
        # video settings
        self.fov = 60
        self.max_fps = 60
        self.window_mode = "full_screen"
        self.display = pg.display.set_mode((0, 0), flags=pg.OPENGL | pg.HWSURFACE | pg.DOUBLEBUF | pg.FULLSCREEN)
        self.resolution = self.display.get_size()
        self.version = "ALPHA 0.1"

        # TODO - make window mode do something
        # Config Loading
        try:
            logger.info("Loading Config: 'config.yml'")
            with open("config.yml", mode='r') as self.config_file:
                for line in self.config_file:
                    if line.startswith("\t"):
                        if "fov:" in line:
                            s = line.strip("\tfov: ")
                            self.fov = int(s)
                        elif "max_fps:" in line:
                            s = line.strip("\tmax_fps: ")
                            self.max_fps = int(s)
                        elif "display_mode: " in line:
                            s = line.strip("\tdisplay_mode: ")
                            if s == "full_screen":
                                self.window_mode = s
                            elif s == "windowed":
                                self.window_mode = s
                            elif s == "borderless":
                                self.window_mode = s
                        elif "res: " in line:
                            s = line.replace("\tres: (", "")
                            s = s.replace(")", "")
                            self.resolution = tuple(map(int, s.split(", ")))

        # if config is not existent (throws OSError) -> Create Default Config
        except OSError:
            logger.info("Config does not exist creating a new one")
            with open("config.yml", mode='w') as self.config_file:
                self.config_file.write(f"version: {self.version}\n")

                # default video settings
                self.config_file.write("video_settings:\n")
                self.config_file.write(f"\tfov: {self.fov}\n")
                self.config_file.write(f"\tmax_fps: {self.max_fps}\n")

                # default display settings
                self.config_file.write("display_settings:\n")
                self.config_file.write(f"\tdisplay_mode: {self.window_mode}\n")
                self.config_file.write(f"\tres: {self.display.get_size()}\n")

    # TODO - save config from game
    def save_config(self, *args):
        pass


class ShaderProgram(object):

    def __init__(self, vertex_path, fragment_path):
        logger.debug("Loading Shader:\n"
                     f"  vertex_shader: '{vertex_path}'\n"
                     f"fragment_shader: '{fragment_path}\n'")
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
                    logger.exception(f"Couldn't Compile Shader '{vertex_path}'")

        except IOError:
            logger.error(f"Couldn't open or write to file ({vertex_path})")

        # could raise exception if file doesn't exist.
        try:
            with open(fragment_path) as file:

                # could raise exception if compiling is going wrong.
                try:
                    fragment_shader = shaders.compileShader(file.read(), GL.GL_FRAGMENT_SHADER)
                except shaders.ShaderCompilationError:
                    logger.exception(f"Couldn't Compile Shader '{fragment_path}' ")

        except IOError:
            logger.error(f"Couldn't open or write to file ({fragment_path})")

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
        aspect_ratio = float(config.display.get_size()[1]) / float(config.display.get_size()[0])
        # 0.00555555555 = 1/180° (so we don'T have to use division lol)
        fov_rad = 1 / math.tan(config.fov * 0.5 * math.pi * 0.00555555555)

        # projection matrix
        p_matrix = array([aspect_ratio * fov_rad, 0.0, 0.0, 0.0,
                          0.0, fov_rad, 0.0, 0.0,
                          0.0, 0.0, (z_far + z_near) / (z_far - z_near), -1.0,
                          0.0, 0.0, 2 * (z_far * z_near) / (z_far - z_near), 0.0], 'f')

        # create value 'pMatrixUniform' and pass it to our 'shader program'
        # set 'pMatrixUniform' to our 'pMatrix' (Projection Matrix)
        self.set_uniform_matrix4fv('projection', 1, GL.GL_FALSE, p_matrix)

        # get the default transformation matrix from Opengl
        self.set_uniform_matrix4fv('model', 1, GL.GL_FALSE, GL.glGetFloat(GL.GL_MODELVIEW_MATRIX))

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
def main(game_name):
    pg.init()
    pg.display.set_caption(game_name)

    # Load Config:
    global config
    config = Config()

    # set active scene
    active_scene = SceneMainMenu()

    # use time.perf_counter()
    # because it is more accurate than time.time()
    global delta_time
    delta_time = time.perf_counter()

    while active_scene is not None:
        active_scene.update()
        active_scene.render()
        active_scene = active_scene.next


if __name__ == '__main__':
    main("Der Zerstörer")
    logger.info("Closing Game")
    pg.quit()
    quit()
