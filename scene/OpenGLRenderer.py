"""
OpenGL based renderer for generating input images from synthetic hair dataset for Hair-GS.
"""

from typing import List

import numpy as np
import cv2
from OpenGL.GL import *
import glfw

vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 normal;

uniform mat4 model;
uniform mat3 normal_model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fPos;
out vec4 fColor;
out vec3 fNormal;
void main()
{
    gl_Position = projection * view * model * vec4(position, 1.0);
    fPos = vec3(model * vec4(position, 1.0));
    fNormal = normal_model * normal;
    fColor = color;
}
"""

fragment_shader_source = """
#version 330 core
in vec3 fPos;
in vec4 fColor;
in vec3 fNormal;

uniform vec3 lightPos;
uniform vec4 ambientColor;
uniform vec4 diffuseColor;
uniform float ka;
uniform float kd;
uniform int use_lighting;

out vec4 FragColor;

void main()
{
    vec4 lightingColor = vec4(1.0, 1.0, 1.0, 1.0);
    if (use_lighting == 1)
    {
        vec3 normal_ = normalize(fNormal);
        vec3 lightDir = normalize(lightPos - fPos);
        vec4 ambient = ka * ambientColor;
        vec4 diffuse = kd * max(dot(normal_, lightDir), 0.0) * diffuseColor;
        lightingColor = ambient + diffuse;
    }
    FragColor = lightingColor * fColor;
}
"""


def create_shader(shader_type, source):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader


class OpenGLModel:
    type = ""
    vertices = None
    colors = None
    normals = None
    indices = None
    model = None

    vao = None
    vbo = None
    ebo = None
    model_uniform_id = None
    normal_model_uniform_id = None
    use_lighting_uniform_id = None
    ka_uniform_id = None
    kd_uniform_id = None

    def __init__(
        self,
        vertices: np.ndarray,
        colors: np.ndarray = None,
        normals: np.ndarray = None,
        edges: np.ndarray = None,
        faces: np.ndarray = None,
        model: np.ndarray = np.eye(4),
        use_lighting: bool = True,
        line_width: float = 1.0,
        ka: float = 0.5,
        kd: float = 0.5,
    ):
        assert (edges is not None or faces is not None) and (
            edges is None or faces is None
        ), "Either edges or faces must be provided, but not both"
        self.vertices = vertices.astype(np.float32)
        self.indices = (
            edges.astype(np.uint32) if edges is not None else faces.astype(np.uint32)
        )
        self.type = GL_LINES if edges is not None else GL_TRIANGLES
        if colors is None:
            colors = np.array([1, 1, 1, 1])
        if colors.shape[0] != vertices.shape[0]:
            colors = np.tile(colors, (vertices.shape[0], 1))
        colors = colors.astype(np.float32)
        self.colors = colors
        if normals is None:
            normals = np.ones(vertices.shape)
        self.normals = normals.astype(np.float32)
        self.model = model.astype(np.float32)
        self.use_lighting = use_lighting
        self.line_width = line_width
        self.ka = ka
        self.kd = kd

    def setup(self, program):
        # vao
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        # vbo: vertices and colors
        buffer_data = np.hstack([self.vertices, self.colors, self.normals])
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, buffer_data.nbytes, buffer_data, GL_STATIC_DRAW)
        glVertexAttribPointer(
            0, 3, GL_FLOAT, GL_FALSE, buffer_data.shape[1] * sizeof(GLfloat), None
        )  # vertices
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            1,
            self.colors.shape[1],
            GL_FLOAT,
            GL_FALSE,
            buffer_data.shape[1] * sizeof(GLfloat),
            ctypes.c_void_p(3 * sizeof(GLfloat)),
        )  # colors
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(
            2,
            3,
            GL_FLOAT,
            GL_FALSE,
            buffer_data.shape[1] * sizeof(GLfloat),
            ctypes.c_void_p(7 * sizeof(GLfloat)),
        )  # normals
        glEnableVertexAttribArray(2)
        # ebo: edges or faces
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW
        )
        # unbind
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
        # model matrix uniform
        self.model_uniform_id = glGetUniformLocation(program, "model")
        self.normal_model_uniform_id = glGetUniformLocation(program, "normal_model")
        self.use_lighting_uniform_id = glGetUniformLocation(program, "use_lighting")
        self.ka_uniform_id = glGetUniformLocation(program, "ka")
        self.kd_uniform_id = glGetUniformLocation(program, "kd")

    def draw(self):
        assert self.vao is not None, "OpenGLModel.setup() must be called before drawing"
        normal_model = np.linalg.inv(self.model[:3, :3]).T
        glUniformMatrix4fv(self.model_uniform_id, 1, GL_TRUE, self.model)
        glUniformMatrix3fv(self.normal_model_uniform_id, 1, GL_TRUE, normal_model)
        glUniform1i(self.use_lighting_uniform_id, 1 if self.use_lighting else 0)
        glUniform1f(self.ka_uniform_id, self.ka)
        glUniform1f(self.kd_uniform_id, self.kd)
        glBindVertexArray(self.vao)
        if self.type == GL_LINES:
            glLineWidth(self.line_width)
        glDrawElements(self.type, self.indices.size, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def cleanup(self):
        glDeleteVertexArrays(1, self.vao)
        glDeleteBuffers(1, self.vbo)
        glDeleteBuffers(1, self.ebo)


class OpenGLCamera:
    view = None
    projection = None

    view_uniform_id = None
    projection_uniform_id = None

    def __init__(self, view: np.ndarray, projection: np.ndarray):
        self.view = view.astype(np.float32)
        self.projection = projection.astype(np.float32)

    def setup(self, program):
        self.view_uniform_id = glGetUniformLocation(program, "view")
        self.projection_uniform_id = glGetUniformLocation(program, "projection")
        glUniformMatrix4fv(self.view_uniform_id, 1, GL_TRUE, self.view)
        glUniformMatrix4fv(self.projection_uniform_id, 1, GL_TRUE, self.projection)


class OpenGLLighting:
    _dark = np.array([0.0, 0.0, 0.0, 0.0])
    light_pos = None
    diffuse_color = None
    ambient_color = None
    specular_color = None

    light_pos_uniform_id = None
    diffuse_color_uniform_id = None
    ambient_color_uniform_id = None
    specular_color_uniform_id = None
    ka_uniform_id = None
    kd_uniform_id = None

    def __init__(
        self,
        light_pos: np.ndarray = np.array([10, 10, 10]),
        diffuse_color: np.ndarray = _dark,
        ambient_color: np.ndarray = _dark,
        specular_color: np.ndarray = _dark,
    ):
        self.light_pos = light_pos.astype(np.float32)
        self.diffuse_color = diffuse_color.astype(np.float32)
        self.ambient_color = ambient_color.astype(np.float32)
        self.specular_color = specular_color.astype(np.float32)

    def setup(self, program):
        self.light_pos_uniform_id = glGetUniformLocation(program, "lightPos")
        self.diffuse_color_uniform_id = glGetUniformLocation(program, "diffuseColor")
        self.ambient_color_uniform_id = glGetUniformLocation(program, "ambientColor")
        self.specular_color_uniform_id = glGetUniformLocation(program, "specularColor")
        glUniform3fv(self.light_pos_uniform_id, 1, self.light_pos)
        glUniform4fv(self.diffuse_color_uniform_id, 1, self.diffuse_color)
        glUniform4fv(self.ambient_color_uniform_id, 1, self.ambient_color)
        glUniform4fv(self.specular_color_uniform_id, 1, self.specular_color)


class OpenGLRenderer:
    resolution = None  # (width, height)
    models = []
    camera = None
    lighting = None
    program = None

    def __init__(self, resolution):
        self.resolution = resolution

    def setup(self):
        # offscreen rendering (window is created but not shown)
        glfw.init()
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(
            self.resolution[0], self.resolution[1], "", None, None
        )
        glfw.make_context_current(window)

        self.program = glCreateProgram()
        vertex_shader = create_shader(GL_VERTEX_SHADER, vertex_shader_source)
        fragment_shader = create_shader(GL_FRAGMENT_SHADER, fragment_shader_source)
        glAttachShader(self.program, vertex_shader)
        glAttachShader(self.program, fragment_shader)
        glLinkProgram(self.program)
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)
        if not glGetProgramiv(self.program, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(self.program))
        glUseProgram(self.program)
        glViewport(0, 0, self.resolution[0], self.resolution[1])
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Set clear color
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        # glEnable(GL_LINE_SMOOTH)
        self.setup_meshes()
        self.setup_camera()
        self.setup_lighting()
        glUseProgram(0)

    def setup_meshes(self, idx: int = None):
        glUseProgram(self.program)
        if idx is not None:
            self.models[idx].setup(self.program)
        else:
            for mesh in self.models:
                mesh.setup(self.program)
        glUseProgram(0)

    def setup_camera(self):
        glUseProgram(self.program)
        if self.camera is not None:
            self.camera.setup(self.program)
        glUseProgram(0)

    def setup_lighting(self):
        glUseProgram(self.program)
        if self.lighting is not None:
            self.lighting.setup(self.program)
        glUseProgram(0)

    def render(
        self, mesh_indices: List[int] = None, background_color=[0.0, 0.0, 0.0, 1.0]
    ) -> np.array:
        # render
        glUseProgram(self.program)
        glClearColor(
            background_color[0],
            background_color[1],
            background_color[2],
            background_color[3],
        )
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        for i, mesh in enumerate(self.models):
            if mesh_indices is None or i in mesh_indices:
                mesh.draw()
        glUseProgram(0)
        glFlush()
        # get image
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(
            0, 0, self.resolution[0], self.resolution[1], GL_RGBA, GL_UNSIGNED_BYTE
        )
        image = np.frombuffer(data, dtype="uint8")
        image = image.reshape((self.resolution[1], self.resolution[0], 4))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.flip(image, 0)
        return image
