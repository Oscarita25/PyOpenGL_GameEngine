#version 330 core

layout (location = 0 ) in vec4 pos;
//layout (location = 1) in vec3 aColor;
//layout (location = 2) in vec2 aTexCoord;


uniform mat4 projection; // projection matrix (perspective)
uniform mat4 view; // view  matrix
uniform mat4 model; // world matrix

void main() {
    gl_Position = projection * view * model * pos;
}