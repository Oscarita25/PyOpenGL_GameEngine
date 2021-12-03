#version 330 core

layout (location = 0 ) in vec3 pos;
//layout (location = 1) in vec3 aColor;
//layout (location = 2) in vec2 aTexCoord;


uniform mat4 projection; // projection matrix (perspective)
uniform mat4 modelview; // world matrix

void main() {
    gl_Position = projection* modelview * vec4(pos,1);
}