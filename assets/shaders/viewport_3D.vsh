#version 330 core

layout (location = 0 ) in vec3 pos;

uniform mat4 projection; // projection matrix (perspective)
uniform mat4 model; // model matrix
uniform mat4 view; //view matrix | camera

void main()
{
    gl_Position = projection * view * model * vec4(pos,1);
}