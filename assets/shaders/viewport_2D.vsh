#version 330 core

layout (location = 0 ) in vec2 pos;
layout (location = 1 ) in vec2 cords;

uniform mat4 projection; // projection matrix (perspective)
uniform mat4 modelview; // model matrix
uniform mat4 camera; //camera

out vec2 tex_cords;
void main()
{
    tex_cords = cords;
    gl_Position = projection * modelview * vec4(pos,0,1);
}
