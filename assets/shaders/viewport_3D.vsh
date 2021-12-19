#version 330 core

layout (location = 0 ) in vec3 pos; // LocalSpace

uniform mat4 projection; // projection matrix - ScreenSpace&Clipspace
uniform mat4 view;       // view matrix - ViewSpace
uniform mat4 viewpoint;  // Camera viewpoint - Camera pos
uniform mat4 model;      // model matrix - WorldSpace


void main()
{
    gl_Position = projection * (view * viewpoint) * model * vec4(pos,1);
}