#version 330 core

out vec4 frag_color;
uniform vec4 vertex_color;

void main() {
    frag_color = vertex_color;
}
