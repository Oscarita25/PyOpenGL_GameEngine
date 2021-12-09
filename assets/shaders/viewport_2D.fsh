#version 330 core

in vec2 tex_cords;
out vec4 frag_color;

uniform sampler2D tex;
uniform vec3 tex_color;

void main()
{
    frag_color = vec4(tex_color, 1.0) * texture(tex, tex_cords);
}

