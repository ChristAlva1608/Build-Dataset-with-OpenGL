#version 330 core

// input attribute variable, given per vertex
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;

uniform mat4 projection, modelview;
out vec2 texcoord_interp;

void main(){
    texcoord_interp = texcoord;
    gl_Position = projection * modelview * vec4(position, 1.0);
}
