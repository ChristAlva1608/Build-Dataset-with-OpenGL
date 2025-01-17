#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texcoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 normal_interp;
out vec3 vert_pos;
out vec2 texcoord_interp;

void main(){
  mat4 modelview = view * model;
  vec4 vert_pos4 =  modelview * vec4(position, 1.0);
  vert_pos = vec3(vert_pos4) / vert_pos4.w;

  mat4 normal_matrix = transpose(inverse(modelview));
  normal_interp = vec3(normal_matrix * vec4(normal, 0.0));

  texcoord_interp = texcoord;
  gl_Position = projection * vert_pos4;
}