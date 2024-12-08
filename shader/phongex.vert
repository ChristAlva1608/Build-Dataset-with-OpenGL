// Vertex Shader
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texCoord;
layout(location = 2) in vec3 normal;

uniform mat4 modelview;
uniform mat4 projection;

out vec3 frag_normal;
out vec3 frag_pos;
out vec2 frag_texCoord;

void main() {
    vec4 viewPos = modelview * vec4(position, 1.0);
    gl_Position = projection * viewPos;
    
    // Transform normal to view space
    frag_normal = mat3(transpose(inverse(modelview))) * normal;
    
    // Pass view space position and texture coordinates to fragment shader
    frag_pos = viewPos.xyz;
    frag_texCoord = texCoord;
}