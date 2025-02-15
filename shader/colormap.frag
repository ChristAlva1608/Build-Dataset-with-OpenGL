#version 330 core

in vec2 texcoord_interp;
out vec4 FragColor;
uniform sampler2D DepthTexture;

void main() {
    vec3 out_color = texture(DepthTexture, texcoord_interp).rgb;
    FragColor = vec4(out_color, 1.0);
}