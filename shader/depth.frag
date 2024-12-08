#version 330 core
out vec4 FragColor;

uniform float near;
uniform float far;

float LinearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {
    float depth = gl_FragCoord.z;
    float linearDepth = LinearizeDepth(depth) / far;
    FragColor = vec4(vec3(linearDepth), 1.0);
}