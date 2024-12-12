#version 330 core
out vec4 FragColor;

uniform float near;
uniform float far;
uniform vec3 nearColor;
uniform vec3 farColor;

float LinearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {
    float depth = gl_FragCoord.z;
    float linearDepth = LinearizeDepth(depth) / far;
    
    // Define colors for near and far
    vec3 nearColor = vec3(1.0, 0.0, 0.0);  // Red
    vec3 farColor = vec3(0.0, 0.0, 1.0);   // Blue
    
    // Interpolate between colors based on depth
    vec3 color = mix(nearColor, farColor, linearDepth);
    FragColor = vec4(color, 1.0);
}