#version 330 core
out vec4 FragColor;

uniform float near;
uniform float far;
uniform vec3 nearColor;
uniform vec3 farColor;
uniform int colormap_selection; // New uniform for colormap selection

uniform float magma_data[256*3];

float LinearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {
    float depth = gl_FragCoord.z;
    float linearDepth = LinearizeDepth(depth) / far;
    int depth_value = int(linearDepth * 255);

    vec3 value;
    switch(colormap_selection) {
        case 0: // Greys
            value = vec3(linearDepth);
            break;
        case 1: // Magma
            value = vec3(magma_data[depth_value*3], magma_data[depth_value*3+1], magma_data[depth_value*3+2]); // numCols = 3
            break;
    }
    FragColor = vec4(value, 1.0);
}