#version 330 core
out vec4 FragColor;

uniform float near;
uniform float far;
uniform vec3 nearColor;
uniform vec3 farColor;
uniform int colormap_selection; // New uniform for colormap selection

vec3 greys(float t) {
    return vec3(t);
}

vec3 plasma(float t) {
    
    vec3 c0 = vec3(0.050383, 0.029803, 0.527975);
    vec3 c1 = vec3(0.458465, 0.013806, 0.557215);
    vec3 c2 = vec3(0.846974, 0.075725, 0.532256);
    vec3 c3 = vec3(0.992172, 0.424599, 0.158234);
    
    return c0 + c1 * t + c2 * t * t + c3 * t * t * t;
}

vec3 cividis(float t) {
    vec3 c0 = vec3(0.0, 0.135112, 0.304751);
    vec3 c1 = vec3(0.208790, 0.398924, 0.522732);
    vec3 c2 = vec3(0.597109, 0.640309, 0.444608);
    vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
    
    return c0 + c1 * t + c2 * t * t + c3 * t * t * t;
}

vec3 magma(float t) {
    vec3 c0 = vec3(1.0, 0.0, 0.0);     // Red
    vec3 c1 = vec3(1.0, 0.4, 0.0);     // Orange-red
    vec3 c2 = vec3(1.0, 0.7, 0.0);     // Orange-yellow
    vec3 c3 = vec3(1.0, 1.0, 0.0);     // Yellow
    
    return c0 + c1 * t + c2 * t * t + c3 * t * t * t;
}

vec3 inferno(float t) {
    vec3 c0 = vec3(0.001462, 0.000466, 0.013866);
    vec3 c1 = vec3(0.355778, 0.068317, 0.377916);
    vec3 c2 = vec3(0.929416, 0.275019, 0.304751);
    vec3 c3 = vec3(0.988362, 0.998364, 0.644924);
    
    return c0 + c1 * t + c2 * t * t + c3 * t * t * t;
}

vec3 viridis(float t) {
    vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
    vec3 c1 = vec3(0.229739, 0.322470, 0.545267);
    vec3 c2 = vec3(0.127568, 0.566949, 0.550556);
    vec3 c3 = vec3(0.369214, 0.788888, 0.382914);
    vec3 c4 = vec3(0.993248, 0.906157, 0.143936);
    
    float t2 = t * t;
    float t3 = t2 * t;
    
    return c0 + c1 * t + c2 * t2 + c3 * t3 + c4 * t3 * t;
}

float LinearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {
    float depth = gl_FragCoord.z;
    float linearDepth = LinearizeDepth(depth) / far;
    
    vec3 color;
    switch(colormap_selection) {
        case 0: // Greys
            color = greys(linearDepth);
            break;
        case 1: // Plasma
            color = plasma(linearDepth);
            break;
        case 2: // Cividis
            color = cividis(linearDepth);
            break;
        case 3: // Magma
            color = magma(linearDepth);
            break;
        case 4: // Inferno
            color = inferno(linearDepth);
            break;
        case 5: // Viridis
            color = viridis(linearDepth);
            break;
        default: // Default to original mix behavior
            color = mix(nearColor, farColor, linearDepth);
            break;
    }
    
    FragColor = vec4(color, 1.0);
}