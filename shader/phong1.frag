#version 330 core

precision mediump float;
in vec3 normal_interp;  // Surface normal
in vec3 vertPos;       // Vertex position
in vec3 colorInterp;
in vec2 TextCoord;

uniform mat3 K_materials;
uniform mat3 I_light;

uniform int mode;       // Rendering mode
uniform float shininess; // Shininess
uniform vec3 light_pos; // Light position
uniform sampler3D ourTexture;
uniform bool useTexCoord; 


out vec4 fragColor;

void main() {
    vec3 N = normalize(normal_interp);
    vec3 L = normalize(light_pos - vertPos);
    vec3 R = reflect(-L, N);      // Reflected light vector
    vec3 V = normalize(-vertPos); // Vector to viewer

    float specAngle = max(dot(R, V), 0.0);
    float specular = pow(specAngle, shininess);
    vec3 g = vec3(max(dot(L, N), 0.0), specular, 1.0);
    vec3 rgb = 0.5 * matrixCompMult(K_materials, I_light) * g + 0.5 * colorInterp;

    if (useTexCoord) {
        // Normalize vertex position to get appropriate texture coordinates
        vec3 texCoords = (vertPos + vec3(0.5)) * vec3(1.0); // Mapping [-0.5, 0.5] to [0.0, 1.0]
        fragColor = texture(ourTexture, texCoords); // Use texCoords for sampling
        fragColor.a = 1.0; // Set alpha to fully opaque
    } else {
        fragColor = vec4(rgb, 1.0);
    }
}