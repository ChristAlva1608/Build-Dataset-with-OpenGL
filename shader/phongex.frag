// Fragment Shader
#version 330 core

in vec3 frag_normal;
in vec3 frag_pos;
in vec2 frag_texCoord;

uniform vec3 light_pos;  // Light position in view space
uniform mat3 I_light;    // Light intensities (ambient, diffuse, specular)
uniform mat3 K_materials; // Material properties (ambient, diffuse, specular)
uniform float shininess;
uniform int mode;
uniform sampler2D diffuse_texture;
uniform int use_texture;

out vec4 FragColor;

void main() {
    // Normalize vectors
    vec3 N = normalize(frag_normal);
    vec3 L = normalize(light_pos - frag_pos);
    vec3 V = normalize(-frag_pos);
    vec3 R = reflect(-L, N);
    
    // Calculate light components
    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(V, R), 0.0), shininess);
    
    // Get material colors (either from texture or material properties)
    vec3 material_color;
    if (use_texture == 1) {
        vec4 tex_color = texture(diffuse_texture, frag_texCoord);
        material_color = tex_color.rgb;
    } else {
        material_color = K_materials[1]; // Use diffuse color from material
    }
    
    // Calculate final color components
    vec3 ambient = I_light[2] * K_materials[2];
    vec3 diffuse = I_light[0] * material_color * diff;
    vec3 specular = I_light[1] * K_materials[1] * spec;
    
    // Combine all components
    vec3 final_color = ambient + diffuse + specular;
    
    // Apply different rendering modes if specified
    if (mode == 0) {
        // Normal visualization mode
        FragColor = vec4((N + 1.0) / 2.0, 1.0);
    } else if (mode == 1) {
        // Standard lighting mode
        FragColor = vec4(final_color, 1.0);
    } else if (mode == 2) {
        // Texture only mode (if available)
        FragColor = use_texture == 1 ? texture(diffuse_texture, frag_texCoord) : vec4(material_color, 1.0);
    }
}