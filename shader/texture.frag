#version 330 core

precision mediump float;
in vec3 vert_pos;       // Vertex position
in vec3 normal_interp;  // Surface normal
in vec2 texcoord_interp;

uniform mat3 K_materials;
uniform mat3 I_light;

uniform float shininess; // Shininess
uniform vec3 lightPos; // Light position
uniform vec3 lightColor;
uniform vec3 viewPos;
uniform vec3 objectColor;
out vec4 fragColor;

uniform sampler2D texture_ambient;
uniform sampler2D texture_diffuse;
uniform sampler2D texture_specular;
uniform sampler2D texture_refl;
uniform sampler2D texture_bump;

uniform int use_texture;

void main() {
    // ambient
    float ambientStrength = 0.5;
    vec3 ambient = ambientStrength * lightColor;

    // diffuse
    vec3 norm = normalize(normal_interp);
    vec3 lightDir = normalize(lightPos - vert_pos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    // specular
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - vert_pos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    // Combine lighting components
    vec3 lighting = (ambient + diffuse + specular);

    // Determine the final color based on use_texture
    vec4 finalColor;
    if (use_texture == 1) {
        // Use texture
        vec4 ambientColor = texture(texture_ambient, texcoord_interp);
        vec4 diffuseColor = texture(texture_diffuse, texcoord_interp);
        vec4 specularColor = texture(texture_specular, texcoord_interp);
        vec4 reflColor = texture(texture_refl, texcoord_interp);
        vec4 bumpColor = texture(texture_bump, texcoord_interp);

        // Combine the textures
        vec4 textureColor = diffuseColor * 0.8 + ambientColor * 0.05 + specularColor * 0.05 + reflColor * 0.05 + specularColor * 0.05; // Example blend
        finalColor = textureColor * vec4(lighting, 1.0);
    } else {
        // Use object color
        finalColor = vec4(lighting * objectColor, 1.0);
    }

    fragColor = finalColor;
}

