#version 330 core

precision mediump float;
in vec3 vert_pos;       // Vertex position
in vec3 normal_interp;  // Surface normal
in vec2 texcoord_interp;

out vec4 fragColor;

uniform float shininess; // Shininess
uniform vec3 lightPos; // Light position
uniform vec3 lightColor;
uniform vec3 viewPos;
uniform vec3 objectColor;

uniform vec3 ambientStrength;
uniform vec3 diffuseStrength;
uniform vec3 specularStrength;

uniform sampler2D texture_ambient;
uniform sampler2D texture_diffuse;
uniform sampler2D texture_specular;
uniform sampler2D texture_refl;
uniform sampler2D texture_bump;

uniform float near;
uniform float far;
uniform int colormap_selection; // New uniform for colormap selection
uniform float magma_data[256*3];

uniform int use_texture;
uniform int mode;

float LinearizeDepth(float depth) {
    float z = depth * 2.0 - 1.0; // Back to NDC 
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main() {
    if (mode==1){
        // ambient
        float ambientStrength = 0.5;
        vec3 ambient = ambientStrength * lightColor;

        // diffuse
        float diffuseStrength = 0.5;
        vec3 norm = normalize(normal_interp);
        vec3 lightDir = normalize(lightPos - vert_pos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diffuseStrength * lightColor;

        // specular
        float specularStrength = 0.5;
        vec3 viewDir = normalize(viewPos - vert_pos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = specularStrength * spec * lightColor;

        // Combine lighting components
        vec3 lighting = (ambient + diffuse + specular);
        
        // Determine the final color based on use_texture
        vec4 finalColor;
        if (use_texture == 1) {
            if (texture(texture_diffuse, texcoord_interp).a < 0.1) // discard for the RGBA
               discard;

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
    else {
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
        fragColor = vec4(value, 1.0);
    }
}