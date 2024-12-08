import glm
import numpy as np
import math

class TrackballV2:
    def __init__(self):
        self.center = glm.vec3(0, 0, 0)
        self.camera_distance = 10.0
        self.camera_position = glm.vec3(0, 0, self.camera_distance)
        self.up = glm.vec3(0, 1, 0)
        self.rotation = glm.quat(1, 0, 0, 0)
        self.pan_offset = glm.vec3(0)
        self.fov = 45.0

    def project_to_sphere(self, x, y, width, height):
        # Convert screen coordinates to [-1, 1] range
        x = (2.0 * x - width) / width
        y = (height - 2.0 * y) / height
        
        # Project onto sphere or hyperbolic sheet
        d = math.sqrt(x*x + y*y)
        if d < 1.0:
            z = math.sqrt(1.0 - d*d)  # Project onto sphere
        else:
            z = 1.0 / (2.0 * d)  # Project onto hyperbolic sheet
        return glm.normalize(glm.vec3(x, y, z))

    def drag(self, old_pos, new_pos, window_size):
        if old_pos == new_pos:
            return

        # Get vectors from positions
        va = self.project_to_sphere(old_pos[0], old_pos[1], window_size[0], window_size[1])
        vb = self.project_to_sphere(new_pos[0], new_pos[1], window_size[0], window_size[1])

        # Calculate rotation angle and axis
        angle = math.acos(min(1.0, glm.dot(va, vb)))
        axis = glm.cross(va, vb)

        if glm.length(axis) > 0:
            # Create rotation quaternion and combine with current rotation
            q = glm.quat(math.cos(angle/2), 
                        axis.x * math.sin(angle/2),
                        axis.y * math.sin(angle/2),
                        axis.z * math.sin(angle/2))
            self.rotation = q * self.rotation

    def pan(self, old_pos, new_pos):
        # Pan speed factor
        speed = 0.01
        dx = (new_pos[0] - old_pos[0]) * speed
        dy = (new_pos[1] - old_pos[1]) * speed
        
        # Update pan offset
        right = glm.cross(glm.normalize(self.camera_position - self.center), self.up)
        self.pan_offset += right * -dx + self.up * dy

    def zoom(self, delta, size):
        """ Zoom trackball by a factor delta normalized by windows size """
        self.distance = max(0.001, self.camera_distance * (1 - 50*delta/size))

    def view_matrix(self):
        # Calculate camera position in world space
        rot_mat = glm.mat4_cast(self.rotation)
        camera_pos = glm.vec3(rot_mat * glm.vec4(0, 0, self.camera_distance, 1.0))
        camera_pos += self.pan_offset
        
        # Create view matrix
        return glm.lookAt(camera_pos, self.center + self.pan_offset, self.up)

    def projection_matrix(self, window_size):
        aspect = window_size[0] / window_size[1]
        return glm.perspective(glm.radians(self.fov), aspect, 0.1, 100.0)