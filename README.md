## Illusionary Render
### Introduction
Illusionary Render is a render engine constructed by illusionary. In this work, I meant to improve the speed of rendering applying cuda programming. I divide the whole assignment into two parts:
- Cuda rasterization.
- Cuda raytracer.

I will introduce seperately these two parts in the following text.

### Cuda rasterization
#### Brief Intro
In my opinion, each rendering engine has several important parts. Even though different renderers have distinct approaches to optimization and implementation, they all consider the following key issues:
- The geometric representation of the model
- Transformation from world coordinates to the camera plane
- Lighting and shading of the model
  
The keywords are: **geometry, transformation, shading**. As for I_render, I addressed these three issues separately and combined them with CUDA parallelization, achieving good results.

#### Implementation: geometric representation
In I_render, all objects is presented by triangle meshes. Each triangle has three vertices. And each vertices store two attributes: normal and position. Because this work focus on speed, I only apply the basic attributes. To extend the function of I_render, it's easy to add some uv coordinates, albedo and so on to the vertices.
<p align="center">
  <img src="README_image/The-Stanford-Bunny-shown-on-the-left-is-reconstructed-shown-on-the-right-397-points.png" alt="meshes of stanford bunny" />
</p>

#### Implementation: transformation
As for transformation, I borrowed a method from computer vision that is more fundamental and straightforward compared to the MVP (Model-View-Projection) transformation. Each camera has its own **intrinsic matrix**, **extrinsic matrix**, and **projection matrix**:

- **Intrinsic Matrix**:
    The intrinsic matrix \( I_{3 \times 3} \) represents the camera's internal parameters, such as focal length and the aspect ratio of the x and y coordinates. It can be understood as the **viewport transformation**, which maps the coordinates from the camera's image plane to the screen space. The intrinsic matrix is typically defined as:

    $$
    I = 
    \begin{bmatrix}
    f_x & 0 & c_x \\
    0 & f_y & c_y \\
    0 & 0 & 1
    \end{bmatrix}
    $$

    Where:
    - \( f_x \) and \( f_y \) are the focal lengths in the x and y directions, respectively.
    - \( c_x \) and \( c_y \) are the coordinates of the optical center (principal point) in the image plane.

- **Extrinsic Matrix**:
    The extrinsic matrix \( E_{4 \times 4} \) describes the camera's position and orientation in the world space. It transforms points from world space to view space by applying the translation and rotation of the camera. The extrinsic matrix can be written as:

    $$
    E = 
    \begin{bmatrix}
    R & t \\
    0 & 1
    \end{bmatrix}
    $$

    Where:
    - \( R \) is the 3x3 rotation matrix that represents the camera's orientation (how the camera is rotated in space).
    - \( t \) is the 3x1 translation vector representing the camera's position in world space.

- **Projection Matrix**:
    The projection matrix \( P_{3 \times 4} \) transforms points from view space to screen space, accounting for the perspective projection and creating the effect of depth in the 3D scene. For a simple perspective projection, the projection matrix is given by:

    $$
    P = 
    \begin{bmatrix}
    \frac{1}{z} & 0 & 0 & 0 \\
    0 & \frac{1}{z} & 0 & 0 \\
    0 & 0 & \frac{1}{z} & 0
    \end{bmatrix}
    $$

    Where \( z \) is the depth of the point being projected.

- **Final Transformation**:
    To transform a point \( \mathbf{X}_{world} \) in world space to screen space, you combine these three matrices. The process is as follows:
    1. First, transform the point from world space to view space using the extrinsic matrix \( E \).
    2. Then, apply the intrinsic matrix \( I \) to map from view space to camera space.
    3. Finally, apply the projection matrix \( P \) to convert from camera space to screen space.

    The final transformation can be written as:

    $$
    \mathbf{X}_{screen} = P \cdot I \cdot E \cdot \mathbf{X}_{world}
    $$

    This combination of intrinsic, extrinsic, and projection matrices defines the complete transformation pipeline from world space to screen space in computer vision and 3D rendering.

#### Implementation: shading
In `I_render`, I apply the Blinn-Phong shading method to the shader. The Blinn-Phong model mainly considers three components of illumination: ambient illumination, specular illumination, and diffuse illumination.

- **Ambient Illumination**:
Ambient illumination represents the constant light that affects all objects in the scene equally, regardless of their position or orientation. It simulates the indirect light that is scattered throughout the environment. This component ensures that objects are never completely dark, even when they are not directly illuminated.

- **Specular Illumination**:
Specular illumination is the light that is reflected in a particular direction, creating highlights on shiny surfaces. It depends on the angle between the reflected light and the viewer's direction. The Blinn-Phong model uses a more efficient approximation by calculating the half-vector between the light direction and the viewer's direction. This makes the model faster while producing visually similar results.

- **Diffuse Illumination**:
Diffuse illumination is the light that hits a surface and is scattered in all directions, giving the surface a matte appearance. It is directly dependent on the angle between the surface normal and the incoming light direction, as described by Lambert's cosine law.

The final illumination \(I\) for a point on a surface is the sum of these three components:
$$ I = I_{ambient} + I_{diffuse} + I_{specular} $$

Where:
- **Ambient illumination** is calculated as:  
  $$ I_{ambient} = k_{ambient} \cdot I_{light} $$

- **Diffuse illumination** is calculated using Lambert's cosine law:  
  $$ I_{diffuse} = k_{diffuse} \cdot I_{light} \cdot \max(\mathbf{N} \cdot \mathbf{L}, 0) $$

- **Specular illumination** is calculated as:  
  $$ I_{specular} = k_{specular} \cdot I_{light} \cdot \left( \max(\mathbf{H} \cdot \mathbf{N}, 0) \right)^n $$  
  Where:
  - \( \mathbf{H} \) is the half-vector:  
    $$ \mathbf{H} = \frac{\mathbf{L} + \mathbf{V}}{|\mathbf{L} + \mathbf{V}|} $$  
    \( \mathbf{L} \) is the light direction and \( \mathbf{V} \) is the view direction.
  - \( n \) is the shininess exponent.

This combination of ambient, diffuse, and specular illumination produces realistic lighting effects, simulating how light interacts with surfaces to create depth and highlight details in 3D rendering.
