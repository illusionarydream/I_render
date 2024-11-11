# Illusionary Render

<!-- <p align="center">
  <img src="README_image/image.png" alt="meshes of stanford bunny" />
</p> -->
![](README_image/image.png)


## Introduction
`I_render` is a rendering engine developed by Illusionary. In this project, my goal is to improve the rendering speed by applying CUDA programming. The entire task is divided into two parts:
- **CUDA rasterization**
  The CUDA rasterization part is primarily responsible for converting 3D scenes into 2D images. By leveraging the parallel computing power of the GPU, we can significantly speed up the rasterization process, resulting in faster rendering times.

- **CUDA ray tracing**
  The CUDA ray tracing part is used to simulate the propagation and reflection of light within the scene. Through CUDA programming, we can efficiently compute the interactions between light and objects, producing high-quality images.

The combination of these two parts allows `I_render` to achieve faster rendering speeds while maintaining high image quality.

## Cuda rasterization
### Brief Intro
In my opinion, each rendering engine has several important parts. Even though different renderers have distinct approaches to optimization and implementation, they all consider the following key issues:
- The geometric representation of the model
- Transformation from world coordinates to the camera plane
- Lighting and shading of the model
  
The keywords are: **geometry, transformation, shading**. As for `I_render`, I addressed these three issues separately and combined them with CUDA parallelization, achieving good results.

### Implementation: geometric representation
In `I_render`, all objects are represented by triangle meshes. Each triangle consists of three vertices, and each vertex stores two attributes: normal and position. This work focuses on speed, so only the essential attributes are applied. However, extending the functionality of I_render is straightforward. Additional attributes such as UV coordinates, albedo, and others can be easily added to the vertices if needed. The following image shows the meshes of the famous stanford bunny.

```
struct Vertice{
    V3f position; // V3f is Vector-3D-float
    V3f normal;
    // other attributes.
}
```

<!-- <p align="center">
  <img src="README_image/The-Stanford-Bunny-shown-on-the-left-is-reconstructed-shown-on-the-right-397-points.png" alt="meshes of stanford bunny" />
</p> -->
![](README_image/The-Stanford-Bunny-shown-on-the-left-is-reconstructed-shown-on-the-right-397-points.png)

### Implementation: transformation
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

### Implementation: shading
In ``I_render``, I apply the Blinn-Phong shading method to the shader. The Blinn-Phong model mainly considers three components of illumination: ambient illumination, specular illumination, and diffuse illumination.

- **Ambient Illumination**:
Ambient illumination represents the constant light that affects all objects in the scene equally, regardless of their position or orientation. It simulates the indirect light that is scattered throughout the environment. This component ensures that objects are never completely dark, even when they are not directly illuminated.

- **Specular Illumination**:
Specular illumination is the light that is reflected in a particular direction, creating highlights on shiny surfaces. It depends on the angle between the reflected light and the viewer's direction. The Blinn-Phong model uses a more efficient approximation by calculating the half-vector between the light direction and the viewer's direction. This makes the model faster while producing visually similar results.

- **Diffuse Illumination**:
Diffuse illumination is the light that hits a surface and is scattered in all directions, giving the surface a matte appearance. It is directly dependent on the angle between the surface normal and the incoming light direction, as described by Lambert's cosine law.

![](/README_image/phong.png)

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

![](/README_image/Phong_components_version_4.png)

### Pipeline
In I_render, our pipeline can be roughly divided into the following steps:
- The first step is to read the data and initialize various parameters.
- The second step is to project the triangular mesh onto a plane based on depth, obtaining the spatial point and its related information for each pixel.
- The third step is to perform shading for each spatial point corresponding to the pixel.
  
I will seperately introduce these three steps.

![](/README_image/Untitled%20Diagram.drawio.png)

#### First step: initialize
During the initialization process, we need to do the following tasks:
- We need to read the triangular mesh information in OBJ format and store it in the GPU.
- We need to set the camera viewpoint, camera position, and the camera intrinsic matrix (focal length).
- We need to set up the point light source, including its position, color, and intensity information and and store it in the GPU.

#### Second step: projection
During the process of projecting the triangular mesh, we need to transform the mesh from the world coordinate system to the camera coordinate system, as described earlier. This allows us to perform depth visibility testing more easily.

This transformation process is straightforward due to the relative independence of the triangular meshes. **We can perform parallel computation between the triangles**. Each CUDA kernel function only needs to multiply the coordinates of the three vertices of a triangle by the camera's extrinsic matrix. For the three vertices' normals, we multiply by the inverse of the extrinsic matrix to obtain the transformed **vertex coordinates** and **normal directions** in the camera coordinate system.

Next, we need to perform depth testing. Following the **Z-buffer method**, we again leverage parallelism at the triangular mesh level. Each CUDA kernel function traverses the entire triangle and, for each pixel projected onto the view plane, compares the distance from the point to the view plane with the value in the Z-buffer. If the distance is smaller, it indicates that the point is closer to the view plane, and we need to update the Z-buffer value. Additionally, we store the information of this point, such as the interpolated normal, its position in the camera coordinate system, and so on.

However, because the depth testing between each triangle is parallel, there is a possibility of two triangles competing for the same Z-buffer position, leading to data synchronization and conflict issues. To address this, I have redefined an **atomic operation** for the class to ensure that the comparison and update of the data are done safely and correctly.

#### Third step: shading
In the shading stage, since we previously obtained the spatial information for each pixel in the second step, we can use **pixel-level parallel computation** here. Each kernel function handles the shading of a single pixel. I applied the Blinn-Phong shading model for this step. The input and output of this stage is:
- Input: view position, point position, point normal, light position, light emission color, light intensity.
- Output: pixel color

The specific computation progress has been explained on the former text. And as with other rendering engines, the shader module is inherently **extensible**, allowing us to design custom shaders to achieve various effects. `I_render` primarily establishes a basic rasterization model within a parallel processing framework to achieve optimal rendering speed.

### Gallery
In this chapter, I will showcase the results achieved by `I_render`.
#### Depth map
<video width="400" height="400" controls>
  <source src="README_image/Peek 2024-11-11 13-40.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Normal map
<video width="400" height="400" controls>
  <source src="README_image/Peek 2024-11-10 22-03.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Diffuse model
<video width="400" height="400" controls>
  <source src="README_image/Peek 2024-11-11 13-45.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Metal model
<video width="400" height="400" controls>
  <source src="README_image/Peek 2024-11-11 13-46.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>