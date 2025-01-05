#define STB_IMAGE_IMPLEMENTATION
#include "dataload.cuh"
#include "math_materials.cuh"
#include "stb_image.h"

// ! temporarily only support triangles.
// ! need to have normals.
std::vector<Triangle> load_obj(const std::string& filename, bool if_texture, V3f corner, float scale) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return std::vector<Triangle>();
    }
    // normalize the obj file
    float min_x = MAX, max_x = -MAX;
    float min_y = MAX, max_y = -MAX;
    float min_z = MAX, max_z = -MAX;

    // maximum faces
    int cnt_faces = 0;

    // read obj file
    std::string line;
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> texcoords;
    std::vector<int> indices_vertices;
    std::vector<int> indices_normals;
    std::vector<int> indices_texcoords;
    while (std::getline(file, line)) {
        if (line.substr(0, 2) == "v ") {
            std::istringstream iss(line.substr(2));
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);

            // normalize the obj file
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
        }
        if (line.substr(0, 2) == "vn") {
            std::istringstream iss(line.substr(2));
            float x, y, z;
            iss >> x >> y >> z;
            normals.push_back(x);
            normals.push_back(y);
            normals.push_back(z);
        }
        if (line.substr(0, 2) == "vt") {
            std::istringstream iss(line.substr(2));
            float u, v;
            iss >> u >> v;
            texcoords.push_back(u);
            texcoords.push_back(v);
        }
        if (line.substr(0, 2) == "f ") {
            std::istringstream iss(line.substr(2));
            int v0, v1, v2;
            int n0, n1, n2;
            int t0, t1, t2;
            char c;

            if (if_texture)
                iss >> v0 >> c >> t0 >> c >> n0 >> v1 >> c >> t1 >> c >> n1 >> v2 >> c >> t2 >> c >> n2;
            else {
                iss >> v0 >> c >> c >> n0 >> v1 >> c >> c >> n1 >> v2 >> c >> c >> n2;
            }

            // if the number of faces exceeds the maximum number of faces, break
            if (cnt_faces++ >= MAX_mesh) {
                std::cerr << "Error: the number of faces exceeds the maximum number of faces " << cnt_faces << std::endl;
                break;
            }

            indices_vertices.push_back(v0 - 1);
            indices_vertices.push_back(v1 - 1);
            indices_vertices.push_back(v2 - 1);
            indices_normals.push_back(n0 - 1);
            indices_normals.push_back(n1 - 1);
            indices_normals.push_back(n2 - 1);

            if (if_texture) {
                indices_texcoords.push_back(t0 - 1);
                indices_texcoords.push_back(t1 - 1);
                indices_texcoords.push_back(t2 - 1);
            }
        }
    }
    file.close();

    // normalize the obj file
    for (int i = 0; i < vertices.size(); i += 3) {
        float ratio = std::max(max_x - min_x, std::max(max_y - min_y, max_z - min_z));
        ratio = scale / ratio;
        vertices[i] = (vertices[i] - min_x) * ratio + corner[0];
        vertices[i + 1] = (vertices[i + 1] - min_y) * ratio + corner[1];
        vertices[i + 2] = (vertices[i + 2] - min_z) * ratio + corner[2];
    }

    // ? print the information
    printf("Info: load_obj %s, vertices %d, normals %d, texcoords %d, faces %d, \n",
           filename.c_str(),
           vertices.size() / 3,
           normals.size() / 3,
           texcoords.size() / 2,
           indices_vertices.size() / 3);

    // convert to triangles
    std::vector<Triangle> triangles;

    for (int i = 0; i < indices_vertices.size(); i += 3) {
        // get the vertices
        V3f v0(vertices[3 * indices_vertices[i]],
               vertices[3 * indices_vertices[i] + 1],
               vertices[3 * indices_vertices[i] + 2]);
        V3f v1(vertices[3 * indices_vertices[i + 1]],
               vertices[3 * indices_vertices[i + 1] + 1],
               vertices[3 * indices_vertices[i + 1] + 2]);
        V3f v2(vertices[3 * indices_vertices[i + 2]],
               vertices[3 * indices_vertices[i + 2] + 1],
               vertices[3 * indices_vertices[i + 2] + 2]);
        // get the normals
        V3f n0(normals[3 * indices_normals[i]],
               normals[3 * indices_normals[i] + 1],
               normals[3 * indices_normals[i] + 2]);
        V3f n1(normals[3 * indices_normals[i + 1]],
               normals[3 * indices_normals[i + 1] + 1],
               normals[3 * indices_normals[i + 1] + 2]);
        V3f n2(normals[3 * indices_normals[i + 2]],
               normals[3 * indices_normals[i + 2] + 1],
               normals[3 * indices_normals[i + 2] + 2]);

        V3f t0, t1, t2;
        if (if_texture) {
            // get the texcoords
            t0 = V3f(texcoords[2 * indices_texcoords[i]],
                     texcoords[2 * indices_texcoords[i] + 1],
                     0.0f);
            t1 = V3f(texcoords[2 * indices_texcoords[i + 1]],
                     texcoords[2 * indices_texcoords[i + 1] + 1],
                     0.0f);
            t2 = V3f(texcoords[2 * indices_texcoords[i + 2]],
                     texcoords[2 * indices_texcoords[i + 2] + 1],
                     0.0f);
        }
        if (if_texture)
            triangles.push_back(Triangle(v0, v1, v2, n0, n1, n2, t0, t1, t2));
        else
            triangles.push_back(Triangle(v0, v1, v2, n0, n1, n2));
    }
    return triangles;
}

std::vector<V3f> load_texture(const std::string& filename, int& width, int& height) {
    int channels;
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    if (data == nullptr) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return std::vector<V3f>();
    }
    std::vector<V3f> texture(width * height);
    for (int i = 0; i < width * height; i++) {
        texture[i] = V3f(data[3 * i] / 255.0f, data[3 * i + 1] / 255.0f, data[3 * i + 2] / 255.0f);
    }
    stbi_image_free(data);

    printf("Info: load_texture %s, width %d, height %d\n", filename.c_str(), width, height);
    return texture;
}