#include "dataload.cuh"

std::vector<Triangle> load_obj(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return std::vector<Triangle>();
    }
    // read obj file
    std::string line;
    std::vector<float> vertices;
    std::vector<int> indices;
    while (std::getline(file, line)) {
        if (line.substr(0, 2) == "v ") {
            std::istringstream iss(line.substr(2));
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
        if (line.substr(0, 2) == "f ") {
            std::istringstream iss(line.substr(2));
            int v0, v1, v2;
            iss >> v0 >> v1 >> v2;
            indices.push_back(v0 - 1);
            indices.push_back(v1 - 1);
            indices.push_back(v2 - 1);
        }
    }
    file.close();

    // convert to triangles
    std::vector<Triangle> triangles;

    for (int i = 0; i < indices.size(); i += 3) {
        V3f v0(vertices[3 * indices[i]], vertices[3 * indices[i] + 1], vertices[3 * indices[i] + 2]);
        V3f v1(vertices[3 * indices[i + 1]], vertices[3 * indices[i + 1] + 1], vertices[3 * indices[i + 1] + 2]);
        V3f v2(vertices[3 * indices[i + 2]], vertices[3 * indices[i + 2] + 1], vertices[3 * indices[i + 2] + 2]);
        triangles.push_back(Triangle(v0, v1, v2));
    }
    return triangles;
}