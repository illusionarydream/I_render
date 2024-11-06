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
    std::vector<float> normals;
    std::vector<int> indices_vertices;
    std::vector<int> indices_normals;
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
            int n0, n1, n2;
            char c;
            iss >> v0 >> c >> c >> n0 >> v1 >> c >> c >> n1 >> v2 >> c >> c >> n2;

            indices_vertices.push_back(v0 - 1);
            indices_vertices.push_back(v1 - 1);
            indices_vertices.push_back(v2 - 1);
            indices_normals.push_back(n0 - 1);
            indices_normals.push_back(n1 - 1);
            indices_normals.push_back(n2 - 1);
        }
        if (line.substr(0, 2) == "vn") {
            std::istringstream iss(line.substr(2));
            float x, y, z;
            iss >> x >> y >> z;
            normals.push_back(x);
            normals.push_back(y);
            normals.push_back(z);
        }
    }
    file.close();

    // convert to triangles
    std::vector<Triangle> triangles;

    for (int i = 0; i < indices_vertices.size(); i += 3) {
        // get the vertices
        V3f v0(vertices[3 * indices_vertices[i]], vertices[3 * indices_vertices[i] + 1], vertices[3 * indices_vertices[i] + 2]);
        V3f v1(vertices[3 * indices_vertices[i + 1]], vertices[3 * indices_vertices[i + 1] + 1], vertices[3 * indices_vertices[i + 1] + 2]);
        V3f v2(vertices[3 * indices_vertices[i + 2]], vertices[3 * indices_vertices[i + 2] + 1], vertices[3 * indices_vertices[i + 2] + 2]);
        // get the normals
        V3f n0(normals[3 * indices_normals[i]], normals[3 * indices_normals[i] + 1], normals[3 * indices_normals[i] + 2]);
        V3f n1(normals[3 * indices_normals[i + 1]], normals[3 * indices_normals[i + 1] + 1], normals[3 * indices_normals[i + 1] + 2]);
        V3f n2(normals[3 * indices_normals[i + 2]], normals[3 * indices_normals[i + 2] + 1], normals[3 * indices_normals[i + 2] + 2]);

        triangles.push_back(Triangle(v0, v1, v2, n0, n1, n2));
    }
    return triangles;
}