#include "bbox.hpp"
#include <iostream>

int main() {
    // Create points
    Point p1(1, 2, 3);
    Point p2(4, 5, 6);

    // Create bounding box
    Bbox bbox(p1, p2);

    // Test Union
    Point p3(7, 8, 9);
    Bbox unionBbox = bbox.Union(p3);
    std::cout << "Union Bbox pMin: " << unionBbox[0] << ", pMax: " << unionBbox[1] << std::endl;

    // Test Overlaps
    Bbox bbox2(Point(5, 6, 7), Point(8, 9, 10));
    std::cout << "Overlaps: " << (bbox.Overlaps(bbox2) ? "true" : "false") << std::endl;

    // Test Inside
    Point p4(2, 3, 4);
    std::cout << "Inside: " << (bbox.Inside(p4) ? "true" : "false") << std::endl;

    // Test Expand
    bbox.Expand(1.0);
    std::cout << "Expanded Bbox pMin: " << bbox[0] << ", pMax: " << bbox[1] << std::endl;

    // Test SurfaceArea
    std::cout << "Surface Area: " << bbox.SurfaceArea() << std::endl;

    // Test Volume
    std::cout << "Volume: " << bbox.Volume() << std::endl;

    // Test MaximumExtent
    std::cout << "Maximum Extent: " << bbox.MaximumExtent() << std::endl;

    // Test SetpMin and SetpMax
    bbox.SetpMin(Point(0, 0, 0));
    bbox.SetpMax(Point(10, 10, 10));
    std::cout << "Set Bbox pMin: " << bbox[0] << ", pMax: " << bbox[1] << std::endl;

    // Test Lerp
    Vector t(0.5, 0.5, 0.5);
    std::cout << "Lerp: " << bbox.Lerp(t) << std::endl;

    // Test Offset
    std::cout << "Offset: " << bbox.Offset(p4) << std::endl;

    // Test BoundingSphere
    Point c;
    float rad;
    bbox.BoundingSphere(&c, &rad);
    std::cout << "Bounding Sphere center: " << c << ", radius: " << rad << std::endl;

    return 0;
}