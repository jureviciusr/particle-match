//
// Created by rokas on 17.11.30.
//

#include "fastmatch-dataset/Vector3d.hpp"

double Vector3d::getX() const {
    return x;
}

void Vector3d::setX(double x) {
    Vector3d::x = x;
}

double Vector3d::getY() const {
    return y;
}

void Vector3d::setY(double y) {
    Vector3d::y = y;
}

double Vector3d::getZ() const {
    return z;
}

void Vector3d::setZ(double z) {
    Vector3d::z = z;
}

Vector3d::Vector3d() = default;


Vector3d::Vector3d(double x, double y, double z) : x(x), y(y), z(z) {}

Eigen::Vector3d Vector3d::toEigen() {
    return {x, y, z};
}

