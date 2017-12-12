//
// Created by rokas on 17.11.30.
//

#include "fastmatch-dataset/Quaternion.hpp"



double Quaternion::getX() const {
    return x;
}

void Quaternion::setX(double x) {
    Quaternion::x = x;
}

double Quaternion::getY() const {
    return y;
}

void Quaternion::setY(double y) {
    Quaternion::y = y;
}

double Quaternion::getZ() const {
    return z;
}

void Quaternion::setZ(double z) {
    Quaternion::z = z;
}

double Quaternion::getW() const {
    return w;
}

void Quaternion::setW(double w) {
    Quaternion::w = w;
}

Quaternion::Quaternion(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) {}

Vector3d Quaternion::toRPY() const {
    auto rpy = toEigen().toRotationMatrix().eulerAngles(0, 1, 2);
    return {rpy[0], rpy[1], rpy[2]};
}

Eigen::Quaterniond Quaternion::toEigen() const {
    return Eigen::Quaterniond(w, x, y, z);
}

Vector3d Quaternion::toRPYdegrees() const {
    auto rpy = toEigen().toRotationMatrix().eulerAngles(2, 0, 2);
    return {rpy[0] * (180.0 / M_PI), rpy[1] * (180.0 / M_PI), rpy[2] * (180.0 / M_PI)};
}

Quaternion::Quaternion() = default;
