//
// Created by rokas on 17.11.30.
//

#include "Quaternion.hpp"

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

Quaternion::Quaternion() = default;
