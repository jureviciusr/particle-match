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
    double pitch = asin(-2.0*(x*z - w*y));
    double roll = atan2(2.0*(y*z + w*x), w*w - x*x - y*y + z*z);
    double heading = atan2(2.0*(x*y + w*z), w*w + x*x - y*y - z*z);

    // Normalize heading to according to earth north pole.
    // The heading is now an angle in range of [0, 2*PI] from north pole clockwise.

    // Shift to range
    heading += M_PI;

    // Reverse and offset
    heading = ((2. * M_PI) - heading) + M_PI + M_PI_2;

    // Normalize again
    if(heading > (2. * M_PI)) {
        heading -= 2. * M_PI;
    }

    return {roll, pitch, heading};
}

Eigen::Quaterniond Quaternion::toEigen() const {
    return Eigen::Quaterniond(w, x, y, z);
}

Vector3d Quaternion::toRPYdegrees() const {
    auto rpy = toRPY();
    return {rpy.getX() * (180.0 / M_PI), rpy.getY() * (180.0 / M_PI), rpy.getZ() * (180.0 / M_PI)};
}

Quaternion::Quaternion(const Eigen::Quaterniond &eq) : Quaternion(eq.x(), eq.y(), eq.z(), eq.w()) {}

Quaternion::Quaternion() = default;
