//
// Created by rokas on 17.11.30.
//

#pragma once

#include <Eigen/Eigen>
#include "Vector3d.hpp"

class Quaternion {
public:
    double getX() const;

    void setX(double x);

    double getY() const;

    void setY(double y);

    double getZ() const;

    void setZ(double z);

    double getW() const;

    void setW(double w);

    Vector3d toRPY() const;

    Vector3d toRPYdegrees() const;

    Eigen::Quaterniond toEigen() const;

protected:
    double x, y, z, w;

public:
    Quaternion(double x, double y, double z, double w);

    Quaternion();
};


