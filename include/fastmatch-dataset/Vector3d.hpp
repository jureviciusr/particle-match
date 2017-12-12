//
// Created by rokas on 17.11.30.
//

#pragma once

#include <Eigen/Eigen>

class Vector3d {
protected:
    double x, y ,z;
public:
    double getX() const;

    void setX(double x);

    double getY() const;

    void setY(double y);

    Vector3d();

    Vector3d(double x, double y, double z);

    double getZ() const;

    void setZ(double z);

    Eigen::Vector3d toEigen();
};


