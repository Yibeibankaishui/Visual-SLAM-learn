//
// Created by 高翔 on 2017/12/19.
// 本程序演示如何从Essential矩阵计算R,t
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;

#include <sophus/so3.hpp>

#include <iostream>

using namespace std;

int main(int argc, char **argv) {

    // 给定Essential矩阵
    Matrix3d E;
    E << -0.0203618550523477, -0.4007110038118445, -0.03324074249824097,
            0.3939270778216369, -0.03506401846698079, 0.5857110303721015,
            -0.006788487241438284, -0.5815434272915686, -0.01438258684486258;

    // 待计算的R,t
    Matrix3d R;
    Vector3d t;

    // SVD and fix sigular values
    // START YOUR CODE HERE
    JacobiSVD<MatrixXd> svd(E, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    MatrixXd A = svd.singularValues();

    // 处理Sigma
    Matrix3d Sigma = Matrix3d::Zero();
    Sigma(0) = Sigma(4) = (A(0) + A(1)) / 2;
    // END YOUR CODE HERE

    // set t1, t2, R1, R2 
    // START YOUR CODE HERE
    // 计算R_Z(\frac{\pi}{2})
    AngleAxisd RZ(M_PI_2, Vector3d(0, 0, 1)); 
    AngleAxisd RZ_(-M_PI_2, Vector3d(0, 0, 1)); 
    Matrix3d RZm = RZ.toRotationMatrix();
    Matrix3d RZ_m = RZ_.toRotationMatrix();

    Matrix3d t_wedge1 = U * RZm * Sigma * U.transpose();
    Matrix3d t_wedge2 = U * RZ_m * Sigma * U.transpose();

    Matrix3d R1 = U * RZm.transpose() * V.transpose();
    Matrix3d R2 = U * RZ_m.transpose() * V.transpose();
    // END YOUR CODE HERE

    cout << "R1 = " << R1 << endl;
    cout << "R2 = " << R2 << endl;
    cout << "t1 = " << Sophus::SO3d::vee(t_wedge1) << endl;
    cout << "t2 = " << Sophus::SO3d::vee(t_wedge2) << endl;

    // check t^R=E up to scale
    Matrix3d tR = t_wedge1 * R1;
    cout << "t^R = " << tR << endl;

    return 0;
}