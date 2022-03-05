#include <iostream>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;


#define MATRIX_SIZE 100

int main(int argc, char **argv)
{
    Matrix<double, Dynamic, Dynamic> matrix_A = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
    matrix_A = matrix_A * matrix_A.transpose();
    Matrix<double, MATRIX_SIZE, 1> vector_b = MatrixXd::Random(MATRIX_SIZE, 1);

    Matrix<double, Dynamic, Dynamic> x;

    x = matrix_A.colPivHouseholderQr().solve(vector_b);
    cout << "The result of QR decomposition is (using householder): " << endl;
    cout << "x = " << x.transpose() << endl;
    cout << endl;

    x = matrix_A.ldlt().solve(vector_b);
    cout << "The result of Cholesky decomposition is : " << endl;
    cout << "x = " << x.transpose() << endl;
    cout << endl;

    return 0;

}


