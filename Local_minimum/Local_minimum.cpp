#include<iostream>
#include<Eigen/Dense>
#include<Eigen/Eigenvalues>
#include<complex>
#include<Math.h>

using Eigen::MatrixXf;
using Eigen::VectorXf;
using namespace std;

float function(VectorXf X) {
	return 100 * pow(X(1) - pow(X(0), 2), 2) + pow(1 - X(0), 2);
}
VectorXf gradientF(VectorXf X) {
	float xDerivative = -400 * (X(1) - pow(X(0), 2)) * X(0) - 2 * (1 - X(0));
	float yDerivative = 200 * (X(1) - pow(X(0), 2));
	VectorXf vectorOfDerivatives(2);
	vectorOfDerivatives(0) = xDerivative;
	vectorOfDerivatives(1) = yDerivative;
	return vectorOfDerivatives;
}
MatrixXf hesianF(VectorXf X) {
	float xxDerivative = 800 * pow(X(0), 2) - 400 * (X(1) - pow(X(0), 2)) + 2;
	float xyDerivative = -400 * X(0);
	float yyDerivative = 200;
	MatrixXf hesian(2, 2);
	hesian(0, 0) = xxDerivative;
	hesian(1, 0) = xyDerivative;
	hesian(0, 1) = xyDerivative;
	hesian(1, 1) = yyDerivative;
	return hesian;
}
float norm(VectorXf X, int n) {
	float sum = 0;
	for (int i = 0; i < n; i++) {
		sum = sum + pow(X(i), 2);
	}
	float norm = sqrt(sum);
	return norm;
}
float adjunctFunction(VectorXf X, VectorXf d, float x) {
	return 100 * pow(X(1) - x * d(1) - pow(X(0) - x * d(0), 2), 2) + pow(1 - X(0) + x * d(0), 2);
}
VectorXf adjunctFunctionDerivative(VectorXf X, VectorXf d) {
	VectorXf coefficients(3);
	float coef3 = 400 * pow(d(0), 4);
	float coef2 = (-1200 * X(0) * pow(d(0), 3) + 600 * pow(d(0), 2) * d(1)) / coef3;
	coefficients(2) = coef2;
	float coef1 = (1200 * pow(X(0), 2) * pow(d(0), 2) - 800 * X(0) * d(0) * d(1) - 400 * X(1) * pow(d(0), 2) + 2 * pow(d(0), 2) + 200 * pow(d(1), 2)) / coef3;
	coefficients(1) = coef1;
	float coef0 = (-400 * pow(X(0), 3) * d(0) + 200 * pow(X(0), 2) * d(1) + 400 * X(0) * X(1) * d(0) - 2 * X(0) * d(0) - 200 * X(1) * d(1) + 2 * d(0)) / coef3;
	coefficients(0) = coef0;
	return coefficients;
}
VectorXf eigenvalues(VectorXf X, VectorXf d) {
	MatrixXf matrix(3, 3);
	matrix = MatrixXf::Zero(3, 3);
	for (int i = 0; i < 2; i++) {
		for (int j = 1; j < 3; j++) {
			if (i + 1 == j) {
				matrix(i, j) = 1;
			}
		}
	}
	VectorXf coefficients;
	coefficients = adjunctFunctionDerivative(X, d);
	for (int i = 0; i < 3; i++) {
		matrix(2, i) = -coefficients(i);
	}
	Eigen::EigenSolver<MatrixXf> s(matrix);
	VectorXf eigenvaluesVector = VectorXf::Zero(3);
	for (int i = 0; i < 3; i++) {
		complex<double> l1 = s.eigenvalues()[i];
		if (l1.imag() == 0) {
			double l = l1.real();
			eigenvaluesVector(i) = l;
		}
	}
	return eigenvaluesVector;
}
float minFunction(VectorXf X, VectorXf d) {
	VectorXf eigenvaluesVector;
	eigenvaluesVector = eigenvalues(X, d);
	VectorXf functionValues(3);
	for (int i = 0; i < 3; i++) {
		functionValues(i) = adjunctFunction(X, d, eigenvaluesVector(i));
	}
	float min = functionValues(0);
	int index = 0;
	for (int i = 1; i < 3; i++) {
		if (functionValues(i) < min) {
			min = functionValues(i);
			index = i;
		}
	}
	return eigenvaluesVector(index);
}
int main() {
	VectorXf X(2);
	X(0) = 0;
	X(1) = 0;
	float eps;
	cout << "Assign epsylon value:" << endl;
	cin >> eps;
	VectorXf gradient(2);
	MatrixXf hesian(2, 2);
	MatrixXf m1(2, 2);
	VectorXf d(2);
	gradient = gradientF(X);
	float alpha;
	while (norm(gradient, 2) > eps) {
		hesian = hesianF(X);
		m1 = hesian.inverse();
		d = m1 * gradient;
		alpha = minFunction(X, d);
		X = X - alpha * d;
		gradient = gradientF(X);
	}
	cout << X << endl;
	int k;
	cin >> k;
}