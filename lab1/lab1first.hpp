#ifndef LAB1_FIRST_HPP
#define LAB1_FIRST_HPP

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mpi.h>

const int VALID_ARGC = 2;
const double TAU = 0.001;
const double EPSILON = 1e-5;

void simpleIterationMethod(int argc, char** argv);

void initLocalMatrix(std::vector<double>& localMatrix, int N, int rank, int rowsPerProc, int localCountRows);
void initU(std::vector<double>& u, int N);
void initB(std::vector<double>& b, const std::vector<double>& u, const std::vector<double>& localMatrix, int localCountRows, int size, int N, int rank);
double computeResidual(const std::vector<double>& x, const std::vector<double>& b, const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size);
void updateX(std::vector<double>& x, const std::vector<double>& b, const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size);

void printVector(std::vector<double> vec, int rank, int rowLen);
void printVector(std::vector<double> vec, int rank);
bool checkResult(std::vector<double>& x, std::vector<double>& u, int N);
#endif