#ifndef LAB1_SECOND_HPP
#define LAB1_SECOND_HPP

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mpi.h>

#define VALID_ARGC 2
#define TAU 0.001
#define EPSILON 1e-5

void simpleIterationMethod(int argc, char** argv);

void initLocalMatrix(std::vector<double>& localMatrix, int N, int rank, int rowsPerProc, int localCountRows);
void initU(std::vector<double>& u, int N, int rank, int rowsPerProc);
void initB(std::vector<double>& localB, const std::vector<double>& u, const std::vector<double>& localMatrix, int localCountRows, int size, int N, int rank);
double computeResidual(const std::vector<double>& localX, const std::vector<double>& localB, const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size);
void updateLocalX(std::vector<double>& localX, const std::vector<double>& localB, const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size);

void printVector(std::vector<double> vec, int rank, int rowLen);
void printVector(std::vector<double> vec, int rank);
#endif