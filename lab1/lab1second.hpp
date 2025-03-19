#ifndef LAB1SECOND_HPP
#define LAB1SECOND_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <mpi.h>

const double EPSILON = 1e-5;
const double TAU = 0.001;
const int VALID_ARGC = 2;
const int MAX_COUNT_PROCESS = 16;

void simpleIterationMethod(int argc, char** argv);
void initLocalMatrix(std::vector<double>& localMatrix, int N, int rank, int rowsPerProc, int localCountRows);
void initU(std::vector<double>& u, int N);
void initB(std::vector<double>& localB, const std::vector<double>& u, const std::vector<double>& localMatrix, int localCountRows, int size, int N, int rank);
double computeResidual(const std::vector<double>& localX, const std::vector<double>& localB, const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size, int extra);
void updateLocalX(std::vector<double>& localX, const std::vector<double>& localB, const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size, int extra);
bool checkResult(std::vector<double>& localX, std::vector<double>& u, int localCountRows, int rank);

#endif 
