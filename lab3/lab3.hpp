#ifndef LAB3_HPP
#define LAB3_HPP

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <math.h>
#include <mpi.h>

#define VALID_ARGC 4

void generateMatrix(std::vector<int>& matrix, int rows, int cols);
void multiplySubmatrices(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int rowsA, int colsA, int colsB);
void printMatrix(const std::vector<int>& matrix, int rows, int cols, const std::string& name);
bool verifyMatrixMultiplication(const std::vector<int>& A, const std::vector<int>& B, const std::vector<int>& C_parallel, int N1, int N2, int N3);

#endif

