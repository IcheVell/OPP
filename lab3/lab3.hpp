#ifndef LAB3_HPP
#define LAB3_HPP

#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <mpi.h>

#define VALID_ARGC 4

void matrixMult(int argc, char** argv, int N1, int N2, int N3);
void computeGridSize(int size, int* p1, int* p2, int N1, int N3);
void generateMatrix(std::vector<int>& matrix, int rows, int cols);
void printMatrix(const std::vector<int>& matrix, int rows, int cols, const std::string& name);
void multiplySubmatrices(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int rowsA, int colsA, int colsB);

#endif

