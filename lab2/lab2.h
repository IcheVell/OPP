#ifndef LAB1_H
#define LAB1_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define MIN_COUNT_ARGS 2
#define EPSILON 0.0001
#define TAY 0.01
#define MAX_COUNT_ITERATIONS 10000000

void simpleIterationMethod(char** argv);

double** getMatrix(int N);
void initMatrix(double** A, int N);
void freeMatrix(double** A, int N);
void initBDefaultVector(double* b, int N);
void initBRandomVector(double** A, double* b, double* u, int N);
void initUVector(double* u, int N);
void initZeroVector(double* x, int N);
void printMatrix(double** A, int N);
void printVector(double* vect, int N);

double calculateDoubleNorm(double* tempVector, double** A, double* x, double* b, int N);
void matrixMult(double* tempVector, double** A, double* x, int N);
void vectorSub(double* tempVector, double* b, int N);
void scalarMult(double* tempVector, double scalar, int N);
void calculateNewIteration(double* tempVector, double** A, double* x, double* b, int N);

#endif