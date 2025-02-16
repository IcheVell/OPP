#include "lab2.h"

int main(int argc, char** argv) {
    if (argc < MIN_COUNT_ARGS) {
        fprintf(stderr, "Invalid argc: %d, must be more than 2\n", argc);
        return 1;
    }

    simpleIterationMethod(argv);
    return 0;
}

void simpleIterationMethod(char** argv) {
    int N = atoi(argv[1]);
    if (N == 0) {
        fprintf(stderr, "Invalid matrix size: %s\n", argv[1]);
        return;
    }

    double** A = getMatrix(N);
    double* u = (double*)malloc(sizeof(double) * N);
    double* b = (double*)malloc(sizeof(double) * N);
    double* x = (double*)malloc(sizeof(double) * N);
    double* tempVector = (double*)malloc(sizeof(double) * N);

    initMatrix(A, N);
    initUVector(u, N);
    initBRandomVector(A, b, u, N);
    initZeroVector(x, N);
    initZeroVector(tempVector, N);

    while ((calculateDoubleNorm(tempVector, A, x, b, N) > EPSILON)) {
        calculateNewIteration(tempVector, A, x, b, N);
    }

    freeMatrix(A, N);
    free(b);
    free(x);
    free(u);
    free(tempVector);
}

double** getMatrix(int N) {
    double** matrix = (double**)malloc(sizeof(double*) * N);
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        matrix[i] = (double*)malloc(sizeof(double) * N);
    }
    return matrix;
}

void initMatrix(double** matrix, int N) {
    #pragma omp parallel for collapse(2) schedule(guided)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                matrix[i][j] = 2.0;
            } else {
                matrix[i][j] = 1.0;
            }
        }
    }
}

void freeMatrix(double** matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void initBDefaultVector(double* b, int N) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        b[i] = N + 1;
    }
}

void initBRandomVector(double** A, double* b, double* u, int N) {
    matrixMult(b, A, u, N);
}

void initZeroVector(double* x, int N) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        x[i] = 0;
    }
}

void initUVector(double* u, int N) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        u[i] = sin(2 * M_PI * i / N);
    }
}

double calculateDoubleNorm(double* tempVector, double** A, double* x, double* b, int N) {
    matrixMult(tempVector, A, x, N);
    vectorSub(tempVector, b, N);

    double upSum = 0, downSum = 0;
    #pragma omp parallel for reduction(+:upSum, downSum) schedule(guided)
    for (int i = 0; i < N; i++) {
        upSum += tempVector[i] * tempVector[i]; 
        downSum += b[i] * b[i];
    }

    return (sqrt(upSum) / sqrt(downSum));
}

void matrixMult(double* tempVector, double** A, double* x, int N) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        double sum = 0;
        for (int j = 0; j < N; j++) {
            sum += A[i][j] * x[j];
        }
        tempVector[i] = sum;
    }
}

double calculateTAY(double** A, int N) {
    double maxSum = 0;
    #pragma omp parallel for reduction(max:maxSum) schedule(guided)
    for (int i = 0; i < N; i++) {
        double rowSum = 0;
        for (int j = 0; j < N; j++) {
            rowSum += fabs(A[i][j]);
        }
        if (rowSum > maxSum) {
            maxSum = rowSum;
        }
    }
    return 1.0 / maxSum; 
}

void vectorSub(double* tempVector, double* b, int N) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        tempVector[i] -= b[i];
    }
}


void scalarMult(double* tempVector, double scalar, int N) {
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        tempVector[i] *= scalar;
    }
}

void calculateNewIteration(double* tempVector, double** A, double* x, double* b, int N) {
    matrixMult(tempVector, A, x, N);
    vectorSub(tempVector, b, N);
    scalarMult(tempVector, calculateTAY(A, N), N);
    vectorSub(x, tempVector, N);
}


void printMatrix(double** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", A[i][j]);
        }
        printf("\n");
    }
}

void printVector(double* vect, int N) {
    for (int i = 0; i < N; i++) {
        printf("%lf ", vect[i]);
    }
    printf("\n");
}
