#include "lab3.hpp"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

void computeGridSize(int size, int* p1, int* p2, int N1, int N3);
void generateMatrix(std::vector<int>& matrix, int rows, int cols);
void printMatrix(const std::vector<int>& matrix, int rows, int cols, const std::string& name);
void multiplySubmatrices(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int rowsA, int colsA, int colsB);

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " N1 N2 N3\n";
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N1 = std::atoi(argv[1]);
    int N2 = std::atoi(argv[2]);
    int N3 = std::atoi(argv[3]);

    int p1, p2;
    computeGridSize(size, &p1, &p2, N1, N3);

    int blockRows = N1 / p1;
    int blockCols = N3 / p2;

    std::vector<int> A, B, C(N1 * N3);
    if (rank == 0) {
        A.resize(N1 * N2);
        B.resize(N2 * N3);
        srand(time(nullptr));
        generateMatrix(A, N1, N2);
        generateMatrix(B, N2, N3);
        printMatrix(A, N1, N2, "A");
        printMatrix(B, N2, N3, "B");
    }

    std::vector<int> localA(blockRows * N2);
    std::vector<int> localB(N2 * blockCols);
    std::vector<int> localC(blockRows * blockCols, 0);

    MPI_Comm gridComm;
    int dims[2] = {p1, p2};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &gridComm);

    int coords[2];
    MPI_Cart_coords(gridComm, rank, 2, coords);

    MPI_Comm rowComm, colComm;
    MPI_Comm_split(gridComm, coords[0], rank, &rowComm);
    MPI_Comm_split(gridComm, coords[1], rank, &colComm);

    std::vector<int> sendcounts(size, blockRows * N2);
    std::vector<int> displs(size);

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            displs[i] = i * blockRows * N2;
        }
    }

    MPI_Scatterv(A.data(), sendcounts.data(), displs.data(), MPI_INT,
                 localA.data(), blockRows * N2, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> sendcountsB(size, N2 * blockCols);
    std::vector<int> displsB(size);

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            displsB[i] = i * N2 * blockCols;
        }
    }

    MPI_Scatterv(B.data(), sendcountsB.data(), displsB.data(), MPI_INT,
                 localB.data(), N2 * blockCols, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(localA.data(), blockRows * N2, MPI_INT, 0, colComm);
    MPI_Bcast(localB.data(), N2 * blockCols, MPI_INT, 0, rowComm);

    multiplySubmatrices(localA, localB, localC, blockRows, N2, blockCols);

    std::vector<int> recvcounts(size, blockRows * blockCols);
    std::vector<int> displsC(size);

    if (rank == 0) {
        for (int i = 0; i < size; ++i) {
            displsC[i] = i * blockRows * blockCols;
        }
    }

    std::vector<int> fullC(N1 * N3);
    MPI_Gatherv(localC.data(), blockRows * blockCols, MPI_INT,
                fullC.data(), recvcounts.data(), displsC.data(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printMatrix(fullC, N1, N3, "C");
    }

    MPI_Finalize();
    return 0;
}

void computeGridSize(int size, int* p1, int* p2, int N1, int N3) {
    switch (size) {
        case 1:
            *p1 = 1;
            *p2 = 1;
            break;
        case 2:
            *p1 = (N1 > N3) ? 2 : 1;
            *p2 = (N1 > N3) ? 1 : 2;
            break;
        case 4:
            *p1 = 2;
            *p2 = 2;
            break;
        case 8:
            *p1 = (N1 > N3) ? 4 : 2;
            *p2 = (N1 > N3) ? 2 : 4;
            break;
        case 16:
            *p1 = 4;
            *p2 = 4;
            break;
        default:
            *p1 = 1;
            *p2 = size;
    }
}

void generateMatrix(std::vector<int>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 10;
    }
}

void printMatrix(const std::vector<int>& matrix, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

void multiplySubmatrices(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i * colsB + j] = 0;
            for (int k = 0; k < colsA; ++k) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}
