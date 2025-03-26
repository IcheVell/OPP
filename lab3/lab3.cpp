#include "lab3.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    double startTime = 0.0, endTime = 0.0;
    startTime = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " N1 N2 N3\n";
        }
        MPI_Finalize();
        return 1;
    }

    int N1 = std::atoi(argv[1]);
    int N2 = std::atoi(argv[2]);
    int N3 = std::atoi(argv[3]);

    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int p1 = dims[0];  
    int p2 = dims[1];  

    int blockRows = N1 / p1;
    int blockCols = N3 / p2;

    std::vector<int> A, B, C(N1 * N3, 0);
    if (rank == 0) {
        A.resize(N1 * N2);
        B.resize(N2 * N3);
        srand(time(nullptr));
        generateMatrix(A, N1, N2);
        generateMatrix(B, N2, N3);
    }

    std::vector<int> localA(blockRows * N2, 0);
    std::vector<int> localB(N2 * blockCols, 0);
    std::vector<int> localC(blockRows * blockCols, 0);

    int periods[2] = {0, 0};
    MPI_Comm gridComm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &gridComm);

    int coords[2];
    MPI_Cart_coords(gridComm, rank, 2, coords);

    MPI_Comm rowComm, colComm;
    MPI_Comm_split(gridComm, coords[0], rank, &rowComm);
    MPI_Comm_split(gridComm, coords[1], rank, &colComm);

    std::vector<int> sendcountsA(p1, 0);
    std::vector<int> displsA(p1, 0);
    std::vector<int> sendcountsB(p2, 0);
    std::vector<int> displsB(p2, 0);

    if (rank == 0) {
        for (int i = 0; i < p1; i++) {
            sendcountsA[i] = blockRows * N2;
            displsA[i] = i * blockRows * N2;
        }
        for (int j = 0; j < p2; j++) {
            sendcountsB[j] = 1;  
            displsB[j] = j;      
        }
    }

    if (coords[1] == 0) {
        MPI_Scatterv(A.data(), sendcountsA.data(), displsA.data(), MPI_INT, localA.data(), blockRows * N2, MPI_INT, 0, colComm);
    }

    MPI_Datatype columnType, columnTypeResized;
    MPI_Type_vector(N2, blockCols, N3, MPI_INT, &columnType);
    MPI_Type_create_resized(columnType, 0, blockCols * sizeof(int), &columnTypeResized);
    MPI_Type_commit(&columnTypeResized);
    MPI_Type_free(&columnType); 

    if (coords[0] == 0) {
        MPI_Scatter(B.data(), 1, columnTypeResized, localB.data(), N2 * blockCols, MPI_INT, 0, rowComm);
    }

    MPI_Type_free(&columnTypeResized);

    MPI_Bcast(localA.data(), blockRows * N2, MPI_INT, 0, rowComm);
    MPI_Bcast(localB.data(), N2 * blockCols, MPI_INT, 0, colComm);

    multiplySubmatrices(localA, localB, localC, blockRows, N2, blockCols);

    std::vector<int> fullC(N1 * N3);

    MPI_Datatype locC, locCResized;
    int sizes[2] = { N1, N3 };  
    int subsizes[2] = { blockRows, blockCols };
    int starts[2] = { 0, 0 };  
    MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_INT, &locC);
    MPI_Type_create_resized(locC, 0, sizeof(int), &locCResized);
    MPI_Type_commit(&locCResized);
    MPI_Type_free(&locC); 

    std::vector<int> recvcounts(size, 1);
    std::vector<int> displsC(size, 0);

    for (int i = 0; i < p1; ++i) {
        for (int j = 0; j < p2; ++j) {
            int idx = i * p2 + j;
            displsC[idx] = i * blockRows * N3 + j * blockCols;
        }
    }

    MPI_Gatherv(localC.data(), blockRows * blockCols, MPI_INT, fullC.data(), recvcounts.data(), displsC.data(), locCResized, 0, gridComm);

    MPI_Type_free(&locCResized);

    endTime = MPI_Wtime();
 

    if (rank == 0) {
        bool correct = verifyMatrixMultiplication(A, B, fullC, N1, N2, N3);
        std::cout << (correct ? "Correct result!" : "Incorrect result!") << std::endl;
        std::cout << "Exec time: " << (endTime - startTime) << " sec" << std::endl;
    }

    MPI_Finalize();
    return 0;
}

void generateMatrix(std::vector<int>& matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = rand() % 10;
    }
}

void multiplySubmatrices(const std::vector<int>& A, const std::vector<int>& B, std::vector<int>& C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i * colsB + j] = 0;
            for (int k = 0; k < colsA; k++) {
                C[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
}

void printMatrix(const std::vector<int>& matrix, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

bool verifyMatrixMultiplication(const std::vector<int>& A, const std::vector<int>& B, const std::vector<int>& C_parallel, int N1, int N2, int N3) {
    std::vector<int> C_serial(N1 * N3, 0);

    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N3; j++) {
            for (int k = 0; k < N2; k++) {
                C_serial[i * N3 + j] += A[i * N2 + k] * B[k * N3 + j];
            }
        }
    }

    for (int i = 0; i < N1 * N3; i++) {
        if (C_serial[i] != C_parallel[i]) {
            return false;
        }
    }

    return true;
}
