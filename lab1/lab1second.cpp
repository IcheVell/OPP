#include "lab1second.hpp"

int main(int argc, char** argv) {
    if (argc != VALID_ARGC) {
        std::cerr << "Not valid argc. Now " << argc << ", need " << VALID_ARGC;
        return 1;
    }
    try {
        simpleIterationMethod(argc, argv);

    } catch (std::exception& e) {
        std::cerr << e.what();
    }

    return 0;
}

void simpleIterationMethod(int argc, char** argv) {
    int N = std::stoi(argv[1]);
    if (N <= 0) {
        throw std::invalid_argument("Invalid matrix size: " + N);
    }

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int extra = N % size;
    int rowsPerProcess = N / size;
    int localCountRows = (rank == size - 1) ? rowsPerProcess + extra : rowsPerProcess;

    std::vector<double> localMatrix(localCountRows * N);
    std::vector<double> localB(localCountRows, 0);
    std::vector<double> localX(localCountRows, 0);
    std::vector<double> u(N, 0);

    initLocalMatrix(localMatrix, N, rank, rowsPerProcess, localCountRows);
    initU(u, N, rank, rowsPerProcess);
    initB(localB, u, localMatrix, localCountRows, size, N, rank);

    double residual;
    int iter = 0;

    do {
        residual = computeResidual(localX, localB, localMatrix, localCountRows, N, rank, size);

        updateLocalX(localX, localB, localMatrix, localCountRows, N, rank, size);
        
        if (rank == 0) {
            std::cout << "Iteration " << iter << ", Residual: " << residual << std::endl;
        }

        iter++;
    } while (residual > EPSILON);

    MPI_Finalize();
}

void initLocalMatrix(std::vector<double>& localMatrix, int N, int rank, int rowsPerProc, int localCountRows) {
    for (int i = 0; i < localCountRows; i++) {
        for (int j = 0; j < N; j++) {
            localMatrix[i * N + j] = ((i + rank * rowsPerProc) == j) ? 2.0 : 1.0;
        }
    }
} 

void initU(std::vector<double>& u, int N, int rank, int rowsPerProcess) {
    for (int i = 0; i < N; i++) {
        u[i] = sin(2.0 * M_PI * i / N);
        if (std::abs(u[i]) < 1e-15) {
            u[i] = 0.0;
        }
    }
}

void initB(std::vector<double>& localB, const std::vector<double>& u, const std::vector<double>& localMatrix, int localCountRows, int size, int N, int rank) {
    for (int i = 0; i < localCountRows; i++) {
        for (int j = 0; j < N; j++) {
            localB[i] += localMatrix[i * N + j] * u[j];
        }
    }
}


double computeResidual(const std::vector<double>& localX, const std::vector<double>& localB, 
                       const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size) {
    double localResidual = 0.0;
    double localNormB = 0.0;
    
    std::vector<double> globalX(N, 0.0);
    std::vector<int> recvCounts(size, 0);
    std::vector<int> displs(size, 0);

    int rowsPerProc = N / size;
    int extra = N % size;

    for (int i = 0; i < size; i++) {
        recvCounts[i] = rowsPerProc + ((size - 1 == i) ? extra : 0);
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + recvCounts[i - 1]);
    }

    MPI_Allgatherv(localX.data(), localCountRows, MPI_DOUBLE, 
                   globalX.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    std::vector<double> localAx(localCountRows, 0.0);

    for (int i = 0; i < localCountRows; i++) {
        for (int j = 0; j < N; j++) {
            localAx[i] += localMatrix[i * N + j] * globalX[j];
        }
    }

    for (int i = 0; i < localCountRows; i++) {
        localResidual += std::pow(localAx[i] - localB[i], 2);
        localNormB += std::pow(localB[i], 2);
    }

    double globalResidual = 0.0, globalNormB = 0.0;
    MPI_Allreduce(&localResidual, &globalResidual, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localNormB, &globalNormB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return std::sqrt(globalResidual) / std::sqrt(globalNormB);
}


void updateLocalX(std::vector<double>& localX, const std::vector<double>& localB, 
                  const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size) {
    std::vector<double> globalX(N, 0.0);
    std::vector<double> localAx(localCountRows, 0.0);
    
    std::vector<int> recvCounts(size, 0);
    std::vector<int> displs(size, 0);

    int rowsPerProc = N / size;
    int extra = N % size;

    for (int i = 0; i < size; i++) {
        recvCounts[i] = rowsPerProc + ((size - 1 == i) ? extra : 0);
        displs[i] = (i == 0) ? 0 : (displs[i - 1] + recvCounts[i - 1]);
    }

    MPI_Allgatherv(localX.data(), localCountRows, MPI_DOUBLE, 
                   globalX.data(), recvCounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < localCountRows; i++) {
        for (int j = 0; j < N; j++) {
            localAx[i] += localMatrix[i * N + j] * globalX[j];
        }
    }

    for (int i = 0; i < localCountRows; i++) {
        localX[i] = globalX[i + displs[rank]] - TAU * (localAx[i] - localB[i]);
    }
}




void printVector(std::vector<double> vec, int rank) {
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::cout << "Current rank is " << rank << std::endl;

    for (auto it = vec.begin(); it != vec.end(); it++) {
        std::cout << *it << " ";
    }
    
    std::cout << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
}

void printVector(std::vector<double> vec, int rank, int N) {
    MPI_Barrier(MPI_COMM_WORLD);

    std::cout << "Current rank is " << rank << std::endl;
    int counter = 0;
    for (auto it = vec.begin(); it != vec.end(); it++) {
        std::cout << *it << " ";

        if (counter == N - 1) {
            std::cout << std::endl;
            counter = -1;
        }

        counter++;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}