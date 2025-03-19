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
        throw std::invalid_argument("Invalid matrix size: " + std::to_string(N));
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
    initU(u, N);
    initB(localB, u, localMatrix, localCountRows, size, N, rank);

    double residual;
    int iter = 0;

    do {
        residual = computeResidual(localX, localB, localMatrix, localCountRows, N, rank, size, extra);
        updateLocalX(localX, localB, localMatrix, localCountRows, N, rank, size, extra);

        // if (rank == 0) {
        //     std::cout << "Iteration " << iter << ", Residual: " << residual << std::endl;
        // }

        iter++;
    } while (residual > EPSILON);

    //std::cout << checkResult(localX, u, localCountRows, rank) << std::endl;

    char procNames[MPI_MAX_PROCESSOR_NAME];
    int nameLength;

    MPI_Get_processor_name(procNames, &nameLength);

    std::cout << "Process " << rank << " of " << size << " on " << procNames << std::endl;

    MPI_Finalize();
}

void initLocalMatrix(std::vector<double>& localMatrix, int N, int rank, int rowsPerProc, int localCountRows) {
    for (int i = 0; i < localCountRows; i++) {
        for (int j = 0; j < N; j++) {
            localMatrix[i * N + j] = ((i + rank * rowsPerProc) == j) ? 2.0 : 1.0;
        }
    }
}

void initU(std::vector<double>& u, int N) {
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

double computeResidual(const std::vector<double>& localX, const std::vector<double>& localB, const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size, int extra) {
    double localResidual = 0.0;
    double localNormB = 0.0;
    std::vector<double> receivedX(localCountRows, 0.0);
    std::vector<double> localAx(localCountRows, 0.0);

    for (int i = 0; i < localCountRows; i++) {
        for (int j = 0; j < localCountRows; j++) {
            localAx[i] += localMatrix[i * N + (j + rank * localCountRows)] * localX[j];
        }
    }

    MPI_Request requests[2];

    for (int step = 0; step < size - 1; step++) {
        int sendTo = (rank + step + 1) % size;
        int recvFrom = (rank - step - 1 + size) % size;

        MPI_Isend(localX.data(), localCountRows, MPI_DOUBLE, sendTo, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(receivedX.data(), localCountRows, MPI_DOUBLE, recvFrom, 0, MPI_COMM_WORLD, &requests[1]);

        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < localCountRows; i++) {
            for (int j = 0; j < localCountRows; j++) {
                localAx[i] += localMatrix[i * N + (j + recvFrom * localCountRows)] * receivedX[j];
            }
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



void updateLocalX(std::vector<double>& localX, const std::vector<double>& localB, const std::vector<double>& localMatrix, int localCountRows, int N, int rank, int size, int extra) {
    std::vector<double> receivedX(localCountRows, 0.0);
    std::vector<double> localAx(localCountRows, 0.0);

    for (int i = 0; i < localCountRows; i++) {
        for (int j = 0; j < localCountRows; j++) {
            localAx[i] += localMatrix[i * N + (j + rank * localCountRows)] * localX[j];
        }
    }

    MPI_Request requests[2];

    for (int step = 0; step < size - 1; step++) {
        int sendTo = (rank + step + 1) % size;
        int recvFrom = (rank - step - 1 + size) % size;

        MPI_Isend(localX.data(), localCountRows, MPI_DOUBLE, sendTo, 0, MPI_COMM_WORLD, &requests[0]);
        MPI_Irecv(receivedX.data(), localCountRows, MPI_DOUBLE, recvFrom, 0, MPI_COMM_WORLD, &requests[1]);

        MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

        for (int i = 0; i < localCountRows; i++) {
            for (int j = 0; j < localCountRows; j++) {
                localAx[i] += localMatrix[i * N + (j + recvFrom * localCountRows)] * receivedX[j];
            }
        }
    }

    for (int i = 0; i < localCountRows; i++) {
        localX[i] -= TAU * (localAx[i] - localB[i]);
    }
}


bool checkResult(std::vector<double>& localX, std::vector<double>& u, int localCountRows, int rank) {
    for (int i = 0; i < localCountRows; i++) {
        if (std::abs(localX[i] - u[i + rank * localCountRows]) > EPSILON) {
            return false;
        }
    }
    return true;
}
