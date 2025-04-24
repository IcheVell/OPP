#include "lab4rhoonce.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0)
            std::cerr << "Usage: " << argv[0] << " Nx Ny Nz\n";
        MPI_Finalize();
        return 1;
    }

    const int Nx = std::atoi(argv[1]);
    const int Ny = std::atoi(argv[2]);
    const int Nz = std::atoi(argv[3]);

    const double hx = Dx / (Nx - 1);
    const double hy = Dy / (Ny - 1);
    const double hz = Dz / (Nz - 1);

    const int Nz_local = Nz / size;

    if (Nz % size != 0) {
        std::cerr << "Invalid Nz: " << Nz;
        MPI_Finalize();
        return 1;
    }

    const int k_start = rank * Nz_local;

    double start_time = MPI_Wtime();

    std::vector<double> phi_old((Nz_local + 2) * Ny * Nx, 0.0);
    std::vector<double> phi_new((Nz_local + 2) * Ny * Nx, 0.0);
    std::vector<double> rho_grid((Nz_local + 2) * Ny * Nx, 0.0);
    for (int k = 0; k < Nz_local + 2; ++k) {
        int gk = k_start + k - 1;
        if (gk < 0 || gk >= Nz) continue;
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                double x = x_start + i * hx;
                double y = y_start + j * hy;
                double z = z_start + gk * hz;
                rho_grid[get_idx(i, j, k, Nx, Ny)] = rho(x, y, z);
            }
        }
    }

    for (int k = 0; k < Nz_local + 2; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int k_global = k_start + k - 1;
                if (is_boundary(i, j, k_global, Nx, Ny, Nz)) {
                    double x = x_start + i * hx;
                    double y = y_start + j * hy;
                    double z = z_start + k_global * hz;
                    phi_old[get_idx(i, j, k, Nx, Ny)] = exact_phi(x, y, z);
                }
            }
        }
    }

    int iter = 0;
    double max_diff = 0.0;
    do {
        MPI_Request reqs[4];

        if (rank > 0) {
            MPI_Irecv(&phi_old[get_idx(0,0,0,Nx,Ny)], Nx*Ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(&phi_old[get_idx(0,0,1,Nx,Ny)], Nx*Ny, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, &reqs[1]);
        }
        if (rank < size-1) {
            MPI_Irecv(&phi_old[get_idx(0,0,Nz_local+1,Nx,Ny)], Nx*Ny, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD, &reqs[2]);
            MPI_Isend(&phi_old[get_idx(0,0,Nz_local,Nx,Ny)], Nx*Ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &reqs[3]);
        }   

        phi_new = phi_old;

        double invA = 1.0 / (2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + a);

        for (int k = 2; k <= Nz_local-1; ++k) {
            for (int j = 1; j <= Ny-2; ++j) {
                for (int i = 1; i <= Nx-2; ++i) {
                    phi_new[get_idx(i,j,k,Nx,Ny)] = compute_next_phi(phi_old, rho_grid, i, j, k, hx, hy, hz, Nx, Ny, invA, rank, Nz_local);
                }
            }
        }

        if (rank > 0) {
            MPI_Waitall(2, &reqs[0], MPI_STATUSES_IGNORE);
        }
        if (rank < size-1) {
            MPI_Waitall(2, &reqs[2], MPI_STATUSES_IGNORE);
        }

        if (rank > 0) {
            int k = 1;
            for (int j = 1; j <= Ny-2; ++j) {
                for (int i = 1; i <= Nx-2; ++i) {
                    int id = get_idx(i,j,k, Nx, Ny);
                    phi_new[id] = compute_next_phi(phi_old, rho_grid, i,j,k, hx,hy,hz, Nx,Ny, invA, rank, Nz_local);
                }
            }
        }

        if (rank < size-1) {
            int k = Nz_local;
            for (int j = 1; j <= Ny-2; ++j) {
                for (int i = 1; i <= Nx-2; ++i) {
                    int id = get_idx(i,j,k, Nx, Ny);
                    phi_new[id] = compute_next_phi(phi_old, rho_grid, i,j,k, hx,hy,hz, Nx,Ny, invA, rank, Nz_local);
                }
            }
        }

        double local_max = 0.0;
        for (int k = 1; k <= Nz_local; ++k) {
            for (int j = 1; j <= Ny-2; ++j) {
                for (int i = 1; i <= Nx-2; ++i) {
                    int id = get_idx(i,j,k,Nx,Ny);
                    local_max = std::max(local_max, std::abs(phi_new[id] - phi_old[id]));
                }
            }
        }

        MPI_Allreduce(&local_max, &max_diff, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        phi_old.swap(phi_new);
        ++iter;
    } while (max_diff > epsilon);


    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Converged in " << iter << " iterations\n";
        std::cout << "Elapsed time: " << end_time - start_time << " sec\n";

    }

    check_against_exact_solution(phi_new, Nx, Ny, Nz_local, rank, k_start, hx, hy, hz, epsilon, size);

    MPI_Finalize();
    return 0;
}

double exact_phi(double x, double y, double z) {
    return x * x + y * y + z * z;
}

double rho(double x, double y, double z) {
    return 6.0 - a * exact_phi(x, y, z);
}

int get_idx(int i, int j, int k, int Nx, int Ny) {
    return k * Ny * Nx + j * Nx + i;
}

bool is_boundary(int i, int j, int k_global, int Nx, int Ny, int Nz) {
    return i == 0 || i == Nx - 1 || j == 0 || j == Ny - 1 || k_global == 0 || k_global == Nz - 1;
}

double compute_next_phi(const std::vector<double>& phi_old, const std::vector<double>& rho_grid, int i, int j, int k, double hx, double hy, double hz, int Nx, int Ny, double invA, int rank, int layer_height) {
    int idx = get_idx(i, j, k, Nx, Ny);
    double x_comp = (phi_old[idx+1] + phi_old[idx-1]) / (hx*hx);
    x_comp = (phi_old[get_idx(i+1,j,k,Nx,Ny)] + phi_old[get_idx(i-1,j,k,Nx,Ny)])/(hx*hx);
    double y_comp = (phi_old[get_idx(i,j+1,k,Nx,Ny)] + phi_old[get_idx(i,j-1,k,Nx,Ny)])/(hy*hy);
    double z_comp = (phi_old[get_idx(i,j,k+1,Nx,Ny)] + phi_old[get_idx(i,j,k-1,Nx,Ny)])/(hz*hz);

    double rhs = rho_grid[idx];
    return invA*(x_comp + y_comp + z_comp - rhs);
}


void check_against_exact_solution(const std::vector<double>& phi, int Nx, int Ny, int Nz_local, int rank, int k_start, double hx, double hy, double hz, double epsilon, int size) {
    bool local_fail = false;

    for (int k = 1; k <= Nz_local && !local_fail; ++k) {
        int global_k = k_start + k - 1;  
        for (int j = 1; j < Ny-1 && !local_fail; ++j) {
            for (int i = 1; i < Nx-1; ++i) {

                int id = get_idx(i, j, k, Nx, Ny);
                double x = x_start + i * hx;
                double y = y_start + j * hy;
                double z = z_start + global_k * hz;

                double exact = exact_phi(x, y, z);
                double approx = phi[id];
                if (std::abs(approx - exact) > epsilon * 2.5) {
                    local_fail = true;
                    break;
                }
            }
        }
    }

    int global_fail = 0;
    MPI_Allreduce(&local_fail, &global_fail, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);

    if (rank == 0) {
        if (global_fail)
            std::cerr << "ОШИБКА: найдено расхождение более epsilon\n";
        else
            std::cout << "Успех: все внутренние узлы в пределах epsilon\n";
    }
}


