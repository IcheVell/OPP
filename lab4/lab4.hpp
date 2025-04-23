#ifndef LAB4_HPP
#define LAB4_HPP

#include <mpi.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <functional>

constexpr double x_start = -1.0, y_start = -1.0, z_start = -1.0;
constexpr double Dx = 2.0, Dy = 2.0, Dz = 2.0;
constexpr double a = 1e5;
constexpr double epsilon = 1e-8;

double rho(double x, double y, double z);
double exact_phi(double x, double y, double z);
int get_idx(int i, int j, int k, int Nx, int Ny);
bool is_boundary(int i, int j, int k_global, int Nx, int Ny, int Nz);
double compute_next_phi(const std::vector<double>& phi_old, int i, int j, int k, double hx, double hy, double hz, int Nx, int Ny, double invA, int rank, int layer_height);
void check_against_exact_solution(const std::vector<double>& phi_new, int Nx, int Ny, int Nz_local, int rank, int k_start, double hx, double hy, double hz, double epsilon, int size);

#endif