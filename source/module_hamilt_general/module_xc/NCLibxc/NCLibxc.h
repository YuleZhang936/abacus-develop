#ifndef NCLIBXC_H
#define NCLIBXC_H

#ifdef USE_LIBXC

#include <vector>
#include <array>
#include <complex>
#include <utility>

// 2x2 complex matrix
using Matrix2x2 = std::array<std::array<std::complex<double>, 2>, 2>;

class NCLibxc {
public:
    // Pauli matrices
    static const Matrix2x2 sigma_x;
    static const Matrix2x2 sigma_y;
    static const Matrix2x2 sigma_z;

    static Matrix2x2 add(const Matrix2x2 &a, const Matrix2x2 &b);

    static Matrix2x2 scalar_multiply(const std::complex<double> &scalar, const Matrix2x2 &matrix);


    // build a Pauli matrix from Cartesian coordinates
    static Matrix2x2 construct_pauli_matrix(double x, double y, double z);

    // identity matrix
    static Matrix2x2 identity_matrix();

    // post-processing of libxc. get partial derivatives from libxc and integrate them to get the 0th,1st,2nd derivatives of the functional
    static void postlibxc_lda(int xc_id, const std::vector<double>& rho_up, const std::vector<double>& rho_down, 
                              std::vector<double>& e, std::vector<double>& v1, std::vector<double>& v2, 
                              std::vector<double>& f1, std::vector<double>& f2, std::vector<double>& f3);

    // returns energy density & potential at each grid point
    static std::pair<std::vector<double>, std::vector<Matrix2x2>> lda_mc(int xc_id, const std::vector<double>& n, 
                                                                  const std::vector<double>& mx, const std::vector<double>& my, const std::vector<double>& mz);

    // returns energy density & potential at each grid point
    static std::pair<std::vector<double>, std::vector<Matrix2x2>> lda_lc(int xc_id, const std::vector<double>& n, 
                                                                  const std::vector<double>& mx, const std::vector<double>& my, const std::vector<double>& mz);
    // print message and citation of the program
    static void print_NCLibxc();
};

std::vector<std::array<double, 4>> MakeAngularGrid(int grid_level);

#endif // USE_LIBXC

#endif // NCLIBXC_H
