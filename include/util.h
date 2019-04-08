#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Householder>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include <random>
#ifdef USE_MKL
#include <mkl_cblas.h>
#include <mkl_lapacke.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

typedef Eigen::SparseMatrix<double, 0, int> SpMat;
typedef Eigen::VectorBlock<Eigen::Matrix<double, -1,  1, 0, -1,  1>, -1> Segment;
typedef Eigen::Block<Eigen::Matrix<double, -1, -1, 0, -1, -1>, -1, -1> MatrixBlock;

bool are_connected(Eigen::VectorXi &a, Eigen::VectorXi &b, SpMat &A);
bool should_be_disconnected(int lvl1, int lvl2, int sep1, int sep2);
double elapsed(timeval& start, timeval& end);
void swap2perm(Eigen::VectorXi* swap, Eigen::VectorXi* perm);
bool isperm(Eigen::VectorXi* perm);
SpMat symmetric_graph(SpMat& A);

typedef timeval timer;
timer wctime();

/**
 * C <- alpha A   * B   + beta C
 * C <- alpha A^T * B   + beta C
 * C <- alpha A   * B^T + beta C
 * C <- alpha A^T * B^T + beta C
 * Gemm
 */
void gemm(Eigen::MatrixXd* A, Eigen::MatrixXd* B, Eigen::MatrixXd* C, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, double alpha, double beta);

/** Return a new
 * C <- alpha A^(/T) * B^(/T)
 **/
Eigen::MatrixXd* gemm_new(Eigen::MatrixXd* A, Eigen::MatrixXd* B, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, double alpha);

/**
 * C <- C - A * A^T
 */
void syrk(Eigen::MatrixXd* A, Eigen::MatrixXd* C);

/** 
 * A <- L, L L^T = A
 * Return != 0 if potf failed (not spd)
 */ 
int potf(Eigen::MatrixXd* A);

/**
 * A <- [L\U] (lower and upper)
 * p <- swap (NOT a permutation)
 * A[p] = L*U
 * L is unit diagonal
 * U is not
 * Return != 0 if getf failed (singular)
 */
int getf(Eigen::MatrixXd* A, Eigen::VectorXi* swap);

/**
 * Compute an estimated 1-norm condition number of A using its LU or Cholesky factorization
 */
double rcond_1_getf(Eigen::MatrixXd* A_LU, double A_1_norm);
double rcond_1_potf(Eigen::MatrixXd* A_LLT, double A_1_norm);

/**
 * B <- B * L^(-1)
 * B <- B * L^(-T)
 * B <- B * U^(-1)
 * B <- B * U^(-T)
 */
void trsm_right(Eigen::MatrixXd* L, Eigen::MatrixXd* B, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag);

/**
 * B <- L^(-1) * B
 * B <- L^(-T) * B
 * B <- U^(-1) * B
 * B <- U^(-T) * B
 */
void trsm_left(Eigen::MatrixXd* L, Eigen::MatrixXd* B, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag);

/**
 * x <- L^(-1) * x
 * x <- L^(-T) * x
 * x <- U^(-1) * x
 * x <- U^(-T) * x
 */
void trsv(Eigen::MatrixXd* L, Segment* x, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag);

/**
 * x <- L^T * x
 */
void trmv_trans(Eigen::MatrixXd* L, Segment* x);

/**
 * A <- L^T * A
 */
void trmm_trans(Eigen::MatrixXd* L, Eigen::MatrixXd* A);

/**
 * x2 <- x2 - A21 * x1
 */
void gemv_notrans(Eigen::MatrixXd* A21, Segment* x1, Segment* x2);

/**
 * x2 <- x2 - A12^T * x1
 */
void gemv_trans(Eigen::MatrixXd* A12, Segment* x1, Segment* x2);

/**
 * AP = QR
 */
void geqp3(Eigen::MatrixXd* A, Eigen::VectorXi* jpvt, Eigen::VectorXd* tau);

/**
 * A = U S VT
 * Compute the full SVD, where if A is mxn, U is mxm, V is nxn, and S is min(M,N)
 * VT is V^T, *not* V.
 */
void gesvd(Eigen::MatrixXd* A, Eigen::MatrixXd* U, Eigen::VectorXd* S, Eigen::MatrixXd* VT);

/**
 * x <- Q * x
 * A <- Q * A
 */
void ormqr_notrans(Eigen::MatrixXd* v, Eigen::VectorXd* h, Segment* x);
void ormqr_notrans_left(Eigen::MatrixXd* v, Eigen::VectorXd* h, Eigen::MatrixXd* A);

/**
 * x <- Q^T * x
 * A <- Q^T * A
 */
void ormqr_trans(Eigen::MatrixXd* v, Eigen::VectorXd* h, Segment* x);
void ormqr_trans_left(Eigen::MatrixXd* v, Eigen::VectorXd* h, Eigen::MatrixXd* A);

/**
 * A <- A   * Q
 * A <- A   * Q^T
 * A <- Q   * A
 * A <- Q^T * A
 */
void ormqr(Eigen::MatrixXd* v, Eigen::VectorXd* h, Eigen::MatrixXd* A, char side, char trans);

/**
 * Create the thin Q
 */
void orgqr(Eigen::MatrixXd* v, Eigen::VectorXd* h);

/** 
 * Create a square Q
 */
// void orgqr_fat(Eigen::MatrixXd* v, Eigen::VectorXd* h);

/**
 * A = QR
 */
void geqrf(Eigen::MatrixXd* A, Eigen::VectorXd* tau);

int choose_rank(Eigen::VectorXd& s, double tol);

std::size_t hashv(std::vector<size_t> vals);

// Hash function for Eigen matrix and vector.
// The code is from `hash_combine` function of the Boost library. See
// http://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine .
template<typename T>
struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
    // Note that it is oblivious to the storage order of Eigen matrix (column- or
    // row-major). It will give you the same hash value for two different matrices if they
    // are the transpose of each other in different storage order.
    size_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

void block2dense(Eigen::VectorXi &rowval, Eigen::VectorXi &colptr, Eigen::VectorXd &nnzval, int i, int j, int li, int lj, Eigen::MatrixXd *dst, bool transpose);

Eigen::MatrixXd linspace_nd(int n, int dim);

// Returns A[p,p]
SpMat symm_perm(SpMat &A, Eigen::VectorXi &p);

// Random vector with seed
Eigen::VectorXd random(int size, int seed);

Eigen::MatrixXd random(int rows, int cols, int seed);

// Print vector
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    for(auto v_ : v) {
        os << v_ << " " ;
    }
    os << std::endl;
    return os;
}

#endif
