#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <stdio.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Householder>
#include <Eigen/LU>
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

#include "spaND.h"

namespace spaND {

bool are_connected(Eigen::VectorXi &a, Eigen::VectorXi &b, SpMat &A);
bool should_be_disconnected(int lvl1, int lvl2, int sep1, int sep2);
double elapsed(timeval& start, timeval& end);
void swap2perm(Eigen::VectorXi* swap, Eigen::VectorXi* perm);
bool isperm(const Eigen::VectorXi* perm);
Eigen::VectorXi invperm(const Eigen::VectorXi& perm);
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
 * C <- alpha A * A^T + beta C (or A^T * A)
 */
void syrk(Eigen::MatrixXd* A, Eigen::MatrixXd* C, CBLAS_TRANSPOSE tA, double alpha, double beta);

/** 
 * A <- L, L L^T = A
 * Return != 0 if potf failed (not spd)
 */ 
int potf(Eigen::MatrixXd* A);

int ldlt(Eigen::MatrixXd* A, Eigen::MatrixXd* L, Eigen::VectorXd* d, Eigen::VectorXi* p, double* rcond);

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
 * A = P L U Q
 * A <- [L\U] (lower and upper)
 * L is unit diagonal
 * U is not
 */
void fpgetf(Eigen::MatrixXd* A, Eigen::VectorXi* p, Eigen::VectorXi* q);

/**
 * A = L U
 * breaks down into L & U, and split the diagonal between the two
 * A = L * (D + U') = L * D (I + D^-1 U') = { L * |D|^1/2 } { |D|^1/2 sign(D) (I + D^-1 U') }
 */
void split_LU(Eigen::MatrixXd* A, Eigen::MatrixXd* L, Eigen::MatrixXd* U);

/**
 * Compute an estimated 1-norm condition number of A using its LU or Cholesky factorization
 */
double rcond_1_getf(Eigen::MatrixXd* A_LU, double A_1_norm);
double rcond_1_potf(Eigen::MatrixXd* A_LLT, double A_1_norm);
double rcond_1_trcon(Eigen::MatrixXd* LU, char uplo, char diag);

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
 * A = U S U^T 
 * A <- U
 * Full Symmetric EVD
 */
void syev(Eigen::MatrixXd* A, Eigen::VectorXd* S);

/**
 * x <- Q * x
 */
void ormqr_notrans(Eigen::MatrixXd* v, Eigen::VectorXd* h, Segment* x);

/**
 * x <- Q^T * x
 * A <- Q^T * A
 */
void ormqr_trans(Eigen::MatrixXd* v, Eigen::VectorXd* h, Segment* x);

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

// Set Eigen::MatrixXd to zero using memset
void setZero(Eigen::MatrixXd* A);

// Print vector
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    for(auto v_ : v) {
        os << v_ << " " ;
    }
    os << std::endl;
    return os;
}

// Range
template<typename T>
struct ItRange
{
    public:
        ItRange(const T& begin, const T& end) : begin_(begin), end_(end) {}
        const T& begin() const {
            return this->begin_;
        }
        const T& end() {
            return this->end_;
        }
    private:
        T begin_;
        T end_;
};

/**
 * Statistics and Logging
 **/

struct Profile {
    public:
        double elim;
        double scale;
        double spars;
        double merge;

        double elim_pivot;
        double elim_panel;
        double elim_schur;

        double scale_pivot;
        double scale_panel;

        double spars_assmb;
        double spars_spars;
        double spars_scatt;

        double merge_alloc;
        double merge_copy;

        double potf;
        double ldlt;
        double trsm;
        double gemm;
        double geqp3;
        double geqrf;
        double syev;
        double gesvd;
        double getf;

        double buildq;
        double scattq;
        double perma;
        double scatta;
        double assmb;
        double phi;

        Profile() :
            elim(0), scale(0), spars(0), merge(0), 
            elim_pivot(0), elim_panel(0), elim_schur(0),
            scale_pivot(0), scale_panel(0), 
            spars_assmb(0), spars_spars(0), spars_scatt(0),
            merge_alloc(0), merge_copy(0),
            potf(0), ldlt(0), trsm(0), gemm(0), geqp3(0), geqrf(0), syev(0), gesvd(0), getf(0),
            buildq(0), scattq(0), perma(0), scatta(0), assmb(0), phi(0)
            {}
};

struct PivotFlops {
    long rows;
    double time;
};

struct PanelFlops {
    long rows;
    long cols;
    double time;
};

struct GemmFlops {
    long rows;
    long cols;
    long inner;
    double time;
};

struct RRQRFlops {
    long rows;
    long cols;
    double time;
};

struct ProfileFlops {
    public:
        std::vector<PivotFlops> pivot;
        std::vector<PanelFlops> panel;
        std::vector<GemmFlops>  gemm;
        std::vector<RRQRFlops>  rrqr;
        ProfileFlops() {};
};

template<typename T>
struct Stats {
    private:
        T min;
        T max;
        T sum;
        int count;
    public:
        Stats(): min(std::numeric_limits<T>::max()), max(std::numeric_limits<T>::lowest()), sum(0), count(0) {};
        void addData(T value) {
            this->min = (this->min < value ? this->min : value);
            this->max = (this->max > value ? this->max : value);
            this->sum += value;
            this->count += 1;
        }
        T      getMin()   const { return this->count == 0 ? 0 : this->min; }
        T      getMax()   const { return this->count == 0 ? 0 : this->max; }
        double getMean()  const { return ((double)this->sum)/((double)this->count); }
        int    getCount() const { return this->count; }
        T      getSum()   const { return this->sum; }
};

struct Log {
    public:
        int dofs_nd;
        int dofs_left_nd;
        int dofs_left_elim;
        int dofs_left_spars;
        long long int fact_nnz;
        Stats<int> rank_before;
        Stats<int> rank_after;
        int ignored;
        Stats<int> nbrs;

        Stats<double> cond_diag_elim;
        Stats<double> norm_diag_elim;
        Stats<double> cond_U_elim;
        Stats<double> cond_L_elim;

        Stats<double> cond_diag_scal;
        Stats<double> norm_diag_scal;
        Stats<double> cond_U_scal;
        Stats<double> cond_L_scal;

        double Asym_before_scaling;
        double Asym_after_scaling;        
        Stats<double> Anorm_before_lower;
        Stats<double> Anorm_before_upper;
        Stats<double> Anorm_after_lower;
        Stats<double> Anorm_after_upper;
    
        Log() :
            dofs_nd(0),
            dofs_left_nd(0), dofs_left_elim(0), dofs_left_spars(0),
            fact_nnz(0), ignored(0),
            Asym_before_scaling(0), Asym_after_scaling(0)
            {}
};

}

#endif
