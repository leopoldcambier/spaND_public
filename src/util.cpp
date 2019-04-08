#include "util.h"

using namespace Eigen;
using namespace std;

bool are_connected(VectorXi &a, VectorXi &b, SpMat &A) {
    int  bsize = b.size();
    auto b_begin = b.data();
    auto b_end   = b.data() + b.size();
    for(int ia = 0; ia < a.size(); ia++) {
        // Look at all the neighbors of ia
        int node = a[ia];
        for(SpMat::InnerIterator it(A,node); it; ++it) {
            auto neigh = it.row();
            auto id = lower_bound(b_begin, b_end, neigh);
            int pos = id - b_begin;
            if(pos < bsize && b[pos] == neigh) // Found one in b! They are connected.
                return true;
        }
    }
    return false;
}

// lvl=0=leaf
// assumes a ND binary tree
bool should_be_disconnected(int lvl1, int lvl2, int sep1, int sep2) {
    while (lvl2 > lvl1) {
        lvl1 += 1;
        sep1 /= 2;
    }
    while (lvl1 > lvl2) {
        lvl2 += 1;
        sep2 /= 2;
    }
    if (sep1 != sep2) {
        return true;
    } else {
        return false;
    }
}

/** 
 * Given A, returns |A|+|A^T|+I
 */
SpMat symmetric_graph(SpMat& A) {
    assert(A.rows() == A.cols());
    int n = A.rows();
    vector<Triplet<double>> vals(2 * A.nonZeros() + n);
    int l = 0;
    for (int k=0; k < A.outerSize(); ++k) {
        vals[l++] = Triplet<double>(k, k, 1.0);
        for (SpMat::InnerIterator it(A,k); it; ++it) {
            vals[l++] = Triplet<double>(it.col(), it.row(), abs(it.value()));
            vals[l++] = Triplet<double>(it.row(), it.col(), abs(it.value()));
        }
    }
    assert(l == vals.size());
    SpMat AAT(n,n);
    AAT.setFromTriplets(vals.begin(), vals.end());
    return AAT;
}

double elapsed(timeval& start, timeval& end) {
    return (end.tv_sec + end.tv_usec * 1e-6) - (start.tv_sec + start.tv_usec * 1e-6);
}

timer wctime() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return time;
}

// All are base-0
void swap2perm(Eigen::VectorXi* swap, Eigen::VectorXi* perm) {
    int n = perm->size();
    assert(swap->size() == n);
    for(int i = 0; i < n; i++) {
        (*perm)[i] = i;
    }
    for(int i = 0; i < n; i++) {
        int ipiv = (*swap)[i];
        int tmp = (*perm)[ipiv];
        (*perm)[ipiv] = (*perm)[i];
        (*perm)[i] = tmp;
    }
}

bool isperm(Eigen::VectorXi* perm) {
    int n = perm->size();
    VectorXi count = VectorXi::Zero(n);
    for(int i = 0;i < n; i++) {
        int pi = (*perm)[i];
        if(pi < 0 || pi >= n) { return false; }
        count[pi] += 1;
    }
    return (count.cwiseEqual(1)).all();
}

size_t hashv(vector<size_t> vals) {
    size_t seed = 0;
    for (size_t i = 0; i < vals.size(); ++i) {
      seed ^= vals[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
}

/**
 * C = alpha A^(/T) * B^(/T) + beta C
 */
void gemm(MatrixXd* A, MatrixXd* B, MatrixXd* C, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, double alpha, double beta) {
    int m = C->rows();
    int n = C->cols();    
    int lda = A->rows();
    int ldb = B->rows();
    int ldc = C->rows();
    int k  = (tA == CblasNoTrans ? A->cols() : A->rows());
    int k2 = (tB == CblasNoTrans ? B->rows() : B->cols());        
    int m2 = (tA == CblasNoTrans ? A->rows() : A->cols());
    int n2 = (tB == CblasNoTrans ? B->cols() : B->rows());
    assert(k == k2);
    assert(m == m2); 
    assert(n == n2);
    if(m == 0 || n == 0 || k == 0)
        return;
    cblas_dgemm(CblasColMajor, tA, tB, m, n, k, alpha, A->data(), lda, B->data(), ldb, beta, C->data(), ldc);
}

MatrixXd* gemm_new(Eigen::MatrixXd* A, Eigen::MatrixXd* B, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB, double alpha) {
    int m = (tA == CblasNoTrans ? A->rows() : A->cols());
    int n = (tB == CblasNoTrans ? B->cols() : B->rows());
    MatrixXd* C = new MatrixXd(m, n);
    gemm(A, B, C, tA, tB, alpha, 0.0);
    return C;
}

void syrk(MatrixXd* A, MatrixXd* C) {
    int n = C->rows();
    int k = A->cols();
    assert(C->cols() == n);
    if (n == 0 || k == 0)
        return;
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, k, -1.0, A->data(), n, 1.0, C->data(), n);
}

int potf(MatrixXd* A) {
    int n = A->rows();
    assert(A->cols() == n);
    if (n == 0)
        return 0;
    int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n, A->data(), n);
    return info;
}

int getf(Eigen::MatrixXd* A, Eigen::VectorXi* swap) {
    int n = A->rows();
    assert(A->cols() == n);
    assert(swap->size() == n);
    if(n == 0)
        return 0;
    int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, n, n, A->data(), n, swap->data());
    for(int i = 0; i < n; i++) {
        (*swap)[i] -= 1;
    }
    return info;
}

double rcond_1_getf(Eigen::MatrixXd* A_LU, double A_1_norm) {
    int n = A_LU->rows();
    assert(A_LU->cols() == n);
    double rcond = 10.0;
    int info = LAPACKE_dgecon(LAPACK_COL_MAJOR, '1', n, A_LU->data(), n, A_1_norm, &rcond);
    assert(info == 0);
    return rcond;
}

double rcond_1_potf(Eigen::MatrixXd* A_LLT, double A_1_norm) {
    int n = A_LLT->rows();
    assert(A_LLT->cols() == n);
    double rcond = 10.0;
    int info = LAPACKE_dpocon(LAPACK_COL_MAJOR, 'L', n, A_LLT->data(), n, A_1_norm, &rcond);
    assert(info == 0);
    return rcond;
}

void trsm_right(MatrixXd* L, MatrixXd* B, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag) {
    int m = B->rows();
    int n = B->cols();
    assert(L->rows() == n);
    assert(L->cols() == n);
    if (m == 0 || n == 0)
        return;
    cblas_dtrsm(CblasColMajor, CblasRight, uplo, trans, diag, m, n, 1.0, L->data(), n, B->data(), m);
}

void trsm_left(MatrixXd* L, MatrixXd* B, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag) {
    int m = B->rows();
    int n = B->cols();
    assert(L->rows() == m);
    assert(L->cols() == m);
    if (m == 0 || n == 0)
        return;
    cblas_dtrsm(CblasColMajor, CblasLeft, uplo, trans, diag, m, n, 1.0, L->data(), m, B->data(), m);
}

void trsv(MatrixXd* LU, Segment* x, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag) {
    int n = LU->rows();
    assert(LU->cols() == n);
    assert(x->size() == n);
    if (n == 0)
        return;
    cblas_dtrsv(CblasColMajor, uplo, trans, diag, n, LU->data(), n, x->data(), 1);
}

void trmv_trans(MatrixXd* L, Segment* x) {
    int n = L->rows();
    int m = L->cols();
    assert(x->size() == n);
    assert(n == m);
    if (n == 0)
        return;
    cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit, L->rows(), L->data(), L->rows(), x->data(), 1);
}

// A <- L^T * A
void trmm_trans(MatrixXd* L, MatrixXd* A) {
    int m = A->rows();
    int n = A->cols();
    assert(L->rows() == m);
    assert(L->cols() == m);
    if (m == 0 || n == 0)
        return;
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit, m, n, 1.0, L->data(), m, A->data(), m);
}

// x2 -= A21 * x1
void gemv_notrans(MatrixXd* A21, Segment* x1, Segment* x2) {
    int m = A21->rows();
    int n = A21->cols();
    assert(x1->size() == n);
    assert(x2->size() == m);
    if (n == 0 || m == 0)
        return;
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, -1.0, A21->data(), m, x1->data(), 1, 1.0, x2->data(), 1);
}

// x2 -= A12^T x1
void gemv_trans(MatrixXd* A12, Segment* x1, Segment* x2) {
    int m = A12->rows();
    int n = A12->cols();
    assert(x1->size() == m);
    assert(x2->size() == n);
    if (n == 0 || m == 0)
        return;
    cblas_dgemv(CblasColMajor, CblasTrans, m, n, -1.0, A12->data(), m, x1->data(), 1, 1.0, x2->data(), 1);
}

// x <- Q * x
void ormqr_notrans(MatrixXd* v, VectorXd* h, Segment* x) {
    int m = v->rows();
    int n = v->cols();
    assert(h->size() == n);
    assert(x->size() == m);
    if (m == 0) 
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'N', m, 1, n, v->data(), m, h->data(), x->data(), m); 
    assert(info == 0);
}

// x <- Q^T * x
void ormqr_trans(MatrixXd* v, VectorXd* h, Segment* x) {
    int m = x->size();
    // n = 1
    int k = v->cols();
    assert(h->size() == k);
    assert(v->rows() == m);
    if (m == 0) 
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', m, 1, k, v->data(), m, h->data(), x->data(), m); 
    assert(info == 0);
}

// A <- Q^T * A
void ormqr_trans_left(MatrixXd* v, VectorXd* h, MatrixXd* A) {
    int m = A->rows();
    int n = A->cols();
    int k = v->cols();
    assert(h->size() == k);
    assert(v->rows() == m);
    if (m == 0 || n == 0)
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'T', m, n, k, v->data(), m, h->data(), A->data(), m);
    assert(info == 0);
}

// A <- Q * A
void ormqr_notrans_left(MatrixXd* v, VectorXd* h, MatrixXd* A) {
    int m = A->rows();
    int n = A->cols();
    int k = v->cols();
    assert(h->size() == k);
    assert(v->rows() == m);
    if (m == 0 || n == 0)
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'L', 'N', m, n, k, v->data(), m, h->data(), A->data(), m);
    assert(info == 0);
}

// A <- A * Q
void ormqr_notrans_right(MatrixXd* v, VectorXd* h, MatrixXd* A) {
    int m = A->rows();
    int n = A->cols();
    int k = v->cols();
    assert(h->size() == k);
    assert(v->rows() == n);
    if (m == 0 || n == 0)
        return;
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, 'R', 'N', m, n, k, v->data(), n, h->data(), A->data(), m);
    assert(info == 0);
}

// A <- (Q^/T) * A * (Q^/T)
void ormqr(MatrixXd* v, VectorXd* h, MatrixXd* A, char side, char trans) {
    int m = A->rows();
    int n = A->cols();
    int k = v->cols(); // number of reflectors
    assert(h->size() == k);
    if (m == 0 || n == 0)
        return;
    if(side == 'L') // Q * A or Q^T * A
        assert(k <= m);
    if(side == 'R') // A * Q or A * Q^T
        assert(k <= n);
    int info = LAPACKE_dormqr(LAPACK_COL_MAJOR, side, trans, m, n, k, v->data(), v->rows(), h->data(), A->data(), m);
    assert(info == 0);
}

// Create the thin Q in v
void orgqr(Eigen::MatrixXd* v, Eigen::VectorXd* h) {
    int m = v->rows();
    int k = v->cols();
    assert(h->size() == k);
    if(m == 0)
        return;
    int info = LAPACKE_dorgqr(LAPACK_COL_MAJOR, m, k, k, v->data(), m, h->data());
    assert(info == 0);
}

// RRQR
void geqp3(MatrixXd* A, VectorXi* jpvt, VectorXd* tau) {
    int m = A->rows();
    int n = A->cols();
    if (m == 0 || n == 0)
        return;
    assert(jpvt->size() == n);
    assert(tau->size() == min(m,n));
    int info = LAPACKE_dgeqp3(LAPACK_COL_MAJOR, m, n, A->data(), m, jpvt->data(), tau->data());
    assert(info == 0);
    for (int i = 0; i < jpvt->size(); i++)
        (*jpvt)[i] --;
}

// Full SVD
void gesvd(Eigen::MatrixXd* A, Eigen::MatrixXd* U, Eigen::VectorXd* S, Eigen::MatrixXd* VT) {
    int m = A->rows();
    int n = A->cols();
    int k = min(m,n);
    assert(U->rows() == m && U->cols() == m);
    assert(VT->rows() == n && VT->cols() == n);
    assert(S->size() == k);
    if(k == 0)
        return;
    VectorXd superb(k-1);
    int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', m, n, A->data(), m, S->data(), U->data(), m, VT->data(), n, superb.data());
    assert(info == 0);
}

// QR
void geqrf(MatrixXd* A, VectorXd* tau) {
    int m = A->rows();
    int n = A->cols();
    if (m == 0 || n == 0)
        return;
    assert(tau->size() == min(m,n));
    int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR, m, n, A->data(), m, tau->data());
    assert(info == 0);
}

int choose_rank(VectorXd& s, double tol) {
    if (tol == 0) {
        return s.size();
    } else if (tol >= 1.0) {
        return 0;
    } else {
        if (s.size() <= 1) {
            return s.size();
        } else {
            double sref = abs(s[0]);
            int rank = 1;
            while(rank < s.size() && abs(s[rank]) / sref >= tol) {
                rank++;
            }
            assert(rank <= s.size());
            return rank;
        }
    }
}

void block2dense(VectorXi &rowval, VectorXi &colptr, VectorXd &nnzval, int i, int j, int li, int lj, MatrixXd *dst, bool transpose) {
    if(transpose) {
        assert(dst->rows() == lj && dst->cols() == li);
    } else {
        assert(dst->rows() == li && dst->cols() == lj);
    }
    for(int col = 0; col < lj; col++) {
        // All elements in column c
        int start_c = colptr[j + col];
        int end_c = colptr[j + col + 1];
        int size = end_c - start_c;
        auto start = rowval.data() + start_c;
        auto end = rowval.data() + end_c;
        // Find i
        auto found = lower_bound(start, end, i);
        int id = distance(start, found);
        // While we are between i and i+i...
        while(id < size) {
            int row = rowval[start_c + id];
            if(row >= i + li) {
                break;
            }
            row = row - i;
            double val = nnzval[start_c + id];
            if(transpose) {
                (*dst)(col,row) = val;
            } else {
                (*dst)(row,col) = val;
            }
            id ++;
        }
    }
}

MatrixXd linspace_nd(int n, int dim) {
    MatrixXd X = MatrixXd::Zero(dim, pow(n, dim));
    if (dim == 1) {
        for(int i = 0; i < n; i++) {
            X(0,i) = double(i);
        }
    } else if (dim == 2) {
        int id = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                X(0,id) = i;
                X(1,id) = j;
                id ++;
            }
        }
    } else if (dim == 3) {
        int id = 0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                for(int k = 0; k < n; k++) {
                    X(0,id) = i;
                    X(1,id) = j;
                    X(2,id) = k;
                    id ++;
                }
            }
        }
    }
    return X;
}

// Compute A[p,p]
SpMat symm_perm(SpMat &A, VectorXi &p) {
    // Create inverse permutation
    VectorXi pinv(p.size());
    for(int i = 0; i < p.size(); i++)
        pinv[p[i]] = i;
    // Initialize A[p,p]
    int n = A.rows();
    int nnz = A.nonZeros();
    assert(n == A.cols()); 
    SpMat App(n, n);
    App.reserve(nnz);
    // Create permuted (I, J, V) values
    vector<Triplet<double>> vals(nnz);
    int l = 0;
    for (int k = 0; k < A.outerSize(); k++){
        for (SpMat::InnerIterator it(A, k); it; ++it){
            int i = it.row();
            int j = it.col();
            double v = it.value();
            vals[l] = Triplet<double>(pinv[i],pinv[j],v);
            l ++;
        }
    }
    // Create App
    App.setFromTriplets(vals.begin(), vals.end());
    return App;
}

// Random [-1,1]
VectorXd random(int size, int seed) {
    mt19937 rng;
    rng.seed(seed);
    uniform_real_distribution<double> dist(-1.0,1.0);
    VectorXd x(size);
    for(int i = 0;i < size; i++) {
        x[i] = dist(rng);
    }
    return x;
}

MatrixXd random(int rows, int cols, int seed) {
    mt19937 rng;
    rng.seed(seed);
    uniform_real_distribution<double> dist(-1.0,1.0);
    MatrixXd A(rows, cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            A(i,j) = dist(rng);
        }
    }
    return A;
}
