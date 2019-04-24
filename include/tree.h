#ifndef TREE_H
#define TREE_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <list>
#include <set>
#include <tuple>
#include <utility>
#include <queue>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Cholesky>
#include <Eigen/QR>
#include <Eigen/Householder> 
#include <Eigen/SVD>
#include <metis.h>
#include <numeric>
#include <assert.h>
#include <limits>
#include <memory>
#include "util.h"
#include "partition.h"

typedef Eigen::SparseMatrix<double, 0, int> SpMat;
typedef std::unique_ptr<Eigen::MatrixXd> pMatrixXd;
typedef std::unique_ptr<Eigen::VectorXd> pVectorXd;
typedef std::unique_ptr<Eigen::VectorXi> pVectorXi;

enum class ScalingKind { SVD, PLU, EVD, LLT };
enum class PartKind { MND, RB };
enum class SymmKind { SPD, SYM, GEN };

struct Edge;
struct Operation;
struct Cluster;

typedef std::unique_ptr<Cluster> pCluster;
typedef std::unique_ptr<Edge>    pEdge;

struct pEdgeIt
{
    public:
        pEdgeIt(const std::list<pEdge>::iterator& it);
        pEdgeIt(const pEdgeIt& other);
        Edge* operator*() const;
        pEdgeIt& operator=(const pEdgeIt& other);
        bool operator!=(const pEdgeIt& other);
        pEdgeIt& operator++();
    private:
        std::list<pEdge>::iterator current;
};

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

struct Cluster {
    public:
        /* Cluster info */
        int start; // start = -1 ==> not a base (bottom leaves) node
        int size; // x->size() >= this->size. size is the curent size (<= rank). x can be larger is this node got sparsified.
        int order;
        bool eliminated;
        /* Merging business */
        ClusterID id;
        ClusterID parentid;
        Cluster* parent; // not owned
        std::vector<Cluster*> children; // not owned
        int posparent;
        /* Other */
        pMatrixXd phi;
        /* Edges holding pieces of the matrix, i.e., an A12 and A21 */
        std::list<pEdge> edgesOut; // Edges to other, self <= other  (including self)
        std::list<Edge*> edgesIn;  // Edges from other, other < self (excluding self)
        /* The solution */
        pVectorXd x; // x->size() >= this->size
        /* Temporary values - not owned */
        Eigen::VectorXd* diag;
        Eigen::VectorXi* p;
        Eigen::VectorXd* s;
        Eigen::MatrixXd* U;
        Eigen::MatrixXd* VT;
        Eigen::MatrixXd* Ass;
        /* Methods */
        Cluster(int start, int size, ClusterID id, int order) : 
            start(start), size(size), order(order), eliminated(false), id(id), parentid(ClusterID()),
            parent(nullptr), posparent(-1),
            phi(nullptr), x(nullptr), diag(nullptr), p(nullptr), s(nullptr), U(nullptr), VT(nullptr), Ass(nullptr) {
                assert(start >= 0);
                set_size(size);
            }
        int get_level();
        void set_size(int size_){
            size = size_;
            this->x = std::make_unique<Eigen::VectorXd>(size);
            this->x->setZero();
        }
        bool is_eliminated() {
            return eliminated;
        }
        void set_eliminated() {
            assert(! eliminated);
            eliminated = true;
        }
        Edge* pivot();
        ItRange<pEdgeIt> edgesOutNbr();
        ItRange<pEdgeIt> edgesOutAll();
        ItRange<std::list<Edge*>::iterator> edgesInNbr();
        int nnbr_in_self_out();
        void set_vector(const Eigen::VectorXd& b);
        void extract_vector(Eigen::VectorXd& b);
        Segment head_x();
};

/* An edge holding a piece of the (trailing) matrix */
struct Edge {
    public:
        Cluster* n1;
        Cluster* n2;
        pMatrixXd A21; // Lower triangular part or Complete pivot, n2 x n1
        pMatrixXd A12; // Upper triangular part, n1 x n2
        Edge(Cluster* n1, Cluster* n2, pMatrixXd A);
        Edge(Cluster* n1, Cluster* n2, pMatrixXd A, pMatrixXd AT);
        Eigen::MatrixXd* ALow();
        Eigen::MatrixXd* AUpp();
        Eigen::MatrixXd* APiv();
        void set_APiv(pMatrixXd);
        void set_ALow(pMatrixXd);
        void set_AUpp(pMatrixXd);
        pMatrixXd get_APiv();
        pMatrixXd get_AUpp();
        pMatrixXd get_ALow();
};

/** An operation applied on the matrix **/
struct Operation {
    public:
        virtual void fwd() {};
        virtual void bwd() {};
        virtual void diag() {};
        virtual long long nnz() = 0;
        virtual std::string name() {
            return "Operation";
        };
        virtual ~Operation() {};
};

struct Gemm : public Operation {
    private:
        Segment xs;
        Segment xn;
        pMatrixXd Ans;
        pMatrixXd Asn; // Asn == nullptr <=> symmetric
    public:
        Gemm(Cluster* n1, Cluster* n2, pMatrixXd Ans) : 
            xs(n1->head_x()), xn(n2->head_x()), Ans(std::move(Ans)), Asn(nullptr) {}
        Gemm(Cluster* n1, Cluster* n2, pMatrixXd Ans, pMatrixXd Asn) : 
            xs(n1->head_x()), xn(n2->head_x()), Ans(std::move(Ans)), Asn(std::move(Asn)) {}
        void fwd() {
            if(Asn == nullptr) {
                gemv_notrans(Ans.get(), &xs, &xn);
            } else {
                gemv_notrans(Ans.get(), &xs, &xn);
            }
        }
        void bwd() {
            if(Asn == nullptr) {
                gemv_trans(Ans.get(), &xn, &xs);
            } else {
                gemv_notrans(Asn.get(), &xn, &xs);
            }
        }
        long long nnz() {
            return xs.size() * xn.size() * (Asn == nullptr ? 1 : 2);
        }
        std::string name() {
            return "Gemm";
        }
};

struct GemmDiag : public Operation {
    private:
        Segment xs;
        Segment xn;
        Eigen::VectorXd* diag;
        pMatrixXd Ans;        
    public:
        GemmDiag(Cluster* n1, Cluster* n2, pMatrixXd Ans, Eigen::VectorXd* diag) : 
            xs(n1->head_x()), xn(n2->head_x()), diag(diag), Ans(std::move(Ans)) {}
        void fwd() {   
            xn -= ((*Ans) * diag->asDiagonal() * xs);
        }
        void bwd() {
            xs -= (diag->asDiagonal() * Ans->transpose() * xn);
        }
        long long nnz() {
            return xs.size() * xn.size();
        }
        std::string name() {
            return "GemmDiag";
        }
};

struct Scaling : public Operation {
    private:
        Segment xs;
        pMatrixXd LUss;
        pVectorXi p; // p == nullptr <=> LLT / p != nullptr <=> PLU
    public:
        Scaling(Cluster* n1, pMatrixXd Lss) : 
            xs(n1->head_x()), LUss(std::move(Lss)),  p(nullptr) {}
        Scaling(Cluster* n1, pMatrixXd LUss, pVectorXi p) : 
            xs(n1->head_x()), LUss(std::move(LUss)), p(std::move(p)) {}
        void fwd() {
            if(p == nullptr) {
                trsv(LUss.get(), &xs, CblasLower, CblasNoTrans, CblasNonUnit);
            } else {
                xs = p->asPermutation().transpose() * xs;
                trsv(LUss.get(), &xs, CblasLower, CblasNoTrans, CblasUnit);                
            }
        }
        void bwd() {
            if(p == nullptr) {
                trsv(LUss.get(), &xs, CblasLower, CblasTrans, CblasNonUnit);
            } else {
                trsv(LUss.get(), &xs, CblasUpper, CblasNoTrans, CblasNonUnit);
            }
        }
        long long nnz() {
            if(p == nullptr) {
                return (xs.size() * (xs.size()+1))/2;
            } else {
                return (xs.size() * xs.size());
            }
        }
        std::string name() {
            return "Scaling";
        }
};

struct ScalingEVD : public Operation {
    private:
        Segment xs;
        pMatrixXd U;
        pVectorXd Srsqrt;
    public:
        ScalingEVD(Cluster* n, pMatrixXd U, pVectorXd Srsqrt) : 
            xs(n->head_x()), U(std::move(U)), Srsqrt(std::move(Srsqrt)) {
                assert(this->Srsqrt->size() == this->U->rows());
                assert(this->Srsqrt->size() == this->U->cols());
            }
        void fwd() {
            xs = Srsqrt->asDiagonal() * U->transpose() * xs;
        }
        void bwd() {
            xs = (*U) * Srsqrt->asDiagonal() * xs;
        }
        long long nnz() {
            return xs.size() * xs.size() + xs.size();
        }
        std::string name() {
            return "ScalingEVD";
        }
};

struct ScalingDiag : public Operation {
    private:
        Segment xs;
        pVectorXd diagV;
    public:
        ScalingDiag(Cluster* n, pVectorXd diag) :
            xs(n->head_x()), diagV(std::move(diag)) { assert(xs.size() == diagV->size()); };
        void diag() {
            xs = diagV->asDiagonal() * xs;
        }
        long long nnz() {
            return xs.size();
        }
        std::string name() {
            return "ScalingDiag";
        }
};

struct ScalingSVD : public Operation {
    private:
        Segment xs;
        pMatrixXd U;
        pMatrixXd VT;
        pVectorXd Srsqrt;
    public:
        ScalingSVD(Cluster* n1, pMatrixXd U, pMatrixXd VT, pVectorXd Srsqrt) : 
            xs(n1->head_x()), U(std::move(U)), VT(std::move(VT)), Srsqrt(std::move(Srsqrt)) {}
        void fwd() {
            xs = Srsqrt->asDiagonal() * U->transpose() * xs;
        }
        void bwd() {
            xs = VT->transpose() * Srsqrt->asDiagonal() * xs;
        }
        long long nnz() {
            return xs.size() * xs.size() * 2 + xs.size();
        }
        std::string name() {
            return "ScalingSVD";
        }
};

struct SelfElimEVD : public Operation {
    private:
        Segment xc;
        Segment xf;        
        pMatrixXd U;
        pVectorXd Srsqrt;
        pMatrixXd Cfc;
        pVectorXd diagV;
    public:
        SelfElimEVD(Cluster* self, pMatrixXd U, pVectorXd Srsqrt, pMatrixXd Cfc, pVectorXd diag) :
            xc(self->x->segment(0, Cfc->cols())),
            xf(self->x->segment(Cfc->cols(), Cfc->rows())),
            U(std::move(U)), Srsqrt(std::move(Srsqrt)), Cfc(std::move(Cfc)), diagV(std::move(diag))
        {
            assert(this->Cfc->rows()  + this->Cfc->cols() == self->size);
            assert(this->Cfc->rows() == this->U->rows());
            assert(this->U->rows()   == this->U->cols());
            assert(this->Cfc->rows() == this->diagV->rows());
        }
        void fwd() {
            xf = Srsqrt->asDiagonal() * U->transpose() * xf;
            xc -= Cfc->transpose() * xf;            
        }
        void bwd() {
            xf -= (*Cfc) * xc;
            xf = (*U) * Srsqrt->asDiagonal() * xf;
        }
        void diag() {
            xf = diagV->asDiagonal() * xf;
        }
        long long nnz() {
            return 2 * diagV->size() + U->rows() * U->cols() + Cfc->rows() * Cfc->cols();
        }
        std::string name() {
            return "SelfElimEVD";
        }
};

struct SelfElim : public Operation {
    private:
        Segment xc;
        Segment xf;
        pMatrixXd Cff;
        pMatrixXd Ccf;
        pMatrixXd Cfc; // nullptr = symmetric
        pVectorXi lup; // nullptr = symmetric
    public:
        SelfElim(Cluster* self, pMatrixXd Cff, pMatrixXd Ccf) :
                 xc(self->x->segment(0, Ccf->rows())),
                 xf(self->x->segment(Ccf->rows(), Ccf->cols())),
                 Cff(std::move(Cff)), Ccf(std::move(Ccf)), Cfc(nullptr), lup(nullptr) {
            assert(this->Ccf->rows() + this->Ccf->cols() == self->size);   
        }
        SelfElim(Cluster* self, pMatrixXd Cff, pMatrixXd Ccf,
                 pMatrixXd Cfc, pVectorXi lup) :
                 xc(self->x->segment(0, Ccf->rows())),
                 xf(self->x->segment(Ccf->rows(), Ccf->cols())),
                 Cff(std::move(Cff)), Ccf(std::move(Ccf)), Cfc(std::move(Cfc)), lup(std::move(lup)) {
            assert(this->Ccf->rows() + this->Ccf->cols() == self->size);
        }
        void fwd() {
            if(Cfc == nullptr) {
                trsv(Cff.get(), &xf, CblasLower, CblasNoTrans, CblasNonUnit);
                gemv_notrans(Ccf.get(), &xf, &xc);
            } else {
                xf = lup->asPermutation().transpose() * xf;
                trsv(Cff.get(), &xf, CblasLower, CblasNoTrans, CblasUnit);
                gemv_notrans(Ccf.get(), &xf, &xc); 
            }
        }
        void bwd() {
            if(Cfc == nullptr) {
                gemv_trans(Ccf.get(), &xc, &xf); 
                trsv(Cff.get(), &xf, CblasLower, CblasTrans, CblasNonUnit);
            } else {
                gemv_notrans(Cfc.get(), &xc, &xf);
                trsv(Cff.get(), &xf, CblasUpper, CblasNoTrans, CblasNonUnit);
            }
        }
        long long nnz() {
            if(Cfc == nullptr) {
                return (xf.size() * (xf.size()+1))/2 + xf.size() * xc.size();
            } else {
                return (xf.size() * xf.size()) + 2 * xf.size() * xc.size();
            }
        } 
        std::string name() {
            return "SelfElim";
        }     

};

struct Permutation : public Operation {
    private:
        Segment xs;
        pVectorXi perm;
    public:
        Permutation(Cluster* n1, pVectorXi perm) : 
            xs(n1->head_x()), perm(std::move(perm)) {}
        void fwd() {
            xs = perm->asPermutation().transpose() * xs;
        }
        void bwd() {
            xs = perm->asPermutation() * xs;
        }
        long long nnz() {
            return xs.size();
        }
        std::string name() {
            return "Permutation";
        }
};

struct Interpolation : public Operation {
    private:
        Segment xc;
        Segment xf;
        pMatrixXd Tcf;
    public:
        Interpolation(Cluster* n1, pMatrixXd Tcf) : 
            xc(n1->x->segment(0, Tcf->rows())),
            xf(n1->x->segment(Tcf->rows(), Tcf->cols())),
            Tcf(std::move(Tcf)) {
            assert(this->Tcf->rows() + this->Tcf->cols() == n1->size);
        }
        void fwd() {
            gemv_trans(Tcf.get(), &xc, &xf);
        }
        void bwd() {            
            gemv_notrans(Tcf.get(), &xf, &xc);
        }
        long long nnz() {
            return xc.size() * xf.size();
        }
        std::string name() {
            return "Interpolation";
        }
};

struct Orthogonal : public Operation {
    private:
        Segment xs;
        pMatrixXd v;
        pVectorXd h;
    public:
        Orthogonal(Cluster* n1, pMatrixXd v, pVectorXd h) : 
            xs(n1->head_x()), v(std::move(v)), h(std::move(h)) {}
        void fwd() {
            ormqr_trans(v.get(), h.get(), &xs);            
        }
        void bwd() {
            ormqr_notrans(v.get(), h.get(), &xs);
        }
        long long nnz() {
            return xs.size() * xs.size();
        }
        std::string name() {
            return "Orthogonal";
        }
};

struct Merge : public Operation {
    Cluster* parent;
    // Children are in parent->children
    public:
        Merge(Cluster* parent) : parent(parent) {}
        void fwd() {
            int k = 0;
            for(auto c: parent->children) {
                for(int i = 0; i < c->size; i++) {
                    (*parent->x)[k] = (*c->x)[i];
                    k++;
                }
            }
            assert(k == parent->x->size());
        }
        void bwd() {
            int k = 0;
            for(auto c: parent->children) {
                for(int i = 0; i < c->size; i++) {
                    (*c->x)[i] = (*parent->x)[k];
                    k++;
                }
            }
            assert(k == parent->x->size());
        }
        long long nnz() {
            return 0;
        }
        std::string name() {
            return "Merge";
        }
};

class Tree
{
    /**
     * Basic data structure
     * Given
     * A = [Ass  Asn1  Asn2]
     *     [An1s .     .   ]
     *     [An2s .     .   ]
     * Ass, An1s, An2s are stored in 'low' edges
     *      Asn1, Asn2 are stored in 'upp' edges
     * The first upp edge is always nullptr
     * The upp edges are all nullptr if symmetry
     * The low edges are all non nullptr
     * All edges are from lower -> higher order
     */

    private:

        // Parameters
        bool verb;              // Verbose (true) or not
        bool geo;               // Wether to use geometry (true) or not
        PartKind part_kind;     // The kind of partitioning (modified ND or recursive bissection)
        bool use_vertex_sep;    // ModifiedND: Wether to use a vertex separator in algebraic partitioning (true) or bipartition (false)
        bool preserve;          // Wether to preserve phi (true) or not
        int nphis;              // Number of vectors to preserve
        int N;                  // (Square) matrix size
        int ilvl;               // Level [0...lvl) have been eliminated ; -1 is nothing eliminated
        int nlevels;            // Maximum tree depth        
        double tol;             // Compression tolerance
        int skip;               // #levels to skip for sparsification
        bool scale;             // Wether to scale the pivot (true) or not
        ScalingKind scale_kind; // The kind of scaling
        SymmKind symm_kind;     // Wether the matrix is SPD (SPD), symmetric indefinite (SYM) or general unsymmetric (GEN)
        bool ortho;             // Wether to use orthogonal transformation (true) or not
        bool use_want_sparsify; // Wether to use want_sparsify (true) or not
        bool monitor_condition_pivots; // Compute condition number of all pivots (expensive)
        // Helper parameters
        bool adaptive;
        // External data (NOT owned)
        Eigen::MatrixXd* Xcoo;  // The dim x N coordinates matrix
        Eigen::MatrixXd* phi;   // The N x k phi matrix
        
        /** Stats and info **/
        long long nnz() ;
        int ndofs_left();
        int nclusters_left();
        void stats();
        bool symmetry() const;

        /** Helpers */
        void init(int lvl);        
        // Eliminating
        int  eliminate_cluster(Cluster*);
        int  potf_cluster(Cluster*);
        void trsm_edgeIn(Edge*);
        void trsm_edgeOut(Edge*);
        void gemm_edges(Edge*,Edge*);        
        void update_eliminated_edges_and_delete(Cluster*);
        // Scaling
        int  scale_cluster(Cluster*);
        // Sparsification        
        bool want_sparsify(Cluster*);
        int  sparsify_cluster(Cluster*);
        void sparsify_adaptive_only(Cluster*);
        void sparsify_preserve_only(Cluster*);
        void sparsify_preserve_adaptive(Cluster*);
        pMatrixXd assemble_Asn(Cluster*);
        pMatrixXd assemble_Asphi(Cluster*);
        int  sparsify_interp(Cluster*);
        void drop_all(Cluster*);
        void scatter_Q(Cluster*, Eigen::MatrixXd*);
        void scatter_Asn(Cluster*, Eigen::MatrixXd*);
        // Merge
        void update_size(Cluster*);
        void update_edges(Cluster*);

        // The permutation computed by assembly
        Eigen::VectorXi perm;

        // Store the operations
        std::vector<std::unique_ptr<Operation>> ops;

        // Stores the clusters at each level of the cluster hierarchy
        int current_bottom;
        std::vector<std::list<pCluster>> bottoms; // bottoms.size() = nlevels
        const std::list<pCluster>& bottom_current() const;
        const std::list<pCluster>& bottom_original() const;

    public:    

        // Set all sorts of options
        void set_verb(bool);
        void set_Xcoo(Eigen::MatrixXd*);
        void set_use_geo(bool);
        void set_phi(Eigen::MatrixXd*);
        void set_preserve(bool);
        void set_use_vertex_sep(bool);
        void set_tol(double);
        void set_skip(int);
        void set_scale(bool);
        void set_ortho(bool);
        void set_scaling_kind(ScalingKind);
        void set_symm_kind(SymmKind);
        void set_part_kind(PartKind);
        void set_use_sparsify(bool);
        void set_monitor_condition_pivots(bool);

        // Basic info
        int get_N() const;
        void print_summary() const;
        Eigen::VectorXi get_assembly_perm() const;
        SpMat get_trailing_mat() const;
        Eigen::MatrixXd get_current_x() const;
        std::vector<std::vector<ClusterID>> get_clusters_levels() const;
        void print_ordering_clustering() const;
        bool is_factorized() const;

        // Publicly visible profiling & other log info
        std::vector<Profile> tprof;
        std::vector<Log> log;

        // Store the ordering & partitioning after partition
        std::vector<ClusterID> part;

        /** Constructor 
         * lvl is the tree depth
         */
        Tree(int lvl);

        /** Partitioning and Ordering 
         * Assumes the matrix A has a symmetric pattern
         */
        void partition(SpMat&);
        void partition_rb(SpMat&);

        /** Initial Matrix Assembly 
         * A can be anything, though to make sense, its pattern should match the one in partition
         */
        void assemble(SpMat&);

        /** Factorization
         */
        int factorize();

        /** Solve
         * X should have the same size as the matrix
         */
        void solve(Eigen::VectorXd&) const;

        /** Analysis
         * Print some timings and other rank information
         **/
        void print_log() const;
};

#endif
