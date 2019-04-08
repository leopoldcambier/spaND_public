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
#include "util.h"

typedef Eigen::SparseMatrix<double, 0, int> SpMat;

// Describes the ordering
struct SepID {
    public:
        int lvl;
        int sep;
        SepID(int l, int s) : lvl(l), sep(s) {};
        SepID() : lvl(-1), sep(0) {} ;
        // Some lexicographics order
        // NOT the matrix ordering
        bool operator==(const SepID& other) const {
            return (this->lvl == other.lvl && this->sep == other.sep);
        }
        bool operator<(const SepID& other) const {
            return (this->lvl < other.lvl) || (this->lvl == other.lvl && this->sep < other.sep);
        }
};

// Describes the merging of the separators
struct ClusterID {
    public:
        SepID self;
        SepID l;
        SepID r;
        ClusterID(SepID self) {
            this->self = self;
            this->l    = SepID();
            this->r    = SepID();
        }
        ClusterID() {
            this->self = SepID();
            this->l    = SepID();
            this->r    = SepID();
        }
        ClusterID(SepID self, SepID left, SepID right) {
            this->self = self;
            this->l    = left;
            this->r    = right;
        }
        // Some lexicographics order
        // NOT the matrix ordering
        bool operator==(const ClusterID& other) const {
            return      (this->self == other.self)
                     && (this->l    == other.l)
                     && (this->r    == other.r);
        }
        bool operator<(const ClusterID& other) const {
            return     (this->self <  other.self)
                    || (this->self == other.self && this->l <  other.l) 
                    || (this->self == other.self && this->l == other.l && this->r < other.r);
        }
        bool operator>(const ClusterID& other) const {
            return ! ( (*this) == other || (*this < other) );
        }
};

std::ostream& operator<<(std::ostream& os, const SepID& s);
std::ostream& operator<<(std::ostream& os, const ClusterID& c);
SepID merge(SepID& s);
ClusterID merge_if(ClusterID& c, int lvl);

struct Edge;
struct Operation;
struct OutNbr;
struct Cluster;

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
        Cluster* parent;
        std::vector<Cluster*> children;
        int posparent;
        /* Other */
        Eigen::MatrixXd* phi;
        /* Edges holding pieces of the matrix, i.e., an A12 and A21 */
        std::list<Edge*> edgesOut; // Edges to other, self <= other  (including self)
        std::list<Edge*> edgesIn;  // Edges from other, other < self (excluding self)
        /* The solution */
        Eigen::VectorXd* x; // x->size() >= this->size
        Eigen::VectorXi* p;
        /* Methods */
        Cluster(int start, int size, ClusterID id, int order) : 
            start(start), size(size), order(order), eliminated(false), id(id), parentid(ClusterID()),
            parent(nullptr), posparent(-1),
            phi(nullptr), x(nullptr), p(nullptr) {
                if(start >= 0) {
                    set_size(size);
                }
            }
        int get_level();
        void set_size(int size_){
            size = size_;
            delete this->x;
            this->x = new Eigen::VectorXd(size);
            this->x->setZero();
        }
        bool is_eliminated() {
            return eliminated;
        }
        void set_eliminated() {
            assert(! eliminated);
            eliminated = true;
        }
        // int get_sep();
        Edge* pivot();
        OutNbr edgesOutNbr();
        int nnbr_in_self_out();
        void set_vector(const Eigen::VectorXd& b);
        void extract_vector(Eigen::VectorXd& b);
        Segment head();
};

/* An edge holding a piece of the (trailing) matrix */
struct Edge {
    public:
        Cluster* n1;
        Cluster* n2;
        Eigen::MatrixXd* A21; // Lower triangular part or Complete pivot, n2 x n1
        Eigen::MatrixXd* A12; // Upper triangular part, n1 x n2
        Edge(Cluster* n1, Cluster* n2, Eigen::MatrixXd* A);
        Edge(Cluster* n1, Cluster* n2, Eigen::MatrixXd* A, Eigen::MatrixXd* AT);
        Eigen::MatrixXd* ALow();
        Eigen::MatrixXd* AUpp();
        Eigen::MatrixXd* APiv();
        void set_APiv(Eigen::MatrixXd* A);
        void set_ALow(Eigen::MatrixXd* A);
        void set_AUpp(Eigen::MatrixXd* A);
};

/** An operation applied on the matrix **/
struct Operation {
    public:
        virtual void fwd() = 0;
        virtual void bwd() = 0;
        virtual long long nnz() = 0;
        virtual ~Operation() {};
};

struct Gemm : public Operation {
    private:
        Segment xs;
        Segment xn;
        Eigen::MatrixXd* Ans;
        Eigen::MatrixXd* Asn; // Asn == nullptr <=> symmetric
    public:
        Gemm(Cluster* n1, Cluster* n2, Eigen::MatrixXd* Ans) : 
            xs(n1->head()), xn(n2->head()), Ans(Ans), Asn(nullptr) {}
        Gemm(Cluster* n1, Cluster* n2, Eigen::MatrixXd* Ans, Eigen::MatrixXd* Asn) : 
            xs(n1->head()), xn(n2->head()), Ans(Ans), Asn(Asn) {}
        void fwd() {
            if(Asn == nullptr) {
                gemv_notrans(Ans, &xs, &xn);
            } else {
                gemv_notrans(Ans, &xs, &xn);
            }
        }
        void bwd() {
            if(Asn == nullptr) {
                gemv_trans(Ans, &xn, &xs);
            } else {
                gemv_notrans(Asn, &xn, &xs);
            }
        }
        long long nnz() {
            return xs.size() * xn.size() * (Asn == nullptr ? 1 : 2);
        }
        ~Gemm() {
            delete Ans;
            delete Asn;
        }
};

struct Scaling : public Operation {
    private:
        Segment xs;
        Eigen::MatrixXd* LUss;
        Eigen::VectorXi* p; // p == nullptr <=> LLT / p != nullptr <=> PLU
    public:
        Scaling(Cluster* n1, Eigen::MatrixXd* Lss) : 
            xs(n1->head()), LUss(Lss),  p(nullptr) {}
        Scaling(Cluster* n1, Eigen::MatrixXd* LUss, Eigen::VectorXi* p) : 
            xs(n1->head()), LUss(LUss), p(p) {}
        void fwd() {
            if(p == nullptr) {
                trsv(LUss, &xs, CblasLower, CblasNoTrans, CblasNonUnit);
                // std::cout << "Scaling\n" << xs << "\n" << *LUss << "\n*********" << std::endl;
            } else {
                xs = p->asPermutation().transpose() * xs;
                trsv(LUss, &xs, CblasLower, CblasNoTrans, CblasUnit);                
            }
        }
        void bwd() {
            if(p == nullptr) {
                trsv(LUss, &xs, CblasLower, CblasTrans, CblasNonUnit);
            } else {
                trsv(LUss, &xs, CblasUpper, CblasNoTrans, CblasNonUnit);
            }
        }
        long long nnz() {
            if(p == nullptr) {
                return (xs.size() * (xs.size()+1))/2;
            } else {
                return (xs.size() * xs.size());
            }
        }
        ~Scaling() {
            delete LUss;
            delete p;
        }
};

struct ScalingSVD : public Operation {
    private:
        Segment xs;
        Eigen::MatrixXd* U;
        Eigen::MatrixXd* VT;
        Eigen::VectorXd* Srsqrt;
    public:
        ScalingSVD(Cluster* n1, Eigen::MatrixXd* U, Eigen::MatrixXd* VT, Eigen::VectorXd* Srsqrt) : 
            xs(n1->head()), U(U), VT(VT), Srsqrt(Srsqrt) {}
        void fwd() {
            xs = Srsqrt->asDiagonal() * U->transpose() * xs;
        }
        void bwd() {
            xs = VT->transpose() * Srsqrt->asDiagonal() * xs;
        }
        long long nnz() {
            return xs.size() * xs.size() * 2 + xs.size();
        }
        ~ScalingSVD() {
            delete U;
            delete VT;
            delete Srsqrt;
        }
};

struct SelfElim : public Operation {
    private:
        Segment xc;
        Segment xf;
        Eigen::MatrixXd* Cff;
        Eigen::MatrixXd* Ccf;
        Eigen::MatrixXd* Cfc; // nullptr = symmetric
        Eigen::VectorXi* lup; // nullptr = symmetric
    public:
        SelfElim(Cluster* n1, Eigen::MatrixXd* Cff, Eigen::MatrixXd* Ccf) :
                 xc(n1->x->segment(0, Ccf->rows())),
                 xf(n1->x->segment(Ccf->rows(), Ccf->cols())),
                 Cff(Cff), Ccf(Ccf), Cfc(nullptr), lup(nullptr) {
            assert(Ccf->rows() + Ccf->cols() == n1->size);   
        }
        SelfElim(Cluster* n1, Eigen::MatrixXd* Cff, Eigen::MatrixXd* Ccf,
                 Eigen::MatrixXd* Cfc, Eigen::VectorXi* lup) :
                 xc(n1->x->segment(0, Ccf->rows())),
                 xf(n1->x->segment(Ccf->rows(), Ccf->cols())),
                 Cff(Cff), Ccf(Ccf), Cfc(Cfc), lup(lup) {
            assert(Ccf->rows() + Ccf->cols() == n1->size);
        }
        void fwd() {
            if(Cfc == nullptr) {
                trsv(Cff, &xf, CblasLower, CblasNoTrans, CblasNonUnit);
                gemv_notrans(Ccf, &xf, &xc);
                // std::cout << "SelfElim\n" << xc << "\n" << xf << "\n" << *Cff << "\n" << *Ccf << "\n*********" << std::endl;
            } else {
                xf = lup->asPermutation().transpose() * xf;
                trsv(Cff, &xf, CblasLower, CblasNoTrans, CblasUnit);
                gemv_notrans(Ccf, &xf, &xc); 
            }
        }
        void bwd() {
            if(Cfc == nullptr) {
                gemv_trans(Ccf, &xc, &xf); 
                trsv(Cff, &xf, CblasLower, CblasTrans, CblasNonUnit);
            } else {
                gemv_notrans(Cfc, &xc, &xf);
                trsv(Cff, &xf, CblasUpper, CblasNoTrans, CblasNonUnit);
            }
        }
        long long nnz() {
            if(Cfc == nullptr) {
                return (xf.size() * (xf.size()+1))/2 + xf.size() * xc.size();
            } else {
                return (xf.size() * xf.size()) + 2 * xf.size() * xc.size();
            }
        }
        ~SelfElim() {
            delete Cff;
            delete Ccf;
            delete Cfc;
            delete lup;
        }
        
};

struct Permutation : public Operation {
    private:
        Segment xs;
        Eigen::VectorXi* perm;
    public:
        Permutation(Cluster* n1, Eigen::VectorXi* perm) : 
            xs(n1->head()), perm(perm) {}
        void fwd() {
            xs = perm->asPermutation().transpose() * xs;
        }
        void bwd() {
            xs = perm->asPermutation() * xs;
        }
        long long nnz() {
            return xs.size();
        }
        ~Permutation() {
            delete perm;
        }
};

struct Interpolation : public Operation {
    private:
        Segment xc;
        Segment xf;
        Eigen::MatrixXd* Tcf;
    public:
        Interpolation(Cluster* n1, Eigen::MatrixXd* Tcf) : 
            xc(n1->x->segment(0, Tcf->rows())),
            xf(n1->x->segment(Tcf->rows(), Tcf->cols())),
            Tcf(Tcf) {
            assert(Tcf->rows() + Tcf->cols() == n1->size);
        }
        void fwd() {
            gemv_trans(Tcf, &xc, &xf);
        }
        void bwd() {            
            gemv_notrans(Tcf, &xf, &xc);
        }
        long long nnz() {
            return xc.size() * xf.size();
        }
        ~Interpolation() {
            delete Tcf;
        }
};

struct Orthogonal : public Operation {
    private:
        Segment xs;
        Eigen::MatrixXd* v;
        Eigen::VectorXd* h;
    public:
        Orthogonal(Cluster* n1, Eigen::MatrixXd* v, Eigen::VectorXd* h) : 
            xs(n1->head()), v(v), h(h) {}
        void fwd() {
            ormqr_trans(v, h, &xs);
        }
        void bwd() {
            ormqr_notrans(v, h, &xs);
        }
        long long nnz() {
            return xs.size() * xs.size();
        }
        ~Orthogonal() {
            delete v;
            delete h;
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
        ~Merge() {}
};

struct OutNbr
{
    public:
        OutNbr(Cluster* n_) : n(n_) {
            assert(n->edgesOut.size() > 0);
        }
        std::list<Edge*>::iterator begin() const {
            auto iter = n->edgesOut.begin();
            iter++;
            return iter;
        }
        std::list<Edge*>::iterator end() const {   
            return n->edgesOut.end();
        }
    private:
        Cluster* n;
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

        double mergealloc;
        double mergecopy;

        double geqp3;
        double geqrf;
        double potf;
        double trsm;
        double gemm;

        double buildq;
        double scattq;
        double perma;
        double scatta;
        double prese;
        double assmb;
        double phi;

        Profile() :
            elim(0), scale(0), spars(0), merge(0), 
            mergealloc(0), mergecopy(0),
            geqp3(0), geqrf(0), potf(0), trsm(0), gemm(0),
            buildq(0), scattq(0), perma(0), scatta(0), prese(0), assmb(0), phi(0)
            {}
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
        Stats<double> cond_diag;
        Stats<double> norm_diag;
    
        Log() :
            dofs_nd(0),
            dofs_left_nd(0), dofs_left_elim(0), dofs_left_spars(0),
            fact_nnz(0),
            rank_before(Stats<int>()), rank_after(Stats<int>()),
            cond_diag(Stats<double>()), norm_diag(Stats<double>())
            {}
};

enum class ScalingKind { SVD, PLU };

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
        bool symmetry;          // Symmetry (true) or not
        bool geo;               // Wether to use geometry (true) or not
        bool use_vertex_sep;    // Wether to use a vertex separator in algebraic partitioning (true) or bipartition (false)
        bool preserve;          // Wether to preserve phi (true) or not
        int nphis;              // Number of vectors to preserve
        int N;                  // (Square) matrix size
        int ilvl;               // Level [0...lvl) have been eliminated ; -1 is nothing eliminated
        int nlevels;            // Maximum tree depth        
        double tol;             // Compression tolerance
        int skip;               // #levels to skip for sparsification
        bool scale;             // Wether to scale the pivot (true) or not
        ScalingKind scale_kind; // The kind of scaling
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

        /** Helpers */
        void init(int lvl);        
        // Eliminating
        int  eliminate_cluster(Cluster*);
        int  potf_cluster(Cluster*);
        void trsm_edge(Edge*);
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
        Eigen::MatrixXd* assemble_Asn(Cluster*);
        Eigen::MatrixXd* assemble_Asphi(Cluster*);
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
        std::vector<Operation*> ops;  

        // Stores the clusters at each level of the cluster hierarchy
        int current_bottom;
        std::vector<std::list<Cluster*>> bottoms; // bottoms.size() = nlevels
        const std::list<Cluster*>& bottom_current() const;
        const std::list<Cluster*>& bottom_original() const;

    public:    

        // Set all sorts of options
        void set_verb(bool);
        void set_symmetry(bool);
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
        void set_use_sparsify(bool);
        void set_monitor_condition_pivots(bool);

        // Basic info
        int get_N() const;
        void print_summary() const;
        Eigen::VectorXi get_assembly_perm() const;
        SpMat get_assembly_mat() const;
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

        /** Destructor */
        ~Tree();
};

#endif
