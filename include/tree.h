#ifndef __TREE_H__
#define __TREE_H__

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
#include <Eigen/LU>
#include <numeric>
#include <assert.h>
#include <limits>
#include <memory>

#include "spaND.h"

namespace spaND {

class Tree
{ 
    protected:

        // Parameters
        bool verb;              // Verbose (true) or not
        bool geo;               // Wether to use geometry (true) or not
        PartKind part_kind;     // The kind of partitioning (modified ND or recursive bissection)
        bool use_vertex_sep;    // ModifiedND: Wether to use a vertex separator in algebraic partitioning (true) or bipartition (false)
        bool preserve;          // Wether to preserve phi (true) or not
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
        bool monitor_unsymmetry;       // Monitor symmetry of the matrix (expensive)
        bool monitor_Rdiag;            // Print diagonal of R to file. Each line is <lvl sigma_1 ... sigma_k>
        bool monitor_flops;

        // Helper parameters
        int max_order;

        // External data (NOT owned)
        Eigen::MatrixXd* Xcoo;  // The dim x N coordinates matrix
        Eigen::MatrixXd* phi;   // The N x k phi matrix
        
        /** Stats and info **/        
        int ndofs_left() const;
        int nclusters_left() const;
        void stats() const;
        bool symmetry() const;
        void assert_symmetry();
        int nphis() const;

        /** Helpers */
        void init(int lvl);

        // Eliminating & scaling
        void eliminate_cluster(Cluster*);
        void scale_cluster(Cluster*);

        // LLT
        void potf_cluster(Cluster*);
        void panel_potf(Cluster*);
        void trsm_potf_edgeIn(Edge*);
        void trsm_potf_edgeOut(Edge*);
        
        // PLUQ
        void getf_cluster(Cluster*, pMatrixXd*, pMatrixXd*, pVectorXi*, pVectorXi*);
        void panel_getf(Cluster*, Eigen::MatrixXd*, Eigen::MatrixXd*, Eigen::VectorXi*, Eigen::VectorXi*);
        void trsm_getf_edgeIn(Edge*, Eigen::MatrixXd*, Eigen::VectorXi*);
        void trsm_getf_edgeOut(Edge*, Eigen::MatrixXd*, Eigen::VectorXi*);

        // LDLT
        void ldlt_cluster(Cluster*, pMatrixXd*, pVectorXd*, pVectorXi*);
        void panel_ldlt(Cluster*, Eigen::MatrixXd*, Eigen::VectorXi*);
        void trsm_ldlt_edgeIn(Edge*, Eigen::MatrixXd*, Eigen::VectorXi*);
        void trsm_ldlt_edgeOut(Edge*, Eigen::MatrixXd*, Eigen::VectorXi*);
        
        void schur_symmetric(Cluster*);
        void schur_symmetric(Cluster*, Eigen::VectorXd*);
        void record_schur_symmetric(Cluster*, Eigen::VectorXd*);
        void gemm_edges(Edge*, Edge*, Eigen::VectorXd*, bool, bool);

        // Sparsification        
        bool want_sparsify(Cluster*);
        void sparsify_cluster(Cluster*);
        void sparsify_cluster_farfield(Cluster*);
        pMatrixXd assemble_Asn(Cluster*, std::function<bool(Edge*)>);                
        void sparsify_adaptive_only(Cluster* self, std::function<bool(Edge*)> pred);
        pMatrixXd assemble_Asphi(Cluster*);
        void sparsify_preserve_only(Cluster*);
        void sparsify_preserve_adaptive(Cluster*);

        // Merge
        void merge_all();
        pOperation reset_size(Cluster*, std::map<Cluster*,int>*);
        void update_edges(Cluster*, std::map<Cluster*,int>*);
        Cluster* shrink_split_scatter_phi(Cluster*, int, std::function<bool(Edge*)>, Eigen::MatrixXd*, Eigen::VectorXd*, Eigen::MatrixXd*, bool, bool);

        // The permutation computed by assembly
        Eigen::VectorXi perm;

        // Store the operations
        std::list<std::unique_ptr<Operation>> ops;

        // Stores the clusters at each level of the cluster hierarchy
        int current_bottom;
        std::vector<std::list<pCluster>> bottoms; // bottoms.size() = nlevels
        std::vector<pCluster> others;
        const std::list<pCluster>& bottom_current() const;
        const std::list<pCluster>& bottom_original() const;

        // Generate new order ID's
        int get_new_order();

    public:

        // Set all sorts of options
        void set_verb(bool);
        void set_Xcoo(Eigen::MatrixXd*);
        void set_use_geo(bool);
        void set_phi(Eigen::MatrixXd*);
        void set_preserve(bool);
        void set_tol(double);
        void set_skip(int);
        void set_scaling_kind(ScalingKind);
        void set_symm_kind(SymmKind);
        void set_part_kind(PartKind);
        void set_use_sparsify(bool);
        void set_monitor_condition_pivots(bool);
        void set_monitor_unsymmetry(bool);
        void set_monitor_Rdiag(bool);
        void set_monitor_flops(bool);

        // Basic info, export of data and monitoring
        int get_N() const;
        Eigen::VectorXi get_assembly_perm() const;
        SpMat get_trailing_mat() const;
        Eigen::MatrixXd get_current_x() const;
        void print_clusters_hierarchy() const;
        void print_connectivity() const;
        std::list<const Cluster*> get_clusters() const;
        std::vector<int> get_dof2ID() const;
        int get_nlevels() const;
        int get_stop() const;
        long long nnz() const ;

        // Publicly visible profiling & other log info
        std::vector<Profile> tprof;
        std::vector<Log> log;
        std::vector<ProfileFlops> tprof_flops;        

        /** Constructor 
         * lvl is the tree depth
         */
        Tree(int lvl);

        /** Partitioning and Ordering 
         * Assumes the matrix A has a symmetric pattern
         * Returns the partitioning (self, left, right) used, in natural ordering
         */
        std::vector<ClusterID> partition(SpMat&);
        void partition_lorasp(SpMat&);

        /** Initial Matrix Assembly 
         * A can be anything, though to make sense, its pattern should match the one in partition
         */
        void assemble(SpMat&);

        /** Factorization
         */
        void factorize();
        void factorize_lorasp();

        /** Solve
         * X should have the same size as the matrix
         */
        void solve(Eigen::VectorXd&) const;

        /** Analysis
         * Print some timings and other rank information
         **/
        void print_log() const;
};

/** 
 * Functions operating on Trees
 */

/** Save data to files **/

/** 
 * Each ID is an integer, starting from 0, uniquely identitying all clusters.
 * The cleaf clusters have the first ID's, followed by their parents, then their parents, etc   
 */

/**
 * This file gives the coordinates and unique cluster ID for every row/col in the matrix (in its natural ordering)
 * The first 2 lines are
 * >>   N ndims L
 * >>   X;id
 * where N is the matrix size, ndims is the space dimension (0 if no geometry) and L the number of ND levels
 * All subsequent lines correspond to a row/col in the matrix (in its natural ordering).
 * >>   xi_0 ... xi_ndims-1 ; order
 *      ^-----------------^   ^---^
 *      Coordinates (if any)   ID
 */
void write_clustering(const Tree& t, const Eigen::MatrixXd& X, std::string fn);

/**
 * This file gives the cluster hierarchy. This is done by listing all children -> parent edges using their ID
 * The file line is
 * >>    child;parent
 * followed by one line for each unique Left clusterID (in no particular order)
 * >>   child ; parent
 *      ^---^   ^----^
 *       ID      ID
 *  So each line has 2 integers
 */
void write_merging(const Tree& t, std::string fn);

/**
 * This file gives clusters medatada. The first line is
 * >>    id;lvl;mergeLvl;name
 * Each line is then
 * >>    ID ; lvl ; merge level ; string
 * where lvl is the cluster ND level, merge level is the level at which this cluster "exists" and is merged, and string any string.
 */
void write_clusters(const Tree& t, std::string fn);

/**
 * This prints the cluster ranks and sizes to a file. Lines are
 * >> ID  size rank
 *        ^--^ ^--^
 *         |    Final cluster size (size())
 *         *-- Original cluster size (original_size())
 * where there is 1 line per cluster, for all levels (_not_ N*nlevels _nor_ N)
 */
void write_stats(const Tree& t, std::string fn);

void write_log_flops(const Tree& t, std::string fn);

}

#endif
