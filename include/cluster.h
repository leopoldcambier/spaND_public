#ifndef __CLUSTER_H__
#define __CLUSTER_H__

#include <vector>
#include <list>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <assert.h>
#include <memory>
#include <string>

#include "spaND.h"

namespace spaND {

struct Cluster {
    private:
        /* Cluster info */
        // Start of the cluster in the assembly-permuted matrix
        int start_;
        // Current size of the cluster in the assembly-permuted matrix
        int size_;
        // Level
        int level_;
        // Global order of the cluster
        int order_;
        // Wether the cluster has been eliminated
        bool eliminated_;
        // Wether to sparsify this cluster or not
        bool sparsify_;
        /* Hierarchy */
        Cluster* parent_;
        std::vector<Cluster*> children_;
        /* Vector to preserve */
        pMatrixXd phi_;
        /* Edges holding pieces of the matrix, i.e., an A12 and A21 */
        std::list<pEdge> edgesOut; // Edges to other, in the same column as self in the matrix
        std::list<Edge*> edgesIn;  // Edges from other, in the same row as self in the matrix
        /* The solution */
        pVectorXd x_; // x->size() == this->size
        /* Look for duplicates. Assumes edges are sorted */
        void assertDuplicates();
    public:
        /* Diagnostics */
        std::vector<double> Rdiag;
    public:
        // Construct a cluser
        Cluster(int start, int size, int level, int order, bool should_sparsify);
        // Wether this cluster should be sparsified
        bool should_sparsify() const;
        // Reset the cluster size and its corresponding 'x'. Any previous x is discarded
        void reset_size(int);
        // Set the size of the cluster. Keep previous x
        void set_size(int);
        // Wether this cluster has been eliminated
        bool is_eliminated() const;
        // Set cluster as eliminated and remove edges
        void set_eliminated();
        // Clear a node's Out (owned) edges
        void clear_edges();
        // Return the pivot
        Edge* pivot() const;
        // Iterate over all outgoing edges, except pivot
        ItRange<pEdgeIt> edgesOutNbr();
        // Iterate over all outgoing edges, including pivot
        ItRange<pEdgeIt> edgesOutAll();
        // Iterate over all incoming edges, except pivot
        ItRange<std::list<Edge*>::iterator> edgesInNbr();
        // Number of in/out/pivot edges
        int nnbr_in_self_out() const;
        // Number of out/pivot edges
        int nnbr_self_out() const;
        // Set x to b
        void set_vector(const Eigen::VectorXd& b);
        // Extract x into b
        void extract_vector(Eigen::VectorXd& b) const;
        // Return the head of x, corresponding to the first this->size() entries of this->get_x()
        Segment head_x();
        // Return the full x
        const pVectorXd& get_x() const;
        // Start of the cluster in permuted ordering
        int start() const;
        // Size of cluster
        int size() const;
        // Original cluster size
        int original_size() const;
        // The level at which this cluster should be eliminated
        int level() const;
        // A global cluster ID
        int order() const;
        // Return phi
        Eigen::MatrixXd* phi() const;
        // Set phi
        void set_phi(pMatrixXd);
        // Return a list over the children
        const std::vector<Cluster*>& children() const;
        // Add a children
        void add_children(Cluster*);
        // Return the parent
        Cluster* parent() const;
        // Set the parent
        void set_parent(Cluster*);
        // Add an edge (u, v) to u->outEdges and v->inEdges (if u != v) or to u->outEdges (if u == v), at the front
        void add_edge(pEdge e);
        // Sort out edges (u, v) according to v->order()
        void sort_edges();
        // Returns the depth of this in its cluster hierarchy. 0 is the root.
        int depth() const;
        // Returns the level at which this cluster should be merged
        int merge_level() const;
        // An informative name about a cluster (any string do)
        virtual std::string get_name() const;
        // virtual function -> need virtual dtor
        virtual ~Cluster() {};

};

}

#endif
