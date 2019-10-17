#ifndef __EDGE_H__
#define __EDGE_H__

#include <vector>
#include <iterator>
#include <list>
#include <numeric>
#include <assert.h>
#include <limits>
#include <memory>

#include "spaND.h"

namespace spaND {

/** Iterate over pEdge and return Edge* **/
struct pEdgeIt : public std::iterator<std::input_iterator_tag, Edge*>
{
    public:
        pEdgeIt(const std::list<pEdge>::iterator& it);
        pEdgeIt(const pEdgeIt& other);
        Edge* operator*() const;
        pEdgeIt& operator=(const pEdgeIt& other);
        bool operator!=(const pEdgeIt& other) const;
        bool operator==(const pEdgeIt& other) const;
        pEdgeIt& operator++();
    private:
        std::list<pEdge>::iterator current;
};

/* An edge holding a piece of the (trailing) matrix */
struct Edge {
    public:
        Cluster* n1;
        Cluster* n2;
        bool original;
        pMatrixXd A21;
        Edge(Cluster* n1, Cluster* n2, pMatrixXd A21, bool original);
        Eigen::MatrixXd* A();
        void set_A(pMatrixXd);
        pMatrixXd get_A();
        void set_original();
        bool is_original();
};

}

#endif