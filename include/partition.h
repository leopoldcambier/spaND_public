#ifndef PARTITION_H
#define PARTITION_H

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <list>
#include <set>
#include <tuple>
#include <utility>
#include <queue>
#include <metis.h>
#include <numeric>
#include <assert.h>
#include <limits>

#include "spaND.h"

namespace spaND {

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

SepID merge(SepID& s);
ClusterID merge_if(ClusterID& c, int lvl);

std::ostream& operator<<(std::ostream& os, const SepID& s);
std::ostream& operator<<(std::ostream& os, const ClusterID& c);

std::vector<int> partition_RB(SpMat &A, int nlevels, bool verb, Eigen::MatrixXd* Xcoo);
void partition_metis(std::vector<int> &colptr, std::vector<int> &rowval, std::vector<int> &colptrtmp, std::vector<int> &rowvaltmp, std::vector<int> &dofs, std::vector<int> &parts, bool useVertexSep);
void partition_geo(std::vector<int> &colptr, std::vector<int> &rowval, std::vector<int> &dofs, std::vector<int> &parts, Eigen::MatrixXd *X, std::vector<int> &invp);
std::vector<ClusterID> partition_modifiedND(SpMat &A, int nlevels, bool verb, bool useVertexSep, Eigen::MatrixXd* Xcoo);

int part_at_lvl(int part_leaf, int lvl, int nlevels);
SepID find_highest_common(SepID n1, SepID n2);
std::vector<ClusterID> partition_recursivebissect(SpMat &A, int nlevels, bool verb, Eigen::MatrixXd* Xcoo);

}

#endif