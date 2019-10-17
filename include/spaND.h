#ifndef __SPAND_H__
#define __SPAND_H__

#include <memory>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <metis.h>

namespace spaND {

struct Edge;
struct Cluster;
struct pEdgeIt;
struct Operation;
struct ClusterID;
struct Profile;
struct Log;

enum class ScalingKind { SVD, PLU, PLUQ, EVD, LLT, LDLT };
enum class PartKind { MND, RB };
enum class SymmKind { SPD, SYM, GEN };

inline std::string scaling2str(ScalingKind sk) {
    return (sk == ScalingKind::LLT  ? "LLT" :
           (sk == ScalingKind::PLU  ? "PLU" :
           (sk == ScalingKind::PLUQ ? "PLUQ" :
           (sk == ScalingKind::SVD  ? "SVD" :
           (sk == ScalingKind::EVD  ? "EVD" :
           (sk == ScalingKind::LDLT ? "LDLT" :
           "ErrorUnknownScaling"))))));    
};

inline std::string symm2str(SymmKind sk) {
    return (sk == SymmKind::SPD ? "SPD" :
           (sk == SymmKind::SYM ? "SYM" :
           (sk == SymmKind::GEN ? "GEN" :
           "ErrorUnknownSymmetry")));    
};

inline std::string part2str(PartKind pk) {
    return (pk == PartKind::MND ? "MND" :
           (pk == PartKind::RB  ? "RB" :
           "ErrorUnknownPart"));    
};

typedef std::unique_ptr<Eigen::MatrixXd> pMatrixXd;
typedef std::unique_ptr<Eigen::VectorXd> pVectorXd;
typedef std::unique_ptr<Eigen::VectorXi> pVectorXi;
typedef std::unique_ptr<Cluster>         pCluster;
typedef std::unique_ptr<Edge>            pEdge;
typedef std::unique_ptr<Operation>       pOperation;

typedef Eigen::SparseMatrix<double, 0, int> SpMat;
typedef Eigen::VectorBlock<Eigen::Matrix<double, -1,  1, 0, -1,  1>, -1> Segment;
typedef Eigen::Block<Eigen::Matrix<double, -1, -1, 0, -1, -1>, -1, -1> MatrixBlock;

}

#include "util.h"
#include "tree.h"
#include "operations.h"
#include "partition.h"
#include "edge.h"
#include "cluster.h"
#include "is.h"


#endif