#ifndef __OPERATIONS_H__
#define __OPERATIONS_H__

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
#include <Eigen/LU>
#include <metis.h>
#include <numeric>
#include <assert.h>
#include <limits>
#include <memory>

#include "spaND.h"

namespace spaND {

/** An operation applied on the matrix **/

struct Operation {
    public:
        virtual void bwd() = 0;
        virtual void fwd() = 0;
        virtual void diag() {}; // Default is nothing to do
        virtual long long nnz() = 0;
        virtual std::string name() {
            return "Operation";
        };
        virtual ~Operation() {};
};

struct Gemm : public Operation {
    protected:
        Segment self;
        Segment nbr;
        pMatrixXd A;
        Eigen::VectorXd* Adiag;
    public:
        Gemm(Cluster* self, Cluster* nbr, pMatrixXd A);
        Gemm(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag);        
        long long nnz();
        std::string name();
};

struct GemmIn : public Gemm {
    public:
        GemmIn(Cluster* self, Cluster* nbr, pMatrixXd A);
        GemmIn(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag);
        void fwd();
        void bwd();
};

struct GemmOut : public Gemm {
    public:
        GemmOut(Cluster* self, Cluster* nbr, pMatrixXd A);
        GemmOut(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag);
        void fwd();
        void bwd();
};

struct GemmSymmIn : public Gemm {    
    public:
        GemmSymmIn(Cluster* self, Cluster* nbr, pMatrixXd A);
        GemmSymmIn(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag);
        void fwd();
        void bwd();
};

struct GemmSymmOut : public Gemm {    
    public:
        GemmSymmOut(Cluster* self, Cluster* nbr, pMatrixXd A);
        GemmSymmOut(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag);
        void fwd();
        void bwd();
};

struct ScalingLLT : public Operation {
    private:
        Segment s;
        pMatrixXd LLT;
    public:
        ScalingLLT(Cluster* n1, pMatrixXd LLT);
        void fwd();
        void bwd();
        long long nnz();
        std::string name();
};

struct ScalingPLUQ : public Operation {
    private:
        Segment xs;
        pMatrixXd L; // _not_ unit-diagonal
        pMatrixXd U; 
        pVectorXi p;
        pVectorXi q;
    public:
        ScalingPLUQ(Cluster* n1, pMatrixXd L, pMatrixXd U, pVectorXi p, pVectorXi q);
        void fwd();
        void bwd();
        long long nnz();
        std::string name();
};

struct ScalingLDLT : public Operation {
    private:
        Segment xs;
        pMatrixXd L;
        pVectorXd s;
        pVectorXi p;
    public:
        ScalingLDLT(Cluster* n1, pMatrixXd L, pVectorXd s, pVectorXi p);
        void fwd();
        void bwd();
        void diag();
        long long nnz();
        std::string name();
};

struct Orthogonal : public Operation {
    private:
        Segment s;
        pMatrixXd v;
        pVectorXd h;
    public:
        Orthogonal(Cluster* self, pMatrixXd v, pVectorXd h);
        void fwd();
        void bwd();
        long long nnz();
        std::string name();
};

struct Merge : public Operation {
    Cluster* parent;
    public:
        Merge(Cluster* parent);
        void fwd();
        void bwd();
        long long nnz();
        std::string name();
};

struct Split : public Operation {
    Cluster* original;
    Cluster* sibling;
    public:
        Split(Cluster* original, Cluster* sibling);
        void fwd();
        void bwd();       
        long long nnz();
        std::string name();
};

}

#endif