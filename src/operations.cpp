#include "spaND.h"

namespace spaND {

Gemm::Gemm(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag) : 
    self(self->head_x()), nbr(nbr->head_x()), A(move(A)), Adiag(Adiag) {};

Gemm::Gemm(Cluster* self, Cluster* nbr, pMatrixXd A) : 
    self(self->head_x()), nbr(nbr->head_x()), A(move(A)), Adiag(nullptr) {};

GemmIn::GemmIn(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag) : Gemm(self, nbr, move(A), Adiag) {};
GemmIn::GemmIn(Cluster* self, Cluster* nbr, pMatrixXd A) : Gemm(self, nbr, move(A)) {};

GemmOut::GemmOut(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag) : Gemm(self, nbr, move(A), Adiag) {};
GemmOut::GemmOut(Cluster* self, Cluster* nbr, pMatrixXd A) : Gemm(self, nbr, move(A)) {};

GemmSymmIn::GemmSymmIn(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag) : Gemm(self, nbr, move(A), Adiag) {};
GemmSymmIn::GemmSymmIn(Cluster* self, Cluster* nbr, pMatrixXd A) : Gemm(self, nbr, move(A)) {};

GemmSymmOut::GemmSymmOut(Cluster* self, Cluster* nbr, pMatrixXd A, Eigen::VectorXd* Adiag) : Gemm(self, nbr, move(A), Adiag) {};
GemmSymmOut::GemmSymmOut(Cluster* self, Cluster* nbr, pMatrixXd A) : Gemm(self, nbr, move(A)) {};

long long Gemm::nnz() {
    return self.size() * nbr.size() + (Adiag == nullptr ? 0 : self.size());
}

std::string Gemm::name() {
    return "Gemm";
}

void GemmOut::fwd() {    
    if(nbr.size() > 0 && self.size() > 0) {
        if(Adiag != nullptr) {
            nbr -= ((*A) * Adiag->asDiagonal() * self);
        } else {
            gemv_notrans(A.get(), &self, &nbr);
        }  
    }      
}

void GemmOut::bwd() {};

void GemmIn::fwd() {};

void GemmIn::bwd() {
    if(nbr.size() > 0 && self.size() > 0) {
        if(Adiag != nullptr) {
            self -= (Adiag->asDiagonal() * (*A) * nbr);
        } else {
            gemv_notrans(A.get(), &nbr, &self);
        }       
    } 
}

void GemmSymmOut::fwd() {
    if(nbr.size() > 0 && self.size() > 0) {
        if(Adiag != nullptr) {
            nbr -= ((*A) * Adiag->asDiagonal() * self);
        } else {            
            gemv_notrans(A.get(), &self, &nbr);
        }
    }
}

void GemmSymmOut::bwd() {
    if(nbr.size() > 0 && self.size() > 0) {
        if(Adiag != nullptr) {
            self -= Adiag->asDiagonal() * ((*A).transpose() * nbr);
        } else {            
            gemv_trans(A.get(), &nbr, &self);
        }
    }
}

void GemmSymmIn::fwd() {
    if(nbr.size() > 0 && self.size() > 0) {
        if(Adiag != nullptr) {
            nbr -= ((*A).transpose() * Adiag->asDiagonal() * self);
        } else {
            gemv_trans(A.get(), &self, &nbr);
        }
    }
}

void GemmSymmIn::bwd() {
    if(nbr.size() > 0 && self.size() > 0) {
        if(Adiag != nullptr) {
            self -= Adiag->asDiagonal() * ((*A) * nbr);
        } else {            
            gemv_notrans(A.get(), &nbr, &self);
        }
    }
}

ScalingLLT::ScalingLLT(Cluster* n1, pMatrixXd LLT) : s(n1->head_x()), LLT(std::move(LLT)) {}

void ScalingLLT::fwd() {
    trsv(LLT.get(), &s, CblasLower, CblasNoTrans, CblasNonUnit);
}

void ScalingLLT::bwd() {
    trsv(LLT.get(), &s, CblasLower, CblasTrans, CblasNonUnit);
}

long long ScalingLLT::nnz() {
    return s.size() * (s.size() + 1) / 2;
}
std::string ScalingLLT::name() {
    return "ScalingLLT";
}

ScalingPLUQ::ScalingPLUQ(Cluster* s, pMatrixXd L, pMatrixXd U, pVectorXi p, pVectorXi q) : 
    xs(s->head_x()), L(std::move(L)), U(std::move(U)), p(std::move(p)), q(std::move(q)) {}
void ScalingPLUQ::fwd() {
    xs = p->asPermutation().transpose() * xs;
    trsv(L.get(), &xs, CblasLower, CblasNoTrans, CblasNonUnit);
}
void ScalingPLUQ::bwd() {
    trsv(U.get(), &xs, CblasUpper, CblasNoTrans, CblasNonUnit);
    xs = q->asPermutation().transpose() * xs;
}
long long ScalingPLUQ::nnz() {            
    return (xs.size() * xs.size()) + 2 * xs.size();
}
std::string ScalingPLUQ::name() {
    return "ScalingPLUQ";
}

ScalingLDLT::ScalingLDLT(Cluster* n1, pMatrixXd L, pVectorXd s, pVectorXi p) : 
    xs(n1->head_x()), L(std::move(L)), s(move(s)), p(move(p)) {}
void ScalingLDLT::fwd() {
    xs = p->asPermutation().transpose() * xs;
    trsv(L.get(), &xs, CblasLower, CblasNoTrans, CblasNonUnit);
}
void ScalingLDLT::bwd() {
    trsv(L.get(), &xs, CblasLower, CblasTrans, CblasNonUnit);
    xs = p->asPermutation() * xs;
}
void ScalingLDLT::diag() {
    xs = s->cwiseInverse().asDiagonal() * xs;
}
long long ScalingLDLT::nnz() {
    return (xs.size() * (xs.size()+1))/2 + 2 * xs.size();
}
std::string ScalingLDLT::name() {
    return "ScalingLDLT";
}

Orthogonal::Orthogonal(Cluster* self, pMatrixXd v, pVectorXd h) : 
    s(self->head_x()), v(std::move(v)), h(std::move(h)) {
        assert(s.size() == this->v->rows());
    }
void Orthogonal::fwd() {
    ormqr_trans(v.get(), h.get(), &s);            
}
void Orthogonal::bwd() {
    ormqr_notrans(v.get(), h.get(), &s);
}
long long Orthogonal::nnz() {
    return s.size() * s.size();
}
std::string Orthogonal::name() {
    return "Orthogonal";
}

Merge::Merge(Cluster* parent) : parent(parent) {}
void Merge::fwd() {
    int k = 0;
    for(auto c: parent->children()) {
        for(int i = 0; i < c->size(); i++) {
            (*parent->get_x())[k] = (*c->get_x())[i];
            k++;
        }
    }
    assert(k == parent->get_x()->size());
}
void Merge::bwd() {
    int k = 0;
    for(auto c: parent->children()) {
        for(int i = 0; i < c->size(); i++) {
            (*c->get_x())[i] = (*parent->get_x())[k];
            k++;
        }
    }
    assert(k == parent->get_x()->size());
}
long long Merge::nnz() {
    return 0;
}
std::string Merge::name() {
    return "Merge";
}

Split::Split(Cluster* original, Cluster* sibling): original(original), sibling(sibling) {}
void Split::fwd() {
    assert(original->size() + sibling->size() == original->original_size());
    *sibling->get_x() = original->get_x()->segment(original->size(), sibling->size());
}
void Split::bwd() {
    assert(original->size() + sibling->size() == original->original_size());
    original->get_x()->segment(original->size(), sibling->size()) = *sibling->get_x();
}
long long Split::nnz() {
    return 0;
}
std::string Split::name() {
    return "Split";
}

}