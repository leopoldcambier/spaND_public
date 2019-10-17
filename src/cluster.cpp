#include "spaND.h"

using namespace std;
using namespace Eigen;

namespace spaND {

Cluster::Cluster(int start, int size, int level, int order, bool should_sparsify) : 
    start_(start), size_(size), level_(level), order_(order), eliminated_(false), sparsify_(should_sparsify),
    parent_(nullptr),
    phi_(nullptr), 
    x_(nullptr)
{
    assert(start >= 0);
    reset_size(size);
}

bool Cluster::should_sparsify() const {
    return this->sparsify_;
}
void Cluster::reset_size(int size) {
    this->size_ = size;
    this->x_ = std::make_unique<Eigen::VectorXd>(this->size());
    this->x_->setZero();
}
void Cluster::set_size(int size) {
    this->size_ = size;
}
bool Cluster::is_eliminated() const {
    return this->eliminated_;
}
void Cluster::set_eliminated() {
    assert(! this->eliminated_);
    // Remove self from neighbors's edges
    for(const auto& e : this->edgesOut){        
        e->n2->edgesIn.remove_if([this](Edge* e2){return e2->n1 == this;});
    }
    for(const auto& e : this->edgesIn){
        e->n1->edgesOut.remove_if([this](const pEdge& e2){ return e2->n2 == this; });
    }
    // Delete self's edges
    edgesIn.clear();
    edgesOut.clear();
    this->eliminated_ = true;
}
void Cluster::clear_edges() {
    edgesOut.clear();
    edgesIn.clear();
}
Edge* Cluster::pivot() const {
    assert(this->edgesOut.size() > 0);
    Edge* first = this->edgesOut.begin()->get();
    assert(first->n1 == first->n2);
    return first;
}
ItRange<pEdgeIt> Cluster::edgesOutNbr() {
    assert(this->edgesOut.size() > 0);
    auto begin = pEdgeIt(this->edgesOut.begin());
    ++begin;
    auto end   = pEdgeIt(this->edgesOut.end());
    return ItRange<pEdgeIt>(begin, end);
}
ItRange<pEdgeIt> Cluster::edgesOutAll() {
    auto begin = pEdgeIt(this->edgesOut.begin());
    auto end   = pEdgeIt(this->edgesOut.end());
    return ItRange<pEdgeIt>(begin, end);
}
ItRange<std::list<Edge*>::iterator> Cluster::edgesInNbr() {
    auto begin = this->edgesIn.begin();
    auto end   = this->edgesIn.end();
    return ItRange<std::list<Edge*>::iterator>(begin, end);
}

int Cluster::nnbr_in_self_out() const {
    return this->edgesOut.size() + this->edgesIn.size();
}

int Cluster::nnbr_self_out() const {
    return this->edgesOut.size();
}

void Cluster::set_vector(const VectorXd& b) {
    assert(x_ != nullptr);
    if(children_.size() == 0) { // No children -> leaf        
        (*x_) = b.segment(start_, x_->size());
    } else {
        x_->setZero();        
    }
}

void Cluster::extract_vector(VectorXd& b) const {
    assert(x_ != nullptr);
    if(children_.size() == 0) { // No children -> leaf
        b.segment(start_, x_->size()) = (*x_);
    }
}

int Cluster::start() const { return start_; }
int Cluster::size() const { return size_; }
int Cluster::original_size() const { return x_->size(); }
int Cluster::level() const { return level_; }
int Cluster::order() const { return order_; }
MatrixXd* Cluster::phi() const { return phi_.get(); }
void Cluster::set_phi(pMatrixXd phi) { 
    assert(phi->rows() == this->size());
    phi_ = move(phi);
}
const std::vector<Cluster*>& Cluster::children() const { return children_; }
void Cluster::add_children(Cluster* c) { children_.push_back(c); }
Cluster* Cluster::parent() const { return parent_; }
void Cluster::set_parent(Cluster* p) { parent_ = p; }

void Cluster::add_edge(pEdge e) {
    assert(this == e->n1);
    // Pivot goes in front
    if(e->n1 == e->n2) {
        this->edgesOut.insert(this->edgesOut.begin(), move(e));
    // Rest doesn't matter
    } else {
        e->n2->edgesIn.push_back(e.get());
        this->edgesOut.push_back(move(e));
    }
}

Segment Cluster::head_x() {
    return this->x_->segment(0, this->size());
}

const pVectorXd& Cluster::get_x() const {
    return this->x_;
}

void Cluster::assertDuplicates() {
    // Check for duplicates
    if(this->edgesIn.size() > 1) {
        auto first   = this->edgesIn.begin();
        auto second  = this->edgesIn.begin();
        second++;
        auto end     = this->edgesIn.end();
        while(second != end) {
            assert( (*first)->n1 != (*second)->n1 );
            assert( (*first)->n1->order() != (*second)->n1->order() );
            first++;
            second++;
        }
    }
    if(this->edgesOut.size() > 1) {
        auto first   = this->edgesOut.begin();
        auto second  = this->edgesOut.begin();
        second++;
        auto end     = this->edgesOut.end();
        while(second != end) {
            assert( (*first)->n2 != (*second)->n2 );
            assert( (*first)->n2->order() != (*second)->n2->order() );
            first++;
            second++;
        }
    }
}

// Put the out edge in some stable order, with the pivot in front
void Cluster::sort_edges() {
    this->edgesOut.sort([](const pEdge& lhs, const pEdge& rhs){
        if(lhs->n1 == lhs->n2) { // lhs is a pivot
            return true;
        } else if(rhs->n1 == rhs->n2) { // rhs is a pivot
            return false;
        } else {
            return lhs->n2->order() < rhs->n2->order(); 
        }
    });
    this->edgesIn.sort([](const Edge* lhs, const Edge* rhs){
        return lhs->n1->order() < rhs->n1->order();
    });
    this->assertDuplicates();
}

void compute_depth(const Cluster* n, int& depth) {
    if(n->parent() != nullptr) {
        depth += 1;
        compute_depth(n->parent(), depth);
    }
}

int Cluster::depth() const {
    int depth = 0;
    compute_depth(this, depth);
    return depth;
}

int Cluster::merge_level() const {
    return this->level() - this->depth();
}

string Cluster::get_name() const {
    return to_string(this->order_);
}

}
