#include "spaND.h"

using namespace std;
using namespace Eigen;

namespace spaND {

/** Iterators **/
pEdgeIt::pEdgeIt(const std::list<pEdge>::iterator& it) {
    this->current = it;
}
pEdgeIt::pEdgeIt(const pEdgeIt& other) {
    this->current = other.current;
}
Edge* pEdgeIt::operator*() const {
    return this->current->get();
}        
pEdgeIt& pEdgeIt::operator=(const pEdgeIt& other) {
    this->current = other.current;
    return *this;
}
bool pEdgeIt::operator!=(const pEdgeIt& other) const {
    return this->current != other.current;
}
bool pEdgeIt::operator==(const pEdgeIt& other) const {
    return !(*this != other);
}
pEdgeIt& pEdgeIt::operator++() {
    this->current++;
    return *this;
}

/** Edge **/
Edge::Edge(Cluster* n1, Cluster* n2, pMatrixXd A, bool original) : n1(n1), n2(n2), original(original) {
    assert(A != nullptr);
    A21 = move(A);
    assert(A21->rows() == n2->size());
    assert(A21->cols() == n1->size());
}
MatrixXd* Edge::A() {
    return this->A21.get();
}
void Edge::set_A(pMatrixXd A) {
    this->A21 = move(A);
}
pMatrixXd Edge::get_A() {
    return move(this->A21);
}
bool Edge::is_original() {
    return this->original;
}
void Edge::set_original() {
    this->original = true;
}

}