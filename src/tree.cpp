#include "tree.h"

using namespace std;
using namespace Eigen;

/** Cluster & Edges & Iterators **/

int   Cluster::get_level() {
    return this->id.self.lvl;
}
Edge* Cluster::pivot() {
    assert(this->edgesOut.size() > 0);
    Edge* first = *this->edgesOut.begin();
    assert(first->n1 == first->n2);
    return first;
}
OutNbr Cluster::edgesOutNbr() {    
    return OutNbr(this);
}
int Cluster::nnbr_in_self_out() {
    return this->edgesOut.size() + this->edgesIn.size();
}
void Cluster::set_vector(const Eigen::VectorXd& b) {
    assert(x != nullptr);
    if(start >= 0) {
        for(int i = 0; i < x->size(); i++) {
            (*x)[i] = b[start + i];
        }
    } else {
        for(int i = 0; i < x->size(); i++) {
            (*x)[i] = 0;
        }
    }
}
void Cluster::extract_vector(Eigen::VectorXd& b) {
    assert(x != nullptr);
    assert(start >= 0);
    for(int i = 0; i < x->size(); i++) {
        b[start + i] = (*x)[i];
    }
}

ClusterID invalidClusterID() {
    return ClusterID(SepID());
}

Edge::Edge(Cluster* n1, Cluster* n2, Eigen::MatrixXd* A) : n1(n1), n2(n2) {
    assert(n1->order <= n2->order);
    assert(A != nullptr);
    A21 = A;
    assert(A21->rows() == n2->size);
    assert(A21->cols() == n1->size);
    A12 = nullptr;
}
Edge::Edge(Cluster* n1, Cluster* n2, Eigen::MatrixXd* A, Eigen::MatrixXd* AT) : n1(n1), n2(n2) {
    assert(n1->order <= n2->order);
    assert(A != nullptr);
    
    A21 = A;
    assert(A21->rows() == n2->size);
    assert(A21->cols() == n1->size);

    A12 = AT;
    if(AT != nullptr) { 
        assert(A12->cols() == n2->size);
        assert(A12->rows() == n1->size);
    }
}
Eigen::MatrixXd* Edge::ALow() {
    return this->A21;
}
Eigen::MatrixXd* Edge::AUpp() {
    return this->A12;
}
Eigen::MatrixXd* Edge::APiv() {
    assert( this->n1 == this->n2);
    assert( this->A12 == nullptr);
    return this->A21;
}
void Edge::set_APiv(Eigen::MatrixXd* A) {
    assert( this->n1 == this->n2);    
    assert( A != nullptr );
    this->A21 = A;
}
void Edge::set_ALow(Eigen::MatrixXd* A) {
    this->A21 = A;
}
void Edge::set_AUpp(Eigen::MatrixXd* A) {
    this->A12 = A;
}

std::ostream& operator<<(std::ostream& os, const SepID& s) {
    os << "(" << s.lvl << " " << s.sep << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const ClusterID& c) {
    os << "(" << c.self << ":" << c.l << ";" << c.r << ")";
    return os;
}

SepID merge(SepID& s) {
    return SepID(s.lvl + 1, s.sep / 2);
}

ClusterID merge_if(ClusterID& c, int lvl) {
    auto left  = c.l.lvl < lvl ? merge(c.l) : c.l;
    auto right = c.r.lvl < lvl ? merge(c.r) : c.r;
    return ClusterID(c.self, left, right);
}

/**
 * Tree
 */

/** 
 * Indicates default values for all parameters
 */
void Tree::init(int nlevels) {
    assert(nlevels > 0);
    this->verb = true;
    this->symmetry = true;
    this->geo = false;
    this->use_vertex_sep = true;
    this->preserve = false;
    this->nphis = -1;
    this->N = -1;
    this->ilvl = 0;
    this->nlevels = nlevels;
    this->tol = 10.0;
    this->skip = 100;
    this->scale = true;
    this->scale_kind = ScalingKind::PLU;
    this->ortho = true;
    this->Xcoo = nullptr;
    this->phi = nullptr;
    this->adaptive = true;
    this->use_want_sparsify = true;
    this->monitor_condition_pivots = false;

    this->tprof = vector<Profile>(nlevels, Profile());
    this->log   = vector<Log>(nlevels, Log());
    this->bottoms = vector<list<Cluster*>>(nlevels);
    this->current_bottom = 0;
}

Tree::Tree(int nlevels) {
    this->init(nlevels);
}

void Tree::set_verb(bool verb) {
    this->verb = verb;
}

void Tree::set_symmetry(bool symm) {
    this->symmetry = symm;
}

void Tree::set_Xcoo(Eigen::MatrixXd* Xcoo) {
    this->Xcoo = Xcoo;
}

void Tree::set_use_geo(bool geo) {
    this->geo = geo;
}

void Tree::set_phi(Eigen::MatrixXd* phi) {
    this->phi = phi;
}

void Tree::set_preserve(bool preserve) {
    this->preserve = preserve;
}

void Tree::set_use_vertex_sep(bool use_vertex_sep) {
    this->use_vertex_sep = use_vertex_sep;
}

void Tree::set_tol(double tol) {
    this->tol = tol;
}

void Tree::set_skip(int skip) {
    this->skip = skip;
}

void Tree::set_scale(bool scale) {
    this->scale = scale;
}

void Tree::set_ortho(bool ortho) {
    this->ortho = ortho;
}

void Tree::set_scaling_kind(ScalingKind scaling_kind) {
    this->scale_kind = scaling_kind;
}

void Tree::set_use_sparsify(bool use) {
    this->use_want_sparsify = use;
}

void Tree::set_monitor_condition_pivots(bool monitor) {
    this->monitor_condition_pivots = monitor;
}

int Tree::nclusters_left() {
    int n = 0;
    for(auto self : bottom_current()){
        if(! self->is_eliminated()) n++;
    }
    return n;
}

int Tree::ndofs_left() {
    int n = 0;
    for(auto self : bottom_current()){
        if(! self->is_eliminated()) n += self->size;
    }
    return n;
}

int Tree::get_N() const {
    return this->N;
}

VectorXi Tree::get_assembly_perm() const {
    return this->perm;
}

bool Tree::is_factorized() const {
    assert(this->ilvl >= 0 && this->ilvl <= this->nlevels);
    return this->ilvl == this->nlevels;
}

const list<Cluster*>& Tree::bottom_current() const {
    assert(current_bottom < bottoms.size());
    return bottoms[current_bottom];
}

const list<Cluster*>& Tree::bottom_original() const {
    assert(bottoms.size() > 0);
    return bottoms[0];
}

/** 
 * Goes over all edges and collect stats on size and count
 */
void Tree::stats() {
    Stats<int> cluster_size = Stats<int>();
    Stats<int> edge_size = Stats<int>();
    Stats<int> edge_count = Stats<int>();
    for(auto self : bottom_original()) {
        cluster_size.addData(self->size);
        edge_count.addData(self->edgesOut.size());
        for(auto edge : self->edgesOut) {
            assert(edge->n1 == self);
            edge_size.addData(edge->n1->size * edge->n2->size);
        }
    }
    printf("    Cluster size: %9d | %9d | %9d | %9f\n", cluster_size.getCount(), cluster_size.getMin(), cluster_size.getMax(), cluster_size.getMean());
    printf("    Edge sizes:   %9d | %9d | %9d | %9f\n", edge_size.getCount(),    edge_size.getMin(),    edge_size.getMax(),    edge_size.getMean());
    printf("    Edge count:   %9d | %9d | %9d | %9f\n", edge_count.getCount(),   edge_count.getMin(),   edge_count.getMax(),   edge_count.getMean());
}

// We sparsify a cluster when both his left and right have been eliminated
bool Tree::want_sparsify(Cluster* self) {
    assert(self->id.self.lvl >= this->ilvl);
    if(this->use_want_sparsify) {        
        return (self->id.l.lvl == this->ilvl) && (self->id.r.lvl == this->ilvl);
    } else {
        return true;
    }
}

void Tree::print_summary() const {
    printf("Tree with %d levels (%d eliminated so far)\n", this->nlevels, this->ilvl);
}

/**
 * PARTITIONING SUBROUTINES
 */
void partition_metis(VectorXi &colptr, VectorXi &rowval, VectorXi &colptrtmp, VectorXi &rowvaltmp, VectorXi &dofs, VectorXi &parts, int *options, bool useVertexSep) {
    int size = dofs.size();
    int sepsize = 0;
    // 1) Build smaller matrix
    colptrtmp[0] = 0;
    for(int i = 0; i < size; i++) { // New dof
        colptrtmp[i+1] = colptrtmp[i];
        int i_ = dofs[i];
        for(int k = colptr[i_]; k < colptr[i_+1]; k++) { // Go through its edges in A
            int k_ = rowval[k];
            if(k_ == i_) // Skip diagonal
                continue;
            int *id = lower_bound(dofs.data(), dofs.data()+size, k_); // Is k a neighbor of i ?
            int pos = id - dofs.data();
            if(pos < size && dofs[pos] == k_) {
                rowvaltmp[colptrtmp[i+1]] = pos;                
                colptrtmp[i+1] += 1;
            }
        }
    }
    // 2) Call metis
    if(size > 0 && useVertexSep) {
        assert(METIS_OK == METIS_ComputeVertexSeparator(&size, colptrtmp.data(), rowvaltmp.data(), nullptr, options, &sepsize, parts.data()));
    } else if (size > 0 && (! useVertexSep)) {
        int one = 1;
        int two = 2;
        int objval = 0;
        assert(METIS_OK == METIS_PartGraphRecursive(&size, &one, colptrtmp.data(), rowvaltmp.data(), nullptr, nullptr, nullptr, &two, nullptr, nullptr, options, &objval, parts.data()));
        // Update part with vertex sep
        for(int i = 0; i < size; i++) { // Those with 0, connected to > 1, get a 2
            if (parts[i] != 0) {
                continue;
            }
            for(int k = colptrtmp[i]; k < colptrtmp[i+1]; k++) { 
                int j = rowvaltmp[k];
                if (parts[j] == 1) {
                    parts[i] = 2;
                    break;
                }
            }
        }
    }
}

void partition_geo(VectorXi &colptr, VectorXi &rowval, VectorXi &dofs, VectorXi &parts, MatrixXd *X, VectorXi &invp) {
    assert(X->cols() == colptr.size()-1);
    assert(invp.size() >= X->cols());
    assert(X->cols() == parts.size());
    int N = dofs.size();
    if(N == 0)
        return;
    VectorXi dofs_tmp = dofs;
    // Get dimension over which to cut
    int bestdim = -1;
    double maxrange = -1.0;
    for(int d = 0; d < X->rows(); d++) {
        double maxi = numeric_limits<double>::lowest();
        double mini = numeric_limits<double>::max();
        for(int i = 0; i < dofs.size(); i++) {
            int idof = dofs[i];
            maxi = max(maxi, (*X)(d,idof));
            mini = min(mini, (*X)(d,idof));
        }
        double range = maxi - mini;
        if(range > maxrange) {
            bestdim = d;
            maxrange = range;
        }
    }
    assert(bestdim >= 0);
    for(int i = 0; i < N; i++) {
        parts[i] = dofs[i];
    }
    // Sort dofs based on X[dim,:]
    std::sort(parts.data(), parts.data() + N, [&](int i, int j) { return (*X)(bestdim,i) < (*X)(bestdim,j); });
    // Get middle value
    int mid = parts[N/2];
    double midv = (*X)(bestdim,mid);
    // First, zero (-1) out all the neighbors of dofs
    for(int i = 0; i < N; i++) {
        int dof = dofs[i];
        invp[dof] = -1;
        for(int k = colptr[dof]; k < colptr[dof+1]; k++) {
            int dofn = rowval[k];
            invp[dofn] = -1;
        }
    }
    // Put in part & get inverse perm
    for(int i = 0; i < N; i++) {
        int dof = dofs[i];
        invp[dof] = i;
        if ( (*X)(bestdim,dof) < midv ) {
            parts[i] = 0;
        } else {
            parts[i] = 1;
        }
    }
    // Compute separator
    for(int i = 0; i < N; i++) {
        int dof = dofs[i];
        // If in RIGHT
        if(parts[i] == 0)
            continue;
        // If neighbor in LEFT (parts[i] = 1 here)
        for(int k = colptr[dof]; k < colptr[dof+1]; k++) {
            int dofn = rowval[k];
            int in = invp[dofn];
            if(in == -1) // Skip: no in dofs
                continue;
            if(parts[in] == 0) {
                parts[i] = 2;
                break;
            }
        }
    }
}

/**
 * Partition & Order
 * A is assumed to have a symmetric pattern. 
 * The diagonal is irrelevant.
 */
void Tree::partition(SpMat &A) {
    assert(this->ilvl == 0);
    timer tstart = wctime();
    // Basic stuff
    double t_nd = 0.0;
    int nlevels = this->nlevels;
    int N = A.rows();
    this->N = N;
    assert(A.rows() == A.cols());
    assert(nlevels > 0);
    if(this->geo) {
        assert(this->Xcoo != nullptr);
        assert(this->Xcoo->cols() == this->N);
    }
    MatrixXd *X = this->Xcoo;
    bool useVertexSep = this->use_vertex_sep;
    // Print
    if(this->verb) {
        if (this->geo) {
            cout << "Geometric partitioning of matrix with " << N << " dofs with " << nlevels << " levels in " << X->rows() << "D" << endl;
        } else {
            cout << "Algebraic (with vertex sep ? " << useVertexSep << ") partitioning of matrix with " << N << " dofs with " << nlevels << " levels" << endl;
        }
    }
    // Metis options
    int options[METIS_NOPTIONS];
    assert(METIS_OK == METIS_SetDefaultOptions(options));
    // CSC format
    A.makeCompressed();
    int nnz = A.nonZeros();
    VectorXi rowval = Map<VectorXi>(A.innerIndexPtr(), nnz);
    VectorXi colptr = Map<VectorXi>(A.outerIndexPtr(), N + 1);
    // Workspace for Metis partitioning
    VectorXi colptrtmp(N+1);
    VectorXi rowvaltmp(nnz);
    VectorXi parttmp(N);
    // The data to partition
    vector<VectorXi> doms(1);
    auto init = VectorXi::LinSpaced(N, 0, N-1);
    doms[0] = init;
    // Partitioning results
    this->part = vector<ClusterID>(N, ClusterID(SepID(nlevels-1,0),SepID(nlevels-1,0),SepID(nlevels-1,0)));
    // Where we store results
    for(int depth = 0; depth < nlevels - 1; depth++) {
        Stats<int> sepsstats = Stats<int>();
        int level = nlevels - depth - 1;
        timer t0000 = wctime();
        // Get all separators
        vector<VectorXi> newdoms(pow(2, depth+1));
        for(int sep = 0; sep < pow(2, depth); sep++) {
            SepID idself = SepID(level, sep);
            // Prepare sorted dofs vector            
            VectorXi dofssep = doms[sep];
            sort(dofssep.data(), dofssep.data() + dofssep.size());
            // Generate matrix & perform ND
            timer t_ = wctime();
            if(geo) {
                partition_geo(colptr, rowval, dofssep, parttmp, X, colptrtmp); // colptrtmp can be any array with >= N elements
            } else {
                partition_metis(colptr, rowval, colptrtmp, rowvaltmp, dofssep, parttmp, options, useVertexSep); // result in parttmp[0...size-1]
            }
            timer t__ = wctime();
            t_nd += elapsed(t_, t__);
            // Update left/right
            SepID idleft  = SepID(level - 1, 2 * sep);
            SepID idright = SepID(level - 1, 2 * sep + 1);
            for (int i = 0; i < dofssep.size(); i++)
            {
                int j = dofssep[i];
                // Update self
                if (this->part[j].self == idself) {
                    if (parttmp[i] == 0) {
                        this->part[j].self = idleft;    
                    } else if(parttmp[i] == 1) {
                        this->part[j].self = idright;
                    }
                }
                // Update left/right               
                if (parttmp[i] == 0) { // Boundary \union interior                    
                    if (this->part[j].l == idself) {
                        this->part[j].l = idleft;
                    } 
                    if (this->part[j].r == idself) {
                        this->part[j].r = idleft;
                    }
                } else if (parttmp[i] == 1) { // Boundary \union interior                    
                    if (this->part[j].l == idself) {
                        this->part[j].l = idright;
                    } 
                    if (this->part[j].r == idself) {
                        this->part[j].r = idright;
                    }                
                } else if (parttmp[i] == 2) { // Separator \inter interior
                    if (this->part[j].self == idself) {                        
                        this->part[j].l = idleft;
                        this->part[j].r = idright;
                    }
                }
            }
            // Build left/right
            auto isleft   = [this, &idleft ](int i){return (this->part[i].self == idleft || this->part[i].l == idleft || this->part[i].r == idleft);};
            auto isright  = [this, &idright ](int i){return (this->part[i].self == idright || this->part[i].l == idright || this->part[i].r == idright);};
            auto isnewsep = [this, &idself, &idleft, &idright ](int i){return (this->part[i].self == idself && this->part[i].l == idleft && this->part[i].r == idright);};
            auto nleft  = count_if(dofssep.data(), dofssep.data() + dofssep.size(), isleft);
            auto nright = count_if(dofssep.data(), dofssep.data() + dofssep.size(), isright);
            auto nnewsep = count_if(dofssep.data(), dofssep.data() + dofssep.size(), isnewsep);
            sepsstats.addData(nnewsep);
            VectorXi newleft  = VectorXi(nleft);
            VectorXi newright = VectorXi(nright);            
            copy_if(dofssep.data(), dofssep.data() + dofssep.size(), newleft.data(),  isleft);
            copy_if(dofssep.data(), dofssep.data() + dofssep.size(), newright.data(), isright);
            newdoms[2 * sep]     = newleft;
            newdoms[2 * sep + 1] = newright;
        }
        doms = newdoms;
        timer t0001 = wctime();
        if(this->verb) printf("  Depth %2d: %3.2e s. (%5d separators, [%5d %5d], mean %6.1f)\n", 
                    depth+1, elapsed(t0000, t0001), sepsstats.getCount(), sepsstats.getMin(), sepsstats.getMax(), sepsstats.getMean());
    }         
    // Logging
    for(int i = 0; i < this->part.size(); i++) {
        this->log[this->part[i].self.lvl].dofs_nd += 1;
    }
    this->log[nlevels-1].dofs_left_nd = 0;
    for(int l = nlevels-2; l >= 0; l--) {
        this->log[l].dofs_left_nd = this->log[l+1].dofs_nd + this->log[l+1].dofs_left_nd;
    }
    // Compute the ordering & associated permutation
    perm = VectorXi::LinSpaced(N, 0, N-1);
    // Sort according to the ND ordering FIRST & the cluster merging process THEN
    vector<ClusterID> partmerged = this->part;
    auto compIJ = [&partmerged](int i, int j){return (partmerged[i] < partmerged[j]);};
    stable_sort(perm.data(), perm.data() + perm.size(), compIJ); // !!! STABLE SORT MATTERS !!!
    for(int lvl = 1; lvl < nlevels; lvl++) {
        transform(partmerged.begin(), partmerged.end(), partmerged.begin(), [&lvl](ClusterID s){return merge_if(s, lvl);}); // lvl matters    
        stable_sort(perm.data(), perm.data() + perm.size(), compIJ); // !!! STABLE SORT MATTERS !!!
    }
    // Apply permutation
    vector<ClusterID> partpermed(N);
    transform(perm.data(), perm.data() + perm.size(), partpermed.begin(), [this](int i){return this->part[i];});
    // Create the initial clusters
    vector<Stats<int>> clustersstats(nlevels, Stats<int>());
    int order = 0;
    for(int k = 0; k < N; ) {
        int knext = k+1;
        ClusterID id = partpermed[k];
        while(knext < N && partpermed[knext] == id) { knext += 1; }        
        Cluster* self = new Cluster(k, knext - k, id, order);
        bottoms[0].push_back(self);
        clustersstats[self->get_level()].addData(self->size);
        k = knext;
        order ++;
    }
    if(this->verb) {
        printf("Clustering size statistics (# of leaf-clusters at each level of the ND hierarchy)\n");
        printf("Lvl     Count       Min       Max      Mean\n");
        for(int lvl = 0; lvl < nlevels; lvl++) {
            printf("%3d %9d %9d %9d %9.0f\n", lvl, clustersstats[lvl].getCount(), clustersstats[lvl].getMin(), clustersstats[lvl].getMax(), clustersstats[lvl].getMean());
        }
    }
    // Create the cluster hierarchy
    if(this->verb) printf("Hierarchy numbers (# of cluster at each level of the cluster-hierarchy)\n");
    if(this->verb) printf("%3d %9lu\n", 0, bottoms[0].size());
    for(int lvl = 1; lvl < nlevels; lvl++) {
        auto begin = find_if(bottoms[lvl-1].begin(), bottoms[lvl-1].end(), [lvl](Cluster* s){
                        return s->get_level() > lvl; // All others should have been eliminated by now
                     });
        auto end   = bottoms[lvl-1].end();
        // Merge clusters
        for(auto self = begin; self != end; self++) {
            assert((*self)->get_level() > lvl);            
            (*self)->parentid = merge_if((*self)->id, lvl);
        }
        // Figure out who gets merged together, setup children/parent, parentID
        int order = 0;
        for(auto k = begin; k != end;) {
            // Figures out who gets merged together
            auto idparent = (*k)->parentid;
            vector<Cluster*> children;
            // Find all the guys that get merged with him
            (*k)->posparent = 0;
            children.push_back(*k);
            int children_start = (*k)->start;
            int children_size  = (*k)->size;
            k++;
            while(k != end && idparent == (*k)->parentid) {
                children.push_back(*k);
                children_size += (*k)->size;
                k++;                
            }
            Cluster* parent = new Cluster(children_start, children_size, idparent, order);
            for(auto c: children) {
                c->parent = parent;
            }
            parent->children = children;
            bottoms[lvl].push_back(parent);
            order ++;
        }
        if(this->verb) printf("%3d %9lu\n", lvl, bottoms[lvl].size());
    }
    timer tend = wctime();    
    if(this->verb) printf("Partitioning time : %3.2e s. (%3.2e s. ND)\n", elapsed(tstart, tend), t_nd);
}

/** Returns L such that
    L[l][i] = (self, left, right) 
    is the ClusterID for node i (in natural ordering) at level l of the merging process
    l=0 is the cluster-level level
 */
vector<vector<ClusterID>> Tree::get_clusters_levels() const {
    vector<vector<ClusterID>> clusters_levels(this->nlevels, vector<ClusterID>(this->N, invalidClusterID()));    
    queue<tuple<Cluster*,int>> left;
    for(auto c: bottom_original()) {
        left.push({c,0});
    }
    while(! left.empty()) {
        Cluster* n; int l;
        tie(n, l) = left.front();
        left.pop();
        for(int i = n->start; i < n->start + n->size; i++) {
            clusters_levels[l][i] = n->id;
        }
        if(n->parent != nullptr && n->parent->children.front() == n) {
            left.push({n->parent, l+1});
        }
    }
    for(int l = 0; l < this->nlevels; l++) {
        vector<ClusterID> tmp(this->N, invalidClusterID());
        for(int i = 0; i < this->N; i++) {
            tmp[this->perm[i]] = clusters_levels[l][i];
        }
        clusters_levels[l] = tmp;
    }
    return clusters_levels;
}

/** 
 * Assemble the matrix
 */
void Tree::assemble(SpMat& A) {
    assert(this->ilvl == 0);
    assert(this->N >= 0);
    timer tstart = wctime();    
    int N = this->N;
    int nlevels = this->nlevels;
    bool symmetry = this->symmetry;
    if(this->verb) cout << "Assembling (Size " << N << " with " << nlevels << " levels and symmetry " << symmetry << ")" << endl;    
    // Permute & compress the matrix for assembly
    timer t0 = wctime();
    SpMat App = symm_perm(A, perm);
    App.makeCompressed();
    SpMat App_T;
    if(! this->symmetry) {
        App_T = App.transpose();
        App_T.makeCompressed();
    }
    timer t1 = wctime();
    // Get CSC format    
    int nnz = App.nonZeros();
    VectorXi rowval = Map<VectorXi>(App.innerIndexPtr(), nnz);
    VectorXi colptr = Map<VectorXi>(App.outerIndexPtr(), N + 1);
    VectorXd nnzval = Map<VectorXd>(App.valuePtr(), nnz);
    VectorXi rowval_T;
    VectorXi colptr_T;
    VectorXd nnzval_T;
    if(! this->symmetry) {
        rowval_T = Map<VectorXi>(App_T.innerIndexPtr(), nnz);
        colptr_T = Map<VectorXi>(App_T.outerIndexPtr(), N + 1);
        nnzval_T = Map<VectorXd>(App_T.valuePtr(), nnz);
    }
    // Some edge stats
    vector<Stats<int>> edgesizestats(nlevels, Stats<int>());
    vector<Stats<int>> nedgestats(nlevels, Stats<int>());
    // Create all edges
    vector<Cluster*> cmap(N);
    for(auto self : bottom_original()) {
        assert(self->start >= 0);
        for(int k = self->start; k < self->start + self->size; k++) { cmap[k] = self; }
    }
    for(auto self : bottom_original()) {
        // Get all neighbors located after
        int col  = self->start;
        int size = self->size;
        set<Cluster*> nbrs;
        for (int j = col; j < col + size; ++j) {
            // Lower part
            for (SpMat::InnerIterator it(App,j); it; ++it) {
                Cluster* n = cmap[it.row()];  
                if ( self->order <= n->order ) { // Edges always go lower order -> higher order            
                    nbrs.insert(n);
                }
            }
            // Upper part, if necessary
            if(! this->symmetry) {
                for (SpMat::InnerIterator it(App_T,j); it; ++it) {
                    Cluster* n = cmap[it.row()];  
                    if ( self->order <= n->order ) { // Edges always go lower order -> higher order            
                        nbrs.insert(n);
                    }
                }
            }
        }
        nbrs.insert(self);
        // Go and get the edges
        for (auto nbr : nbrs) {
            MatrixXd* A = new MatrixXd();
            *A = MatrixXd::Zero(nbr->size, self->size);
            block2dense(rowval, colptr, nnzval, nbr->start, self->start, nbr->size, self->size, A, false);
            edgesizestats[self->get_level()].addData(nbr->size * self->size);
            if( (!this->symmetry) && (self != nbr) ) { // Non-pivot in non-symmetric case
                MatrixXd* A_T = new MatrixXd();
                *A_T = MatrixXd::Zero(self->size, nbr->size);
                block2dense(rowval_T, colptr_T, nnzval_T, nbr->start, self->start, nbr->size, self->size, A_T, true);
                Edge* e = new Edge(self, nbr, A, A_T);
                self->edgesOut.push_back(e);
                nbr->edgesIn.push_back(e);
                edgesizestats[self->get_level()].addData(nbr->size * self->size);
            } else { // Pivot, and rest in symmetric case
                Edge* e = new Edge(self, nbr, A);
                self->edgesOut.push_back(e);
                if(self != nbr) {
                    nbr->edgesIn.push_back(e);
                }
            }            
        }
        nedgestats[self->get_level()].addData(nbrs.size());
        // Bring pivot in front
        self->edgesOut.sort([](Edge* a, Edge* b){return a->n2->order < b->n2->order;});
    }
    if(this->verb) {
        printf("Edge size statistics (Leaf-cluster edge size at each level of the ND hierarchy)\n");
        printf("Lvl     Count       Min       Max      Mean\n");
        for(int lvl = 0; lvl < nlevels; lvl++) {
            printf("%3d %9d %9d %9d %9.0f\n", lvl, edgesizestats[lvl].getCount(), edgesizestats[lvl].getMin(), edgesizestats[lvl].getMax(), edgesizestats[lvl].getMean());
        }
        printf("Edge count statistics (Leaf-cluster edge count at each level of the ND hierarchy)\n");
        printf("Lvl     Count       Min       Max      Mean\n");
        for(int lvl = 0; lvl < nlevels; lvl++) {
            printf("%3d %9d %9d %9d %9.0f\n", lvl, nedgestats[lvl].getCount(), nedgestats[lvl].getMin(), nedgestats[lvl].getMax(), nedgestats[lvl].getMean());
        }
    }
    timer tend = wctime();
    if(this->verb) printf("Assembly time : %3.2e s. (%3.2e permuting A)\n", elapsed(tstart, tend), elapsed(t0, t1));
}

/**
 * Factorize
 */

// Scale pivot to make them identity
// Optionally, scale the phi as well
int Tree::scale_cluster(Cluster* self) {  
    assert(this->ilvl <= self->get_level());
    // Factor pivot in-place
    MatrixXd* Lss = self->pivot()->APiv();
    VectorXi* p = nullptr;
    MatrixXd* U = nullptr;
    MatrixXd* VT = nullptr;
    VectorXd* Srsqrt = nullptr;
    double Ass_1_norm = Lss->cwiseAbs().colwise().sum().maxCoeff();
    this->log[this->ilvl].norm_diag.addData(Ass_1_norm);
    if(this->symmetry) {
        timer t_ = wctime();
        int info = potf(Lss); // Ass = L L^T
        timer t__ = wctime();
        this->tprof[this->ilvl].potf += elapsed(t_, t__);
        if(info != 0) {
            cout << "Not SPD!" << endl;
            return 1;
        }
        if(monitor_condition_pivots){
            double rcond = rcond_1_potf(Lss, Ass_1_norm);
            this->log[this->ilvl].cond_diag.addData(1.0/rcond);
        }
        this->ops.push_back(new Scaling(self, Lss));
    } else {
        if(this->scale_kind == ScalingKind::PLU) {            
            VectorXi swap(self->size);
            p = new VectorXi(self->size);
            timer t_ = wctime();
            int info = getf(Lss, &swap); // Ass = P L U
            timer t__ = wctime();
            this->tprof[this->ilvl].potf += elapsed(t_, t__);   
            if(info != 0) {
                cout << "Singular Pivot!" << endl;
                return 1;
            }
            if(monitor_condition_pivots){
                double rcond = rcond_1_getf(Lss, Ass_1_norm);
                this->log[this->ilvl].cond_diag.addData(1.0/rcond);
            }
            swap2perm(&swap, p);
            this->ops.push_back(new Scaling(self, Lss, p));
        } else if (this->scale_kind == ScalingKind::SVD) {
            U   = new MatrixXd(self->size, self->size);
            VT  = new MatrixXd(self->size, self->size);
            Srsqrt = new VectorXd(self->size);
            gesvd(Lss, U, Srsqrt, VT);
            if(monitor_condition_pivots){
                double cond = Srsqrt->maxCoeff() / Srsqrt->minCoeff();
                this->log[this->ilvl].cond_diag.addData(cond);
            }
            *Srsqrt = (Srsqrt->cwiseInverse()).cwiseSqrt();
            this->ops.push_back(new ScalingSVD(self, U, VT, Srsqrt));
            delete Lss;
        }
    }
    if(preserve) {
        trmm_trans(Lss, self->phi);
    }
    // Replace with identity
    MatrixXd* Iss = new MatrixXd();
    *Iss = MatrixXd::Identity(self->size, self->size);
    self->pivot()->set_APiv(Iss);
    // Factor panel in place after
    for(auto edge : self->edgesOutNbr()) {
        MatrixXd* Ans = edge->ALow();
        MatrixXd* Asn = edge->AUpp();
        timer t_ = wctime();
        if(this->symmetry) {
            trsm_right(Lss, Ans, CblasLower, CblasTrans, CblasNonUnit); // Ans Lss^-T                
        } else {
            if(this->scale_kind == ScalingKind::PLU) {  
                trsm_right(Lss, Ans, CblasUpper, CblasNoTrans, CblasNonUnit); // Ans Uss^-1                
                (*Asn) = p->asPermutation().transpose() * (*Asn);
                trsm_left( Lss, Asn, CblasLower, CblasNoTrans, CblasUnit); // Lss^-1 Pss^-1 Asn
            } else if (this->scale_kind == ScalingKind::SVD) {
                (*Asn) = Srsqrt->asDiagonal() * U->transpose() * (*Asn); // S^(-1/2) U^T Asn
                (*Ans) = (*Ans) * VT->transpose() * Srsqrt->asDiagonal(); // Ans V S^(-1/2)
            }
        }
        timer t__ = wctime();
        this->tprof[this->ilvl].trsm += elapsed(t_, t__);
    }
    // Factor panel in place before
    for(auto edge : self->edgesIn) {
        MatrixXd* Asn = edge->ALow();
        MatrixXd* Ans = edge->AUpp();
        timer t_ = wctime();
        if(this->symmetry) {
            trsm_left(Lss, Asn, CblasLower, CblasNoTrans, CblasNonUnit); // Lss^(-1) Asn
        } else {
            if(this->scale_kind == ScalingKind::PLU) {  
                trsm_right(Lss, Ans, CblasUpper, CblasNoTrans, CblasNonUnit); // Ans Uss^-1                
                (*Asn) = p->asPermutation().transpose() * (*Asn);
                trsm_left( Lss, Asn, CblasLower, CblasNoTrans, CblasUnit); // Lss^-1 Pss^-1 Asn
            } else if (this->scale_kind == ScalingKind::SVD) {
                (*Asn) = Srsqrt->asDiagonal() * U->transpose() * (*Asn); // S^(-1/2) U^T Asn
                (*Ans) = (*Ans) * VT->transpose() * Srsqrt->asDiagonal(); // Ans V S^(-1/2)
            }
        }
        timer t__ = wctime();
        this->tprof[this->ilvl].trsm += elapsed(t_, t__);
    }
    return 0;
}

int Tree::potf_cluster(Cluster* self){
    MatrixXd* Ass = self->pivot()->APiv();
    double Ass_1_norm = Ass->cwiseAbs().colwise().sum().maxCoeff();
    this->log[this->ilvl].norm_diag.addData(Ass_1_norm);
    VectorXi* p = nullptr;
    if( (this->ilvl <= skip) || (! ortho) ) {
        if(this->symmetry) {
            timer t_ = wctime();
            int info = potf(Ass); // Ass = L L^T
            timer t__ = wctime();
            this->tprof[this->ilvl].potf += elapsed(t_, t__); 
            if(info != 0) {
                cout << "Not SPD!" << endl;
                return 1;
            }
            if(monitor_condition_pivots){                
                double rcond = rcond_1_potf(Ass, Ass_1_norm);
                this->log[this->ilvl].cond_diag.addData(1.0/rcond);
            }
            this->ops.push_back(new Scaling(self, Ass));
        } else {            
            VectorXi swap(self->size);
            p = new VectorXi(self->size);
            timer t_ = wctime();
            int info = getf(Ass, &swap); // Ass = P L U       
            timer t__ = wctime();            
            this->tprof[this->ilvl].potf += elapsed(t_, t__); 
            if(info != 0) {
                cout << "Singular Pivot!" << endl;
                return 1;
            }
            if(monitor_condition_pivots){                
                double rcond = rcond_1_getf(Ass, Ass_1_norm);
                this->log[this->ilvl].cond_diag.addData(1.0/rcond);
            }
            swap2perm(&swap, p);
            this->ops.push_back(new Scaling(self, Ass, p));
            self->p = p;
        }        
    } else {
        delete Ass;
    }
    return 0;
}

void Tree::trsm_edge(Edge* edge){
    Cluster* self = edge->n1;
    Cluster* nbr  = edge->n2;
    MatrixXd* Ans = edge->ALow();
    MatrixXd* Asn = edge->AUpp();
    VectorXi* p   = self->p;
    MatrixXd* Ass = self->pivot()->APiv();
    if( (this->ilvl <= skip) || (! ortho) ) {
        timer t_ = wctime();
        if( this->symmetry ) {
            trsm_right(Ass, Ans, CblasLower, CblasTrans, CblasNonUnit); // Ans Lss^-T                
        } else {          
            trsm_right(Ass, Ans, CblasUpper, CblasNoTrans, CblasNonUnit); // Ans Uss^-1                
            (*Asn) = p->asPermutation().transpose() * (*Asn);
            trsm_left( Ass, Asn, CblasLower, CblasNoTrans, CblasUnit); // Lss^-1 Pss^-1 Asn
        }
        timer t__ = wctime();
        this->tprof[this->ilvl].trsm += elapsed(t_, t__);
    }
    if( this->symmetry ) {
        this->ops.push_back(new Gemm(self, nbr, Ans));
    } else {
        this->ops.push_back(new Gemm(self, nbr, Ans, Asn));
    }
}

void Tree::gemm_edges(Edge* edge1, Edge* edge2){
    assert(edge1->n1 == edge2->n1);    
    Cluster* c1     = edge1->n2;        
    MatrixXd* An1s  = edge1->ALow();
    MatrixXd* Asn1  = edge1->AUpp();
    Cluster* c2     = edge2->n2;
    MatrixXd* An2s  = edge2->ALow();
    MatrixXd* Asn2  = edge2->AUpp();    
    assert(c1->order <= c2->order);
    if( c1 == c2 ) { // Diagonal
        MatrixXd* An1n1 = c1->pivot()->APiv();
        timer t_ = wctime();
        if(this->symmetry) {
            syrk(An1s, An1n1); // Ann -= Ans Ans^T
        } else {
            gemm(An1s, Asn1, An1n1, CblasNoTrans, CblasNoTrans, -1.0, 1.0); // Ann -= Ans Asn
        }
        timer t__ = wctime();
        this->tprof[this->ilvl].gemm += elapsed(t_, t__);
    } else if (c1->order < c2->order) { // Off-diagonal             
        auto found = find_if(c1->edgesOut.begin(), c1->edgesOut.end(), [&c2](Edge* e){ return e->n2 == c2; } );
        if (found == c1->edgesOut.end()) { // Doesn't exist - fill in
            MatrixXd* An2n1 = new MatrixXd;
            MatrixXd* An1n2 = nullptr;
            *An2n1 = MatrixXd::Zero(c2->size, c1->size);
            if(! this->symmetry) {
                An1n2 = new MatrixXd;
                *An1n2 = MatrixXd::Zero(c1->size, c2->size);
            }
            timer t_ = wctime();
            if(this->symmetry) {
                gemm(An2s, An1s, An2n1, CblasNoTrans, CblasTrans, -1.0, 1.0); // An2n1 = - An2s An1s^T
            } else {
                gemm(An2s, Asn1, An2n1, CblasNoTrans, CblasNoTrans, -1.0, 1.0); // An2n1 = - An2s Asn1
                gemm(An1s, Asn2, An1n2, CblasNoTrans, CblasNoTrans, -1.0, 1.0); // An1n2 = - An1s Ans2
            }
            timer t__ = wctime();
            this->tprof[this->ilvl].gemm += elapsed(t_, t__);
            Edge* e = new Edge(c1, c2, An2n1, An1n2);
            c1->edgesOut.push_back(e);
            c2->edgesIn.push_back(e);
        } else { // Exists
            MatrixXd* An2n1 = (*found)->ALow();
            MatrixXd* An1n2 = (*found)->AUpp();
            timer t_ = wctime();
            if(this->symmetry) {
                gemm(An2s, An1s, An2n1, CblasNoTrans, CblasTrans, -1.0, 1.0); // An2n1 -= An2s An1s^T
            } else {
                gemm(An2s, Asn1, An2n1, CblasNoTrans, CblasNoTrans, -1.0, 1.0); // An2n1 -= An2s Asn1
                gemm(An1s, Asn2, An1n2, CblasNoTrans, CblasNoTrans, -1.0, 1.0); // An1n2 -= An1s Ans2
            }
            timer t__ = wctime();
            this->tprof[this->ilvl].gemm += elapsed(t_, t__);
        }
    }   
}

// Eliminate a cluster
int Tree::eliminate_cluster(Cluster* self){
    // Factor pivot in-place
    int error = potf_cluster(self);
    if(error > 0) return error;
    // Factor panel in place
    for(auto edge : self->edgesOutNbr()){
        trsm_edge(edge);
    }
    // Schur complement
    for(auto edge1 : self->edgesOutNbr()){ 
        for(auto edge2 : self->edgesOutNbr()){
            Cluster* c1 = edge1->n2;        
            Cluster* c2 = edge2->n2;
            if(c1->order <= c2->order) {
                gemm_edges(edge1, edge2);
            }
        }
    }
    // Update data structure
    assert(self->parent == nullptr);
    self->set_eliminated();
    update_eliminated_edges_and_delete(self);
    return 0;
}

void Tree::update_size(Cluster* snew) {
    assert(this->ilvl > 0); // No merge at lvl == 0
    assert(this->ilvl == current_bottom);
    // Update sizes & posparent
    int size = 0;
    for(auto sold : snew->children){
        sold->posparent = size;            
        size += sold->size;
    }
    snew->set_size(size);
    this->ops.push_back(new Merge(snew));
}

void Tree::update_edges(Cluster* snew) {
    assert(this->ilvl > 0);
    assert(this->ilvl == current_bottom);
    // Concatenate phis
    if(preserve) {
        assert(snew->children.size() > 0);
        auto kid = snew->children[0];
        snew->phi = new MatrixXd(snew->size, kid->phi->cols());
        int row = 0;
        for(auto sold : snew->children) {
            assert(sold->phi->rows() == sold->size);
            assert(sold->phi->cols() == snew->phi->cols());
            snew->phi->middleRows(row, sold->size) = *(sold->phi);
            row += sold->size;
            delete sold->phi;
        }  
        assert(row == snew->size);
    }
    // Figure out edges that gets together
    set<Cluster*> edges_merged;
    for(auto sold : snew->children){
        for(auto eold : sold->edgesOut){
            auto nold = eold->n2;
            auto nnew = nold->parent;
            edges_merged.insert(nnew);
        }
    }
    // Allocate memory, create new edges 
    for(auto nnew : edges_merged) {
        timer t0 = wctime();  
        MatrixXd* A = new MatrixXd(nnew->size, snew->size);   
        MatrixXd* A_T = nullptr;  
        A->setZero();
        if( (! this->symmetry) && (nnew != snew) ) {
            A_T = new MatrixXd(snew->size, nnew->size);
            A_T->setZero();
        }
        timer t1 = wctime();
        this->tprof[this->ilvl].mergealloc += elapsed(t0, t1);
        Edge* e = new Edge(snew, nnew, A, A_T);
        snew->edgesOut.push_back(e);
        if(snew != nnew) {
            nnew->edgesIn.push_back(e);
        }
    }
    snew->edgesOut.sort([](Edge* a, Edge* b){return a->n2->order < b->n2->order;});
    // Fill edges, delete previous edges
    for(auto sold : snew->children){
        for(auto eold : sold->edgesOut){
            auto nold = eold->n2;
            auto nnew = nold->parent;
            auto found = find_if(snew->edgesOut.begin(), snew->edgesOut.end(), [&nnew](Edge* e){return e->n2 == nnew;});
            assert(found != snew->edgesOut.end());                            
            timer t0 = wctime();
            /**  [x x] . [. x]
            *    [x x] . [x .]
            *     . .
            *    [. x]
            *    [x .]           **/
            (*found)->ALow()->block(nold->posparent, sold->posparent, nold->size, sold->size) = *eold->ALow(); // Diagonal and below
            if (! this->symmetry) {
                if ( snew == nnew && sold != nold ) { // New diagonal, old off-diagonal
                    (*found)->APiv()->block(sold->posparent, nold->posparent, sold->size, nold->size) = *eold->AUpp(); // On the diagonal pivot
                } else if (snew != nnew) { // Above diagonal
                    (*found)->AUpp()->block(sold->posparent, nold->posparent, sold->size, nold->size) = *eold->AUpp();
                }
            }
            timer t1 = wctime();
            this->tprof[this->ilvl].mergecopy += elapsed(t0, t1);
            delete eold->ALow();
            if (! this->symmetry) {
                delete eold->AUpp();
            }
            delete eold;                    
        }            
    }
}

// Get [Asn] for all n != s (symmetric and unsymmetric case)
// Order is always
// [before low, before up, after low, after up]
MatrixXd* Tree::assemble_Asn(Cluster* self) {
    timer t0 = wctime();
    int rows = self->size;
    int cols = 0;
    for(auto edge : self->edgesIn) {
        cols += edge->n1->size;
    }    
    for(auto edge : self->edgesOutNbr()) {
        cols += edge->n2->size;
    }
    if(! this->symmetry) cols *= 2;
    MatrixXd* Asn = new MatrixXd(rows, cols);
    int c = 0;
    for(auto edge : self->edgesIn) {
        int cols = edge->n1->size;
        Asn->middleCols(c, cols) = *edge->ALow();
        c += cols;
        if(! this->symmetry) {
            Asn->middleCols(c, cols) = edge->AUpp()->transpose();
            c += cols;
        }
    }
    for(auto edge : self->edgesOutNbr()) {
        int cols = edge->n2->size;
        Asn->middleCols(c, cols) = edge->ALow()->transpose();
        c += cols;
        if(! this->symmetry) {
            Asn->middleCols(c, cols) = *edge->AUpp();
            c += cols;
        }
    }
    assert(c == cols);
    timer t1 = wctime();
    this->tprof[this->ilvl].assmb += elapsed(t0, t1);
    return Asn;
}

// Get [Asn phi, phi]
MatrixXd* Tree::assemble_Asphi(Cluster* self) {
    assert(this->symmetry);
    int rows = self->size;
    // How many neighbors ?
    int nphis = this->nphis;
    int nnbr = self->nnbr_in_self_out();
    int cols = nphis * nnbr;
    // Build matrix [phis, Asn*phin] into Q1
    MatrixXd* Asnp = new MatrixXd(rows, cols);
    int c = 0;
    for(auto edge : self->edgesIn) {
        Asnp->middleCols(c, nphis) = (*edge->ALow()) * (*(edge->n1->phi));
        c += nphis;
    }
    for(auto edge : self->edgesOutNbr()) {
        Asnp->middleCols(c, nphis) = edge->ALow()->transpose() * (*(edge->n2->phi));
        c += nphis;
    }
    Asnp->middleCols(c, nphis) = (*self->phi);
    c += nphis;
    assert(c == cols);
    return Asnp;
}

// Asn -> Q^T Asn
// Ans -> Ans Q
void Tree::scatter_Q(Cluster* self, MatrixXd* Q) {
    timer t0 = wctime();
    int rank = Q->cols();
    assert(self->size == rank);
    for(auto edge : self->edgesIn) {
        MatrixXd* Asn = edge->ALow();
        edge->set_ALow(gemm_new(Q, Asn, CblasTrans, CblasNoTrans, 1.0));
        delete Asn;        
        if(! this->symmetry) {
            MatrixXd* Ans = edge->AUpp();
            edge->set_AUpp(gemm_new(Ans, Q, CblasNoTrans, CblasNoTrans, 1.0));
            delete Ans;
        }
    }
    for(auto edge : self->edgesOutNbr()) {
        MatrixXd* Ans = edge->ALow();
        edge->set_ALow(gemm_new(Ans, Q, CblasNoTrans, CblasNoTrans, 1.0));
        delete Ans;
        if(! this->symmetry) {
            MatrixXd* Asn = edge->AUpp();
            edge->set_AUpp(gemm_new(Q, Asn, CblasTrans, CblasNoTrans, 1.0));
            delete Asn;
        }
    }
    timer t1 = wctime();
    this->tprof[this->ilvl].scattq += elapsed(t0, t1);
}

void Tree::scatter_Asn(Cluster* self, MatrixXd* Asn) {
    timer t0 = wctime();
    int rank = Asn->rows();
    assert(self->size == rank);
    int cols = Asn->cols();
    int c = 0;
    for(auto edge : self->edgesIn) {
        int cols = edge->n1->size;
        (*edge->ALow()) = Asn->middleCols(c, cols);
        c += cols;
        if(! this->symmetry) {
            (*edge->AUpp()) = Asn->middleCols(c, cols).transpose();
            c += cols;
        }
    }
    for(auto edge : self->edgesOutNbr()) {
        int cols = edge->n2->size;
        (*(edge->ALow())) = Asn->middleCols(c, cols).transpose();
        c += cols;
        if(! this->symmetry) {
            (*edge->AUpp()) = Asn->middleCols(c, cols);
            c += cols;
        }
    }
    assert(c == cols);
    timer t1 = wctime();
    this->tprof[this->ilvl].scatta += elapsed(t0, t1);
}

// Preserve only
void Tree::sparsify_preserve_only(Cluster* self) {
    assert(this->symmetry);
    int rows = self->size;
    int nphis = this->nphis;
    int nnbr = self->nnbr_in_self_out();
    int cols = nphis * nnbr;
    if (cols >= rows) return; // No point to move forward
    // Get edge
    MatrixXd* Asnp = this->assemble_Asphi(self); // new
    assert(Asnp->cols() == cols);
    assert(Asnp->rows() == rows);
    // Orthogonalize
    int rank = min(rows, cols);
    VectorXd* h = new VectorXd(rank); // new
    timer tgeqrf_0 = wctime();
    geqrf(Asnp, h);
    timer tgeqrf_1 = wctime();
    this->tprof[this->ilvl].geqrf += elapsed(tgeqrf_0, tgeqrf_1);
    Asnp->conservativeResize(rows, rank);
    MatrixXd* v = Asnp;
    // Build Q
    MatrixXd Q = *Asnp;
    orgqr(&Q, h);
    // Record
    this->ops.push_back(new Orthogonal(self, v, h));
    // Update self
    self->size = rank;
    (*(self->pivot()->APiv())) = MatrixXd::Identity(rank, rank);
    *self->phi = Q.transpose() * (*self->phi);
    // Update edges
    this->scatter_Q(self, &Q);  
}

// RRQR only
void Tree::sparsify_adaptive_only(Cluster* self) {
    // Asn = [Asn_1 Asn_2 ... Asn_k]
    MatrixXd* Asn = this->assemble_Asn(self);
    int rows = Asn->rows();
    int cols = Asn->cols();
    VectorXi jpvt = VectorXi::Zero(cols);
    VectorXd ht   = VectorXd(min(rows, cols));
    // GEQP3
    timer tgeqp3_0 = wctime();
    geqp3(Asn, &jpvt, &ht);
    timer tgeqp3_1 = wctime();
    this->tprof[this->ilvl].geqp3 += elapsed(tgeqp3_0, tgeqp3_1);
    // Truncate ?
    VectorXd diag = Asn->diagonal();
    int rank = choose_rank(diag, tol);
    if (rank >= rows) { // No point, nothing to do
        delete Asn;
        return;
    }    
    // Save Q
    timer tQ_0 = wctime();
    MatrixXd* v = new MatrixXd(rows, rank); // new
    *v = Asn->leftCols(rank);
    VectorXd* h = new VectorXd(rank); // new
    *h = ht.topRows(rank);
    timer tQ_1 = wctime();
    this->tprof[this->ilvl].buildq += elapsed(tQ_0, tQ_1);
    // Record
    this->ops.push_back(new Orthogonal(self, v, h));
    // Update self
    self->size = rank;
    (*(self->pivot()->APiv())) = MatrixXd::Identity(rank, rank);
    // Update edges - adjust
    timer tS_0 = wctime();
    MatrixXd AsnP = Asn->topRows(rank).triangularView<Upper>();
    AsnP = AsnP * (jpvt.asPermutation().transpose());
    timer tS_1 = wctime();
    this->tprof[this->ilvl].perma += elapsed(tS_0, tS_1);
    assert(AsnP.rows() == rank);
    this->scatter_Asn(self, &AsnP);
    // Forget about Asn
    delete Asn;
}

// Preserve + RRQR
void Tree::sparsify_preserve_adaptive(Cluster* self) {
    assert(this->symmetry);
    int rows = self->size;
    int nphis = this->nphis;
    int nnbr = self->nnbr_in_self_out();
    int cols1 = nphis * nnbr;
    if (cols1 >= rows) return;
    // (1) Get edge
    MatrixXd* Asnphi = this->assemble_Asphi(self); // [Asn phi, phi] - new
    assert(Asnphi->rows() == rows);
    assert(Asnphi->cols() == cols1);
    // QR
    int rank1 = min(rows, cols1);
    VectorXd h1 = VectorXd(rank1);
    timer tgeqrf_0 = wctime();
    geqrf(Asnphi, &h1);
    timer tgeqrf_1 = wctime();
    this->tprof[this->ilvl].geqrf += elapsed(tgeqrf_0, tgeqrf_1);
    // Build Q1
    Asnphi->conservativeResize(rows, rank1);
    MatrixXd* Q1 = Asnphi;
    orgqr(Q1, &h1);
    // (2) Get edge
    MatrixXd* Asn = this->assemble_Asn(self); // Asn - new
    int cols2 = Asn->cols();
    VectorXi jpvt = VectorXi::Zero(cols2);
    VectorXd h2   = VectorXd(min(rows, cols2));
    // Remove Q1
    (*Asn) -= (*Q1) * (Q1->transpose() * (*Asn));
    // GEQP3
    timer tgeqp3_0 = wctime();
    geqp3(Asn, &jpvt, &h2);
    timer tgeqp3_1 = wctime();
    this->tprof[this->ilvl].geqp3 += elapsed(tgeqp3_0, tgeqp3_1);
    // Truncate ?
    VectorXd diag = Asn->diagonal();
    int rank2 = choose_rank(diag, tol);
    int rank = rank1 + rank2;
    if(rank >= rows) {
        delete Asnphi;
        delete Asn;
        return;
    }
    // Build Q2
    Asn->conservativeResize(rows, rank2);
    h2.conservativeResize(rank2);
    orgqr(Asn, &h2);
    MatrixXd* Q2 = Asn;
    // Concatenate [Q1, Q2] & orthogonalize
    MatrixXd* v = new MatrixXd(rows, rank); // new
    *v << *Q1, *Q2;
    VectorXd* h = new VectorXd(rank); // new
    timer tgeqrf_2 = wctime();
    geqrf(v, h);
    timer tgeqrf_3 = wctime();
    this->tprof[this->ilvl].geqrf += elapsed(tgeqrf_2, tgeqrf_3);
    MatrixXd Q = *v;
    orgqr(&Q, h);
    // Scatter
    this->ops.push_back(new Orthogonal(self, v, h));
    // Update self
    self->size = rank;
    (*(self->pivot()->APiv())) = MatrixXd::Identity(rank, rank);
    *self->phi = Q.transpose() * (*self->phi);
    // Update edges
    this->scatter_Q(self, &Q);
    // Delete
    delete Asnphi;
    delete Asn;
}

void Tree::drop_all(Cluster* self) {
    int rows = self->size;
    int rank = 0;
    // rank-0 orthogonal trans
    MatrixXd Q = MatrixXd(rows, rank);
    // Update self
    self->size = rank;
    (*(self->pivot()->APiv())) = MatrixXd::Identity(rank, rank);
    // Update edges
    this->scatter_Q(self, &Q);
}

int Tree::sparsify_cluster(Cluster* self) {
    if(want_sparsify(self)) {
        this->log[this->ilvl].rank_before.addData(self->size);
        if (ortho) {
            if (preserve) {
                if (adaptive) {
                    this->sparsify_preserve_adaptive(self);
                } else {
                    this->sparsify_preserve_only(self);
                }
            } else {
                if (adaptive) {
                    this->sparsify_adaptive_only(self);                            
                } else {
                    this->drop_all(self);
                }
            }
        } else {
            int error = this->sparsify_interp(self);
            if(error != 0) return error;
        }
        this->log[this->ilvl].rank_after.addData(self->size);
    }
    return 0;
}

// Interpolative
int Tree::sparsify_interp(Cluster* self) {
    // Get edge
    MatrixXd* Asn = this->assemble_Asn(self);
    // Transpose
    MatrixXd Ans = Asn->transpose();
    delete Asn;
    MatrixXd Ans_ = Ans;
    int cols = Ans.cols(); // s
    int rows = Ans.rows(); // n
    int rank = 0;
    VectorXi* perm = new VectorXi(cols);
    // Interpolative Decomp
    if (rows == 0) { // Sparsify everything
        (*perm) = VectorXi::LinSpaced(cols, 0, cols-1);
        rank = 0;
    } else {
        (*perm) = VectorXi::Zero(cols);
        VectorXd h = VectorXd(min(rows, cols));
        // geqp3
        timer tgeqp3_0 = wctime();
        geqp3(&Ans, perm, &h);
        timer tgeqp3_1 = wctime();
        this->tprof[this->ilvl].geqp3 += elapsed(tgeqp3_0, tgeqp3_1);
        // Truncate
        VectorXd diag = Ans.diagonal();
        rank = choose_rank(diag, tol);
    }
    int discard = cols - rank;
    // Record local permutation
    this->ops.push_back(new Permutation(self, perm));
    // Build f, c, Tfc
    MatrixXd R = Ans.triangularView<Upper>();
    auto R11 = Ans.block(0, 0, rank, rank).triangularView<Upper>();
    MatrixXd R12 = Ans.block(0, rank, rank, discard);
    MatrixXd *Tcf = new MatrixXd(rank, discard);
    if(rank > 0)
       *Tcf = R11.solve(R12);
    // Record interpolation
    this->ops.push_back(new Interpolation(self, Tcf));
    // Build Acc, Ccf, Cfc, Cff
    auto P = perm->asPermutation();
    MatrixXd Ass;
    if(this->symmetry) {
        Ass = self->pivot()->APiv()->selfadjointView<Lower>();
    } else {
        Ass = *self->pivot()->APiv();
    }
    Ass = P.transpose() * Ass * P;
    MatrixXd Acc = Ass.block(0, 0, rank, rank);
    MatrixXd Afc = Ass.block(rank, 0, discard, rank);
    MatrixXd Aff = Ass.block(rank, rank, discard, discard);
    MatrixXd Acf = Ass.block(0, rank, rank, discard);
    MatrixXd *Cff = new MatrixXd();
    MatrixXd *Ccf = new MatrixXd();    
    (*Cff) = Aff - Afc * (*Tcf) - Tcf->transpose() * Acf + (Tcf->transpose()) * Acc * (*Tcf);
    (*Ccf) = Acf - Acc * (*Tcf);    
    // Eliminate f (|f| = discard)
    if(this->symmetry) {
        // Pivot
        int info = potf(Cff);
        if(info != 0) {
            cout << "Not SPD!" << endl;
            return 1;
        }
        // Panel
        trsm_right(Cff, Ccf, CblasLower, CblasTrans, CblasNonUnit);        
        this->ops.push_back(new SelfElim(self, Cff, Ccf));
        // Schur Complement
        (*self->pivot()->APiv()) = Acc - (*Ccf) * Ccf->transpose();
    } else {
        MatrixXd *Cfc = new MatrixXd();
        (*Cfc) = Afc - Tcf->transpose() * Acc;
        // Pivot
        VectorXi swap(discard);
        VectorXi* lup = new VectorXi(discard);
        int info = getf(Cff, &swap);
        if(info != 0) {
            cout << "Singular Pivot!" << endl;
            return 1;
        }
        swap2perm(&swap, lup);
        this->ops.push_back(new SelfElim(self, Cff, Ccf, Cfc, lup));
        // Panel
        trsm_right(Cff, Ccf, CblasUpper, CblasNoTrans, CblasNonUnit);
        (*Cfc) = lup->asPermutation().transpose() * (*Cfc);
        trsm_left( Cff, Cfc, CblasLower, CblasNoTrans, CblasUnit);
        // Schur Complement
        (*self->pivot()->APiv()) = Acc - (*Ccf) * (*Cfc);
    }
    // Update self
    self->size = rank;
    // Update edges (except self)
    for(auto edge : self->edgesIn) {
        (*edge->ALow()) = ( P.transpose() * (*edge->ALow()) ).topRows(rank); // (P^T Asn)[1:rank,:]
        if(! this->symmetry) {
            (*edge->AUpp()) = ( (*edge->AUpp()) * P ).leftCols(rank); // (Ans P)[:,1:rank]
        }
    }
    for(auto edge : self->edgesOutNbr()) {
        (*edge->ALow()) = ( (*edge->ALow()) * P ).leftCols(rank); // (Ans P)[:,1:rank]
        if(! this->symmetry) {
            (*edge->AUpp()) = ( P.transpose() * (*edge->AUpp()) ).topRows(rank); // (P^T Asn)[1:rank,:]
        }
    }
    return 0;
}

void Tree::update_eliminated_edges_and_delete(Cluster* self){
    assert(self->get_level() == this->ilvl);
    assert(self->edgesIn.size() == 0);
    delete self->phi;
    for(auto e : self->edgesOut){
        e->n2->edgesIn.remove_if([self](Edge* e){return e->n1 == self;});
        delete e;
    }
}

int Tree::factorize() {

    timer tstart = wctime();
    if(preserve) {
        assert(phi != nullptr);
        assert(phi->rows() == N);
        assert(ortho && symmetry);
        this->nphis = phi->cols();
    }
    if(ortho) {
        assert(scale);
    }
    this->adaptive = (tol <= 1.0);
    if(this->verb) {
        cout << "Factorization started" << endl;
        cout << "  N:          " << N        << endl;
        cout << "  #levels:    " << nlevels  << endl;
        cout << "  verbose?:   " << verb     << endl;
        cout << "  symmetry?:  " << symmetry << endl;
        cout << "  adaptive?:  " << adaptive << endl;
        cout << "  tol?:       " << tol      << endl;
        cout << "  #skip:      " << skip     << endl;
        cout << "  scale?:     " << scale    << endl;
        cout << "  ortho?:     " << ortho    << endl;
        cout << "  scalingkd?  " << (symmetry ? "LLT" : (scale_kind == ScalingKind::SVD ? "SVD" : "PLU"))<< endl;    
        cout << "  want_spars? " << use_want_sparsify << endl;
        cout << "  mon cond?   " << monitor_condition_pivots << endl;
        cout << "  preserving? " << preserve << endl;
        if(preserve)
            cout << "  preserving " << this->nphis << " vectors" << endl;
    }

    // Create the phi
    if(preserve) {
        MatrixXd phi_ = this->perm.asPermutation().transpose() * (*phi);        
        // Store phi
        for(auto self : bottom_original()) {
            self->phi = new MatrixXd(self->size, phi_.cols());
            (*self->phi) = phi_.middleRows(self->start, self->size);
        }    
    }

    // Factorization
    for(this->ilvl = 0; this->ilvl < nlevels; this->ilvl++) {
        
        if(this->verb) printf("Level %d, %d dofs left, %d clusters left\n", this->ilvl, this->ndofs_left(), this->nclusters_left());

        // Eliminate
        {
            // for(auto self : this->bottom_current()) {
            //     if(self->get_level() == this->ilvl) {
            //         auto Ass = self->pivot()->APiv();
            //         auto U = new MatrixXd(self->size, self->size);
            //         auto VT  = new MatrixXd(self->size, self->size);
            //         auto S = new VectorXd(self->size);
            //         gesvd(Ass, U, S, VT);
            //         cout << "----" << endl;
            //         cout << *S << endl;
            //     }
            // }
            // exit(0);
            timer telim_0 = wctime();        
            for(auto self : this->bottom_current()) {
                if(self->get_level() == this->ilvl) {
                    int error = this->eliminate_cluster(self);
                    if(error != 0) return error;
                }
            }
            timer telim_1 = wctime();
            this->tprof[this->ilvl].elim += elapsed(telim_0, telim_1);
            this->log[this->ilvl].dofs_left_elim = this->ndofs_left();
            if(this->verb) printf("  Elim: %3.2e s., %d dofs left, %d clusters left\n", elapsed(telim_0, telim_1), this->log[this->ilvl].dofs_left_elim, this->nclusters_left());
        }

        // Merge
        if(this->ilvl > 0){
            timer tmerge_0 = wctime();        
            current_bottom++;
            for(auto self : this->bottom_current()) {
                this->update_size(self);
            }
            for(auto self : this->bottom_current()) {
                this->update_edges(self);
            }
            timer tmerge_1 = wctime();
            this->tprof[this->ilvl].merge += elapsed(tmerge_0, tmerge_1);
            if(this->verb) printf("  Merge: %3.2e s., %d dofs left, %d clusters left\n", elapsed(tmerge_0, tmerge_1), this->ndofs_left(), this->nclusters_left());
        }

        // Scale
        if (this->ilvl >= skip) {
            if (scale) {
                timer tscale_0 = wctime();
                for(auto self : this->bottom_current()) {
                    if(self->get_level() > this->ilvl) {
                        int error = this->scale_cluster(self);
                        if(error != 0) return error;
                    }
                }
                timer tscale_1 = wctime();
                this->tprof[this->ilvl].scale += elapsed(tscale_0, tscale_1);
                if(this->verb) printf("  Scaling: %3.2e s.\n", elapsed(tscale_0, tscale_1));
            }
        } // Scale ... if skip

        // Sparsify
        if (this->ilvl >= skip) {
            timer tspars_0 = wctime();
            for(auto self : this->bottom_current()) {
                if(self->get_level() > this->ilvl) {
                    int error = sparsify_cluster(self);
                    if(error != 0) return error;
                }
            }
            timer tsparse_1 = wctime();
            this->tprof[this->ilvl].spars += elapsed(tspars_0, tsparse_1);
            if(this->verb) printf("  Sparsification: %3.2e s., %d dofs left, geqp3 %3.2e, geqrf %3.2e, assmb %3.2e, buildQ %3.2e, scatterQ %3.2e, permA %3.2e, scatterA %3.2e\n", 
                    elapsed(tspars_0, tsparse_1), this->ndofs_left(), this->tprof[this->ilvl].geqp3, this->tprof[this->ilvl].geqrf, 
                    this->tprof[this->ilvl].assmb, this->tprof[this->ilvl].buildq, this->tprof[this->ilvl].scattq, this->tprof[this->ilvl].perma, this->tprof[this->ilvl].scatta);
        } // Sparsify ... if skip

        this->log[this->ilvl].dofs_left_spars = this->ndofs_left();
        this->log[this->ilvl].fact_nnz = this->nnz();

    } // TOP LEVEL for(int this->ilvl = 0 ...)
    timer tend = wctime();
    if(this->verb) printf("Factorization: %3.2e s.\n", elapsed(tstart, tend));
    return 0;
}

void Tree::solve(VectorXd& x) const {
    // Permute
    VectorXd b = perm.asPermutation().transpose() * x;
    // Set solution
    for(auto cluster : bottom_original()) {
        cluster->set_vector(b);
    }
    // Fwd
    for(auto io = ops.begin(); io != ops.end(); io++) {
        (*io)->fwd();
    }
    // Bwd
    for(auto io = ops.rbegin(); io != ops.rend(); io++) {
        (*io)->bwd();
    }
    // Extract solution
    for(auto cluster : bottom_original()) {
        cluster->extract_vector(b);
    }
    // Permute back
    x = perm.asPermutation() * b;
}

Segment Cluster::head() {
    return this->x->segment(0, this->size);
}

long long Tree::nnz() {
    long long nnz = 0;
    for(auto op: ops) {
        nnz += op->nnz();
    }
    return nnz;
}

void Tree::print_log() const {
    // Timings
    printf("&&&& Lvl |         Elim        Scale     Sparsify        Merge |     Preserve        geqp3        geqrf         potf         trsm         gemm       buildq       scattq        assmb       scatta          phi   mergealloc    mergecopy\n");
    for(int lvl = 0; lvl < this->nlevels; lvl++) {
        printf("&&&& %3d | %e %e %e %e | %e %e %e %e %e %e %e %e %e %e %e %e %e\n", 
            lvl,
            this->tprof[lvl].elim,
            this->tprof[lvl].scale,
            this->tprof[lvl].spars,
            this->tprof[lvl].merge,
            this->tprof[lvl].prese,
            this->tprof[lvl].geqp3,
            this->tprof[lvl].geqrf,
            this->tprof[lvl].potf,
            this->tprof[lvl].trsm,
            this->tprof[lvl].gemm,
            this->tprof[lvl].buildq,
            this->tprof[lvl].scattq,
            this->tprof[lvl].assmb,
            this->tprof[lvl].scatta,
            this->tprof[lvl].phi,
            this->tprof[lvl].mergealloc,
            this->tprof[lvl].mergecopy
            );
    }
    // Sizes and ranks
    printf("++++ Lvl        ND    ND lft    El lft    Sp lft   Fct nnz    Rk Bfr    Rk Aft   Tot Bfr   Tot Aft   Cl Sped  CondDiag   NormDiag\n");
    for(int lvl = 0; lvl < this->nlevels; lvl++) {
        printf("++++ %3d %9d %9d %9d %9d %9lld %9.0f %9.0f %9d %9d %9d   %4.1e    %4.1e\n", 
            lvl,
            this->log[lvl].dofs_nd, 
            this->log[lvl].dofs_left_nd, 
            this->log[lvl].dofs_left_elim, 
            this->log[lvl].dofs_left_spars, 
            this->log[lvl].fact_nnz,
            this->log[lvl].rank_before.getMean(),
            this->log[lvl].rank_after.getMean(),
            this->log[lvl].rank_before.getSum(),
            this->log[lvl].rank_after.getSum(),
            this->log[lvl].rank_after.getCount(),
            this->log[lvl].cond_diag.getMax(),
            this->log[lvl].norm_diag.getMean()
            );
    }
}

/** Return the current trailing matrix **/
SpMat Tree::get_assembly_mat() const {
    assert(this->ilvl == 0); // Not well defined exept at the leaves
    // Build matrix at current stage
    vector<Triplet<double>> values(0);
    for(auto s : bottom_original()) {            
        int s1 = s->start;
        assert(s1 >= 0);
        int s2 = s->size;
        for(auto e : s->edgesOut) {
            assert(e->n1 == s);
            int n1 = e->n2->start;
            assert(n1 >= 0);
            int n2 = e->n2->size;
            // Neighbors are stored as A[n,s] in the lower part
            for(int i_ = 0; i_ < n2; i_++) {
                for(int j_ = 0; j_ < s2; j_++) {
                    int i = n1 + i_;
                    int j = s1 + j_;
                    if(this->symmetry) {
                        if(i > j) {
                            double v = (*e->ALow())(i_,j_);
                            values.push_back(Triplet<double>(i,j,v));
                            values.push_back(Triplet<double>(j,i,v));
                        } else if (i == j) {
                            assert(s == e->n2);
                            double v = (*e->APiv())(i_,j_);
                            values.push_back(Triplet<double>(i,j,v));
                        }
                    } else {
                        if(i > j && e->n2 == s) { // Pivot
                            double v1 = (*e->APiv())(i_,j_);
                            values.push_back(Triplet<double>(i,j,v1));
                            double v2 = (*e->APiv())(j_,i_);
                            values.push_back(Triplet<double>(j,i,v2));
                        } else if (i == j && e->n2 == s) { // Pivot
                            double v = (*e->APiv())(i_,j_);
                            values.push_back(Triplet<double>(i,j,v));
                        } else if (i > j) { // Below
                            double v1 = (*e->ALow())(i_,j_);
                            values.push_back(Triplet<double>(i,j,v1));
                            double v2 = (*e->AUpp())(j_,i_);
                            values.push_back(Triplet<double>(j,i,v2));
                        }
                    }
                }
            }                
        }
    }
    int N = this->N;
    SpMat A(N,N);
    A.setFromTriplets(values.begin(), values.end());   
    return A;
}

Tree::~Tree() {
    for(auto o : this->ops) {
        delete o;
    }
    for(int lvl = 0; lvl < this->ilvl; lvl++) {
        for(auto s : bottoms[lvl]){
            delete s->x;
            delete s;
        }
    }
    for(int lvl = this->ilvl; lvl < this->nlevels; lvl++) {
        for(auto s : bottoms[lvl]){
            delete s->phi;
            delete s->x;
            for(auto e : s->edgesOut) {
                delete e;
            }
            delete s;
        }
    }
}
