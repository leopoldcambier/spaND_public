#include "spaND.h"

using namespace std;
using namespace Eigen;

namespace spaND {

SepID merge(SepID& s) {
    return SepID(s.lvl + 1, s.sep / 2);
}

ClusterID merge_if(ClusterID& c, int lvl) {
    auto left  = c.l.lvl < lvl ? merge(c.l) : c.l;
    auto right = c.r.lvl < lvl ? merge(c.r) : c.r;
    return ClusterID(c.self, left, right);
}

/** Returns the CSC undirected-graph structure of the symmetric sparse matrix A. Edges (u,v) and (v,u) are both present for u != v
 *  There are no self loops
 */
tuple<vector<int>,vector<int>> SpMat2CSC(SpMat& A) {
    int size = A.rows();
    assert(A.cols() == A.rows());
    vector<int> colptr(size+1);
    vector<int> rowval;
    colptr[0] = 0;
    for(int i = 0; i < size; i++) {
        colptr[i+1] = colptr[i];
        for (SpMat::InnerIterator it(A,i); it; ++it) {
            assert(i == it.col());
            int j = it.row();
            if(i != j) {
                rowval.push_back(j);
                colptr[i + 1] += 1;
            }
        }
    }
    return make_tuple(colptr, rowval);
}

std::ostream& operator<<(std::ostream& os, const SepID& s) {
    os << "(" << s.lvl << " " << s.sep << ")";
    return os;
}

std::ostream& operator<<(std::ostream& os, const ClusterID& c) {
    os << "(" << c.self << ":" << c.l << ";" << c.r << ")";
    return os;
}

void partition_metis(vector<int> &colptr, vector<int> &rowval, vector<int> &colptrtmp, vector<int> &rowvaltmp, vector<int> &dofs, vector<int> &parts, bool useVertexSep) {
    // Metis options
    assert(sizeof(idx_t) == sizeof(int));
    int options[METIS_NOPTIONS];
    assert(METIS_OK == METIS_SetDefaultOptions(options));
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
            auto found = lower_bound(dofs.begin(), dofs.end(), k_); // Is k a neighbor of i ?
            int pos = distance(dofs.begin(), found);
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

void bissect_geo(vector<int> &colptr, vector<int> &rowval, vector<int> &dofs, vector<int> &parts, MatrixXd *X, int start) {
    assert(X->cols() == colptr.size()-1);
    assert(X->cols() == parts.size());
    int N = dofs.size();
    if(N == 0)
        return;
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
    vector<int> dofstmp = dofs;
    // Sort dofs based on X[dim,:]
    std::sort(dofstmp.begin(), dofstmp.end(), [&](int i, int j) { return (*X)(bestdim,i) < (*X)(bestdim,j); });
    int imid = dofstmp[N/2];
    double midv = (*X)(bestdim,imid);
    // X <= midv -> 0, X > midv -> 1
    for(int i = 0; i < N; i++) {
        int j = dofs[i];
        if( (*X)(bestdim,j) <= midv ) {
            parts[j] = start;
        } else {
            parts[j] = start+1;
        }
    }
}

void partition_nd_geo(vector<int> &colptr, vector<int> &rowval, vector<int> &dofs, vector<int> &parts, MatrixXd *X, vector<int> &invp) {
    assert(X->cols() == colptr.size()-1);
    assert(invp.size() >= X->cols());
    assert(X->cols() == parts.size());
    int N = dofs.size();
    if(N == 0)
        return;
    vector<int> dofs_tmp = dofs;
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
    std::sort(parts.begin(), parts.begin() + N, [&](int i, int j) { return (*X)(bestdim,i) < (*X)(bestdim,j); });
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

int part_at_lvl(int part_leaf, int lvl, int nlevels) {
    assert(lvl > 0);
    for(int l = 0; l < lvl-1; l++) {
        part_leaf /= 2;
    }
    assert(part_leaf >= 0 && part_leaf < pow(2, nlevels-lvl));
    return part_leaf;
}

SepID find_highest_common(SepID n1, SepID n2) {
    int lvl1 = n1.lvl;
    int lvl2 = n2.lvl;
    int sep1 = n1.sep;
    int sep2 = n2.sep;
    while (lvl1 < lvl2) {
        lvl1 ++;
        sep1 /= 2;
    }
    while (lvl2 < lvl1) {
        lvl2 ++;
        sep2 /= 2;
    }
    while (sep1 != sep2) {
        lvl1 ++;
        lvl2 ++;
        sep1 /= 2;
        sep2 /= 2;
    }
    assert(lvl1 == lvl2);
    assert(sep1 == sep2);
    return SepID(lvl1, sep1);
}

vector<int> partition_RB(SpMat &A, int nlevels, bool verb, MatrixXd* Xcoo) {
    bool geo = (Xcoo != nullptr);
    if(verb && geo) printf("Geometric RB partitioning & ordering\n");
    if(verb && !geo) printf("Algebraic RB partitioning & ordering\n");
    int size = A.rows();
    auto csc = SpMat2CSC(A);
    vector<int> colptr = get<0>(csc);
    vector<int> rowval = get<1>(csc);
    vector<int> parts(size);
    if(geo) {
        fill(parts.begin(), parts.end(), 0);
        for(int depth = 0; depth < nlevels-1; depth++) {
            // Create dofs lists
            vector<vector<int>> dofs(pow(2, depth));
            for(int i = 0; i < parts.size(); i++) {
                assert(0 <= parts[i] && parts[i] < pow(2, depth));
                dofs[parts[i]].push_back(i);
            }
            // Do a ND partition
            for(int k = 0; k < pow(2, depth); k++) {
                bissect_geo(colptr, rowval, dofs[k], parts, Xcoo, 2*k);
            }
        }
    } else {
        // Metis recursive bissection        
        int one = 1;
        int objval;
        int nparts = pow(2, nlevels-1);
        if(nparts == 1) {
            // Meties doesn't like nparts == 1 and memcheck finds some errors
            parts = vector<int>(size, 0);
        } else {
            int options[METIS_NOPTIONS];
            options[METIS_OPTION_SEED] = 7103855;        
            assert(METIS_OK == METIS_SetDefaultOptions(options));
            assert(METIS_OK == METIS_PartGraphRecursive(&size, &one, colptr.data(), rowval.data(),
                                                        nullptr, nullptr, nullptr, &nparts, nullptr, nullptr, options, &objval, parts.data()));
        }        
    }
    return parts;
}

/** A should be a symmetric matrix **/
vector<ClusterID> partition_recursivebissect(SpMat &A, int nlevels, bool verb, MatrixXd* Xcoo) {
    timer t0 = wctime();
    int size = A.rows();
    vector<int> parts = partition_RB(A, nlevels, verb, Xcoo);
    timer t1 = wctime();
    if(verb) printf("Recursive bisection time: %3.2e s.\n", elapsed(t0, t1));
    /**
     * ND root         20
     *             10      11
     * NF leaves 00  01  02  03
     * part       1   1   2   3
     */
    vector<ClusterID> clusters(size, ClusterID(SepID(-1,0),SepID(-1,0),SepID(-1,0)));
    // Find separators
    for(int depth = 0; depth < nlevels-1; depth++) {
        int l = nlevels - depth - 1;
        vector<int> sepsizes(pow(2, depth), 0);
        for(int i = 0; i < size; i++) {
            if(clusters[i].self.lvl == -1) { // Not yet a separator
                bool sep = false;
                int pi = part_at_lvl(parts[i], l, nlevels);
                if(pi % 2 == 0) { // Separators are always on the left partition
                    for (SpMat::InnerIterator it(A,i); it; ++it) {
                        assert(i == it.col());
                        int j = it.row();
                        if(clusters[j].self.lvl == -1) { // Neighbor should also not be a separator
                            int pj = part_at_lvl(parts[j], l, nlevels);
                            if(pj == pi+1) {
                                sep = true;
                                break;
                            }
                        }
                    }
                    if(sep) { // Ok, it has a non-sep neighbor with lvl-partition pi+1 -> separator
                        clusters[i].self = SepID(l, pi/2);
                        assert(pi/2 >= 0 && pi/2 < sepsizes.size());
                        sepsizes[pi/2]++;
                    }
                }
            }
        }
        { // Stats
            Stats<int> sepsstats = Stats<int>();
            for(auto v: sepsizes) sepsstats.addData(v);
            if(verb) printf("  Depth %2d: %5d separators, [%5d %5d], mean %6.1f\n", 
                 depth+1, sepsstats.getCount(), sepsstats.getMin(), sepsstats.getMax(), sepsstats.getMean());
        }
    }
    for(int i = 0; i < size; i++) {
        if(clusters[i].self.lvl == -1) { 
            clusters[i].self = SepID(0, parts[i]); // Leaves
        }
    }
    // Leaf-clusters
    for(int i = 0; i < size; i++) {
        SepID self = clusters[i].self;
        // If it's a separator
        if(self.lvl > 0) {
            // Define left/right part
            // Find left & right SepID
            SepID left  = SepID(0, parts[i]);
            SepID right = SepID(-1, 0);
            int pi = part_at_lvl(parts[i], self.lvl, nlevels);
            assert(self.sep == pi/2);
            for (SpMat::InnerIterator it(A,i); it; ++it) {
                int j = it.row();
                SepID nbr = clusters[j].self;
                if(nbr.lvl < self.lvl) {
                    int pj = part_at_lvl(parts[j], self.lvl, nlevels);
                    assert(pi == pj || pi+1 == pj);
                    if(pj <= pi) { // Update left. It could be empty.
                        left  = find_highest_common(nbr, left);
                    } else { // Update right. It cannot be empty.
                        if(right.lvl == -1) {
                            right = clusters[j].self;
                        } else {
                            right = find_highest_common(nbr, right);
                        }
                    }
                }
            }
            assert(right.lvl != -1); // *cannot* be empty
            clusters[i].l = left;
            clusters[i].r = right;
        }
    }
    for(int i = 0; i < size; i++) {
        SepID self = clusters[i].self;
        if(self.lvl == 0) {
            clusters[i].l = self;
            clusters[i].r = self;
        }
    }
    return clusters;
}

vector<ClusterID> partition_modifiedND(SpMat &A, int nlevels, bool verb, bool useVertexSep, MatrixXd* Xcoo) {
    int N = A.rows();
    bool geo = (Xcoo != nullptr);
    if(verb && geo) printf("Geometric MND partitioning & ordering\n");
    if(verb && !geo) printf("Algebraic MND partitioning & ordering\n");
    // CSC format
    int nnz = A.nonZeros();
    auto csc = SpMat2CSC(A);
    vector<int> colptr = get<0>(csc);
    vector<int> rowval = get<1>(csc);
    // Workspace for Metis partitioning
    vector<int> colptrtmp(N+1);
    vector<int> rowvaltmp(nnz);
    vector<int> parttmp(N);
    // The data to partition
    vector<vector<int>> doms(1, vector<int>(N));
    iota(doms[0].begin(), doms[0].end(), 0);
    // Partitioning results
    vector<ClusterID> part = vector<ClusterID>(N, ClusterID(SepID(nlevels-1,0),SepID(nlevels-1,0),SepID(nlevels-1,0)));
    // Where we store results
    for(int depth = 0; depth < nlevels - 1; depth++) {
        Stats<int> sepsstats = Stats<int>();
        int level = nlevels - depth - 1;
        timer t0000 = wctime();
        // Get all separators
        vector<vector<int>> newdoms(pow(2, depth+1));
        for(int sep = 0; sep < pow(2, depth); sep++) {
            SepID idself = SepID(level, sep);
            // Prepare sorted dofs vector            
            vector<int> dofssep = doms[sep];
            sort(dofssep.begin(), dofssep.end());
            // Generate matrix & perform ND
            if(geo) {
                partition_nd_geo(colptr, rowval, dofssep, parttmp, Xcoo, colptrtmp); // colptrtmp can be any array with >= N elements
            } else {
                partition_metis(colptr, rowval, colptrtmp, rowvaltmp, dofssep, parttmp, useVertexSep); // result in parttmp[0...size-1]
            }
            // Update left/right
            SepID idleft  = SepID(level - 1, 2 * sep);
            SepID idright = SepID(level - 1, 2 * sep + 1);
            for (int i = 0; i < dofssep.size(); i++)
            {
                int j = dofssep[i];
                // Update self
                if (part[j].self == idself) {
                    if (parttmp[i] == 0) {
                        part[j].self = idleft;    
                    } else if(parttmp[i] == 1) {
                        part[j].self = idright;
                    }
                }
                // Update left/right               
                if (parttmp[i] == 0) { // Boundary \union interior                    
                    if (part[j].l == idself) {
                        part[j].l = idleft;
                    } 
                    if (part[j].r == idself) {
                        part[j].r = idleft;
                    }
                } else if (parttmp[i] == 1) { // Boundary \union interior                    
                    if (part[j].l == idself) {
                        part[j].l = idright;
                    } 
                    if (part[j].r == idself) {
                        part[j].r = idright;
                    }                
                } else if (parttmp[i] == 2) { // Separator \inter interior
                    if (part[j].self == idself) {                        
                        part[j].l = idleft;
                        part[j].r = idright;
                    }
                }
            }
            // Build left/right
            auto isleft   = [&part, &idleft ](int i){return (part[i].self == idleft || part[i].l == idleft || part[i].r == idleft);};
            auto isright  = [&part, &idright ](int i){return (part[i].self == idright || part[i].l == idright || part[i].r == idright);};
            auto isnewsep = [&part, &idself, &idleft, &idright ](int i){return (part[i].self == idself && part[i].l == idleft && part[i].r == idright);};
            auto nleft  = count_if(dofssep.begin(), dofssep.end(), isleft);
            auto nright = count_if(dofssep.begin(), dofssep.end(), isright);
            auto nnewsep = count_if(dofssep.begin(), dofssep.end(), isnewsep);
            sepsstats.addData(nnewsep);
            vector<int> newleft(nleft);
            vector<int> newright(nright);            
            copy_if(dofssep.begin(), dofssep.end(), newleft.begin(),  isleft);
            copy_if(dofssep.begin(), dofssep.end(), newright.begin(), isright);
            newdoms[2 * sep]     = newleft;
            newdoms[2 * sep + 1] = newright;
        }
        doms = newdoms;
        timer t0001 = wctime();
        if(verb) printf("  Depth %2d: %3.2e s. (%5d separators, [%5d %5d], mean %6.1f)\n", 
                 depth+1, elapsed(t0000, t0001), sepsstats.getCount(), sepsstats.getMin(), sepsstats.getMax(), sepsstats.getMean());
    }
    return part;   
}

}