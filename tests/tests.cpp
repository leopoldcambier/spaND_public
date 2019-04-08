#include <gtest/gtest.h>
#include <iostream>
#include <assert.h>
#include <algorithm>
#include <random>
#include "tree.h"
#include "mmio.hpp"
#include "cxxopts.hpp"


using namespace std;
using namespace Eigen;

bool VERB = false;
int  N_THREADS = 4;

SpMat neglapl(int n, int d) {
    stringstream s;
    s << "../mats/neglapl_" << d << "_" << n << ".mm";
    string file = s.str();
    SpMat A = mmio::sp_mmread<double,int>(file);
    return A;
}

SpMat neglapl_unsym(int n, int d, int seed) {
    SpMat A = neglapl(n, d);
    default_random_engine gen;
    gen.seed(seed);
    uniform_real_distribution<double> rand(-1, 1);
    for(int k = 0; k < A.outerSize(); ++k) {
        for(SpMat::InnerIterator it(A, k); it; ++it) {
            A.coeffRef(it.row(), it.col()) += rand(gen);
        }
    }
    return A;
}

SpMat random_SpMat(int n, double p, int seed) {
    default_random_engine gen;
    gen.seed(seed);
    uniform_real_distribution<double> dist(0.0,1.0);
    vector<Triplet<double>> triplets;
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            auto v_ij = dist(gen);
            if(v_ij < p) {
                triplets.push_back(Triplet<double>(i,j,v_ij));
            }
        }
    }
    SpMat A(n,n);
    A.setFromTriplets(triplets.begin(), triplets.end()); 
    return A;
}

SpMat identity_SpMat(int n) {
    vector<Triplet<double>> triplets;
    for(int i = 0; i < n; ++i) {
        triplets.push_back(Triplet<double>(i,i,1.0));
    }
    SpMat A(n,n);
    A.setFromTriplets(triplets.begin(), triplets.end()); 
    return A;
}

TEST(MatrixMarket, Sparse) {
    // 1
    SpMat A = mmio::sp_mmread<double,int>("../mats/test1.mm");
    SpMat Aref(2, 3);
    Aref.insert(0, 0) = 1;
    Aref.insert(0, 1) = -2e2;
    Aref.insert(1, 1) = 3e3;
    Aref.insert(1, 2) = -4.4e4;
    EXPECT_EQ(A.nonZeros(), 4);
    EXPECT_EQ((Aref - A).norm(), 0.0);
    // 2
    A = mmio::sp_mmread<double,int>("../mats/test2.mm");
    Aref = SpMat(3, 3);
    Aref.insert(0, 0) = 1.1;
    Aref.insert(1, 1) = 2e2;
    Aref.insert(2, 0) = -3.3;
    Aref.insert(0, 2) = -3.3;
    EXPECT_EQ(A.nonZeros(), 4);
    EXPECT_EQ((Aref - A).norm(), 0.0);
    // 3
    A = mmio::sp_mmread<double,int>("../mats/test3.mm");
    Aref = SpMat(4, 1);
    Aref.insert(3, 0) = -1;
    EXPECT_EQ(A.nonZeros(), 1);
    EXPECT_EQ((Aref - A).norm(), 0.0);
    // 4
    A = mmio::sp_mmread<double,int>("../mats/test4.mm");
    Aref = SpMat(2, 2);
    Aref.insert(1, 0) = -3.3;
    Aref.insert(0, 1) = -3.3;
    EXPECT_EQ(A.nonZeros(), 2);
    EXPECT_EQ((Aref - A).norm(), 0.0);
}

TEST(MatrixMarket, Array) {
    // 5
    MatrixXd A = mmio::dense_mmread<double>("../mats/test5.mm");
    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 3);
    MatrixXd Aref(2, 3);
    Aref << 1, 3, -5, 2, 4, 1e6; // row-wise filling in eigen
    EXPECT_EQ((Aref - A).norm(), 0.0);
    // 6
    A = mmio::dense_mmread<double>("../mats/test6.mm");
    Aref = MatrixXd(2, 2);
    EXPECT_EQ(A.rows(), 2);
    EXPECT_EQ(A.cols(), 2);
    Aref << 1, -2, -2, 3; // row-wise filling in eigen
    EXPECT_EQ((Aref - A).norm(), 0.0);
}

/** Util.cpp tests **/

TEST(Util, AreConnected) {
    // 3x3 laplacian
    SpMat A = mmio::sp_mmread<double,int>("../mats/neglapl_2_3.mm");
    VectorXi a(2);
    VectorXi b(3);
    a << 0, 1;
    b << 6, 7, 8;
    EXPECT_FALSE(are_connected(a, b, A));
    a = VectorXi(2);
    b = VectorXi(3);
    a << 0, 1;
    b << 2, 5, 8;
    EXPECT_TRUE(are_connected(a, b, A));
    a = VectorXi(2);
    b = VectorXi(1);
    a << 3, 4;
    b << 5;
    EXPECT_TRUE(are_connected(a, b, A));
    a = VectorXi(1);
    b = VectorXi(1);
    a << 6;
    b << 6;
    EXPECT_TRUE(are_connected(a, b, A));
}

TEST(Util, ShouldBeDisconnected) {
    EXPECT_TRUE(should_be_disconnected(0, 0, 0, 2));
    EXPECT_TRUE(should_be_disconnected(0, 0, 1, 2));
    EXPECT_TRUE(should_be_disconnected(0, 0, 4, 2));
    EXPECT_TRUE(should_be_disconnected(0, 0, 5, 2));
    EXPECT_TRUE(should_be_disconnected(0, 0, 1000, 2));

    EXPECT_TRUE(should_be_disconnected(1, 2, 2, 0));
    EXPECT_TRUE(should_be_disconnected(0, 1, 1, 1));
    EXPECT_TRUE(should_be_disconnected(0, 2, 1, 1));
    EXPECT_TRUE(should_be_disconnected(2, 0, 0, 5));
    EXPECT_TRUE(should_be_disconnected(2, 2, 0, 1));

    EXPECT_FALSE(should_be_disconnected(0, 1, 0, 0));
    EXPECT_FALSE(should_be_disconnected(0, 2, 0, 0));
    EXPECT_FALSE(should_be_disconnected(0, 10, 0, 0));

    EXPECT_FALSE(should_be_disconnected(2, 0, 1, 5));
    EXPECT_FALSE(should_be_disconnected(2, 0, 1, 6));
    EXPECT_FALSE(should_be_disconnected(2, 1, 1, 2));
    EXPECT_FALSE(should_be_disconnected(2, 1, 1, 3));
}

TEST(Util, ChooseRank) {
    VectorXd errs = VectorXd(5);
    errs << 1.0, -0.1, 0.01, -0.001, 1e-4;
    EXPECT_EQ(choose_rank(errs, 1e-1), 2);
    EXPECT_EQ(choose_rank(errs, 1e-2), 3);
    EXPECT_EQ(choose_rank(errs, 1.0), 0);
    EXPECT_EQ(choose_rank(errs, 0), 5);
    EXPECT_EQ(choose_rank(errs, 1e-16), 5);
}

TEST(Util, Block2Dense) {
    // Usual case
    {
        SpMat A(5, 5);
        A.insert(0, 0) = 1.0;
        A.insert(2, 2) = -2.0;
        A.insert(1, 3) = 3.0;
        A.makeCompressed();
        VectorXi rowval = Map<VectorXi>(A.innerIndexPtr(), A.nonZeros());
        VectorXi colptr = Map<VectorXi>(A.outerIndexPtr(), 6);
        VectorXd nnzval = Map<VectorXd>(A.valuePtr(), A.nonZeros());
        MatrixXd Ad = MatrixXd::Zero(3, 3);
        block2dense(rowval, colptr, nnzval, 1, 1, 3, 3, &Ad, false);
        MatrixXd Adref = MatrixXd::Zero(3, 3);
        Adref << 0, 0, 3, 0, -2, 0, 0, 0, 0;
        EXPECT_EQ((Adref - Ad).norm(), 0);
    }
    // Transpose
    {
        SpMat A(3, 4);
        A.insert(0, 1) = 1.0;
        A.insert(1, 0) = 2.0;
        A.insert(1, 2) = 3.0;
        A.insert(2, 1) = 4.0;
        A.insert(0, 3) = 5.0;
        A.makeCompressed();
        VectorXi rowval = Map<VectorXi>(A.innerIndexPtr(), A.nonZeros());
        VectorXi colptr = Map<VectorXi>(A.outerIndexPtr(), 5);
        VectorXd nnzval = Map<VectorXd>(A.valuePtr(), A.nonZeros());
        MatrixXd Ad = MatrixXd::Zero(3, 2);
        block2dense(rowval, colptr, nnzval, 0, 0, 2, 3, &Ad, true);
        MatrixXd Adref = MatrixXd::Zero(3, 2);
        Adref << 0, 2, 1, 0, 0, 3;
        EXPECT_EQ((Adref - Ad).norm(), 0);
    }
}

TEST(Util, LinspaceNd) {
    MatrixXd X2 = linspace_nd(3, 2);
    MatrixXd X2ref(2, 9);
    X2ref << 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2;
    EXPECT_EQ((X2ref - X2).norm(), 0);
    MatrixXd X3 = linspace_nd(2, 3);
    MatrixXd X3ref(3, 8);
    X3ref << 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1;
    EXPECT_EQ((X3ref - X3).norm(), 0);
}

TEST(Util, SymmPerm) {
    vector<int> dims  = {2, 2,  2,  3, 3,  3,  3 };
    vector<int> sizes = {5, 10, 20, 5, 15, 25, 30};
    for(int test = 0; test < dims.size(); test++) {
        int s = sizes[test];
        int d = dims[test];
        stringstream ss;
        ss << "../mats/neglapl_" << d << "_" << s << ".mm";
        SpMat A = mmio::sp_mmread<double,int>(ss.str());
        // Create random perm
        int N = A.rows();
        VectorXi p = VectorXi::LinSpaced(N, 0, N-1);
        random_device rd;
        mt19937 g(rd());
        shuffle(p.data(), p.data() + N, g);
        // Compare
        SpMat pAp = symm_perm(A, p);
        SpMat pApref = p.asPermutation().inverse() * A * p.asPermutation();
        EXPECT_EQ((pAp - pApref).norm(), 0.0);
    }
}

TEST(Util, isperm) {
    VectorXi perm1(10);
    VectorXi perm2(10);
    VectorXi noperm1(10);
    VectorXi noperm2(5);
    perm1 << 0, 9, 8, 1, 4, 2, 3, 7, 5, 6;
    perm2 << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9;
    noperm1 << 0, 9, 8, 1, 4, 2, 3, 5, 5, 6;
    noperm2 << 0, 9, 8, 1, 4;
    EXPECT_TRUE(isperm(&perm1));
    EXPECT_TRUE(isperm(&perm2));
    EXPECT_FALSE(isperm(&noperm1));
    EXPECT_FALSE(isperm(&noperm2));
}

TEST(Util, swap2perm) {
    VectorXi swap(6);
    VectorXi perm(6);
    VectorXi permRef(6);
    swap << 3, 3, 2, 5, 4, 5;
    permRef << 3, 0, 2, 5, 4, 1;
    swap2perm(&swap, &perm); // perm.asPermutation().transpose() * x <=> x[perm]
    EXPECT_EQ((perm - permRef).norm(), 0.0);
}

SpMat symmetric_graph_ref(SpMat A) {
    SpMat ATabs = A.cwiseAbs().transpose();
    return A.cwiseAbs() + ATabs + identity_SpMat(A.rows());
}

TEST(Util, symmetric_graph) {
    for(int i = 1; i < 100; i++) {
        SpMat A = random_SpMat(i, 0.2, i);
        SpMat AAT = symmetric_graph(A);
        EXPECT_LT( (AAT - symmetric_graph_ref(A)).norm(), 1e-12);
    }
}


/** Partitioning tests **/

/**
 * Check the partitioning of a square laplacian 5x5
 */
TEST(PartitionTest, Square) {
    SpMat A = mmio::sp_mmread<double,int>("../mats/neglapl_2_5.mm");
    MatrixXd X = linspace_nd(5, 2);
    Tree t(3);
    t.set_verb(VERB);
    t.set_use_geo(true);
    t.set_Xcoo(&X);
    t.partition(A);
    vector<SepID> sepidref { 
        SepID(0,0), SepID(0,0), SepID(1,0), SepID(0,1), SepID(0,1),
        SepID(0,0), SepID(0,0), SepID(1,0), SepID(0,1), SepID(0,1),
        SepID(2,0), SepID(2,0), SepID(2,0), SepID(2,0), SepID(2,0),
        SepID(0,2), SepID(0,2), SepID(1,1), SepID(0,3), SepID(0,3),
        SepID(0,2), SepID(0,2), SepID(1,1), SepID(0,3), SepID(0,3), 
    } ;
    vector<SepID> leftref { 
        SepID(0,0), SepID(0,0), SepID(0,0), SepID(0,1), SepID(0,1),
        SepID(0,0), SepID(0,0), SepID(0,0), SepID(0,1), SepID(0,1),
        SepID(0,0), SepID(0,0), SepID(1,0), SepID(0,1), SepID(0,1),
        SepID(0,2), SepID(0,2), SepID(0,2), SepID(0,3), SepID(0,3),
        SepID(0,2), SepID(0,2), SepID(0,2), SepID(0,3), SepID(0,3), 
    } ;
    vector<SepID> rightref { 
        SepID(0,0), SepID(0,0), SepID(0,1), SepID(0,1), SepID(0,1),
        SepID(0,0), SepID(0,0), SepID(0,1), SepID(0,1), SepID(0,1),
        SepID(0,2), SepID(0,2), SepID(1,1), SepID(0,3), SepID(0,3),
        SepID(0,2), SepID(0,2), SepID(0,3), SepID(0,3), SepID(0,3),
        SepID(0,2), SepID(0,2), SepID(0,3), SepID(0,3), SepID(0,3), 
    } ;
    for(int i = 0; i < t.part.size(); i++) {
        ASSERT_TRUE(t.part[i].self  == sepidref[i]);
        ASSERT_TRUE(t.part[i].l     == leftref[i]);
        ASSERT_TRUE(t.part[i].r     == rightref[i]);
    }
}

/**
 * Check consistency of the partitioning
 */
TEST(PartitionTest, Consistency) {
    vector<int> dims  = {2, 2,  2,   3, 3,  3 };
    vector<int> sizes = {5, 20, 100, 5, 15, 25};
    for(int test = 0; test < dims.size(); test++) {
        int s = sizes[test];
        int d = dims[test];
        stringstream ss;
        ss << "../mats/neglapl_" << d << "_" << s << ".mm";
        int n = pow(s, d);
        string file = ss.str();
        for(int nlevels = 1; nlevels < 8; nlevels++) {
            for(int geoi = 0; geoi < 2; geoi++) {
                bool geo = (geoi == 0);
                // Partition tree
                MatrixXd X = linspace_nd(s, d);
                Tree t(nlevels);
                t.set_verb(VERB);
                SpMat A = mmio::sp_mmread<double,int>(file);
                t.set_use_geo(geo);
                t.set_Xcoo(&X);
                t.partition(A);
                // (1) Lengths
                ASSERT_EQ(t.part.size(), n);
                // (2) Check ordering integrity
                for(int i = 0; i < n; i++) {
                    auto pi = t.part[i].self;
                    for (SpMat::InnerIterator it(A,i); it; ++it) {
                        int j = it.row();
                        auto pj = t.part[j].self;  
                        ASSERT_FALSE(should_be_disconnected(pi.lvl, pj.lvl, pi.sep, pj.sep));
                    }
                }
                // (3) Check left/right integrity          
                for(int i = 0; i < n; i++) {
                    auto pi = t.part[i].self;
                    auto li = t.part[i].l;
                    auto ri = t.part[i].r;
                    if(pi.lvl == 0) {
                        ASSERT_TRUE(pi == li);
                        ASSERT_TRUE(pi == ri);
                    } else {
                        ASSERT_TRUE(pi.lvl > li.lvl);
                        ASSERT_TRUE(pi.lvl > ri.lvl);
                        while(li.lvl < pi.lvl - 1) {
                            li.lvl += 1;
                            li.sep /= 2;
                        }
                        while(ri.lvl < pi.lvl - 1) {
                            ri.lvl += 1;
                            ri.sep /= 2;
                        }
                        ASSERT_TRUE(li.lvl == pi.lvl - 1);
                        ASSERT_TRUE(ri.lvl == pi.lvl - 1);
                        ASSERT_TRUE(li.sep == 2 * pi.sep);
                        ASSERT_TRUE(ri.sep == 2 * pi.sep + 1);
                    }
                    
                }
            }
        }
    }
}

/** Assembly tests **/

/** 
 * Check assembly
 */
TEST(Assembly, Consistency) {
    vector<int> dims  = {2, 2,  2,  3, 3,  3};
    vector<int> sizes = {5, 10, 20, 5, 10, 15};
    for(int test = 0; test < dims.size(); test++) {        
        int s = sizes[test];
        int d = dims[test];
        SpMat Aref = neglapl(s, d);
        SpMat Arefunsym = neglapl_unsym(s, d, test);
        for(int nlevels = 2; nlevels < 5 ; nlevels++) {
            /**
             * Symmetric case
             */
            {
                // Partition and assemble
                Tree t(nlevels);
                t.set_verb(VERB);
                t.set_use_geo(false);
                t.partition(Aref);
                t.assemble(Aref);
                // Couple things to check
                ASSERT_FALSE(t.is_factorized());
                // Get permutation
                VectorXi p = t.get_assembly_perm();
                // Check it's indeed a permutation
                ASSERT_TRUE(isperm(&p));
                auto P = p.asPermutation();
                // Check get_mat()
                SpMat A2 = t.get_assembly_mat();
                EXPECT_EQ((P.inverse() * Aref * P - A2).norm(), 0.0);
            }
            /**
             * Unsymmetric case
             */
            {
                // Partition and assemble
                Tree t(nlevels);
                t.set_verb(VERB);
                t.set_symmetry(false);
                t.set_use_geo(false);
                t.partition(Arefunsym);
                t.assemble(Arefunsym);                
                // Couple things to check
                ASSERT_FALSE(t.is_factorized());
                // Get permutation
                VectorXi p = t.get_assembly_perm();
                // Check it's indeed a permutation
                ASSERT_TRUE(isperm(&p));
                auto P = p.asPermutation();
                // Check get_mat()
                SpMat A2 = t.get_assembly_mat();
                EXPECT_EQ((P.inverse() * Arefunsym * P - A2).norm(), 0.0);
            }
        }
    }
}

/** Factorization tests **/

/**
 * Test that with eps=0, we get exact solutions
 */
TEST(ApproxTest, Exact) {
    vector<int> dims  = {2, 2,  2,  3, 3};
    vector<int> sizes = {5, 10, 20, 5, 15};
    for(int test = 0; test < dims.size(); test++) {
        cout << "Test " << test << "..." << endl;
        int s = sizes[test];
        int d = dims[test];
        int n = pow(s, d);
        int nlevelsmin = n < 1000 ? 1 : 8;
        SpMat Aref = neglapl(s, d);
        SpMat Arefunsym = neglapl_unsym(s, d, test);
        for(int nlevels = nlevelsmin; nlevels < nlevelsmin+5 ; nlevels++) {
            for(int so = 0; so < 3; so++) {
                bool scale = true;
                bool ortho = false;
                if (so == 0) { scale = false; ortho = false; }
                if (so == 1) { scale = true ; ortho = false; }
                if (so == 2) { scale = true ; ortho = true ; }
                { // just skip everything
                    { // Symmetric                        
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.partition(Aref);
                        t.assemble(Aref);
                        t.set_tol(0.0);
                        t.set_skip(1000);
                        t.set_scale(scale);
                        t.set_ortho(ortho);
                        t.factorize();
                        VectorXd b = random(n, test+nlevels+so+2019+0);
                        auto x = b;
                        t.solve(x);
                        double err = (Aref*x-b).norm() / b.norm();
                        EXPECT_LE(err, 1e-14);
                    }
                    { // Unsymmetric
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.set_symmetry(false);
                        t.partition(Arefunsym);
                        t.assemble(Arefunsym);
                        t.set_tol(0.0);
                        t.set_skip(1000);
                        t.set_scale(scale);
                        t.set_ortho(ortho);
                        t.factorize();
                        VectorXd b = random(n, test+nlevels+so+2019+1);
                        auto x = b;
                        t.solve(x);
                        double err = (Arefunsym*x-b).norm() / b.norm();
                        EXPECT_LE(err, 1e-12) << err;
                    }                    
                }
                { // Really small tolerance
                    { // Symmetric
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.partition(Aref);
                        t.assemble(Aref);
                        t.set_tol(1e-14);
                        t.set_skip(0);
                        t.set_scale(scale);
                        t.set_ortho(ortho);
                        t.factorize();
                        VectorXd b = random(n, test+nlevels+so+2019+2);
                        auto x = b;
                        t.solve(x);
                        double err = (Aref*x-b).norm() / b.norm();
                        EXPECT_LE(err, 1e-12) << scale << ortho << nlevels;
                    }
                    { // Unsymmetric
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.set_symmetry(false);
                        t.partition(Arefunsym);
                        t.assemble(Arefunsym);
                        t.set_tol(1e-14);
                        t.set_skip(0);
                        t.set_scale(scale);
                        t.set_ortho(ortho);
                        t.factorize();
                        VectorXd b = random(n, test+nlevels+so+2019+3);
                        auto x = b;
                        t.solve(x);
                        double err = (Arefunsym*x-b).norm() / b.norm();
                        EXPECT_LE(err, 1e-10);
                    }
                }
                if (scale && ortho) { // Preserve on top of that
                    Tree t(nlevels);
                    t.set_verb(VERB);                    
                    t.partition(Aref);
                    t.assemble(Aref);
                    t.set_tol(1e-14);
                    t.set_skip(0);
                    t.set_scale(scale);
                    t.set_ortho(ortho);
                    MatrixXd phi = random(Aref.rows(), 3, test+nlevels+so+2019);
                    t.set_preserve(true);
                    t.set_phi(&phi);
                    t.factorize();
                    VectorXd b = random(n, test+nlevels+so+2019+4);
                    auto x = b;
                    t.solve(x);
                    double err = (Aref*x-b).norm() / b.norm();
                    EXPECT_TRUE(err < 1e-12);
                }
            }
        }
    }
}

/** 
 * Test conservation is correct
 */
TEST(ApproxTest, Preservation) {
    vector<int> dims  = {2, 2,  2,  3, 3,  3};
    vector<int> sizes = {5, 10, 20, 5, 10, 25};
    for(int test = 0; test < dims.size(); test++) {
        cout << "Test " << test << "..." << endl;
        int s = sizes[test];
        int d = dims[test];
        stringstream ss;
        ss << "../mats/neglapl_" << d << "_" << s << ".mm";
        int n = pow(s, d);
        string file = ss.str();
        int nlevelsmin = n < 1000 ? 1 : 8;
        vector<double> tols = {10, 1e-1, 1e-2, 1e-4, 1e-6, 0.0};
        for(int nlevels = nlevelsmin; nlevels < nlevelsmin + 5; nlevels++) {
            for(int it = 0; it < tols.size(); it++) {
                for(int skip = 0; skip < 3; skip++) {
                    bool scale = true;
                    bool ortho = true;
                    SpMat A = mmio::sp_mmread<double,int>(file);
                    // Check a 1 is preserved
                    {
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.partition(A);
                        t.assemble(A);
                        MatrixXd phi = MatrixXd::Ones(n, 1);
                        t.set_tol(tols[it]);
                        t.set_skip(skip);
                        t.set_scale(scale);
                        t.set_ortho(ortho);
                        t.set_preserve(true);
                        t.set_phi(&phi);
                        t.factorize();
                        for(int c = 0; c < phi.cols(); c++) {
                            VectorXd b = A * phi.col(c);
                            VectorXd x = b;
                            t.solve(x);
                            double err1 = (A*x-b).norm() / b.norm();
                            double err2 = (x-phi.col(c)).norm() / phi.col(c).norm();
                            EXPECT_TRUE(err1 < 1e-12) << "err1 = " << err1 << " | " << skip << " " << it << " " << nlevels << " " << test << endl;
                            EXPECT_TRUE(err2 < 1e-12) << "err2 = " << err2 << " | " << skip << " " << it << " " << nlevels << " " << test << endl;
                        }
                        VectorXd b = random(n, nlevels+it+skip+2019);
                        auto x = b;
                        t.solve(x);
                        double err = (A*x-b).norm() / b.norm();
                        if (tols[it] == 0.0) {
                            EXPECT_TRUE(err < 1e-12);
                        } else {
                            EXPECT_TRUE(err < tols[it] * 1e2);
                        }
                    }
                    // Check that a multiple random b are preserved
                    {
                        Tree t(nlevels);
                        t.set_verb(VERB);
                        t.partition(A);
                        t.assemble(A);
                        MatrixXd phi = random(n, 5, nlevels+it+skip+2019);
                        t.set_tol(tols[it]);
                        t.set_skip(skip);
                        t.set_scale(scale);
                        t.set_ortho(ortho);
                        t.set_preserve(true);
                        t.set_phi(&phi);
                        t.factorize();
                        for(int c = 0; c < phi.cols(); c++) {
                            VectorXd b = A * phi.col(c);
                            VectorXd x = b;
                            t.solve(x);
                            double err1 = (A*x-b).norm() / b.norm();
                            double err2 = (x-phi.col(c)).norm() / phi.col(c).norm();
                            EXPECT_TRUE(err1 < 1e-12) << "err1 = " << err1 << " | " << skip << " " << it << " " << nlevels << " " << test << endl;
                            EXPECT_TRUE(err2 < 1e-12) << "err2 = " << err2 << " | " << skip << " " << it << " " << nlevels << " " << test << endl;
                        }
                        VectorXd b = random(n, nlevels+it+skip+2019);
                        auto x = b;
                        t.solve(x);
                        double err = (A*x-b).norm() / b.norm();
                        if (tols[it] == 0.0) {
                            EXPECT_TRUE(err < 1e-12);
                        } else {
                            EXPECT_TRUE(err < tols[it] * 1e2);
                        }
                    }
                }
            }
        }
    }
}

/**
 * Test that the approximations are reasonnable accurate 
 * with and without preservation
 */
TEST(ApproxTest, Approx) {
    vector<int> dims  = {2, 2,  2,  3, 3};
    vector<int> sizes = {5, 10, 20, 5, 15};
    vector<double> tols = {0.0, 1e-10, 1e-6, 1e-2, 10};
    matrix_hash<VectorXd> hash;
    for(int test = 0; test < dims.size(); test++) {
        vector<size_t> allhashes;
        cout << "Test " << test << "... ";
        int s = sizes[test];
        int d = dims[test];
        SpMat Aref = neglapl(s, d);
        SpMat Arefunsym = neglapl_unsym(s, d, test+2019);
        int n = pow(s, d);
        int nlevelsmin = n < 1000 ? 1 : 8;
        for(int nlevels = nlevelsmin; nlevels < nlevelsmin + 5; nlevels++) {
            for(int it = 0; it < tols.size(); it++) {
                for(int skip = 0; skip < 3; skip++) {
                    for(int so = 0; so < 3; so++) {
                        for(int symm = 0; symm < 2; symm++) {
                            for(int pr = 0; pr < 2; pr++) {
                                bool scale = true;
                                bool ortho = false;
                                if (so == 0) { scale = false; ortho = false; }
                                if (so == 1) { scale = true ; ortho = false; }
                                if (so == 2) { scale = true ; ortho = true ; }
                                Tree t(nlevels);
                                t.set_verb(VERB);
                                t.set_symmetry(symm);
                                SpMat A = (symm ? Aref : Arefunsym);
                                if( pr && (! (symm && scale && ortho) ) ) continue;
                                t.partition(A);
                                t.assemble(A);
                                ASSERT_FALSE(t.is_factorized());
                                MatrixXd phi = random(A.rows(), 2, nlevels+it+skip+so+symm+pr+2019);
                                t.set_tol(tols[it]);
                                t.set_skip(skip);
                                t.set_scale(scale);
                                t.set_ortho(ortho);
                                t.set_preserve(pr);
                                t.set_phi(&phi);
                                t.factorize();
                                ASSERT_TRUE(t.is_factorized());
                                VectorXd b = random(n, nlevels+it+skip+so+symm+pr+2019);
                                auto x = b;
                                t.solve(x);
                                double err = (A*x-b).norm() / b.norm();
                                auto hb = hash(b);
                                auto hx = hash(x);
                                allhashes.push_back(hb);
                                allhashes.push_back(hx);
                                if (tols[it] == 0.0) {
                                    EXPECT_LE(err, 1e-12);
                                } else {
                                    EXPECT_LE(err, tols[it] * 1e2) << hb << " " << hx << " " << test << " " << it << " " << nlevels << " " << skip << " " << so << " " << symm << " " << pr << " " << tols[it];
                                }
                            }
                        }
                    }
                }
            }
        }
        size_t h = hashv(allhashes);
        cout << "Overall hash(x,b) = " << h << endl;
    }
}

/** 
 * Test that the code produce reproducable results
 */
TEST(ApproxTest, Repro) {
    int    dims[3]      = {2, 2, 2};
    int    sizes[3]     = {20, 64, 16};
    double tols[6]      = {1e-5, 1e-5, 10, 10, 1e-8, 0.5};
    double skips[6]     = {1, 1, 1, 0, 2, 1};
    bool   scales[6]    = {true, true, true, true, true, false};
    bool   orthos[6]    = {true, true, true, true, false, false};
    bool   preserves[6] = {false, true, false, true, false, false};
    int repeat = 100;
    matrix_hash<VectorXd> hash;
    for(int test = 0; test < 3; test++) {
        cout << "Test " << test << "..." << endl;
        int s = sizes[test];
        int d = dims[test];
        int n = pow(s, d);
        SpMat Aref = neglapl(s, d);
        SpMat Arefunsym = neglapl_unsym(s, d, test);        
        for(int nlevels = 5; nlevels < 7; nlevels++) {
            for(int symm = 0; symm < 2; symm++) {
                for(int pr = 0; pr < 6; pr++) {
                    Tree t(nlevels);
                    t.set_verb(VERB);
                    t.set_symmetry(symm);
                    SpMat A = (symm ? Aref : Arefunsym);
                    MatrixXd phi = random(A.rows(), 3, test+nlevels+symm+pr+2019);
                    t.partition(A);
                    t.assemble(A);
                    t.set_tol(tols[pr]);
                    t.set_skip(skips[pr]);
                    t.set_scale(scales[pr]);
                    t.set_ortho(orthos[pr]);
                    t.set_preserve(preserves[pr]);
                    t.set_phi(&phi);
                    if(preserves[pr] && (! symm)) continue;
                    t.factorize();
                    VectorXd b = random(n, nlevels+test);
                    auto xref = b;
                    t.solve(xref);
                    for(int i = 0; i < repeat; i++) {
                        Tree t2(nlevels);
                        t2.set_verb(VERB);
                        t2.set_symmetry(symm);
                        t2.partition(A);
                        t2.assemble(A);
                        t2.set_tol(tols[pr]);
                        t2.set_skip(skips[pr]);
                        t2.set_scale(scales[pr]);
                        t2.set_ortho(orthos[pr]);
                        t2.set_preserve(preserves[pr]);
                        t2.set_phi(&phi);
                        t2.factorize();
                        auto x = b;
                        t2.solve(x);
                        auto hb = hash(b);
                        auto hx = hash(x);
                        EXPECT_EQ((xref - x).norm(), 0.0) << hb << " " << hx << " " << test << " " << nlevels << " " << symm << " " << pr << " " << tols[pr] << " " << skips[pr] << " " << scales[pr] << " " << orthos[pr] << " " << preserves[pr];
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    cxxopts::Options options("spaND tests", "Test suite for the spaND algorithms.");
    options.add_options()
        ("help", "Print help")
        ("v,verb", "Verbose (default: false)", cxxopts::value<bool>()->default_value("false"))
        ("n_threads", "Number of threads", cxxopts::value<int>()->default_value("4"))
        ;
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    VERB = result["verb"].as<bool>();
    N_THREADS = result["n_threads"].as<int>();
    cout << "n_threads: " << N_THREADS << endl;
    cout << "verb: " << VERB << endl;

    return RUN_ALL_TESTS();
}