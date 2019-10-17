#include <iostream>
#include <fstream>
#include "spaND.h"
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/IterativeSolvers>
#include "mmio.hpp"

using namespace Eigen;
using namespace std;
using namespace spaND;

int main(int argc, char* argv[]) {
    assert(argc == 5);
    string folder(argv[1]);
    int nlevels = atoi(argv[2]);
    double tol = atof(argv[3]);
    int skip = atoi(argv[4]);
    cout << "Inputs folder " << folder << " nlevels = " << nlevels << " tol = " << tol << " skip = " << skip << endl;
    string file  = folder + "/jac4.mm";
    string xfile = folder + "/xCoords.mm";
    string yfile = folder + "/yCoords.mm";
    cout << "Reading from " << file << " " << xfile << " " << yfile << endl;
    SpMat A = mmio::sp_mmread<double,int>(file);
    int N = A.rows();
    cout << "A has " << N << " rows" << endl;
    VectorXd xcoords = mmio::dense_mmread<double>(xfile);
    VectorXd ycoords = mmio::dense_mmread<double>(yfile);
    // Create X coordinate matrix
    MatrixXd X(2, N);
    assert(xcoords.size() == N/2);
    assert(ycoords.size() == N/2);
    for(int i = 0; i < N; i++) {
        X(0,i) = xcoords(i/2);
        X(1,i) = ycoords(i/2);
    }
    Tree t = Tree(nlevels);
    t.set_tol(tol);
    t.set_skip(skip);
    t.set_use_geo(true);
    t.set_Xcoo(&X);
    // Run algo    
    timer start = wctime();
    t.partition(A);
    // Rest
    timer part = wctime();
    t.assemble(A);
    timer ass = wctime();
    try {
        t.factorize();
    } catch (exception& ex) {
        cout << ex.what();
        exit(1);
    }
    timer end = wctime();
    t.print_log();
    cout << "Timings:" << endl << "  Partition: " << elapsed(start,part) << " s." << endl << "  Assembly: " << elapsed(part,ass) << " s." << endl << "  Factorization: " << elapsed(ass, end) << " s." << endl;
    // Use CG
    VectorXd x = VectorXd::Zero(N);
    VectorXd b = VectorXd::Random(N);
    timer cg0 = wctime();
    int iter = cg(A, b, x, t, 500, 1e-12, true);
    timer cg1 = wctime();
    cout << "CG: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
    cout << "  CG: " << elapsed(cg0, cg1) << " s." << endl;
    return 0;
}
