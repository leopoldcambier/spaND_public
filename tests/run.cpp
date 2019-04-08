#include <functional>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <stdio.h>

#include "tree.h"
#include "util.h"
#include "mmio.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char* argv[]) {
  vector<int> dims  = {2,  2,  2,   3, 3,  3 };
  vector<int> sizes = {5,  16, 64,  5, 10, 15};
  int ntests = dims.size();
  if (argc > 1) {
      ntests = atoi(argv[1]);
      printf("Running max %d different tests problems\n", ntests);
  }
  matrix_hash<VectorXd> hash;
  vector<size_t> allhashes;
  for(int test = 0; test < ntests; test++) {
    int s = sizes[test];
    int d = dims[test];
    stringstream ss;
    ss << "../mats/neglapl_" << d << "_" << s << ".mm";
    string file = ss.str();
    cout << file << endl;
    SpMat Asymm = mmio::sp_mmread<double,int>(file);
    default_random_engine gen;
    uniform_real_distribution<double> dist(0.5,1.0);
    double a = dist(gen);
    SpMat Aunsymm = a * Asymm.triangularView<Lower>() + (1 - a) * Asymm.triangularView<Upper>(); 
    SpMat* A = nullptr;
    int N = Asymm.rows();
    int nlevelsmin = N < 1000 ? 1 : 8;
    for(int nlevels = nlevelsmin; nlevels < nlevelsmin+3; nlevels++) {
      vector<double> tols = {0.0, 1e-2, 1.0, 10.0};
      for(double tol : tols) {
        for(int skip = 0; skip < 3; skip++) {
          for(int pres = 0; pres < 2; pres++) {
            for(int symm = 0; symm < 2; symm++) {
              for(int geo = 0; geo < 2; geo++) {
                for(int scale = 0; scale < 2; scale++) {
                  for(int ortho = 0; ortho < 2; ortho++) {
                    // Only valid combinations
                    if(ortho && (!scale))   continue;
                    if(pres  && ((!ortho) || (!symm)) ) continue;
                    Tree t = Tree(nlevels);
                    t.set_verb(false);
                    t.set_tol(tol);
                    t.set_skip(skip);
                    t.set_scale(scale);
                    t.set_ortho(ortho);
                    t.set_preserve(pres);
                    t.set_use_geo(geo);
                    t.set_symmetry(symm);
                    MatrixXd phi = MatrixXd::Ones(N,1);
                    MatrixXd X = linspace_nd(s, d);
                    t.set_phi(&phi);                                        
                    t.set_Xcoo(&X);  
                    if(symm) {
                        A = &Asymm;                                            
                    } else {
                        A = &Aunsymm;                                            
                    }
                    t.partition(*A);
                    t.assemble(*A);
                    int errors = t.factorize();                        
                    assert(errors == 0);
                    VectorXd b = VectorXd::Random(N);
                    auto x = b;
                    t.solve(x);
                    double res = ((*A)*x-b).norm() / b.norm();
                    SparseLU<SpMat> lu((*A));
                    VectorXd xref;
                    xref = lu.solve(b);
                    double err = (xref - x).norm() / xref.norm();
                    auto h = hash(x);
                    allhashes.push_back(h);
                    printf("%6d %4d %d] %3d %3.2e %2d %d %d %d %d %d %3.2e %3.2e | %lu\n", N, s, d, nlevels, tol, skip, pres, symm, geo, scale, ortho, res, err, hash(x));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  size_t h = hashv(allhashes);
  cout << "Overall hash: " << h << endl;
  ofstream f;
  f.open("allhashes.log");
  for(auto v : allhashes) {
    f << v << "\n";
  }
  f.close();
  return 1;
}

