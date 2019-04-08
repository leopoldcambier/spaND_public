#include <iostream>
#include <fstream>
#include <sstream>
#include "tree.h"
#include "util.h"
#include "is.h"
#include "cxxopts.hpp"
#include "mmio.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char* argv[]) {

    cxxopts::Options options("spaND", "General driver for spaND. Load a matrix, an optional coordinate file. Then, assemble the matrix, partition it, factor it and solve Ax=b using GMRES.");
    options.add_options()
        // Generic
        ("help", "Print help")
        ("verbose", "Verbose mode", cxxopts::value<bool>()->default_value("true"))
        // General
        ("m,matrix", "Matrix MM coordinate file (mandatory)", cxxopts::value<string>())
        ("l,lvl", "# of levels (mandatory)", cxxopts::value<int>())
        ("symmetric", "Wether the matrix is symmetric or not", cxxopts::value<bool>()->default_value("true"))
        // Geometry
        ("coordinates", "Coordinates MM array file. If provided, will do a geometric partitioning.", cxxopts::value<string>())
        ("n,coordinates_n", "If provided with -n, will use a tensor n^d & geometric partitioning. Overwrites --coordinates", cxxopts::value<int>()->default_value("-1"))
        ("d,coordinates_d", "If provided with -d, will use a tensor n^d & geometric partitioning. Overwrites --coordinates", cxxopts::value<int>()->default_value("-1"))
        // Compression choices
        ("t,tol", "Tolerance", cxxopts::value<double>()->default_value("1e-1"))            
        ("skip", "Skip sparsification", cxxopts::value<int>()->default_value("0"))        
        ("scale", "Wether to scale or not", cxxopts::value<bool>()->default_value("true"))
        ("ortho", "Wether to use orthogonal basis or not", cxxopts::value<bool>()->default_value("true"))
        ("preserve", "Wether to preserve the 1 vector or not (default: false)", cxxopts::value<bool>()->default_value("false"))
        ("scaling_kind", "The scaling kind, PLU or SVD (unsymmetric only)", cxxopts::value<string>()->default_value("PLU"))
        ("use_want_sparsify", "Wether to use want_sparsify (true) or not (default: true)", cxxopts::value<bool>()->default_value("true"))
        // Iterative method        
        ("solver","Wether to use GMRES or CG", cxxopts::value<string>()->default_value("GMRES"))
        ("i,iterations","Iterative solver iterations", cxxopts::value<int>()->default_value("100"))
        // Problem transformation, to try some functions of A
        ("flip_sign", "Wether to solve with -A (true) or not (default: false)", cxxopts::value<bool>()->default_value("false"))
        ("rationale", "Solve with a_0 I + a_1 A + a_2 A^2 ... instead of A", cxxopts::value<vector<double>>())
        // Output more data
        ("output_ordering", "Output ordering in text file Nx(6*nlevels) matrix", cxxopts::value<string>())
        ("monitor_cond", "Monitor condition number of pivots", cxxopts::value<bool>()->default_value("false"))
        // Detail
        ("n_threads", "Number of threads", cxxopts::value<int>()->default_value("1"))        
        ;
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    if ( (!result.count("matrix")) || (!result.count("lvl")) ) {
        cout << "--matrix and --lvl are mandatory" << endl;
        exit(0);
    }
    string matrix = result["matrix"].as<string>();
    string coordinates;
    int cn = result["coordinates_n"].as<int>();
    int cd = result["coordinates_d"].as<int>();
    bool geo_file = (result.count("coordinates") > 0);
    if ( (cn == -1 && cd >= 0) || (cd == -1 && cn >= 0) ) {
        cout << "cn and cd should be both provided, or none should be provided" << endl;
        return 1;
    }
    bool geo_tensor = (cn >= 0 && cd >= 0);
    bool geo = geo_file || geo_tensor;
    if(geo_file) {
        coordinates = result["coordinates"].as<string>();
    }
    double tol = result["tol"].as<double>();
    int skip = result["skip"].as<int>();
    int nlevels = result["lvl"].as<int>();
    bool scale = result["scale"].as<bool>();
    bool ortho = result["ortho"].as<bool>();
    bool preserve = result["preserve"].as<bool>();
    bool symmetric = result["symmetric"].as<bool>();
    bool verb = result["verbose"].as<bool>();
    ScalingKind sk = (result["scaling_kind"].as<string>() == "PLU" ? ScalingKind::PLU : ScalingKind::SVD);
    bool useCG = (result["solver"].as<string>() == "CG");
    bool useGMRES = (result["solver"].as<string>() == "GMRES");
    int iterations = result["iterations"].as<int>();
    if(!useCG && !useGMRES) {
        cout << "Wrong solver picked. Should be CG or GMRES" << endl;
        return 1;
    }
    bool use_want_sparsify = result["use_want_sparsify"].as<bool>();
    int n_threads = result["n_threads"].as<int>();
    bool flip_sign = result["flip_sign"].as<bool>();
    vector<double> rational_coeffs;
    if (result.count("rationale")) {
        for(auto v: result["rationale"].as<vector<double>>()) {
            rational_coeffs.push_back(v);
        }
        cout << "Using rational coeffs: ";
        for(int i = 0; i < rational_coeffs.size(); i++) {
            cout << rational_coeffs[i] << " A^" << i << " + ";
        }
        cout << endl;
    }
    bool monitor_cond = result["monitor_cond"].as<bool>();
    
    // Load a matrix
    SpMat A = mmio::sp_mmread<double,int>(matrix);
    int N = A.rows();
    cout << "Matrix " << N << "x" << N << " loaded from " << matrix << endl;
    if(flip_sign) {
        A = -A;
        cout << "Solving with -A instead" << endl;
    } else if(rational_coeffs.size() > 0) {
        cout << "Solving with rational coefficients instead" << endl;
        // Create identity
        vector<Triplet<double>> tripletList;
        for(int i = 0; i < N; i++) { tripletList.push_back({i,i,1.0}); }
        SpMat I(N,N);
        I.setFromTriplets(tripletList.begin(), tripletList.end());
        // Creating a_0 I + a_1 A + a_2 A^2 ...
        SpMat An = I;
        SpMat RatA = SpMat(N,N);
        for(int i = 0; i < rational_coeffs.size(); i++) {
            RatA = RatA + rational_coeffs[i] * An;
            An = An * A;
        }
        A = RatA;
    }
    cout << A.block(0, 0, 10, 10) << endl;

    // Load coordinates ?
    MatrixXd X;
    if(geo_tensor) {
        if(pow(cn, cd) != N) {
            cout << "Error: cn and cd where both provided, but cn^cd != N where A is NxN" << endl;
            return 1;
        }
        X = linspace_nd(cn, cd);
        cout << "Tensor coordinate matrix of size " << cn << "^" << cd << " built" << endl;
    } else if(geo_file) {
        X = mmio::dense_mmread<double>(coordinates);
        cout << "Coordinate file " << X.rows() << "x" << X.cols() << " loaded from " << coordinates << endl;
        if(X.cols() != N) {
            cout << "Error: coordinate file should hold a matrix of size d x N" << endl;
        }
    }

    // Vector to preserve (maybe)
    MatrixXd phi = MatrixXd::Ones(N,1);

    // Initialize a tree
    Tree t(nlevels);
    // Basic options
    t.set_verb(verb);
    t.set_symmetry(symmetric);
    t.set_tol(tol);
    t.set_skip(skip);
    t.set_scale(scale);
    t.set_ortho(ortho);
    t.set_preserve(preserve);
    t.set_use_geo(geo);
    if(geo) {
        t.set_Xcoo(&X);
    }
    t.set_scaling_kind(sk);
    t.set_use_sparsify(use_want_sparsify);
    t.set_phi(&phi);
    t.set_monitor_condition_pivots(monitor_cond);
    t.print_summary();

    // Partition
    timer tpart_0 = wctime();
    SpMat AAT = symmetric_graph(A);
    t.partition(AAT);
    timer tpart = wctime();

    // Output some diagnostics
    if(result.count("output_ordering")) {
        string ordering_fn = result["output_ordering"].as<string>();
        cout << "Writing ordering to " << ordering_fn << endl;
        ofstream ordering_s(ordering_fn);
        auto clusters_ordering = t.get_clusters_levels();
        ordering_s << N << " " << X.rows() << " " << clusters_ordering.size() << "\n";
        for(int i = 0; i < N; i++) {
            stringstream ss;
            for(int d = 0; d < X.rows(); d++) {
                ss << X(d,i) << " ";
            }
            for(int l = 0; l < clusters_ordering.size(); l++) {
                ClusterID id = clusters_ordering[l][i];
                ss << id.self.lvl << " " << id.self.sep << " " << id.l.lvl << " " << id.l.sep << " " << id.r.lvl << " " << id.r.sep << " ";
            }
            ordering_s << ss.str() << "\n";
        }
        ordering_s.close();
    }

    // Assembly
    timer tass_0 = wctime();
    t.assemble(A);
    timer tass = wctime();

    // Factorize    
    timer tfact_0 = wctime();
    int err = t.factorize();
    timer tfact = wctime();

    t.print_log();
    cout << "Timings [s.]:" << endl;
    cout << "<<<<tpart=" << elapsed(tpart_0, tpart) << endl;
    cout << "<<<<tassm=" << elapsed(tass_0,  tass)  << endl;
    cout << "<<<<tfact=" << elapsed(tfact_0, tfact) << endl;
    cout << "<<<<stop=" << t.log[nlevels-2].dofs_left_elim << endl;
    cout << "<<<<error=" << err << endl;
    // Run one solve
    {
        matrix_hash<VectorXd> hash;
        // Random b
        {
            
            VectorXd b = random(N, 2019);
            VectorXd x = b;
            timer tsolv_0 = wctime();
            t.solve(x);
            timer tsolv = wctime();
            cout << "<<<<tsolv=" << elapsed(tsolv_0, tsolv) << endl;
            cout << "One-time solve (Random b):" << endl;            
            cout << "<<<<|Ax-b|/|b| : " << (A*x-b).norm() / b.norm() << endl;
            cout << "<<<<hash(b) : "    << hash(b) << endl;
            cout << "<<<<hash(x) : "    << hash(x) << endl;
        }
        // Random x
        {
            VectorXd xtrue = random(N, 2019);
            VectorXd b = A*xtrue;
            VectorXd x = b;
            t.solve(x);
            cout << "One-time solve (Random x):" << endl;
            cout << "<<<<|Ax-b|/|b| : "    << (A*x-b).norm() / b.norm() << endl;
            cout << "<<<<|x-xtrue|/|x| : " << (x-xtrue).norm() / xtrue.norm() << endl;
            cout << "<<<<hash(xtrue) : "   << hash(xtrue) << endl;
            cout << "<<<<hash(b) : "       << hash(b) << endl;
            cout << "<<<<hash(x) : "       << hash(x) << endl;  
        }
    }
    // Solve
    if(err == 0) {
        VectorXd x = VectorXd::Zero(N);
        VectorXd b = VectorXd::Random(N);
        if(useCG) {            
            timer cg0 = wctime();
            int iter = cg(A, b, x, t, iterations, 1e-12, verb);
            timer cg1 = wctime();
            cout << "CG: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
            cout << "  CG: " << elapsed(cg0, cg1) << " s." << endl;
            cout << "<<<<CG=" << iter << endl;
        } else if(useGMRES) {
            timer gmres0 = wctime();
            int iter = gmres(A, b, x, t, iterations, iterations, 1e-12, verb);
            timer gmres1 = wctime();
            cout << "GMRES: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
            cout << "  GMRES: " << elapsed(gmres0, gmres1) << " s." << endl;
            cout << "<<<<GMRES=" << iter << endl;
        }
    }
    if(preserve)  {
        cout << "Checking preservation" << endl;
        VectorXd b = A*phi.col(0);
        VectorXd x = b;
        t.solve(x);
        cout << "Residual |Ax-b|/|b| with b = A*phi: " << (A*x-b).norm() / b.norm() << endl;
    }
    return err;
}
