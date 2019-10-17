#include <iostream>
#include <fstream>
#include <sstream>
#include "spaND.h"
#include "cxxopts.hpp"
#include "mmio.hpp"

using namespace Eigen;
using namespace std;
using namespace spaND;

int main(int argc, char* argv[]) {

    cxxopts::Options options("spaND", "General driver for spaND. Load a matrix, an optional coordinate file. Then, assemble the matrix, partition it, factor it and solve Ax=b using GMRES.");
    options.add_options()
        // Generic
        ("help", "Print help")
        ("verbose", "Verbose mode", cxxopts::value<bool>()->default_value("true"))
        // General
        ("m,matrix", "Matrix MM coordinate file (mandatory)", cxxopts::value<string>())
        ("l,lvl", "# of levels (mandatory). If 0, uses lvl=ceil(log2(N/64))+1 instead", cxxopts::value<int>())
        ("symm_kind", "Wether the matrix is SPD, symmetric SYM or general GEN", cxxopts::value<string>()->default_value("SPD"))
        ("part_kind", "Partition kind (modified ND - MND) or recursive bissection based (RB)", cxxopts::value<string>()->default_value("MND"))
        ("lorasp", "Wether to use LoRaSp (true) or spaND (false) (default: false)", cxxopts::value<bool>()->default_value("false"))
        // Geometry
        ("coordinates", "Coordinates MM array file. If provided, will do a geometric partitioning.", cxxopts::value<string>())
        ("n,coordinates_n", "If provided with -n, will use a tensor n^d & geometric partitioning. Overwrites --coordinates", cxxopts::value<int>()->default_value("-1"))
        ("d,coordinates_d", "If provided with -d, will use a tensor n^d & geometric partitioning. Overwrites --coordinates", cxxopts::value<int>()->default_value("-1"))
        // Compression choices
        ("t,tol", "Tolerance", cxxopts::value<double>()->default_value("1e-1"))            
        ("skip", "Skip sparsification", cxxopts::value<int>()->default_value("0"))        
        ("preserve", "Wether to preserve the 1 vector or not (default: false)", cxxopts::value<bool>()->default_value("false"))
        ("scaling_kind", "The scaling kind, LLT, EVD, SVD, PLU or PLUQ", cxxopts::value<string>()->default_value("LLT"))
        ("use_want_sparsify", "Wether to use want_sparsify (true) or not (default: true)", cxxopts::value<bool>()->default_value("true"))
        // Iterative method
        ("solver","Wether to use CG, GMRES or IR", cxxopts::value<string>()->default_value("CG"))
        ("i,iterations","Iterative solver iterations", cxxopts::value<int>()->default_value("100"))
        ("solver_tol","Iterative solver tolerance", cxxopts::value<double>()->default_value("1e-12"))
        // Problem transformation, to try some functions of A
        ("flip_sign", "Wether to solve with -A (true) or not (default: false)", cxxopts::value<bool>()->default_value("false"))
        ("rationale", "Solve with a_0 I + a_1 A + a_2 A^2 ... instead of A", cxxopts::value<vector<double>>())
        // Output more data
        ("clustering", "Output clustering (before any merging) in text file", cxxopts::value<string>())
        ("clusters", "Output clusters metadata in text file", cxxopts::value<string>())
        ("merging", "Output merging process in text file", cxxopts::value<string>())
        ("stats", "Output cluster data throughout elimination in text file", cxxopts::value<string>())
        ("monitor_cond", "Monitor condition number of pivots", cxxopts::value<bool>()->default_value("false"))
        ("monitor_unsym", "Monitor |A-AT|/|A|", cxxopts::value<bool>()->default_value("false"))
        ("monitor_Rdiag", "Monitor the diagonal of R (~ svds) and save to --stats", cxxopts::value<bool>()->default_value("false"))
        ("print_clusters_hierarchy", "Print the clusters hierarchy (lots of output!)", cxxopts::value<bool>()->default_value("false"))
        ("write_log_flops", "Print the flops log to file", cxxopts::value<string>())
        // Detail
        ("n_threads", "Number of threads", cxxopts::value<int>()->default_value("1"))
        // For indefinite saddle point systems
        ("sp_mid", "Saddle point middle point", cxxopts::value<int>()->default_value("-1"))
        ("sp_eps", "Saddle point epsilon point", cxxopts::value<double>()->default_value("0.0"))       
        ;
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help({"", "Group"}) << endl;
        exit(0);
    }

    if ( (!result.count("matrix")) || (!result.count("lvl")) ) {
        std::cout << "--matrix and --lvl are mandatory" << endl;
        exit(0);
    }
    string matrix = result["matrix"].as<string>();
    string coordinates;
    int cn = result["coordinates_n"].as<int>();
    int cd = result["coordinates_d"].as<int>();
    std::cout << "<<<<cn=" << cn << std::endl;
    std::cout << "<<<<cd=" << cd << std::endl;
    bool geo_file = (result.count("coordinates") > 0);
    if ( (cn == -1 && cd >= 0) || (cd == -1 && cn >= 0) ) {
        std::cout << "cn and cd should be both provided, or none should be provided" << endl;
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
    bool preserve = result["preserve"].as<bool>();
    string s0 = result["symm_kind"].as<string>();
    SymmKind symmk = (s0 == "SPD" ? SymmKind::SPD :
                     (s0 == "SYM" ? SymmKind::SYM :
                     (SymmKind::GEN)));
    bool verb = result["verbose"].as<bool>();
    string s1 = result["scaling_kind"].as<string>();
    ScalingKind sk = (s1 == "LLT"  ? ScalingKind::LLT  :
                     (s1 == "PLU"  ? ScalingKind::PLU  :
                     (s1 == "PLUQ" ? ScalingKind::PLUQ :
                     (s1 == "SVD"  ? ScalingKind::SVD  :
                     (s1 == "LDLT" ? ScalingKind::LDLT :
                     (ScalingKind::EVD))))));
    PartKind pk = (result["part_kind"].as<string>() == "MND" ? PartKind::MND : PartKind::RB);
    bool useCG = (result["solver"].as<string>() == "CG");
    bool useGMRES = (result["solver"].as<string>() == "GMRES");
    bool useIR = (result["solver"].as<string>() == "IR");
    int iterations = result["iterations"].as<int>();
    double solver_tol = result["solver_tol"].as<double>();
    if(!useCG && !useGMRES && !useIR) {
        std::cout << "Wrong solver picked. Should be CG, IR or GMRES" << endl;
        return 1;
    }
    bool use_want_sparsify = result["use_want_sparsify"].as<bool>();
    int n_threads = result["n_threads"].as<int>();
    (void) n_threads;
    bool flip_sign = result["flip_sign"].as<bool>();
    vector<double> rational_coeffs;
    if (result.count("rationale")) {
        for(auto v: result["rationale"].as<vector<double>>()) {
            rational_coeffs.push_back(v);
        }
        std::cout << "Using rational coeffs: ";
        for(int i = 0; i < rational_coeffs.size(); i++) {
            std::cout << rational_coeffs[i] << " A^" << i << " + ";
        }
        std::cout << endl;
    }
    bool monitor_cond = result["monitor_cond"].as<bool>();
    bool monitor_unsymmetry = result["monitor_unsym"].as<bool>();
    bool monitor_Rdiag = result["monitor_Rdiag"].as<bool>();
    bool use_lorasp = result["lorasp"].as<bool>();
    
    // Load a matrix
    SpMat A = mmio::sp_mmread<double,int>(matrix);
    int N = A.rows();
    std::cout << "<<<<matrix=" << matrix << std::endl;
    std::cout << "Matrix " << N << "x" << N << " loaded from " << matrix << endl;
    
    if(nlevels <= 0) {
        std::cout << "nlevels = " << nlevels << " -> using log(N/64)/log(2) - |nlevels| instead" << std::endl;
        nlevels = round(log((double)N/64.0)/log(2.0)) + nlevels;
    }

    // Modifiy A for some cases/experiments
    if(flip_sign) {
        A = -A;
        std::cout << "Solving with -A instead" << endl;
    } else if(rational_coeffs.size() > 0) {
        std::cout << "Solving with rational coefficients instead" << endl;
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

    // Matrix unsymmetry
    SpMat AT = A.transpose();
    printf("|A - AT|/|A| = %e\n", (A - AT).norm() / A.norm());

    // Saddle point modifications
    SpMat Aprec = A;
    bool is_saddlepoint = result.count("sp_mid") && result.count("sp_eps");
    if(is_saddlepoint) {
        int sp_mid = result["sp_mid"].as<int>();
        double sp_eps = result["sp_eps"].as<double>();
        std::cout << "A[~sp_mid,~sp_mid]:\n" << A.block(sp_mid-2, sp_mid-2, 5, 5) << endl;
        std::cout << "System is saddle point" << endl;
        std::cout << "Rebuilding matrix by adding " << sp_eps << " on diagonal at entries [" << sp_mid << ";" << N << "[" << endl;
        vector<Triplet<double>> tripletList;
        for(int i = sp_mid; i < N; i++) { tripletList.push_back({i,i,sp_eps}); }
        for(int k = 0; k < A.outerSize(); ++k) {
            for (SpMat::InnerIterator it(A,k); it; ++it) { tripletList.push_back({int(it.row()),int(it.col()),it.value()}); }
        }
        Aprec = SpMat(N,N);
        Aprec.setFromTriplets(tripletList.begin(), tripletList.end());
    }
    // std::cout << "A:\n" << A.block(0, 0, 10, 10) << endl;
    // std::cout << "Aprec:\n" << Aprec.block(0, 0, 10, 10) << endl;

    // Load coordinates ?
    MatrixXd X;
    if(geo_tensor) {
        if(pow(cn, cd) != N) {
            std::cout << "Error: cn and cd where both provided, but cn^cd != N where A is NxN" << endl;
            return 1;
        }
        X = linspace_nd(cn, cd);
        std::cout << "Tensor coordinate matrix of size " << cn << "^" << cd << " built" << endl;
    } else if(geo_file) {
        X = mmio::dense_mmread<double>(coordinates);
        std::cout << "Coordinate file " << X.rows() << "x" << X.cols() << " loaded from " << coordinates << endl;
        if(X.cols() != N) {
            std::cout << "Error: coordinate file should hold a matrix of size d x N" << endl;
        }
    }

    // Vector to preserve (maybe)
    MatrixXd phi = MatrixXd::Ones(N,1);

    // Basic options
    Tree t(nlevels);    
    t.set_verb(verb);
    t.set_symm_kind(symmk);
    t.set_tol(tol);
    t.set_skip(skip);
    t.set_preserve(preserve);
    t.set_use_geo(geo);
    if(geo) {
        t.set_Xcoo(&X);
    }
    t.set_scaling_kind(sk);
    t.set_part_kind(pk);
    t.set_use_sparsify(use_want_sparsify);
    t.set_phi(&phi);
    t.set_monitor_condition_pivots(monitor_cond);
    t.set_monitor_unsymmetry(monitor_unsymmetry);
    t.set_monitor_Rdiag(monitor_Rdiag);
    if(result.count("write_log_flops")) t.set_monitor_flops(true);

    // Print basic parameters
    std::cout << "<<<<N=" << N << endl;
    std::cout << "<<<<nlevels=" << nlevels << endl;
    std::cout << "<<<<tol=" << tol << endl;
    std::cout << "<<<<skip=" << skip << endl;
    std::cout << "<<<<preserve=" << preserve << endl;
    std::cout << "<<<<lorasp=" << use_lorasp << endl;

    // Partition
    timer tpart_0 = wctime();
    SpMat AAT = symmetric_graph(Aprec);
    if(use_lorasp) {
        t.partition_lorasp(AAT);
    } else {
        t.partition(AAT);
    }
    timer tpart = wctime();

    // Clustering
    if(result.count("clustering")) {
        string clustering_fn = result["clustering"].as<string>();        
        std::cout << "Writing clustering to " << clustering_fn << endl;
        write_clustering(t, X, clustering_fn);
    }
    // Hierarchy
    if(result.count("merging")) {
        string merging_fn = result["merging"].as<string>();
        std::cout << "Writing merging to " << merging_fn << endl;
        write_merging(t, merging_fn);
    }
    // Cluster metadata
    if(result.count("clusters")) {
        string clusters_fn = result["clusters"].as<string>();
        std::cout << "Writing clusters to " << clusters_fn << endl;
        write_clusters(t, clusters_fn);
    }

    // Assembly
    timer tass_0 = wctime();
    t.assemble(Aprec);
    timer tass = wctime();

    if(result.count("print_clusters_hierarchy") && result["print_clusters_hierarchy"].as<bool>()) {
        t.print_clusters_hierarchy();
    }

    // Factorize    
    timer tfact_0 = wctime();
    try {
        if(use_lorasp) {
            t.factorize_lorasp();
        } else {
            t.factorize();
        }
    } catch (exception& ex) {
        cout << ex.what();
        cout << "<<<<FAILED" << endl;
        exit(1);
    }
    timer tfact = wctime();

    t.print_log();
    std::cout << "Timings [s.]:" << endl;
    std::cout << "<<<<tpart=" << elapsed(tpart_0, tpart) << endl;
    std::cout << "<<<<tassm=" << elapsed(tass_0,  tass)  << endl;
    std::cout << "<<<<tfact=" << elapsed(tfact_0, tfact) << endl;
    std::cout << "<<<<stop="  << t.get_stop() << endl;
    std::cout << "<<<<nnzfact=" << t.nnz() << endl;

    if(result.count("write_log_flops")) {
        string log_flops_fn = result["write_log_flops"].as<string>();
        std::cout << "Writing flops log to " << log_flops_fn << endl;
        write_log_flops(t, log_flops_fn);
    }

    // Sizes & ranks
    if(result.count("stats")) {
        string stats_fn = result["stats"].as<string>();
        std::cout << "Writing cluster stats to " << stats_fn << endl;        
        write_stats(t, stats_fn);
    }

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
            std::cout << "<<<<tsolv=" << elapsed(tsolv_0, tsolv) << endl;
            std::cout << "One-time solve (Random b):" << endl;            
            std::cout << "<<<<|Ax-b|/|b| : " << (A*x-b).norm() / b.norm() << endl;
            std::cout << "<<<<hash(b) : "    << hash(b) << endl;
            std::cout << "<<<<hash(x) : "    << hash(x) << endl;
        }
        // Random x
        {
            VectorXd xtrue = random(N, 2019);
            VectorXd b = A*xtrue;
            VectorXd x = b;
            t.solve(x);
            std::cout << "One-time solve (Random x):" << endl;
            std::cout << "<<<<|Ax-b|/|b| : "    << (A*x-b).norm() / b.norm() << endl;
            std::cout << "<<<<|x-xtrue|/|x| : " << (x-xtrue).norm() / xtrue.norm() << endl;
            std::cout << "<<<<hash(xtrue) : "   << hash(xtrue) << endl;
            std::cout << "<<<<hash(b) : "       << hash(b) << endl;
            std::cout << "<<<<hash(x) : "       << hash(x) << endl;  
        }
    }
    // Solve
    {
        VectorXd x = VectorXd::Zero(N);
        VectorXd b = VectorXd::Random(N);
        if(useCG) {            
            timer cg0 = wctime();
            int iter = cg(A, b, x, t, iterations, solver_tol, verb);
            timer cg1 = wctime();
            std::cout << "CG: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
            std::cout << "  CG: " << elapsed(cg0, cg1) << " s." << endl;
            std::cout << "<<<<CG=" << iter << endl;
            std::cout << "<<<<tCG=" << elapsed(cg0, cg1) << endl;
        } else if(useGMRES) {
            timer gmres0 = wctime();
            int iter = gmres(A, b, x, t, iterations, iterations, solver_tol, verb);
            timer gmres1 = wctime();
            std::cout << "GMRES: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
            std::cout << "  GMRES: " << elapsed(gmres0, gmres1) << " s." << endl;
            std::cout << "<<<<GMRES=" << iter << endl;
            std::cout << "<<<<tGMRES=" << elapsed(gmres0, gmres1) << endl;
        } else if(useIR) {
            timer ir0 = wctime();
            int iter = ir(A, b, x, t, iterations, solver_tol, verb);
            timer ir1 = wctime();
            std::cout << "IR: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A*x-b).norm() / b.norm() << endl;
            std::cout << "  IR: " << elapsed(ir0, ir1) << " s." << endl;
            std::cout << "<<<<IR=" << iter << endl;
            std::cout << "<<<<tIR=" << elapsed(ir0, ir1) << endl;
        }
    }
    if(preserve)  {
        std::cout << "Checking preservation" << endl;
        VectorXd b = A*phi.col(0);
        VectorXd x = b;
        t.solve(x);
        std::cout << "Residual |Ax-b|/|b| with b = A*phi: " << (A*x-b).norm() / b.norm() << endl;
    }
    return 0;
}
