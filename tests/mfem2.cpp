// This code comes from MFEM
// Published under LGPL v2.1 license
// See https://github.com/mfem/mfem/blob/master/LICENSE for detail


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "tree.h"
#include "util.h"
#include "is.h"
#include "mfem_util.h"

using namespace std;
using namespace mfem;
using namespace Eigen;

int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
    const char *mesh_file = "../data/beam-tri.mesh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 0;
    double tol = 1e-2;
    int skip = 4;
    bool preserve = false;
    int target = 5000;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    // args.AddOption(&order, "-o", "--order",
    //                "Finite element order (polynomial degree).");
    // args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
    //                "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&tol, "-t", "--tolerance", "RRQR tolerance");
    args.AddOption(&skip, "-s", "--skip", "Skip some ND levels");
    args.AddOption(&preserve, "-p", "--preserve", "-no-p", "--no-preserve", "Wether or not to use piecewise const+linear preservation");
    args.AddOption(&target, "-n", "--n", "Target element count (no greater than)");

    args.Parse();
    if (!args.Good())
    {
       args.PrintUsage(cout);
       return 1;
    }
    args.PrintOptions(cout);

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral or hexahedral elements with the same code.
    mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    if (mesh->attributes.Max() < 2 || mesh->bdr_attributes.Max() < 2)
    {
        cerr << "\nInput mesh should have at least two materials and "
            << "two boundary attributes! (See schematic in ex2.cpp)\n"
            << endl;
        return 3;
    }

    cout << mesh->attributes.Max() << endl;
    cout << mesh->bdr_attributes.Max() << endl;

    cout << "Mesh attributes" << endl;
    mesh->attributes.Print();
    cout << "Bndry attributes" << endl;
    mesh->bdr_attributes.Print();

    // 3. Select the order of the finite element discretization space. For NURBS
    //    meshes, we increase the order by degree elevation.
    if (mesh->NURBSext)
    {
       mesh->DegreeElevate(order, order);
    }

    // 4. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than 5,000
    //    elements.
    {
        int ref_levels =
            (int)floor(log(double(target)/mesh->GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh->UniformRefinement();
        }
   }

    int N = mesh->GetNV();
    cout << "Dimension? " << dim << endl;
    cout << "Vertices? " << N << endl;

    Eigen::MatrixXd Xcoo(dim, N * dim);
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < dim; j++) { // The three spatial dimensions
            for (int k = 0; k < dim; k++) { // The three PDEs components
                Xcoo(j, i + k * N) = mesh->GetVertex(i)[j];
            }
        }
    }

    cout << "Coo matrix" << endl;
    cout << Xcoo.leftCols(10).transpose() << endl;
    int lvl = (int)ceil(log(N * dim / 64.0)/log(2.0));

    // 5. Define a finite element space on the mesh. Here we use vector finite
    //    elements, i.e. dim copies of a scalar finite element space. The vector
    //    dimension is specified by the last argument of the FiniteElementSpace
    //    constructor. For NURBS meshes, we use the (degree elevated) NURBS space
    //    associated with the mesh nodes.
    mfem::FiniteElementCollection *fec;
    mfem::FiniteElementSpace *fespace;
    if (mesh->NURBSext)
    {
       fec = NULL;
       fespace = mesh->GetNodes()->FESpace();
    }
    else
    {
       fec = new mfem::H1_FECollection(order, dim);
       fespace = new mfem::FiniteElementSpace(mesh, fec, dim);
    }
    cout << "Number of finite element unknowns: " << fespace->GetTrueVSize()
         << endl << "Assembling: " << flush;

    // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking only
    //    boundary attribute 1 from the mesh as essential and converting it to a
    //    list of true dofs.
    mfem::Array<int> ess_tdof_list, ess_bdr(mesh->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1;
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // 7. Set up the linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system. In this case, b_i equals the boundary integral
    //    of f*phi_i where f represents a "pull down" force on the Neumann part
    //    of the boundary and phi_i are the basis functions in the finite element
    //    fespace. The force is defined by the VectorArrayCoefficient object f,
    //    which is a vector of Coefficient objects. The fact that f is non-zero
    //    on boundary attribute 2 is indicated by the use of piece-wise constants
    //    coefficient for its last component.
    mfem::VectorArrayCoefficient f(dim);
    for (int i = 0; i < dim-1; i++)
    {
        f.Set(i, new ConstantCoefficient(0.0));
    }
    {
        mfem::Vector pull_force(mesh->bdr_attributes.Max());
        pull_force = 0.0;
        pull_force(1) = -1.0e-2;
        f.Set(dim-1, new PWConstCoefficient(pull_force));
    }


    mfem::LinearForm *b = new mfem::LinearForm(fespace);
    b->AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(f));
    cout << "r.h.s. ... " << flush;
    b->Assemble();

    // 8. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    mfem::GridFunction x(fespace);
    x = 0.0;

    // 9. Set up the bilinear form a(.,.) on the finite element space
    //    corresponding to the linear elasticity integrator with piece-wise
    //    constants coefficient lambda and mu.
    //    [ lambda=mu=high | lambda=mu=low]
    Vector lambda(mesh->attributes.Max());
    lambda = 1.0;
    lambda(0) = lambda(1)*50;
    PWConstCoefficient lambda_func(lambda);
    Vector mu(mesh->attributes.Max());
    mu = 1.0;
    mu(0) = mu(1)*50;
    PWConstCoefficient mu_func(mu);

    BilinearForm *a = new BilinearForm(fespace);
    a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func,mu_func));

    // 10. Assemble the bilinear form and the corresponding linear system,
    //     applying any necessary transformations such as: eliminating boundary
    //     conditions, applying conforming constraints for non-conforming AMR,
    //     static condensation, etc.
    cout << "matrix ... " << flush;
    if (static_cond) { a->EnableStaticCondensation(); }
    a->Assemble();

    mfem::SparseMatrix A;
    mfem::Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
    cout << "done." << endl;

    cout << "Size of linear system: " << A.Height() << endl;

    // Run Algo
    SpMat A2 = mfem2eigen(A); 
    VectorXd b2 = mfem2eigen(B);
    VectorXd x2 = VectorXd::Zero(B.Size());
    MatrixXd phi;
    assert(dim == 3);
    assert(N % 3 == 0);
    if(preserve) {
        int N = A2.rows();
        int N3 = N/3;
        phi = MatrixXd::Zero(N, 12);
        // piecewise 1
        for(int i = 0; i < N/3; i++) {
            phi(     i,0) = 1;
            phi(  N3+i,1) = 1;
            phi(2*N3+i,2) = 1;
        }
        // piecewise x
        for(int i = 0; i < N/3; i++) {
            phi(     i,3) = Xcoo(0,i);
            phi(  N3+i,4) = Xcoo(0,i);
            phi(2*N3+i,5) = Xcoo(0,i);
        }
        // piecewise y
        for(int i = 0; i < N/3; i++) {
            phi(     i,6) = Xcoo(1,i);
            phi(  N3+i,7) = Xcoo(1,i);
            phi(2*N3+i,8) = Xcoo(1,i);
        }
        // piecewise z
        for(int i = 0; i < N/3; i++) {
            phi(     i,9)  = Xcoo(2,i);
            phi(  N3+i,10) = Xcoo(2,i);
            phi(2*N3+i,11) = Xcoo(2,i);
        }
    }
    if(preserve) {
        tol = 10.0; // Adaptive = false -> only preservation
    }
    Tree t = Tree(lvl);
    t.set_tol(tol);
    t.set_skip(skip);
    t.set_preserve(preserve);
    t.set_phi(&phi);
    t.set_use_geo(true);
    t.set_Xcoo(&Xcoo);
    timer t0 = wctime();
    t.partition(A2);
    t.assemble(A2);
    int err = t.factorize();
    timer t3 = wctime();
    cout << ">>>>t_F=" << elapsed(t0, t3) << endl;
    t.print_log();
    if(err == 0) {
        int iter = cg(A2, b2, x2, t, 500, 1e-12, true);
        timer t4 = wctime();
        cout << "CG: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A2*x2-b2).norm() / b2.norm() << endl;
        cout << ">>>>t_S=" << elapsed(t3, t4) << endl;
        cout << ">>>>CG=" << iter << endl;
        eigen2mfem(x2, X);
        if(preserve) {
            for(int i = 0; i < phi.cols(); i++) {
                VectorXd b3 = A2 * phi.col(i);
                VectorXd x3 = b3;
                t.solve(x3);
                double error = (A2*x3-b3).norm() / b3.norm();
                cout << ">>>>Preservation error=" << error << endl;
            }
        }
    } else {
        cout << ">>>>t_S=0" << endl;
        cout << ">>>>NOT SPD" << endl;
    }

    // 12. Recover the solution as a finite element grid function.
    a->RecoverFEMSolution(X, *b, x);

    // 13. For non-NURBS meshes, make the mesh curved based on the finite element
    //     space. This means that we define the mesh elements through a fespace
    //     based transformation of the reference element. This allows us to save
    //     the displaced mesh as a curved mesh when using high-order finite
    //     element displacement field. We assume that the initial mesh (read from
    //     the file) is not higher order curved mesh compared to the chosen FE
    //     space.
    if (!mesh->NURBSext)
    {
       mesh->SetNodalFESpace(fespace);
    }

    // 14. Save the displaced mesh and the inverted solution (which gives the
    //     backward displacements to the original grid). This output can be
    //     viewed later using GLVis: "glvis -m displaced.mesh -g sol.gf".
    {
        mfem::GridFunction *nodes = mesh->GetNodes();
        *nodes += x;
        x *= -1;
        ofstream mesh_ofs("displaced.mesh");
        mesh_ofs.precision(8);
        mesh->Print(mesh_ofs);
        ofstream sol_ofs("sol.gf");
        sol_ofs.precision(8);
        x.Save(sol_ofs);
    }

    // 15. Send the above data by socket to a GLVis server. Use the "n" and "b"
    //     keys in GLVis to visualize the displacements.
    if (visualization)
    {
       char vishost[] = "localhost";
       int  visport   = 19916;
       mfem::socketstream sol_sock(vishost, visport);
       sol_sock.precision(8);
       sol_sock << "solution\n" << *mesh << x << flush;
    }

    // 16. Free the used memory.
    delete a;
    delete b;
    if (fec)
    {
       delete fespace;
       delete fec;
    }
    delete mesh;

    return 0;
}
