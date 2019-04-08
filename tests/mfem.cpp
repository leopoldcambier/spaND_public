// This code comes from MFEM
// Published under LGPL v2.1 license
// See https://github.com/mfem/mfem/blob/master/LICENSE for detail
//
//                                MFEM Example 1
//
// Compile with: make ex1
//
// Sample runs:  ex1 -m ../data/square-disc.mesh
//               ex1 -m ../data/star.mesh
//               ex1 -m ../data/escher.mesh
//               ex1 -m ../data/fichera.mesh
//               ex1 -m ../data/square-disc-p2.vtk -o 2
//               ex1 -m ../data/square-disc-p3.mesh -o 3
//               ex1 -m ../data/square-disc-nurbs.mesh -o -1
//               ex1 -m ../data/disc-nurbs.mesh -o -1
//               ex1 -m ../data/pipe-nurbs.mesh -o -1
//               ex1 -m ../data/star-surf.mesh
//               ex1 -m ../data/square-disc-surf.mesh
//               ex1 -m ../data/inline-segment.mesh
//               ex1 -m ../data/amr-quad.mesh
//               ex1 -m ../data/amr-hex.mesh
//               ex1 -m ../data/fichera-amr.mesh
//               ex1 -m ../data/mobius-strip.mesh
//               ex1 -m ../data/mobius-strip.mesh -o -1 -sc
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 1 with homogeneous Dirichlet boundary conditions.
//               Specifically, we discretize using a FE space of the specified
//               order, or if order < 1 using an isoparametric/isogeometric
//               space (i.e. quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
//               The example highlights the use of mesh refinement, finite
//               element grid functions, as well as linear and bilinear forms
//               corresponding to the left-hand side and right-hand side of the
//               discrete linear system. We also cover the explicit elimination
//               of essential boundary conditions, static condensation, and the
//               optional connection to the GLVis tool for visualization.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "tree.h"
#include "util.h"
#include "is.h"
#include "mfem_util.h"

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
    const char *mesh_file = "../data/star.mesh";
    int order = 1;
    bool static_cond = false;
    bool visualization = 1;

    mfem::OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                    "Finite element order (polynomial degree) or -1 for"
                    " isoparametric space.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                    "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                    "--no-visualization",
                    "Enable or disable GLVis visualization.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        return 1;
    }
    args.PrintOptions(cout);

    // 2. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
    //    the same code.
    mfem::Mesh *mesh = new mfem::Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 3. Refine the mesh to increase the resolution. In this example we do
    //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    //    largest number that gives a final mesh with no more than 50,000
    //    elements.
    {
        int ref_levels =
            (int)floor(log(50000./mesh->GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh->UniformRefinement();
        }
    }

    // 4. Define a finite element space on the mesh. Here we use continuous
    //    Lagrange finite elements of the specified order. If order < 1, we
    //    instead use an isoparametric/isogeometric space.
    mfem::FiniteElementCollection *fec;
    if (order > 0)
    {
        fec = new mfem::H1_FECollection(order, dim);
    }
    else if (mesh->GetNodes())
    {
        fec = mesh->GetNodes()->OwnFEC();
        cout << "Using isoparametric FEs: " << fec->Name() << endl;
    }
    else
    {
        fec = new mfem::H1_FECollection(order = 1, dim);
    }
    mfem::FiniteElementSpace *fespace = new mfem::FiniteElementSpace(mesh, fec);
    cout << "Number of finite element unknowns: "
            << fespace->GetTrueVSize() << endl;

    // 5. Determine the list of true (i.e. conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined by marking all
    //    the boundary attributes from the mesh as essential (Dirichlet) and
    //    converting them to a list of true dofs.
    mfem::Array<int> ess_tdof_list;
    if (mesh->bdr_attributes.Size())
    {
        mfem::Array<int> ess_bdr(mesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // 6. Set up the linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system, which in this case is (1,phi_i) where phi_i are
    //    the basis functions in the finite element fespace.
    mfem::LinearForm *b = new mfem::LinearForm(fespace);
    mfem::ConstantCoefficient one(1.0);
    b->AddDomainIntegrator(new mfem::DomainLFIntegrator(one));
    b->Assemble();

    // 7. Define the solution vector x as a finite element grid function
    //    corresponding to fespace. Initialize x with initial guess of zero,
    //    which satisfies the boundary conditions.
    mfem::GridFunction x(fespace);
    x = 0.0;

    // 8. Set up the bilinear form a(.,.) on the finite element space
    //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
    //    domain integrator.
    mfem::BilinearForm *a = new mfem::BilinearForm(fespace);
    a->AddDomainIntegrator(new mfem::DiffusionIntegrator(one));

    // 9. Assemble the bilinear form and the corresponding linear system,
    //    applying any necessary transformations such as: eliminating boundary
    //    conditions, applying conforming constraints for non-conforming AMR,
    //    static condensation, etc.
    if (static_cond) { a->EnableStaticCondensation(); }
    a->Assemble();

    mfem::SparseMatrix A;
    mfem::Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);

    cout << "Size of linear system: " << A.Height() << endl;

    // 9b. Use spaND
    // Run Algo
    MatrixXd Xcoo(3, mesh->GetNV());
    for(int i = 0; i < mesh->GetNV(); i++) {
        Xcoo(0, i) = mesh->GetVertex(i)[0];
        Xcoo(1, i) = mesh->GetVertex(i)[1];
        Xcoo(2, i) = mesh->GetVertex(i)[2];
    }
   
    SpMat A2 = mfem2eigen(A); 
    VectorXd b2 = mfem2eigen(B);

    cout << b2.topRows(150) << endl;
    cout << A2.block(0, 0, 150, 150) << endl;

    VectorXd x2 = VectorXd::Zero(B.Size());
    
    Tree t = Tree(11);
    t.set_tol(1e-2);
    t.set_skip(2);
    t.set_use_geo(true);
    t.set_Xcoo(&Xcoo);
    t.partition(A2);
    t.assemble(A2);
    int err = t.factorize();
    assert(err == 0);
    int iter = cg(A2, b2, x2, t, 500, 1e-12, true);
    cout << "CG: #iterations: " << iter << ", residual |Ax-b|/|b|: " << (A2*x2-b2).norm() / b2.norm() << endl;
    eigen2mfem(x2, X);

    // 11. Recover the solution as a finite element grid function.
    a->RecoverFEMSolution(X, *b, x);

    // 12. Save the refined mesh and the solution. This output can be viewed later
    //     using GLVis: "glvis -m refined.mesh -g sol.gf".
    ofstream mesh_ofs("refined.mesh");
    mesh_ofs.precision(8);
    mesh->Print(mesh_ofs);
    ofstream sol_ofs("sol.gf");
    sol_ofs.precision(8);
    x.Save(sol_ofs);

    // 13. Send the solution by socket to a GLVis server.
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        mfem::socketstream sol_sock(vishost, visport);
        sol_sock.precision(8);
        sol_sock << "solution\n" << *mesh << x << flush;
    }

    // 14. Free the used memory.
    delete a;
    delete b;
    delete fespace;
    if (order > 0) { delete fec; }
    delete mesh;

    return 0;
}
