// This code comes from MFEM
// Published under LGPL v2.1 license
// See https://github.com/mfem/mfem/blob/master/LICENSE for detail
// https://github.com/mfem/mfem/blob/master/examples/ex5.cpp
//
//                                MFEM Example 5
//
// Compile with: make ex5
//
// Sample runs:  ex5 -m ../data/square-disc.mesh
//               ex5 -m ../data/star.mesh
//               ex5 -m ../data/beam-tet.mesh
//               ex5 -m ../data/beam-hex.mesh
//               ex5 -m ../data/escher.mesh
//               ex5 -m ../data/fichera.mesh
//
// Description:  This example code solves a simple 2D/3D mixed Darcy problem
//               corresponding to the saddle point system
//                                 k*u + grad p = f
//                                 - div u      = g
//               with natural boundary condition -p = <given pressure>.
//               Here, we use a given exact solution (u,p) and compute the
//               corresponding r.h.s. (f,g).  We discretize with Raviart-Thomas
//               finite elements (velocity u) and piecewise discontinuous
//               polynomials (pressure p).
//
//               The example demonstrates the use of the BlockMatrix class, as
//               well as the collective saving of several grid functions in a
//               VisIt (visit.llnl.gov) visualization format.
//
//               We recommend viewing examples 1-4 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include "spaND.h"
#include "mfem_util.h"
#include "mmio.hpp"

using namespace std;
using namespace mfem;
using namespace spaND;

// Define the analytical solution and forcing terms / boundary conditions
void uFun_ex(const Vector & x, Vector & u);
double pFun_ex(const Vector & x);
void fFun(const Vector & x, Vector & f);
double gFun(const Vector & x);
double f_natural(const Vector & x);

int main(int argc, char *argv[])
{
   StopWatch chrono;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool visualization = 1;
   int target = 10000;
   double tol = 1e-2;
   double sp_eps = 1e-2;
   int skip = 2;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&target, "-t", "--target",
                  "Target FE mesh size");
   args.AddOption(&tol, "-tol", "--tol",
                  "spaND tol");
   args.AddOption(&sp_eps, "-eps", "--eps",
                  "(1,1) bloc negative shift");
   args.AddOption(&skip, "-s", "--skip",
                  "spaND skip");
   
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
   Mesh *mesh = new Mesh(mesh_file, 1, 1);
   int dim = mesh->Dimension();

   // 3. Refine the mesh to increase the resolution. In this example we do
   //    'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
   //    largest number that gives a final mesh with no more than 10,000
   //    elements.
   {
      int ref_levels =
         (int)floor(log(double(target)/mesh->GetNE())/log(2.)/dim);
      for (int l = 0; l < ref_levels; l++)
      {
         mesh->UniformRefinement();
      }
   }
   cout << "Number of elements: " << mesh->GetNE() << endl;

   // 4. Define a finite element space on the mesh. Here we use the
   //    Raviart-Thomas finite elements of the specified order.
   FiniteElementCollection *hdiv_coll(new RT_FECollection(order, dim));
   FiniteElementCollection *l2_coll(new L2_FECollection(order, dim));

   FiniteElementSpace *R_space = new FiniteElementSpace(mesh, hdiv_coll);
   FiniteElementSpace *W_space = new FiniteElementSpace(mesh, l2_coll);

   cout << "R Vdim: " << R_space->GetVDim() << endl;
   cout << "R NDof: " << R_space->GetNDofs() << endl;
   cout << "R VSize: " << R_space->GetVSize() << endl;
   cout << "R NV/NE/NF dofs: " << R_space->GetNV() << " " << R_space->GetNE() << " " << R_space->GetNBE() << " " << R_space->GetNF() << endl;

   cout << "W Vdim: " << W_space->GetVDim() << endl;
   cout << "W NDof: " << W_space->GetNDofs() << endl;
   cout << "W VSize: " << W_space->GetVSize() << endl;
   cout << "W NV/NE/NF dofs: " << W_space->GetNV() << " " << W_space->GetNE() << " " << W_space->GetNBE() << " " << W_space->GetNF() << endl;

   // 5. Define the BlockStructure of the problem, i.e. define the array of
   //    offsets for each variable. The last component of the Array is the sum
   //    of the dimensions of each block.
   Array<int> block_offsets(3); // number of variables + 1
   block_offsets[0] = 0;
   block_offsets[1] = R_space->GetVSize();
   block_offsets[2] = W_space->GetVSize();
   block_offsets.PartialSum();

   int Rsize = block_offsets[1] - block_offsets[0];
   int Wsize = block_offsets[2] - block_offsets[1];

   std::cout << "***********************************************************\n";
   std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
   std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
   std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
   std::cout << "***********************************************************\n";

   // 6. Define the coefficients, analytical solution, and rhs of the PDE.
   ConstantCoefficient k(1.0);

   VectorFunctionCoefficient fcoeff(dim, fFun);
   FunctionCoefficient fnatcoeff(f_natural);
   FunctionCoefficient gcoeff(gFun);

   VectorFunctionCoefficient ucoeff(dim, uFun_ex);
   FunctionCoefficient pcoeff(pFun_ex);

   // 7. Allocate memory (x, rhs) for the analytical solution and the right hand
   //    side.  Define the GridFunction u,p for the finite element solution and
   //    linear forms fform and gform for the right hand side.  The data
   //    allocated by x and rhs are passed as a reference to the grid functions
   //    (u,p) and the linear forms (fform, gform).
   BlockVector x(block_offsets), rhs(block_offsets);

   LinearForm *fform(new LinearForm);
   fform->Update(R_space, rhs.GetBlock(0), 0);
   fform->AddDomainIntegrator(new VectorFEDomainLFIntegrator(fcoeff));
   fform->AddBoundaryIntegrator(new VectorFEBoundaryFluxLFIntegrator(fnatcoeff));
   fform->Assemble();

   LinearForm *gform(new LinearForm);
   gform->Update(W_space, rhs.GetBlock(1), 0);
   gform->AddDomainIntegrator(new DomainLFIntegrator(gcoeff));
   gform->Assemble();

   // 8. Assemble the finite element matrices for the Darcy operator
   //
   //                            D = [ M  B^T ]
   //                                [ B   0  ]
   //     where:
   //
   //     M = \int_\Omega k u_h \cdot v_h d\Omega   u_h, v_h \in R_h
   //     B   = -\int_\Omega \div u_h q_h d\Omega   u_h \in R_h, q_h \in W_h
   BilinearForm *mVarf(new BilinearForm(R_space));
   MixedBilinearForm *bVarf(new MixedBilinearForm(R_space, W_space));

   mVarf->AddDomainIntegrator(new VectorFEMassIntegrator(k));
   mVarf->Assemble();
   mVarf->Finalize();
   SparseMatrix &M(mVarf->SpMat());

   bVarf->AddDomainIntegrator(new VectorFEDivergenceIntegrator);
   bVarf->Assemble();
   bVarf->Finalize();
   SparseMatrix & B(bVarf->SpMat());
   B *= -1.;
   SparseMatrix *BT = Transpose(B);

   BlockMatrix darcyMatrix(block_offsets);
   darcyMatrix.SetBlock(0,0, &M);
   darcyMatrix.SetBlock(0,1, BT);
   darcyMatrix.SetBlock(1,0, &B);

   // Get the matrix
   int Nout = darcyMatrix.NumRows();
   cout << darcyMatrix.NumRows() << " x " << darcyMatrix.NumCols() << endl;
   SpMat Me = mfem2eigen(M);
   SpMat Be = mfem2eigen(B);
   cout << "Mfem: " << M.NumNonZeroElems() << " + 2x " << B.NumNonZeroElems() << endl;
   cout << Me.cols() << "x" << Me.rows() << " - " << Be.cols() << "x" << Be.rows() << endl;
   // M to triplets
   vector<Eigen::Triplet<double>> triplets;
   for (int k = 0; k < Me.outerSize(); ++k) {
      for (SpMat::InnerIterator it(Me,k); it; ++it) {
         triplets.push_back({it.row(), it.col(), it.value()});
      }
   }
   // B & B^T
   for (int k = 0; k < Be.outerSize(); ++k) {
      for (SpMat::InnerIterator it(Be,k); it; ++it) {
         triplets.push_back({Rsize + it.row(), it.col(),         it.value()});
         triplets.push_back({it.col()        , Rsize + it.row(), it.value()});
      }
   }

   // All together for A
   SpMat A(Nout, Nout);
   A.setFromTriplets(triplets.begin(), triplets.end());
   cout << A.rows() << "x" << A.cols() << " NNZ ? " << A.nonZeros() << endl;

   // Add negative bottom right diagonal for Aprec
   // double Anorm = A.norm();
   for(int i = 0; i < Wsize; i++) {
      triplets.push_back({Rsize + i, Rsize + i, -sp_eps});
   }
   SpMat Aprec(Nout, Nout);
   Aprec.setFromTriplets(triplets.begin(), triplets.end());

   // Get coordinates
   // For R
   FiniteElementSpace *R_vfes = new FiniteElementSpace(mesh, hdiv_coll, dim);
   GridFunction R_coords(R_vfes);
   {
      DenseMatrix coords, coords_t;
      Array<int> rt_vdofs;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const FiniteElement *rt_fe = R_vfes->GetFE(i);
         const IntegrationRule &rt_nodes = rt_fe->GetNodes();
         ElementTransformation *T = mesh->GetElementTransformation(i);
         T->Transform(rt_nodes, coords);
         coords_t.Transpose(coords);
         R_vfes->GetElementVDofs(i, rt_vdofs);
         FiniteElementSpace::AdjustVDofs(rt_vdofs);
         R_coords.SetSubVector(rt_vdofs, coords_t.GetData());
      }
   }
   // For W
   FiniteElementSpace *W_vfes = new FiniteElementSpace(mesh, l2_coll, dim);
   GridFunction W_coords(W_vfes);
   {
      DenseMatrix coords, coords_t;
      Array<int> wt_vdofs;
      for (int i = 0; i < mesh->GetNE(); i++)
      {
         const FiniteElement *wt_fe = W_vfes->GetFE(i);
         const IntegrationRule &wt_nodes = wt_fe->GetNodes();
         ElementTransformation *T = mesh->GetElementTransformation(i);
         T->Transform(wt_nodes, coords);
         coords_t.Transpose(coords);
         W_vfes->GetElementVDofs(i, wt_vdofs);
         W_coords.SetSubVector(wt_vdofs, coords_t.GetData());
      }
   }

   // Build Xcoord
   Eigen::MatrixXd Xcoo = Eigen::MatrixXd::Zero(dim, Nout);
   assert(Rsize == R_coords.Size() / dim);
   assert(Wsize == W_coords.Size() / dim);   
   for(int i = 0; i < Rsize; i++) {
      for(int d = 0; d < dim; d++) {
         Xcoo(d, i) = R_coords.GetData()[i + d * Rsize];
      }
   }
   for(int i = 0; i < Wsize; i++) {
      for(int d = 0; d < dim; d++) {
         Xcoo(d, Rsize + i) = W_coords.GetData()[i + d * Wsize];
      }
   }
   cout << Xcoo.leftCols(10) << endl;
   cout << Xcoo.rightCols(10) << endl;

#if 0
      cout << "***********" << endl;
      cout << "R_coords.Size() = " << R_coords.Size() << endl;
      cout << "W_coords.Size() = " << W_coords.Size() << endl;
      std::ofstream Rfs("R_coords.txt", std::ofstream::out);
      Rfs << R_coords;
      Rfs.close();
      std::ofstream Wfs("W_coords.txt", std::ofstream::out);
      Wfs << W_coords;
      Wfs.close();
      mmio::sp_mmwrite("A.txt", Aprec);
      cout << "***********" << endl;
#endif
   
   // Try solving
   SpMat AAT = symmetric_graph(Aprec);
   int lvl = (int)ceil(log( double(Nout) / 64.0)/log(2.0))-2;     
   Tree t(lvl);
   t.set_symm_kind(SymmKind::GEN);
   t.set_tol(tol);
   t.set_skip(skip);
   t.set_Xcoo(&Xcoo);
   t.set_use_geo(true);
   t.partition(AAT);
   t.assemble(Aprec);
   t.set_monitor_condition_pivots(true);
   t.set_scaling_kind(ScalingKind::SVD);
   try {
      t.factorize();
   } catch (std::exception& ex) {
      cout << ex.what();
   }
   t.print_log();
   Eigen::VectorXd rhse = mfem2eigen(rhs);
   Eigen::VectorXd sole = Eigen::VectorXd::Zero(rhse.rows());
   int iter = gmres(A, rhse, sole, t, 100, 100, 1e-12, true);
   cout << "GMRES: " << iter << " |Ax-b|/|b|: " << (A*sole-rhse).norm() / rhse.norm() << endl;

   {
      for(int i = 0; i < Nout; i++) {
         x.Elem(i) = sole[i];
      }
   }

   // 9. Construct the operators for preconditioner
   //
   //                 P = [ diag(M)         0         ]
   //                     [  0       B diag(M)^-1 B^T ]
   //
   //     Here we use Symmetric Gauss-Seidel to approximate the inverse of the
   //     pressure Schur Complement
//    SparseMatrix *MinvBt = Transpose(B);
//    Vector Md(M.Height());
//    M.GetDiag(Md);
//    for (int i = 0; i < Md.Size(); i++)
//    {
//       MinvBt->ScaleRow(i, 1./Md(i));
//    }
//    SparseMatrix *S = Mult(B, *MinvBt);

//    Solver *invM, *invS;
//    invM = new DSmoother(M);
// #ifndef MFEM_USE_SUITESPARSE
//    invS = new GSSmoother(*S);
// #else
//    invS = new UMFPackSolver(*S);
// #endif

//    invM->iterative_mode = false;
//    invS->iterative_mode = false;

//    BlockDiagonalPreconditioner darcyPrec(block_offsets);
//    darcyPrec.SetDiagonalBlock(0, invM);
//    darcyPrec.SetDiagonalBlock(1, invS);

//    // 10. Solve the linear system with MINRES.
//    //     Check the norm of the unpreconditioned residual.
//    int maxIter(10);
//    double rtol(1.e-6);
//    double atol(1.e-10);

//    chrono.Clear();
//    chrono.Start();
//    MINRESSolver solver;
//    solver.SetAbsTol(atol);
//    solver.SetRelTol(rtol);
//    solver.SetMaxIter(maxIter);
//    solver.SetOperator(darcyMatrix);
//    solver.SetPreconditioner(darcyPrec);
//    solver.SetPrintLevel(1);
//    x = 0.0;
//    solver.Mult(rhs, x);
//    chrono.Stop();

   // if (solver.GetConverged())
   //    std::cout << "MINRES converged in " << solver.GetNumIterations()
   //              << " iterations with a residual norm of " << solver.GetFinalNorm() << ".\n";
   // else
   //    std::cout << "MINRES did not converge in " << solver.GetNumIterations()
   //              << " iterations. Residual norm is " << solver.GetFinalNorm() << ".\n";
   // std::cout << "MINRES solver took " << chrono.RealTime() << "s. \n";

   // 11. Create the grid functions u and p. Compute the L2 error norms.
   GridFunction u, p;
   u.MakeRef(R_space, x.GetBlock(0), 0);
   p.MakeRef(W_space, x.GetBlock(1), 0);

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_u  = u.ComputeL2Error(ucoeff, irs);
   double norm_u = ComputeLpNorm(2., ucoeff, *mesh, irs);
   double err_p  = p.ComputeL2Error(pcoeff, irs);
   double norm_p = ComputeLpNorm(2., pcoeff, *mesh, irs);

   std::cout << "|| u_h - u_ex || / || u_ex || = " << err_u / norm_u << "\n";
   std::cout << "|| p_h - p_ex || / || p_ex || = " << err_p / norm_p << "\n";

   // 12. Save the mesh and the solution. This output can be viewed later using
   //     GLVis: "glvis -m ex5.mesh -g sol_u.gf" or "glvis -m ex5.mesh -g
   //     sol_p.gf".
   {
      ofstream mesh_ofs("ex5.mesh");
      mesh_ofs.precision(8);
      mesh->Print(mesh_ofs);

      ofstream u_ofs("sol_u.gf");
      u_ofs.precision(8);
      u.Save(u_ofs);

      ofstream p_ofs("sol_p.gf");
      p_ofs.precision(8);
      p.Save(p_ofs);
   }

   // 13. Save data in the VisIt format
   VisItDataCollection visit_dc("Example5", mesh);
   visit_dc.RegisterField("velocity", &u);
   visit_dc.RegisterField("pressure", &p);
   visit_dc.Save();

   // 14. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream u_sock(vishost, visport);
      u_sock.precision(8);
      u_sock << "solution\n" << *mesh << u << "window_title 'Velocity'" << endl;
      socketstream p_sock(vishost, visport);
      p_sock.precision(8);
      p_sock << "solution\n" << *mesh << p << "window_title 'Pressure'" << endl;
   }

   // 15. Free the used memory.
   delete fform;
   delete gform;
   // delete invM;
   // delete invS;
   // delete S;
   // delete MinvBt;
   delete BT;
   delete mVarf;
   delete bVarf;
   delete W_space;
   delete R_space;
   delete l2_coll;
   delete hdiv_coll;
   delete mesh;

   return 0;
}


void uFun_ex(const Vector & x, Vector & u)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);
   if (x.Size() == 3)
   {
      zi = x(2);
   }

   u(0) = - exp(xi)*sin(yi)*cos(zi);
   u(1) = - exp(xi)*cos(yi)*cos(zi);

   if (x.Size() == 3)
   {
      u(2) = exp(xi)*sin(yi)*sin(zi);
   }
}

// Change if needed
double pFun_ex(const Vector & x)
{
   double xi(x(0));
   double yi(x(1));
   double zi(0.0);

   if (x.Size() == 3)
   {
      zi = x(2);
   }

   return exp(xi)*sin(yi)*cos(zi);
}

void fFun(const Vector & x, Vector & f)
{
   f = 0.0;
}

double gFun(const Vector & x)
{
   if (x.Size() == 3)
   {
      return -pFun_ex(x);
   }
   else
   {
      return 0;
   }
}

double f_natural(const Vector & x)
{
   return (-pFun_ex(x));
}
