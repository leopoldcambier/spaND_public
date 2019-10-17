// Code partially from Eigen http://eigen.tuxfamily.org
// Originally published under the MPLv2 License https://www.mozilla.org/en-US/MPL/2.0/
// See http://eigen.tuxfamily.org/index.php?title=Main_Page#License

#include "spaND.h"

using namespace Eigen;
using namespace spaND;

namespace spaND {

int ir(const SpMat& mat, const VectorXd& rhs, VectorXd& x, const Tree& precond, int iters, double tol, bool verb)
{
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> VectorType;
    VectorType residual = rhs - mat * x;
    double rhsNorm2 = rhs.squaredNorm();
    if(rhsNorm2 == 0){
        x.setZero();
        return 0;
    }
    double threshold = tol*tol*rhsNorm2;
    double residualNorm2 = residual.squaredNorm();
    int i = 1;
    while(residualNorm2 >= threshold && i < iters) {
        precond.solve(residual);
        x += residual;
        residual = rhs - mat * x;
        residualNorm2 = residual.squaredNorm();
        if(verb) printf("%d: |Ax-b|/|b| = %3.2e <? %3.2e\n", i, sqrt(residualNorm2 / rhsNorm2), tol);
        if(residualNorm2 < threshold) {
            if(verb) printf("Converged!\n");
            break;
        }
        i++;
    }
    return i;
}

int cg(const SpMat& mat, const VectorXd& rhs, VectorXd& x, const Tree& precond, int iters, double tol, bool verb)
{
    using std::sqrt;
    using std::abs;
    typedef Eigen::Matrix<double,Eigen::Dynamic,1> VectorType;
    
    double t_matvec = 0.0;
    double t_preco  = 0.0;
    timer start     = wctime();
    
    int maxIters = iters;
    
    int n = mat.cols();

    timer t00 = wctime();
    VectorType residual = rhs - mat * x; // r_0 = b - A x_0
    timer t01 = wctime();
    t_matvec += elapsed(t00, t01);

    double rhsNorm2 = rhs.squaredNorm();
    if(rhsNorm2 == 0) 
    {
        x.setZero();
        iters = 0;
        return iters;
    }
    double threshold = tol*tol*rhsNorm2;
    double residualNorm2 = residual.squaredNorm();
    if (residualNorm2 < threshold)
    {
        iters = 0;
        return iters;
    }
   
    VectorType p(n);
    p = residual;
    timer t02 = wctime();
    precond.solve(p);      // p_0 = M^-1 r_0
    timer t03 = wctime();
    t_preco += elapsed(t02, t03);

    VectorType z(n), tmp(n);
    double absNew = residual.dot(p);  // the square of the absolute value of r scaled by invM
    int i = 0;
    while(i < maxIters)
    {
        timer t0 = wctime();
        tmp.noalias() = mat * p;                    // the bottleneck of the algorithm
        timer t1 = wctime();
        t_matvec += elapsed(t0, t1);

        double alpha = absNew / p.dot(tmp);         // the amount we travel on dir
        x += alpha * p;                             // update solution
        residual -= alpha * tmp;                    // update residual
        
        residualNorm2 = residual.squaredNorm();
        if(verb) printf("%d: |Ax-b|/|b| = %3.2e <? %3.2e\n", i, sqrt(residualNorm2 / rhsNorm2), tol);
        if(residualNorm2 < threshold) {
            if(verb) printf("Converged!\n");
            break;
        }
     
        z = residual; 
        timer t2 = wctime();
        precond.solve(z);                           // approximately solve for "A z = residual"
        timer t3 = wctime();
        t_preco += elapsed(t2, t3);
        double absOld = absNew;
        absNew = residual.dot(z);                   // update the absolute value of r
        double beta = absNew / absOld;              // calculate the Gram-Schmidt value used to create the new search direction
        p = z + beta * p;                           // update search direction
        i++;
    }
    iters = i+1;
    if(verb) {
        timer stop = wctime();
        printf("# of iter:  %d\n", iters);
        printf("Total time: %3.2e s.\n", elapsed(start, stop));
        printf("  Matvec:   %3.2e s.\n", t_matvec);
        printf("  Precond:  %3.2e s.\n", t_preco);
    }
    return iters;
};

int gmres(const SpMat& mat, const VectorXd& rhs, VectorXd& x, const Tree& precond, int iters, int restart, double tol_error, bool verb) {

    timer start     = wctime();
    double t_matvec = 0.0;
    double t_preco  = 0.0;

    using std::sqrt;
    using std::abs;

    typedef typename VectorXd::RealScalar RealScalar;
    typedef typename VectorXd::Scalar Scalar;
    typedef Matrix < Scalar, Dynamic, 1 > VectorType;
    typedef Matrix < Scalar, Dynamic, Dynamic, ColMajor> FMatrixType;

    const RealScalar considerAsZero = (std::numeric_limits<RealScalar>::min)();

    if(rhs.norm() <= considerAsZero) {
        x.setZero();
        tol_error = 0;
        return true;
    }

    RealScalar tol = tol_error;
    const int maxIters = iters;
    iters = 0;

    const int m = mat.rows();

    // residual and preconditioned residual
    timer t0 = wctime();    
    VectorType p0 = rhs - mat*x;
    timer t1 = wctime();
    t_matvec += elapsed(t0, t1);
    VectorType r0 = p0;
    precond.solve(r0);
    timer t2 = wctime();
    t_preco += elapsed(t1, t2);

    const RealScalar r0Norm = r0.norm();

    // is initial guess already good enough?
    if(r0Norm == 0) {
        printf("Converged!");
        return true;
    }

    // storage for Hessenberg matrix and Householder data
    FMatrixType H   = FMatrixType::Zero(m, restart + 1);
    VectorType w    = VectorType::Zero(restart + 1);
    VectorType tau  = VectorType::Zero(restart + 1);

    // storage for Jacobi rotations
    std::vector < JacobiRotation < Scalar > > G(restart);

    // storage for temporaries
    VectorType t(m), v(m), workspace(m), x_new(m);

    // generate first Householder vector
    Ref<VectorType> H0_tail = H.col(0).tail(m - 1);
    RealScalar beta;
    r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
    w(0) = Scalar(beta);

    for (int k = 1; k <= restart; ++k)
    {
        ++iters;

        v = VectorType::Unit(m, k - 1);

        // apply Householder reflections H_{1} ... H_{k-1} to v
        // TODO: use a HouseholderSequence
        for (int i = k - 1; i >= 0; --i) {
            v.tail(m - i).applyHouseholderOnTheLeft(H.col(i).tail(m - i - 1), tau.coeffRef(i), workspace.data());
        }

        // apply matrix M to v:  v = mat * v;
        timer t0 = wctime();
        t.noalias() = mat * v;
        timer t1 = wctime();
        t_matvec += elapsed(t0, t1);
        v = t;
        precond.solve(v);
        timer t2 = wctime();
        t_preco += elapsed(t1, t2);

        // apply Householder reflections H_{k-1} ... H_{1} to v
        // TODO: use a HouseholderSequence
        for (int i = 0; i < k; ++i) {
            v.tail(m - i).applyHouseholderOnTheLeft(H.col(i).tail(m - i - 1), tau.coeffRef(i), workspace.data());
        }

        if (v.tail(m - k).norm() != 0.0) {
            if (k <= restart)
            {
            // generate new Householder vector
            Ref<VectorType> Hk_tail = H.col(k).tail(m - k - 1);
            v.tail(m - k).makeHouseholder(Hk_tail, tau.coeffRef(k), beta);

            // apply Householder reflection H_{k} to v
            v.tail(m - k).applyHouseholderOnTheLeft(Hk_tail, tau.coeffRef(k), workspace.data());
            }
        }

        if (k > 1) {
            for (int i = 0; i < k - 1; ++i) {
                // apply old Givens rotations to v
                v.applyOnTheLeft(i, i + 1, G[i].adjoint());
            }
        }

        if (k<m && v(k) != (Scalar) 0) {
            // determine next Givens rotation
            G[k - 1].makeGivens(v(k - 1), v(k));

            // apply Givens rotation to v and w
            v.applyOnTheLeft(k - 1, k, G[k - 1].adjoint());
            w.applyOnTheLeft(k - 1, k, G[k - 1].adjoint());
        }

        // insert coefficients into upper matrix triangle
        H.col(k-1).head(k) = v.head(k);

        tol_error = abs(w(k)) / r0Norm;
        bool stop = (k==m || tol_error < tol || iters == maxIters);
        if(verb) printf("%d: |Ax-b|/|b| = %3.2e <? %3.2e\n", iters, tol_error, tol);

        if (stop || k == restart) {
            // solve upper triangular system
            Ref<VectorType> y = w.head(k);
            H.topLeftCorner(k, k).template triangularView <Upper>().solveInPlace(y);

            // use Horner-like scheme to calculate solution vector
            x_new.setZero();
            for (int i = k - 1; i >= 0; --i) {
                x_new(i) += y(i);
                // apply Householder reflection H_{i} to x_new
                x_new.tail(m - i).applyHouseholderOnTheLeft(H.col(i).tail(m - i - 1), tau.coeffRef(i), workspace.data());
            }

            x += x_new;

            if(stop) {
                printf("GMRES converged!\n");
                if(verb) {
                    timer stop = wctime();
                    printf("# of iter:  %d\n", iters);
                    printf("Total time: %3.2e s.\n", elapsed(start, stop));
                    printf("  Matvec:   %3.2e s.\n", t_matvec);
                    printf("  Precond:  %3.2e s.\n", t_preco);
                }
                return iters;
            } else {
                k=0;

                // reset data for restart
                timer t0 = wctime();
                p0.noalias() = rhs - mat*x;
                timer t1 = wctime();
                t_matvec += elapsed(t0, t1);                
                r0 = p0;
                precond.solve(r0);
                timer t2 = wctime();
                t_preco += elapsed(t1, t2);

                // clear Hessenberg matrix and Householder data
                H.setZero();
                w.setZero();
                tau.setZero();

                // generate first Householder vector
                r0.makeHouseholder(H0_tail, tau.coeffRef(0), beta);
                w(0) = Scalar(beta);
            }
        }
    }
    printf("GMRES failed to converge in %d iterations\n", iters);
    return iters;
}

}