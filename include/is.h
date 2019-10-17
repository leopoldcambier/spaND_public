#ifndef IS_H
#define IS_H

#include <iostream>
#include <stdio.h>
#include <Eigen/Core>

#include "spaND.h"

namespace spaND {

int cg(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int iters, double tol, bool verb);
int gmres(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int iters, int restart, double tol_error, bool verb);
int ir(const SpMat& mat, const Eigen::VectorXd& rhs, Eigen::VectorXd& x, const Tree& precond, int iters, double tol, bool verb);

}

#endif
