Eigen::VectorXd mfem2eigen(mfem::Vector &b) {
    int N = b.Size();
    Eigen::VectorXd x(N);
    for(int i = 0; i < N; i++) {
        x[i] = b(i);
    }
    return x;
}

void eigen2mfem(Eigen::VectorXd &a, mfem::Vector &b) {
    int N = b.Size();
    assert(a.size() == N);
    for(int i = 0; i < N; i++) {
        b(i) = a[i];
    }
}

SpMat mfem2eigen(mfem::SparseMatrix &A) {
    int N = A.Height();
    int M = A.Width();
    int nnz = A.NumNonZeroElems();
    std::vector<Eigen::Triplet<double>> triplets(nnz);
    int l = 0;
    for(int i = 0; i < A.Height(); i++) {
       for(int k = A.GetI()[i]; k < A.GetI()[i+1]; k++) {
           assert(l < nnz);
           int j = A.GetJ()[k];
           double v = A.GetData()[k];
           assert(i >= 0 && i < N && j >= 0 && j < M);
           triplets[l] = Eigen::Triplet<double>(i,j,v);
           l += 1;
       }
    }
    assert(l == nnz);
    SpMat B(N,M);
    B.setFromTriplets(triplets.begin(), triplets.end());
    return B;
}
