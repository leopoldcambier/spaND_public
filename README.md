# spaND
Repository for the spaND algorithm (https://arxiv.org/abs/1901.02971).

## Disclaimer
This is very researchy-hacky code. 
The goal is to easily experiments various kinds of sparsification techniques, with reasonnable performances.
This is *not* production-ready code. I am not a C++ guru and many things are very imperfect and may break if you try too much :-)

Some short pieces of code (or some examples) have been taken from other open source project. 
This is indicated when needed.

## Build instructions

Necessary:
- Download Eigen (header library only) : http://eigen.tuxfamily.org/index.php?title=Main_Page
- Download linalgCpp (header library only) : https://github.com/leopoldcambier/linalgCpp
- Download, build and install Metis in **Int 32 bits mode** : http://glaros.dtc.umn.edu/gkhome/metis/metis/overview
- Download, build and install Openblas : https://www.openblas.net/ or Intel MKL

Optional:
- Download, build and install Mfem : https://mfem.org/
- Download, build and install googletest : https://github.com/google/googletest

Then

1. Create and `obj/` directory:

    `mkdir obj`.
    
2. Copy one of the file in the `tests/Makefile-confs` folder and create a `tests/Makefile.conf` file with your system settings. For instance 

    `cp tests/Makefile-confs/Makefile-sherlock tests/Makefile.conf`.
    
    You need to set the following variables in `Makefile.conf`:
    
    - `USE_MKL` to either 0 (not using MKL) or 1 (using MKL - this is because MKL requires specific build flags). If you have both, you can set it at build time (see below.).
    - `CC` should be your C++ compiler (needs to support -std=c++11)
    - `EIGENDIR` should point to Eigen's root
    - `BLASDIR` should point to BLAS's include
    - `BLASLIB` should point to BLAS's lib
    - `METISDIR` should point to METIS's include
    - `METISLIB` should point to METIS's libmetis
    - `MMIODIR` should point to MMIO's (a subcomponent of linalgCpp) directory, like $(HOME)/git/linalgCpp/mmio if you cloned linalgCpp in $(HOME)/git
    - `GTEST` should point to googletest folder (optional)
    - `MFEMDIR` should point to MFEM's root (optional)
    - `MFEMLIB` should point to MFEM's root (optional)
        
3. Then, move to `tests/`. You can create any of the executables with `make target` or all with `make`. 
If you have both Openblas and MKL, you can pass `USE_MKL=0` or `USE_MKL=1` to make to chose between the two.

## Some additional things to do (maybe) when you run:
- You may have to add Metis's LIB or BLAS's LIB folder to `LD_LIBRARY_PATH`. For instance, with MKL on the Stanford Sherlock system, it looks like this (in this case, it's automatically set, no need to do anything)

    ```
    [lcambier@sh-ln07 login ~]$ module load icc metis imkl eigen
    [lcambier@sh-ln07 login ~]$ echo $LD_LIBRARY_PATH
    /share/software/user/restricted/imkl/2019/mkl/lib/intel64:/share/software/user/open/metis/5.1.0/lib:/share/software/user/restricted/icc/2019/lib/intel64
    ```
    
    or with Openblas

    ```
    [lcambier@sh-ln07 login ~]$ module load icc metis openblas eigen
    [lcambier@sh-ln07 login ~]$ echo $LD_LIBRARY_PATH
    /share/software/user/open/openblas/0.3.4/lib:/share/software/user/open/metis/5.1.0/lib:/share/software/user/restricted/icc/2019/lib/intel64
    ```
    
    If you don't have modules, you may have to set `LD_LIBRARY_PATH` yourself.
    
- You may have to change the Openblas link flag from `-lblas` to `-lopenblas`. Not clear why :-()

## Example

Let's compile the general driver code and run it on a sample arbitrary matrix. Set-up all the above. We assume you have Openblas installed, with 'USE_MKL=0' set in your `Makefile.conf` file.
1. Compile the ```spaND.cpp``` example

    ```make spaND```

2. Run it

    ```./spaND -m ../mats/neglapl_2_32.mm --coordinates ../mats/32x32.mm -t 1e-2 -l 5```
    
    You can find more after this function by doing `./spaND --help`.

    This should output something similar to this

```
Gandalf:tests lcambier$ ./spaND -m ../mats/neglapl_2_32.mm --coordinates ../mats/32x32.mm -t 1e-2 -l 5
Matrix 1024x1024 loaded from ../mats/neglapl_2_32.mm
Coordinate file 2x1024 loaded from ../mats/32x32.mm
Tree with 5 levels (0 eliminated so far)
Geometric partitioning of matrix with 1024 dofs with 5 levels in 2D
  Depth  1: 9.99e-05 s. (    1 separators, [   32    32], mean   32.0)
  Depth  2: 4.70e-05 s. (    2 separators, [   15    16], mean   15.5)
  Depth  3: 9.89e-05 s. (    4 separators, [   15    16], mean   15.2)
  Depth  4: 9.18e-05 s. (    8 separators, [    7     8], mean    7.4)
Partitioning time : 1.66e-03 s. (1.64e-04 s. ND)
Assembling (Size 1024 with 5 levels and symmetry 1)
Clustering size statistics
Lvl     Count       Min       Max      Mean
  0        16        49        64        53
  1         8         7         8         7
  2        12         1         8         5
  3         6         1         8         5
  4         7         1         8         5
Edge size statistics
Lvl     Count       Min       Max      Mean
  0        64       343      4096       979
  1        20         7        64        26
  2        24         1        64        21
  3        12         1        64        21
  4        13         1        64        20
Edge count statistics
Lvl     Count       Min       Max      Mean
  0        16         3         5         4
  1         8         2         3         2
  2        12         1         3         2
  3         6         1         3         2
  4         7         1         3         2
Assembly time : 2.33e-03 s. (7.30e-04 permuting A)
Factorization started
  N:          1024
  #levels:    5
  verbose?:   1
  symmetry?:  1
  adaptive?:  1
  tol?:       0.01
  #skip:      0
  scale?:     1
  ortho?:     1
  scalingkd?  LLT
  preserving? 0
Level 0, 1024 dofs left, 49 clusters left
  Elim: 3.18e-03 s., 183 dofs left, 33 clusters left
  Scaling: 4.31e-04 s.
  Sparsification: 6.31e-04 s., 129 dofs left, geqp3 2.22e-04, geqrf 0.00e+00, assmb 1.88e-05, buildQ 1.96e-05, scatterQ 0.00e+00, permA 2.81e-04, scatterA 4.24e-05
Level 1, 129 dofs left, 33 clusters left
  Elim: 1.14e-04 s., 89 dofs left, 25 clusters left
  Merge: 1.49e-04 s., 89 dofs left, 15 clusters left
  Scaling: 1.79e-04 s.
  Sparsification: 1.11e-04 s., 58 dofs left, geqp3 7.22e-05, geqrf 0.00e+00, assmb 5.25e-06, buildQ 2.86e-06, scatterQ 0.00e+00, permA 8.34e-06, scatterA 8.82e-06
Level 2, 58 dofs left, 15 clusters left
  Elim: 4.20e-05 s., 37 dofs left, 11 clusters left
  Merge: 4.41e-05 s., 37 dofs left, 5 clusters left
  Scaling: 1.39e-04 s.
  Sparsification: 5.20e-05 s., 17 dofs left, geqp3 2.65e-05, geqrf 0.00e+00, assmb 3.10e-06, buildQ 3.81e-06, scatterQ 0.00e+00, permA 3.81e-06, scatterA 8.82e-06
Level 3, 17 dofs left, 5 clusters left
  Elim: 1.19e-05 s., 9 dofs left, 3 clusters left
  Merge: 1.00e-05 s., 9 dofs left, 1 clusters left
  Scaling: 3.10e-06 s.
  Sparsification: 2.15e-06 s., 0 dofs left, geqp3 1.19e-06, geqrf 0.00e+00, assmb 0.00e+00, buildQ 0.00e+00, scatterQ 0.00e+00, permA 0.00e+00, scatterA 0.00e+00
Level 4, 0 dofs left, 1 clusters left
  Elim: 1.19e-06 s., 0 dofs left, 0 clusters left
  Merge: 3.10e-06 s., 0 dofs left, 0 clusters left
  Scaling: 0.00e+00 s.
  Sparsification: 0.00e+00 s., 0 dofs left, geqp3 0.00e+00, geqrf 0.00e+00, assmb 0.00e+00, buildQ 0.00e+00, scatterQ 0.00e+00, permA 0.00e+00, scatterA 0.00e+00
Factorization: 5.29e-03 s.
&&&& Lvl         Elim        Scale     Sparsify        Merge     Preserve        geqp3        geqrf         potf         trsm         gemm       buildq       scattq        assmb       scatta          phi   mergealloc    mergecopy
&&&&   0 3.175020e-03 4.310608e-04 6.308556e-04 0.000000e+00 0.000000e+00 2.222061e-04 0.000000e+00 2.013206e-03 1.103163e-03 2.038479e-04 1.955032e-05 0.000000e+00 1.883507e-05 4.243851e-05 0.000000e+00 0.000000e+00 0.000000e+00
&&&&   1 1.139641e-04 1.790524e-04 1.111031e-04 1.490116e-04 0.000000e+00 7.224083e-05 0.000000e+00 1.144409e-05 1.499653e-04 4.410744e-05 2.861023e-06 0.000000e+00 5.245209e-06 8.821487e-06 0.000000e+00 1.955032e-05 1.406670e-05
&&&&   2 4.196167e-05 1.389980e-04 5.197525e-05 4.410744e-05 0.000000e+00 2.646446e-05 0.000000e+00 8.583069e-06 1.225471e-04 2.479553e-05 3.814697e-06 0.000000e+00 3.099442e-06 8.821487e-06 0.000000e+00 3.337860e-06 8.821487e-06
&&&&   3 1.192093e-05 3.099442e-06 2.145767e-06 1.001358e-05 0.000000e+00 1.192093e-06 0.000000e+00 2.145767e-06 0.000000e+00 7.152557e-06 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
&&&&   4 1.192093e-06 0.000000e+00 0.000000e+00 3.099442e-06 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00 0.000000e+00
++++ Lvl        ND    ND lft    El lft    Sp lft   Fct nnz    Rk Bfr    Rk Aft
++++   0       841      1024       183       129     43846         7         5
++++   1        59       183        89        58     46846         8         5
++++   2        61       124        37        17     48789         9         4
++++   3        31        63         9         0     50011         9         0
++++   4        32        32         0         0     51035       nan       nan
Timings:
  Partition: 0.00209999 s.
  Assembly: 0.00233698 s.
  Factorization: 0.00577521 s.
1: |Ax-b|/|b| = 8.48e-04 <? 1.00e-12
2: |Ax-b|/|b| = 1.92e-05 <? 1.00e-12
3: |Ax-b|/|b| = 2.45e-08 <? 1.00e-12
4: |Ax-b|/|b| = 8.72e-11 <? 1.00e-12
5: |Ax-b|/|b| = 2.34e-13 <? 1.00e-12
GMRES converged!
# of iter:  5
Total time: 2.10e-03 s.
  Matvec:   3.96e-05 s.
  Precond:  1.15e-03 s.
GMRES: #iterations: 5, residual |Ax-b|/|b|: 5.2265e-13
  GMRES: 0.00211883 s.
<<<<GMRES=5
```
