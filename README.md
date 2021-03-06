# numerical_electromagnetics

basic numerical solutions to some electromagnetic problems.  Only very basic methods are demonstrated - see comments.  

## Special functions

A script located at ```./specfun/basic_specfuns.py``` will show how use scipy to compute some common special functions (Legendre polynomials, Bessel functions, spherical Bessel functions, and Spherical Harmonics) and then how to visualize them.  

A script located at ```./specfun/Jackson_310_311.py``` computes and plots the series solutions to problems 3.10-3.11 in the book.

## Numerical solutions for electrostatic problems

See ```./electrostatics/2Dpotential.py``` for an example of how to set up finite difference equations for electrostatic problems and how to solve them with Jacobi and Gauss-Seidel iterations (discussed in the book).  

Additionally, there is a demo using a solution via LU factorization from UMFpack which is *much* faster than the iterative solutions for this type of problem.  Generally speaking for 2D solution an LU factorization is going to be best.  Note also that the optimized approaches like UMFPACK and MUMPS are going to be much faster than if you implement your own LU factorization, e.g. via the Doolittle algorithm.  

## Magnetic diffusion solutions

See ``` ./magnetic_diffusion/diffusion1D.py ``` for example of 1D diffusion problems from the book (section 5.18).  There is also a demo of how to do a simple 2D solution using a locally one-dimensional approximation ``` ./magnetic_diffusion/diffusion2D.py ```.

## Solutions to wave equations

See ``` ./waves/waves1D.py ``` for a simple solution to a scalar hyperbolic equations using the simple Lax-Wendroff method.  ``` ./waves/EMwaves1D.py ``` show how to extend Lax-Wendroff to solve vector wave equations, such as those describing plane EM wave propagation.  

There is also a demo of 2D wave excitation in ``` ./waves/EMwaves2D.py ```.  