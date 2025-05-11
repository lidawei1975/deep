#include <cmath>
#include "lmminimizer_n.h"



levmarq::levmarq()
{
	verbose = 0;
	max_it = 250;
	lambda = 0.0001;
	up = 4.0;
	down = 1 / 4.0;
	target_derr = 1e-12;
	TOL = 1e-12;
};

levmarq::~levmarq(){};


/* perform least-squares minimization using the Levenberg-Marquardt
   algorithm.  The arguments are as follows:
   npar    number of parameters
   par     array of parameters to be varied
   ny      number of measurements to be fit
   dysq    array of error in measurements, squared
		   (set dysq=NULL for unweighted least-squares)
   func    function to be fit
   grad    gradient of "func" with respect to the input parameters
   lmstat  pointer to the "status" structure, where minimization parameters
		   are set and the final status is returned.
   Before calling levmarq, several of the parameters in lmstat must be set.
   For default values, call levmarq_init(lmstat).
 */
int levmarq::solve(double **par, double **newpar, int ny, double *dysq,double **g, double *ypre, ldwcostfunction &func)
{
	int i, j, it, ill;
	double mult, weight, err, newerr, derr;

	double h[npar * npar];
	double ch[npar * npar];
	double d[npar];
	double delta[npar];


	weight = 1;
	derr = newerr = 0;

	/* calculate the initial error ("chi-squared") */
	err = error_func(par, ny, dysq, ypre,func);

	/* main iteration */
	for (it = 0; it < max_it; it++)
	{

		/* calculate the approximation to the Hessian and the "derivative" d */
		for (int i = 0; i < npar; i++)
		{
			d[i] = 0;
			for (j = 0; j <= i; j++)
			{
				h[i*npar+j] = 0;
			}
		}

		func.Evaluate(par, ypre, g);


		for (int m = 0; m < ny; m++)
		{
			if (dysq)
			{
				weight = 1 / dysq[m]; /* for weighted least-squares */
			}

			for (i = 0; i < npar; i++)
			{
				d[i] +=  -ypre[m] * g[i][m] * weight;
				for (j = 0; j <= i; j++)
				{
					h[i*npar+j] += g[i][m] * g[j][m] * weight;
				}
			}
		}

		/*  make a step "delta."  If the step is rejected, increase
		   lambda and try again */
		mult = 1 + lambda;
		ill = 1; /* ill-conditioned? */
		while (ill && (it < max_it))
		{
			for (i = 0; i < npar; i++)
				h[i*npar+i] = h[i*npar+i] * mult;

			ill = cholesky_decomp(npar, ch, h);

			if (!ill)
			{
				solve_axb_cholesky(npar, ch, delta, d);
				for (i = 0; i < npar; i++)
				{
					newpar[i][0] = par[i][0] + delta[i];
				}
				newerr = error_func(newpar, ny, dysq, ypre,func);
				derr = newerr - err;
				ill = (derr > 0);
			}
			if (verbose)
            {
                std::cout<<"it = "<<it<<",   lambda = "<<lambda<<",   err = "<<err<<",   derr = "<<derr<<std::endl;
            }
				
			if (ill)
			{
				mult = (1 + lambda * up) / (1 + lambda);
				lambda *= up;
				it++;
			}
		}
		for (i = 0; i < npar; i++)
		{
			par[i][0] = newpar[i][0];
		}
		err = newerr;
		lambda *= down;

		if ((!ill) && (-derr < target_derr))
			break;
	}

	final_it = it;
	final_err = err;
	final_derr = derr;

	return (it == max_it);

}

/* calculate the error function (chi-squared) */
double levmarq::error_func(double **par, int ny, double *dysq,double *ypre, ldwcostfunction &func)
{
	double res, e = 0;

	func.Evaluate(par, ypre, NULL);

	for (int i = 0; i < ny; i++)
	{
		res = ypre[i];
		if (dysq) /* weighted least-squares */
			e += res * res / dysq[i];
		else
			e += res * res;
	}


	return e;
}


/* solve the equation Ax=b for a symmetric positive-definite matrix A,
   using the Cholesky decomposition A=LL^T.  The matrix L is passed in "l".
   Elements above the diagonal are ignored.
*/
void levmarq::solve_axb_cholesky(int n, double *matrix_l, double *x, double *b)
{
	int i, j;
	double sum;

	/* solve L*y = b for y (where x[] is used to store y) */

	for (i = 0; i < n; i++)
	{
		sum = 0;
		for (j = 0; j < i; j++)
			sum += matrix_l[i*n+j] * x[j];
		x[i] = (b[i] - sum) / matrix_l[i*n+i];
	}

	/* solve L^T*x = y for x (where x[] is used to store both y and x) */

	for (i = n - 1; i >= 0; i--)
	{
		sum = 0;
		for (j = i + 1; j < n; j++)
			sum += matrix_l[j*n+i] * x[j];
		x[i] = (x[i] - sum) / matrix_l[i*n+i];
	}
}

/* This function takes a symmetric, positive-definite matrix "a" and returns
   its (lower-triangular) Cholesky factor in "l".  Elements above the
   diagonal are neither used nor modified.  The same array may be passed
   as both l and a, in which case the decomposition is performed in place.
*/
int levmarq::cholesky_decomp(int n, double *matrix_l, double *matrix_a)
{
	int i, j, k;
	double sum;

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < i; j++)
		{
			sum = 0;
			for (k = 0; k < j; k++)
				sum += matrix_l[i*n+k] * matrix_l[j*n+k];
			matrix_l[i*n+j] = (matrix_a[i*n+j] - sum) / matrix_l[j*n+j];
		}

		sum = 0;
		for (k = 0; k < i; k++)
			sum += matrix_l[i*n+k] * matrix_l[i*n+k];
		sum = matrix_a[i*n+i] - sum;
		if (sum < TOL)
			return 1; /* not positive-definite */
		matrix_l[i*n+i] = sqrt(sum);
	}
	return 0;
}

