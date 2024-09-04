#include <iostream>
#include <vector>


/**
 * A baseclass for the cost function
*/

#ifndef LDWCOSTFUNCTION_H
#define LDWCOSTFUNCTION_H

class ldwcostfunction
{
public:
    ldwcostfunction(){};
    virtual ~ldwcostfunction(){};
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobian) const = 0;
};


class levmarq
{
private:
	int verbose;
	int max_it;
	double lambda;
	double up;
	double down;
	double target_derr;
	int final_it;
	double final_err;
	double final_derr;	
	double TOL;

	
	void solve_axb_cholesky(int n, double * l, double *x, double *b);
	int cholesky_decomp(int n, double *l, double *a);

public:
	levmarq();
	~levmarq();

	double error_func(double **par, int ny, double *dysq, ldwcostfunction &func);
	int solve(int npar, double **par, int ny, double *dysq,ldwcostfunction &func);

	/**
	 * Below are functions to solve with mutiple cost functions
	 */
	int solve(int nshared_par, int npar, double **par, int ny, double *dysq, std::vector<ldwcostfunction *> funcs);
	double error_func(int nshared_par, double **par, int ny, double *dysq, std::vector<ldwcostfunction*> funcs);

};
#endif