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

	std::vector<double> lower_bounds;
	std::vector<double> upper_bounds;
	bool has_bounds;

	void solve_axb_cholesky(int n, double * l, double *x, double *b);
	int solve_axb_simplicial_ldlt(int n, double *matrix_a, double *x, double *b);
	int cholesky_decomp(int n, double *l, double *a);

public:
	levmarq();
	~levmarq();

	void set_bounds(const std::vector<double>& lb, const std::vector<double>& ub);
	void clear_bounds();

	double error_func(double **par, int ny, double *dysq, ldwcostfunction &func);
	int solve(int npar, double **par, int ny, double *dysq,ldwcostfunction &func, int linear_solver_flag = 0);

	/**
	 * Below are functions to solve with mutiple cost functions
	 */
	int solve(int nshared_par, int npar, double **par, int ny, double *dysq, std::vector<ldwcostfunction *> funcs);
	double error_func(int nshared_par, double **par, int ny, double *dysq, std::vector<ldwcostfunction*> funcs);

};
#endif