#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <valarray>
#include <complex>
#include <string>
#include "omp.h"
#include <Eigen/Dense>
#include <Eigen/Cholesky>


#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::CostFunction;
using ceres::AutoDiffCostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#include "json/json.h"
#include "commandline.h"
#include "spectrum_fit.h"


//cost functions for ceres-solver
//Gaussian
mycostfunction_gaussian::~mycostfunction_gaussian(){};
mycostfunction_gaussian::mycostfunction_gaussian(int xdim_, int ydim_, double *z_)
{
    xdim=xdim_;
    ydim=ydim_;
    n_datapoint=xdim*ydim;
    z=z_;   
};



bool mycostfunction_gaussian::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a=xx[0][0];
    double x0=xx[1][0];
    double sigmax=fabs(xx[2][0]);
    double y0=xx[3][0];
    double sigmay=fabs(xx[4][0]);
 

    if (jaco != NULL ) //both residual errors and jaco are required.
    {
        for(int i=0;i<xdim;i++)
        {
            double x_sigmax=(i-x0)/sigmax;
            double gx=exp((x0-i)*x_sigmax);
            for(int j=0;j<ydim;j++)
            {
                double y_sigmay=(j-y0)/sigmay;
                double g=gx*exp((y0-j)*y_sigmay);
                double ag=a*g;
                int ii=i*ydim+j;

                residual[ii]=ag-z[ii]; 
                jaco[0][ii]=g; //with respect to a
                jaco[1][ii]=ag*2*x_sigmax; //x0
                jaco[2][ii]=ag*x_sigmax*x_sigmax; //sigmax
                jaco[3][ii]=ag*2*y_sigmay; //y0
                jaco[4][ii]=ag*y_sigmay*y_sigmay; //sigmay
            }
        }
    }
    else //only require residual errors
    {
        for(int i=0;i<xdim;i++)
        {
            double gx=a*exp(-(i-x0)*(i-x0)/sigmax);
            for(int j=0;j<ydim;j++)
            { 
                int ii=i*ydim+j;
                residual[ii]=gx*exp(-(j-y0)*(j-y0)/sigmay)-z[ii];  
            }
        }
    }
    return true;
};
//Gaussian v2
mycostfunction_gaussian_v2::~mycostfunction_gaussian_v2(){};
mycostfunction_gaussian_v2::mycostfunction_gaussian_v2(int n, double *x_, double *y_, double *z_)
{
    n_datapoint=n;
    x=x_;
    y=y_;
    z=z_;   
};


bool mycostfunction_gaussian_v2::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a=xx[0][0];
    double x0=xx[1][0];
    double sigmax=fabs(xx[2][0]);
    double y0=xx[3][0];
    double sigmay=fabs(xx[4][0]);
 

    if (jaco != NULL ) //both residual errors and jaco are required.
    {
        for(int i=0;i<n_datapoint;i++)
        {
            double x_sigmax=(x[i]-x0)/sigmax;
            double y_sigmay=(y[i]-y0)/sigmay;
            double g=exp((x0-x[i])*x_sigmax+(y0-y[i])*y_sigmay);
            double ag=a*g;

            residual[i]=ag-z[i]; 
            jaco[0][i]=g; //with respect to a
            jaco[1][i]=ag*2*x_sigmax; //x0
            jaco[2][i]=ag*x_sigmax*x_sigmax; //sigmax
            jaco[3][i]=ag*2*y_sigmay; //y0
            jaco[4][i]=ag*y_sigmay*y_sigmay; //sigmay
        }
    }
    else //only require residual errors
    {
        for(int i=0;i<n_datapoint;i++)
        { 
            residual[i]=a*exp(-(x[i]-x0)*(x[i]-x0)/sigmax-(y[i]-y0)*(y[i]-y0)/sigmay)-z[i];  
        }
    }
    return true;
};

//Voigt 
mycostfunction_voigt::~mycostfunction_voigt(){};
mycostfunction_voigt::mycostfunction_voigt(int xdim_, int ydim_, double *z_)
{
    xdim=xdim_;
    ydim=ydim_;
    n_datapoint=xdim*ydim;
    z=z_;   
};

void mycostfunction_voigt::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;
    double sigma3=sigma*sigma2;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    
    double t1=z_i*l-z_r*v;
    double t2=z_r*l+z_i*v;

    *r_x0=t1/(sigma2*M_SQRT_PI); 
    *r_gamma=(t2-M_1_SQRT_PI)/(sigma2*M_SQRT_PI);
    *r_sigma=-v/M_SQRT_2PI/sigma2 - t1*x0/sigma3/M_SQRT_PI - (t2-M_1_SQRT_PI)*gamma/sigma3/M_SQRT_PI;

    return;
};

void mycostfunction_voigt::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    return;
};

bool mycostfunction_voigt::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a=xx[0][0];
    double x0=xx[1][0];
    double sigmax=fabs(xx[2][0]);
    double gammax=fabs(xx[3][0]);
    double y0=xx[4][0];
    double sigmay=fabs(xx[5][0]);
    double gammay=fabs(xx[6][0]);
    
    double vvx,vvy,r_x,r_y,r_sigmax,r_sigmay,r_gammax,r_gammay;

    voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
    voigt_helper(y0, sigmay, gammay, &vvy, &r_y, &r_sigmay, &r_gammay);

    if (jaco != NULL ) //both residual errors and jaco are required.
    {
        for(int i=0;i<xdim;i++)
        {
            voigt_helper(i-x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax); 
            for(int j=0;j<ydim;j++)
            {
                int ii=i*ydim+j;
                voigt_helper(j-y0, sigmay, gammay, &vvy, &r_y, &r_sigmay, &r_gammay);
                residual[ii]=a*vvx*vvy-z[ii]; 
                jaco[0][ii]=vvx*vvy; //with respect to a
                jaco[1][ii]=-a*r_x*vvy; //x0
                jaco[2][ii]=a*r_sigmax*vvy; //sigmax
                jaco[3][ii]=a*r_gammax*vvy; //gammax
                jaco[4][ii]=-a*r_y*vvx; //y0
                jaco[5][ii]=a*r_sigmay*vvx; //sigmay
                jaco[6][ii]=a*r_gammay*vvx; //gammay
            }
        }
    }
    else //only require residual errors
    {
        for(int i=0;i<xdim;i++)
        {
            voigt_helper(i-x0, sigmax, gammax, &vvx);      
            for(int j=0;j<ydim;j++)
            {
                int ii=i*ydim+j;
                voigt_helper(j-y0, sigmay, gammay, &vvy);   
                residual[ii]=a*vvx*vvy-z[ii];    
            }
        }
    }
    return true;
};

//voigt_v2
mycostfunction_voigt_v2::~mycostfunction_voigt_v2(){};
mycostfunction_voigt_v2::mycostfunction_voigt_v2(int n, double *x_, double *y_, double *z_)
{
    n_datapoint=n;
    x=x_;
    y=y_;
    z=z_;   
};

void mycostfunction_voigt_v2::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;
    double sigma3=sigma*sigma2;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    
    double t1=z_i*l-z_r*v;
    double t2=z_r*l+z_i*v;

    *r_x0=t1/(sigma2*M_SQRT_PI); 
    *r_gamma=(t2-M_1_SQRT_PI)/(sigma2*M_SQRT_PI);
    *r_sigma=-v/M_SQRT_2PI/sigma2 - t1*x0/sigma3/M_SQRT_PI - (t2-M_1_SQRT_PI)*gamma/sigma3/M_SQRT_PI;

    return;
};

void mycostfunction_voigt_v2::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    return;
};

bool mycostfunction_voigt_v2::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a=xx[0][0];
    double x0=xx[1][0];
    double sigmax=fabs(xx[2][0]);
    double gammax=fabs(xx[3][0]);
    double y0=xx[4][0];
    double sigmay=fabs(xx[5][0]);
    double gammay=fabs(xx[6][0]);
    
    double vvx,vvy,r_x,r_y,r_sigmax,r_sigmay,r_gammax,r_gammay;

    voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
    voigt_helper(y0, sigmay, gammay, &vvy, &r_y, &r_sigmay, &r_gammay);

    if (jaco != NULL ) //both residual errors and jaco are required.
    {
        for(int i=0;i<n_datapoint;i++)
        {
            voigt_helper(x[i]-x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax); 
            voigt_helper(y[i]-y0, sigmay, gammay, &vvy, &r_y, &r_sigmay, &r_gammay);
            residual[i]=a*vvx*vvy-z[i]; 
            jaco[0][i]=vvx*vvy; //with respect to a
            jaco[1][i]=-a*r_x*vvy; //x0
            jaco[2][i]=a*r_sigmax*vvy; //sigmax
            jaco[3][i]=a*r_gammax*vvy; //gammax
            jaco[4][i]=-a*r_y*vvx; //y0
            jaco[5][i]=a*r_sigmay*vvx; //sigmay
            jaco[6][i]=a*r_gammay*vvx; //gammay
        }
    }
    else //only require residual errors
    {
        for(int i=0;i<n_datapoint;i++)
        {
            voigt_helper(x[i]-x0, sigmax, gammax, &vvx);      
            voigt_helper(y[i]-y0, sigmay, gammay, &vvy);   
            residual[i]=a*vvx*vvy-z[i];    
        }
    }
    return true;
};

//Voigt: intensity only
/*
mycostfunction_voigt_a::~mycostfunction_voigt_a(){};
mycostfunction_voigt_a::mycostfunction_voigt_a(int xdim_, int ydim_, double *z_,double x0_,double sigmax_,double gammax_,double y0_,double sigmay_,double gammay_)
{
    xdim=xdim_;
    ydim=ydim_;
    n_datapoint=xdim*ydim;
    z=z_;   

    x0=x0_;
    sigmax=sigmax_;
    gammax=gammax_;
    y0=y0_;
    sigmay=sigmay_;
    gammay=gammay_;
};

void mycostfunction_voigt_a::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;
    double sigma3=sigma*sigma2;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    
    double t1=z_i*l-z_r*v;
    double t2=z_r*l+z_i*v;

    *r_x0=t1/(sigma2*M_SQRT_PI); 
    *r_gamma=(t2-M_1_SQRT_PI)/(sigma2*M_SQRT_PI);
    *r_sigma=-v/M_SQRT_2PI/sigma2 - t1*x0/sigma3/M_SQRT_PI - (t2-M_1_SQRT_PI)*gamma/sigma3/M_SQRT_PI;

    return;
};

void mycostfunction_voigt_a::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    return;
};

bool mycostfunction_voigt_a::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a=xx[0][0];    
    double vvx,vvy,r_x,r_y,r_sigmax,r_sigmay,r_gammax,r_gammay;

    voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
    voigt_helper(y0, sigmay, gammay, &vvy, &r_y, &r_sigmay, &r_gammay);

    if (jaco != NULL ) //both residual errors and jaco are required.
    {
        for(int i=0;i<xdim;i++)
        {
            voigt_helper(i-x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
            for(int j=0;j<ydim;j++)
            {
                int ii=i*ydim+j;
                voigt_helper(j-y0, sigmay, gammay, &vvy, &r_y, &r_sigmay, &r_gammay);
                residual[ii]=a*vvx*vvy-z[ii]; 
                jaco[0][ii]=vvx*vvy; //with respect to a
            }
        }
    }
    else //only require residual errors
    {
        for(int i=0;i<xdim;i++)
        {
            voigt_helper(i-x0, sigmax, gammax, &vvx);
            for(int j=0;j<ydim;j++)
            {
                int ii=i*ydim+j;
                voigt_helper(j-y0, sigmay, gammay, &vvy);
                residual[ii]=a*vvx*vvy-z[ii]; 
            }
        }
    }
    return true;
};

*/

//class gaussian_fit

gaussian_fit::gaussian_fit()
{
    type=null_type; //need to be overwritten to gaussian or voigt
    to_remove.clear();
    sigmax.clear();
    sigmay.clear();
    gammax.clear();
    gammay.clear();
    a.clear();
    num_sum.clear();
    err.clear();
    gaussian_fit_wx=10000.0;
    gaussian_fit_wy=10000.0; //default use all data points to fit each peak
    nround=0;
    rmax=100;
    too_near_cutoff=0.20;
    //std::cout<<"creat gaussain fit "<<std::endl;

    options.max_num_iterations = 250;
    options.function_tolerance = 1e-12;
    options.parameter_tolerance =1e-12;
    options.initial_trust_region_radius = 150;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;


};
gaussian_fit::~gaussian_fit()
{
    //std::cout<<"delete gaussain fit "<<std::endl;
};



//gaussian fit of one peak
bool gaussian_fit::one_fit_gaussian(int xsize,int ysize,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay, double *e)
{
    ceres::Solver::Summary summary;
    ceres::Problem problem;

    sigmax=2*sigmax*sigmax;
    sigmay=2*sigmay*sigmay;

    mycostfunction_gaussian *cost_function = new mycostfunction_gaussian(xsize,ysize,zz->data());
    cost_function->set_n_residuals(zz->size());
    for(int m=0;m<5;m++) cost_function->parameter_block_sizes()->push_back(1); 
    problem.AddResidualBlock(cost_function, NULL, &a,&x0,&sigmax,&y0,&sigmay);

    
    // std::cout<<"before, xsize= "<<xsize<<" zzsize= "<<zz->size()<<" a= "<<a<<" x0= "<<x0<<" y0= "<<y0<<std::endl;
    ceres::Solve(options, &problem, &summary);
    // std::cout<<"after,  xsize= "<<xsize<<" zzsize= "<<zz->size()<<" a= "<<a<<" x0= "<<x0<<" y0= "<<y0<<std::endl<<std::endl;

    *e = sqrt(summary.final_cost / zz->size());

    // a*=max_a;

    sigmax=sqrt(fabs(sigmax)/2.0);
    sigmay=sqrt(fabs(sigmay)/2.0);

    // delete cost_function; //There is no need to do so because problem class will take care of it 
    return true;
};

bool gaussian_fit::one_fit_gaussian_v2(std::vector<double> *xx,std::vector<double> *yy,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay, double *e)
{
    ceres::Solver::Summary summary;
    ceres::Problem problem;

    sigmax=2*sigmax*sigmax;
    sigmay=2*sigmay*sigmay;

    mycostfunction_gaussian_v2 *cost_function = new mycostfunction_gaussian_v2(xx->size(),xx->data(),yy->data(),zz->data());
    cost_function->set_n_residuals(xx->size());
    for(int m=0;m<5;m++) cost_function->parameter_block_sizes()->push_back(1); 
    problem.AddResidualBlock(cost_function, NULL, &a,&x0,&sigmax,&y0,&sigmay);

    ceres::Solve(options, &problem, &summary);
    *e = sqrt(summary.final_cost / xx->size());

    sigmax=sqrt(fabs(sigmax)/2.0);
    sigmay=sqrt(fabs(sigmay)/2.0);

    return true;
};

bool gaussian_fit::one_fit_gaussian_intensity_only(int xsize,int ysize,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay, double *e)
{
    sigmax=2*sigmax*sigmax;
    sigmay=2*sigmay*sigmay;

    double s1,s2;
    s1=s2=0.0;
    for(int i=0;i<xsize;i++)
    {
        double v1=exp(-(x0-i)*(x0-i)/sigmax);
        for(int j=0;j<ysize;j++)
        {
            double v11=exp(-(y0-j)*(y0-j)/sigmay)*v1;
            double v2=zz->at(i*ysize+j);
            s1+=v11*v11;
            s2+=v11*v2;
        }
    }
    a=s2/s1;
    return true;
};

//gaussian fit of one peak in multiple spectra
bool gaussian_fit::multiple_fit_gaussian(int xsize,int ysize, std::vector< std::vector<double> > &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay, double *e)
{
    std::vector<mycostfunction_gaussian *> cost_functions(zz.size());

    ceres::Solver::Summary summary;
    ceres::Problem problem;
    sigmax=2*sigmax*sigmax;
    sigmay=2*sigmay*sigmay;
    for (unsigned int k = 0; k < zz.size(); k++)
    {
        cost_functions[k] = new mycostfunction_gaussian(xsize,ysize,zz[k].data());
        cost_functions[k]->set_n_residuals(zz[k].size());
        for (int m = 0; m < 5; m++) cost_functions[k]->parameter_block_sizes()->push_back(1);
        problem.AddResidualBlock(cost_functions[k], NULL, &(a[k]), &x, &sigmax, &y, &sigmay);
    }
    ceres::Solve(options, &problem, &summary);
    *e = sqrt(summary.final_cost);
    // std::cout << "after run: a=" << a[0] << " x=" << x << " y=" << y << " sx=" << sigmax << " sy=" << sigmay << std::endl;

    sigmax=sqrt(fabs(sigmax)/2.0);
    sigmay=sqrt(fabs(sigmay)/2.0);

    return true;
};

bool gaussian_fit::multiple_fit_gaussian_v2(std::vector<double> &xx,std::vector<double> &yy, std::vector< std::vector<double> > &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay, double *e)
{

    ceres::Solver::Summary summary;
    ceres::Problem problem;
    sigmax=2*sigmax*sigmax;
    sigmay=2*sigmay*sigmay;
    for (unsigned int k = 0; k < zz.size(); k++)
    {
        mycostfunction_gaussian_v2 * cost_function = new mycostfunction_gaussian_v2(xx.size(),xx.data(),yy.data(),zz[k].data());
        cost_function->set_n_residuals(zz[k].size());
        for (int m = 0; m < 5; m++) cost_function->parameter_block_sizes()->push_back(1);
        problem.AddResidualBlock(cost_function, NULL, &(a[k]), &x, &sigmax, &y, &sigmay);
    }
    ceres::Solve(options, &problem, &summary);
    *e = sqrt(summary.final_cost);
    // std::cout << "after run: a=" << a[0] << " x=" << x << " y=" << y << " sx=" << sigmax << " sy=" << sigmay << std::endl;

    sigmax=sqrt(fabs(sigmax)/2.0);
    sigmay=sqrt(fabs(sigmay)/2.0);

    return true;
};

//fit one Voigt intensity only, with jacobian!!
bool gaussian_fit::one_fit_voigt_intensity_only(int xsize,int ysize,std::vector<double> *zz, double &a,double x0,double y0,double sigmax,double sigmay,double gammax,double gammay,double *e)
{
    // double ee;
    // ceres::Solver::Options options;
    // options.max_num_iterations = 25;
    // options.linear_solver_type = ceres::DENSE_QR;
    // options.minimizer_progress_to_stdout = false;
    // ceres::Solver::Summary summary;
    // ceres::Problem problem;

    // mycostfunction_voigt_a *t = new mycostfunction_voigt_a(xsize,ysize,zz->data(),x0,sigmax,gammax,y0,sigmay,gammay);
    // t->set_n_residuals(zz->size());
    // t->parameter_block_sizes()->push_back(1); //number of fitting parameters
    // problem.AddResidualBlock(t, NULL, &a);

    // ceres::Solve(options, &problem, &summary);
    // *e = sqrt(summary.final_cost / zz->size());

    double s1,s2;
    s1=s2=0.0;
    for(int i=0;i<xsize;i++)
    {
        double v1=voigt(x0-i,sigmax,gammax);
        for(int j=0;j<ysize;j++)
        {
            double v11=voigt(y0-j,sigmay,gammay)*v1;
            double v2=zz->at(i*ysize+j);
            s1+=v11*v11;
            s2+=v11*v2;
        }
    }
    a=s2/s1;

    return true;
};



//fit one Voigt, with jacobian!!
bool gaussian_fit::one_fit_voigt(int xsize,int ysize,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay,double &gammax,double &gammay,double *e, int n)
{

    if(n==0)
    {
    
        double ee;
        double test_gammax[3],test_gammay[3];

        double fwhhx=0.5346*gammax*2+std::sqrt(0.2166*4*gammax*gammax+sigmax*sigmax*8*0.6931);
        double fwhhy=0.5346*gammay*2+std::sqrt(0.2166*4*gammay*gammay+sigmay*sigmay*8*0.6931);

        test_gammax[0]=1e-9;
        test_gammax[1]=fwhhx/4.0;
        test_gammax[2]=fwhhx/2.0*0.98;

        test_gammay[0]=1e-9;
        test_gammay[1]=fwhhy/4.0;
        test_gammay[2]=fwhhy/2.0*0.98;

        double inten=a*voigt(0,sigmax,gammax)*voigt(0,sigmay,gammay);
        double min_e=std::numeric_limits<double>::max();

        double x0_try,y0_try,a_try;

        for(int i=0;i<3;i++)
        {
            double gammax_try=test_gammax[i];
            double sigmax_try=sqrt(((fwhhx-1.0692*gammax_try)*(fwhhx-1.0692*gammax_try)-0.2166*4*gammax_try*gammax_try)/(8*0.6931));
            for(int j=0;j<3;j++)
            {
                double gammay_try=test_gammay[j];
                double sigmay_try=sqrt(((fwhhy-1.0692*gammay_try)*(fwhhy-1.0692*gammay_try)-0.2166*4*gammay_try*gammay_try)/(8*0.6931));
                a_try=inten/(voigt(0,sigmax_try,gammax_try)*voigt(0,sigmay_try,gammay_try));
                x0_try=x0;
                y0_try=y0;
                // std::cout<<" sigmax="<<sigmax_try<<" sigmay="<<sigmay_try<<" gammax="<<gammax_try<<" gammay_try="<<gammay_try;
                one_fit_voigt_core(xsize,ysize, zz, x0_try, y0_try, a_try, sigmax_try, sigmay_try,gammax_try,gammay_try,&ee); 
                // std::cout<<" ee="<<ee<<std::endl;
                if(ee<min_e)
                {
                    min_e=ee;
                    *e=ee;
                    x0=x0_try;
                    y0=y0_try;
                    a=a_try;
                    sigmax=sigmax_try;
                    sigmay=sigmay_try;
                    gammax=gammax_try;
                    gammay=gammay_try;
                }
            }
        }
    }
    
    else
    {
        one_fit_voigt_core(xsize,ysize, zz, x0, y0, a, sigmax, sigmay,gammax,gammay,e);
    }
    
    
    return true;
};

bool gaussian_fit::one_fit_voigt_core(int xsize,int ysize,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay,double &gammax,double &gammay,double *e)
{

    ceres::Solver::Summary summary;
    ceres::Problem problem;

    mycostfunction_voigt *cost_function = new mycostfunction_voigt(xsize,ysize,zz->data());
    cost_function->set_n_residuals(zz->size());
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    problem.AddResidualBlock(cost_function, NULL, &a,&x0,&sigmax,&gammax,&y0,&sigmay,&gammay);

    ceres::Solve(options, &problem, &summary);
    *e = sqrt(summary.final_cost / zz->size());


    sigmax=fabs(sigmax);
    sigmay=fabs(sigmay);
    gammax=fabs(gammax);
    gammay=fabs(gammay);
    

    return true;
};

bool gaussian_fit::one_fit_voigt_v2(std::vector<double> *xx,std::vector<double> *yy,std::vector<double> *zz, double &x0,double &y0,double &a,double &sigmax,double &sigmay,double &gammax,double &gammay,double *e, int n)
{
    

    double ee;
    //std::cout<<"before guassian one_fit, x="<<x0<<" y="<<y0<<" a="<<a<<" sx="<<sigmax<<" sy="<<sigmay<<std::endl;
    if(n==0)
    {
        //run gaussian fit first, then add Voigt term gamma
        
        //conversion from voigt to gaussian, firstly make a voigt with 0 gammay, then convert to unit in Gaussian fitting
        a=a*(voigt(0.0,sigmax,gammax)*voigt(0.0,sigmay,gammay));
        sigmax=(0.5346*gammax*2+std::sqrt(0.2166*4*gammax*gammax+sigmax*sigmax*8*0.6931))/2.355;
        sigmay=(0.5346*gammay*2+std::sqrt(0.2166*4*gammay*gammay+sigmay*sigmay*8*0.6931))/2.355;

        //gaussian fit
        one_fit_gaussian_v2(xx,yy,zz,x0,y0,a,sigmax,sigmay,&ee);
        
        //convert back to voigt representation, suppose gammay --> 0
        gammax=2.0;
        gammay=1e-9;

        a/=voigt(0,sigmax,gammax)*voigt(0,sigmay,gammay);
    }
    //std::cout<<"after guassian one_fit, x="<<x0<<" y="<<y0<<" a="<<a<<" sx="<<sigmax<<" sy="<<sigmay<<std::endl;

    // double x[7];

    // x[0]=a;
    // x[1]=x0;
    // x[2]=sigmax;
    // x[3]=gammax;
    // x[4]=y0;
    // x[5]=sigmay;
    // x[6]=gammay;

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Problem problem;


    mycostfunction_voigt_v2 *cost_function = new mycostfunction_voigt_v2(xx->size(),xx->data(),yy->data(),zz->data());
    cost_function->set_n_residuals(xx->size());
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    cost_function->parameter_block_sizes()->push_back(1); //number of fitting parameters
    problem.AddResidualBlock(cost_function, NULL, &a,&x0,&sigmax,&gammax,&y0,&sigmay,&gammay);

    ceres::Solve(options, &problem, &summary);
    *e = sqrt(summary.final_cost / xx->size());

    std::cout<<"Cost is "<<*e<<std::endl;

    // a = x[0];
    // x0 = x[1];
    // sigmax = fabs(x[2]);
    // gammax = fabs(x[3]);
    // y0 = x[4];
    // sigmay = fabs(x[5]);
    // gammay = fabs(x[6]);

    sigmax=fabs(sigmax);
    sigmay=fabs(sigmay);
    gammax=fabs(gammax);
    gammay=fabs(gammay);
    

    //std::cout<<"aftert voigt one_fit, x="<<x0<<" y="<<y0<<" a="<<a<<" sx="<<sigmax<<" sy="<<sigmay<<" gx="<<gammax<<" gy="<<gammay<<std::endl;

    return true;
};

bool gaussian_fit::multiple_fit_voigt(int xsize,int ysize, std::vector<std::vector<double>> &zz, double &x0, double &y0, std::vector<double> &a, double &sigmax, double &sigmay, double &gammax, double &gammay, double *e, int n)
{
    if(n==0)
    {
    
        double ee;
        double test_gammax[3],test_gammay[3];

        double fwhhx=0.5346*gammax*2+std::sqrt(0.2166*4*gammax*gammax+sigmax*sigmax*8*0.6931);
        double fwhhy=0.5346*gammay*2+std::sqrt(0.2166*4*gammay*gammay+sigmay*sigmay*8*0.6931);

        test_gammax[0]=1e-9;
        test_gammax[1]=fwhhx/4.0;
        test_gammax[2]=fwhhx/2.0*0.98;

        test_gammay[0]=1e-9;
        test_gammay[1]=fwhhy/4.0;
        test_gammay[2]=fwhhy/2.0*0.98;

        std::vector<double> inten(a.size()),a_try(a.size());
        for(int i=0;i<a.size();i++)
        {
            inten[i]=a[i]*voigt(0,sigmax,gammax)*voigt(0,sigmay,gammay); 
        }
        
        double min_e=std::numeric_limits<double>::max();

        double x0_try,y0_try;

        for(int i=0;i<3;i++)
        {
            double gammax_try=test_gammax[i];
            double sigmax_try=sqrt(((fwhhx-1.0692*gammax_try)*(fwhhx-1.0692*gammax_try)-0.2166*4*gammax_try*gammax_try)/(8*0.6931));
            for(int j=0;j<3;j++)
            {
                double gammay_try=test_gammay[j];
                double sigmay_try=sqrt(((fwhhy-1.0692*gammay_try)*(fwhhy-1.0692*gammay_try)-0.2166*4*gammay_try*gammay_try)/(8*0.6931));
                for(int i=0;i<a.size();i++)
                {
                    a_try[i]=inten[i]/(voigt(0,sigmax_try,gammax_try)*voigt(0,sigmay_try,gammay_try));
                }
                ;
                x0_try=x0;
                y0_try=y0;
                // std::cout<<" sigmax="<<sigmax_try<<" sigmay="<<sigmay_try<<" gammax="<<gammax_try<<" gammay_try="<<gammay_try;
                multiple_fit_voigt_core(xsize,ysize, zz, x0_try, y0_try, a_try, sigmax_try, sigmay_try,gammax_try,gammay_try,&ee); 
                // std::cout<<" ee="<<ee<<std::endl;
                if(ee<min_e)
                {
                    min_e=ee;
                    *e=ee;
                    x0=x0_try;
                    y0=y0_try;
                    a=a_try;
                    sigmax=sigmax_try;
                    sigmay=sigmay_try;
                    gammax=gammax_try;
                    gammay=gammay_try;
                }
            }
        }
    }

    else
    {
        multiple_fit_voigt_core(xsize,ysize,zz,x0,y0,a,sigmax,sigmay,gammax,gammay,e);    
    }
   

    return true;
};



bool gaussian_fit::multiple_fit_voigt_core(int xsize,int ysize, std::vector<std::vector<double>> &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay, double &gammax, double &gammay, double *e)
{

    ceres::Solver::Summary summary;
    ceres::Problem problem;

    for (unsigned int k = 0; k < zz.size(); k++)
    {
        mycostfunction_voigt *cost_function = new mycostfunction_voigt(xsize,ysize, zz[k].data());
        cost_function->set_n_residuals(zz[k].size());
        for (int m = 0; m < 7; m++)
            cost_function->parameter_block_sizes()->push_back(1);
        problem.AddResidualBlock(cost_function, NULL, &(a[k]), &x, &sigmax, &gammax, &y, &sigmay, &gammay);
    }
    ceres::Solve(options, &problem, &summary);
    *e = sqrt(summary.final_cost);

    sigmax = fabs(sigmax);
    sigmay = fabs(sigmay);
    gammax = fabs(gammax);
    gammay = fabs(gammay);

    return true;
};

bool gaussian_fit::multiple_fit_voigt_v2(std::vector<double> &xx,std::vector<double> &yy, std::vector<std::vector<double>> &zz, double &x, double &y, std::vector<double> &a, double &sigmax, double &sigmay, double &gammax, double &gammay, double *e, int n)
{
    if (n == 0)
    {
        //run gaussian fit first, then add Voigt term gamma
        //conversion from voigt to gaussian, firstly make a voigt with 0 gammay, then convert to unit in Gaussian fitting

        for (int i = 0; i < a.size(); i++)
        {
            a[i] = a[i] * (voigt(0.0, sigmax, gammax) * voigt(0.0, sigmay, gammay));
        }

        sigmax = (0.5346 * gammax * 2 + std::sqrt(0.2166 * 4 * gammax * gammax + sigmax * sigmax * 8 * 0.6931)) / 2.355;
        sigmay = (0.5346 * gammay * 2 + std::sqrt(0.2166 * 4 * gammay * gammay + sigmay * sigmay * 8 * 0.6931)) / 2.355;


        // gaussian fit
        ceres::Solver::Summary summary;
        ceres::Problem problem;

        sigmax=2*sigmax*sigmax;
        sigmay=2*sigmay*sigmay;

        for (unsigned int k = 0; k < zz.size(); k++)
        {
            mycostfunction_gaussian_v2 * cost_function = new mycostfunction_gaussian_v2(xx.size(),xx.data(),yy.data(),zz[k].data());
            cost_function->set_n_residuals(zz[k].size());
            for (int m = 0; m < 5; m++) cost_function->parameter_block_sizes()->push_back(1);
            problem.AddResidualBlock(cost_function, NULL, &(a[k]), &x, &sigmax, &y, &sigmay);
        }
        ceres::Solve(options, &problem, &summary);

        sigmax=sqrt(fabs(sigmax)/2.0);
        sigmay=sqrt(fabs(sigmay)/2.0);

        //convert back to voigt representation, suppose gammay --> 0
        gammax = 0.0000001;
        gammay = 0.0000001;
        sigmax = sqrt(sigmax / 2.0);
        sigmay = sqrt(sigmay / 2.0);
        for (int i = 0; i < a.size(); i++)
        {
            a[i] /= voigt(0, sigmax, gammax) * voigt(0, sigmay, gammay);
        }
    }


    ceres::Solver::Summary summary;
    ceres::Problem problem;

    for (unsigned int k = 0; k < zz.size(); k++)
    {
        mycostfunction_voigt_v2 *cost_function = new mycostfunction_voigt_v2(xx.size(),xx.data(),yy.data(),zz[k].data());
        cost_function->set_n_residuals(zz[k].size());
        for (int m = 0; m < 7; m++)
            cost_function->parameter_block_sizes()->push_back(1);
        problem.AddResidualBlock(cost_function, NULL, &(a[k]), &x, &sigmax, &gammax, &y, &sigmay, &gammay);
    }
    ceres::Solve(options, &problem, &summary);
    *e = sqrt(summary.final_cost);

    sigmax = fabs(sigmax);
    sigmay = fabs(sigmay);
    gammax = fabs(gammax);
    gammay = fabs(gammay);

    return true;
};

bool gaussian_fit::gaussain_convolution(double a, double x, double y, double sigmax, double sigmay, std::vector<double> *kernel,double scale)
{
    float wx=2.3548*sigmax*scale;
    float wy=2.3548*sigmay*scale;
    
    i0=std::max(0,int(x-wx+0.5));
    i1=std::min(xdim,int(x+wx+0.5));
    j0=std::max(0,int(y-wy+0.5));
    j1=std::min(ydim,int(y+wy+0.5));

    kernel->clear();
    kernel->resize((i1-i0)*(j1-j0));

    sigmax=2*sigmax*sigmax;
    sigmay=2*sigmay*sigmay;
    
    
    for (int i =i0; i < i1; i++)
    {
        double t1=x-i;
        for (int j = j0; j < j1; j++)
        {
            double t2=y-j;
            kernel->at((i-i0)*(j1-j0)+j-j0)=a*exp(-(t1 * t1 ) / sigmax-(t2*t2)/sigmay);
        }
    }
    return true;
};

bool gaussian_fit::gaussain_convolution_within_region(int ndx,double a, double x, double y, double sigmax, double sigmay, std::vector<double> *kernel,double scale)
{
    float wx=2.3548*sigmax*scale;
    float wy=2.3548*sigmay*scale;
    
    std::array<int,4> add_limit=valid_fit_region.at(ndx);

    i0=std::max(std::max(0,int(x-wx+0.5)),add_limit[0]);
    i1=std::min(std::min(xdim,int(x+wx+0.5)),add_limit[1]);
    j0=std::max(std::max(0,int(y-wy+0.5)),add_limit[2]);
    j1=std::min(std::min(ydim,int(y+wy+0.5)),add_limit[3]);


    kernel->clear();
    kernel->resize((i1-i0)*(j1-j0));

    sigmax=2*sigmax*sigmax;
    sigmay=2*sigmay*sigmay;
    
    
    for (int i =i0; i < i1; i++)
    {
        double t1=x-i;
        for (int j = j0; j < j1; j++)
        {    
            double t2=y-j;
            kernel->at((i-i0)*(j1-j0)+j-j0)=a*exp(-(t1 * t1 ) / (sigmax)-(t2*t2)/(sigmay));
        }
    }
    return true;
};

bool gaussian_fit::gaussain_convolution_region(double x, double y, double sigmax, double sigmay,double scale)
{
    float wx=2.3548*sigmax*scale;
    float wy=2.3548*sigmay*scale;
    
    i0=std::max(0,int(x-wx+0.5));
    i1=std::min(xdim,int(x+wx+0.5));
    j0=std::max(0,int(y-wy+0.5));
    j1=std::min(ydim,int(y+wy+0.5));
    return true;
};

bool gaussian_fit::voigt_convolution(double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay, std::vector<double> *kernel,double scale)
{
    float wx=(1.0692*gammax+sqrt(0.8664*gammax*gammax+5.5452*sigmax*sigmax))*scale;
    float wy=(1.0692*gammay+sqrt(0.8664*gammay*gammay+5.5452*sigmay*sigmay))*scale;

    i0=std::max(0,int(x-wx+0.5));
    i1=std::min(xdim,int(x+wx+0.5));
    j0=std::max(0,int(y-wy+0.5));
    j1=std::min(ydim,int(y+wy+0.5));

    kernel->clear();
    kernel->resize((i1-i0)*(j1-j0));
    
    for (int i =i0; i < i1; i++)
    {
        for (int j = j0; j < j1; j++)
        {
            double z1=voigt ( i-x, sigmax, gammax );
            double z2=voigt ( j-y, sigmay, gammay );
            kernel->at((i-i0)*(j1-j0)+j-j0)=a*z1*z2;
        }
    }
    return true;
};

bool gaussian_fit::voigt_convolution_within_region(int ndx,double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay, std::vector<double> *kernel,double scale)
{
    float wx=(1.0692*gammax+sqrt(0.8664*gammax*gammax+5.5452*sigmax*sigmax))*scale;
    float wy=(1.0692*gammay+sqrt(0.8664*gammay*gammay+5.5452*sigmay*sigmay))*scale;

    std::array<int,4> add_limit=valid_fit_region.at(ndx);

    i0=std::max(std::max(0,int(x-wx+0.5)),add_limit[0]);
    i1=std::min(std::min(xdim,int(x+wx+0.5)),add_limit[1]);
    j0=std::max(std::max(0,int(y-wy+0.5)),add_limit[2]);
    j1=std::min(std::min(ydim,int(y+wy+0.5)),add_limit[3]);

    kernel->clear();
    kernel->resize((i1-i0)*(j1-j0));
    
    for (int i =i0; i < i1; i++)
    {
        for (int j = j0; j < j1; j++)
        {
            double z1=voigt ( i-x, sigmax, gammax );
            double z2=voigt ( j-y, sigmay, gammay );
            kernel->at((i-i0)*(j1-j0)+j-j0)=a*z1*z2;
        }
    }
    return true;
};

bool gaussian_fit::voigt_convolution_region(double x, double y, double sigmax, double sigmay, double gammax, double gammay,double scale)
{
    float wx=(1.0692*gammax+sqrt(0.8664*gammax*gammax+5.5452*sigmax*sigmax))*scale;
    float wy=(1.0692*gammay+sqrt(0.8664*gammay*gammay+5.5452*sigmay*sigmay))*scale;
    i0=std::max(0,int(x-wx+0.5));
    i1=std::min(xdim,int(x+wx+0.5));
    j0=std::max(0,int(y-wy+0.5));
    j1=std::min(ydim,int(y+wy+0.5));
    return true;
};


bool gaussian_fit::init(int x00,int y00,int xdim_, int ydim_, std::vector< std::vector<double> >surface_, std::vector<double> x_, std::vector<double> y_, 
                    std::vector< std::vector<double> > aa, std::vector<double> sx, std::vector<double> sy, std::vector<int> ns, std::vector<int> move_)
{
    //at beginning, x and y are all integers!
    xstart=x00;
    ystart=y00;
    xdim=xdim_;
    ydim=ydim_;
    surface=surface_;
    x=x_;
    y=y_;

    sigmax=sx;  //This is actually fwhh at this time, will be converted to sigma below
    sigmay=sy;
    
    a=aa;
    original_ndx=ns;
    cannot_move=move_;

    //std::cout<<"In this region, npeak is "<<x.size()<<std::endl;
    for(int i=x.size()-1;i>=0;i--)
    {
        if(x[i]<0.1 || y[i]<0.1 || x[i]>xdim-1.1 || y[i]>ydim-1.1)
        {
            a.erase(a.begin()+i);
            x.erase(x.begin() + i);
            y.erase(y.begin() + i);
            sigmax.erase(sigmax.begin() + i);
            sigmay.erase(sigmay.begin() + i);
            original_ndx.erase(original_ndx.begin()+i);
            cannot_move.erase(cannot_move.begin()+i);
        }
    }

    original_ratio.resize(x.size(),0);
    for (int i = 0; i < x.size(); i++)
    {
        int nt = int(round(x[i])) * ydim + int(round(y[i]));
        original_ratio[i] = surface[0][nt];
    }

    //std::cout<<"In this region, after removal process, npeak is "<<x.size()<<std::endl;
    int npeak=x.size();
    
    x_int.clear();  //x_int and y_int is not used, to be removed.
    y_int.clear();
    
    gammax.clear();
    gammay.clear();
    err.clear();
    num_sum.clear();

    to_remove.clear();
    to_remove.resize(npeak,0);


    for(int i=0;i<npeak;i++)
    {
        x_int.push_back((int)round(x.at(i)));
        y_int.push_back((int)round(y.at(i)));
        sigmax[i]=fabs(sigmax[i]/2.355);
        sigmay[i]=fabs(sigmay[i]/2.355);
    }

    peaks_total.clear();
    peaks_total.resize(xdim*ydim,0.0);
    gammax.resize(npeak,0.00000001);
    gammay.resize(npeak,0.00000001);
    err.resize(npeak,0.0);
    num_sum.resize(npeak,std::vector<double>(surface.size(),0.0));

    //In gaussian fitting, sigma is actually 2*sigma*sigma.
    //But in voigt fitting, sigma is sigma and gamma is gamma.
    //in init, we always receive 2*sigma*sigma in sigma vector. 
    if(type==voigt_type)
    {
        for(int i=0;i<npeak;i++)
        {
            //becase a[i] is the max height of the peak we estiamted before fitting
            //After voigt fitting, a[i]*voigt(x)*voigt(y) is the max height
            //After Gaussian fittting, a[i] is the max height. 
            //In other words, voigt function has the prefactor in its defination but gaussian's is always 1.0
            for(int j=0;j<surface.size();j++)
            {
                a[i][j]/=(voigt(0.0,sigmax[i],gammax[i])*voigt(0.0,sigmay[i],gammay[i]));  
            }
        }
    }
    return true;
};

bool gaussian_fit::find_highest_neighbor(int xx,int yy,int &mm,int &nn)
{
    bool b_already_at_max=true;
    double current_a=surface[0][xx*ydim+yy];
    double a_difference=0.0;
    mm=0;
    nn=0;

    for (int m = -1; m <= 1; m++)
    {
        for (int n = -1; n <= 1; n++)
        {
            if (surface[0][(xx+m)*ydim+yy+n]-current_a>a_difference)
            {
                a_difference=surface[0][(xx+m)*ydim+yy+n]-current_a;  
                mm=m;
                nn=n; 
                b_already_at_max=false;
            }
        }
    }

    return b_already_at_max;
}


bool gaussian_fit::limit_fitting_region_of_each_peak()
{
    int npeak=x.size();
    for(int ndx=0;ndx<npeak;ndx++)
    {
        int xx=round(x[ndx]);
        int yy=round(y[ndx]);
        int m,n;

        std::array<int,4> region={0,xdim,0,ydim};
        find_highest_neighbor(xx,yy,m,n);xx+=m;yy+=n;
        find_highest_neighbor(xx,yy,m,n);xx+=m;yy+=n; //move the local  maximal
        if (find_highest_neighbor(xx, yy, m, n) == false) //this is a shoulder peak, restore initial coordinate
        {
            xx = round(x[ndx]);
            yy = round(y[ndx]);
        }

        double current_a = surface[0][xx * ydim + yy];
        bool b;

        b=true;
        for (int i = xx - 1; i >= std::max(0, xx - int(gaussian_fit_wx)); i--)
        {
            region[0] = i + 1;
            if (surface[0][i * ydim + yy] > surface[0][(i + 1) * ydim + yy] && surface[0][(i+2) * ydim + yy] > surface[0][(i + 1) * ydim + yy])
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[0]=0;
        }

        b=true;
        for (int i = xx + 1; i < std::min(xdim, xx + int(gaussian_fit_wx)); i++)
        {
            region[1] = i;
            if (surface[0][i * ydim + yy] > surface[0][(i - 1) * ydim + yy] && surface[0][(i-2) * ydim + yy] > surface[0][(i - 1) * ydim + yy] )
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[1]=xdim;
        }

        b=true;
        for (int j = yy - 1; j >= std::max(0, yy - int(gaussian_fit_wy)); j--)
        {
            region[2] = j + 1;
            if (surface[0][xx * ydim + j] > surface[0][xx * ydim + j + 1] && surface[0][xx * ydim + j+2] > surface[0][xx * ydim + j + 1])
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[2]=0;
        }

        b=true;
        for (int j = yy + 1; j < std::min(ydim, yy + int(gaussian_fit_wy)); j++)
        {
            region[3] = j;
            if (surface[0][xx * ydim + j] > surface[0][xx * ydim + j - 1] && surface[0][xx * ydim + j-2] > surface[0][xx * ydim + j - 1])
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[3]=ydim;
        }

        //expand by 1 point
        if(region[0]>0) region[0]-=1;
        if(region[2]>0) region[2]-=1;
        if(region[1]<xdim-1) region[1]+=1;
        if(region[3]<ydim-1) region[3]+=1;
        
    

        valid_fit_region.push_back(region);

        // std::cout<<"Peak "<<ndx<<" x="<<x[ndx]<<" y="<<y[ndx]<<" region is "<<region[0]<<" "<<region[1]<<" "<<region[2]<<" "<<region[3]<<std::endl;

    }
    return true;
};

bool gaussian_fit::limit_fitting_region_of_one_peak()
{
    int npeak=x.size();
    for(int ndx=0;ndx<npeak;ndx++)
    {
        int xx=round(x[ndx]);
        int yy=round(y[ndx]);
        int m,n;

        std::array<int,4> region={0,xdim,0,ydim};
        find_highest_neighbor(xx,yy,m,n);xx+=m;yy+=n;
        find_highest_neighbor(xx,yy,m,n);xx+=m;yy+=n; //move the local  maximal
        if (find_highest_neighbor(xx, yy, m, n) == false) //this is a shoulder peak, restore initial coordinate
        {
            xx = round(x[ndx]);
            yy = round(y[ndx]);
        }

        double current_a = surface[0][xx * ydim + yy];
        bool b;

        b=true;
        for (int i = xx - 1; i >= std::max(0, xx - int(gaussian_fit_wx)); i--)
        {
            region[0] = i + 1;
            if (surface[0][i * ydim + yy] < noise_level*3.0)
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[0]=0;
        }

        b=true;
        for (int i = xx + 1; i < std::min(xdim, xx + int(gaussian_fit_wx)); i++)
        {
            region[1] = i;
            if (surface[0][i * ydim + yy] < noise_level*3.0 )
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[1]=xdim;
        }

        b=true;
        for (int j = yy - 1; j >= std::max(0, yy - int(gaussian_fit_wy)); j--)
        {
            region[2] = j + 1;
            if (surface[0][xx * ydim + j] < noise_level*3.0)
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[2]=0;
        }

        b=true;
        for (int j = yy + 1; j < std::min(ydim, yy + int(gaussian_fit_wy)); j++)
        {
            region[3] = j;
            if (surface[0][xx * ydim + j] < noise_level*3.0)
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[3]=ydim;
        }

        //expand by 1 point
        if(region[0]>0) region[0]-=1;
        if(region[2]>0) region[2]-=1;
        if(region[1]<xdim-1) region[1]+=1;
        if(region[3]<ydim-1) region[3]+=1;
        
    

        valid_fit_region.push_back(region);

        // std::cout<<"Peak "<<ndx<<" x="<<x[ndx]<<" y="<<y[ndx]<<" region is "<<region[0]<<" "<<region[1]<<" "<<region[2]<<" "<<region[3]<<std::endl;

    }
    return true;
};


bool gaussian_fit::run()
{
    if (type == null_type)
        return false;
    
    bool b;
    

    if(surface.size()==1) //single spectrum to fit
    {
        if(x.size()==1) 
        {
            limit_fitting_region_of_each_peak();//this is done using first spectru only at this time!!!!
            b=run_single_peak();
        }
        else 
        {
            limit_fitting_region_of_each_peak();//this is done using first spectru only at this time!!!!
            b=run_multi_peaks(rmax);
        }
    }
    else //multiple spectra from pseudo 3D 
    {
        if(x.size()==1) 
        {
            limit_fitting_region_of_each_peak();//this is done using first spectru only at this time!!!!
            int c=0;
            b=multi_spectra_run_single_peak();
            while(b && c<0 ) //run at least 1 round
            {
                b=multi_spectra_run_single_peak();
                ++c;
            }
        }
        else 
        {
            limit_fitting_region_of_each_peak();//this is done using first spectru only at this time!!!!
            b=multi_spectra_run_multi_peaks(rmax);
        }
    }

    return b;
}


bool gaussian_fit::run_single_peak()
{
    bool b=false;
    std::vector<double> zz0;
    std::vector<double> xx,yy,zz;
    double total_z = 0.0;
    double e;

    std::array<int,4> add_limit=valid_fit_region.at(0);

    float wx=(1.0692*gammax[0]+sqrt(0.8664*gammax[0]*gammax[0]+5.5452*sigmax[0]*sigmax[0]))*2.0;
    float wy=(1.0692*gammay[0]+sqrt(0.8664*gammay[0]*gammay[0]+5.5452*sigmay[0]*sigmay[0]))*2.0;

    i0=std::max(std::max(0,int(x[0]-wx+0.5)),add_limit[0]);
    i1=std::min(std::min(xdim,int(x[0]+wx+0.5)),add_limit[1]);
    j0=std::max(std::max(0,int(y[0]-wy+0.5)),add_limit[2]);
    j1=std::min(std::min(ydim,int(y[0]+wy+0.5)),add_limit[3]);

    for (int ii = i0; ii < i1; ii++)
    {
        for (int jj = j0; jj < j1; jj++)
        {
            double temp = surface[0][ii * ydim + jj];
            zz0.push_back(temp);

            if(temp>noise_level*3.0)
            {
                xx.push_back(ii-i0);
                yy.push_back(jj-j0);
                zz.push_back(temp);
            }

            total_z += temp;
        }
    }

    double current_x=x[0]-i0;
    double current_y=y[0]-j0;

    int i = 0;
    if (type == gaussian_type) //gaussian
    {
        one_fit_gaussian(i1-i0,j1-j0, &zz0, current_x, current_y, a[i][0], sigmax.at(i), sigmay.at(i),&e);
        // one_fit_gaussian_v2(&xx,&yy, &zz, current_x, current_y, a[i][0], sigmax.at(i), sigmay.at(i),&e);
        
        if (fabs(sigmax.at(i)) < 0.02 || fabs(sigmay.at(i)) < 0.02 || current_x < 0 || current_x >= xdim || current_y < 0 || current_y >= ydim)
        {
            to_remove[i] = 1; //this is a flag only
            std::cout<<"remove failed peak because "<<fabs(sigmax.at(i))<<" "<<fabs(sigmay.at(i))<<" "<<current_x<<" "<<current_y<<std::endl;
        }
        if(a[i][0]<minimal_height)
        {
            to_remove[i] = 1; //this is a flag only
            std::cout<<"remove too lower peak "<<a[i][0]<<" "<<fabs(sigmax.at(i))<<" "<<fabs(sigmay.at(i))<<" "<<fabs(gammax.at(i))<<" "<<fabs(gammay.at(i))<<" "<<current_x<<" "<<current_y<<std::endl;
        }
    }
    else if(type == voigt_type)//voigt
    {
        one_fit_voigt(i1-i0,j1-j0, &zz0, current_x, current_y, a[i][0], sigmax.at(i), sigmay.at(i),gammax.at(i),gammay.at(i),&e,0);
        // one_fit_voigt_v2(&xx,&yy, &zz, current_x, current_y, a[i][0], sigmax.at(i), sigmay.at(i),gammax.at(i),gammay.at(i),&e,0);
        
        if (fabs(sigmax.at(i)) + fabs(gammax.at(i)) < 0.2 || fabs(sigmay.at(i)) + fabs(gammay.at(i)) < 0.2 || current_x < 0 || current_x >= xdim || current_y < 0 || current_y >= ydim)
        {
            std::cout<<"remove failed peak because "<<a[i][0]<<" "<<fabs(sigmax.at(i))<<" "<<fabs(sigmay.at(i))<<" "<<fabs(gammax.at(i))<<" "<<fabs(gammay.at(i))<<" "<<current_x<<" "<<current_y<<std::endl;
            to_remove[i] = 1; //this is a flag only
        }

        double true_height=a[i][0]*(voigt(0.0,sigmax[i],gammax[i])*voigt(0.0,sigmay[i],gammay[i]));
        if(true_height<minimal_height)
        {
            std::cout<<"remove too lower peak "<<a[i][0]<<" "<<fabs(sigmax.at(i))<<" "<<fabs(sigmay.at(i))<<" "<<fabs(gammax.at(i))<<" "<<fabs(gammay.at(i))<<" "<<current_x<<" "<<current_y<<std::endl;
            to_remove[i] = 1; //this is a flag only
        }
    }
    nround=1;
    x[0]=current_x+i0;
    y[0]=current_y+j0;

    
    if(to_remove[i]==0)
    {
        num_sum[i][0] = total_z;
        err.at(i)=e;
        sigmax[i]=fabs(sigmax[i]);
        sigmay[i]=fabs(sigmay[i]);
        gammax[i]=fabs(gammax[i]);
        gammay[i]=fabs(gammay[i]);
        b= true;
    }
    else
    {
        a.clear();
        sigmax.clear();
        sigmay.clear();
        gammax.clear();
        gammay.clear();
        num_sum.clear();
        err.clear();
        x.clear();
        y.clear();
        b= false;
    }
    return b;
}

bool gaussian_fit::run_multi_peaks(int loop_max)
{

    int npeak = x.size();

    double e;

    x_old.clear();
    y_old.clear();

    bool flag_break = false;
    int loop;
    for (loop = 0; loop < loop_max; loop++)
    {
        analytical_spectra.clear();
        analytical_spectra.resize(x.size());

        peaks_total.clear();
        peaks_total.resize(xdim*ydim,0.0);
       
        #pragma omp parallel for
        for (unsigned int i_peak = 0; i_peak < x.size(); i_peak++)
        {
            if(to_remove[i_peak]==1) //peak has been removed. 
            {
                analytical_spectra[i_peak].clear();
                continue;   
            }
            
            if (type == gaussian_type)
                gaussain_convolution(a[i_peak][0], x.at(i_peak), y.at(i_peak), sigmax.at(i_peak), sigmay.at(i_peak), &(analytical_spectra[i_peak]),2.0);
            else if(type == voigt_type)
                voigt_convolution(a[i_peak][0], x.at(i_peak), y.at(i_peak), sigmax.at(i_peak), sigmay.at(i_peak), gammax.at(i_peak), gammay.at(i_peak), &(analytical_spectra[i_peak]),2.0);

            #pragma omp critical
            {
                for(int ii=i0;ii<i1;ii++)
                {
                    for(int jj=j0;jj<j1;jj++)
                    {
                        peaks_total[ii*ydim+jj]+=analytical_spectra[i_peak][(ii-i0)*(j1-j0)+jj-j0]; 
                    }
                }
            }
        }
        
        std::vector<int> peak_remove_flag;
        peak_remove_flag.resize(x.size(),0);

        //save old values so that we can check for convergence!
        x_old.push_back(x);
        y_old.push_back(y);

        #pragma omp parallel for
        for (int i_peak = 0; i_peak < x.size(); i_peak++)
        {
            if(to_remove[i_peak]==1) continue;  //peak has been removed. 
           
            std::vector<double> zz;
            int current_xdim,current_ydim;
            double total_z = 0.0;
            double current_x,current_y;

            if (type == gaussian_type)
            {
                gaussain_convolution_within_region(i_peak,a[i_peak][0], x.at(i_peak), y.at(i_peak), sigmax.at(i_peak), sigmay.at(i_peak), &(analytical_spectra[i_peak]),0.5);
            }
            if (type == voigt_type)
            {
                //voigt_convolution_region(x[current_peak_index],y[current_peak_index],sigmax[current_peak_index],sigmay[current_peak_index],gammax[current_peak_index],gammay[current_peak_index]);

                voigt_convolution_within_region(i_peak,a[i_peak][0], x.at(i_peak), y.at(i_peak), sigmax.at(i_peak), sigmay.at(i_peak), gammax.at(i_peak), gammay.at(i_peak), &(analytical_spectra[i_peak]),0.5);
            }

            current_xdim=i1-i0;
            current_ydim=j1-j0;
            current_x=x.at(i_peak)-i0;
            current_y=y.at(i_peak)-j0;

            // std::cout<<"before fit, peak "<<original_ndx[i]<<" x="<<x[i]<<" y="<<y[i]<<" xdim="<<xdim<<" ydim="<<ydim<<" region x is "<<i0<<" "<<i1<<" y is "<<j0<<" "<<j1<<std::endl;
            // if(original_ndx[current_peak_index]==400)
            // {
            //     std::cout<<"x= "<<x.at(current_peak_index)<<" y="<<y.at(current_peak_index)<<" fitting regison is x "<<i0<<" "<<i1<<" y:"<<j0<<" "<<j1<<std::endl;
            // }

            for (int ii = i0; ii < i1; ii++)
            {
                for (int jj = j0; jj < j1; jj++)
                {
                    double inten1 = analytical_spectra[i_peak][(ii-i0) * (j1-j0) + jj-j0];
                    double inten2 = peaks_total[ii * ydim + jj];
                    double scale_factor;
                    if (fabs(inten2) > 1e-100)
                        scale_factor = inten1 / inten2;
                    else
                        scale_factor = 0.0;

                    scale_factor = std::min(scale_factor, 1.0);
                    scale_factor = std::max(scale_factor, -1.0);
                    double temp = scale_factor * surface[0][ii * ydim + jj];
                    zz.push_back(temp);

                    // std::cout<<temp<<" ";
                    total_z += temp;
                }
                // std::cout<<std::endl;
            }
            num_sum[i_peak][0] = total_z;
            // std::cout<<std::endl;
            

            //std::cout <<"Before " <<loop<<" "<< original_ndx[i] << " " << x.at(i) << " " << y.at(i) << " " << a[i][0] << " " << sigmax.at(i) << " " << sigmay.at(i)<< " " << gammax.at(i) << " " << gammay.at(i) << " " << total_z << std::endl;
            if (type == gaussian_type)
            {
                if(cannot_move[i_peak]==1)
                {
                    one_fit_gaussian_intensity_only(current_xdim, current_ydim, &zz, current_x, current_y, a[i_peak][0], sigmax.at(i_peak), sigmay.at(i_peak), &e);
                }
                else
                {
                    one_fit_gaussian(current_xdim, current_ydim, &zz, current_x, current_y, a[i_peak][0], sigmax.at(i_peak), sigmay.at(i_peak), &e);
                }
            }
            else if (type == voigt_type)
            {
                if(cannot_move[i_peak]==1)
                {
                    one_fit_voigt_intensity_only(current_xdim, current_ydim, &zz, a[i_peak][0], current_x, current_y, sigmax.at(i_peak), sigmay.at(i_peak), gammax.at(i_peak), gammay.at(i_peak), &e);       
                }
                else
                {
                    one_fit_voigt(current_xdim, current_ydim, &zz, current_x, current_y, a[i_peak][0], sigmax.at(i_peak), sigmay.at(i_peak), gammax.at(i_peak), gammay.at(i_peak), &e,loop);            
                }
            }

            err.at(i_peak) = e;
            // std::cout<<"after fit, peak "<<original_ndx[i]<<" x="<<x[i]<<" y="<<y[i]<<" xdim="<<xdim<<" ydim="<<ydim<<" region x is "<<i0<<" "<<i1<<" y is "<<j0<<" "<<j1<<std::endl;


            if (type == gaussian_type)
            {
                if (fabs(sigmax.at(i_peak)) < 0.2 || fabs(sigmay.at(i_peak)) < 0.2 || fabs(sigmax.at(i_peak)) > 60.0 || fabs(sigmay.at(i_peak)) >60.0 )
                {
                    std::cout << original_ndx[i_peak] << " will be removed because x=" << current_x +i0 << " y=" << current_y +j0<< " a=" << a[i_peak][0] << " simgax=" << sigmax.at(i_peak) << " sigmay=" << sigmay.at(i_peak) << " totalz=" << total_z << std::endl;
                    peak_remove_flag[i_peak]=1;
                }

            }
            else  if (type == voigt_type)
            {
                if (fabs(sigmax.at(i_peak)) + fabs(gammax.at(i_peak)) < 0.2 || fabs(sigmay.at(i_peak)) + fabs(gammay.at(i_peak)) < 0.2 || fabs(sigmax.at(i_peak)) + fabs(gammax.at(i_peak)) >100.0 || fabs(sigmay.at(i_peak)) + fabs(gammay.at(i_peak)) >100.0)
                {
                    std::cout << original_ndx[i_peak] << " will be removed because " << current_x + i0 << " " << current_y + j0 << " " << a[i_peak][0] << " " << sigmax.at(i_peak) << " " << gammax.at(i_peak) << " " << sigmay.at(i_peak) << " " << gammay.at(i_peak) << " " << total_z << std::endl;
                    peak_remove_flag[i_peak]=1;
                }
            }

            if (current_x < 0 || current_x > current_xdim || current_y < 0 || current_y > current_ydim)
            {
                peak_remove_flag[i_peak] = 2;
                std::cout << original_ndx[i_peak] << " will be removed because x=" << current_x + i0 << " y=" << current_y + j0 << " a=" <<a[i_peak][0]
                          << " , moved out of fitting area x from " << i0  << " to " << i0+ current_xdim << " and y from " << j0 << " to " << j0+ current_ydim << std::endl;
            }
            x[i_peak]=current_x+i0;
            y[i_peak]=current_y+j0;
        } //end of parallel for(int i = 0; i < x.size(); i++)

        //also remove peaks if two peaks become very close.
        //that is, the program fit one peak with two overallped peaks, which can happen occasionally
       
        if(loop>5)
        {            
            double cut_near = gaussian_fit_wx*gaussian_fit_wy*too_near_cutoff*too_near_cutoff;
            // std::cout<<"too_near_cutoff is "<<too_near_cutoff<<std::endl;
            // std::cout<<"cut_near is "<<cut_near<<std::endl;
            for (int k1 = 0; k1 < a.size(); k1++)
            {
                if(to_remove[k1]==1) continue;  //peak has already been removed. 
                if(peak_remove_flag[k1]==2) continue;  //moved out of region peaks, do not use it too near calculation
                for (int k2 = k1 + 1; k2 < a.size(); k2++)
                {
                    if(to_remove[k2]==1) continue;  //peak has already been removed. 
                    if(peak_remove_flag[k2]==2) continue; //moved out of region peaks, do not use it too near calculation
                    double dx = x[k1] - x[k2];
                    double dy = y[k1] - y[k2];
                    
                    if (dx * dx + dy * dy < cut_near) //too close peaks
                    {
                        
                        if (fabs(a[k1][0]) > fabs(a[k2][0]))
                        {
                            //a[k2] = 0.0;
                            peak_remove_flag[k2]=1;
                            std::cout << original_ndx[k2] << " will be removed because too near " << original_ndx[k1] <<", distance is "<<sqrt(dx * dx + dy * dy) <<"<"<<cut_near<< std::endl;
                            if(peak_remove_flag[k1]==1)
                            {
                                peak_remove_flag[k1]=0;  //restore k1, because of effect of k2 on it.
                            }
                        }
                        else
                        {
                            //a[k1] = 0.0;
                            peak_remove_flag[k1]=1;
                            std::cout << original_ndx[k1] << " will be removed because too near " << original_ndx[k2] <<", distance is "<<sqrt(dx * dx + dy * dy)<<"<"<<cut_near<< std::endl;
                            if(peak_remove_flag[k2]==1)
                            {
                                peak_remove_flag[k2]=0;  //restore k2, because of effect of k1 on it.
                            }
                        }
                        flag_break = false;
                    }
                }
            }
        }

        //lable peak to remove!!
        for (int i = peak_remove_flag.size() - 1; i >= 0; i--)
        {
            if (peak_remove_flag.at(i) !=0)
            {
                to_remove[i]=1;
            }
        }

        // find peaks that moved too far away from starting point and remove them
        // if (type == voigt_type)
        {
            for(int i = 0; i < x.size(); i++)
            {
                if(to_remove[i]==1) continue;
                // if(cannot_move[i]==1) continue;
                int nt=int(round(x[i]))*ydim+int(round(y[i]));
                double current_ratio=surface[0][nt];
                if(current_ratio/original_ratio[i]>3.0 || current_ratio/original_ratio[i]<1/3.0)
                {
                    to_remove[i]=1;
                }
            }
        }

        if (flag_break)
        {
            break;
        }

        //test convergence. If so, we can break out of loop early
        bool bcon = false;
        for (int i = x_old.size() - 1; i >= std::max(int(x_old.size()) - 2, 0); i--)
        {
            if (x.size() != x_old[i].size())
            {
                continue;
            }

            bool b = true;
            for (int j = 0; j < x.size(); j++)
            {
                if(to_remove[j]==1) continue;
                // if(cannot_move[j]==1) continue;
                if (fabs(x[j] - x_old[i][j]) > 0.01)
                {
                    b = false;
                    break;
                }
                if (fabs(y[j] - y_old[i][j]) > 0.01)
                {
                    b = false;
                    break;
                }
            }
            if (b == true)
            {
                bcon = true;
                break;
            }
        }

        if (bcon == true || a.size() == 0)
        {
            flag_break = true;
        }
        std::cout<<"\r"<<"Iteration "<<loop+1<<"  "<<std::flush;
    } //loop
    std::cout<<std::endl;

    nround = loop;

    for (int i = sigmax.size() - 1; i >= 0; i--)
    {
        sigmax[i] = fabs(sigmax[i]);
        sigmay[i] = fabs(sigmay[i]);
        gammax[i] = fabs(gammax[i]);
        gammay[i] = fabs(gammay[i]);
    }

    //actual remove peaks
    for (int i = to_remove.size() - 1; i >= 0; i--)
    {
        if (to_remove[i]==1)
        {
            a.erase(a.begin() + i);
            x.erase(x.begin() + i);
            y.erase(y.begin() + i);
            sigmax.erase(sigmax.begin() + i);
            sigmay.erase(sigmay.begin() + i);
            gammax.erase(gammax.begin() + i);
            gammay.erase(gammay.begin() + i);
            num_sum.erase(num_sum.begin() + i);
            err.erase(err.begin() + i);
            original_ndx.erase(original_ndx.begin() + i);
        }
    }

    return true;
};

bool gaussian_fit::multi_spectra_run_single_peak()
{
    bool b=false;
    std::vector<double> xx,yy;
    std::vector< std::vector<double> > zz;
    std::vector<double> total_z(surface.size(),0.0);
    double e;

    float wx=(1.0692*gammax[0]+sqrt(0.8664*gammax[0]*gammax[0]+5.5452*sigmax[0]*sigmax[0]))*1.0;
    float wy=(1.0692*gammay[0]+sqrt(0.8664*gammay[0]*gammay[0]+5.5452*sigmay[0]*sigmay[0]))*1.0;

    // float wx=gaussian_fit_wx;
    // float wy=gaussian_fit_wx;
    

    i0=std::max(0,int(x[0]-wx+0.5));
    i1=std::min(xdim,int(x[0]+wx+0.5));
    j0=std::max(0,int(y[0]-wy+0.5));
    j1=std::min(ydim,int(y[0]+wy+0.5));

    for(int k=0;k<surface.size();k++)
    {
        std::vector<double> tz;
        for (int ii = i0; ii < i1; ii++)
        {
            for (int jj = j0; jj < j1; jj++)
            {
                double temp = surface[k][ii * ydim + jj];
                tz.push_back(temp);
                total_z[k] += temp;
            }
        }
        zz.push_back(tz);
    }

    double current_x=x[0]-i0;
    double current_y=y[0]-j0;

    // std::cout<<"before fitting, X0="<<x[0]<<" y0="<<y[0]<<" i0="<<i0<<" j0="<<j0<<std::endl;

    if (type == gaussian_type) //gaussian
    {
        multiple_fit_gaussian(i1-i0,j1-j0,zz,current_x,current_y,a[0], sigmax.at(0), sigmay.at(0),&e);
        if (fabs(sigmax.at(0)) < 0.02 || fabs(sigmay.at(0)) < 0.02 || current_x < 0 || current_x >= i1-i0 || current_y < 0 || current_y >= j1-j0)
        {
            to_remove[0] = 1; //this is a flag only
            std::cout<<"remove failed peak because "<<fabs(sigmax.at(0))<<" "<<fabs(sigmay.at(0))<<" "<<current_x<<" "<<current_y<<std::endl;
        }

        bool b_tolow=true;
        for(int k=0;k<surface.size();k++)
        {
            if(a[0][k]>=minimal_height)
            {
                b_tolow=false;
                break;
            }
        }

        if(b_tolow)
        {
            to_remove[0] = 1; //this is a flag only
            std::cout<<"remove too lower peak "<<fabs(sigmax.at(0))<<" "<<fabs(sigmay.at(0))<<" "<<fabs(gammax.at(0))<<" "<<fabs(gammay.at(0))<<" "<<current_x<<" "<<current_y<<std::endl;
        }
    }
    else if(type == voigt_type)//voigt
    {
        multiple_fit_voigt(i1-i0,j1-j0, zz, current_x, current_y, a[0], sigmax.at(0), sigmay.at(0),gammax.at(0),gammay.at(0),&e,0);
        if (fabs(sigmax.at(0)) + fabs(gammax.at(0)) < 0.2 || fabs(sigmay.at(0)) + fabs(gammay.at(0)) < 0.2 || current_x < 0 || current_x >= i1-i0 || current_y < 0 || current_y >= j1-j0)
        {
            std::cout<<"remove failed peak because "<<fabs(sigmax.at(0))<<" "<<fabs(sigmay.at(0))<<" "<<fabs(gammax.at(0))<<" "<<fabs(gammay.at(0))<<" "<<current_x<<" "<<current_y<<std::endl;
            to_remove[0] = 1; //this is a flag only
        }
        
        bool b_tolow=true;
        for(int k=0;k<surface.size();k++)
        {
            if(a[0][k]*(voigt(0.0,sigmax[0],gammax[0])*voigt(0.0,sigmay[0],gammay[0]))>=minimal_height)
            {
                b_tolow=false;
                break;
            }
        }
       
        if(b_tolow)
        {
            std::cout<<"remove too lower peak "<<a[0][0]<<" "<<fabs(sigmax.at(0))<<" "<<fabs(sigmay.at(0))<<" "<<fabs(gammax.at(0))<<" "<<fabs(gammay.at(0))<<" "<<current_x<<" "<<current_y<<std::endl;
            to_remove[0] = 1; //this is a flag only
        }
    }
    
    nround=1;
    x[0]=current_x+i0;
    y[0]=current_y+j0;
    // std::cout<<"After fitting, X0="<<x[0]<<" y0="<<y[0]<<" i0="<<i0<<" j0="<<j0<<std::endl;
    
    if(to_remove[0]==0)
    {
        num_sum[0] = total_z;
        err.at(0)=e;
        sigmax[0]=fabs(sigmax[0]);
        sigmay[0]=fabs(sigmay[0]);
        gammax[0]=fabs(gammax[0]);
        gammay[0]=fabs(gammay[0]);
        b= true;
    }
    else
    {
        a.clear();
        sigmax.clear();
        sigmay.clear();
        gammax.clear();
        gammay.clear();
        num_sum.clear();
        err.clear();
        x.clear();
        y.clear();
        b= false;
    }
    
    return b;

};

bool gaussian_fit::multi_spectra_run_multi_peaks(int loop_max)
{
    int npeak = x.size();

    double e;

    x_old.clear();
    y_old.clear();


    bool flag_break = false;
    int loop;
    for (loop = 0; loop < loop_max; loop++)
    {

        //save old values so that we can check for convergence!
        x_old.push_back(x);
        y_old.push_back(y);

        std::vector<int> peak_remove_flag;
        peak_remove_flag.resize(x.size(),0);

        std::vector< std::vector< std::vector<double> > > zzs(npeak); //zzs[peak_ndx][spectrum_ndx][data_point]

        std::vector<int> i0s,i1s,j0s,j1s; //save fitting region of each peak

        //spectrum by spectrum peak decovolution
        for(int spectrum_index=0;spectrum_index<surface.size();spectrum_index++)
        {
            analytical_spectra.clear();
            analytical_spectra.resize(x.size());

            peaks_total.clear();
            peaks_total.resize(xdim*ydim,0.0);

            #pragma omp parallel for
            for (unsigned int i_peak = 0; i_peak < x.size(); i_peak++)
            {
                if(to_remove[i_peak]==1) //peak has been removed. 
                {
                    analytical_spectra[i_peak].clear();
                    continue;   
                }
                
                if (type == gaussian_type)
                    gaussain_convolution(a[i_peak][spectrum_index], x.at(i_peak), y.at(i_peak), sigmax.at(i_peak), sigmay.at(i_peak), &(analytical_spectra[i_peak]),2.0);
                else if(type == voigt_type)
                    voigt_convolution(a[i_peak][spectrum_index], x.at(i_peak), y.at(i_peak), sigmax.at(i_peak), sigmay.at(i_peak), gammax.at(i_peak), gammay.at(i_peak), &(analytical_spectra[i_peak]),2.0);
            
                #pragma omp critical
                {
                    for(int ii=i0;ii<i1;ii++)
                    {
                        for(int jj=j0;jj<j1;jj++)
                        {
                            peaks_total[ii*ydim+jj]+=analytical_spectra[i_peak][(ii-i0)*(j1-j0)+jj-j0]; 
                        }
                    }
                }
            }

            #pragma omp parallel for
            for (int i_peak = 0; i_peak < x.size(); i_peak++)
            {
                if(to_remove[i_peak]==1) continue;  //peak has been removed.
            
                double total_z = 0.0;
                if (type == gaussian_type) 
                {
                    gaussain_convolution_within_region(i_peak,a[i_peak][spectrum_index], x.at(i_peak), y.at(i_peak), sigmax.at(i_peak), sigmay.at(i_peak), &(analytical_spectra[i_peak]),2.0);
                }
                if (type == voigt_type)
                {
                    voigt_convolution_within_region(i_peak,a[i_peak][spectrum_index], x.at(i_peak), y.at(i_peak), sigmax.at(i_peak), sigmay.at(i_peak), gammax.at(i_peak), gammay.at(i_peak), &(analytical_spectra[i_peak]),2.0);
                }

                if(spectrum_index==0)
                {
                    i0s.push_back(i0);
                    i1s.push_back(i1);
                    j0s.push_back(j0);
                    j1s.push_back(j1);
                }

                 
                
                // std::cout<<"before fit, peak "<<original_ndx[i]<<" x="<<x[i]<<" y="<<y[i]<<" xdim="<<xdim<<" ydim="<<ydim<<" region x is "<<i0<<" "<<i1<<" y is "<<j0<<" "<<j1<<std::endl;
                std::vector<double> tz;
                for (int ii = i0; ii < i1; ii++)
                {
                    for (int jj = j0; jj < j1; jj++)
                    {
                        double inten1 = analytical_spectra[i_peak][(ii-i0) * (j1-j0) + jj-j0];
                        double inten2 = peaks_total[ii * ydim + jj];
                        double scale_factor;
                        if (fabs(inten2) > 1e-100)
                            scale_factor = inten1 / inten2;
                        else
                            scale_factor = 0.0;

                        scale_factor = std::min(scale_factor, 1.0);
                        scale_factor = std::max(scale_factor, -1.0);
                        double temp = scale_factor * surface[spectrum_index][ii * ydim + jj];
                        tz.push_back(temp);
                        total_z += temp;
                    }
                }
                num_sum[i_peak][spectrum_index] = total_z;
                zzs[i_peak].push_back(tz);//zzs[peak_index][spec_ndx][data_ndx]
            }
        }

        #pragma omp parallel for
        int i_peak2=0;
        for (int i_peak = 0; i_peak < x.size(); i_peak++)
        {
            if(to_remove[i_peak]==1) continue;  //peak has been removed. 

            i0=i0s[i_peak2];
            i1=i1s[i_peak2];
            j0=j0s[i_peak2];
            j1=j1s[i_peak2];

            ++i_peak2;
            
            int current_xdim=i1-i0;
            int current_ydim=j1-j0;
            double current_x=x.at(i_peak)-i0;
            double current_y=y.at(i_peak)-j0;

            //std::cout <<"Before " <<loop<<" "<< original_ndx[i] << " " << x.at(i) << " " << y.at(i) << " " << a[i][0] << " " << sigmax.at(i) << " " << sigmay.at(i)<< " " << gammax.at(i) << " " << gammay.at(i) << " " << total_z << std::endl;
            if (type == gaussian_type)
            {
                multiple_fit_gaussian(current_xdim, current_ydim, zzs[i_peak], current_x, current_y, a[i_peak], sigmax.at(i_peak), sigmay.at(i_peak), &e);
            }
            else if (type == voigt_type)
            {
                multiple_fit_voigt(current_xdim, current_ydim, zzs[i_peak], current_x, current_y, a[i_peak], sigmax.at(i_peak), sigmay.at(i_peak), gammax.at(i_peak), gammay.at(i_peak), &e,loop);            
            }

            err.at(i_peak) = e;
           

            if (type == gaussian_type)
            {
                if (fabs(sigmax.at(i_peak)) < 0.2 || fabs(sigmay.at(i_peak)) < 0.2)
                {
                    peak_remove_flag[i_peak]=1;
                }

            }
            else  if (type == voigt_type)
            {
                if (fabs(sigmax.at(i_peak)) + fabs(gammax.at(i_peak)) < 0.2 || fabs(sigmay.at(i_peak)) + fabs(gammay.at(i_peak)) < 0.2)
                {
                    peak_remove_flag[i_peak]=1;
                }
            }

           
            if (current_x < 0 || current_x > current_xdim || current_y < 0 || current_y > current_ydim)
            {
                peak_remove_flag[i_peak] = 2;
                std::cout << original_ndx[i_peak] << " will be removed because x=" << current_x + i0 << " y=" << current_y + j0 << " a=" <<a[i_peak][0]
                          << " , moved out of fitting area x from " << i0  << " to " << i0+ current_xdim << " and y from " << j0 << " to " << j0+ current_ydim << std::endl;
            }
            x[i_peak]=current_x+i0;
            y[i_peak]=current_y+j0;
        } //end of parallel for(int i = 0; i < x.size(); i++)

        //also remove peaks if two peaks become very close.
        //that is, the program fit one peak with two overallped peaks, which can happen occasionally
        
        if(loop>5)
        {
            double cut_near = gaussian_fit_wx*gaussian_fit_wy*too_near_cutoff*too_near_cutoff;
            for (int k1 = 0; k1 < a.size(); k1++)
            {
                if(to_remove[k1]==1) continue;  //peak has already been removed. 
                if(peak_remove_flag[k1]==2) continue;  //moved out of region peaks, do not use it too near calculation
                for (int k2 = k1 + 1; k2 < a.size(); k2++)
                {
                    if(to_remove[k2]==1) continue;  //peak has already been removed. 
                    if(peak_remove_flag[k2]==2) continue; //moved out of region peaks, do not use it too near calculation
                    double dx = x[k1] - x[k2];
                    double dy = y[k1] - y[k2];
                    
                    if (dx * dx + dy * dy < cut_near) //too close peaks
                    {
                        if (fabs(a[k1][0]) > fabs(a[k2][0]))
                        {
                            //a[k2] = 0.0;
                            peak_remove_flag[k2]=1;
                            std::cout << original_ndx[k2] << " will be removed because too near " << original_ndx[k1] <<", distance is "<<sqrt(dx * dx + dy * dy) <<"<"<<cut_near<< std::endl;
                            if(peak_remove_flag[k1]==1)
                            {
                                peak_remove_flag[k1]=0;  //restore k1, because of effect of k2 on it.
                            }
                        }
                        else
                        {
                            //a[k1] = 0.0;
                            peak_remove_flag[k1]=1;
                            std::cout << original_ndx[k1] << " will be removed because too near " << original_ndx[k2] <<", distance is "<<sqrt(dx * dx + dy * dy)<<"<"<<cut_near<< std::endl;
                            if(peak_remove_flag[k2]==1)
                            {
                                peak_remove_flag[k2]=0;  //restore k2, because of effect of k1 on it.
                            }
                        }
                        flag_break = false;
                    }
                }
            }
        }
        

        //lable peak to remove!!
        for (int i = peak_remove_flag.size() - 1; i >= 0; i--)
        {
            if (peak_remove_flag.at(i) !=0)
            {
                to_remove[i]=1;
            }
        }

        if (flag_break)
        {
            break;
        }

        //test convergence. If so, we can break out of loop early
        bool bcon = false;
        for (int i = x_old.size() - 1; i >= std::max(int(x_old.size()) - 2, 0); i--)
        {
            if (x.size() != x_old[i].size())
            {
                continue;
            }

            bool b = true;
            for (int j = 0; j < x.size(); j++)
            {
                if (fabs(x[j] - x_old[i][j]) > 0.01)
                {
                    b = false;
                    break;
                }
                if (fabs(y[j] - y_old[i][j]) > 0.01)
                {
                    b = false;
                    break;
                }
            }
            if (b == true)
            {
                bcon = true;
                break;
            }
        }

        if (bcon == true || a.size() == 0)
        {
            flag_break = true;
        }
        std::cout<<"\r"<<"Iteration "<<loop+1<<"   "<<std::flush;
    } //loop
    std::cout<<std::endl;

    nround = loop;

    for (int i = sigmax.size() - 1; i >= 0; i--)
    {
        sigmax[i] = fabs(sigmax[i]);
        sigmay[i] = fabs(sigmay[i]);
        gammax[i] = fabs(gammax[i]);
        gammay[i] = fabs(gammay[i]);
    }

    //actual remove peaks
    for (int i = to_remove.size() - 1; i >= 0; i--)
    {
        if (to_remove[i]==1)
        {
            a.erase(a.begin() + i);
            x.erase(x.begin() + i);
            y.erase(y.begin() + i);
            sigmax.erase(sigmax.begin() + i);
            sigmay.erase(sigmay.begin() + i);
            gammax.erase(gammax.begin() + i);
            gammay.erase(gammay.begin() + i);
            num_sum.erase(num_sum.begin() + i);
            err.erase(err.begin() + i);
            original_ndx.erase(original_ndx.begin() + i);
        }
    }

    return true;
};


//for debug only. make sure I didn't fotget to update some variables in peak removal steps.
bool gaussian_fit::assess_size()
{
    bool b=true;

    if(a.size()==0) return true; //empty fit or single peak fit, we don't care

    if(a.size()!=x.size() || a.size()!=y.size() || a.size()!=sigmax.size() || a.size()!=sigmay.size() || 
    a.size()!=gammax.size() || a.size()!=gammay.size() || a.size()!=num_sum.size() || a.size()!=err.size() || a.size()!=original_ndx.size() )
    {
        std::cout<<"ERROR:  vector size is not consistent in gaussian_fit, cluster"<<my_index<<std::endl;
        std::cout<<"size of a is "<<a.size()<<std::endl;
        std::cout<<"size of x is "<<x.size()<<std::endl;
        std::cout<<"size of y is "<<y.size()<<std::endl;
        std::cout<<"size of simgax is "<<sigmax.size()<<std::endl;
        std::cout<<"size of sigmay is "<<sigmay.size()<<std::endl;
        std::cout<<"size of gammax is "<<gammax.size()<<std::endl;
        std::cout<<"size of gammay is "<<gammay.size()<<std::endl;
        std::cout<<"size of num_sum is "<<num_sum.size()<<std::endl;
        std::cout<<"size of err is "<<err.size()<<std::endl;
        std::cout<<"size of original_ndx is "<<original_ndx.size()<<std::endl;
        b=false;
    }
    return b;
};

int gaussian_fit::get_my_index() { return my_index; };
int gaussian_fit::get_nround() { return nround; };

void gaussian_fit::set_everything(fit_type t, int r, int index)
{
    type = t;
    rmax = r;
    my_index = index;
}
void gaussian_fit::set_peak_paras(double x, double y, double noise, double height, double near)
{
    gaussian_fit_wx = x;
    gaussian_fit_wy = y;
    noise_level = noise;
    minimal_height = height;
    too_near_cutoff = near;
};



//class spectrum_fit
spectrum_fit::spectrum_fit()
{
    peak_shape=null_type;

    wx=0.0;
    wy=0.0; 

    nspect=0; //number of spectra
    spectrum_type=null_spectrum; //general type
    maxround=20;

    median_width_x=0.0; //we don't know median peak width
    median_width_y=0.0;
    too_near_cutoff=0.2e-10; 

};

spectrum_fit::~spectrum_fit()
{
};

//read in all spectra and save the pointer to each spectrum into vector spects
//at the same time, varible spect will point to first spectrum
//all partition will be done using the first spectrum only
bool spectrum_fit::init_all_spectra(std::vector<std::string> fnames_)
{
    fnames=fnames_;
    for (int i = fnames.size() - 1; i >= 0; i--)
    {
        spectrum_io::init(fnames[i]);
        spects.push_back(spect);
    }
    std::reverse(spects.begin(),spects.end());
    nspect=spects.size();
    return true;
};


bool spectrum_fit::clear_memory()
{
    nspect=0;

    //no need to free spect because spects[0] === spect
    for(int i=0;i<spects.size();i++)
    {
        if(spects[i]!=NULL)
        {
            delete [] spects[i];
        }
    }
    return true;
}

bool spectrum_fit::initflags_fit(int n,double c,int im, int zf_)
{
    if(im==1) peak_shape=gaussian_type;
    else if(im==2) peak_shape=voigt_type;
    else peak_shape=null_type;

    too_near_cutoff=c;
    maxround=n;
    zf=zf_;
    return true;
}


//this fucntion decide how peaks are fitted together (depends on whether they overlap or not)
//only first spectrum is used !!
bool spectrum_fit::peak_partition()
{
    std::vector<int> peak_map2;

    if(wx>0.0 )
    {
        wx/=fabs(step1);
    }
    else
    {
        wx=median_width_x*1.60;
    }

    if(wy>0.0 )
    {
        wy/=fabs(step2);
    }
    else
    {
        wy=median_width_y*1.60;
    }


    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<"**********************************************************************************************************************************"<<std::endl;
    std::cout<<"IMPORTANT, make sure these two values are reasonable. If not, set them using -wx and -wy command line arguments. Unit is ppm in commandline arguments!"<<std::endl;
    std::cout<<"wx="<<wx<<" and wy="<<wy<<" (points) in this fitting."<<std::endl;
    std::cout<<"Firstly, peaks that are within wx*2 (wy*2) along direct(indirect) dimension are fitted tagather, if they are also connected by data points above noise*user_scale2."<<std::endl;
    std::cout<<"Secondly, after peak deconvolution in mixed Gaussian algorithm, fitting of each peak are done on an area of wx*3 by wy*3."<<std::endl;
    std::cout<<"**********************************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;

    peak_map.resize(xdim*ydim,-1);
    peak_map2.resize(xdim*ydim,0);
    // pos_x_correction=new double[p1.size()];
    // pos_y_correction=new double[p1.size()];
    
    for(int i=0;i<p1.size();i++)
    {
        int xx=(int)(p1.at(i)+0.5);
        int yy=(int)(p2.at(i)+0.5);
        if(xx>=xdim || xx<0 || yy<0 || yy>=ydim)
        {
            std::cout<<"Sth is wrong with the coordinates. in peak_parttition."<<std::endl;
        }
        peak_map[xx*ydim+yy]=i;
        // pos_x_correction[i]=p1.at(i)-xx;
        // pos_y_correction[i]=p2.at(i)-yy;
    }
    
    for(unsigned int i=0;i<p1.size();i++)
    {
        double d_range=1.5;
        int xfrom=int(p1.at(i)-wx*d_range+0.5);
        int xto=int(p1.at(i)+wx*d_range+0.5);
        int yfrom=int(p2.at(i)-wy*d_range+0.5)+1;
        int yto=int(p2.at(i)+wy*d_range+0.5)+1;

        if(xfrom<0) xfrom=0;
        if(xto>xdim) xto=xdim;
        if(yfrom<0) yfrom=0;
        if(yto>ydim) yto=ydim;
        for(int m=xfrom;m<xto;m++)
        {
            for(int n=yfrom;n<yto;n++)
            {
                if(m>=xdim || m<0 || n<0 || n>=ydim)
                {
                    std::cout<<"Sth is wrong with the coordinates. in peak_parttition."<<std::endl;
                }
                peak_map2[m*ydim+n]=1;
            }
        }
    }

    double lowest_level=noise_level*user_scale2;
    std::vector< std::vector<int> > used;
    b.resize(ydim);
    s.resize(ydim);
    used.resize(ydim);

    for(int j=0;j<ydim;j++)
    {
        if(spect[j*xdim+0]>=lowest_level && peak_map2[j]==1) b[j].push_back(0);
        for(int i=1;i<xdim;i++)
        {
            if((spect[j*xdim+i-1]<lowest_level || peak_map2[j+(i-1)*ydim]==0) && (spect[j*xdim+i]>=lowest_level && peak_map2[j+i*ydim]==1)) b[j].push_back(i);
            if((spect[j*xdim+i-1]>=lowest_level && peak_map2[j+(i-1)*ydim]==1) && (spect[j*xdim+i]<lowest_level || peak_map2[j+i*ydim]==0)) s[j].push_back(i);
            // if(spect[j*xdim+i-1]<lowest_level  && spect[j*xdim+i]>=lowest_level ) b[j].push_back(i);
            // if(spect[j*xdim+i-1]>=lowest_level && spect[j*xdim+i]<lowest_level ) s[j].push_back(i);
        }
        if(s[j].size()<b[j].size()) s[j].push_back(xdim);
        for(int i=0;i<s[j].size();i++) used[j].push_back(0);
    }

    if(b.size()!=s.size() || b.size()!=used.size())
    {
        std::cout<<"ERROR!!! Size of b is "<<b.size()<<" s is "<<s.size()<<" used is "<<used.size()<<std::endl;
    }
    for(int j=0;j<b.size();j++)
    {
        if(b[j].size()!=s[j].size() || b[j].size()!=used[j].size())
        {
            std::cout<<"ERROR, size of b,s,used inconsistent!!!"<<b[j].size()<<" "<<s[j].size()<<" "<<used[j].size()<<std::endl;
        }
    }
    std::cout<<"check size done."<<std::endl;

    std::deque< std::pair<int,int> > work;
    int position;

    for(int j=0;j<ydim;j++)
    {
        for(int i=0;i<used[j].size();i++)
        {
            if(used[j][i]==0)
            {
                used[j][i]=1;
                work.clear();
                work.push_back(std::pair<int,int>(j,i));
                position=0;
                while(position<work.size())
                {
                    std::pair<int,int> c=work[position];
                    position++;
                    for(int jj=std::max(0,c.first-1);jj<std::min(ydim,c.first+2);jj++)
                    {
                        if(jj==c.first) continue;
                        for(int ii=0;ii<used[jj].size();ii++)
                        {
                            if( (jj==c.first && ii==c.second) || used[jj][ii]==1)
                            {
                                continue;
                            }

                            if (s[jj][ii]>=b[c.first][c.second] && b[jj][ii]<=s[c.first][c.second])
                            {
                                work.push_back(std::pair<int,int>(jj,ii));
                                used[jj][ii]=1;
                            }
                        }
                    }
                }
                clusters2.push_back(work);
            }
        }
    }
    std::cout<<"Total "<<clusters2.size()<<" peak clusters."<<std::endl;
    return true;
}

bool spectrum_fit::prepare_fit()
{
    int min1,min2,max1,max2;
    int counter=0;
    for(unsigned int i0=0;i0<clusters2.size();i0++)
    {
        // std::cout<<"parepare, "<<i0<<" out of "<<clusters2.size()<<std::endl;
        min1=min2=1000000;
        max1=max2=-1000000;

        for (unsigned int i1 = 0; i1 < clusters2[i0].size(); i1++)
        {
            std::pair<int,int> k = clusters2[i0][i1];
            int j=k.first;
            int begin=b[j][k.second];
            int stop=s[j][k.second];

            if (begin <= min1)
                min1 = begin;
            if (stop >= max1)
                max1 = stop;
            if (j <= min2)
                min2 = j;
            if (j >= max2)
                max2 = j;
        }
        max1++;
        max2++;

        // std::cout<<"cluster "<<i0<< " x = "<<min1<<" "<<max1<<", y= "<<min2<<" "<<max2<<std::endl;


        if( max1-min1<3 || max2-min2<3 )
        {
            // std::cout<<"cluster "<<i0<< " is too small, remove it."<<std::endl;
            continue;
        }


        std::vector< std::vector<double> > spect_parts(nspect,std::vector<double>((max1-min1)*(max2-min2),0.0));
        std::vector< std::vector<double> > aas;
        aas.clear();
        
        std::vector<double> xx,yy,sx,sy;
        std::vector<int> ori_index;
        std::vector<int> region_peak_cannot_move_flag;
        xx.clear();yy.clear();sx.clear();sy.clear();ori_index.clear(); //do not need


        for(unsigned int i1=0;i1<clusters2[i0].size();i1++)
        {
            std::pair<int,int> k = clusters2[i0][i1];
            int j=k.first;
            int begin=b[j][k.second];
            int stop=s[j][k.second];
            for(int kk=begin;kk<stop;kk++)
            {
                if((kk-min1)*(max2-min2)+j-min2<0 || (kk-min1)*(max2-min2)+j-min2>=spect_parts[0].size()) 
                {
                    std::cout<<"Spect_part MEMOR ERROR !!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
                }
                if(kk+j*xdim<0 || kk+j*xdim>=xdim*ydim) std::cout<<"Spect MEMOR ERROR !!!!!!!!!!!!!!!!!!!!!!!!"<<std::endl;
              
                for(int k=0;k<nspect;k++)
                {
                    spect_parts[k][(kk-min1)*(max2-min2)+j-min2]=spects[k][kk+j*xdim];
                }
                int peak_ndx=peak_map[kk*ydim+j];
                //std::cout<<"kk="<<kk<<" j="<<j<<" peak_ndx="<<peak_ndx<<std::endl;
                if(peak_ndx>=0)
                {
                    xx.push_back(kk-min1);
                    yy.push_back(j-min2);

                    // std::cout<<"kk="<<kk<<" min1="<<min1<<" j="<<j<<" min2="<<min2<<std::endl;

                    double fwhhx=1.0692*gammax[peak_ndx]+sqrt(0.8664*gammax[peak_ndx]*gammax[peak_ndx]+5.5452*sigmax[peak_ndx]*sigmax[peak_ndx]);
                    double fwhhy=1.0692*gammay[peak_ndx]+sqrt(0.8664*gammay[peak_ndx]*gammay[peak_ndx]+5.5452*sigmay[peak_ndx]*sigmay[peak_ndx]);

                    sx.push_back(fwhhx); 
                    sy.push_back(fwhhy); 
                    ori_index.push_back(peak_index[peak_ndx]);  
                    region_peak_cannot_move_flag.push_back(peak_cannot_move_flag[peak_ndx]); 
                    aas.push_back(p_intensity_all_spectra[peak_ndx]);          
                }
            }
        }


        if(xx.size()>0)
        {
            gaussian_fit myfit;

            myfit.set_everything(peak_shape,maxround,counter);

            if(zf==0)
            {
                myfit.init(min1, min2, max1 - min1 , max2 - min2 , spect_parts, xx, yy, aas, sx, sy,ori_index,region_peak_cannot_move_flag);
                myfit.set_peak_paras(wx*1.5,wy*1.5,noise_level,noise_level*user_scale2,too_near_cutoff); 
                // if(counter==177 || counter==572)
                // {
                //     std::cout<<counter<<std::endl;
                //     std::cout<<"Size of xx is "<<xx.size()<<std::endl;
                //     std::cout<<"Size of sx is "<<sx.size()<<std::endl;
                //     std::cout<<"Size of ori_index is "<<ori_index.size()<<std::endl;    

                //     myfit.assess_size();                
                // }
            }
            else if(zf==1)
            {
                int d1=max1-min1;
                int d2=max2-min2;
                for(int m=0;m<xx.size();m++)
                {
                    xx[m]=xx[m]*2.0+1.0;
                    yy[m]=yy[m]*2.0+1.0;
                    sx[m]*=4.0; //becasue sx is actually 2*sigma*sigma
                    sy[m]*=4.0;
                }
                //interp2 of spect_parts[m]
                for(int m=0;m<spect_parts.size();m++)
                {
                    std::vector<double> final_data;
                    std::vector<float> input_data(d1*d2); 
                    //spect_parts[m] is column by column
                    //input_data need to be row by row
                    for(int n1=0;n1<d1;n1++)
                    {
                        for (int n2 = 0; n2 < d2; n2++)
                        {
                            input_data[n2*d1+n1] = spect_parts[m][n1*d2+n2];
                        }
                    }
                    
                    ldw_math_spectrum_2d::spline_expand(d1,d2,input_data.data(),final_data);
                    spect_parts[m]=final_data;
                }

                myfit.init(min1, min2, d1*2-1 , d2*2-1 , spect_parts, xx, yy, aas, sx, sy,ori_index,region_peak_cannot_move_flag);
                myfit.set_peak_paras(wx*3,wy*3,noise_level,noise_level*user_scale2,too_near_cutoff*2.0); 
            }
            fits.emplace_back(myfit);
            counter++;
        }
        // std::cout<<"Initialization of cluster "<<i0<<" is done."<<std::endl;
    }
    return true;
};


bool spectrum_fit::peak_fitting()
{
    peak_partition();
    prepare_fit();
    real_peak_fitting();
    fit_gather();
    return true;
}

bool spectrum_fit::real_peak_fitting()
{
    std::cout<<"Total "<<fits.size()<<" regions to be fitted individually."<<std::endl;

    for(int i=0;i<fits.size();i++)
    {
        if(fits[i].assess_size()==false)
        {
            std::cout<<"SOMETHING IS WRONG!!!! for Cluster "<<i<<std::endl;
        }
    }

    for(int i=0;i<fits.size();i++)
    // for(int i=85;i<88;i++)
    {
        if (fits[i].x.size() == 1)
        {
            std::cout << "Cluster " << fits[i].get_my_index() << " has " << fits[i].x.size() << " peaks before fitting. Region is " << fits[i].xstart << " " << fits[i].xstart + fits[i].xdim << " " << fits[i].ystart << " " << fits[i].ystart + fits[i].ydim<< std::endl;
            if (fits[i].run() == false || fits[i].a.size()==0)
            {
                std::cout << "Cluster " << fits[i].get_my_index() << ", region is " << fits[i].xstart << " " << fits[i].xstart + fits[i].xdim << " " << fits[i].ystart << " " << fits[i].ystart + fits[i].ydim << " ";
                std::cout << "To be removed" << std::endl;
            }

            std::cout << "Cluster " << fits[i].get_my_index() << ", region is " << fits[i].xstart << " " << fits[i].xstart + fits[i].xdim << " " << fits[i].ystart << " " << fits[i].ystart + fits[i].ydim << " ";
            std::cout << "fitted " << fits[i].x.size() << " peaks." << std::endl
                      << std::endl;
        }
   
        if (fits[i].x.size() > 1)
        {
            std::cout << "Cluster " << fits[i].get_my_index() << " has " << fits[i].x.size() << " peaks before fitting." << " Region is " << fits[i].xstart << " " << fits[i].xstart + fits[i].xdim << " " << fits[i].ystart << " " << fits[i].ystart + fits[i].ydim << " "<< std::endl;
            if (fits[i].run() == false || fits[i].a.size()==0 )
            {
                std::cout << "Cluster " << fits[i].get_my_index();
                std::cout << "To be removed" << std::endl;
            }

            std::cout << "Cluster " << fits[i].get_my_index() << ", region is " << fits[i].xstart << " " << fits[i].xstart + fits[i].xdim << " " << fits[i].ystart << " " << fits[i].ystart + fits[i].ydim << " ";
            std::cout << "fitted " << fits[i].x.size() << " peaks." << std::endl
                      << std::endl;
        }
    }

   return true;
};


bool spectrum_fit::fit_gather()
{
    p1.clear();
    p2.clear();
    sigmax.clear();
    sigmay.clear();
    gammax.clear();
    gammay.clear();
    nround.clear();
    group.clear();
    err.clear();
    peak_index.clear();
    
    p_intensity.clear();
    p_intensity_all_spectra.clear();
    num_sums.clear();



    double zf_step=1.0;
    if(zf==0) zf_step=1.0;
    else if(zf==1) zf_step=0.5;

    for(int i=0;i<fits.size();i++)
    {
        if(fits[i].a.size()==0) continue; 

        if(fits[i].assess_size()==false)
        {
            std::cout<<"SOMETHING IS WRONG!!!!"<<std::endl;
        }

        for (unsigned int ii = 0; ii < fits[i].x.size(); ii++)
        {
            p1.push_back(fits[i].x.at(ii)*zf_step + fits[i].xstart);
            p2.push_back(fits[i].y.at(ii)*zf_step + fits[i].ystart);
            group.push_back(i);
            nround.push_back(fits[i].get_nround());
            p_intensity.push_back(fits[i].a[ii][0]); //intensity of first spectrum only.
        }

        //std::cout<<"In fits"<<i<<" first part is done."<<std::endl;

        sigmax.insert(sigmax.end(), fits[i].sigmax.begin(), fits[i].sigmax.end());
        sigmay.insert(sigmay.end(), fits[i].sigmay.begin(), fits[i].sigmay.end());
        peak_index.insert(peak_index.end(),fits[i].original_ndx.begin(),fits[i].original_ndx.end());
        err.insert(err.end(), fits[i].err.begin(), fits[i].err.end());

        num_sums.insert(num_sums.end(), fits[i].num_sum.begin(), fits[i].num_sum.end());
        p_intensity_all_spectra.insert(p_intensity_all_spectra.end(), fits[i].a.begin(), fits[i].a.end()); 
    
        if (peak_shape==voigt_type)
        {
            gammax.insert(gammax.end(), fits[i].gammax.begin(), fits[i].gammax.end());
            gammay.insert(gammay.end(), fits[i].gammay.begin(), fits[i].gammay.end());
        }
        // std::cout<<"In fits"<<i<<" gathering part is done out of total"<<fits.size()<<std::endl;
        // std::cout<<"Current size of p_intensitu is "<<p_intensity.size()<<std::endl;
        assess_size();
    }

    //if it is a load, fits[i].comments will be empty. we set all names to "peak".
    if(user_comments.size()==0)
    {
        user_comments.resize(p1.size(),"peak");    
        std::cout<<"set up user comments section."<<std::endl;  
    }
    //if not, it is a picking or reading, we already have all user_comments, but some of them have been removed in the fitting process
    else
    {
        std::vector< std::string> temp_comments;
        // std::cout<<"size of user_comment is "<<user_comments.size()<<" size of peak_index is "<<peak_index.size()<<std::endl;
        for(int i=0;i<peak_index.size();i++)
        {
            temp_comments.push_back(user_comments[peak_index[i]]);   
        }
        user_comments=temp_comments;
    }

    for(int i=0;i<sigmax.size();i++)
    {
        sigmax[i]*=zf_step;
        sigmay[i]*=zf_step;
        gammax[i]*=zf_step;
        gammay[i]*=zf_step;
        for(int j=0;j<num_sums[i].size();j++)
        {
            num_sums[i][j]*=(zf_step*zf_step);
        }
    }

    std::cout<<"Totally fitted "<<p1.size()<<" peaks from the raw spectrum"<<std::endl;


    p1_ppm.clear();
    p2_ppm.clear();

    for (unsigned int i = 0; i < p1.size(); i++)
    {
        double f1 = begin1 + step1 * (p1[i]);  //direct dimension
        double f2 = begin2 + step2 * (p2[i]);  //indirect dimension
        p1_ppm.push_back(f1);
        p2_ppm.push_back(f2);

        sigmax[i]=fabs(sigmax[i]);
        sigmay[i]=fabs(sigmay[i]);
    }
    //std::cout<<"Done fitting"<<std::endl;

    std::cout<<"Finish gathering fitting result."<<std::endl;
    return true;
}


bool spectrum_fit::generate_recon_and_diff_spectrum()
{
    std::vector< std::vector<double> > spe;

    for(int file_ndx=0;file_ndx<fnames.size();file_ndx++)
    {

        std::string path_name;
        std::string file_name;
        ldw_math_spectrum_2d::SplitFilename (fnames[file_ndx],path_name,file_name);

        spe.clear();

        std::vector<double> intens;
        for(int i=0;i<p_intensity_all_spectra.size();i++)
        {
            intens.push_back(p_intensity_all_spectra[i][file_ndx]);
        }

        if (peak_shape==voigt_type)
        {
            generate_spectrum_voigt(intens, sigmax, sigmay, gammax, gammay, p1, p2, spe);
        }
        else if (peak_shape==gaussian_type)
        {
            generate_spectrum_gaussian(intens, sigmax, sigmay, p1, p2, spe);
        }

        std::vector< std::vector<float> > spe_float;
        spe_float.clear();
        for (unsigned int i = 0; i < spe.size(); i++)
        {
            std::vector<float> t;
            t.clear();
            for(int j=0;j<spe[i].size();j++)
            {
                t.push_back(spe[i][j]);
            }
            spe_float.push_back(t);
        }
        std::cout<<"Write recon to "<<path_name+"/recon_"+file_name<<std::endl;
        write_pipe(spe_float,path_name+"/recon_"+file_name);
    
        spe_float.clear();
        for (unsigned int i = 0; i < spe.size(); i++)
        {
            std::vector<float> t;
            t.clear();
            for(int j=0;j<spe[i].size();j++)
            {
                t.push_back(spects[file_ndx][i*xdim+j]-spe[i][j]);
            }
            spe_float.push_back(t);
        }
        write_pipe(spe_float,path_name+"/diff_"+file_name);
    }
    return true;
};




bool spectrum_fit::generate_spectrum_gaussian(std::vector<double> inten, std::vector<double> sigmax, std::vector<double> sigmay, std::vector<double> centerx, std::vector<double> centery, std::vector< std::vector<double> > &spectrum)
{
    int min_w=15;
    int max_w=150;

    spectrum.resize(ydim);
    for (unsigned int i = 0; i < ydim; i++)
        spectrum[i].resize(xdim);

    double *kernel;

   
    for (unsigned int m = 0; m < centerx.size(); m++)
    {

        float s1=sigmax[m]*2.355f;
        float s2=sigmay[m]*2.355f;
            
        int height=std::max(int(fabs(s2*3)),min_w);
        int width =std::max(int(fabs(s1*3)),min_w);

        height=std::min(height,max_w);
        width=std::min(width,max_w);
        
        sigmax[m]=2*sigmax[m]*sigmax[m];
        sigmay[m]=2*sigmay[m]*sigmay[m];
        

        int p1 = int(centerx[m]+0.5);
        int p2 = int(centery[m]+0.5);

        double r1=centerx[m]-p1;
        double r2=centery[m]-p2;

        kernel=new double[(width*2+1)*(height*2+1)];

        
        for (int i = -height; i <= height; i++)
        {
            for (int j = -width; j <= width; j++)
            {
                kernel[(i + height)*(2*width+1)+j + width] = exp(- (i-r2)*(i-r2)/fabs(sigmay[m]) -(j-r1)*(j-r1)/fabs(sigmax[m]));
            }
        }
    

        for (int i = std::max(-height, -p2); i <= std::min(height, ydim - p2 -1); i++)
        {
            for (int j = std::max(-width, -p1); j <= std::min(width, xdim - p1 -1 ); j++)
            {
                spectrum[p2 + i][p1 + j] += kernel[(i + height)*(2*width+1)+j + width] * inten[m];
            }
        }

        delete [] kernel;
    }
    return 1;
};


bool spectrum_fit::generate_spectrum_voigt(std::vector<double> inten, std::vector<double> sigmax, std::vector<double> sigmay, std::vector<double> gammax, std::vector<double> gammay, 
std::vector<double> centerx, std::vector<double> centery, std::vector< std::vector<double> > &spectrum)
{

    int min_w=15;
    int max_w=150;

    spectrum.resize(ydim);
    for (unsigned int i = 0; i < ydim; i++)
        spectrum[i].resize(xdim);

    for (unsigned int m = 0; m < centerx.size(); m++)
    {

        float s1=0.5346*gammax[m]*2+std::sqrt(0.2166*4*gammax[m]*gammax[m]+sigmax[m]*sigmax[m]*8*0.6931);
        float s2=0.5346*gammay[m]*2+std::sqrt(0.2166*4*gammay[m]*gammay[m]+sigmay[m]*sigmay[m]*8*0.6931);


        int height=std::max(int(fabs(s2*5)),min_w);
        int width =std::max(int(fabs(s1*5)),min_w);



        height=std::min(height,max_w);
        width=std::min(width,max_w);

        int p2 = int(centerx[m]+0.5);
        int p1 = int(centery[m]+0.5);

        double r2=centerx[m]-p2;
        double r1=centery[m]-p1;

        //std::cout<<p2<<" "<<p1<<" "<<centerx[m]<<" "<<centery[m]<<" "<<r1<<" "<<r2<<std::endl;

        std::vector< std::vector<double> > kernel(height * 2 + 1, std::vector<double>(width * 2 + 1));
        for (int i = -height; i <= height; i++)
        {
            for (int j = -width; j <= width; j++)
            {
                double z1=voigt ( i-r1, sigmay[m], gammay[m]);
                double z2=voigt ( j-r2, sigmax[m], gammax[m] );
                kernel[i + height][j + width] = z1*z2;
            }
        }

        for (int i = std::max(-height, -p1); i <= std::min(height, ydim - p1 -1); i++)
        {
            for (int j = std::max(-width, -p2); j <= std::min(width, xdim - p2 -1 ); j++)
            {
                spectrum[p1 + i][p2 + j] += kernel[i + height][j + width] * inten[m];
                //std::cout<<p1+i<<" "<<kernel[i + height][j + width]<<std::endl;
            }
        }
    }
    return 1;
};

bool spectrum_fit::print_peaks(std::string outfname)
{
    assess_size();

    std::vector<int> ndx;
    std::vector<double> amplitudes,fitted_volume;
    std::vector< std::vector<double> > amplitudes_all_spectra;

    if(peak_shape==gaussian_type)
    {
        amplitudes=p_intensity;
        for(int i=0;i<p_intensity.size();i++)
        {
            fitted_volume.push_back(p_intensity[i]*sqrt(fabs(sigmax[i]*sigmay[i]))*3.14159265358979);
        }
    }
    else if(peak_shape==voigt_type)
    {
        fitted_volume=p_intensity;
        for(int i=0;i<p_intensity.size();i++)
        {
            amplitudes.push_back(p_intensity[i]*(voigt(0.0,sigmax[i],gammax[i])*voigt(0.0,sigmay[i],gammay[i])));   
        }
    }
    ldw_math_spectrum_2d::sortArr(amplitudes,ndx);


    //if multiple spectral file. 
    if(spects.size()>1)
    {
        if(peak_shape==gaussian_type)
        {
            amplitudes_all_spectra=p_intensity_all_spectra;       
        }
        else if(peak_shape==voigt_type)
        {
            amplitudes_all_spectra.resize(p_intensity_all_spectra.size());
            for(int i=0;i<p_intensity_all_spectra.size();i++)
            {
                for(int k=0;k<p_intensity_all_spectra[i].size();k++)
                {
                    float temp=p_intensity_all_spectra[i][k]*(voigt(0.0,sigmax[i],gammax[i])*voigt(0.0,sigmay[i],gammay[i]));
                    amplitudes_all_spectra[i].push_back(temp);
                }
            }
        }
    

        //rescale amplitudes_all_spectra using first spectrum
        for (int i = 0; i < p_intensity_all_spectra.size(); i++)
        {
            for (int k = 1; k < p_intensity_all_spectra[i].size(); k++)
            {

                amplitudes_all_spectra[i][k] /= amplitudes_all_spectra[i][0];
            }
            amplitudes_all_spectra[i][0]=1.0;
        }
    }


    std::istringstream iss;
    iss.str(outfname);
    std::string p;
    std::vector<std::string> file_names;

    while (iss >> p)
    {
        file_names.push_back(p);
    }

    std::string stab(".tab");
    std::string slist(".list");
    std::string sjson(".json");

    for(int m=0;m<file_names.size();m++)
    {
        if(std::equal(stab.rbegin(), stab.rend(), file_names[m].rbegin()))
        {
            FILE *fp = fopen(file_names[m].c_str(), "w");
            if(peak_shape==voigt_type)
            {
                //fprintf(fp,"#x y ppm_x ppm_y intensity sigmax sigmay (gammx gammay) fitted_volume num_volume type group\n");
                fprintf(fp,"VARS   INDEX X_AXIS Y_AXIS X_PPM Y_PPM XW YW  X1 X3 Y1 Y3 HEIGHT DHEIGHT ASS CLUSTID INTEGRAL VOL SIMGAX SIGMAY GAMMAX GAMMAY NROUND");
                if(spects.size()>1)
                {
                    for(int i=0;i<spects.size();i++)
                    {
                        fprintf(fp," Z_A%d",i);
                    }
                }
                fprintf(fp,"\n");

                fprintf(fp,"FORMAT %%5d %%9.3f %%9.3f %%8.3f %%8.3f %%7.3f %%7.3f %%4d %%4d %%4d %%4d %%+e %%+e %%s %%4d %%+e %%+e %%f %%f %%f %%f %%4d");
                if(spects.size()>1)
                {
                    for(int i=0;i<spects.size();i++)
                    {
                        fprintf(fp," %%7.4f");
                    }
                }
                fprintf(fp,"\n");
                for (unsigned int i0 = 0; i0 < p1.size(); i0++)
                {
                    //fitted p_tensity is actually area because external function voigt is correctly rescaled to make sure integral is 1. 
                    //Here temp is the real intensity. 
                    int i=ndx[i0];
                    float s1=0.5346*gammax[i]*2+std::sqrt(0.2166*4*gammax[i]*gammax[i]+sigmax[i]*sigmax[i]*8*0.6931);
                    float s2=0.5346*gammay[i]*2+std::sqrt(0.2166*4*gammay[i]*gammay[i]+sigmay[i]*sigmay[i]*8*0.6931);

                    fprintf(fp,"%5d %9.3f %9.3f %8.3f %8.3f %7.3f %7.3f %4d %4d %4d %4d %+e %+e %s %4d %+e %+e %f %f %f %f %4d",
                                peak_index[i],p1[i]+1,p2[i]+1,p1_ppm[i], p2_ppm[i],s1,s2,
                                int(p1[i]-3),int(p1[i]+3),int(p2[i]-3),int(p2[i]+3),amplitudes[i],err[i],user_comments[i].c_str(),group[i]+1,num_sums[i][0],p_intensity[i],sigmax[i],sigmay[i],gammax[i],gammay[i],nround[i]);
                    if(spects.size()>1)
                    {
                        for(int j=0;j<spects.size();j++)
                        {
                            fprintf(fp," %7.4f",amplitudes_all_spectra[i][j]);
                        }
                    }
                    fprintf(fp,"\n");
                }
                fclose(fp);
            }
            
            if(peak_shape==gaussian_type)
            {
                fprintf(fp,"VARS   INDEX X_AXIS Y_AXIS X_PPM Y_PPM XW YW  X1 X3 Y1 Y3 HEIGHT DHEIGHT ASS CLUSTID INTEGRAL VOL NROUND");
                for(int i=0;i<spects.size();i++)
                {
                    fprintf(fp," Z_A%d",i);
                }
                fprintf(fp,"\n");

                fprintf(fp,"FORMAT %%5d %%9.3f %%9.3f %%8.3f %%8.3f %%7.3f %%7.3f %%4d %%4d %%4d %%4d %%+e %%+e %%s %%4d %%+e %%+e %%4d");
                for(int i=0;i<spects.size();i++)
                {
                    fprintf(fp," %%7.4f");
                }
                fprintf(fp,"\n");

                for (unsigned int i0 = 0; i0 < p1.size(); i0++)
                {
                    int i=ndx[i0];
                    float s1=std::sqrt(sigmax[i]/2.0)*2.355f;
                    float s2=std::sqrt(sigmay[i]/2.0)*2.355f;
                    fprintf(fp,"%5d %9.3f %9.3f %8.3f %8.3f %7.3f %7.3f %4d %4d %4d %4d %+e %+e %s %4d %+e %+e %4d",peak_index[i],p1[i]+1,p2[i]+1,p1_ppm[i], p2_ppm[i],s1,s2,
                                int(p1[i]-3),int(p1[i]+3),int(p2[i]-3),int(p2[i]+3),amplitudes[i],err[i],user_comments[i].c_str(),group[i]+1,num_sums[i][0],fitted_volume[i],nround[i]);
                    if(spects.size()>1)
                    {
                        for(int j=0;j<spects.size();j++)
                        {
                            fprintf(fp," %7.4f",amplitudes_all_spectra[i][j]);
                        }
                    }
                    fprintf(fp,"\n");
                }
                fclose(fp);
            }
        }

        else if(std::equal(slist.rbegin(), slist.rend(), file_names[m].rbegin()))
        {
            //for Sparky format
            FILE *fp = fopen(file_names[m].c_str(), "w");
            fprintf(fp, "Assignment         w1         w2   Height\n");
            for (unsigned int i = 0; i < p1.size(); i++)
            {
                fprintf(fp, "?-? %9.3f %9.3f %+e\n", p2_ppm[i], p1_ppm[i], amplitudes[i]);
            }
            fclose(fp);
        }
    

        else if(std::equal(sjson.rbegin(), sjson.rend(), file_names[m].rbegin()))
        {
            // json format output, Gaussian only at this time.
            FILE *fp3 = fopen(file_names[m].c_str(), "w");
            if(peak_shape==gaussian_type)
            {
                fprintf(fp3, "\"fitted_peaks\":[");

                int npeak = p_intensity.size();
                for (unsigned int i = 0; i < npeak - 1; i++)
                {
                    float temp = p_intensity[i] * sqrt(fabs(sigmax[i] * sigmay[i])) * 3.14159265358979f;
                    fprintf(fp3, "{\"cs_x\": %f,\"cs_y\": %f, \"intergral\": %f, \"intergral2\": %f, \"sigmax\": %f, \"sigmay\": %f, \"type\": 1, \"index\": %f},", p1_ppm[i], p2_ppm[i], fitted_volume[i], num_sums[i][0], sigmax[i], sigmay[i], amplitudes[i]);
                }
                float temp = p_intensity[npeak - 1] * sqrt(fabs(sigmax[npeak - 1] * sigmay[npeak - 1])) * 3.14159265358979f;
                fprintf(fp3, "{\"cs_x\": %f,\"cs_y\": %f, \"intergral\": %f, \"intergral2\": %f, \"sigmax\": %f, \"sigmay\": %f,  \"type\": 1, \"index\": %f}", p1_ppm[npeak - 1], p2_ppm[npeak - 1], fitted_volume[npeak - 1], num_sums[npeak - 1][0], sigmax[npeak - 1], sigmay[npeak - 1], amplitudes[npeak - 1]);

                fprintf(fp3, "]");
            }
            fclose(fp3);
        }
    }

    return 1;
};



// bool spectrum_fit::print_intensities(std::string outfname)
// {
//     FILE *fp=fopen(outfname.c_str(), "w");
//     for(int i=0;i<p_intensity_all_spectra.size();i++)
//     {
//         fprintf(fp,"%10d ",i+1);
//         for(int k=0;k<p_intensity_all_spectra[i].size();k++)
//         {
//             float temp=p_intensity_all_spectra[i][k]*(voigt(0.0,sigmax[i],gammax[i])*voigt(0.0,sigmay[i],gammay[i]));
//             fprintf(fp,"%+e ",temp);
//         }
//         fprintf(fp,"\n");
//     }
//     fclose(fp);
//     return true;
// }

bool spectrum_fit::assess_size()
{
    bool b=true;

    if(p_intensity.size()!=p1.size() || p_intensity.size()!=p2.size() || p_intensity.size()!=sigmax.size() || p_intensity.size()!=sigmay.size() || 
    p_intensity.size()!=num_sums.size() || p_intensity.size()!=peak_index.size()  || p_intensity.size()!=err.size() )
    {
        std::cout<<"ERROR:  vector size is not consistent in spectrum_fit."<<std::endl;
        std::cout<<"size of p_intensity is "<<p_intensity.size()<<std::endl;
        std::cout<<"size of p1 is "<<p1.size()<<std::endl;
        std::cout<<"size of p2 is "<<p2.size()<<std::endl;
        std::cout<<"size of simgax is "<<sigmax.size()<<std::endl;
        std::cout<<"size of sigmay is "<<sigmay.size()<<std::endl;
        std::cout<<"size of num_sum is "<<num_sums.size()<<std::endl;
        std::cout<<"size of peak_index is "<<peak_index.size()<<std::endl;
        std::cout<<"size of err is "<<err.size()<<std::endl; 
        b=false;
    }
    return b;
}


bool spectrum_fit::peak_reading(std::string infname)
{
    bool b_read;
    std::string stab(".tab");
    std::string slist(".list");
    std::string sjson(".json");

    std::cout<<"read peaks from file "<<infname<<std::endl;


    if(std::equal(stab.rbegin(), stab.rend(), infname.rbegin()))
    {
        b_read=peak_reading_pipe(infname);
    }
    else if(std::equal(slist.rbegin(), slist.rend(), infname.rbegin()))
    {
        b_read=peak_reading_sparky(infname);
    }
    else if(std::equal(sjson.rbegin(), sjson.rend(), infname.rbegin()))
    {
        b_read=peak_reading_json(infname);
    }
    else
    {
        b_read=false;
        std::cout<<"ERROR: unknown peak list file format. Skip peaks reading."<<std::endl;
    }

    //set gamma if it is not readed in.
    gammax.resize(p1.size(),1e-20);
    gammay.resize(p1.size(),1e-20);


    //remove out of range peaks
    for(int i=p_intensity.size()-1;i>=0;i--)
    {
        if(p1[i]<1 || p1[i]>xdim-2 || p2[i]<1 || p2[i]>ydim-2)
        {
            p1.erase(p1.begin()+i);
            p2.erase(p2.begin()+i);
            p1_ppm.erase(p1_ppm.begin()+i);
            p2_ppm.erase(p2_ppm.begin()+i);
            p_intensity.erase(p_intensity.begin()+i);
            sigmax.erase(sigmax.begin()+i);
            sigmay.erase(sigmay.begin()+i);
            gammax.erase(gammax.begin()+i);
            gammay.erase(gammay.begin()+i);
            user_comments.erase(user_comments.begin()+i);
        }
    }
    std::cout<<"Remove out of bound peaks done."<<std::endl;

    for(int i=0;i<p_intensity.size();i++)
    {
        peak_index.push_back(i);
    }

    //lable peaks that should not move in fitting stage.
    peak_cannot_move_flag.resize(p_intensity.size(),0);

    for(int i=0;i<p_intensity.size();i++)
    {
        int ntemp=int(round(p1[i]))+int(round(p2[i]))*xdim;
        if(p_intensity[i]/spects[0][ntemp]<0.5)
        {
            peak_cannot_move_flag[i]=1; //peak can't move because of low reliability    
        }
    }

    //fill p_intensity_all_spectra using spectra data if needed.
    for(int i=0;i<p1.size();i++)
    {
        std::vector<double> temp;
        temp.push_back(p_intensity[i]); //intensity of first spec, either from peak list or from spects[0]
        for(int n=1;n<spects.size();n++)
        {
            int n1=int(p1[i]+0.5)-1; // -1 because start at 0 in this program but start from 1 in pipe's tab file.
            int n2=int(p2[i]+0.5)-1;
            if(n1<0) n1=0; if(n1>xdim-1) n1=xdim-1;
            if(n2<0) n2=0; if(n2>ydim-1) n2=ydim-1; //water proof
            temp.push_back(spects[n][n2*xdim+n1]); //n1 is direct dimension; n2 is indirect
        }
        p_intensity_all_spectra.push_back(temp); 
    }


    //required for fitting stage because we need to set up wx and wy, which are important
    std::vector<double> sx,sy;

    sx.clear();
    sy.clear();
    for (unsigned int i = 0; i < p1.size(); i++)
    {

        float s1 = 0.5346 * gammax[i] * 2 + std::sqrt(0.2166 * 4 * gammax[i] * gammax[i] + sigmax[i] * sigmax[i] * 8 * 0.6931);
        float s2 = 0.5346 * gammay[i] * 2 + std::sqrt(0.2166 * 4 * gammay[i] * gammay[i] + sigmay[i] * sigmay[i] * 8 * 0.6931);
        sx.push_back(s1);
        sy.push_back(s2);
    }
    median_width_x=ldw_math_spectrum_2d::calcualte_median(sx);
    median_width_y=ldw_math_spectrum_2d::calcualte_median(sy);


    std::cout<<"Median peak width is estimated to be "<<median_width_x<<" "<<median_width_y<< " points from picking."<<std::endl;
    
    if(median_width_x<3.0)
    {
        median_width_x=3.0;
        std::cout<<"Set median peak width along x to 5.0"<<std::endl;    
    }
    
    if(median_width_y<3.0)
    {
        median_width_y=3.0;
        std::cout<<"Set median peak width along y to 5.0"<<std::endl;    
    }

    std::cout<<"loaded in "<<p1.size()<<" opeaks."<<std::endl;
    if(p1.size()==0) b_read=false;
    return b_read;
}


bool spectrum_fit::peak_reading_sparky(std::string fname)
{
    std::string line,p;
    std::vector< std::string> ps;
    std::stringstream iss;

    int xpos=-1;
    int ypos=-1;
    int ass=-1;

    std::ifstream fin(fname);
    getline(fin,line);
    iss.str(line);
    while(iss>>p)
    {
        ps.push_back(p);
    }
    
    for(int i=0;i<ps.size();i++)
    {
        if(ps[i]=="w2") {xpos=i;}  //in sparky, w2 is direct dimension
        else if(ps[i]=="w1") {ypos=i;}
        else if(ps[i]=="Assignment") {ass=i;}   
    }

    if( xpos==-1 || ypos==-1 )
    {
        std::cout<<"One or more required varibles are missing."<<std::endl;
        return false;
    }

    int c=0;
    while(getline(fin,line))
    {
        iss.clear();
        iss.str(line);
        ps.clear();
        while(iss>>p)
        {
            ps.push_back(p);
        }

        if(ps.size()<3) continue; //empty line??

        c++;
        p1_ppm.push_back(atof(ps[xpos].c_str()));
        p2_ppm.push_back(atof(ps[ypos].c_str()));
        p_intensity.push_back(0.0);
        sigmax.push_back(2.0);
        sigmay.push_back(2.0);

        if(ass!=-1)
        {
            user_comments.push_back(ps[ass]);
        }
        else
        {
            user_comments.push_back("peaks"+std::to_string(c));
        }
    }

    //get points from ppm
    p1.clear();
    p2.clear();

    for (unsigned int i = 0; i < p1_ppm.size(); i++)
    {
        p1.push_back((p1_ppm[i] - begin1) / step1); //direct dimension
        p2.push_back((p2_ppm[i] - begin2) / step2); //indirect dimension
    }

    for (int i = 0; i < p1.size(); i++)
    {
        int n1 = int(p1[i] + 0.5) - 1; // -1 because start at 0 in this program but start from 1 in pipe's tab file.
        int n2 = int(p2[i] + 0.5) - 1;

        if (n1 < 0)
            n1 = 0;
        if (n1 > xdim - 1)
            n1 = xdim - 1;
        if (n2 < 0)
            n2 = 0;
        if (n2 > ydim - 1)
            n2 = ydim - 1;                      //water proof
        p_intensity[i] = spect[n2 * xdim + n1]; //n1 is direct dimension; n2 is indirect
    }

    return true;
}


bool spectrum_fit::peak_reading_pipe(std::string fname)
{
    std::string line,p;
    std::vector< std::string> ps;
    std::stringstream iss;

    int index=-1;
    int xpos=-1;
    int ypos=-1;
    int xpos_ppm=-1;
    int ypos_ppm=-1;
    int xw=-1;
    int yw=-1;
    int height=-1;
    int ass=-1;

    std::ifstream fin(fname);
    getline(fin,line);
    iss.str(line);
    while(iss>>p)
    {
        ps.push_back(p);
    }
    if(ps[0]!="VARS")
    {
        std::cout<<"First word of first line must be VARS."<<std::endl;
        return false;
    }
    ps.erase(ps.begin()); //remove first words (VARS)
    for(int i=0;i<ps.size();i++)
    {
        if(ps[i]=="INDEX") {index=i;}
        else if(ps[i]=="X_AXIS") {xpos=i;}
        else if(ps[i]=="Y_AXIS") {ypos=i;}
        else if(ps[i]=="X_PPM") {xpos_ppm=i;}
        else if(ps[i]=="Y_PPM") {ypos_ppm=i;}
        else if(ps[i]=="XW") {xw=i;}
        else if(ps[i]=="YW") {yw=i;}       
        else if(ps[i]=="HEIGHT") {height=i;}   
        else if(ps[i]=="ASS") {ass=i;}   
    }

    if( (xpos==-1 || ypos==-1) && (xpos_ppm==-1 || ypos_ppm==-1) )
    {
        std::cout<<"One or more required varibles are missing."<<std::endl;
        return false;
    }
   
    getline(fin,line); //read in second line
    user_comments.clear();
    int c=0;
    while(getline(fin,line))
    {
        c++;
        iss.clear();
        iss.str(line);
        ps.clear();
        while(iss>>p)
        {
            ps.push_back(p);
        }

        if(ps.size()<6) continue; //empty line??

        if(xpos!=-1 && ypos!=-1)
        {
            p1.push_back(atof(ps[xpos].c_str())-1);
            p2.push_back(atof(ps[ypos].c_str())-1);
        }
        if(xpos_ppm!=-1 && ypos_ppm!=-1)
        {
            p1_ppm.push_back(atof(ps[xpos_ppm].c_str()));
            p2_ppm.push_back(atof(ps[ypos_ppm].c_str()));
        }


        if(height!=-1)
        {
            p_intensity.push_back(atof(ps[height].c_str()));   
        }
        else p_intensity.push_back(0.0);

        if(xw!=-1) 
        {      
            float s=atof(ps[xw].c_str());
            sigmax.push_back(s/2.35);
        }
        else sigmax.push_back(3.0);
        
        if(yw!=-1)
        {
            float s=atof(ps[yw].c_str());
            sigmay.push_back(s/2.35);
        }
        else sigmay.push_back(3.0);

        if(ass!=-1)
        {
            user_comments.push_back(ps[ass]);
        }
        else
        {
            user_comments.push_back("peaks"+std::to_string(c));
        }
    }

    //fill in intensity information from spectrum
    if(height==-1)
    {
        for(int i=0;i<p1.size();i++)
        {
            int n1=int(p1[i]+0.5)-1; // -1 because start at 0 in this program but start from 1 in pipe's tab file.
            int n2=int(p2[i]+0.5)-1;
            if(n1<0) n1=0; if(n1>xdim-1) n1=xdim-1;
            if(n2<0) n2=0; if(n2>ydim-1) n2=ydim-1; //water proof
            p_intensity[i]=spect[n2*xdim+n1]; //n1 is direct dimension; n2 is indirect
        }
    }


    if(p1_ppm.size()>0) //fill in point from ppm. 
    {
        p1.clear();
        p2.clear();

        for (unsigned int i = 0; i < p1_ppm.size(); i++)
        {
            p1.push_back((p1_ppm[i] - begin1) / step1); //direct dimension
            p2.push_back((p2_ppm[i] - begin2) / step2); //indirect dimension
        }
    }
    else //fill in ppm from points
    {
        p1_ppm.clear();
        p2_ppm.clear();

        for (unsigned int i = 0; i < p1.size(); i++)
        {
            double f1 = begin1 + step1 * (p1[i]);  //direct dimension
            double f2 = begin2 + step2 * (p2[i]);  //indirect dimension
            p1_ppm.push_back(f1);
            p2_ppm.push_back(f2);
        }
    }
    return true;
}

bool spectrum_fit::peak_reading_json(std::string infname)
{
    Json::Value root,peaks;
    std::ifstream fin;
    fin.open(infname);

    fin>>root;
    peaks=root["picked_peaks"];
    
    user_comments.clear();
    for(int i=0;i<peaks.size();i++)
    {
        p1_ppm.push_back(peaks[i]["cs_x"].asDouble());
        p2_ppm.push_back(peaks[i]["cs_y"].asDouble());
        p_intensity.push_back(peaks[i]["index"].asDouble());  
        sigmax.push_back(peaks[i]["sigmax"].asDouble());
        sigmay.push_back(peaks[i]["sigmay"].asDouble());
        gammax.push_back(peaks[i]["gammax"].asDouble());
        gammay.push_back(peaks[i]["gammay"].asDouble());
        user_comments.push_back("peaks" + std::to_string(i));
    }

    p1.clear();
    p2.clear();

    for (unsigned int i = 0; i < p1_ppm.size(); i++)
    {
        p1.push_back((p1_ppm[i] - begin1) / step1); //direct dimension
        p2.push_back((p2_ppm[i] - begin2) / step2); //indirect dimension
    }

    return true;
};


