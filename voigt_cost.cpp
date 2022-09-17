

#include "ceres/ceres.h"
#include "glog/logging.h"


using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#include "voigt_cost.h"


//Gaussian
mycostfunction_gaussian::~mycostfunction_gaussian(){};
mycostfunction_gaussian::mycostfunction_gaussian(int n, double *x_, double *y_, double *z_)
{
    n_datapoint=n;
    x=x_;
    y=y_;
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

//voigt_1d
mycostfunction_voigt1d::~mycostfunction_voigt1d(){};
mycostfunction_voigt1d::mycostfunction_voigt1d(int n, double *x_, double *z_)
{
    n_datapoint=n;
    x=x_;
    z=z_;   
};

void mycostfunction_voigt1d::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
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

void mycostfunction_voigt1d::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    return;
};

bool mycostfunction_voigt1d::get_residual(const double a, const double x0, const double sigma, const double gamma, double *residual)
{
    double vvx;
    for (int i = 0; i < n_datapoint; i++)
    {
        voigt_helper(x[i] - x0, sigma, gamma, &vvx);
        residual[i] = a * vvx - z[i];
    }
    return true;
}

bool mycostfunction_voigt1d::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a=xx[0][0];
    double x0=xx[1][0];
    double sigmax=fabs(xx[2][0]);
    double gammax=fabs(xx[3][0]);
   
    
    double vvx,r_x,r_sigmax,r_gammax;

    voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
   

    if (jaco != NULL ) //both residual errors and jaco are required.
    {
        for(int i=0;i<n_datapoint;i++)
        {
            voigt_helper(x[i]-x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax); 
            residual[i]=a*vvx-z[i]; 
            jaco[0][i]=vvx; //with respect to a
            jaco[1][i]=-a*r_x; //x0
            jaco[2][i]=a*r_sigmax; //sigmax
            jaco[3][i]=a*r_gammax; //gammax
        }
    }
    else //only require residual errors
    {
        for(int i=0;i<n_datapoint;i++)
        {
            voigt_helper(x[i]-x0, sigmax, gammax, &vvx);      
            residual[i]=a*vvx-z[i];    
        }
    }
    return true;
};


//Voigt 
mycostfunction::~mycostfunction(){};
mycostfunction::mycostfunction(int n, double *x_, double *y_, double *z_)
{
    n_datapoint=n;
    x=x_;
    y=y_;
    z=z_;   
};

void mycostfunction::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
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

void mycostfunction::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    return;
};

bool mycostfunction::Evaluate(double const *const *xx, double *residual, double **jaco) const
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





