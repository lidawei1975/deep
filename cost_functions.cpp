#include <cmath>

#ifdef LMMIN
    #include "lmminimizer.h"
#else
    #include "ceres/ceres.h"
    #include "glog/logging.h"
    using ceres::CostFunction;
    using ceres::AutoDiffCostFunction;
    using ceres::NumericDiffCostFunction;
    using ceres::Problem;
    using ceres::Solver;
    using ceres::Solve;
#endif

#include "kiss_fft.h"

#include "cost_functors.h"



/**
 * IMPORTANT:
 * In our Voigt definition, total peak volume is 1.0 (constant). Fittted a is the peak volume.
 * In our Lorentz definition, total peak volume gamma. (not constant). Fitted a is the peak height.
 * In our Gaussian definition, total peak volume is sqrt(sigma*pi). Fitted a is the peak height.
*/

/**
 * cost functions for ceres-solver
 * Gaussian
*/
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

/**
 * mycostfunction_voigt_lorentz
 * Voigt along x and Lorentzian along y
*/
mycostfunction_voigt_lorentz::~mycostfunction_voigt_lorentz(){};
mycostfunction_voigt_lorentz::mycostfunction_voigt_lorentz(int xdim_, int ydim_, double *z_)
{
    xdim=xdim_;
    ydim=ydim_;
    n_datapoint=xdim*ydim;
    z=z_;   
};

void mycostfunction_voigt_lorentz::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
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

void mycostfunction_voigt_lorentz::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    return;
};

bool mycostfunction_voigt_lorentz::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a=xx[0][0];
    double x0=xx[1][0];
    double sigmax=fabs(xx[2][0]);
    double gammax=fabs(xx[3][0]);
    double y0=xx[4][0];
    double gammay=fabs(xx[5][0]);
    
    double vvx,r_x,r_sigmax,r_gammax;

    voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);

    if (jaco != NULL ) //both residual errors and jaco are required.
    {
        for(int i=0;i<xdim;i++)
        {
            voigt_helper(i-x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax); 
            for(int j=0;j<ydim;j++)
            {
                int ii=i*ydim+j;
                
                double v = 1 / (1 + ((j - y0) / gammay) * ( (j - y0) / gammay));
                double v2 = v * v;
                double gammay2 = gammay * gammay;
                double av =  2 * a * v2;

                residual[ii]=a*vvx*v-z[ii]; 

                jaco[0][ii]=vvx*v; //with respect to a
                jaco[1][ii]=-a*r_x*v; //x0
                jaco[2][ii]=a*r_sigmax*v; //sigmax
                jaco[3][ii]=a*r_gammax*v; //gammax

                jaco[4][ii]=av*(j-y0)/gammay2*vvx; //y0
                jaco[5][ii]=av*(j-y0)*(j-y0)/(gammay2*gammay)*vvx; //gammay
            }
        }
    }
    else //only require residual errors
    {
        for(int i=0;i<xdim;i++)
        {
            voigt_helper(i-x0, sigmax, gammax, &vvx);  //vvx is the intensity of the voigt function at x=i     
            for(int j=0;j<ydim;j++)
            {
                int ii=i*ydim+j; //location of the data point
                double v = 1 / (1 + ( (j - y0) / gammay) * ( (j - y0) / gammay)); //v is the the intensity of the Lorentzian function at y=j
                residual[ii]=a*vvx*v-z[ii];    
            }
        }
    }
    return true;
};

bool Exactshape::operator()(const double *const a, const double *const x0, const double *const y0, const double *const r2x, const double *const r2y, const double *const sx, const double *const sy, double *residue) const
{
    std::vector<double> result;
    result.resize(41 * 41, 0.0);
    val(*a, *x0, *y0, *r2x, *r2y, *sx, *sy, result.data());
    for (int i = 0; i < 41 * 41; i++)
    {
        residue[i] = result[i] - zz[i];
    }
    return true;
};

bool Exactshape::val(const double a, const double x0, const double y0, const double r2x, const double r2y, const double sx, const double sy, double *v) const
{

    std::vector<double> resultx, resulty;

    kiss_fft_cfg cfg;
    kiss_fft_cpx *y, *z;

    int nx2 = nx * zfx;
    int ny2 = ny * zfy;

    //x dimension
    y = new kiss_fft_cpx[nx2];
    z = new kiss_fft_cpx[nx2];

    for (int i = 0; i < nx; i++)
    {
        double sp = pow(sin(PI * 0.5 + PI * 0.448 / nx * i), 3.684);
        if (i == 0)
            sp = 0.5;
        y[i].r = cos((0.5 - x0 / nx2) * 2 * PI * (i - sx)) * exp(-fabs(r2x) * i) * sp;
        y[i].i = -sin((0.5 - x0 / nx2) * 2 * PI * (i - sx)) * exp(-fabs(r2x) * i) * sp;
        // std::cout<<y[i].r<<" ";
    }
    // std::cout<<std::endl;

    for (int i = nx; i < nx2; i++)
    {
        y[i].r = 0;
        y[i].i = 0;
    }

    if ((cfg = kiss_fft_alloc(nx2, 0, NULL, NULL)) != NULL)
    {
        kiss_fft(cfg, y, z);
        free(cfg);
    }
    else
    {
        return false;
    }

    for (int i = nx2 / 2 - 20; i <= nx2 / 2 + 20; i++)
    {
        resultx.push_back(z[i].r);
    }

    delete[] y;
    delete[] z;

    //y dimension
    y = new kiss_fft_cpx[ny2];
    z = new kiss_fft_cpx[ny2];

    for (int i = 0; i < ny; i++)
    {
        double sp = pow(sin(PI * 0.5 + PI * 0.448 / ny * i), 3.684);
        if (i == 0)
            sp = 0.5;
        y[i].r = cos((0.5 - y0 / ny2) * 2 * PI * (i - sy)) * exp(-fabs(r2y) * i) * sp;
        y[i].i = -sin((0.5 - y0 / ny2) * 2 * PI * (i - sy)) * exp(-fabs(r2y) * i) * sp;
    }

    for (int i = ny; i < ny2; i++)
    {
        y[i].r = 0;
        y[i].i = 0;
    }

    if ((cfg = kiss_fft_alloc(ny2, 0, NULL, NULL)) != NULL)
    {
        kiss_fft(cfg, y, z);
        free(cfg);
    }
    else
    {
        return false;
    }

    for (int i = ny2 / 2 - 20; i <= ny2 / 2 + 20; i++)
    {
        resulty.push_back(z[i].r);
    }

    delete[] y;
    delete[] z;

    //final result
    for (int i = 0; i < 41; i++)
    {
        for (int j = 0; j < 41; j++)
        {
            v[i * 41 + j] = a * resultx[i] * resulty[j];
        }
    }

    return true;
};


/**
 * 1D part
*/

// Gaussain_1d
mycostfunction_gaussian1d::~mycostfunction_gaussian1d(){};
mycostfunction_gaussian1d::mycostfunction_gaussian1d(int n, double *z_)
{
    n_datapoint = n;
    z = z_;
};
bool mycostfunction_gaussian1d::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a = xx[0][0];
    double x0 = xx[1][0];
    double sigmax = fabs(xx[2][0]);

    if (jaco != NULL) // both residual errors and jaco are required.
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            double x_sigmax = (i - x0) / sigmax;
            double g = exp((x0 - i) * x_sigmax);
            double ag = a * g;

            residual[i] = ag - z[i];
            jaco[0][i] = g;                        // with respect to a
            jaco[1][i] = ag * 2 * x_sigmax;        // x0
            jaco[2][i] = ag * x_sigmax * x_sigmax; // sigmax
        }
    }
    else // only require residual errors
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            residual[i] = a * exp(-(i - x0) * (i - x0) / sigmax) - z[i];
        }
    }
    return true;
};

// lorentz 1d
mycostfunction_lorentz1d::~mycostfunction_lorentz1d(){};
mycostfunction_lorentz1d::mycostfunction_lorentz1d(int n, double *z_)
{
    n_datapoint = n;
    z = z_;
};
bool mycostfunction_lorentz1d::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a = xx[0][0];
    double x0 = xx[1][0];
    double gamma = fabs(xx[2][0]);

    if (jaco != NULL) // both residual errors and jaco are required.
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            double v = 1 / (1 + ( (i - x0) / gamma) * ( (i - x0) / gamma));
            double v2 = v * v;
            double gamma2 = gamma * gamma;
            double av = a * v2 * 2;

            residual[i] = a * v - z[i];
            jaco[0][i] = v;                                             // with respect to a
            jaco[1][i] = av * (i - x0) / gamma2;                       // x0
            jaco[2][i] = av * ((i - x0) * (i - x0)) / (gamma2 * gamma); // gammax
        }
    }
    else // only require residual errors
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            residual[i] = a / (1 + ( (i - x0) / gamma) * ( (i - x0) / gamma)) - z[i];
        }
    }
    return true;
};


/**
 * @brief for pseudo 2D voigt fitting, analytical derivative
 * Peak amplitude is defined as A = A0*exp(-g*g*D) where A0 and D are fitting parameters
 * while g is squared z gradient.
 */
// voigt_1d
mycostfunction_voigt1d_doesy::~mycostfunction_voigt1d_doesy(){};
mycostfunction_voigt1d_doesy::mycostfunction_voigt1d_doesy(double t_,int n_datapoint_, double *z_)
{
    z_gradient_squared = t_; 
    n_datapoint = n_datapoint_;
    z = z_;
};

void mycostfunction_voigt1d_doesy::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
{
    double v, l;
    double z_r = x0 / (sqrt(2) * sigma);
    double z_i = gamma / (sqrt(2) * sigma);
    double sigma2 = sigma * sigma;
    double sigma3 = sigma * sigma2;

    re_im_w_of_z(z_r, z_i, &v, &l);
    *vv = v / sqrt(2 * M_PI * sigma2);

    double t1 = z_i * l - z_r * v;
    double t2 = z_r * l + z_i * v;

    *r_x0 = t1 / (sigma2 * M_SQRT_PI);
    *r_gamma = (t2 - M_1_SQRT_PI) / (sigma2 * M_SQRT_PI);
    *r_sigma = -v / M_SQRT_2PI / sigma2 - t1 * x0 / sigma3 / M_SQRT_PI - (t2 - M_1_SQRT_PI) * gamma / sigma3 / M_SQRT_PI;

    return;
};

void mycostfunction_voigt1d_doesy::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v, l;
    double z_r = x0 / (sqrt(2) * sigma);
    double z_i = gamma / (sqrt(2) * sigma);
    double sigma2 = sigma * sigma;

    re_im_w_of_z(z_r, z_i, &v, &l);
    *vv = v / sqrt(2 * M_PI * sigma2);
    return;
};

bool mycostfunction_voigt1d_doesy::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a0 = xx[0][0]; //This is the amplitude at t=0
    
    double diff_sqrt = xx[4][0]; //This is SQRT of the diffusion coefficient. We fit the SQRT of the diffusion coefficient to avoid diffusion coefficient
    double diff = diff_sqrt*diff_sqrt; //This is the diffusion coefficient. 
    double a_scale=exp(-z_gradient_squared*diff); //This is the amplitude scale factor at t=t
    double a = a0*a_scale; //This is the amplitude at t=t
    double x0 = xx[1][0]; //This is the center position
    double sigmax = fabs(xx[2][0]);
    double gammax = fabs(xx[3][0]);

    double vvx, r_x, r_sigmax, r_gammax;

    voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);

    if (jaco != NULL) // both residual errors and jaco are required.
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            voigt_helper(i - x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
            residual[i] = a * vvx - z[i];
            jaco[0][i] = vvx * a_scale;         //  dz/da=vvx and da/da0 = a_scale 
            jaco[1][i] = -a * r_x;              // derivative with respect to x0
            jaco[2][i] = a * r_sigmax;          // derivative with respect to sigmax
            jaco[3][i] = a * r_gammax;          // derivative with respect to gammax
            jaco[4][i] = -vvx * a0 * a_scale * z_gradient_squared * 2.0 * diff_sqrt;   // dz/da=vvx, da/dD = -a0*t*t*exp(-t*t*D), dD/dsqrtD =2*sqrtD
        }
    }
    else // only require residual errors
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            voigt_helper(i - x0, sigmax, gammax, &vvx);
            residual[i] = a * vvx - z[i];
        }
    }
    return true;
};


/**
 * @brief voigt_1d cost funciton. Single peak fitting
 * 
 */
mycostfunction_voigt1d::~mycostfunction_voigt1d(){};
mycostfunction_voigt1d::mycostfunction_voigt1d(int n, double *z_)
{
    n_datapoint = n;
    z = z_;
};

void mycostfunction_voigt1d::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
{
    double v, l;
    double z_r = x0 / (sqrt(2) * sigma);
    double z_i = gamma / (sqrt(2) * sigma);
    double sigma2 = sigma * sigma;
    double sigma3 = sigma * sigma2;

    re_im_w_of_z(z_r, z_i, &v, &l);
    *vv = v / sqrt(2 * M_PI * sigma2);

    double t1 = z_i * l - z_r * v;
    double t2 = z_r * l + z_i * v;

    *r_x0 = t1 / (sigma2 * M_SQRT_PI);
    *r_gamma = (t2 - M_1_SQRT_PI) / (sigma2 * M_SQRT_PI);
    *r_sigma = -v / M_SQRT_2PI / sigma2 - t1 * x0 / sigma3 / M_SQRT_PI - (t2 - M_1_SQRT_PI) * gamma / sigma3 / M_SQRT_PI;

    return;
};

void mycostfunction_voigt1d::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v, l;
    double z_r = x0 / (sqrt(2) * sigma);
    double z_i = gamma / (sqrt(2) * sigma);
    double sigma2 = sigma * sigma;

    re_im_w_of_z(z_r, z_i, &v, &l);
    *vv = v / sqrt(2 * M_PI * sigma2);
    return;
};

bool mycostfunction_voigt1d::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a = xx[0][0];
    double x0 = xx[1][0];
    double sigmax = fabs(xx[2][0]);
    double gammax = fabs(xx[3][0]);

    double vvx, r_x, r_sigmax, r_gammax;

    voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);

    if (jaco != NULL) // both residual errors and jaco are required.
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            voigt_helper(i - x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
            residual[i] = a * vvx - z[i];
            jaco[0][i] = vvx;          // with respect to a
            jaco[1][i] = -a * r_x;     // x0
            jaco[2][i] = a * r_sigmax; // sigmax
            jaco[3][i] = a * r_gammax; // gammax
        }
    }
    else // only require residual errors
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            voigt_helper(i - x0, sigmax, gammax, &vvx);
            residual[i] = a * vvx - z[i];
        }
    }
    return true;
};

// voigt_1d, n peaks togather
mycostfunction_nvoigt1d::~mycostfunction_nvoigt1d(){};
mycostfunction_nvoigt1d::mycostfunction_nvoigt1d(int np_, int n, double *z_)
{
    np = np_;
    n_datapoint = n;
    z = z_;
};

void mycostfunction_nvoigt1d::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
{
    double v, l;
    double z_r = x0 / (sqrt(2) * sigma);
    double z_i = gamma / (sqrt(2) * sigma);
    double sigma2 = sigma * sigma;
    double sigma3 = sigma * sigma2;

    re_im_w_of_z(z_r, z_i, &v, &l);
    *vv = v / sqrt(2 * M_PI * sigma2);

    double t1 = z_i * l - z_r * v;
    double t2 = z_r * l + z_i * v;

    *r_x0 = t1 / (sigma2 * M_SQRT_PI);
    *r_gamma = (t2 - M_1_SQRT_PI) / (sigma2 * M_SQRT_PI);
    *r_sigma = -v / M_SQRT_2PI / sigma2 - t1 * x0 / sigma3 / M_SQRT_PI - (t2 - M_1_SQRT_PI) * gamma / sigma3 / M_SQRT_PI;
    return;
};

void mycostfunction_nvoigt1d::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v, l;
    double z_r = x0 / (sqrt(2) * sigma);
    double z_i = gamma / (sqrt(2) * sigma);
    double sigma2 = sigma * sigma;

    re_im_w_of_z(z_r, z_i, &v, &l);
    *vv = v / sqrt(2 * M_PI * sigma2);
    return;
};

bool mycostfunction_nvoigt1d::Evaluate(double const *const *xx, double *residual, double **jaco) const
{

    double vvx, r_x, r_sigmax, r_gammax;
    // voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);

    if (jaco != NULL) // both residual errors and jaco are required.
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            residual[i] = -z[i];
        }
        for (int m = 0; m < np; m++)
        {
            /**
             * xx is a 2D array, xx[0] is for the first peak, xx[1] is for the second peak, etc.
             * xx[0][0,1,2,3] are peak intensity, peak position, peak sigma, peak gamma of the first peak
            */
            double a = xx[m][0];
            double x0 = xx[m][1];
            double sigmax = fabs(xx[m][2]);
            double gammax = fabs(xx[m][3]);
            for (int i = 0; i < n_datapoint; i++)
            {
                voigt_helper(i - x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
                residual[i] += a * vvx;

                /**
                 * jacon is a 2D array. jaco[0] is for the first peak, jaco[1] is for the second peak, etc.
                 * jaco[0]'s length is n_datapoint*n_fitting_paramter_of_peak_0
                 * res[0] with respect to a,pos,sigma,gamma, followed by res[1] with respect to a,pos,sigma,gamma, etc.
                */
                jaco[m][i * 4 + 0] = vvx;          // with respect to a
                jaco[m][i * 4 + 1] = -a * r_x;     // x0
                jaco[m][i * 4 + 2] = a * r_sigmax; // sigmax
                jaco[m][i * 4 + 3] = a * r_gammax; // gammax
            }
        }
    }
    else // only require residual errors
    {
        for (int i = 0; i < n_datapoint; i++)
        {
            residual[i] = -z[i];
        }
        for (int m = 0; m < np; m++)
        {
            double a = xx[m][0];
            double x0 = xx[m][1];
            double sigmax = fabs(xx[m][2]);
            double gammax = fabs(xx[m][3]);
            for (int i = 0; i < n_datapoint; i++)
            {
                voigt_helper(i - x0, sigmax, gammax, &vvx);
                residual[i] += a * vvx;
            }
        }
    }
    return true;
};
