#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>
#include <time.h>
#include <sys/time.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>	
#include <Eigen/SparseQR>


#include "kiss_fft.h"
#include "json/json.h"
#include "ldw_math.h"
#include "commandline.h"
#include "fid_1d.h"
#include "cubic_spline.h"
#include "cost_functors.h"
#include "lmminimizer.h"
#include "DeepConfig.h"


class spectrum_filter_1d: public fid_1d
{
private:
    /**
     * @brief center and width (FWHH) of the peak, in ppm, that is to be removed.
     * In intermedate steps, peak is fitted as Voigt profile (convolution of Gaussian (sigma) and Lorentzian(gamma)).
    */
    double center;
    double width; 
    double sigma,gamma;
    double start, end;

    /**
     * Fitted urea signal
    */
    std::vector<double> urea_signal;
    int urea_signal_start;
    int urea_signal_end;

    /**
     * Internal function to calculate the filter.
    */
    bool step1_fit_peak();
    bool step1_set_vars();
    bool step3_fit_baseline();
    

    bool calculate_baseline_solver(int c, const std::vector<double> &a,double b,std::vector<double> &y,Eigen::VectorXd &solution);

public:
    spectrum_filter_1d();
    ~spectrum_filter_1d();

    bool init_signal(double center, double width, double start, double end);
    bool filter(bool b_opt);
    bool apply_filter();
    bool save_filtered_spectrum(std::string fname);

    bool read_urea_signal(std::string fname);
    bool save_urea_signal(std::string fname);
};

spectrum_filter_1d::spectrum_filter_1d()
{
    urea_signal_start=0;
    urea_signal_end=0;
    urea_signal.clear();
};

spectrum_filter_1d::~spectrum_filter_1d()
{
};

bool spectrum_filter_1d::save_filtered_spectrum(std::string fname)
{
    return fid_1d::write_nmrpipe_ft1(fname);
}

bool spectrum_filter_1d::init_signal(double center_, double width_,double start_,double end_)
{
    this->center=center_;
    this->width=width_;
    /**
     * Make sure start > end (in NMR, ppm is decreasing from left to right)
    */
    if(start_ >= end_)
    {
        this->start=start_;
        this->end=end_;
    }
    else
    {
        this->start=end_;
        this->end=start_;
    }

    /**
     * if center is not within start and end, set it to the center of start and end.
    */
    if(center > start || center < end)
    {
        this->center=(start+end)/2.0;
        std::cout<<"Center is not within start and end. Set it to the center of start and end."<<std::endl;
    }

    return true;
}

bool spectrum_filter_1d::filter(bool b_opt=true)
{
    /**
     * Make sure noise level is calculated. Call base class function.
    */
    fid_1d::est_noise_level();

    /**
     * Fit a Voigt profile to determine center and width, if b_opt is true.
     * If b_opt is false, use the center and width provided by user in this->center and this->width.
    */
    if ((b_opt ==true && this->step1_fit_peak()) || (b_opt==false && this->step1_set_vars()))
    {
        /**
         * Fit a baseline to the urea signal.
        */
        if (this->step3_fit_baseline())
        {
            return true;
        }
    }
    return false;
}

bool spectrum_filter_1d::apply_filter()
{
    /**
     * To make sure spect is still continuous, before apply baseline, calculate the original slope.
     */
    double slope = (urea_signal[urea_signal.size() - 1] - urea_signal[0]) / (urea_signal_end - urea_signal_start - 1);
    double intercept = urea_signal[0];

    /**
     * Apply the baseline to remove urea signal from the spectrum.
     * At the same time, add back the original slope to make sure the spectrum is continuous.
     */
    for (int i = urea_signal_start; i < urea_signal_end; i++)
    {
        spectrum_real[i] -= urea_signal[i - urea_signal_start];
        spectrum_real[i] += slope * (i - urea_signal_start) + intercept;
    }

    return true;
}

bool spectrum_filter_1d::calculate_baseline_solver(int c, const std::vector<double> &a,double b,std::vector<double> &y,Eigen::VectorXd &solution)
{
    int xdim=y.size();
    Eigen::MatrixXd d(xdim,xdim);
    Eigen::VectorXd m(xdim);


    for(int i=0;i<xdim;i++)
    {
        for(int j=0;j<xdim;j++)
        {
            d(i,j)=0.0f;
        }
    }


    //0,0
    if(solution(0)>y[0])
    {
        d(0,0) = 2*a[0] +2*b;
    }
    else
    {
        d(0,0) = 2*a[0];
    }
    d(0,1) = -4*a[0];
    d(0,2) = 2*a[0];

    //1,1
    d(1,0) = -4*a[1];
    if(solution(1)>y[1])
    {
        d(1,1) = 10*a[1]+2*b;
    }
    else
    {
        d(1,1) = 10*a[1];
    }
    d(1,2)=-8*a[1];
    d(1,3)=2*a[1];

    //2 to xdim-2
    for(int i=2;i<xdim-2;i++)
    {
        d(i,i-2) = 2*a[i];
        d(i,i-1) = -8*a[i];  
        
            if(solution(i)>y[i])
            {
                d(i,i) = 12*a[i]+2*b;
            }
            else
            {
                d(i,i) = 12*a[i];
            }
        
        d(i,i+1) = -8*a[i];
        d(i,i+2) = 2*a[i];
    }    

    //second last row, xdim-1
    d(xdim-2,xdim-4)=2*a[xdim-2];
    d(xdim-2,xdim-3)=-8*a[xdim-2];
    if(solution(xdim-2)>y[xdim-2])
    {
        d(xdim-2,xdim-2) = 10*a[xdim-2]+2*b;
    }
    else
    {
        d(xdim-2,xdim-2) = 10*a[xdim-2];
    }
    d(xdim-2,xdim-1) = -4*a[xdim-2];
    
    //last row, xdim
    d(xdim-1,xdim-3) = 2*a[xdim-1];
    d(xdim-1,xdim-2) = -4*a[xdim-1];
    if(solution(xdim-1)>y[xdim-1])
    {
        d(xdim-1,xdim-1) = 2*a[xdim-1]+2*b;
    }
    else
    {
        d(xdim-1,xdim-1) = 2*a[xdim-1];
    }

    for (int i = 0; i < xdim; i++)
    {
        if (solution(i) > y[i])
        {
            m(i) = 2 * b * y[i] + 1.0;
        }
        else
        {
            m(i) = 1.0;
        }
    }

    // solve Ax = b,
    solution=d.completeOrthogonalDecomposition().solve(m);

    /**
     * Debug code
     * print d and m and solution
    */
    if(c<0)
    {
        std::ofstream fout("d_"+std::to_string(c)+".txt");
        for(int i=0;i<xdim;i++)
        {
            for(int j=0;j<xdim;j++)
            {
                fout<<d(i,j)<<" ";
            }
            fout<<std::endl;
        }
        fout.close();

        fout.open("m_"+std::to_string(c)+".txt");
        for(int i=0;i<xdim;i++)
        {
            fout<<m(i)<<std::endl;
        }
        fout.close();

        fout.open("solution_"+std::to_string(c)+".txt");
        for(int i=0;i<xdim;i++)
        {
            fout<<solution(i)<<std::endl;
        }
        fout.close();
    }

    
    return true;
}

bool spectrum_filter_1d::step1_set_vars()
{
    /**
     * Set the variables for step1.
     * ppm_value = begin1 + point_value * step1;
    */
    int peak_center = (center-begin1)/step1; 
    int peak_width = width/fabs(step1); //step1 is always negative

    center = peak_center;
    width = peak_width;

    return true;
}

bool spectrum_filter_1d::step1_fit_peak()
{
     /**
     * Set the variables for step1.
     * ppm_value = begin1 + point_value * step1;
    */
    int peak_center = (center-begin1)/step1; 
    int peak_width = width/fabs(step1); //step1 is always negative

    int fitting_start = (this->start-begin1)/step1; //inclusive
    int fitting_end = (this->end-begin1)/step1; //exclusive 
    /**
     * Run nonlinear fitting to fit the peak.
    */
    std::vector<double> z;
    float max_z = *std::max_element(spectrum_real.begin()+fitting_start,spectrum_real.begin()+fitting_end);
    for(int i=fitting_start;i<fitting_end;i++)
    {
        z.push_back(spectrum_real[i]/max_z);
    }

    /**
     * Debug code. print z
    */
    std::ofstream fout("z.txt");
    for(int i=0;i<z.size();i++)
    {
        fout<<z[i]<<std::endl;
    }
    fout.close();

    double **par;
    par = new double*[4];
    for(int i=0;i<4;i++) 
    {
        par[i]=new double[1];
    }

    
    par[1][0]=peak_center - fitting_start; //center
    par[2][0]=peak_width/3; //sigma
    par[3][0]=10; //gamma
    par[0][0]=1/voigt(0.0,par[2][0],par[3][0]);


    mycostfunction_voigt1d cost_function(z.size(),z.data());
    class levmarq minimizer;
    minimizer.solve(4, par, z.size(), NULL /**weight*/, cost_function);


    double h = par[0][0];
    center = par[1][0] + fitting_start;
    sigma=fabs(par[2][0]);
    gamma=fabs(par[3][0]);
    width = 0.5346*sigma*2 + sqrt(0.2166*gamma*gamma*4 + sigma*sigma * 8 * 0.6931);

    for(int i=0;i<4;i++) {delete [] par[i];}
    delete [] par;

    return true;
}

bool spectrum_filter_1d::step3_fit_baseline()
{
    /**
     * At this moment, center and width are points from step1. No need to convert to ppm.
    */
    int peak_center = center;
    int peak_width = width;

    int fitting_start = (this->start-begin1)/step1; 
    int fitting_end = (this->end-begin1)/step1;
    int fitting_length = fitting_end - fitting_start;
    int fitting_step = 1;

    /**
     * Reduce the fitting length by half each time, until it is <=1024.
    */
    while(fitting_length>1024)
    {
        fitting_length = fitting_length/2;
        fitting_step*=2;
    }

    peak_width /= fitting_step;

    /**
     * peak_center is relative to fitting_start and step
    */
    peak_center -= fitting_start; 
    peak_center /= fitting_step; 


    /**
     * Prepare fitting region data. NO need to normalize. because a,b parameters will be normalized against noise level.
    */
    std::vector<double> z;
    for(int i=fitting_start;i<fitting_end;i+=fitting_step)
    {
        z.push_back(spectrum_real[i]/noise_level);
    }
    /**
     * Correct possible fitting length error caused by mutitple rounding.
    */
    fitting_length = z.size();
   
    double a0=pow(peak_width,4)*(1e-6);
    double a1=a0/100;
    double b=1.25;
    

    std::vector<double> a(fitting_length,a0);
    /**
     * a: baseline curvature parameter. b: noise level parameter.
     * a is a1 at peak center, varies from a1 to a0 linearly from peak center to cetner+width and center-width to center.
     * a is a0 at further away from peak center.
    */
    for(int i=0;i<fitting_length;i++)
    {
        if(i>=peak_center-peak_width && i<=peak_center+peak_width)
        {
            a[i]=a1+(a0-a1)*fabs(i-peak_center)/peak_width;
        }
    }


    Eigen::VectorXd solution;
    /**
     * Init solution to be all at noise level.
    */
    solution=Eigen::VectorXd::Constant(fitting_length,noise_level);
    /**
     * 10 iterations is usually enough.
    */
    for(int c=0;c<10;c++){
        calculate_baseline_solver(c,a,b,z,solution);
    }

    /**
     * Print d and z for debug
    */
    // std::cout<<"a = "<<a<<" b = "<<b<<std::endl;
    // std::cout<<"Fitting start = "<<fitting_start<<" end = "<<fitting_end<<std::endl;
    // std::ofstream fout("d-z.txt");
    // for(int i=0;i<fitting_length;i++)
    // {
    //     fout<<solution(i)<<" "<<z[i]<<std::endl;
    // }
    // fout.close();

    std::vector<double> baseline_stride(fitting_length);
    for(int i=0;i<fitting_length;i++)
    {
        baseline_stride[i]=solution(i);
    }


    class cublic_spline cs;

    /**
     * initialize cubic spline, notice x index is k-1,2*k-1,3*k-1,...,n*k-1
    */
    cs.calculate_coefficients(baseline_stride); 
  
    /**
     * calculate baseline at each point
    */
    urea_signal.clear();
    for (int m = 0; m < fitting_length*fitting_step; m++)
    {
        double m_double=m;
        m_double=m_double/fitting_step;
        urea_signal.push_back(cs.calculate_value(m_double)*noise_level); //double to float conversion
    }
    urea_signal_start=fitting_start;
    urea_signal_end=fitting_end;

    /**
     * Debug code
    */
    // std::ofstream fout("urea.txt");
    // for(int i=0;i<fitting_length*fitting_step;i++)
    // {
    //     fout<<fitting_start+i<<"\t"<<urea_signal[i]<<std::endl;
    // }
    // fout.close();

    return true;
};

bool spectrum_filter_1d::read_urea_signal(std::string fname)
{
    std::ifstream fin(fname);
    if(!fin.is_open())
    {
        std::cerr<<"Error: cannot open file "<<fname<<std::endl;
        return false;
    }

    Json::Value root;
    fin>>root;
    fin.close();

    if(root["urea_signal_start"].isNull() || root["urea_signal_end"].isNull() || root["urea_signal"].isNull())
    {
        std::cerr<<"Error: cannot find start, end, or urea_signal in the json file."<<std::endl;
        return false;
    }

    urea_signal_start=root["urea_signal_start"].asInt();
    urea_signal_end=root["urea_signal_end"].asInt();

    Json::Value urea_signal_json=root["urea_signal"];
    for(int i=0;i<urea_signal_json.size();i++)
    {
        urea_signal.push_back(urea_signal_json[i].asDouble());
    }

    return true;
};

bool spectrum_filter_1d::save_urea_signal(std::string fname)
{
    Json::Value root;
    root["urea_signal_start"]=urea_signal_start;
    root["urea_signal_end"]=urea_signal_end;

    Json::Value urea_signal_json;
    for(int i=0;i<urea_signal.size();i++)
    {
        urea_signal_json.append(urea_signal[i]);
    }
    root["urea_signal"]=urea_signal_json;

    std::ofstream fout(fname);
    fout<<root;
    fout.close();

    return true;
};

int main(int argc, char **argv)
{
    std::cout<<"Deep picker package: remove urea signal from 1D NMR spectrum."<<std::endl;
    std::cout << "DEEP Picker package Version " << deep_picker_VERSION_MAJOR << "." << deep_picker_VERSION_MINOR << std::endl;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-f");
    args2.push_back("arguments_urea.txt");
    args3.push_back("input file name for arguments");

    args.push_back("-v");
    args2.push_back("1");
    args3.push_back("verbose level: 0 (minimal) 1 (normal)");

    args.push_back("-in");
    args2.push_back("test.ft1");
    args3.push_back("input file name for 1D FTIR spectrum");

    args.push_back("-out");
    args2.push_back("test_urea.ft1");
    args3.push_back("output file names for baseline corrected spectra in pipe format.");

    args.push_back("-center");
    args2.push_back("5.711");
    args3.push_back("center of the urea signal to be removed (ppm)");

    args.push_back("-width");
    args2.push_back("0.06");
    args3.push_back("width of the urea signal to be removed (ppm)");

    args.push_back("-opt");
    args2.push_back("yes");
    args3.push_back("optimize the urea signal center and width, using a Voigt profile");

    args.push_back("-start");
    args2.push_back("5.90");
    args3.push_back("starting of the urea signal fitting region (ppm)");

    args.push_back("-end");
    args2.push_back("5.50");
    args3.push_back("end of the urea signal fitting region (ppm)");

    args.push_back("-urea-in");
    args2.push_back("urea.json");
    args3.push_back("Input file names for urea signal in json format. None means no input (run estimation)");

    args.push_back("-urea-out");
    args2.push_back("urea.json");
    args3.push_back("output file names for urea signal in json format.");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();



    if (cmdline.query("-h") != "yes")
    {
        std::string infname=cmdline.query("-in");
        std::string outfname=cmdline.query("-out");

        double center=std::stod(cmdline.query("-center"));
        double width=std::stod(cmdline.query("-width"));
        double start=std::stod(cmdline.query("-start"));
        double end=std::stod(cmdline.query("-end"));

        class spectrum_filter_1d x;
        x.read_spectrum(infname);

        std::string urea_in = cmdline.query("-urea-in");
        std::string urea_out = cmdline.query("-urea-out");

        /**
         * If user provide urea signal, use it. Otherwise, estimate it.
        */
        bool b_read = false;
        if (urea_in!="none" && urea_in!="None" && urea_in!="NONE")
        {
            b_read = x.read_urea_signal(urea_in);
        }
        
        if (!b_read)
        {
            x.init_signal(center,width,start,end);    
            bool b_opt = cmdline.query("-opt")[0]=='y' || cmdline.query("-opt")[0]=='Y';
            x.filter(b_opt);
            std::cout<<"Urea signal is estimated."<<std::endl;
        }
        x.apply_filter();

        /**
         * Save the urea signal if user provide the output file name.
        */
        if (urea_out!="none" && urea_out!="None" && urea_out!="NONE")
        {
            x.save_urea_signal(urea_out);
        }

        /**
         * Save the filtered spectrum if user provide the output file name.
        */
        if(outfname!="none" && outfname!="None" && outfname!="NONE")
        {
            x.save_filtered_spectrum(outfname);
        }
    }

    return 0;
}