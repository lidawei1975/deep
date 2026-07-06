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


#include "cubic_spline.h"
#include "kiss_fft.h"
#include "json/json.h"
#include "ldw_math.h"
#include "commandline.h"
#include "fid_1d.h"
#include "spectrum_baseline_1d.h"

// #define LDW_DEBUG 
#define LDW_DEBUG_2

#ifdef WEBASSEMBLY
/**
* Bind to emscripten with exposed class methods.
*/
#include <emscripten/bind.h>
using namespace emscripten;
#endif


double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}


spectrum_baseline_1d::spectrum_baseline_1d()
{
    max_calculation_length=4096; //default value
    n_flexible_end=256;
    flexible_series_min=1e-6;
};

spectrum_baseline_1d::spectrum_baseline_1d(int n1,int n2)
{
    max_calculation_length=n1;
    n_flexible_end=n2;
    flexible_series_min=1e-6;
};

spectrum_baseline_1d::~spectrum_baseline_1d()
{

};

bool spectrum_baseline_1d::work(double a0_,double b0_,int n_water,int flag,std::string fname_baseline)
{
    //set parameters
    a0=a0_;
    b0=b0_;

    /**
     * segment type, 1: normal, 0: water,excluded
    */
    segment_type.clear();
    segment_type.resize(ndata_frq,1);

    /**
     * Even without water region, exclusion of the center of the spectrum has no harm
    */
    for(int n=ndata_frq/2-n_water;n<ndata_frq/2+n_water;n++)
    {
        segment_type[n]=0; //water
    }

    std::vector<double> spect_normalized;

    //find max of spe
    double max_spe = 0.0;
    int max_pos = -1;
   
    double sum=0.0;
    for (int k = 0; k < ndata_frq; k+=1)
    {
        if (fabs(spectrum_real[k]) > max_spe)
        {
            max_spe = fabs(spectrum_real[k]);
            max_pos = k;
        }
        sum+=spectrum_real[k];
    }
    sum/=ndata_frq;

    spect_normalized.clear();
    for (int i = 0; i < ndata_frq; i+=1)
    {
        spect_normalized.push_back(spectrum_real[i]/max_spe);
    }


    /**
     * get mean of spect_normalized and use it as initial baseline
    */
    double mean_t=0.0;
    for(int i=0;i<spect_normalized.size();i++)
    {
        mean_t+=spect_normalized[i];
    }
    mean_t/=spect_normalized.size();
    
    for(int i=0;i<spect_normalized.size();i++)
    {
        baseline.push_back(mean_t);
    }

    calculate_baseline(segment_type,spect_normalized,baseline,noise_level/max_spe,flag);

    /**
     * restore baseline to original scale
    */
    for(int i=0;i<ndata_frq;i++)
    {
        baseline[i]*=max_spe;
    }

    /**
     * if fname_baseline is none or no, skip saving baseline
     * else if fname_baseline's extension is txt, save baseline in text format
     * else save in binary format
    */
    if(fname_baseline=="none" || fname_baseline=="no")
    {
        if(n_verbose>0) std::cout<<"Baseline is not saved."<<std::endl;
    }
    else
    {
        //check file extension
        std::string ext=fname_baseline.substr(fname_baseline.find_last_of(".") + 1);
        if(ext=="txt") //text format
        {
            std::ofstream fout(fname_baseline);
            for(int i=0;i<baseline.size();i++)
            {
                fout<<baseline[i]<<std::endl;
            }
            fout.close();
        }
        else //binary format
        {
            //convert to float first
            std::vector<float> baseline_float;
            for(int i=0;i<baseline.size();i++)
            {
                baseline_float.push_back(baseline[i]);
            }

            std::ofstream fout(fname_baseline,std::ios::binary);
            for(int i=0;i<baseline_float.size();i++)
            {
                fout.write((char*)&baseline_float[i],sizeof(float));
            }
            fout.close();
        }
        
    }

 
    /**
     * subtract baseline from spect
    */
    for(int i=0;i<ndata_frq;i++)
    {
        spectrum_real[i]-=baseline[i];
    }

    return true;
}


/**
 * main working function to calculate baseline from signal
 * flag=0: normal matrix based algorithm
 * flag=1: sparse matrix based algorithm
*/

bool spectrum_baseline_1d::calculate_baseline(const std::vector<int> &water_region_flag,const std::vector<double> &signal,std::vector<float> &baseline,double noise, int flag)
{
    int n=signal.size();
    int n_origal=n;
    int k=1;

    /**
     * To speed up calculation, we use stride format.
    */
    while(n>=max_calculation_length)
    {
        n=n/2;
        k*=2;
    }

    /**
     * To fixed two ends's shape, we use a segment setup:
     * first 1/32 and last 1/32 points are flexible, with a fixed a value of 1e8, instead of user defined a value (usually > 1e10)
     */
    int n_end = n/32;

    std::vector<double> signal_stride(n);
    std::vector<double> baseline_stride(n);
    std::vector<double> index(n);
    std::vector<int> water_region_flag_stride(n);
    int k_2=k/2;
    for(int i=0;i<n;i++)
    {
        index[i]=i*k+k_2;
        signal_stride[i]=signal[i*k+k_2];
        baseline_stride[i]=baseline[i*k+k_2]; //float to double conversion
        water_region_flag_stride[i]=water_region_flag[i*k+k_2];
    }

    /**
     * calculate a and b, modified because stride format is used
     * a and b are also normalized by noise
    */
    double a=a0/noise/std::pow(k,4);
    double b=b0/noise; 

    if(n_verbose>0) std::cout<<"size="<<n<<" stride="<<k<<" a="<<a<<" b="<<b<<std::endl;


    /**
     * initial solution. Notice solution is in stride format and will be converted to original format later
     * solution = baseline_stride but in Eigen::VectorXd format
    */
    Eigen::VectorXd solution_left_end=Eigen::VectorXd::Zero(n_end);
    Eigen::VectorXd solution_right_end=Eigen::VectorXd::Zero(n_end);
    Eigen::VectorXd solution=Eigen::VectorXd::Zero(n-n_end*2);
    for(int i=0;i<n_end;i++)
    {
        solution_left_end(i)=baseline_stride[i];
    }
    for(int i=n_end;i<n-n_end;i++)
    {
        solution(i-n_end)=baseline_stride[i];
    }
    for(int i=n-n_end;i<n;i++)
    {
        solution_right_end(i-n+n_end)=baseline_stride[i];
    }

    // generate_geometric_series(1.0,1.0,n_flexible_end);

    double t1=get_wall_time();
    if(flag==0)
    {
        for(int i=0;i<10;i++)
        {   
            /**
             * Left end, center, right end
             */
            std::vector<int> water_region_flag_end(n_end,1); //no water signal in the left or right end
            std::vector<int> water_region_flag_center=std::vector<int>(water_region_flag_stride.begin()+n_end,water_region_flag_stride.end()-n_end);
            std::vector<double> signal_left_end=std::vector<double>(signal_stride.begin(),signal_stride.begin()+n_end);
            std::vector<double> signal_center=std::vector<double>(signal_stride.begin()+n_end,signal_stride.end()-n_end);
            std::vector<double> signal_right_end=std::vector<double>(signal_stride.end()-n_end,signal_stride.end());
            std::vector<double> baseline_left_end=std::vector<double>(baseline_stride.begin(),baseline_stride.begin()+n_end);
            std::vector<double> baseline_right_end=std::vector<double>(baseline_stride.end()-n_end,baseline_stride.end());
            calculate_baseline_solver_1(1e8,b,water_region_flag_end,signal_left_end,solution_left_end);
            calculate_baseline_solver_1(a,b,water_region_flag_center,signal_center,solution);
            calculate_baseline_solver_1(1e8,b,water_region_flag_end,signal_right_end,solution_right_end);
            if(n_verbose>0) std::cout<<"iteration "<<i<<" done"<<std::endl;
        }
    }
    else if(flag==1)
    {
        for(int i=0;i<10;i++)
        {
            calculate_baseline_solver_2(a,b,water_region_flag_stride,signal_stride,solution);
            if(n_verbose>0) std::cout<<"iteration "<<i<<" done"<<std::endl;
        }
    }
    else
    {
        std::cout<<"flag is not supported"<<std::endl;
        return false;
    }
    double t2=get_wall_time();
    if(n_verbose>0)  std::cout<<"Total run time is "<<t2-t1<<" second."<<std::endl;


    /**
     * Copy solution_left_end, solution, solution_right_end to baseline_stride
     */
    for(int i=0;i<n_end;i++)
    {
        baseline_stride[i]=solution_left_end(i);
    }
    for(int i=n_end;i<n-n_end;i++)
    {
        baseline_stride[i]=solution(i-n_end);
    }
    for(int i=n-n_end;i<n;i++)
    {
        baseline_stride[i]=solution_right_end(i-n+n_end);;
    }


    class cublic_spline cs;

    /**
     * initialize cubic spline, notice x index is k-1,2*k-1,3*k-1,...,n*k-1
    */
    cs.calculate_coefficients(baseline_stride); 
  
    /**
     * calculate baseline at each point
    */
    baseline.clear();
    for (int m = 0; m < n_origal; m++)
    {
        double m_double=m;

        /**
         * m_double map to baseline parameters
         * k-1 ==> 0
         * 2*k-1 ==> 1
         * 3*k-1 ==> 2
        */

        m_double=m_double/k-1.0+1.0/k;
        baseline.push_back(cs.calculate_value(m_double)); //double to float conversion
    }

    return true;
};

/**
 * calculate baseline using matrix based algorithm
 * s is the segment type, 1: normal, 0: water,excluded
 * y is the signal
 * solution is the baseline
*/
bool spectrum_baseline_1d::calculate_baseline_solver_1(double aa,double b,std::vector<int> &s,std::vector<double> &y,Eigen::VectorXd &solution)
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

    if(solution(0)>y[0])
    {
        d(0,0) = 2*aa +2*b;
    }
    else
    {
        d(0,0) = 2*aa; 
    }
    d(0,1) = -4*aa;
    d(0,2) = 2*aa;

    d(1,0) = -4*aa;
    if(solution(1)>y[1])
    {
        d(1,1) = 10*aa+2*b;
    }
    else
    {
        d(1,1) = 10*aa;
    }
    d(1,2)=-8*aa;
    d(1,3)=2*aa;


    //2 to xdim-2
    for(int i=2;i<xdim-2;i++)
    {
        d(i, i - 2) = 2 * aa;
        d(i, i - 1) = -8 * aa;
        if (s[i] == 1) // normal data point
        {
            if (solution(i) > y[i])
            {
                d(i, i) = 12 * aa + 2 * b;
            }
            else
            {
                d(i, i) = 12 * aa;
            }
        }
        else // water data region. Allow baseline to go above signal
        {
            d(i, i) = 12 * aa;
        }
        d(i, i + 1) = -8 * aa;
        d(i, i + 2) = 2 * aa;
    }    

    d(xdim-2,xdim-4)=2*aa;
    d(xdim-2,xdim-3)=-8*aa;
    if(solution(xdim-2)>y[xdim-2])
    {
        d(xdim-2,xdim-2) = 10*aa+2*b;
    }
    else
    {
        d(xdim-2,xdim-2) = 10*aa;
    }
    d(xdim-2,xdim-1) = -4*aa;
    
    d(xdim-1,xdim-3) = 2*aa;
    d(xdim-1,xdim-2) = -4*aa;
    if(solution(xdim-1)>y[xdim-1])
    {
        d(xdim-1,xdim-1) = 2*aa+2*b;

    }
    else
    {
        d(xdim-1,xdim-1) = 2*aa;
    }
    
  

    for(int i=0;i<xdim;i++)
    {
        if(s[i]==1) //normal data point
        {
            if(solution(i)>y[i])
            {
                m(i)=2*b*y[i]+1.0;
            }
            else
            {
                m(i)=1.0;
            }
        }
        else //water region
        {
            m(i)=1.0;
        }
    }

    
    // fill b
    // solve Ax = b
    solution=d.ldlt().solve(m);
    
    return true;
}


/**
 * calculate baseline using sparse matrix based algorithm
 * s is the segment type, 1: normal, 0: water,excluded
 * y is the signal
 * solution is the baseline
*/
bool spectrum_baseline_1d::calculate_baseline_solver_2(double a,double b,std::vector<int> &s,std::vector<double> &y,Eigen::VectorXd &solution)
{
    int xdim=y.size();
    Eigen::SparseMatrix<double,Eigen::RowMajor> d(xdim,xdim);
    Eigen::VectorXd m(xdim);

    d.reserve(Eigen::VectorXf::Constant(xdim,5));

    //0,0
    if(solution(0)>y[0])
    {
        d.insert(0,0) = 2*a + 2*b;
    }
    else
    {
        d.insert(0,0) = 2*a;
    }
    d.insert(0,1) = -4*a;
    d.insert(0,2) = 2*a;

    //1,1
    d.insert(1,0) = -4*a;
    if(solution(1)>y[1])
    {
        d.insert(1,1) = 10*a+2*b;
    }
    else
    {
        d.insert(1,1) = 10*a;
    }
    d.insert(1,2)=-8*a;
    d.insert(1,3)=2*a;

    //2 to xdim-2
    for(int i=2;i<xdim-2;i++)
    {
        d.insert(i,i-2) = 2*a;
        d.insert(i,i-1) = -8*a;  
        if(s[i]==1) //normal data point
        {
            if(solution(i)>y[i])
            {
                d.insert(i,i) = 12*a+2*b;
            }
            else
            {
                d.insert(i,i) = 12*a;
            }
        }
        else
        {
            d.insert(i,i) = 12*a;
        }
        d.insert(i,i+1) = -8*a;
        d.insert(i,i+2) = 2*a;
    }    

    //second last row, xdim-1
    d.insert(xdim-2,xdim-4)=2*a;
    d.insert(xdim-2,xdim-3)=-8*a;
    if(solution(xdim-2)>y[xdim-2])
    {
        d.insert(xdim-2,xdim-2) = 10*a+2*b;
    }
    else
    {
        d.insert(xdim-2,xdim-2) = 10*a;
    }
    d.insert(xdim-2,xdim-1) = -4*a;
    
    //last row, xdim
    d.insert(xdim-1,xdim-3) = 2*a;
    d.insert(xdim-1,xdim-2) = -4*a;
    if(solution(xdim-1)>y[xdim-1])
    {
        d.insert(xdim-1,xdim-1) = 2*a+2*b;
    }
    else
    {
        d.insert(xdim-1,xdim-1) = 2*a;
    }
    
  

    for(int i=0;i<xdim;i++)
    {
        if(s[i]==1) //normal data point
        {
            if(solution(i)>y[i])
            {
                m(i)=2*b*y[i]+1.0;
            }
            else
            {
                m(i)=1.0;
            }
        }
        else //water region
        {
            m(i)=1.0;
        }
    }

    
    // fill b
    // solve Ax = b
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    
    solver.compute(d);
    if(solver.info()!=Eigen::Success) {
        return false;
    }
    solution = solver.solve(m);
    if(solver.info()!=Eigen::Success) {
        return false;
    }
    return true;
}


bool spectrum_baseline_1d::read_baseline(std::string infname)
{
    /**
     * check file extension. If it is txt, read baseline in text format
     * else read in binary format
    */
    std::string ext=infname.substr(infname.find_last_of(".") + 1);
    if(ext=="txt") //text format
    {
        std::ifstream fin(infname);
        if(!fin)
        {
            std::cout<<"Error: cannot open file "<<infname<<std::endl;
            return false;
        }
        std::string line;
        while(std::getline(fin,line))
        {
            baseline.push_back(std::stof(line));
        }
        fin.close();
    }
    else //binary format
    {
        std::ifstream fin(infname,std::ios::binary);
        if(!fin)
        {
            std::cout<<"Error: cannot open file "<<infname<<std::endl;
            return false;
        }
        float temp;
        while(fin.read((char*)&temp,sizeof(float)))
        {
            baseline.push_back(temp);
        }   
        fin.close();
    }

    /**
     * make sure baseline has the same size as spect
     * if too short, append zeros to baseline
     * if too long, truncate baseline. print warning message
    */
    if(baseline.size()<ndata_frq)
    {
        std::cout<<"Warning: baseline size is smaller than spectrum size. Append zeros to baseline."<<std::endl;
        for(int i=baseline.size();i<ndata_frq;i++)
        {
            baseline.push_back(0.0);
        }
    }
    else if(baseline.size()>ndata_frq)
    {
        std::cout<<"Warning: baseline size is larger than spectrum size. Truncate baseline."<<std::endl;
        baseline.resize(ndata_frq);
    }

    /**
     * subtract baseline from spect
    */
    for(int i=0;i<ndata_frq;i++)
    {
        spectrum_real[i]-=baseline[i];
    }

    return true;
}

/**
 * help function to generate a geometric series to change the a scale at the two ends of the spectrum
*/
bool spectrum_baseline_1d::generate_geometric_series(double first_term,double last_term,int num_points)
{
    // double first_term = 1e-6;
    // double last_term = 1.0;
    // int num_points = 256;

    double a = std::log10(first_term);
    double b = std::log10(last_term);


    // Clear then reserve space for efficiency
    flexible_series.clear();
    flexible_series.reserve(num_points); 

    if (num_points == 1) {
        flexible_series.push_back(last_term); // Or std::pow(10.0, b)
    } else {
        // Calculate the linear step size for the exponents
        double step = (b - a) / (num_points - 1);

        for (int i = 0; i < num_points; ++i) {
            // Calculate the current exponent
            double current_exponent = a + i * step;
            
            // Calculate the value (10^exponent) and add to the vector
            flexible_series.push_back(std::pow(10.0, current_exponent));
        }
    }
    return true;
}

#ifdef WEBASSEMBLY


/**
 * Function calls for web assembly to read header and data directly from variables without addtional copy
 */
uintptr_t spectrum_baseline_1d::get_data_of_baseline()
{
    return reinterpret_cast<uintptr_t>(baseline.data());
}

/**
 * Exposed functions
*/
EMSCRIPTEN_BINDINGS(dp_1d_module_baseline) {

    /**
     * Baseclass spectrum_fwhh_1d is exposed in spectrum_pick_1d.cpp
    */

    class_<spectrum_baseline_1d,base<fid_1d>>("spectrum_baseline_1d")
        .constructor()
        .class_property("n_verbose", &shared_data_1d::n_verbose)
        .function("init", &spectrum_baseline_1d::init)
        .function("work", &spectrum_baseline_1d::work)
        .function("get_data_of_baseline", &spectrum_baseline_1d::get_data_of_baseline)
        /**
         * Below are from base class fid_1d
        */
        .function("read_first_spectrum_from_buffer",&spectrum_baseline_1d::read_first_spectrum_from_buffer)  //read ft1 file (from buffer instead of file)
        ;
}
#endif // WEBASSEMBLY