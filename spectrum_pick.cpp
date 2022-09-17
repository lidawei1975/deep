//#include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <valarray>
#include <string>
#include <vector>
// #include "omp.h"

#include <Eigen/Dense>
#include <Eigen/Cholesky>


#ifndef M_PI
#define M_PI 3.14159265358979324
#endif

#include "commandline.h"
#include "dnn_picker.h"
#include "spectrum_pick.h"

extern "C"  
{
    double voigt(double x, double sigma, double gamma);
};

//class spectrum_picking
spectrum_pick::spectrum_pick()
{
    infname="input.ft2";
    user_scale=5.5;  //used defined noise level scale factor
    user_scale2=3.0; 
    model_selection=1; //default peak width 6-20 (model 1) or 4-12 (model 2)
};

spectrum_pick::~spectrum_pick()
{
};

bool spectrum_pick::voigt_convolution(double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay,
std::vector<double> &kernel,int &i0,int &i1, int &j0, int &j1)
{
    float wx=(1.0692*gammax+sqrt(0.8664*gammax*gammax+5.5452*sigmax*sigmax))*1.5f;
    float wy=(1.0692*gammay+sqrt(0.8664*gammay*gammay+5.5452*sigmay*sigmay))*1.5f;

    i0=std::max(0,int(x-wx+0.5));
    i1=std::min(xdim,int(x+wx+0.5));
    j0=std::max(0,int(y-wy+0.5));
    j1=std::min(ydim,int(y+wy+0.5));

    kernel.clear();
    kernel.resize((i1-i0)*(j1-j0));

    double z11=voigt(0,sigmax, gammax );
    double z22=voigt(0,sigmay, gammay );
    
    
    for (int i =i0; i < i1; i++)
    {
        for (int j = j0; j < j1; j++)
        {
            double z1=voigt ( i-x, sigmax, gammax )/z11;
            double z2=voigt ( j-y, sigmay, gammay )/z22;
            kernel.at((i-i0)*(j1-j0)+j-j0)=a*z1*z2;
        }
    }
    return true;
};

bool spectrum_pick::simple_peak_picking(bool b_negative)
{
    for(int j=1;j<ydim-1;j++)
    {
        for(int i=1;i<xdim-1;i++)
        {
            if (spect[j * xdim + i] > noise_level * user_scale 
            && spect[j * xdim + i] > spect[j * xdim + i + 1] 
            && spect[j * xdim + i] > spect[j * xdim + i - 1] 
            && spect[j * xdim + i] > spect[j * xdim + xdim + i] 
            && spect[j * xdim + i] > spect[j * xdim - xdim + i]
            && spect[j * xdim + i] > noise_level*user_scale)
            {
                p1.push_back(i);
                p2.push_back(j);
                p_intensity.push_back(spect[j * xdim + i]);
            }
        }
    }

    if(b_negative)
    {
         for(int j=1;j<ydim-1;j++)
        {
            for(int i=1;i<xdim-1;i++)
            {
                if (-spect[j * xdim + i] > noise_level * user_scale 
                && -spect[j * xdim + i] > -spect[j * xdim + i + 1] 
                && -spect[j * xdim + i] > -spect[j * xdim + i - 1] 
                && -spect[j * xdim + i] > -spect[j * xdim + xdim + i] 
                && -spect[j * xdim + i] > -spect[j * xdim - xdim + i]
                && -spect[j * xdim + i] > -noise_level*user_scale)
                {
                    p1.push_back(i);
                    p2.push_back(j);
                    p_intensity.push_back(spect[j * xdim + i]);
                }
            }
        }   
    }

    //fake values.
    sigmax.resize(p1.size(),3.0);
    sigmay.resize(p1.size(),3.0);
    gammax.resize(p1.size(),0.00001);
    gammay.resize(p1.size(),0.00001);
    p_confidencex.resize(p1.size(),1.0);
    p_confidencey.resize(p1.size(),1.0);


    get_ppm_from_point();

    return true;
}

bool spectrum_pick::ann_peak_picking(int flag,int expand, int flag_t1_noise, bool b_negative)  //default flag is 0, default expand is 0, default b_negative=false
{
    std::vector<int> p_type; //no used at this time

    class peak2d p(flag); //flag==0:  run special case, 1: not run. 2: inertia based method
    p.init_ann(model_selection); //read in ann parameters. 1: protein para set, 2: meta para set.
    class peak2d pp(flag); //flag==0:  run special case, 1: not run. 2: inertia based method
    pp.init_ann(model_selection); //read in ann parameters. 1: protein para set, 2: meta para set.
   

    std::vector<float> sp;
    if(expand==1) //need revision to address b_negative mode!!!!
    {
        //spect[j * xdim + i]  i: xdim, j: ydim; row by row format
        std::vector<double> final_data;
        ldw_math_spectrum_2d::spline_expand(xdim,ydim,spect,final_data);
        //At this time, final data is (2*xdim-1)*(2*ydim-1), column by column format

        sp.resize(final_data.size(),0.0f);
        for(int m=0;m<final_data.size();m++)
        {
            sp[m]=final_data[m];
        }

        p.init_spectrum(xdim*2-1,ydim*2-1,noise_level,user_scale,user_scale2,sp,0);
        p.predict();
        //get p1,p2,p_intensity,sigma,gamma from ANN here
        p.extract_result(p1,p2,p_intensity,sigmax,sigmay,gammax,gammay,p_type,p_confidencex,p_confidencey);
        for(int m=0;m<p1.size();m++)
        {
            p1[m]*=0.5;
            p2[m]*=0.5;   
            sigmax[m]*=0.5;
            sigmay[m]*=0.5;
            gammax[m]*=0.5;
            gammay[m]*=0.5;
        }
    }
    else
    {
        sp.assign(spect,spect+xdim*ydim);
        for(int i=0;i<sp.size();i++)
        {
            if(sp[i]<0.0)
                sp[i]=0.0;
        }
        p.init_spectrum(xdim,ydim,noise_level,user_scale,user_scale2,sp,1);
        p.predict();
        //get p1,p2,p_intensity,sigma,gamma from ANN here

        p.extract_result(p1,p2,p_intensity,sigmax,sigmay,gammax,gammay,p_type,p_confidencex,p_confidencey);

        if(b_negative==true)
        {
           
            sp.clear();
            sp.assign(spect,spect+xdim*ydim);
            for(int i=0;i<sp.size();i++)
            {
                if(sp[i]<0.0)
                    sp[i]=-sp[i];
                else
                    sp[i]=0.0;
            }
            pp.init_spectrum(xdim,ydim,noise_level,user_scale,user_scale2,sp,1);
            pp.predict();
            //get p1,p2,p_intensity,sigma,gamma from ANN here
            int n=p_intensity.size();
            pp.extract_result(p1,p2,p_intensity,sigmax,sigmay,gammax,gammay,p_type,p_confidencex,p_confidencey);
            for(int m=n;m<p_intensity.size();m++)
            {
                p_intensity[m]=-p_intensity[m];    
            }
        }
    }

    //remove peaks that are below noise*user_scale
    for (int i = p1.size() - 1; i >= 0; i--)
    {
        int pp1 = int(p1[i] + 0.5);
        int pp2 = int(p2[i] + 0.5);
        if (fabs(p_intensity[i]) <= noise_level * user_scale )
        {
            p_intensity.erase(p_intensity.begin() + i);
            p_confidencex.erase(p_confidencex.begin() + i);
            p_confidencey.erase(p_confidencey.begin() + i);
            
            sigmax.erase(sigmax.begin() + i);
            sigmay.erase(sigmay.begin() + i);
            gammax.erase(gammax.begin() + i);
            gammay.erase(gammay.begin() + i);
            p1.erase(p1.begin() + i);
            p2.erase(p2.begin() + i);
        }
    }
    std::cout<<"Picked "<<p1.size()<<" peaks."<<std::endl;


    //remove peaks using column by column noise estimation
    if(flag_t1_noise==1)
    {
        for (int i = p1.size() - 1; i >= 0; i--)
        {
            int pp1 = std::min(std::max(0,int(p1[i] + 0.5)),xdim-1);
            int pp2 = int(p2[i] + 0.5);
            if (fabs(p_intensity[i]) <= noise_level_columns[pp1] * user_scale )
            {
                p_intensity.erase(p_intensity.begin() + i);
                p_confidencex.erase(p_confidencex.begin() + i);
                p_confidencey.erase(p_confidencey.begin() + i);
                
                sigmax.erase(sigmax.begin() + i);
                sigmay.erase(sigmay.begin() + i);
                gammax.erase(gammax.begin() + i);
                gammay.erase(gammay.begin() + i);
                p1.erase(p1.begin() + i);
                p2.erase(p2.begin() + i);
            }
        }
        std::cout<<"Number of peak is "<<p1.size()<<" after T1 noise removal."<<std::endl;  
    }


    get_ppm_from_point();
    
    peak_index.clear();
    for(int i=0;i<p1.size();i++)
    {
        peak_index.push_back(i);
    }

    //important, set up median peak width for fitting step!!!!
    std::vector<double> sx,sy;
    sx.clear();
    sy.clear();
    for(unsigned int i=0;i<p1.size();i++)
    {
        if(sigmax[i]>0.0 && sigmay[i]>0.0)
        {
            double s1=sigmax[i];
            double g1=gammax[i];
            double fwhh1=1.0692*g1+sqrt(0.8664*g1*g1+5.5452*s1*s1);

            double s2=sigmay[i];
            double g2=gammay[i];
            double fwhh2=1.0692*g2+sqrt(0.8664*g2*g2+5.5452*s2*s2);

            sx.push_back(fwhh1);
            sy.push_back(fwhh2);
        }
    }
    median_width_x=ldw_math_spectrum_2d::calcualte_median(sx);
    median_width_y=ldw_math_spectrum_2d::calcualte_median(sy);

    std::cout<<"Median peak width is estimated to be "<<median_width_x<<" "<<median_width_y<< " from ann picking."<<std::endl;

    return true;
};

bool spectrum_pick::clear_memory()
{
    if(spect!=NULL)
    {
        delete [] spect;
    }
    return true;
}

bool spectrum_pick::linear_regression()
{
    std::vector<int> peak_map(xdim*ydim,-1);
   
    int npeak=p1.size();
    for(int i=0;i<p1.size();i++)
    {
        int xx=(int)(p1.at(i)+0.5);
        int yy=(int)(p2.at(i)+0.5);
        peak_map[xx*ydim+yy]=i;
    }

    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(npeak,npeak);
    Eigen::VectorXf b = Eigen::VectorXf::Zero(npeak);

    int i0,i1,j0,j1;
    std::vector<double> analytical_spectra;
    for (unsigned int i = 0; i < p1.size(); i++)
    {
        voigt_convolution(p_intensity[i], p1[i], p2[i], sigmax[i], sigmay[i], gammax[i], gammay[i], analytical_spectra, i0, i1, j0, j1);
        for (int ii = i0; ii < i1; ii++)
        {
            for (int jj = j0; jj < j1; jj++)
            {
                if(peak_map[ii*ydim+jj]>=0)
                {
                    A(i,peak_map[ii*ydim+jj])=analytical_spectra[(ii - i0) * (j1 - j0) + jj - j0];
                }
            }
        }
    }

    for(int i=0;i<xdim;i++)
    {
        for(int j=0;j<ydim;j++)
        {
            if(peak_map[i*ydim+j]>=0)
            {
                b(peak_map[i*ydim+j])=spect[i+j*xdim];
            }   
        }
    }

    Eigen::VectorXf c=A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    for(int i=0;i<npeak;i++)
    {
        std::cout<<i<<" "<<p1[i]<<" "<<p2[i]<<" "<<c(i)<<"  "<<p_intensity[i]<<" "<<p_intensity[i]*c(i)<<std::endl;
        if(c(i)<0)
        {
            p_intensity[i]=0.0;    
        }
        else
        {
            p_intensity[i]*=c(i);
        }
    }

    


    return true;
}

bool spectrum_pick::print_peaks_picking(std::string outfname)
{
    if (user_comments.size() == 0) //from picking, not reading
    {
        user_comments.resize(p1.size(), "peak");
    }

    std::vector<int> ndx;
    ldw_math_spectrum_2d::sortArr(p_intensity,ndx);
    std::reverse(ndx.begin(),ndx.end());

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
            //.tab file for nmrDraw
            FILE *fp = fopen(file_names[m].c_str(), "w");
            fprintf(fp,"DATA  X_AXIS 1H           1 %5d %8.3fppm %8.3fppm\n",xdim,begin1,stop1);
            fprintf(fp,"DATA  Y_AXIS 15N          1 %5d %8.3fppm %8.3fppm\n",ydim,begin2,stop2);
            fprintf(fp,"VARS   INDEX X_AXIS Y_AXIS X_PPM Y_PPM XW YW  X1 X3 Y1 Y3 HEIGHT ASS CONFIDENCE POINTER\n");
            fprintf(fp,"FORMAT %%5d %%9.3f %%9.3f %%10.6f %%10.6f %%7.3f %%7.3f %%4d %%4d %%4d %%4d %%+e %%s %%4.2f %%3s\n");
            for (unsigned int ii = 0; ii < ndx.size(); ii++)
            {
                int i=ndx[ii];
                double s1,s2;
                s1=1.0692*gammax[i]+sqrt(0.8664*gammax[i]*gammax[i]+5.5452*sigmax[i]*sigmax[i]);
                s2=1.0692*gammay[i]+sqrt(0.8664*gammay[i]*gammay[i]+5.5452*sigmay[i]*sigmay[i]);
                fprintf(fp,"%5d %9.3f %9.3f %10.6f %10.6f %7.3f %7.3f %4d %4d %4d %4d %+e %s %4.2f <--\n",ii+1,p1[i]+1,p2[i]+1,p1_ppm[i], p2_ppm[i],s1,s2,
                            int(p1[i]-3),int(p1[i]+3),int(p2[i]-3),int(p2[i]+3),p_intensity[i],user_comments[i].c_str(),std::min(p_confidencex[i],p_confidencey[i]));        
            }
            fclose(fp);
        }

        else if(std::equal(slist.rbegin(), slist.rend(), file_names[m].rbegin()))
        {
            FILE *fp=fopen(file_names[m].c_str(),"w");
            fprintf(fp,"Assignment         w1         w2   Height Confidence\n");
            for (unsigned int ii = 0; ii < ndx.size(); ii++)
            {
                int i=ndx[ii];
                fprintf(fp,"?-? %10.6f %10.6f %+e %4.2f\n",p2_ppm[i], p1_ppm[i],p_intensity[i],std::min(p_confidencex[i],p_confidencey[i]));        
            }
            fclose(fp);
        }

        else if(std::equal(sjson.rbegin(), sjson.rend(), file_names[m].rbegin()))
        {
            FILE * fp=fopen(file_names[m].c_str(),"w");
            fprintf(fp,"{\"picked_peaks\":[");
            for (unsigned int ii = 0; ii < ndx.size()-1; ii++)
            {
                int i=ndx[ii];
                fprintf(fp,"{\"cs_x\": %f,\"cs_y\": %f, \"type\": 1, \"index\": %f, \"sigmax\": %f,  \"sigmay\": %f,\"gammax\": %f,  \"gammay\": %f},",p1_ppm[i],p2_ppm[i],p_intensity[i],sigmax[i],sigmay[i],gammax[i],gammay[i]);
            }
            int n=ndx[ndx.size()-1];
            fprintf(fp,"{\"cs_x\": %f,\"cs_y\": %f, \"type\": 1, \"index\": %f, \"sigmax\": %f,  \"sigmay\": %f, \"gammax\": %f,  \"gammay\": %f}",p1_ppm[n],p2_ppm[n],p_intensity[n],sigmax[n],sigmay[n],gammax[n],gammay[n]);

            fprintf(fp,"]}");
            fclose(fp);
        }
    }

    return 1;
};

