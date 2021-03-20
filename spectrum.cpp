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



#ifndef M_PI
#define M_PI 3.14159265358979324
#endif



#include "commandline.h"
#include "dnn_picker.h"
#include "spectrum.h"

namespace ldw_math
{
    double calcualte_median(std::vector<double> scores)
    {
        size_t size = scores.size();

        if (size == 0)
        {
            return 0; // Undefined, really.
        }
        else
        {
            sort(scores.begin(), scores.end());
            if (size % 2 == 0)
            {
                return (scores[size / 2 - 1] + scores[size / 2]) / 2;
            }
            else
            {
                return scores[size / 2];
            }
        }
    };
};

//class spectrum_picking
spectrum_picking::spectrum_picking()
{
    infname="input.ft2";
    user_scale=5.5;  //used defined noise level scale factor
    user_scale2=3.0; 
    model_selection=1; //default peak width 6-20 (model 1) or 4-12 (model 2)
};

spectrum_picking::~spectrum_picking()
{

};


bool spectrum_picking::zero_negative()
{
    for(int i=0;i<xdim*ydim;i++)
    {
        if(spect[i]<0.0)
            spect[i]=0.0;
    }
    return true;
}


bool spectrum_picking::get_ppm_from_point()
{
    //get ppm
    p1_ppm.clear();
    p2_ppm.clear();

    for (unsigned int i = 0; i < p1.size(); i++)
    {
        double f1 = begin1 + step1 * (p1[i]);  //direct dimension
        double f2 = begin2 + step2 * (p2[i]);  //indirect dimension
        p1_ppm.push_back(f1);
        p2_ppm.push_back(f2);
    }

    return true;
}


bool spectrum_picking::ann_peak_picking()
{
    class peak2d p;
    std::vector<float> kernel;
    p.init_ann(model_selection); //read in ann parameters. 1: protein para set, 2: meta para set.
   
    zero_negative();
    std::vector<float> sp;
    sp.assign(spect,spect+xdim*ydim);

    p.init_spectrum(xdim,ydim,noise_level,user_scale,user_scale2,sp,1);
    std::cout<<"init spectrum done."<<std::endl;

    p.predict();

    //get p1,p2,p_intensity,sigma,gamma from ANN here
    p.extract_result(p1,p2,p_intensity,sigmax,sigmay,gammax,gammay,p_type,p_confidencex,p_confidencey);

    //remove peaks that are below noise*user_scale
    for (int i = p1.size() - 1; i >= 0; i--)
    {
        int pp1 = int(p1[i] + 0.5);
        int pp2 = int(p2[i] + 0.5);
        double pp = spect[pp1 + pp2 * xdim];

        if (p_intensity[i] <= noise_level * user_scale && pp <= noise_level * user_scale)
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

    //we need them to be consistent with normal picking method!
    std::cout<<"Total picked "<<p1.size()<<" peaks."<<std::endl;
    
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
    median_width_x=ldw_math::calcualte_median(sx);
    median_width_y=ldw_math::calcualte_median(sy);

    std::cout<<"Median peak width is estimated to be "<<median_width_x<<" "<<median_width_y<< " from ann picking."<<std::endl;


    return true;
}


bool spectrum_picking::output_picking(std::string outfname)
{
    if(user_comments.size()==0) //from picking, not reading
    {
        user_comments.resize(p1.size(),"peak");
    }

    std::string outfname2; //for Sparky format
    std::size_t found = outfname.find_last_of(".");
    if(found!=std::string::npos)
    {
        outfname2=outfname.substr(0,found)+".list";
    }
    else
    {
        outfname2=outfname+".list";
    }

    //.tab file for nmrDraw
    FILE *fp = fopen(outfname.c_str(), "w");
    fprintf(fp,"VARS   INDEX X_AXIS Y_AXIS X_PPM Y_PPM XW YW  X1 X3 Y1 Y3 HEIGHT ASS CONFIDENCE\n");
    fprintf(fp,"FORMAT %%5d %%9.3f %%9.3f %%8.3f %%8.3f %%7.3f %%7.3f %%4d %%4d %%4d %%4d %%+e %%s %%4.2f\n");
    for (unsigned int i = 0; i < p1.size(); i++)
    {
        double s1,s2;
        s1=1.0692*gammax[i]+sqrt(0.8664*gammax[i]*gammax[i]+5.5452*sigmax[i]*sigmax[i]);
        s2=1.0692*gammay[i]+sqrt(0.8664*gammay[i]*gammay[i]+5.5452*sigmay[i]*sigmay[i]);
        fprintf(fp,"%5d %9.3f %9.3f %8.3f %8.3f %7.3f %7.3f %4d %4d %4d %4d %+e %s %4.2f\n",i+1,p1[i]+1,p2[i]+1,p1_ppm[i], p2_ppm[i],s1,s2,
                    int(p1[i]-3),int(p1[i]+3),int(p2[i]-3),int(p2[i]+3),p_intensity[i],user_comments[i].c_str(),std::min(p_confidencex[i],p_confidencey[i]));        
    }
    fclose(fp);

    fp=fopen(outfname2.c_str(),"w");
    fprintf(fp,"Assignment         w1         w2   Height Confidence\n");
    for (unsigned int i = 0; i < p1.size(); i++)
    {
        fprintf(fp,"?-? %9.3f %9.3f %+e %4.2f\n",p2_ppm[i], p1_ppm[i],p_intensity[i],std::min(p_confidencex[i],p_confidencey[i]));        
    }
    fclose(fp);


    // fp=fopen("picked_peaks.json","w");
    // fprintf(fp,"\"picked_peaks\":[");
    // int npeak=p1.size();
    // for(unsigned int i=0;i<npeak-1;i++)
    // {
    //     fprintf(fp,"{\"cs_x\": %f,\"cs_y\": %f, \"type\": 1, \"index\": %f, \"sigmax\": %f,  \"sigmay\": %f,\"gammax\": %f,  \"gammay\": %f},",p1_ppm[i],p2_ppm[i],p_intensity[i],sigmax[i],sigmay[i],gammax[i],gammay[i]);
    // }
    // fprintf(fp,"{\"cs_x\": %f,\"cs_y\": %f, \"type\": 1, \"index\": %f, \"sigmax\": %f,  \"sigmay\": %f, \"gammax\": %f,  \"gammay\": %f}",p1_ppm[npeak-1],p2_ppm[npeak-1],p_intensity[npeak-1],sigmax[npeak-1],sigmay[npeak-1],gammax[npeak-1],gammay[npeak-1]);

    // fprintf(fp,"]");
    // fclose(fp);

    return 1;
};

