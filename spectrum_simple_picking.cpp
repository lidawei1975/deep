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


#include "commandline.h"
#include "spectrum_simple_picking.h"


namespace ldw_math{
    
    std::vector<std::vector<double> > laplacian_of_gaussian(int n,double sigma)
    {
        double sigma2=sigma*sigma;
        double sigma4=sigma2*sigma2;
        std::vector< std::vector<double> > hg(n * 2 + 1, std::vector<double>(n * 2 + 1));
        double sum,sum2;

        sum=0.0;
        for(int i=0;i<2*n+1;i++)
        {
            double x=-n+i;
            for(int j=0;j<2*n+1;j++)
            {
                double y=-n+j;
                double t=exp((-x*x-y*y)/(2.0*sigma2));
                sum+=t;
                hg[i][j]=t;
            }
        }

        sum2=0;
        for(int i=0;i<2*n+1;i++)
        {
            double x=-n+i;
            for(int j=0;j<2*n+1;j++)
            {
                double y=-n+j;
                double t=(x*x+y*y-2*sigma2)/(sigma4)/sum;
                hg[i][j]*=t;
                sum2+=hg[i][j];
            }
        }
        sum2/=(2*n+1)*(2*n+1);

        for(int i=0;i<2*n+1;i++)
        {
            for(int j=0;j<2*n+1;j++)
            {
                hg[i][j]-=sum2;
            }
        }

        return hg;
    };

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
}


//class spectrum_picking
spectrum_simple_pick::spectrum_simple_pick()
{
    infname="input.ft2";
    user_scale=5.5;  //used defined noise level scale factor
    peak_diag=2;
    b_negative=false;
};

spectrum_simple_pick::~spectrum_simple_pick()
{
};

bool spectrum_simple_pick::normal_peak_picking()
{
    double min_intensity=noise_level*user_scale;
    
    std::cout<<"Minimal peak intensity is set to "<<min_intensity<<std::endl;

    p1.clear();
    p2.clear();
    p_intensity.clear();
    p_type.clear();

    int ncurrent=0;
    for (unsigned int i = 0 + 1; i < xdim - 1; i++)
    {
        for (unsigned int j = 0 + 1; j < ydim - 1; j++)
        {

            if (spect[i+j*xdim] > spect[i+(j-1)*xdim] && spect[i+j*xdim] > spect[i+(j+1)*xdim] && spect[i+j*xdim] > spect[(i-1)+j*xdim] && spect[i+j*xdim] > spect[(i+1)+j*xdim] && spect[i+j*xdim]>min_intensity)
            {   
                int ndiag=0;
                if(spect[i+j*xdim] > spect[(i+1)+(j-1)*xdim]) ndiag++; 
                if(spect[i+j*xdim] > spect[(i-1)+(j-1)*xdim]) ndiag++; 
                if(spect[i+j*xdim] > spect[(i+1)+(j+1)*xdim]) ndiag++; 
                if(spect[i+j*xdim] > spect[(i-1)+(j+1)*xdim]) ndiag++; 
                if(ndiag>=peak_diag)
                {
                    p1.push_back(i); //index is from 0  direct dimension
                    p2.push_back(j);
                    p_type.push_back(1);
                }
            }
        }
    }

    std::cout<<"Picked "<<p1.size()-ncurrent<<" positive peaks"<<std::endl;
    ncurrent=p1.size();

    if(b_negative==true)
    {
        for (unsigned int i = 0 + 1; i < xdim - 1; i++)
        {
            for (unsigned int j = 0 + 1; j < ydim - 1; j++)
            {
                if (spect[i+j*xdim] < spect[i+(j-1)*xdim] && spect[i+j*xdim] < spect[i+(j+1)*xdim] 
                && spect[i+j*xdim] < spect[(i-1)+j*xdim] && spect[i+j*xdim] < spect[(i+1)+j*xdim] 
                && spect[i+j*xdim]<-min_intensity)
                {
                    int ndiag=0;
                    if(spect[i+j*xdim] < spect[(i+1)+(j-1)*xdim]) ndiag++;
                    if(spect[i+j*xdim] < spect[(i-1)+(j-1)*xdim]) ndiag++;
                    if(spect[i+j*xdim] < spect[(i+1)+(j+1)*xdim]) ndiag++;
                    if(spect[i+j*xdim] < spect[(i-1)+(j+1)*xdim]) ndiag++;
                    
                    if(ndiag>=peak_diag)
                    {
                        p1.push_back(i); //index is from 0  direct dimension
                        p2.push_back(j);
                        p_type.push_back(-1);
                    }
                }
            }
        }

        std::cout<<"Picked "<<p1.size()-ncurrent<<" negative peaks"<<std::endl;
    }

    return true;
}

bool spectrum_simple_pick::shoulder_peak_picking()
{
    double min_intensity=noise_level*user_scale;
    int ncurrent = p1.size();
    int nshoulder1 = int(median_width_x / 2);
    int nshoulder2 = int(median_width_y / 2);

    std::cout << "In shoulder peaks pikcing, nshoulder is " << nshoulder1 << " and " << nshoulder2 << std::endl;

    std::vector<std::vector<int>> peak_map;
    peak_map.resize(ydim);
    for (unsigned int i = 0; i < ydim; i++)
        peak_map[i].resize(xdim, 0);

    for (unsigned int i = 0; i < p1.size(); i++)
    {
        int k1 = p1[i];
        int k2 = p2[i];
        for (int m = std::max(k1 - nshoulder1, 0); m <= std::min(k1 + nshoulder1, xdim - 1); m++)
        {
            for (int n = std::max(k2 - nshoulder2, 0); n <= std::min(k2 + nshoulder2, ydim - 1); n++)
            {

                peak_map[n][m] = 1;
            }
        }
    }

    std::vector<std::vector<double>> shoulder;
    //laplacing_convolution(shoulder);
    //gaussian_smoothing(shoulder);
    double tem = std::max(median_width_x, median_width_y);
    tem = sqrt(median_width_x * median_width_y);
    laplacing_of_gaussian_convolution(shoulder, tem);
    for (unsigned int i = 0 + 1; i < xdim - 1; i++)
    {
        for (unsigned int j = 0 + 1; j < ydim - 1; j++)
        {
            if (shoulder[j][i] > shoulder[j - 1][i] && shoulder[j][i] > shoulder[j + 1][i] && shoulder[j][i] > shoulder[j][i - 1] && shoulder[j][i] > shoulder[j][i + 1] && spect[i + j * xdim] > min_intensity && peak_map[j][i] == 0)
            {
                int ndiag = 0;
                if (shoulder[j][i] > shoulder[j - 1][i + 1])
                    ndiag++;
                if (shoulder[j][i] > shoulder[j - 1][i - 1])
                    ndiag++;
                if (shoulder[j][i] > shoulder[j + 1][i + 1])
                    ndiag++;
                if (shoulder[j][i] > shoulder[j + 1][i - 1])
                    ndiag++;

                if (ndiag >= peak_diag )
                {
                    p1.push_back(i); //index is from 0  direct dimension
                    p2.push_back(j);
                    p_type.push_back(2);
                }
            }
        }
    }
    std::cout << "Picked " << p1.size() - ncurrent << " positive shoulder peaks" << std::endl;
    ncurrent = p1.size();

    if(b_negative==true)
    {
        for (unsigned int i = 0 + 1; i < xdim - 1; i++)
        {
            for (unsigned int j = 0 + 1; j < ydim - 1; j++)
            {
                if (shoulder[j][i] < shoulder[j - 1][i] && shoulder[j][i] < shoulder[j + 1][i] && shoulder[j][i] < shoulder[j][i - 1] && shoulder[j][i] < shoulder[j][i + 1] && spect[i + j * xdim] < -min_intensity && peak_map[j][i] == 0)
                {
                    int ndiag = 0;
                    if (shoulder[j][i] < shoulder[j - 1][i + 1])
                        ndiag++;
                    if (shoulder[j][i] < shoulder[j - 1][i - 1])
                        ndiag++;
                    if (shoulder[j][i] < shoulder[j + 1][i + 1])
                        ndiag++;
                    if (shoulder[j][i] < shoulder[j + 1][i - 1])
                        ndiag++;

                    if (ndiag >= peak_diag )
                    {
                        p1.push_back(i); //index is from 0  direct dimension
                        p2.push_back(j);
                        p_type.push_back(-2);
                    }
                }
            }
        }
        std::cout << "Picked " << p1.size() - ncurrent << " negative shoulder peaks" << std::endl;
    }

    return true;
}

bool spectrum_simple_pick::laplacing_of_gaussian_convolution(std::vector< std::vector<double> > &s,double width)
{
    double sigma=width/2.355/2.0;
    int n=int(width*0.5);

    std::cout<<"in LoG filtering, sigma is "<<sigma<<" (width is "<<width<<")"<<std::endl;

    std::vector<std::vector<double> > lap=ldw_math::laplacian_of_gaussian(n,sigma);

    s.resize(ydim);
    for (unsigned int i = 0; i < ydim; i++)
        s[i].resize(xdim,0.0);


    for(unsigned int i=n;i<xdim-n;i++)
    {
        for(unsigned int j=n;j<ydim-n;j++)
        {
            double t=spect[i+j*xdim];

            for(int mm=-n;mm<=n;mm++)
            {
                for(int nn=-n;nn<=n;nn++)
                {
                    s[j+mm][i+nn]-=lap[mm+n][nn+n]*t;
                }
            }     
        }
    }

    return true;
}


bool spectrum_simple_pick::sub_pixel_0()
{
    std::vector<double> sx,sy;

    sx.clear();
    sy.clear();
   
    // 5-points 2nd order polynorminal fitting
    for(unsigned int m=0;m<p1.size();m++)
    {
        int i=p1[m];
        int j=p2[m];
        //std::cout<<m<<" "<<i<<" "<<j<<" "<<p_type[m]<<std::endl;
        if (abs(p_type[m])==1) //normal peak
        {
            float result[5];

            result[0]=0.5*spect[i-1+j*xdim]-spect[i+j*xdim]+0.5*spect[i+1+j*xdim];
            result[1]=-0.5*spect[i-1+j*xdim]+0.5*spect[i+1+j*xdim];
            result[2]=0.5*spect[i+(j-1)*xdim]-spect[i+j*xdim]+0.5*spect[i+(j+1)*xdim];
            result[3]=-0.5*spect[i+(j-1)*xdim]+0.5*spect[i+(j+1)*xdim];
            result[4]=spect[i+j*xdim];

            float xc=i-result[1]/(2*result[0]);
            float yc=j-result[3]/(2*result[2]);
            float a=result[4]-result[1]*result[1]/(result[0]*4)-result[3]*result[3]/(result[2]*4);
            float estx,esty;

            estimate_sigma(i-xc,j-yc,spect[i+(j-1)*xdim],spect[i-1+j*xdim],spect[i+j*xdim],spect[i+1+j*xdim],spect[i+(j+1)*xdim],a,estx,esty);

            //estx and esty is actually 2*sigma*sigma

            if(estx<0.5) estx=0.5;
            if(estx>800) estx=800;

            if(esty<0.5) esty=0.5;
            if(esty>800) esty=800;

            //std::cout<<estx<<" "<<esty<<std::endl;
            

            sx.push_back(estx);
            sy.push_back(esty);
        }
    }

    median_width_x=std::sqrt(ldw_math::calcualte_median(sx)/2.0)*2.355;
    median_width_y=std::sqrt(ldw_math::calcualte_median(sy)/2.0)*2.355;

    std::cout<<"Median peak width is estimated to be "<<median_width_x<<" "<<median_width_y<< " from picking."<<std::endl;

    return true;
}

bool spectrum_simple_pick::get_mean_width_from_picking()
{
   std::vector<double> sx,sy;

    sx.clear();
    sy.clear();
    for(unsigned int i=0;i<p1.size();i++)
    {
        if(abs(p_type[i])==1 && sigmax[i]>0.0 && sigmay[i]>0.0)
        {
            sx.push_back(sigmax[i]);
            sy.push_back(sigmay[i]);
        }
    }
    median_width_x=std::sqrt(ldw_math::calcualte_median(sx)/2.0)*2.355;
    median_width_y=std::sqrt(ldw_math::calcualte_median(sy)/2.0)*2.355;

    std::cout<<"Median peak width is estimated to be "<<median_width_x<<" "<<median_width_y<< " from picking."<<std::endl;

    return true;
}

bool spectrum_simple_pick::estimate_sigma(float xc,float yc, float a1,float a2, float a3, float a4, float a5, float aa, float &sx, float &sy)
{
    Eigen::MatrixXf m(5,2);
    m(0,0)=xc*xc;
    m(0,1)=(-1-yc)*(-1-yc);

    m(1,0)=(-1-xc)*(-1-xc);
    m(1,1)=yc*yc;
    
    m(2,0)=xc*xc;
    m(2,1)=yc*yc;
    
    m(3,0)=(1-xc)*(1-xc);
    m(3,1)=yc*yc;

    m(4,0)=xc*xc;
    m(4,1)=(1-yc)*(1-yc);

    Eigen::VectorXf b(5);
    b(0)=log(aa/a1);
    b(1)=log(aa/a2);
    b(2)=log(aa/a3);
    b(3)=log(aa/a4);
    b(4)=log(aa/a5);
        
    Eigen::VectorXf r=m.colPivHouseholderQr().solve(b);

    sx=1.0/r(0)*0.8;
    sy=1.0/r(1)*0.8;  //0.8 here is pure emperical. 

    //water-proof our estimation, 0 means failure
    if(std::isnan(sx) || sx<0.1 || sx>1000 )
    {
        sx=0.0;
    }
    if(std::isnan(sy) || sy<0.1 || sy>1000 )
    {
        sy=0.0;
    }

    return true;
}

bool spectrum_simple_pick::sub_pixel()
{
    std::cout<<"get sub-pixel position and amplitude using 5-points 2nd order polynorminal fitting ......"<<std::flush;

    sigmax.clear();
    sigmay.clear();
    p_intensity.clear();

    // 5-points 2nd order polynorminal fitting
    for(unsigned int m=0;m<p1.size();m++)
    {
        int i=p1[m];
        int j=p2[m];
        //std::cout<<m<<" "<<i<<" "<<j<<" "<<p_type[m]<<std::endl;
        if (abs(p_type[m])==1) //normal peak
        {
            float result[5];

            result[0]=0.5*spect[i-1+j*xdim]-spect[i+j*xdim]+0.5*spect[i+1+j*xdim];
            result[1]=-0.5*spect[i-1+j*xdim]+0.5*spect[i+1+j*xdim];
            result[2]=0.5*spect[i+(j-1)*xdim]-spect[i+j*xdim]+0.5*spect[i+(j+1)*xdim];
            result[3]=-0.5*spect[i+(j-1)*xdim]+0.5*spect[i+(j+1)*xdim];
            result[4]=spect[i+j*xdim];

            float xc=i-result[1]/(2*result[0]);
            float yc=j-result[3]/(2*result[2]);
            float a=result[4]-result[1]*result[1]/(result[0]*4)-result[3]*result[3]/(result[2]*4);
            float estx,esty;

            estimate_sigma(i-xc,j-yc,spect[i+(j-1)*xdim],spect[i-1+j*xdim],spect[i+j*xdim],spect[i+1+j*xdim],spect[i+(j+1)*xdim],a,estx,esty);

            //estx and esty is actually 2*sigma*sigma

            p1[m]=xc;
            p2[m]=yc;
            p_intensity.push_back(a);

            //std::cout<<"Estimated width is "<<estx<<" "<<esty<<std::endl;
            //adjust them to meaning value because estimate_sigma is not very stable for some cases.
            if(estx<0.5) estx=0.5;
            if(estx>800) estx=800;

            if(esty<0.5) esty=0.5;
            if(esty>800) esty=800;
            

            sigmax.push_back(estx);
            sigmay.push_back(esty);
            gammax.push_back(1e-6);
            gammay.push_back(1e-6);
        }
    }
    std::cout<<" Done."<<std::endl;
    return true;
}



bool spectrum_simple_pick::simple_peak_picking(bool b_negative_)
{
    b_negative=b_negative_;

    normal_peak_picking(); 
    sub_pixel_0();
    shoulder_peak_picking();
    sub_pixel();
    
    get_mean_width_from_picking(); //median_width_x = 2.355*sigma, at this time, sigmax and sigmay is actually 2*sigma*sigma

    double wid_x,wid_y;  //sigma 

    wid_x=(median_width_x/2.355);
    wid_y=(median_width_y/2.355);
   

    //set peak width of shoulder peaks to mean value of all true peaks, because we can't estimate shoulder peak width
    //wid_x and wid_y are actually 2*sigma*sigma

    p_intensity.resize(p1.size());
    sigmax.resize(p1.size());
    sigmay.resize(p1.size());
    gammax.resize(p1.size(),1e-20);
    gammay.resize(p1.size(),1e-20);
    

    for(unsigned int i=0;i<p1.size();i++)
    {
        if(abs(p_type[i])==1)
        {
            sigmax[i]=std::sqrt(sigmax[i]/2);
            sigmay[i]=std::sqrt(sigmay[i]/2);
        }
        if(abs(p_type[i])==2)
        {
            p_intensity[i]=spect[int(p1[i])+int(p2[i])*xdim];
            sigmax[i]=wid_x;
            sigmay[i]=wid_y;
        }
    }

    // not real 
    gammax.resize(p1.size(),0.00001);
    gammay.resize(p1.size(),0.00001);
    p_confidencex.resize(p1.size(),1.0);
    p_confidencey.resize(p1.size(),1.0);

    get_ppm_from_point();


    return true;
}



bool spectrum_simple_pick::print_peaks_picking(std::string outfname)
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

