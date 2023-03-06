//#include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <valarray>
#include <string>
#include <cstring>
#include <vector>
#include <array>

#include "ceres/ceres.h"
#include "glog/logging.h"


using ceres::CostFunction;
using ceres::AutoDiffCostFunction;
using ceres::DynamicNumericDiffCostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include "json/json.h"
#include "commandline.h"
#include "dnn_picker.h"
#include "spectrum_pick.h"
#include "spectrum_io_1d.h"
#include "spectrum_pick_1d.h"
#include "spectrum_fit_3d.h"

extern "C"  
{
    double voigt(double x, double sigma, double gamma);
};

//some help function

spectrum_fit_3d::spectrum_fit_3d()
{
    w[0]=0.0;
    w[1]=0.0;
    w[2]=0.0;
    
    maxround=10;
    peak_shape=gaussian_type;

    class_flag=2; //fit_3d
};

spectrum_fit_3d::~spectrum_fit_3d()
{
   
};

bool spectrum_fit_3d::init_fitting_parameter(int r_,int i_)
{
    maxround=r_;
    if(i_==1) peak_shape=gaussian_type;
    else if(i_==2) peak_shape=voigt_type;
    return true;  
}


bool spectrum_fit_3d::get_median_width_of_peaks()
{
    std::vector< std::vector<double> > ss;
    ss.clear();
    ss.resize(3,std::vector<double>(0));

    for (unsigned int i = 0; i < peaks_pos.size(); i++)
    {
        for(int k=0;k<3;k++)
        {
            ss[k].push_back( 0.5346 * gamma[i][k] * 2 + std::sqrt(0.2166 * 4 * gamma[i][k] * gamma[i][k] + sigma[i][k] * sigma[i][k] * 8 * 0.6931));
        }
    }

    for(int k=0;k<3;k++)
    {
        median_widths[k]=ldw_math_3d::calcualte_median(ss[k]);
    }
    
    return true;
    
};

bool spectrum_fit_3d::read_for_fitting(std::string fname1, std::string fname2)  //read for fitting
{
    std::string sft2(".ft3");

    //make sure both filenames are ended with .ft3
    if(!std::equal(sft2.rbegin(), sft2.rend(), fname1.rbegin()) || !std::equal(sft2.rbegin(), sft2.rend(), fname2.rbegin()) )
    {
        std::cout<<"both filenames must end with .ft3"<<std::endl;
        return false;
    }

    //remove .ft3 from both file names
    fname1=fname1.substr(0,fname1.length()-4);
    fname2=fname2.substr(0,fname2.length()-4);
    
    std::size_t found1 = fname1.find_first_of("0123456789");
    std::size_t found2 = fname2.find_first_of("0123456789");

    std::string basename1=fname1.substr(0,found1);
    std::string basename2=fname2.substr(0,found2);

    int n_zero=fname1.length()-basename1.length();
    
    if(basename1!=basename2)
    {
        std::cout<<"The two filenames have different base."<<std::endl;
        return false;
    }

    std::string num1=fname1.substr(found1);
    std::string num2=fname2.substr(found2);

    int n=num1.length();
    int n1=atoi(num1.c_str());
    int n2=atoi(num2.c_str());

    for(int i=n1;i<=n2;i++)
    {
        std::string new_string;
        std::string old_string=std::to_string(i);
        int nt=n_zero - old_string.length();
        if(nt>0)
        {
            new_string = std::string(nt, '0') + old_string;
        }
        else
        {
            new_string = old_string;
        }

        std::string current_fname=basename1+new_string+".ft3";
        std::cout<<"Read in "<<current_fname<<std::endl;

        spectrum_io f;
        if((i-n1)%100==0) f.init(current_fname,1);
        else f.init(current_fname,0);
        spectra_2d.emplace_back(f);

        if(i==n1)
        {
            f.get_ppm_infor(begin,step);
            f.get_dim(&xdim,&ydim);
            begin[2]=f.begin3;
            stop[2]=f.stop3;
            std::cout<<"3rd dimension is from "<<begin[2]<<" to "<<stop[2]<<std::endl;
        }

        std::cout<<std::endl<<std::endl;
    }

    zdim=n2-n1+1;
    step[2]=(stop[2]-begin[2])/zdim;
    begin[2]+=step[2]; //to be consistent with pipe.

   
    std::cout << "Direct dimension size is " << xdim << " indirect dimension is " << ydim << " and " << zdim << std::endl;
    std::cout << "  Direct dimension   offset is " << begin[0] << ", ppm per step is " << step[0] << " ppm and end is " <<begin[0]+step[0]*xdim<< std::endl;
    std::cout << "Indirect dimension 1 offset is " << begin[1] << ", ppm per step is " << step[1] << " ppm and end is" <<begin[1]+step[1]*ydim<< std::endl;
    std::cout << "Indirect dimension 2 offset is " << begin[2] << ", ppm per step is " << step[2] << " ppm and end is"<<begin[2]+step[2]*zdim << std::endl;

    //define median noise of all planes as 3D spectral noise level 
    if(noise_level<0.00001)
    {
        std::vector<float> scores;
        for(int i=0;i<spectra_2d.size();i++)
        {
            if(i%100==0)
            {
                scores.push_back(spectra_2d[i].get_noise_level());
            }
        }
        sort(scores.begin(), scores.end());
        noise_level = scores[scores.size() / 2];
        std::cout<<"Estimated noise level is "<<noise_level<<std::endl;
    }
    else
    {
        std::cout<<"Set noise level directly to "<<noise_level<<std::endl;
    }

    return true;
};


//for positve peaks only
bool spectrum_fit_3d::peak_partition()
{

    double lowest_level=noise_level*user_scale2;
    std::vector< std::vector<int> > used(ydim*zdim);
    peaks_in_this_line.resize(ydim*zdim);
    begin_array.resize(ydim*zdim);
    end_array.resize(ydim*zdim);


    for(int k=0;k<zdim;k++)
    {
        std::vector<int> peak_map2; //0: far away from peak, 1: near peak, >=2: it is a peak and peak index is value-2 
        peak_map2.clear();
        peak_map2.resize(xdim*ydim,0);
        for(unsigned int i=0;i<peaks_pos.size();i++)
        {
            if(abs(peaks_pos[i][2]-k)>=w[2]*2) continue;
            if(intensity[i]<0.0) continue; //only consider positive peaks

            int xfrom=int(peaks_pos[i][0]-w[0]*2+0.5);
            int xto=int(peaks_pos[i][0]+w[0]*2+0.5);
            int yfrom=int(peaks_pos[i][1]-w[1]*2+0.5)+1;
            int yto=int(peaks_pos[i][1]+w[1]*2+0.5)+1;

            if(xfrom<0) xfrom=0;
            if(xto>xdim) xto=xdim;
            if(yfrom<0) yfrom=0;
            if(yto>ydim) yto=ydim;
            for(int m=xfrom;m<xto;m++)
            {
                for(int n=yfrom;n<yto;n++)
                {
                    peak_map2[m*ydim+n]=1;
                }
            }
        }

        for(unsigned int i=0;i<peaks_pos.size();i++)
        {
            if(int(peaks_pos[i][2]+0.5)==k)
            {
                int xint=int(peaks_pos[i][0]+0.5);
                int yint=int(peaks_pos[i][1]+0.5);
                peak_map2[xint*ydim+yint]=i+2; //peak index +2 
            }
        }

        for(int j=0;j<ydim;j++)
        { 
            if(spectra_2d[k].get_spect_data()[j*xdim+0]>=lowest_level && peak_map2[j]>=1) begin_array[k*ydim+j].push_back(0);
            for(int i=1;i<xdim;i++)
            {
                if((spectra_2d[k].get_spect_data()[j*xdim+i-1]<lowest_level || peak_map2[j+(i-1)*ydim]==0) && (spectra_2d[k].get_spect_data()[j*xdim+i]>=lowest_level && peak_map2[j+i*ydim]>=1)) begin_array[k*ydim+j].push_back(i);
                if((spectra_2d[k].get_spect_data()[j*xdim+i-1]>=lowest_level && peak_map2[j+(i-1)*ydim]>=1) && (spectra_2d[k].get_spect_data()[j*xdim+i]<lowest_level || peak_map2[j+i*ydim]==0)) end_array[k*ydim+j].push_back(i);
            }
            if(end_array[k*ydim+j].size()<begin_array[k*ydim+j].size()) end_array[k*ydim+j].push_back(xdim);
            for(int i=0;i<end_array[k*ydim+j].size();i++) used[k*ydim+j].push_back(0);

            //find peaks that belong to each segment (defined by begin_array and s)
            for(int i=0;i<end_array[k*ydim+j].size();i++)
            {
                std::vector<int> tt1(0);
                for(int m=begin_array[k*ydim+j][i];m<end_array[k*ydim+j][i];m++)
                {
                    if(peak_map2[m*ydim+j]>=2) //peak
                    {
                        tt1.push_back(peak_map2[m*ydim+j]-2);   
                    }
                }
                peaks_in_this_line[k*ydim+j].push_back(tt1);
            }
        }
    }


    std::deque< ldw_triple > work;
    int position;

    for(int k=0;k<zdim;k++)
    for(int j=0;j<ydim;j++)
    {
        for(int i=0;i<used[k*ydim+j].size();i++)
        {
            if(used[k*ydim+j][i]==0)
            {
                used[k*ydim+j][i]=1;
                work.clear();
                work.push_back(ldw_triple(k,j,i));
                position=0;
                while(position<work.size())
                {
                    ldw_triple c=work[position];
                    position++;
                    
                    int kk;
                    int jj;

                    kk=c.first;
                    for(jj=std::max(0,c.second-1);jj<std::min(ydim,c.second+2);jj++) //check previous and following line along y dimension
                    {
                        if(jj==c.second) continue; //same line, skip
                        for(int ii=0;ii<used[kk*ydim+jj].size();ii++)
                        {
                            if( used[kk*ydim+jj][ii]==1)
                            {
                                continue;
                            }

                            if (end_array[kk*ydim+jj][ii]>=begin_array[c.first*ydim+c.second][c.third] && begin_array[kk*ydim+jj][ii]<=end_array[c.first*ydim+c.second][c.third])
                            {
                                work.push_back(ldw_triple(kk,jj,ii));
                                used[kk*ydim+jj][ii]=1;
                            }
                        }
                    }

                    jj=c.second;
                    for(kk=std::max(0,c.first-1);kk<std::min(zdim,c.first+2);kk++) //check previous and following line along z dimension
                    {
                        if(kk==c.first) continue; //same line, skip
                        for(int ii=0;ii<used[kk*ydim+jj].size();ii++)
                        {
                            if( used[kk*ydim+jj][ii]==1)
                            {
                                continue;
                            }

                            if (end_array[kk*ydim+jj][ii]>=begin_array[c.first*ydim+c.second][c.third] && begin_array[kk*ydim+jj][ii]<=end_array[c.first*ydim+c.second][c.third])
                            {
                                work.push_back(ldw_triple(kk,jj,ii));
                                used[kk*ydim+jj][ii]=1;
                            }
                        }
                    }
                }
                clusters2.push_back(work);
            }
        }
    }
    std::cout<<"Total "<<clusters2.size()<<" positve peak clusters."<<std::endl;
    
    return true;
};


bool spectrum_fit_3d::peak_partition_neg()
{

    double lowest_level=noise_level*user_scale2;
    std::vector< std::vector<int> > used(ydim*zdim);
    peaks_in_this_line_neg.resize(ydim*zdim);
    begin_array_neg.resize(ydim*zdim);
    end_array_neg.resize(ydim*zdim);


    for(int k=0;k<zdim;k++)
    {
        std::vector<int> peak_map2; //0: far away from peak, 1: near peak, >=2: it is a peak and peak index is value-2 
        peak_map2.clear();
        peak_map2.resize(xdim*ydim,0);
        for(unsigned int i=0;i<peaks_pos.size();i++)
        {
            if(abs(peaks_pos[i][2]-k)>=w[2]*2) continue;
            if(intensity[i]>-lowest_level) continue; //only consider negative peaks

            int xfrom=int(peaks_pos[i][0]-w[0]*2+0.5);
            int xto=int(peaks_pos[i][0]+w[0]*2+0.5);
            int yfrom=int(peaks_pos[i][1]-w[1]*2+0.5)+1;
            int yto=int(peaks_pos[i][1]+w[1]*2+0.5)+1;

            if(xfrom<0) xfrom=0;
            if(xto>xdim) xto=xdim;
            if(yfrom<0) yfrom=0;
            if(yto>ydim) yto=ydim;
            for(int m=xfrom;m<xto;m++)
            {
                for(int n=yfrom;n<yto;n++)
                {
                    peak_map2[m*ydim+n]=1;
                }
            }
        }

        for(unsigned int i=0;i<peaks_pos.size();i++)
        {
            if(int(peaks_pos[i][2]+0.5)==k)
            {
                int xint=int(peaks_pos[i][0]+0.5);
                int yint=int(peaks_pos[i][1]+0.5);
                peak_map2[xint*ydim+yint]=i+2; //peak index +2 
            }
        }

        for(int j=0;j<ydim;j++)
        { 
            if(spectra_2d[k].get_spect_data()[j*xdim+0]<=-lowest_level && peak_map2[j]>=1) begin_array[k*ydim+j].push_back(0);
            for(int i=1;i<xdim;i++)
            {
                if((spectra_2d[k].get_spect_data()[j*xdim+i-1]>-lowest_level || peak_map2[j+(i-1)*ydim]==0) && (spectra_2d[k].get_spect_data()[j*xdim+i]<-lowest_level && peak_map2[j+i*ydim]>=1)) begin_array_neg[k*ydim+j].push_back(i);
                if((spectra_2d[k].get_spect_data()[j*xdim+i-1]<=-lowest_level && peak_map2[j+(i-1)*ydim]>=1) && (spectra_2d[k].get_spect_data()[j*xdim+i]>-lowest_level || peak_map2[j+i*ydim]==0)) end_array_neg[k*ydim+j].push_back(i);
            }
            if(end_array_neg[k*ydim+j].size()<begin_array_neg[k*ydim+j].size()) end_array_neg[k*ydim+j].push_back(xdim);
            for(int i=0;i<end_array_neg[k*ydim+j].size();i++) used[k*ydim+j].push_back(0);

            //find peaks that belong to each segment (defined by begin_array and s)
            for(int i=0;i<end_array_neg[k*ydim+j].size();i++)
            {
                std::vector<int> tt1(0);
                for(int m=begin_array_neg[k*ydim+j][i];m<end_array_neg[k*ydim+j][i];m++)
                {
                    if(peak_map2[m*ydim+j]>=2) //peak
                    {
                        tt1.push_back(peak_map2[m*ydim+j]-2);   
                    }
                }
                peaks_in_this_line_neg[k*ydim+j].push_back(tt1);
            }
        }
    }


    std::deque< ldw_triple > work;
    int position;

    for(int k=0;k<zdim;k++)
    for(int j=0;j<ydim;j++)
    {
        for(int i=0;i<used[k*ydim+j].size();i++)
        {
            if(used[k*ydim+j][i]==0)
            {
                used[k*ydim+j][i]=1;
                work.clear();
                work.push_back(ldw_triple(k,j,i));
                position=0;
                while(position<work.size())
                {
                    ldw_triple c=work[position];
                    position++;
                    
                    int kk;
                    int jj;

                    kk=c.first;
                    for(jj=std::max(0,c.second-1);jj<std::min(ydim,c.second+2);jj++) //check previous and following line along y dimension
                    {
                        if(jj==c.second) continue; //same line, skip
                        for(int ii=0;ii<used[kk*ydim+jj].size();ii++)
                        {
                            if( used[kk*ydim+jj][ii]==1)
                            {
                                continue;
                            }

                            if (end_array_neg[kk*ydim+jj][ii]>=begin_array_neg[c.first*ydim+c.second][c.third] && begin_array_neg[kk*ydim+jj][ii]<=end_array_neg[c.first*ydim+c.second][c.third])
                            {
                                work.push_back(ldw_triple(kk,jj,ii));
                                used[kk*ydim+jj][ii]=1;
                            }
                        }
                    }

                    jj=c.second;
                    for(kk=std::max(0,c.first-1);kk<std::min(zdim,c.first+2);kk++) //check previous and following line along z dimension
                    {
                        if(kk==c.first) continue; //same line, skip
                        for(int ii=0;ii<used[kk*ydim+jj].size();ii++)
                        {
                            if( used[kk*ydim+jj][ii]==1)
                            {
                                continue;
                            }

                            if (end_array_neg[kk*ydim+jj][ii]>=begin_array_neg[c.first*ydim+c.second][c.third] && begin_array_neg[kk*ydim+jj][ii]<=end_array_neg[c.first*ydim+c.second][c.third])
                            {
                                work.push_back(ldw_triple(kk,jj,ii));
                                used[kk*ydim+jj][ii]=1;
                            }
                        }
                    }
                }
                clusters2_neg.push_back(work);
            }
        }
    }
    std::cout<<"Total "<<clusters2_neg.size()<<" negative peak clusters."<<std::endl;
    
    return true;
};


bool spectrum_fit_3d::work()
{
    //set up peak cannot move flags
    peak_cannot_move_flag.resize(intensity.size(), 0);
    for (int i = 0; i < intensity.size(); i++)
    {
        int ntemp = int(round(peaks_pos[i][0])) + int(round(peaks_pos[i][1])) * xdim;
        int k = int(round(peaks_pos[i][2]));
        if (intensity[i] / spectra_2d[k].get_spect_data()[ntemp] < 0.5)
        {
            peak_cannot_move_flag[i] = 1; //peak can't move because of low reliability
        }
    }

    get_median_width_of_peaks();

    for(int k=0;k<3;k++)
    {
        if(w[k]>1e-10)
        {
            w[k]/=fabs(step[k]);
        }
        else
        {
            w[k]=median_widths[k]*1.60;
        }
    }


    std::cout<<std::endl;
    std::cout<<std::endl;
    std::cout<<"**********************************************************************************************************************************"<<std::endl;
    std::cout<<"IMPORTANT, make sure these two values are reasonable. If not, set them using -wx and -wy command line arguments. Unit is ppm in commandline arguments!"<<std::endl;
    std::cout<<"wx="<<w[0]<<" and wy="<<w[1]<<" and wz="<<w[2]<<" (points) in this fitting."<<std::endl;
    std::cout<<"Firstly, peaks that are within wx*4 (wy*4) along direct(indirect) dimension are fitted tagather, if they are also connected by data points above noise*user_scale2."<<std::endl;
    std::cout<<"Secondly, after peak deconvolution in mixed Gaussian algorithm, fitting of each peak are done on an area of wx*3 by wy*3. by wz*3."<<std::endl;
    std::cout<<"**********************************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;
    std::cout<<std::endl;

    
   peak_partition();
   peak_partition_neg();
   


    //part 2
    peak_updated.resize(peaks_pos.size(),0);
    fitted_peaks_pos=peaks_pos;
    fitted_sigma=sigma;
    fitted_gamma=gamma;
    fitted_intensity=intensity;
    numerical_sum.resize(peaks_pos.size(),0.0);

    int n_postive_clusters=clusters2.size();
    int n_negative_clusters=clusters2_neg.size();

    //for a better organized code, we combine positive and negative peaks together
    clusters2.insert(clusters2.end(),clusters2_neg.begin(),clusters2_neg.end());
    

    int min1,min2,min3,max1,max2,max3;
    int counter=0;
    for(unsigned int i0=0;i0<clusters2.size();i0++)
    {
        std::cout<<"parepare, "<<i0<<" out of "<<clusters2.size()<<std::endl;
        min1=min2=min3=1000000;
        max1=max2=max3=-1000000;

        bool b_positve_cluster=i0<n_postive_clusters?true:false;

        for (unsigned int i1 = 0; i1 < clusters2[i0].size(); i1++)
        {
            ldw_triple temp = clusters2[i0][i1];
            int k=temp.first;
            int j=temp.second;
            if(b_positve_cluster)
            {
                int begin=begin_array[k*ydim+j][temp.third];
                int stop=end_array[k*ydim+j][temp.third];

                if (begin <= min1) min1 = begin;
                if (stop >= max1) max1 = stop;
                if (j <= min2) min2 = j;
                if (j >= max2) max2 = j;
                if (k <= min3) min3 = k;
                if (k >= max3) max3 = k;
            }
            else
            {
                int begin=begin_array_neg[k*ydim+j][temp.third];
                int stop=end_array_neg[k*ydim+j][temp.third];

                if (begin <= min1) min1 = begin;
                if (stop >= max1) max1 = stop;
                if (j <= min2) min2 = j;
                if (j >= max2) max2 = j;
                if (k <= min3) min3 = k;
                if (k >= max3) max3 = k;
            }
        }
        max1++;
        max2++;
        max3++;

        if( max1-min1<3 || max2-min2<3  || max3-min3<3 )
        {
            std::cout<<"cluster "<<i0<< " is too small, remove it."<<std::endl;
            continue;
        }

        std::vector<double>  spect_parts((max1-min1)*(max2-min2)*(max3-min3),0.0);
        std::vector<std::array<double,3> > xx,fwhhs;
        std::vector<double> intens;
        std::vector<int> not_move;
        std::vector<int> ori_index;


        for(unsigned int i1=0;i1<clusters2[i0].size();i1++)
        {
            ldw_triple temp = clusters2[i0][i1];
            int k=temp.first;
            int j=temp.second;
            int begin;
            int stop;
            
            if(b_positve_cluster)
            {
                begin=begin_array[k*ydim+j][temp.third];
                stop=end_array[k*ydim+j][temp.third];
            }
            else
            {
                begin=begin_array_neg[k*ydim+j][temp.third];
                stop=end_array_neg[k*ydim+j][temp.third];
            }
           
            for(int kk=begin;kk<stop;kk++)
            {
                spect_parts[(k-min3)*(max2-min2)*(max1-min1)+(j-min2)*(max1-min1)+(kk-min1)]=spectra_2d[k].get_spect_data()[kk+j*xdim];
            }

            int peaks_in_this_line_size=b_positve_cluster?peaks_in_this_line[k*ydim+j][temp.third].size():peaks_in_this_line_neg[k*ydim+j][temp.third].size();

            for (int ii = 0; ii < peaks_in_this_line_size; ii++)
            {
                int peak_ndx = b_positve_cluster?peaks_in_this_line[k * ydim + j][temp.third][ii]:peaks_in_this_line_neg[k * ydim + j][temp.third][ii];

                std::array<double, 3> t1;
                t1[0] = peaks_pos[peak_ndx][0] - min1;
                t1[1] = peaks_pos[peak_ndx][1] - min2;
                t1[2] = peaks_pos[peak_ndx][2] - min3;
                xx.push_back(t1);

                std::array<double, 3> fwhh;
                for (int kk = 0; kk < 3; kk++)
                {
                    fwhh[kk] = std::max(1.0692 * gamma[peak_ndx][kk] + sqrt(0.8664 * gamma[peak_ndx][kk] * gamma[peak_ndx][kk] + 5.5452 * sigma[peak_ndx][kk] * sigma[peak_ndx][kk]), 2.0);
                }
                fwhhs.push_back(fwhh);
                ori_index.push_back(peak_ndx);
                intens.push_back(intensity[peak_ndx]);
                not_move.push_back(peak_cannot_move_flag[peak_ndx]);
            }
        }


        if(xx.size()==0)
        {
            std::cout<<"cluster "<<i0<< " has no peak in it, remove it."<<std::endl;
        }
        else
        {
            std::cout<<"cluster "<<i0<< " has "<<xx.size()<<" peak before fitting."<<std::endl;
            std::array<int,3> dims,starts;
            dims[0]=max1-min1;
            dims[1]=max2-min2;
            dims[2]=max3-min3;
            starts[0]=min1;
            starts[1]=min2;
            starts[2]=min3;
            
            gaussian_fit_3d myfit;  
            myfit.init_optional(starts); 
            myfit.init(peak_shape,maxround,dims,spect_parts,xx,fwhhs,intens,not_move); //spect_parts memory order: z,y,x
            myfit.set_peak_paras(median_widths,noise_level,user_scale2); 
            myfit.run();

            //extract fitted peaks from myfit
            for(int m=0;m<myfit.intensity.size();m++)
            {
                int ndx=ori_index[m];
                if(myfit.to_remove[m]==1)
                {
                    fitted_intensity[ndx]=0.0;  //flag of removed peak.
                }
                else
                {
                    for(int k=0;k<3;k++)
                    {
                        fitted_peaks_pos[ndx][k]=myfit.x[m][k]+starts[k];
                    }
                    fitted_sigma[ndx]=myfit.sigma[m];
                    fitted_gamma[ndx]=myfit.gamma[m];
                    fitted_intensity[ndx]=myfit.intensity[m];
                    numerical_sum[ndx]=myfit.numerical_sum[m];
                    peak_updated[ndx]=1; //peak that has been updated by fitting
                }
            }
        }
    }
    return true;
};

bool spectrum_fit_3d::print_fitted_peaks(std::string fname)
{
    FILE *fp=fopen(fname.c_str(),"w");

    fprintf(fp,"Name F1ppm F2ppm F3ppm intensity F1 F2 F3 sigma1 sigma2 sigma3 gamma1 gamma2 gamma3\n");


    // std::vector<int> ndx;
    // ldw_math_3d::sortArr(intensity,ndx);

    // for(int j=ndx.size()-1;j>=0;j--)
    for(int i=0;i<fitted_intensity.size();i++)
    {
        // int i=ndx[j];
        //input peaks that have not been fitted by the fitter, because they are in an empty region of the spe
        if(peak_updated[i]==0) continue; 

        double ppm[3];
        for(int m=0;m<3;m++)
        {
            ppm[m]=begin[m] + step[m] * fitted_peaks_pos[i][m];
        }
        if(fabs(fitted_intensity[i])>0.0)
        {
            fprintf(fp,"(%d)N-C-H %7.3f %7.3f %7.3f %7.3f ",i+1,ppm[0],ppm[1],ppm[2],fitted_intensity[i]);
            fprintf(fp,"%7.3f %7.3f %7.3f ",fitted_peaks_pos[i][0],fitted_peaks_pos[i][1],fitted_peaks_pos[i][2]);
            fprintf(fp,"%7.3f %7.3f %7.3f ",fitted_sigma[i][0],fitted_sigma[i][1],fitted_sigma[i][2]);
            fprintf(fp,"%7.3f %7.3f %7.3f ",fitted_gamma[i][0],fitted_gamma[i][1],fitted_gamma[i][2]);
            fprintf(fp,"\n");
        }
    }
    fclose(fp);
    return true;
};

//class gaussian_fit_3d
gaussian_fit_3d::gaussian_fit_3d()
{
    peak_shape=gaussian_type;
    user_scale2=3.5;

    options.max_num_iterations = 250;
    options.function_tolerance = 1e-12;
    options.parameter_tolerance =1e-12;
    options.initial_trust_region_radius = 15.0;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
};

gaussian_fit_3d::~gaussian_fit_3d()
{

};

bool gaussian_fit_3d::init_optional(std::array<int,3> x)
{
    start_pos=x;
    return true;
};

bool gaussian_fit_3d::set_peak_paras(std::array<double,3> w,double n,double u)
{
    median_widths=w;   
    noise_level=n;
    user_scale2=u;
    return true;
};

bool gaussian_fit_3d::init(fit_type ft_,int rmax_,std::array<int,3> dims_,std::vector<double> spe_,std::vector<std::array<double,3> > x_,std::vector<std::array<double,3> > s_,std::vector<double> a_,std::vector<int> nm_)
{
    // s_ is FWHH
    // a_ is peak amplitude.

    peak_shape=ft_;
    dims=dims_;
    rmax=rmax_;
    spe=spe_;
    x=x_;
    cannot_move=nm_;

    // std::ofstream fout("input_3d.txt");
    // for(int i=0;i<spe.size();i++)
    // {
    //     fout<<spe[i]<<std::endl;
    // }
    // fout.close();

    for(int i=0;i<s_.size();i++) 
    {
        std::array<double,3> t;
        t[0]=s_[i][0]/2.355;
        t[1]=s_[i][1]/2.355;
        t[2]=s_[i][2]/2.355;
        sigma.push_back(t);
        
        t[0]=t[1]=t[2]=1e-10;
        gamma.push_back(t);
    }

    intensity=a_;
    //Rememver that our Voigt function does not include normalization factor
    //We also need to inverse after fitting (see function run()).
    if(peak_shape==voigt_type) 
    {
        for(int i=0;i<a_.size();i++)
        {
            for(int k=0;k<3;k++)
            {
                intensity[i]/=voigt(0,sigma[i][k],gamma[i][k]);
            }
        }
    }

    original_ratio.resize(x.size(),0);
    for (int i = 0; i < x.size(); i++)
    {
        original_ratio[i] = spe[round(x[i][2]) * dims[0] * dims[1] + round(x[i][1]) * dims[0] + round(x[i][0])];
    }
    return true;
}


bool gaussian_fit_3d::run()
{
    bool b;
    if(x.size()==1)
    {
        b=run_single_peak();
    }
    else
    {
        limit_fitting_region_of_each_peak();
        b=run_multiple_peaks();   //run mixed Gaussian algorithm to decovolute peaks.
    }

    if (b == true && peak_shape == voigt_type)
    {
        //convert amplitude
        for (int i = 0; i < intensity.size(); i++)
        {
            for (int k = 0; k < 3; k++)
            {
                intensity[i] *= voigt(0, sigma[i][k], gamma[i][k]);
            }
        }
    }
    return b;
};

bool gaussian_fit_3d::run_multiple_peaks()
{
    std::vector<std::vector<float> > analytical_spectra;    //analytical spectrum for each peak
    std::vector<float> peaks_total;                        //reconstructed 2D matrix from fitting: sum of analytical_spectra
    
    int npeak=intensity.size();
    to_remove.resize(npeak,0);
    numerical_sum.resize(npeak,0.0);
    err.resize(npeak,0.0);

    bool flag_break=false;
    std::vector< std::vector<std::array<double,3> > > x_old;

    for(int loop=0;loop<rmax;loop++)
    {
        analytical_spectra.clear();
        analytical_spectra.resize(npeak);

        peaks_total.clear();
        peaks_total.resize(dims[0]*dims[1]*dims[2],0.0);

        std::array<int,6> region={0,dims[0],0,dims[1],0,dims[2]};
       
        // #pragma omp parallel for
        for (unsigned int index = 0; index < npeak; index++)
        {
            if(to_remove[index]==1) //peak has been removed. 
            {
                analytical_spectra[index].clear();
                continue;   
            }

            int i0,i1,j0,j1,k0,k1;
            
            if (peak_shape == gaussian_type)
            {
                ldw_math_3d::gaussian_convolution(intensity[index],x[index][0],x[index][1],x[index][2],sigma[index][0],sigma[index][1],sigma[index][2],dims[0],dims[1],dims[2],analytical_spectra[index],i0,i1,j0,j1,k0,k1,region,2.0);
            }
            else if (peak_shape == voigt_type)
            {
                ldw_math_3d::voigt_convolution(intensity[index],x[index][0],x[index][1],x[index][2],sigma[index][0],sigma[index][1],sigma[index][2],gamma[index][0],gamma[index][1],gamma[index][2],dims[0],dims[1],dims[2],analytical_spectra[index],i0,i1,j0,j1,k0,k1,region,2.0);
            }
            // #pragma omp critical
            {
                for(int k=k0;k<k1;k++)
                for(int j=j0;j<j1;j++)
                for(int i=i0;i<i1;i++)
                {
                    peaks_total[k*dims[0]*dims[1]+j*dims[0]+i]+=analytical_spectra[index][(k - k0) * (i1 - i0) * (j1 - j0) + (j - j0) * (i1 - i0) + (i - i0)];    
                }
            }
        }
        x_old.push_back(x);

        std::vector<int> peak_remove_flag;
        peak_remove_flag.resize(x.size(),0);

        #pragma omp parallel for
        for (int index = 0; index < x.size(); index++)
        {
            if(to_remove[index]==1) continue;  //peak has been removed. 
           
            std::vector<double> zz;
            double total_z = 0.0;

            int i0,i1,j0,j1,k0,k1;
            
            if (peak_shape == gaussian_type)
            {
                ldw_math_3d::gaussian_convolution(intensity[index],x[index][0],x[index][1],x[index][2],sigma[index][0],sigma[index][1],sigma[index][2],dims[0],dims[1],dims[2],analytical_spectra[index],i0,i1,j0,j1,k0,k1,valid_fit_region.at(index),2.0);
            }
            else if (peak_shape == voigt_type)
            {
                ldw_math_3d::voigt_convolution(intensity[index],x[index][0],x[index][1],x[index][2],sigma[index][0],sigma[index][1],sigma[index][2],gamma[index][0],gamma[index][1],gamma[index][2],dims[0],dims[1],dims[2],analytical_spectra[index],i0,i1,j0,j1,k0,k1,valid_fit_region.at(index), 2.0);
            }

            for (int k =k0;k<k1;k++)
            for (int j = j0; j < j1; j++)
            for (int i = i0; i < i1; i++)
            {
                    double inten1 = analytical_spectra[index][(k - k0) * (i1 - i0) * (j1 - j0) + (j - j0) * (i1 - i0) + (i - i0)];
                    double inten2 = peaks_total[k*dims[0]*dims[1]+j*dims[0]+i];
                    double scale_factor;
                    if (fabs(inten2) > 1e-100)
                        scale_factor = inten1 / inten2;
                    else
                        scale_factor = 0.0;

                    scale_factor = std::min(scale_factor, 1.0);
                    double temp = scale_factor * spe[k*dims[0]*dims[1]+j*dims[0]+i];
                    zz.push_back(temp);

                    // std::cout<<temp<<" ";
                    total_z += temp;
                
                // std::cout<<std::endl;
            }
            numerical_sum[index] = total_z;
            // std::cout<<std::endl;
            
            //std::cout <<"Before " <<loop<<" "<< original_ndx[i] << " " << x.at(i) << " " << y.at(i) << " " << a[i][0] << " " << sigmax.at(i) << " " << sigmay.at(i)<< " " << gammax.at(i) << " " << gammay.at(i) << " " << total_z << std::endl;
            std::array<int,3> current_dims={i1-i0,j1-j0,k1-k0};
            std::array<double,3> current_x={x[index][0]-i0,x[index][1]-j0,x[index][2]-k0};
            if (peak_shape == gaussian_type)
            {
                if(cannot_move[index]==1)
                {
                    one_fit_gaussian(zz.data(),current_dims,current_x,sigma[index],intensity[index],err[index]);
                }
                else
                {
                    one_fit_gaussian(zz.data(),current_dims,current_x,sigma[index],intensity[index],err[index]);
                }
            }
            else if (peak_shape == voigt_type)
            {
                if(cannot_move[index]==1)
                {
                    one_fit_voigt(zz.data(),current_dims,current_x,sigma[index],gamma[index],intensity[index],err[index],loop);  
                }
                else
                {
                    one_fit_voigt(zz.data(),current_dims,current_x,sigma[index],gamma[index],intensity[index],err[index],loop);  
                }
            }

            //debug information
            // std::cout<<"Loop="<<loop<<" Peak "<<index<<std::endl;
            // std::cout<<current_x[0]+i0+start_pos[0]<<" ";
            // std::cout<<current_x[1]+j0+start_pos[1]<<" ";
            // std::cout<<current_x[2]+k0+start_pos[2]<<" ";
            // std::cout<<std::endl;
            

            if (std::min(std::min(sigma[index][0] + gamma[index][0], sigma[index][1] + gamma[index][1]), sigma[index][2] + gamma[index][2]) < 0.5)
            {
                std::cout << "Peak " << index << " will be removed because it becomes too narrow" << std::endl;
                to_remove[index] = 1;
            }
            if (current_x[0] < 0 || current_x[0] > current_dims[0] || current_x[1] < 0 || current_x[1] > current_dims[1] || current_x[2] < 0 || current_x[2] > current_dims[2])
            {
                std::cout << "Peak " << index << " will be removed because it moves out of fitting area" << std::endl;
                to_remove[index] = 1;
            }
            if (current_x[0]+i0 < valid_fit_region[index][0] || current_x[0]+i0 > valid_fit_region[index][1]-1 
                || current_x[1]+j0 < valid_fit_region[index][2] || current_x[1]+j0 > valid_fit_region[index][3]-1 
                || current_x[2]+k0 < valid_fit_region[index][4] || current_x[2]+k0 > valid_fit_region[index][5]-1)
            {
                std::cout << "Peak " << index << " will be removed because it moves out of valid fitting region" << std::endl;
                to_remove[index] = 1;
            }
            x[index][0]=current_x[0]+i0;
            x[index][1]=current_x[1]+j0;
            x[index][2]=current_x[2]+k0;

        } //end of parallel for(int i = 0; i < npeak; i++)

        //todo: two peak becomes one ?



        //remove peaks that moved too much!!
        {
            for(int i = 0; i < x.size(); i++)
            {
                if(to_remove[i]==1) continue;
                double current_ratio=spe[round(x[i][2])*dims[0]*dims[1]+round(x[i][1])*dims[0]+round(x[i][0])];
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

        bool bcon = false;
        for (int i = x_old.size() - 1; i >= std::max(int(x_old.size()) - 2, 0); i--)
        {
            bool b = true;
            for (int j = 0; j < npeak; j++)
            {
                if(to_remove[j]==1) continue;
                if(cannot_move[j]==1) continue;
                for (int k = 0; k < 3; k++)
                {
                    if (fabs(x[j][k] - x_old[i][j][k]) > 0.01)
                    {
                        b = false;
                        break;
                    }
                }
            }
            if (b == true)
            {
                bcon = true;
                break;
            }
        }

        if (bcon == true)
        {
            flag_break = true;
        }
        // std::cout << "\r" << "Iteration " << loop + 1 << std::flush;
        std::cout << "\r" << "Iteration " << loop + 1 << "    " << std::flush;
        
    }
    return true;
}

bool gaussian_fit_3d::run_single_peak()
{   
    bool b=false;
    double total_z = 0.0;
    double e;

    if (peak_shape == gaussian_type) //gaussian
    {
        b=one_fit_gaussian(spe.data(),dims,x[0],sigma[0],intensity[0],e);
    }
    else
    {
        b=one_fit_voigt(spe.data(),dims,x[0],sigma[0],gamma[0],intensity[0],e,0);    
    }

    to_remove.resize(1,0);
    numerical_sum.resize(1,0.0);

    if(b==false)
    {
        to_remove[0]=1;
    }

    
    return b;
}

bool gaussian_fit_3d::find_highest_neighbor(int xx,int yy,int zz, int &ii,int &jj, int &kk)
{
    bool b_already_at_max=true;
    double current_a=spe[zz*dims[0]*dims[1]+yy*dims[0]+xx];
    double a_difference=0.0;
    ii=0;
    jj=0;
    kk=0;

    for (int k = -1; k <= 1; k++)
    {
        if (zz+k<0 || zz+k>dims[2]-1) continue;
        for (int i = -1; i <= 1; i++)
        {
            if (xx+i<0 || xx+i>dims[0]-1) continue;
            for (int j = -1; j <= 1; j++)
            {
                if (yy+j<0 || yy+j>dims[1]-1) continue;
                if (spe[(zz+k) * dims[0] * dims[1] + (yy+j) * dims[0] + xx+i] - current_a > a_difference)
                {
                    a_difference = spe[(zz+k) * dims[0] * dims[1] + (yy+j) * dims[0] + xx+i] - current_a;
                    ii = i;
                    jj = j;
                    kk = k;
                    b_already_at_max = false;
                }
            }
        }
    }

    return b_already_at_max;
}

bool gaussian_fit_3d::limit_fitting_region_of_each_peak()
{
    int npeak=x.size();
    for(int ndx=0;ndx<npeak;ndx++)
    {
        int xx=round(x[ndx][0]);
        int yy=round(x[ndx][1]);
        int zz=round(x[ndx][2]);
        int m,n,k;

        std::array<int,6> region={0,dims[0],0,dims[1],0,dims[2]};
        find_highest_neighbor(xx,yy,zz,m,n,k);xx+=m;yy+=n;zz+=k;
        find_highest_neighbor(xx,yy,zz,m,n,k);xx+=m;yy+=n;zz+=k; //move the local  maximal
        if (find_highest_neighbor(xx, yy, zz, m, n, k) == false) //this is a shoulder peak, restore initial coordinate
        {
            xx=round(x[ndx][0]);
            yy=round(x[ndx][1]);
            zz=round(x[ndx][2]);
        }

        double current_a =spe[zz*dims[0]*dims[1]+yy*dims[0]+xx];
        bool b;

        //x direction
        b=true;
        for (int i = xx - 1; i >= std::max(0, xx - int(median_widths[0]*2)); i--)
        {
            region[0] = i + 1;
            if (spe[zz*dims[0]*dims[1]+yy*dims[0]+i] > spe[zz*dims[0]*dims[1]+yy*dims[0]+i+1] 
                && spe[zz*dims[0]*dims[1]+yy*dims[0]+i+2] > spe[zz*dims[0]*dims[1]+yy*dims[0]+i+1]) 
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
        for (int i = xx + 1; i < std::min(dims[0], xx + int(median_widths[0]*2)); i++)
        {
            region[1] = i;
            if (spe[zz*dims[0]*dims[1]+yy*dims[0]+i] > spe[zz*dims[0]*dims[1]+yy*dims[0]+i-1] 
                && spe[zz*dims[0]*dims[1]+yy*dims[0]+i-2] > spe[zz*dims[0]*dims[1]+yy*dims[0]+i-1]) 
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[1]=dims[0];
        }

        //y direction
        b=true;
        for (int i = yy - 1; i >= std::max(0, yy - int(median_widths[1]*2)); i--)
        {
            region[2] = i + 1;
            if (spe[zz*dims[0]*dims[1]+i*dims[0]+xx] > spe[zz*dims[0]*dims[1]+(i+1)*dims[0]+xx] 
                && spe[zz*dims[0]*dims[1]+(i+2)*dims[0]+xx] > spe[zz*dims[0]*dims[1]+(i+1)*dims[0]+xx]) 
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
        for (int i = yy + 1; i < std::min(dims[1], yy + int(median_widths[1]*2)); i++)
        {
            region[3] = i;
            if (spe[zz*dims[0]*dims[1]+i*dims[0]+xx] > spe[zz*dims[0]*dims[1]+(i-1)*dims[0]+xx] 
                && spe[zz*dims[0]*dims[1]+(i-2)*dims[0]+xx] > spe[zz*dims[0]*dims[1]+(i-1)*dims[0]+xx]) 
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[3]=dims[1];
        }

        //z direction
        b=true;
        for (int i = zz - 1; i >= std::max(0, zz - int(median_widths[2]*2)); i--)
        {
            region[4] = i + 1;
            if (spe[i*dims[0]*dims[1]+(yy)*dims[0]+xx] > spe[(i+1)*dims[0]*dims[1]+yy*dims[0]+xx] 
                && spe[(i+2)*dims[0]*dims[1]+yy*dims[0]+xx] > spe[(i+1)*dims[0]*dims[1]+yy*dims[0]+xx]) 
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[4]=0;
        }

        b=true;
        for (int i = zz + 1; i < std::min(dims[2], zz + int(median_widths[2]*2)); i++)
        {
            region[5] = i;
            if (spe[i*dims[0]*dims[1]+yy*dims[0]+xx] > spe[(i-1)*dims[0]*dims[1]+yy*dims[0]+xx] 
                && spe[(i-2)*dims[0]*dims[1]+yy*dims[0]+xx] > spe[(i-1)*dims[0]*dims[1]+yy*dims[0]+xx]) 
            {
                b=false;
                break;
            }
        }
        if(b)
        {
            region[5]=dims[2];
        }


        //expand by 1 point
        if(region[0]>0) region[0]-=1;
        if(region[2]>0) region[2]-=1;
        if(region[4]>0) region[4]-=1;
        if(region[1]<dims[0]-1) region[1]+=1;
        if(region[3]<dims[1]-1) region[3]+=1;
        if(region[5]<dims[2]-1) region[5]+=1;
        valid_fit_region.push_back(region);

    }
    return true;
};



bool gaussian_fit_3d::one_fit_gaussian(double *data, std::array<int,3> dims, std::array<double,3> &coors, std::array<double,3> &sigmas,double &inten, double &e)
{
    ceres::Solver::Summary summary;
    ceres::Problem problem;

    int ndata=dims[0]*dims[1]*dims[2];
    for(int k=0;k<3;k++) sigmas[k]=2*sigmas[k]*sigmas[k]; //in mycostfunction_gaussian_3d, we are using 2*sigma*sigma

    mycostfunction_gaussian_3d *cost_function = new mycostfunction_gaussian_3d(dims[0],dims[1],dims[2],data);
    cost_function->set_n_residuals(ndata);
    for(int m=0;m<7;m++) cost_function->parameter_block_sizes()->push_back(1); 
    problem.AddResidualBlock(cost_function, NULL, &inten,&coors[0],&sigmas[0],&coors[1],&sigmas[1],&coors[2],&sigmas[2]);
    ceres::Solve(options, &problem, &summary);
    e = sqrt(summary.final_cost / ndata);

    //restore sigma from 2*sigma*sigma
    for(int k=0;k<3;k++) {
        sigmas[k]=sqrt(fabs(sigmas[k])/2);
    }

    return true;
};

bool gaussian_fit_3d::one_fit_voigt(double *data, std::array<int,3> dims, std::array<double,3> &coors, std::array<double,3> &sigmas,std::array<double,3> &gammas,double &inten, double &e, int loop)
{

    // std::ofstream fout("peak1.txt",std::ofstream::app);
    // for(int i=0;i<dims[0]*dims[1]*dims[2];i++)
    // {
    //     fout<<data[i]<<std::endl;
    // }
    // fout.close();

    if(loop==0)
    {
        //run gaussian fit first, then add Voigt term gamma
        //conversion from voigt to gaussian, firstly make a voigt with 0 gammay, then convert to unit in Gaussian fitting
        for(int k=0;k<3;k++)
        {
            inten*=voigt(0.0,sigmas[k],gammas[k]);
            sigmas[k]=(0.5346*gammas[k]*2+std::sqrt(0.2166*4*gammas[k]*gammas[k]+sigmas[k]*sigmas[k]*8*0.6931))/2.355;
        }
        
        //gaussian fit
        one_fit_gaussian(data,dims,coors,sigmas,inten,e);
        
        //convert back to voigt representation, suppose gammay --> 0        
        for(int k=0;k<3;k++)
        {
            gammas[k]=1e-8;
            inten/=voigt(0.0,sigmas[k],gammas[k]);
        }
    }

    ceres::Solver::Summary summary;
    ceres::Problem problem;

    int ndata=dims[0]*dims[1]*dims[2];

    voigt_fit_3d *cost_function = new voigt_fit_3d(dims[0],dims[1],dims[2],data);
    cost_function->set_n_residuals(ndata);
    for(int m=0;m<10;m++) cost_function->parameter_block_sizes()->push_back(1); 
    problem.AddResidualBlock(cost_function, NULL, &inten,&coors[0],&sigmas[0],&gammas[0],&coors[1],&sigmas[1],&gammas[1],&coors[2],&sigmas[2],&gammas[2]);
    ceres::Solve(options, &problem, &summary);
    e = sqrt(summary.final_cost / ndata);

    for(int k=0;k<3;k++)
    {
        sigmas[k]=fabs(sigmas[k]);
        gammas[k]=fabs(gammas[k]);
    }

    return true;
};

//cost functions: 3D Gaussian and 3D Voigt
//Gaussian
mycostfunction_gaussian_3d::~mycostfunction_gaussian_3d(){};
mycostfunction_gaussian_3d::mycostfunction_gaussian_3d(int xdim, int ydim, int zdim, double *data_)
{
    nx=xdim;
    ny=ydim;
    nz=zdim;
    data=data_;   
};


bool mycostfunction_gaussian_3d::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a=xx[0][0];
    double x0=xx[1][0];
    double sigmax=fabs(xx[2][0]);
    double y0=xx[3][0];
    double sigmay=fabs(xx[4][0]);
    double z0=xx[5][0];
    double sigmaz=fabs(xx[6][0]);

    if (jaco != NULL ) //both residual errors and jaco are required.
    {
        int i=0;
        for(int iz=0;iz<nz;iz++)
        for(int iy=0;iy<ny;iy++)
        for(int ix=0;ix<nx;ix++)
        {
            double x_sigmax=(ix-x0)/sigmax;
            double y_sigmay=(iy-y0)/sigmay;
            double z_sigmaz=(iz-z0)/sigmaz;
            
            double g=exp((x0-ix)*x_sigmax+(y0-iy)*y_sigmay+(z0-iz)*z_sigmaz);
            double ag=a*g;

            residual[i]=ag-data[i]; 
            jaco[0][i]=g; //with respect to a
            jaco[1][i]=ag*2*x_sigmax; //x0
            jaco[2][i]=ag*x_sigmax*x_sigmax; //sigmax
            jaco[3][i]=ag*2*y_sigmay; //y0
            jaco[4][i]=ag*y_sigmay*y_sigmay; //sigmay
            jaco[5][i]=ag*2*z_sigmaz; //z0
            jaco[6][i]=ag*z_sigmaz*z_sigmaz; //sigmaz
            ++i;
        }
    }
    else //only require residual errors
    {
        int i=0;
        for(int iz=0;iz<nz;iz++)
        for(int iy=0;iy<ny;iy++)
        for(int ix=0;ix<nx;ix++)
        { 
            residual[i]=a*exp(-(ix-x0)*(ix-x0)/sigmax-(iy-y0)*(iy-y0)/sigmay-(iz-z0)*(iz-z0)/sigmaz)-data[i];  
            ++i;
        }
    }
    return true;
};

#ifndef LDW_PARAS
    #define LDW_PARAS
    #define SMALL 1e-10
    #define PI 3.14159265358979323846
    #define M_SQRT_PI 1.772453850905516
    #define M_SQRT_2PI 2.506628274631000
    #define M_1_SQRT_PI 0.564189583547756
#endif

//voigt
voigt_fit_3d::~voigt_fit_3d(){};
voigt_fit_3d::voigt_fit_3d(int nx_, int ny_, int nz_, double *data_)
{
    nx=nx_;
    ny=ny_;
    nz=nz_;
    data=data_;
    n_datapoint=nx*ny*nz;
};

void voigt_fit_3d::voigt_helper(const double x0, const double sigma, const double gamma, double *vv, double *r_x0, double *r_sigma, double *r_gamma) const
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

void voigt_fit_3d::voigt_helper(const double x0, const double sigma, const double gamma, double *vv) const
{
    double v,l;
    double z_r=x0/(sqrt(2)*sigma);
    double z_i=gamma/(sqrt(2)*sigma);
    double sigma2=sigma*sigma;

    re_im_w_of_z(z_r,z_i,&v,&l);
    *vv=v/sqrt(2*M_PI*sigma2);
    return;
};

bool voigt_fit_3d::Evaluate(double const *const *xx, double *residual, double **jaco) const
{
    double a=xx[0][0];
    double x0=xx[1][0];
    double sigmax=fabs(xx[2][0]);
    double gammax=fabs(xx[3][0]);
    double y0=xx[4][0];
    double sigmay=fabs(xx[5][0]);
    double gammay=fabs(xx[6][0]);
    double z0=xx[7][0];
    double sigmaz=fabs(xx[8][0]);
    double gammaz=fabs(xx[9][0]);
    
    
    double vvx,vvy,vvz,r_x,r_y,r_z,r_sigmax,r_sigmay,r_sigmaz,r_gammax,r_gammay,r_gammaz;

    voigt_helper(x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
    voigt_helper(y0, sigmay, gammay, &vvy, &r_y, &r_sigmay, &r_gammay);
    voigt_helper(z0, sigmaz, gammaz, &vvz, &r_z, &r_sigmaz, &r_gammaz);

    if (jaco != NULL ) //both residual errors and jaco are required.
    {
        int i = 0;
        for (int iz = 0; iz < nz; iz++)
        {
            voigt_helper(iz - z0, sigmaz, gammaz, &vvz, &r_z, &r_sigmaz, &r_gammaz);
            for (int iy = 0; iy < ny; iy++)
            {
                voigt_helper(iy - y0, sigmay, gammay, &vvy, &r_y, &r_sigmay, &r_gammay);
                for (int ix = 0; ix < nx; ix++)
                {
                    voigt_helper(ix - x0, sigmax, gammax, &vvx, &r_x, &r_sigmax, &r_gammax);
                    residual[i] = a * vvx * vvy * vvz - data[i];
                    jaco[0][i] = vvx * vvy *vvz;           //with respect to a
                    jaco[1][i] = -a * r_x * vvy * vvz;     //x0
                    jaco[2][i] = a * r_sigmax * vvy * vvz; //sigmax
                    jaco[3][i] = a * r_gammax * vvy * vvz; //gammax
                    jaco[4][i] = -a * r_y * vvx * vvz;     //y0
                    jaco[5][i] = a * r_sigmay * vvx * vvz; //sigmay
                    jaco[6][i] = a * r_gammay * vvx * vvz; //gammay
                    jaco[7][i] = -a * r_z * vvx * vvy;     //z0
                    jaco[8][i] = a * r_sigmaz * vvx * vvy; //sigmaz
                    jaco[9][i] = a * r_gammaz * vvx * vvy; //gammaz
                    ++i;
                }
            }
        }
    }
    else //only require residual errors
    {
        int i = 0;
        for (int iz = 0; iz < nz; iz++)
        {
            voigt_helper(iz - z0, sigmaz, gammaz, &vvz);
            for (int iy = 0; iy < ny; iy++)
            {
                voigt_helper(iy - y0, sigmay, gammay, &vvy);
                for (int ix = 0; ix < nx; ix++)
                {
                    voigt_helper(ix - x0, sigmax, gammax, &vvx);
                    residual[i] = a * vvx * vvy * vvz - data[i];
                    ++i;
                }
            }
        }
    }
    return true;
};
