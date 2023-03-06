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

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "spline.h"
#include "json/json.h"
#include "commandline.h"
#include "dnn_picker.h"
#include "spectrum_io.h"
#include "spectrum_io_1d.h"
// #include "spectrum_picking_1d.h"
#include "spectrum_io_3d.h"


extern "C"  
{
    double voigt(double x, double sigma, double gamma);
};

namespace ldw_math_3d
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


    bool voigt_convolution(double a, double x, double y, double z, double sigmax, double sigmay, double sigmaz, double gammax, double gammay, double gammaz, int xdim, int ydim,int zdim, std::vector<float> &kernel, int &i0, int &i1, int &j0, int &j1, int &k0, int &k1,std::array<int,6> region,double scale)
    {
        float wx=(1.0692*gammax+sqrt(0.8664*gammax*gammax+5.5452*sigmax*sigmax))*scale;
        float wy=(1.0692*gammay+sqrt(0.8664*gammay*gammay+5.5452*sigmay*sigmay))*scale;
        float wz=(1.0692*gammaz+sqrt(0.8664*gammaz*gammaz+5.5452*sigmaz*sigmaz))*scale;
        
        i0=std::max(std::max(0,int(x-wx+0.5)),region[0]);
        i1=std::min(std::min(xdim,int(x+wx+0.5)),region[1]);
        j0=std::max(std::max(0,int(y-wy+0.5)),region[2]);
        j1=std::min(std::min(ydim,int(y+wy+0.5)),region[3]);
        k0=std::max(std::max(0,int(z-wz+0.5)),region[4]);
        k1=std::min(std::min(zdim,int(z+wz+0.5)),region[5]);


        kernel.clear();
        kernel.resize((i1-i0)*(j1-j0)*(k1-k0));

        for (int k = k0; k < k1; k++)
        {
            double z3 = voigt(k - z, sigmaz, gammaz);
            for (int j = j0; j < j1; j++)
            {
                double z2 = voigt(j - y, sigmay, gammay);
                for (int i = i0; i < i1; i++)
                {
                    double z1 = voigt(i - x, sigmax, gammax);
                    kernel.at((k - k0) * (i1 - i0) * (j1 - j0) + (j - j0) * (i1 - i0) + (i - i0)) = a * z1 * z2 * z3;
                }
            }
        }
        return true;
    };

    
    bool gaussian_convolution(double a, double x, double y, double z, double sigmax, double sigmay, double sigmaz, int xdim, int ydim,int zdim, std::vector<float> &kernel, int &i0, int &i1, int &j0, int &j1, int &k0, int &k1,std::array<int,6> region, double scale)
    {
        float wx=2.3548*sigmax*scale;
        float wy=2.3548*sigmay*scale;
        float wz=2.3548*sigmaz*scale;
        
        i0=std::max(std::max(0,int(x-wx+0.5)),region[0]);
        i1=std::min(std::min(xdim,int(x+wx+0.5)),region[1]);
        j0=std::max(std::max(0,int(y-wy+0.5)),region[2]);
        j1=std::min(std::min(ydim,int(y+wy+0.5)),region[3]);
        k0=std::max(std::max(0,int(z-wz+0.5)),region[4]);
        k1=std::min(std::min(zdim,int(z+wz+0.5)),region[5]);


        kernel.clear();
        kernel.resize((i1-i0)*(j1-j0)*(k1-k0));

        for (int k = k0; k < k1; k++)
        {
            double z3 = exp(-(k - z) * (k - z) / (2 * sigmaz * sigmaz));
            for (int j = j0; j < j1; j++)
            {
                double z2 = exp(-(j - y) * (j - y) / (2 * sigmay * sigmay));
                for (int i = i0; i < i1; i++)
                {
                    double z1 = exp(-(i - x) * (i - x) / (2 * sigmax * sigmax));
                    kernel.at((k - k0) * (i1 - i0) * (j1 - j0) + (j - j0) * (i1 - i0) + (i - i0) ) = a * z1 * z2 * z3;
                }
            }
        }
        return true;
    };

    bool voigt_convolution_region(double x, double y, double z, double sigmax, double sigmay, double sigmaz, double gammax, double gammay, double gammaz, int xdim, int ydim,int zdim, int &i0, int &i1, int &j0, int &j1, int &k0, int &k1)
    {
        float wx=(1.0692*gammax+sqrt(0.8664*gammax*gammax+5.5452*sigmax*sigmax))*1.5f;
        float wy=(1.0692*gammay+sqrt(0.8664*gammay*gammay+5.5452*sigmay*sigmay))*1.5f;
        float wz=(1.0692*gammaz+sqrt(0.8664*gammaz*gammaz+5.5452*sigmaz*sigmaz))*1.5f;
        
        i0=std::max(0,int(x-wx+0.5));
        i1=std::min(xdim,int(x+wx+0.5));
        j0=std::max(0,int(y-wy+0.5));
        j1=std::min(ydim,int(y+wy+0.5));
        k0=std::max(0,int(z-wz+0.5));
        k1=std::min(zdim,int(z+wz+0.5));
        return true;
    };

    bool gaussian_convolution_region(double x, double y, double z, double sigmax, double sigmay, double sigmaz, int xdim, int ydim,int zdim, int &i0, int &i1, int &j0, int &j1, int &k0, int &k1)
    {
        float wx=2.3548*sigmax*1.5f;
        float wy=2.3548*sigmay*1.5f;
        float wz=2.3548*sigmaz*1.5f;
        
        i0=std::max(0,int(x-wx+0.5));
        i1=std::min(xdim,int(x+wx+0.5));
        j0=std::max(0,int(y-wy+0.5));
        j1=std::min(ydim,int(y+wy+0.5));
        k0=std::max(0,int(z-wz+0.5));
        k1=std::min(zdim,int(z+wz+0.5));
        return true;
    };



    bool calcualte_principal_axis(std::vector<float> data, int xdim, int ydim, int zdim)
    {
        float sx,sy,sz,ss;
        float xx,yy,zz,xy,yz,zx;

        sx=sy=sz=ss=0.0f;
        for(int k=0;k<zdim;k++)
        for(int i=0;i<xdim;i++)
        for(int j=0;j<ydim;j++)
        {
            ss+=data[k*xdim*ydim+i*ydim+j];
            sx+=data[k*xdim*ydim+i*ydim+j]*i;
            sy+=data[k*xdim*ydim+i*ydim+j]*j;
            sz+=data[k*xdim*ydim+i*ydim+j]*k;
        }

        sx/=ss;
        sy/=ss;
        sz/=ss; //center of mass
    
        xx=0.0f;
        yy=0.0f;
        zz=0.0f;
        xy=0.0f;
        yz=0.0f;
        zx=0.0f;

        for (int k = 0; k < zdim; k++)
            for (int i = 0; i < xdim; i++)
                for (int j = 0; j < ydim; j++)
                {
                    float da = data[k * xdim * ydim + i * ydim + j];
                    xx += (i - sx) * (i - sx) * da;
                    yy += (j - sy) * (j - sy) * da;
                    zz += (k - sz) * (k - sz) * da;
                    xy += (i - sx) * (j - sy) * da;
                    yz += (k - sz) * (j - sy) * da;
                    zx += (k - sz) * (i - sx) * da;
                }


        Eigen::Matrix3f covariance_tensor;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eig;
        Eigen::Vector3f values;
        Eigen::Matrix3f evecs;

        covariance_tensor(0,0)=xx/ss;
        covariance_tensor(1,1)=yy/ss;
        covariance_tensor(2,2)=zz/ss;
        
        covariance_tensor(0,1)=xy/ss;
        covariance_tensor(1,0)=xy/ss;

        covariance_tensor(0,2)=zx/ss;
        covariance_tensor(2,0)=zx/ss;

        covariance_tensor(1,2)=yz/ss;
        covariance_tensor(2,1)=yz/ss;

        
        eig.compute(covariance_tensor);
        values = eig.eigenvalues();
        evecs = eig.eigenvectors();

        bool b_tilt=false;
        for(int i=0;i<3;i++)
        {
            double tmax=0.0;
            for(int j=0;j<3;j++)
            {
                if(fabs(evecs(i,j))>tmax)
                {
                    tmax=fabs(evecs(i,j));
                }
            }
            std::cout<<" "<<tmax;
            if(tmax<0.96) b_tilt=true;
        }
        std::cout<<std::endl;

        return b_tilt;
    };

    
    double interp2_point(int min_x, int max_x, int min_y,int max_y, std::vector<double> data, double x,double y)
    {
        //data[i*ny+j]
        int ny=max_y-min_y+1;
        std::vector<double> spe_at_y; 

        std::vector<double> x_input,y_input;
        for(int j=min_y;j<=max_y;j++) y_input.push_back(j);
        for(int i=min_x;i<=max_x;i++) x_input.push_back(i);
        
        for(int i=min_x;i<=max_x;i++)
        {
            std::vector<double> tdata(data.begin()+(i-min_x)*ny,data.begin()+(i-min_x+1)*ny);
            tk::spline st(y_input,tdata);  
            spe_at_y.push_back(st(y));
        }

        tk::spline st(x_input,spe_at_y);  
        return st(x);
    };
    

    bool interp2(int min_x, int max_x, int min_y,int max_y, std::vector<double> data, std::vector<double> x,std::vector<double> y,std::vector<double> &line_v)
    {
        //data[i*ny+j]
        int ndata=x.size();
        int ny=max_y-min_y+1;
        int nx=max_x-min_x+1;
        
        std::vector< std::vector<double> > spe_at_y_bycol; //nx by ndata

        std::vector<double> x_input,y_input;
        for(int j=min_y;j<=max_y;j++) y_input.push_back(j);
        for (int i = min_x; i <= max_x; i++) x_input.push_back(i);

        for (int i = min_x; i <= max_x; i++)
        {
            std::vector<double> t;
            std::vector<double> tdata(data.begin() + (i - min_x) * ny, data.begin() + (i - min_x + 1) * ny);
            tk::spline st(y_input, tdata);
            for (int m = 0; m < y.size(); m++)
            {
                t.push_back(st(y[m]));
            }
            spe_at_y_bycol.push_back(t);
        }

        std::vector< std::vector<double> > spe_at_y_byrow(ndata,std::vector<double>(nx,0.0));
        for(int i=0;i<nx;i++)
        {
            for(int j=0;j<y.size();j++)
            {
                spe_at_y_byrow[j][i]= spe_at_y_bycol[i][j];  
            }
        }

        for(int i=0;i<y.size();i++)
        {
            tk::spline st(x_input,spe_at_y_byrow[i]);       
            line_v.push_back(st(x[i]));
        }

        return true;
    };

    std::vector<int> find_neighboring_peaks(std::vector< std::array<double,3> > peaks_3d_pos,std::vector<double> inten,int p)
    {
        std::vector<int> ndx;
        for(int i=0;i<peaks_3d_pos.size();i++)
        {
            if(i==p) continue;
            double d1=peaks_3d_pos[i][0]-peaks_3d_pos[p][0];
            double d2=peaks_3d_pos[i][1]-peaks_3d_pos[p][1];
            double d3=peaks_3d_pos[i][2]-peaks_3d_pos[p][2];
            
            //neighborhood and same sign
            if(sqrt(d1*d1+d2*d2+d3*d3)<30.0 && inten[i]*inten[p]>0.0)
            {
                ndx.push_back(i);
            }
        }  
        return ndx; 
    };

    std::vector<std::deque<int> > bread_first(std::vector<int> &neighbor, int n)
    {
        std::vector<std::deque<int> > clusters;
        std::deque<int> work, work2;
        std::vector<int> used;

        used.resize(n, 0);

        for (int i = 0; i < n; i++)
        {
            if (used.at(i) != 0)
            {
                continue;
            }

            used.at(i) = 1;
            work.clear();
            work2.clear();
            work.push_back(i);
            work2.push_back(i);

            while (!work.empty())
            {
                int c = work.at(0);
                work.pop_front();

                for (int j = 0; j < n; j++)
                {
                    if (j == c || used.at(j) != 0)
                    {
                        continue;
                    }
                    if (neighbor[j * n + c] == 1)
                    {
                        #pragma omp critical
                        {
                            work.push_back(j);
                            work2.push_back(j);
                        }
                        used.at(j) = 1;
                    }
                }
            }
            if (work2.size() >= 1)
            {
                clusters.push_back(work2);
            }
        }
        return clusters;
    };

    int find_best_from_peaks(std::vector< std::array<double,3> > x1,std::vector< std::array<double,3> > x2, std::vector<int> &ndxs)
    {
        ndxs.clear();
        int npeak=x1.size();

        //distance matrix of peaks pair
        std::vector<double> distance_matrix(npeak*npeak,0.0);
        std::vector<int> distance_binary_matrix(npeak*npeak,0.0);


        for(int i=0;i<npeak;i++)
        {
            for(int j=i+1;j<npeak;j++)
            {
                double t1=(x1[i][0]-x1[j][0])*(x1[i][0]-x1[j][0])+(x1[i][1]-x1[j][1])*(x1[i][1]-x1[j][1])+(x1[i][2]-x1[j][2])*(x1[i][2]-x1[j][2]);
                double t2=(x2[i][0]-x2[j][0])*(x2[i][0]-x2[j][0])+(x2[i][1]-x2[j][1])*(x2[i][1]-x2[j][1])+(x2[i][2]-x2[j][2])*(x2[i][2]-x2[j][2]);
                double t=sqrt(t1)+sqrt(t2);
                distance_matrix[i*npeak+j]=distance_matrix[i+j*npeak]=t;
                if(t<10.0)
                    distance_binary_matrix[i*npeak+j]=distance_binary_matrix[j*npeak+i]=1;
                else
                    distance_binary_matrix[i*npeak+j]=distance_binary_matrix[j*npeak+i]=0;
            }
        }

        std::vector<std::deque<int> > clusters=bread_first(distance_binary_matrix,npeak); //orphan will be skipped

        for(int i=0;i<clusters.size();i++)
        {
            std::vector<double> total_distance;
            for(int j=0;j<clusters[i].size();j++)
            {
                double tsum=0.0;
                for(int jj=0;jj<clusters[i].size();jj++)
                {
                    if(jj==j) continue;
                    tsum+=distance_matrix[j*npeak+jj];
                }
                total_distance.push_back(tsum);
            }
            int minElementIndex = std::min_element(total_distance.begin(),total_distance.end()) - total_distance.begin();
            ndxs.push_back(clusters[i][minElementIndex]);
        }

        return true;
    };

    bool get_perpendicular_line(std::array<double,3> x,std::array<double,3> x0, 
            std::vector<double> &line_x1, std::vector<double> &line_y1, std::vector<double> &line_z1,
            std::vector<double> &line_x2, std::vector<double> &line_y2, std::vector<double> &line_z2,double step)
    {
        line_x1.clear();
        line_y1.clear();
        line_z1.clear();
        line_x2.clear();
        line_y2.clear();
        line_z2.clear();
        
        double direction1[3],direction2[3];
        
        //direct1 is in x-y plane. 
        direction1[0]=x0[1]-x[1];
        direction1[1]=x[0]-x0[0];
        direction1[2]=0.0; 
        double len=sqrt(direction1[0]*direction1[0]+direction1[1]*direction1[1]);
        if(len<1e-20)  //input direction (x0 --> x) is along Z direction
        {
            direction1[0]=1.0;
            direction1[1]=0.0;   
        }
        else
        {
            direction1[0]/=len;
            direction1[1]/=len;
        }

        //dir2 = input cross dir1. Keep in mind dir1[2]=0
        direction2[0]=-(x[2]-x0[2])*direction1[1];
        direction2[1]=(x[2]-x0[2])*direction1[0];
        direction2[2]=(x[0]-x0[0])*direction1[1]-(x[1]-x0[1])*direction1[0];
        len=sqrt(direction2[0]*direction2[0]+direction2[1]*direction2[1]+direction2[2]*direction2[2]);
        direction2[0]/=len;
        direction2[1]/=len;
        direction2[2]/=len;
        
        for(int m=-22;m<=22;m++)
        {
            double double_m=m*step;
            line_x1.push_back(double_m*direction1[0]+x[0]);
            line_y1.push_back(double_m*direction1[1]+x[1]);
            line_z1.push_back(double_m*direction1[2]+x[2]);
            line_x2.push_back(double_m*direction2[0]+x[0]);
            line_y2.push_back(double_m*direction2[1]+x[1]);
            line_z2.push_back(double_m*direction2[2]+x[2]);
        }
        return true;
    };
}

spectrum_io_3d::spectrum_io_3d()
{
    noise_level=-0.1; //negative nosie means we don't have it yet
    //spectrum range 
    begin[0]=100;
    stop[0]=-100;
    begin[1]=1000;
    stop[1]=-1000; //will be updated once read in spe
    begin[2]=1000;
    stop[2]=-1000;

    mod_selection=1;
    zf=0;

    class_flag=0; //base class io_3d
};

spectrum_io_3d::~spectrum_io_3d()
{
   
};

bool spectrum_io_3d::init_parameters(double u1,double u2,double s,int m,int z,bool b_neg)
{
    user_scale1=u1;
    user_scale2=u2;
    if(s>0.00001)
    {
        noise_level=s;
    }
    mod_selection=m;
    zf=z;
    b_negative=b_neg;
    return true;
}


bool spectrum_io_3d::peak_reading(std::string fname)
{
    std::string line,p;
    std::vector< std::string> ps;
    std::stringstream iss;

    int xpos=-1;
    int apos=-1;
    int sigma_pos=-1;
    int gamma_pos=-1;

    
    std::ifstream fin(fname);
    getline(fin,line);
    iss.str(line);
    while(iss>>p)
    {
        ps.push_back(p);
    }
    
    for(int i=0;i<ps.size();i++)
    {
        if(ps[i]=="F1") {xpos=i;}  //in sparky, w2 is direct dimension
        else if(ps[i]=="intensity") {apos=i;}
        else if(ps[i]=="sigma1") {sigma_pos=i;}   
        else if(ps[i]=="gamma1") {gamma_pos=i;}   
        
    }

    if( xpos==-1 || apos==-1 )
    {
        std::cout<<"One or more required varibles are missing."<<std::endl;
        return false;
    }


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

        std::array<double,3> t;

        t[0]=atof(ps[xpos].c_str());
        t[1]=atof(ps[xpos+1].c_str());
        t[2]=atof(ps[xpos+2].c_str());
        peaks_pos.push_back(t);

        t[0]=std::max(atof(ps[sigma_pos].c_str()),2.0);
        t[1]=std::max(atof(ps[sigma_pos+1].c_str()),2.0);
        t[2]=std::max(atof(ps[sigma_pos+2].c_str()),2.0);
        sigma.push_back(t);

        t[0]=std::max(atof(ps[gamma_pos].c_str()),0.01);
        t[1]=std::max(atof(ps[gamma_pos+1].c_str()),0.01);
        t[2]=std::max(atof(ps[gamma_pos+2].c_str()),0.01);
        gamma.push_back(t);


        intensity.push_back(atof(ps[apos].c_str()));
    }
    std::cout<<"Loaded in "<<intensity.size()<<" peaks."<<std::endl;

    //remove out of bound peaks to be water proof
    for(int i=intensity.size()-1;i>=0;i--)
    {
        if(peaks_pos[i][0]<=1 || peaks_pos[i][0]>=xdim-2 || peaks_pos[i][1]<=1 || peaks_pos[i][1]>=ydim-2 || peaks_pos[i][2]<=1 || peaks_pos[i][2]>=zdim-2)
        {
            intensity.erase(intensity.begin()+i);
            peaks_pos.erase(peaks_pos.begin()+i);
            sigma.erase(sigma.begin()+i);
            gamma.erase(gamma.begin()+i);   
        }
    }
    std::cout<<"After removal of out of bound peaks, there are total "<<intensity.size()<<" peaks."<<std::endl;

    return true;
};