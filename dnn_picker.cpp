
#include <utility> 
#include <vector>
#include <valarray>
#include <deque>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "spline.h"

#include "clique.h"
#include "dnn_picker.h"


extern float ann_data[];
extern float ann_data_m2[];
extern float ann_data_m3[];
extern float ann_data_m4[];

extern "C"  
{
    double voigt(double x, double sigma, double gamma);
};

namespace ldw_math_dnn
{
//some help function
    template <class myType>
    void sortArr(std::vector<myType> &arr, std::vector<int> &ndx) 
    { 
        std::vector<std::pair<myType, int> > vp; 
    
        for (int i = 0; i < arr.size(); ++i) { 
            vp.push_back(std::make_pair(arr[i], i)); 
        } 
    
        std::sort(vp.begin(), vp.end()); 
    
        for (int i = 0; i < vp.size(); i++)
        { 
            ndx.push_back(vp[i].second);
        } 
    };

    std::vector<int> find_neighboring_peaks(std::vector<int> cx,std::vector<int>  cy,int p)
    {
        int px=cx[p];
        int py=cy[p];
        std::vector<int> ndx;
        for(int i=0;i<cx.size();i++)
        {
            if(i==p) continue;
            int d1=cx[i]-px;
            int d2=cy[i]-py;
            if(sqrt(d1*d1+d2*d2)<30.0)
            {
                ndx.push_back(i);
            }
        }  
        return ndx;
    };

    bool calcualte_principal_axis(std::vector<double> data, int xdim, int ydim)
    {
        double sx,sy,ss;
        double xx,yy,xy;
        
        sx=sy=ss=0.0;

        for(int i=0;i<xdim;i++)
        {
            for(int j=0;j<ydim;j++)
            {
                sx+=data[i*ydim+j]*i;
                sy+=data[i*ydim+j]*j;
                ss+=data[i*ydim+j];
            }
        }
        sx/=ss;
        sy/=ss; //sx and sy is the weight center. ss is total weight

        Eigen::Matrix2f inertia_tensor;
        
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig;
        Eigen::Vector2f values;
        Eigen::Matrix2f evecs;

        xx=yy=xy=0.0;
        for(int i=0;i<xdim;i++)
        {
            for(int j=0;j<ydim;j++)
            {
                xx+=(i-sx)*(i-sx)*data[i*ydim+j];
                yy+=(j-sy)*(j-sy)*data[i*ydim+j];
                xy+=(i-sx)*(j-sy)*data[i*ydim+j];
            }
        }

        
        inertia_tensor(0,0)=xx/ss;
        inertia_tensor(1,1)=yy/ss;
        inertia_tensor(0,1)=xy/ss;
        inertia_tensor(1,0)=xy/ss;
        eig.compute(inertia_tensor);
        values = eig.eigenvalues();
        evecs = eig.eigenvectors();
        double px=evecs(0,1);
        double py=evecs(0,0);
        
        std::cout<<"xdim="<<xdim<<" ydim="<<ydim<<" sx="<<sx<<" sy="<<sy<<" ss="<<ss<<" cov matrix is "<<xx/ss<<" "<<yy/ss<<" "<<xy/ss<<" px="<<px<<"  py="<<py<<std::endl;

        double til=std::min(fabs(px),fabs(py));
        if(til>0.13 && til<0.62)  //7.5 degree to 38 degree
            return true;
        else
            return false;
    }

    bool voigt_convolution(double a, double x, double y, double sigmax, double sigmay, double gammax, double gammay,int xdim, int ydim, std::vector<double> &kernel, int &i0, int &i1, int &j0, int &j1)
    {
        float wx=(1.0692*gammax+sqrt(0.8664*gammax*gammax+5.5452*sigmax*sigmax))*1.0f;
        float wy=(1.0692*gammay+sqrt(0.8664*gammay*gammay+5.5452*sigmay*sigmay))*1.0f;
        
        i0=std::max(0,int(x-wx+0.5));
        i1=std::min(xdim,int(x+wx+0.5));
        j0=std::max(0,int(y-wy+0.5));
        j1=std::min(ydim,int(y+wy+0.5));

        kernel.clear();
        kernel.resize((i1-i0)*(j1-j0));
        
        for (int i =i0; i < i1; i++)
        {
            for (int j = j0; j < j1; j++)
            {
                double z1=voigt ( i-x, sigmax, gammax );
                double z2=voigt ( j-y, sigmay, gammay );
                kernel.at((i-i0)*(j1-j0)+j-j0)=a*z1*z2;
            }
        }
        return true;
    }

    bool voigt_convolution_region(double x, double y, double sigmax, double sigmay, double gammax, double gammay,int xdim, int ydim,int &i0, int &i1, int &j0, int &j1)
    {
        float wx=(1.0692*gammax+sqrt(0.8664*gammax*gammax+5.5452*sigmax*sigmax))*1.0f;
        float wy=(1.0692*gammay+sqrt(0.8664*gammay*gammay+5.5452*sigmay*sigmay))*1.0f;
        
        i0=std::max(0,int(x-wx+0.5));
        i1=std::min(xdim,int(x+wx+0.5));
        j0=std::max(0,int(y-wy+0.5));
        j1=std::min(ydim,int(y+wy+0.5));

        return true;
    }
    std::vector<int> get_best_partition(std::vector<float> peak_amplitudes, float ratio_cutoff)
    {
        float v_min = 1e30f;
        float v_max = 0.0f;
        int v_max_ndx, v_min_ndx, p;
        std::vector<int> r,r2;

        for (int i = 0; i < peak_amplitudes.size(); i++)
        {
            if (peak_amplitudes[i] > v_max)
            {
                v_max = peak_amplitudes[i];
                v_max_ndx = i;
            }
            if (peak_amplitudes[i] < v_min)
            {
                v_min = peak_amplitudes[i];
                v_min_ndx = i;
            }
        }

        if (v_min / v_max > ratio_cutoff)
        {
            r.clear();
        }
        else
        {
            if (v_min_ndx > v_max_ndx)
            {
                p = v_max_ndx + 1;
                while(peak_amplitudes[p]/v_min>v_max/peak_amplitudes[p])
                {
                    p++;
                }
            }
            else
            {
                p = v_max_ndx -1 ;
                while(peak_amplitudes[p]/v_min>v_max/peak_amplitudes[p])
                {
                    p--;
                }
                p = p + 1;
            }
            std::vector<float> peak_amplitudes_1(peak_amplitudes.begin(), peak_amplitudes.begin() + p);
            std::vector<float> peak_amplitudes_2(peak_amplitudes.begin() + p, peak_amplitudes.end());
            r = get_best_partition(peak_amplitudes_1, ratio_cutoff);
            r2 = get_best_partition(peak_amplitudes_2, ratio_cutoff);

            r.push_back(p);
            for (int k = 0; k < r2.size(); k++)
            {
                r.push_back(r2[k] + p);
            }

        }
        return r;
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

    int find_best_from_peaks(std::vector<double> x1,std::vector<double> y1,std::vector<double> x2,std::vector<double> y2, std::vector<int> &ndxs)
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
                double t1=(x1[i]-x1[j])*(x1[i]-x1[j])+(y1[i]-y1[j])*(y1[i]-y1[j]);
                double t2=(x2[i]-x2[j])*(x2[i]-x2[j])+(y2[i]-y2[j])*(y2[i]-y2[j]);
                double t=sqrt(t1)+sqrt(t2);
                distance_matrix[i*npeak+j]=distance_matrix[i+j*npeak]=t;
                if(t<6.0)
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

/*
  int find_medoid_from_peaks(std::vector<double> x,std::vector<double> y)
    {
        int npeak=x.size();
        std::vector<double> d(npeak*npeak,0.0);

        for(int i=0;i<npeak;i++)
        {
            for(int j=i+1;j<npeak;j++)
            {
                double t=fabs(x[i]-x[j])+fabs(y[i]-y[j]);
                d[i*npeak+j]=t;
                d[j*npeak+i]=t;
            }
        }

        double min_d=10000.0;
        int ndx=-1;
        for(int i=0;i<npeak;i++)
        {
            double current_d=0.0;
            for(int j=0;j<npeak;j++)
            {
                current_d+=d[i*npeak+j];
            }
            if(current_d<min_d)
            {
                min_d=current_d;
                ndx=i;
            }
        }
        return ndx;
    };

    int find_best_from_peaks_old(std::vector<double> x1,std::vector<double> y1,std::vector<double> x2,std::vector<double> y2)
    {
        int npeak=x1.size();
        
        double max_d=0.0;
        int ndx=-1;
        for(int i=0;i<npeak;i++)
        {
            double t=(x1[i]-x2[i])*(x1[i]-x2[i])+(y1[i]-y2[i])*(y1[i]-y2[i]);
            if(t>max_d)
            {
                max_d=t;
                ndx=i;
            }
        }
        return ndx;
    }

    bool cut_one_peak_using_amplitude(std::vector<double> &s, int &anchor)
    {
        int ndata=s.size();

        int center;
        double max_v=0.0;
        for(int i=anchor-2;i<=anchor+2;i++)
        {
            if(s[i]>max_v)
            {
                max_v=s[i];
                center=i;
            }
        } 

        int left_cut=0;
        for(int i=center-1;i>=0;i--)
        {
            if(s[i]>s[i+1])
            {   
                left_cut=i+1;
                break;
            }
        }

        int right_cut=ndata;
        for(int i=center+1;i<ndata;i++)
        {
            if(s[i]>s[i-1])
            {
                right_cut=i;
                break;
            }
        }

        s.erase(s.begin(),s.begin()+left_cut);
        anchor-=left_cut;
        right_cut-=left_cut;
        s.erase(s.begin()+right_cut,s.end());

        return true;
    };
*/

    bool get_perpendicular_line(double x,double y, double x0, double y0, std::vector<double> &line_x, std::vector<double> &line_y)
    {
        line_x.clear();
        line_y.clear();
        double direction_x=y0-y;
        double direction_y=x-x0;
        double len=sqrt(direction_x*direction_x+direction_y*direction_y);
        direction_x/=len;
        direction_y/=len;

        for(int m=-22;m<=22;m++)
        {
            line_x.push_back(m*direction_x+x);
            line_y.push_back(m*direction_y+y);
        }
        return true;
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
};


//funtions start here

pool1d::pool1d() {};
pool1d::~pool1d() {};

bool pool1d::predict(int nlen, std::vector<float> &input, std::vector<float> &output)
{
    output.clear();
    output.resize(nlen*nfilter);

    int nshift = (npool - 1) / 2;
    for (int j1 = 0; j1 < nlen; j1++)
    {
        for (int j2 = 0; j2 < nfilter; j2++)
        {
            std::vector<float> temp;
            temp.clear();
            for (int jj = std::max(0,j1 - nshift); jj < std::min(j1 + nshift+1,nlen); jj++)
            {
                temp.push_back(input[jj * nfilter + j2]);
            }
            output[j1 * nfilter + j2] = *std::max_element(temp.begin(),temp.end());
        }
    }
    return true;
};



base1d::base1d() {};
base1d::~base1d() {};

bool base1d::mat_mul(std::vector<float> &in1, std::vector<float> &in2, std::vector<float> &out, int m, int n, int k)
{
    //in1 is m by n
    //in2 in n by k
    //out will be m by k
    //matrix is saved row by row
    if(in1.size()!=m*n || in2.size()!=n*k)
    {
        return false;
    }

    for(int m0=0;m0<m;m0++)
    {
        for(int k0=0;k0<k;k0++)
        {
            float t=0.0;
            for(int n0=0;n0<n;n0++)
            {
                t+=in1[m0*n+n0]*in2[n0*k+k0];
            }
            out[m0*k+k0]=t;
        }
    }
    return true;
};

bool base1d::print()
{
    for (int i = 0; i < kernel.size(); i++)
    {
        std::cout<< kernel[i]<<" ";
    }
    std::cout<<std::endl;
    for (int i = 0; i < bias.size(); i++)
    {
        std::cout<< bias[i]<<" ";
    }
    std::cout<<std::endl;
    return true;
};



dense::dense() {};
dense::~dense() {};

bool dense::read(std::string fname)
{
    std::ifstream s(fname);
    // float t;

    kernel.resize(ninput*nfilter); 
    bias.resize(nfilter); 
    
    for (int i = 0; i < kernel.size(); i++)
    {
        s >> kernel[i];
    }

    for (int i = 0; i < bias.size(); i++)
    {
        s >> bias[i];
    }
    return true;
};

int dense::read(float *p)
{
    kernel.resize(ninput*nfilter); //3*8*6 
    bias.resize(nfilter); //size is 6  
    for (int i = 0; i < kernel.size(); i++)
    {
        kernel[i]=p[i];
    }
    for (int i = 0; i < bias.size(); i++)
    {
        bias[i]=p[i+kernel.size()];
    } 
    return kernel.size()+bias.size();
}

bool dense::predict(int nlen, std::vector<float> &input, std::vector<float> &output)
{

    output.clear();
    output.resize(nlen*nfilter);


    mat_mul(input,kernel,output,nlen,ninput,nfilter);

    // std::cout<<"in dense::predict, out is"<<std::endl;
    // for(int i=0;i<kernel.size();i++) std::cout<<kernel[i]<<" ";
    // std::cout<<std::endl;


    //apply bias
    for(int i=0;i<nfilter;i++)
    {
        for(int j=0;j<nlen;j++)
        {
            output[j*nfilter+i]+=bias[i];
        }
    }

    //relo activation
    if(a==relo)
    {
        for(int j=0;j<nlen*nfilter;j++)
        {
            output[j]=std::max(output[j],0.0f);    
        }
    }
    else if(a==softmax) //softmax
    {
        for(int i=0;i<nlen;i++)
        {
            float sum=0.0f;
            for(int j=0;j<nfilter;j++)
            {
                float t=exp(output[i*nfilter+j]);
                output[i*nfilter+j]=t;
                sum+=t;
            }
            for(int j=0;j<nfilter;j++)
            {
                output[i*nfilter+j]/=sum;
            }
        }
    }
    //do thing if it is linear activation

    return true;
};





conv1d::conv1d() {};
conv1d::~conv1d() {};


bool conv1d::read(std::string fname)
{
    std::ifstream s(fname);

    kernel.resize(nkernel*ninput*nfilter); //3*8*6 
    bias.resize(nfilter); //size is 6
    
    for (int i = 0; i < kernel.size(); i++)
    {
        s >> kernel[i];
    }
    for (int i = 0; i < bias.size(); i++)
    {
        s >> bias[i];
    }
    return true;
};


int conv1d::read(float *p)
{
    kernel.resize(nkernel*ninput*nfilter); //3*8*6 
    bias.resize(nfilter); //size is 6  
    for (int i = 0; i < kernel.size(); i++)
    {
        kernel[i]=p[i];
    }
    for (int i = 0; i < bias.size(); i++)
    {
        bias[i]=p[i+kernel.size()];
    } 
    return kernel.size()+bias.size();
}


bool conv1d::predict(int nlen, std::vector<float> &input, std::vector<float> &output)
{
    //nlen: 300
    //nkernel: 3
    //ninput: 8
    //nfilter: 6

    //input: 300*8
    //output: 300*6 

    int nblock=ninput*nfilter;    
    
    output.clear();
    output.resize(nlen*nfilter);

    //apply kernel
    for(int i=0;i<nkernel;i++)
    {
        std::vector<float> t1(kernel.begin()+nblock*i,kernel.begin()+nblock*(i+1));
        std::vector<float> out(nlen*nfilter,0.0);
        mat_mul(input,t1,out,nlen,ninput,nfilter);

        int nshift=(nkernel-1)/2-i;

        for(int j1=std::max(0,nshift);j1<std::min(nlen,nlen+nshift);j1++)
        {
            // if(j1-nshift<0) continue;
            // if(j1-nshift>=nlen) continue;
            for(int j2=0;j2<nfilter;j2++)
            {
                output[j1*nfilter+j2]+=out[(j1-nshift)*nfilter+j2];
            }
        }
    }

    //apply bias
    for(int i=0;i<nfilter;i++)
    {
        for(int j=0;j<nlen;j++)
        {
            output[j*nfilter+i]+=bias[i];
        }
    }

    //relo activation
    for(int j=0;j<nlen*nfilter;j++)
    {
        output[j]=std::max(output[j],0.0f);    
    }
    return true;
};



peak1d::peak1d() {
    model_selection=0; // no model
};
peak1d::~peak1d() {};

bool peak1d::load()
{
    model_selection=1;

    int n=0;
    c0.set_size(11,1,40);  //nkernel, ninput, nfilter
    // c0.read("conv1d.txt");
    n+=c0.read(ann_data+n);
    
    c1.set_size(1,40,20); //knernel, ninput, nfilter
    // c1.read("conv1d_1.txt");
    n+=c1.read(ann_data+n);
    
    c2.set_size(11,20,10); //knernel, ninput, nfilter
    // c2.read("conv1d_2.txt");
    n+=c2.read(ann_data+n);

    c3.set_size(1,10,20); //knernel, ninput, nfilter
    // c3.read("conv1d_3.txt");
    n+=c3.read(ann_data+n);

    c4.set_size(1,20,10); //knernel, ninput, nfilter
    // c4.read("conv1d_4.txt");
    n+=c4.read(ann_data+n);

    c5.set_size(11,10,30); //knernel, ninput, nfilter
    // c5.read("conv1d_5.txt");
    n+=c5.read(ann_data+n);

    c6.set_size(1,30,18); //knernel, ninput, nfilter
    // c6.read("conv1d_6.txt");
    n+=c6.read(ann_data+n);

    c7.set_size(1,18,8); //knernel, ninput, nfilter
    // c7.read("conv1d_7.txt");
    n+=c7.read(ann_data+n);

    p1.set_size(18,3);  //ninput, npool
    
    d.set_act(softmax);
    d.set_size(18,3); //ninput=18, nfilter=3
    // d.read("my_layer.txt");
    n+=d.read(ann_data+n);

    return true;
};


bool peak1d::load_m4()
{
    model_selection=4;

    int n=0;
    c0.set_size(11,1,40);  //nkernel, ninput, nfilter
    // c0.read("conv1d.txt");
    n+=c0.read(ann_data_m4+n);
    
    c1.set_size(1,40,20); //knernel, ninput, nfilter
    // c1.read("conv1d_1.txt");
    n+=c1.read(ann_data_m4+n);
    
    c2.set_size(11,20,10); //knernel, ninput, nfilter
    // c2.read("conv1d_2.txt");
    n+=c2.read(ann_data_m4+n);

    c3.set_size(1,10,20); //knernel, ninput, nfilter
    // c3.read("conv1d_3.txt");
    n+=c3.read(ann_data_m4+n);

    c4.set_size(1,20,10); //knernel, ninput, nfilter
    // c4.read("conv1d_4.txt");
    n+=c4.read(ann_data_m4+n);

    c5.set_size(11,10,30); //knernel, ninput, nfilter
    // c5.read("conv1d_5.txt");
    n+=c5.read(ann_data_m4+n);

    c6.set_size(1,30,18); //knernel, ninput, nfilter
    // c6.read("conv1d_6.txt");
    n+=c6.read(ann_data_m4+n);

    c7.set_size(1,18,8); //knernel, ninput, nfilter
    // c7.read("conv1d_7.txt");
    n+=c7.read(ann_data_m4+n);

    p1.set_size(18,3);  //ninput, npool
    
    d.set_act(softmax);
    d.set_size(18,3); //ninput=18, nfilter=3
    // d.read("my_layer.txt");
    n+=d.read(ann_data_m4+n);

    return true;
};

bool peak1d::load_m2()
{
    model_selection=2;

    int n=0;
    c0.set_size(9,1,40);  //nkernel, ninput, nfilter
    c1.set_size(1,40,20); //knernel, ninput, nfilter
    c2.set_size(7,20,10); //knernel, ninput, nfilter
    c3.set_size(1,10,20); //knernel, ninput, nfilter
    c4.set_size(1,20,10); //knernel, ninput, nfilter
    c5.set_size(7,10,30); //knernel, ninput, nfilter
    c6.set_size(1,30,18); //knernel, ninput, nfilter
    c7.set_size(1,18,8); //knernel, ninput, nfilter

    p1.set_size(18,3);  //ninput, npool
    
    d.set_act(softmax);
    d.set_size(18,3); //ninput=18, nfilter=3


    n+=c0.read(ann_data_m2+n);
    n+=c1.read(ann_data_m2+n);
    n+=c2.read(ann_data_m2+n);
    n+=c3.read(ann_data_m2+n);    
    n+=c4.read(ann_data_m2+n);
    n+=c5.read(ann_data_m2+n);
    n+=c6.read(ann_data_m2+n);
    n+=c7.read(ann_data_m2+n);
    n+=d.read(ann_data_m2+n);

    return true;
};

bool peak1d::load_m3() //1D, wide peak with baseline
{
    model_selection=3;

    int n=0;
    c0.set_size(11,1,40);  //nkernel, ninput, nfilter
    n+=c0.read(ann_data_m3+n);
    
    c1.set_size(1,40,20); //knernel, ninput, nfilter
    n+=c1.read(ann_data_m3+n);
    
    c2.set_size(11,20,10); //knernel, ninput, nfilter
    n+=c2.read(ann_data_m3+n);

    c3.set_size(1,10,20); //knernel, ninput, nfilter
    n+=c3.read(ann_data_m3+n);

    c4.set_size(1,20,10); //knernel, ninput, nfilter
    n+=c4.read(ann_data_m3+n);

    c5.set_size(11,10,30); //knernel, ninput, nfilter
    n+=c5.read(ann_data_m3+n);

    c6.set_size(1,30,18); //knernel, ninput, nfilter
    n+=c6.read(ann_data_m3+n);

    c7.set_size(1,18,8); //knernel, ninput, nfilter
    n+=c7.read(ann_data_m3+n);

    p1.set_size(18,3);  //ninput, npool
    
    d.set_act(softmax);
    d.set_size(18,3); //ninput=18, nfilter=3
    n+=d.read(ann_data_m3+n);

    return true;
};

bool peak1d::predict(std::vector<float> input_)
{   
    n_shift=20; 
    ndim0=input_.size()-40;
    input.clear();
    //prevent the ANN to go after noise-level peaks
    //default 60.0
    //if we increase it from 60.0f to 120.0f, we may lost some weak peaks but will be robust against noisy spectra
    scale_factor=std::max(*max_element(input_.begin(),input_.end()),noise_level*60.0f);

    for(int i=0;i<input_.size();i++)
    {
        if(input_[i]<0.0){
            input.push_back(0.0);
        }
        else{
            input.push_back(input_[i]/scale_factor);
        }
    }

    min_flag.clear();
    min_flag.resize(input.size(),0);
   
    std::vector<float> t1,t2,t3,t4,t5,t6,t7,t8;
    ndim=input.size();
    c0.predict(ndim,input,t1);
    c1.predict(ndim,t1,t2);
    c2.predict(ndim,t2,t3);
    c3.predict(ndim,t3,t4);
    c4.predict(ndim,t4,t5);
    c5.predict(ndim,t5,t6);
    c6.predict(ndim,t6,t7);
    c7.predict(ndim,t7,output2);
    p1.predict(ndim,t7,t8);
    d.predict(ndim,t8,output1);
    return true;
};


bool peak1d::predict_step2()
{   
    moving_average_output();

    posits.clear();
    centes.clear();
    sigmas.clear();
    gammas.clear();
    intens.clear();
    ptypes.clear();
    shouls.clear();
    confidence.clear();

    std::vector<int> ps;
    std::vector<double> vs;
    std::vector<int> p_type;
    std::vector<int> p_segment;

    //to do: define cutoff according to predicted peak width
    double cutoff=4.0;
    if(model_selection==2) cutoff=2.5;

    p_segment.push_back(0);
    for(int i=2;i<ndim-2;i++) //exclude terminal 2 data point!!
    {
        // if(output1[i*3+2]+output1[i*3+1]>output1[i*3] && output1[i*3]<output1[i*3-3] && output1[i*3]<output1[i*3+3])
        
        bool b1=output1[i*3+2]+output1[i*3+1]>output1[i*3];
        int ii=i-1;bool b11=output1[ii*3+2]+output1[ii*3+1]>output1[ii*3];
        ii=ii+1;   bool b12=output1[ii*3+2]+output1[ii*3+1]>output1[ii*3];
        b1= b1 || b11 || b12;
        
        bool b1_potential=output1[i*3+2]+output1[i*3+1]>output1[i*3]*0.6 && output1[i*3+2]>0.1;
        
        bool b2=output1[i*3+1]>output1[i*3-3+1] && output1[i*3+1]>output1[i*3+3+1] && output1[i*3+1]>0.1;
        bool b3=output1[i*3+2]>output1[i*3-3+2] && output1[i*3+2]>output1[i*3+3+2] && output1[i*3+2]>0.1;
        bool b4=(min_flag[i]==0);

        //exclude minimal and its two neighbor
        if( output1[i*3]>0.75 && p_type.size()!=p_segment[p_segment.size()-1] )
        {
            p_segment.push_back(p_type.size());    
        }
        
        if( b4 && b1 && ( b2 || b3 ) )
        {
            ps.push_back(i);
            vs.push_back(output1[i*3]); //score of non-peak
            p_type.push_back(1);
        }
        else if( b4 && b1_potential && ( b2 || b3 ) )
        {
            ps.push_back(i);
            vs.push_back(output1[i*3]); //score of non-peak
            p_type.push_back(2); //potential peak !!
        }
    }

    for(int k=0;k<p_segment.size()-1;k++)
    {
        if(p_segment[k+1]-p_segment[k]!=3) continue;
        
        int kk=p_segment[k];

        if(p_type[kk]==2 && p_type[kk+1]==1 && p_type[kk+2]==1)
        {
            p_type[kk]=1; //change potential peak to peak if pattern found
        }
        if(p_type[kk]==1 && p_type[kk+1]==1 && p_type[kk+2]==2){
            p_type[kk+2]=1; //change potential peak to peak if pattern found
        }
    }


    //For this type of pattern, if all 3 peaks are within overlap removal range, it might be benefitical to be consistent, such as
    //always keep the two non-center peaks and remove the central peak.
    for(int k=0;k<p_segment.size()-1;k++)
    {
        if(p_segment[k+1]-p_segment[k]!=3) continue;
        
        int kk=p_segment[k];

        int pos;
        double c1,c2,c3;

        pos=ps[kk];
        if(output1[pos*3+2]>output1[pos*3+1])
            c1=-(output2[pos*8+4]*3-1.5)+pos;
        else
            c1=-(output2[pos*8+0]*3-1.5)+pos;
        
        pos=ps[kk+1];
        if(output1[pos*3+2]>output1[pos*3+1])
            c2=-(output2[pos*8+4]*3-1.5)+pos;
        else
            c2=-(output2[pos*8+0]*3-1.5)+pos;
        
        
        pos=ps[kk+2];
        if(output1[pos*3+2]>output1[pos*3+1])
            c3=-(output2[pos*8+4]*3-1.5)+pos;
        else
            c3=-(output2[pos*8+0]*3-1.5)+pos;

        double d1=fabs(c1-c2);
        double d2=fabs(c3-c2);
        if(p_type[kk]==1 && p_type[kk+1]==1 && p_type[kk+2]==1 && (d1<cutoff || d2<cutoff) )
        {
            p_type[kk+1]=2; //label center peak to be removed, but keep other two peaks. 
        }
    }

    for(int k=p_type.size()-1;k>=0;k--)
    {
        // std::cout<<k<<" "<<ps[k]<<" "<<vs[k]<<" "<<p_type[k]<<std::endl;
        if(p_type[k]==2)  //remove remaining potential peak
        {
            p_type.erase(p_type.begin()+k);
            ps.erase(ps.begin()+k);
            vs.erase(vs.begin()+k);
        }
    }




    std::vector<int> ndx;
    ldw_math_dnn::sortArr(vs,ndx);

    for(int i=0;i<ndx.size();i++)
    {
        int pos=ps[ndx[i]];
        double c,a,w,s,g,t,con;
        if(output1[pos*3+2]>output1[pos*3+1]) //weak
        {
            c=-(output2[pos*8+4]*3-1.5)+pos;
            a=std::min(sqrt(std::max(0.0f,output2[pos*8+5]))/1.17,1.0);
            w=output2[pos*8+6]*20.0;
            g=output2[pos*8+7]*10.0;
            con=1.0-output1[pos*3];
        }
        else //strong
        {
            c=-(output2[pos*8+0]*3-1.5)+pos;
            a=std::min(std::max(0.0f,output2[pos*8+1])/6.6+0.85,1.0);
            w=output2[pos*8+2]*20.0;
            g=output2[pos*8+3]*10.0;
            con=1.0-output1[pos*3];
        }
        // std::cout<<"pos="<<pos<<" con="<<con<<std::endl;
        
        bool b_tonear=0;
        for(int k=0;k<centes.size();k++)
        {
            if(fabs(centes[k]-c)<cutoff)
            {
                b_tonear=1;
                break;
            }
        }

        if(b_tonear){
            continue;
        }
        
        
        t=w-1.0692*g;
        s=std::sqrt(std::max(t*t-0.8664*g*g,0.01)/5.5452);
        a*=input[pos]*scale_factor;
        
        posits.push_back(pos);
        centes.push_back(c);
        sigmas.push_back(s);
        gammas.push_back(g);
        intens.push_back(a);
        confidence.push_back(con);
        if(output1[pos*3+2]>output1[pos*3+1]){
            ptypes.push_back(0); //weak
        }
        else{
            ptypes.push_back(1); //strong
        }
    }


/*  
    for(int i=0;i<intens.size();i++)
    {
        if(ptypes[i]==1) continue;

        int index=-1;
        int min_dis=7;
        for(int j=0;j<intens.size();j++)
        {
            if(j==i) continue;
            if(ptypes[j]==0) continue;
            if(abs(posits[j]-posits[i])<min_dis)
            {
                min_dis=abs(posits[j]-posits[i]);
                index=j;
            }
        }
        if(index>=0)
        {
            //test whether we should keep i or not, using j as master peak
            std::vector<double> xn;
            std::vector<double> yn;
            double y_max=0.0;
            double x=posits[index];
            double a=7.7;
            double sigma=3.0;
            double gamma=0.1;
            double e=0.0;

            if (posits[i] > posits[index]) //shoulder peak is on the right side of main peak
            {
                for (int k = posits[index]; k < posits[i] + 5; k++)
                {
                    xn.push_back(k);
                    yn.push_back(input[k]);
                    if (input[k] > y_max)
                        y_max = input[k];
                }
            }
            else
            {
                for (int k = posits[i] - 4; k <= posits[index]; k++)
                {
                    xn.push_back(k);
                    yn.push_back(input[k]);
                    if (input[k] > y_max)
                        y_max = input[k];
                }
            }
        }
    }
*/

    //lable shoulder peaks so that double-shoulder peak can be removed.
    shouls.clear();
    shouls.resize(intens.size(),0);
    std::vector<int> local_max;
    for(int i=1;i<input.size()-1;i++)
    {
        if(input[i-1]<input[i] && input[i+1]<input[i])
        {
            local_max.push_back(i);
        }
    }
    std::vector<std::vector<int> > local_max_matches,local_max_distances;
    local_max_matches.resize(local_max.size());
    local_max_distances.resize(local_max.size());

    for(int i=0;i<posits.size();i++)
    {
        int min_dis=10000;
        int ndx=-1;
        for(int j=0;j<local_max.size();j++)
        {
            if(abs(local_max[j]-posits[i])<min_dis)
            {
                min_dis=abs(local_max[j]-posits[i]);
                ndx=j;
            }
        }
        if(ndx>-1)
        {
            local_max_matches[ndx].push_back(i);
            local_max_distances[ndx].push_back(min_dis);
        }
    }

    for(int i=0;i<local_max.size();i++)
    {
        if(local_max_matches[i].size()<=1) continue;
        int min_dis=1000;
        for(int j=0;j<local_max_distances[i].size();j++)
        {
            if(local_max_distances[i][j]<min_dis) min_dis=local_max_distances[i][j];
        }
        min_dis=std::max(min_dis,1);
        for(int j=0;j<local_max_distances[i].size();j++)
        {
            if(min_dis>=2 && local_max_distances[i][j]<min_dis*3)
            {
                shouls[local_max_matches[i][j]]=2;
            }
            if(local_max_distances[i][j]>=min_dis*3)
            {
                shouls[local_max_matches[i][j]]=1;
            }
        }
    }

    //remove out of bound peaks, if any
    for(int i=posits.size()-1;i>=0;i--)
    {
        posits[i]-=n_shift;
        centes[i]-=n_shift;
        if(posits[i]<=0 || posits[i]>=ndim0-1)
        {
            posits.erase(posits.begin()+i);
            ptypes.erase(ptypes.begin()+i);
            centes.erase(centes.begin()+i);
            sigmas.erase(sigmas.begin()+i);
            gammas.erase(gammas.begin()+i);
            intens.erase(intens.begin()+i);
            shouls.erase(shouls.begin()+i);
            confidence.erase(confidence.begin()+i);
        }
    }

    if(posits.size()!=confidence.size())
    {
        std::cout<<"ERROR, inconsistent size of pos and confidence in class peak1d."<<std::endl;
        std::cout<<"Sizes are "<<posits.size()<<" "<<centes.size()<<" "<<sigmas.size()<<" "<<gammas.size()<<" "<<intens.size()<<" "<<confidence.size()<<std::endl;
    }

    return true;
};

//helper of peak1d
bool peak1d::move_mean(std::vector<float> &data, int m, int n)
{
    std::vector<float> temp;
    temp=data;
    for(int i=1;i<m-1;i++)
    {
        for(int j=0;j<n;j++)
        {
            data[i*n+j]=(temp[(i-1)*n+j]+temp[i*n+j]+temp[(i+1)*n+j])/3.0f;
        }
    }

    for(int j=0;j<n;j++)
    {
        data[j]=(temp[j]+temp[n+j])/2.0f;
        data[(m-1)*n+j]=(temp[(m-2)*n+j]+temp[(m-1)*n+j])/2.0f;
    }

    return true;
};

bool peak1d::moving_average_output()
{
    move_mean(output1,ndim,3);   
    move_mean(output2,ndim,8);   
    return true;
};




//peak2d class. combine 1D result to get 2D.

peak2d::peak2d() {
    flag1=0;
};

peak2d::peak2d(int n)
{
    flag1=n;
}

peak2d::~peak2d() {};

bool peak2d::init_ann(int index_model)
{
    model_selection=index_model;
    if(index_model==1) return p1.load();
    if(index_model==2) return p1.load_m2();
    if(index_model==3) return p1.load_m3();
    if(index_model==4) return p1.load_m4();
    return true;
}

bool peak2d::column_2_row()
{
    spectrum_row.resize(spectrum_column.size());
    for (int i = 0; i < xdim; i++)
    {
        for (int j = 0; j < ydim; j++)
        {
            spectrum_row[j * xdim + i] = spectrum_column[i * ydim + j];
        }
    }
    return true;
};

bool peak2d::row_2_column()
{
    spectrum_column.resize(spectrum_row.size());
    for (int i = 0; i < xdim; i++)
    {
        for (int j = 0; j < ydim; j++)
        {
            spectrum_column[i * ydim + j] = spectrum_row[j * xdim + i];
        }
    }
    return true;
};

bool peak2d::init_spectrum(int xdim_, int ydim_, float noise, float scale, float scale2, std::vector<float> s, int flag)
{
    xdim=xdim_;
    ydim=ydim_;
    noise_level=noise;  
    user_scale=scale;
    user_scale2=scale2;

    p1.set_noise_level(noise_level);

    if(flag==0)  //input spectrum is in col by col format
    {
        spectrum_column=s;
        column_2_row(); 
    } 
    else
    {
        spectrum_row=s;
        row_2_column();
    }
    return true;
}

bool peak2d::predict_step1()
{
    int n=xdim*ydim;
    r_column.resize(n,-1);
    r_row.resize(n,-1);
    
    c_column.clear();
    a_column.clear();
    s_column.clear();
    g_column.clear();
    sh_column.clear();
        
    c_row.clear();
    a_row.clear();
    s_row.clear();
    g_row.clear();
    sh_row.clear();
    

    int counter=0;
#ifdef LDW_DEBUG
    std::ofstream fout1("vertical.txt");
#endif

    double boundary_cutoff=noise_level*user_scale2;

    //column by column prediction here
    for (int i = 0; i < xdim; i++)
    {
        std::vector<int> signa_boudaries;
        std::vector<int> noise_boudaries;
        std::vector<int> final_segment_begin,final_segment_stop;

        if(spectrum_column[i*ydim]>boundary_cutoff)
        {
             signa_boudaries.push_back(0);            
        }
        for(int j=1;j<ydim;j++)
        {
            if(spectrum_column[i*ydim+j-1]<=boundary_cutoff && spectrum_column[i*ydim+j]>boundary_cutoff)
            {
                signa_boudaries.push_back(std::max(j-10,0));
            }
            else if(spectrum_column[i*ydim+j-1]>boundary_cutoff && spectrum_column[i*ydim+j]<=boundary_cutoff)
            {
                noise_boudaries.push_back(std::min(j+10,ydim));
            }
        }
        if(noise_boudaries.size()<signa_boudaries.size())
        {
            noise_boudaries.push_back(ydim);
        }

        bool b=true;
        while(b)
        {
            b=false;
            for(int j=signa_boudaries.size()-1;j>=1;j--)
            {
                if(signa_boudaries[j]<=noise_boudaries[j-1])
                {
                    signa_boudaries.erase(signa_boudaries.begin()+j);
                    noise_boudaries.erase(noise_boudaries.begin()+j-1); 
                }
            }
        }

        for(int j=0;j<signa_boudaries.size();j++)
        {
            int begin=signa_boudaries[j];
            int stop=noise_boudaries[j];
            int begin0=begin;

            if (stop - begin < 5)
                continue;

            std::vector<float> spe(spectrum_column.begin()+i*ydim,spectrum_column.begin()+(i+1)*ydim);
            std::vector<int> peak_positions, min_positions;
            std::vector<float> peak_amplitudes;
            for (int k = begin + 2; k < stop - 2; k++)
            {
                
                if (spe[k] > noise_level * user_scale && spe[k] > spe[k - 2] && spe[k] > spe[k - 1] && spe[k] > spe[k + 1] && spe[k] > spe[k + 2])
                {
                    peak_positions.push_back(k);
                    peak_amplitudes.push_back(spe[k]);
                    continue;
                }
            }
            if(peak_positions.size()==0) continue;
            for(int k=0;k<peak_positions.size()-1;k++)
            {
                int p=peak_positions[k];
                float v=peak_amplitudes[k];
                for(int m=peak_positions[k];m<peak_positions[k+1];m++)
                {
                    if(spe[m]<v)
                    {
                        v=spe[m];
                        p=m;
                    }
                }
                min_positions.push_back(p);
            }
            min_positions.push_back(stop);
        
            std::vector<int> bs=ldw_math_dnn::get_best_partition(peak_amplitudes,0.2);
            bs.push_back(peak_amplitudes.size());

            for(int k=0;k<bs.size();k++)
            {
                int b,s;
                if(k==0) b=begin;
                else b=min_positions[bs[k-1]-1];
                s=min_positions[bs[k]-1];
                final_segment_begin.push_back(b);
                final_segment_stop.push_back(s);
                // std::cout<<"Cut into "<<b<<" - "<<s<<std::endl;
            }
        }


        for (int j = 0; j < final_segment_begin.size(); j++)
        {
            int begin = final_segment_begin[j];
            int stop = final_segment_stop[j];
            int begin0=begin;

            int left_patch=0;
            int right_patch=0;
            int n=stop-begin+40;

            left_patch=std::max(0,20-begin);
            begin=std::max(0,begin-20);
            right_patch=std::max(0,20-(ydim-stop));
            stop=std::min(ydim,stop+20);
            
            std::vector<float> t;
            t.clear();
            t.resize(n,0.0f);
            std::copy(spectrum_column.begin() + i * ydim + begin, spectrum_column.begin() + i * ydim + stop, t.begin()+left_patch);

            p1.predict(t);
            p1.predict_step2();
            
            for (int k = 0; k < p1.posits.size(); k++)
            {
#ifdef LDW_DEBUG
                fout1 << i<<" " <<p1.posits[k]+begin0<< " "<< p1.confidence[k] << std::endl;
#endif
                r_column[i * ydim + begin0 + p1.posits[k]] = counter;
                c_column.push_back(p1.centes[k]);
                a_column.push_back(p1.intens[k]);
                s_column.push_back(p1.sigmas[k]);
                g_column.push_back(p1.gammas[k]);
                sh_column.push_back(p1.shouls[k]);
                conf_column.push_back(p1.confidence[k]);
                counter++;
            }
        }
        if((i+1)%500==0) {std::cout<<"Finish "<<i+1<<" columns out of "<<xdim<<std::endl;}
    }
#ifdef LDW_DEBUG
    fout1.close();
    std::ofstream fout2("horizental.txt");
#endif
    counter=0;
    for (int i = 0; i < ydim; i++)
    {
        std::vector<int> signa_boudaries;
        std::vector<int> noise_boudaries;
        std::vector<int> final_segment_begin,final_segment_stop;

        if (spectrum_row[i * xdim] > boundary_cutoff)
        {
            signa_boudaries.push_back(0);
        }
        for (int j = 1; j < xdim; j++)
        {
            if (spectrum_row[i * xdim + j - 1] <= boundary_cutoff && spectrum_row[i * xdim + j] > boundary_cutoff)
            {
                signa_boudaries.push_back(std::max(j-10,0));
            }
            else if (spectrum_row[i * xdim + j - 1] > boundary_cutoff && spectrum_row[i * xdim + j] <= boundary_cutoff)
            {
                noise_boudaries.push_back(std::min(j+10,xdim));
            }
        }
        if (noise_boudaries.size() < signa_boudaries.size())
        {
            noise_boudaries.push_back(xdim);
        }

        bool b=true;
        while(b)
        {
            b=false;
            for(int j=signa_boudaries.size()-1;j>=1;j--)
            {
                if(signa_boudaries[j]<=noise_boudaries[j-1])
                {
                    signa_boudaries.erase(signa_boudaries.begin()+j);
                    noise_boudaries.erase(noise_boudaries.begin()+j-1); 
                }
            }
        }

        for (int j = 0; j < signa_boudaries.size(); j++)
        {
            int begin = signa_boudaries[j];
            int stop = noise_boudaries[j];
            int begin0=begin;

            if (stop - begin < 5)
                continue;
            std::vector<float> spe(spectrum_row.begin()+i*xdim,spectrum_row.begin()+(i+1)*xdim);
            std::vector<int> peak_positions, min_positions;
            std::vector<float> peak_amplitudes;
            for (int k = begin + 2; k < stop - 2; k++)
            {
                
                if (spe[k] > noise_level * user_scale && spe[k] > spe[k - 2] && spe[k] > spe[k - 1] && spe[k] > spe[k + 1] && spe[k] > spe[k + 2])
                {
                    peak_positions.push_back(k);
                    peak_amplitudes.push_back(spe[k]);
                    continue;
                }
            }
            if(peak_positions.size()==0) continue;
            for(int k=0;k<peak_positions.size()-1;k++)
            {
                int p=peak_positions[k];
                float v=peak_amplitudes[k];
                for(int m=peak_positions[k];m<peak_positions[k+1];m++)
                {
                    if(spe[m]<v)
                    {
                        v=spe[m];
                        p=m;
                    }
                }
                min_positions.push_back(p);
            }
            min_positions.push_back(stop);
            std::vector<int> bs=ldw_math_dnn::get_best_partition(peak_amplitudes,0.2);
            bs.push_back(peak_amplitudes.size());

            for(int k=0;k<bs.size();k++)
            {
                int b,s;
                if(k==0) b=begin;
                else b=min_positions[bs[k-1]-1];
                s=min_positions[bs[k]-1];
                final_segment_begin.push_back(b);
                final_segment_stop.push_back(s);
                // std::cout<<"Cut into "<<b<<" - "<<s<<std::endl;
            }
        }

        for (int j = 0; j < final_segment_begin.size(); j++)
        {
            int begin = final_segment_begin[j];
            int stop = final_segment_stop[j];
            int begin0=begin;

            int left_patch=0;
            int right_patch=0;
            int n=stop-begin+40;

            left_patch=std::max(0,20-begin);
            begin=std::max(0,begin-20);
            right_patch=std::max(0,20-(xdim-stop));
            stop=std::min(xdim,stop+20);
            
            std::vector<float> t;
            t.clear();
            t.resize(n,0.0f);
            std::copy(spectrum_row.begin() + i * xdim + begin, spectrum_row.begin() + i * xdim + stop, t.begin()+left_patch);


            
            p1.predict(t);
            p1.predict_step2();
            for (int k = 0; k < p1.posits.size(); k++)
            {
#ifdef LDW_DEBUG
                fout2 << p1.posits[k]+begin0 << " " << i << " " << p1.confidence[k] << std::endl;
#endif
                r_row[i * xdim + p1.posits[k] + begin0] = counter;
                c_row.push_back(p1.centes[k]);
                a_row.push_back(p1.intens[k]);
                s_row.push_back(p1.sigmas[k]);
                g_row.push_back(p1.gammas[k]);
                sh_row.push_back(p1.shouls[k]);
                conf_row.push_back(p1.confidence[k]);
                counter++;
            }
        }
        if((i+1)%500==0) {std::cout<<"Finish "<<i+1<<" rows out of "<<ydim<<std::endl;}
    }
#ifdef LDW_DEBUG
    fout2.close();
#endif

    //not needed any longer. save memory!!
    spectrum_column.clear();
    spectrum_row.clear();

    return true;
}

//find lines!!
bool peak2d::predict_step2()
{
    find_lines(xdim,ydim,r_column,column_line_x,column_line_y,column_line_index,column_line_segment);
    find_lines(ydim,xdim,r_row,row_line_y,row_line_x,row_line_index,row_line_segment);

    //column_line_index and row_line_index are index to a,s,g_column and a,s,g_row to get peak paramters!!

    //r_column and r_row is not needed any longer after here. save memory!!
    r_column.clear();
    r_row.clear();

#ifdef LDW_DEBUG
    std::ofstream fout;
    fout.open("line_vertical.txt");
    for(int i=0;i<column_line_x.size();i++)
    {
        fout<<column_line_x[i]<<" "<<column_line_y[i]<<std::endl;
    }
    fout.close();

    fout.open("line_vertical_s.txt");
    for(int i=0;i<column_line_segment.size();i++)
    {
        fout<<column_line_segment[i]<<std::endl;
    }
    fout.close();

    fout.open("line_horizental.txt");
    for(int i=0;i<row_line_x.size();i++)
    {
        fout<<row_line_x[i]<<" "<<row_line_y[i]<<std::endl;
    }
    fout.close();

    fout.open("line_horizental_s.txt");
    for(int i=0;i<row_line_segment.size();i++)
    {
        fout<<row_line_segment[i]<<std::endl;
    }
    fout.close();
#endif

    rl_column.resize(xdim*ydim,-1);
    rl_row.resize(xdim*ydim,-1);
    rl_column_p.resize(xdim*ydim,-1);
    rl_row_p.resize(xdim*ydim,-1);

    // values in rl_row and rl_column are line index
    // values in rl_row_p and rl_column_p are 1D peak index, that can be used to get inten (through a_column,a_row), sigma, gamma.  

    for(int i=0;i<column_line_segment.size();i++)
    {
        int b,s;
        if(i==0) b=0;
        else b=column_line_segment[i-1];
        s=column_line_segment[i];
        for(int j=b;j<s;j++)
        {
            rl_column[column_line_x[j]*ydim+column_line_y[j]]=i;
            rl_column_p[column_line_x[j]*ydim+column_line_y[j]]=column_line_index[j];
        }
    }

    for(int i=0;i<row_line_segment.size();i++)
    {
        int b,s;
        if(i==0) b=0;
        else b=row_line_segment[i-1];
        s=row_line_segment[i];
        for(int j=b;j<s;j++)
        {
            rl_row[row_line_x[j]*ydim+row_line_y[j]]=i;
            rl_row_p[row_line_x[j]*ydim+row_line_y[j]]=row_line_index[j];
        }
    }
    return true;
}

std::vector<int>  peak2d::select_max_nonoverlap_set(std::vector<int> tx,std::vector<int> ty,int cutoff)
{
    std::vector<int> ndxs;
    int n=tx.size();
    std::vector< std::vector<int> > neighbor;
    std::vector< std::vector<int> > cliques;
    neighbor.resize(n,std::vector<int>(n,0));


    for(int i=0;i<n;i++)
    {
        for(int j=i+1;j<n;j++)
        {
            int d=abs(tx[i]-tx[j])+abs(ty[i]-ty[j]);
            if(d>=cutoff)
            {
                neighbor[i][j]=neighbor[j][i]=1;
            }
        }
    }

    class cmaxclique c;
    c.init(&neighbor);
    c.solver();
    cliques=c.output();

    int index=-1;
    int max_n=0;
    for(int i=0;i<cliques.size();i++)
    {
        if(cliques[i].size()>max_n)
        {
            max_n=cliques[i].size();
            index=i;
        }
    }

    if(max_n>0)
    {
        ndxs=cliques[index];
    }
    else
    {
        ndxs.push_back(int(tx.size()/2.0+0.1));
    }
    return ndxs;
}


bool peak2d::predict_step3()
{
    //to do: define cutoff according to predicted peak width
    int ncutoff=6;
    if(model_selection==2) ncutoff=4;

    //new algorithm
    std::vector<int> tx;
    std::vector<int> ty;
    std::vector<double> w;

    for(int i=0;i<row_line_segment.size();i++)
    {
        int b,s;
        if(i==0) b=0;
        else b=row_line_segment[i-1];
        s=row_line_segment[i];
        tx.clear();
        ty.clear();
        int c=-1;
        for (int j = b; j < s; j++)
        {
            int x = row_line_x[j];
            int y = row_line_y[j];            //one point in row_line
            if (rl_column[x * ydim + y] >= 0) //is also a point in column_line
            {
                if(tx.size()>0 && rl_column[x * ydim + y]!=c)
                {
                    std::vector<int> ndxs=select_max_nonoverlap_set(tx,ty,ncutoff);
                    for(int n=0;n<ndxs.size();n++)
                    {
                        cx.push_back(tx[ndxs[n]]);
                        cy.push_back(ty[ndxs[n]]);
                    }
                    tx.clear();
                    ty.clear();
                }
                tx.push_back(x);
                ty.push_back(y);
                c=rl_column[x * ydim + y];
            }
        }
        if (tx.size() > 0)
        {
            std::vector<int> ndxs = select_max_nonoverlap_set(tx, ty, ncutoff);
            for (int n = 0; n < ndxs.size(); n++)
            {
                cx.push_back(tx[ndxs[n]]);
                cy.push_back(ty[ndxs[n]]);
            }
            tx.clear();
            ty.clear();
        }
    }
    for (int j = 0; j < cx.size(); j++)
    {
        int ndxx = cx[j];
        int ndxy = cy[j];
        p_2_column_paras.push_back(rl_column_p[ndxx * ydim + ndxy]);
        p_2_row_paras.push_back(rl_row_p[ndxx * ydim + ndxy]);
        p_2_line_column.push_back(rl_column[ndxx * ydim + ndxy]);
        p_2_line_row.push_back(rl_row[ndxx * ydim + ndxy]);
    }
    //we do not need rl_column,rl_column_p, rl_row or rl_row_p from here. clear them to save memory!!
    rl_column.clear();
    rl_column_p.clear();
    rl_row.clear();
    rl_row_p.clear();

    // check_special_peaks_1(); //actually do nothing in this function. Kept here for future consideration only !!

#ifdef LDW_DEBUG
    setup_peaks_from_p();
    std::ofstream fout2;
    fout2.open("potential_peaks.txt");
    for (int i = 0; i < cx.size(); i++)
    {
        fout2 << cx[i] << " " << cy[i] << " " << inten[i] << " " << shoulx[i] << " " << shouly[i] << " " << confidencex[i] << " " << confidencey[i] << std::endl;
    }
    fout2.close();
#endif

    if(flag1==0)
    {
        setup_peaks_from_p();
        // p_2_column_paras: index to a_column,s_column etc to get inte, sigma ...., 1D peak parameter of column by column picking (indirect dimension). p_2_row_paras: same
        // p_2_line_column: index of column lines,  p_2_line_row: index of row lines
        // check_special_peaks_2();  //label peaks that should be excluded in check_special_peaks_3 function call
        check_special_peaks_3();  //add new peaks to cx,cy, p_2_column_paras,p_2_row_paras. p_2_line_column and p_2_line_row lost track after this function!! Tilted priciple overalpped two peaks will have only one intersection
        setup_peaks_from_p();

        //remove double shoulder
        for (int i = cx.size() - 1; i >= 0; i--)
        {
            if ((fabs(shoulx[i] - 1) < 0.0001 && fabs(shouly[i] - 1) < 0.0001))
            {
                cx.erase(cx.begin() + i);
                cy.erase(cy.begin() + i);
                p_2_column_paras.erase(p_2_column_paras.begin() + i);
                p_2_row_paras.erase(p_2_row_paras.begin() + i);
                inten.erase(inten.begin() + i);
                sigmax.erase(sigmax.begin() + i);
                gammax.erase(gammax.begin() + i);
                shoulx.erase(shoulx.begin() + i);
                sigmay.erase(sigmay.begin() + i);
                gammay.erase(gammay.begin() + i);
                shouly.erase(shouly.begin() + i);
                confidencex.erase(confidencex.begin() + i);
                confidencey.erase(confidencey.begin() + i);
            }
        }
    }
    else if(flag1==2 || flag1==3)
    {
        setup_peaks_from_p(); 
 
        std::vector<double> sum_of_all_peaks;
        sum_of_all_peaks.resize(xdim*ydim,0.0f);

        std::vector< std::vector<double> > spectrum_of_peaks;
        spectrum_of_peaks.resize(inten.size());
        int i0,i1,j0,j1;

        for(int i=0;i<inten.size();i++)
        {
            ldw_math_dnn::voigt_convolution(inten[i],cx[i],cy[i],sigmax[i],sigmay[i],gammax[i],gammay[i],xdim,ydim,spectrum_of_peaks[i],i0,i1,j0,j1);
            for (int ii = i0; ii < i1; ii++)
            {
                for (int jj = j0; jj < j1; jj++)
                {
                    sum_of_all_peaks[ii * ydim + jj] += spectrum_of_peaks[i][(ii - i0) * (j1 - j0) + jj - j0];
                }
            }
        }

        std::vector<int> axis_til;
        for(int i=0;i<inten.size();i++)
        {
            ldw_math_dnn::voigt_convolution_region(cx[i],cy[i],sigmax[i],sigmay[i],gammax[i],gammay[i],xdim,ydim,i0,i1,j0,j1);
            for (int ii = i0; ii < i1; ii++)
            {
                for (int jj = j0; jj < j1; jj++)
                {
                    double z1=spectrum_of_peaks[i][(ii - i0) * (j1 - j0) + jj - j0];
                    double z2=std::max(sum_of_all_peaks[ii * ydim + jj],0.0000000001);
                    spectrum_of_peaks[i][(ii - i0) * (j1 - j0) + jj - j0]=z1/z2*spectrum_column[ii * ydim + jj];
                }
            }
            std::cout<<i<<" "<<cx[i]+1<<" "<<cy[i]+1<<std::endl;
            axis_til.push_back(ldw_math_dnn::calcualte_principal_axis(spectrum_of_peaks[i],i1-i0,j1-j0));
        }
    
        std::ofstream fnewpeak("new_peak.txt");
        for(int i=0;i<inten.size();i++)
        {
            // if(cx[i]+1!=185 || cy[i]+1!=362) continue; //plane of sichun
            // if(cx[i]+1!=496 || cy[i]+1!=101) continue; //rop
            // if(cx[i]+1!=1275 || cy[i]+1!=1523) continue; //asynu reduced
            // if(i!=471) continue; //asynu
            // if(i!=112) continue; //asynu
            
            // if(i!=419) continue; //asynu_a
            // if(i!=281) continue; //asynu_a
            
        
            
            if( inten[i] < noise_level * user_scale ) continue;
            if( shoulx[i]==1 || shouly[i]==1 ) continue; 
            if(axis_til[i]==0) continue;  

            std::vector<int> ndx_neighbors=ldw_math_dnn::find_neighboring_peaks(cx,cy,i);

            std::vector<double> title_angle_x,title_angle_y;
            title_angle_x.clear();
            title_angle_y.clear();

            std::vector<double> new_peak1_x,new_peak1_y,new_peak2_x,new_peak2_y,new_peak_tiltx;
            std::vector<double> new_peak1_inten,new_peak2_inten;
            std::vector<double> new_peak1_sigma, new_peak1_gamma, new_peak2_sigma, new_peak2_gamma;

           
            for (double tx = 10; tx <= 35; tx += 2.0)
            {
                title_angle_x.push_back(cos(tx * M_PI / 180.0));
                title_angle_y.push_back(sin(tx * M_PI / 180.0));
            }
            for (double tx = 55; tx <= 80; tx += 2.0)
            {
                title_angle_x.push_back(cos(tx * M_PI / 180.0));
                title_angle_y.push_back(sin(tx * M_PI / 180.0));
            }
            for (double tx = 100; tx <= 125; tx += 2.0)
            {
                title_angle_x.push_back(cos(tx * M_PI / 180.0));
                title_angle_y.push_back(sin(tx * M_PI / 180.0));
            }
            for (double tx = 145; tx <= 170; tx += 2.0)
            {
                title_angle_x.push_back(cos(tx * M_PI / 180.0));
                title_angle_y.push_back(sin(tx * M_PI / 180.0));
            }

            int x_int=int(cx[i]+0.5);
            int y_int=int(cy[i]+0.5);
            // std::cout<<"i="<<i<<" x="<<cx[i]<<" y="<<cy[i]<<std::endl;
            for(int j=0;j<title_angle_x.size();j++)
            {
                std::vector<double> target_line,target_line_x,target_line_y;
                target_line.clear();
                target_line_x.clear();
                target_line_y.clear();
                for(int m=-22;m<=22;m++)
                {
                    double target_x=cx[i]+m*title_angle_x[j];
                    double target_y=cy[i]+m*title_angle_y[j];
                    if(target_x>=0 && target_x<xdim-1.0 && target_y>=0 && target_y<ydim-1.0)
                    {
                        target_line_x.push_back(target_x);
                        target_line_y.push_back(target_y);                       
                    }
                }
                if(interp2(target_line_x,target_line_y,target_line)==false) 
                    continue;
                

                //cut target_line and get center.
                int anchor_pos=22;
                int pos_start=0;
                int pos_end=target_line_x.size();
                cut_one_peak(target_line_x,target_line_y,i,ndx_neighbors,anchor_pos,pos_start,pos_end); //remove segment belongs to other peaks

                anchor_pos-=pos_start;
                pos_end-=pos_start;

                target_line.erase(target_line.begin(),target_line.begin()+pos_start);
                target_line.erase(target_line.begin()+pos_end,target_line.end());


                std::vector<float> target_line_float;
                for(int ii=0;ii<20;ii++) target_line_float.push_back(target_line[0]);
                for(int ii=0;ii<target_line.size();ii++)
                {
                    target_line_float.push_back(target_line[ii]);
                }
                for(int ii=0;ii<20;ii++) target_line_float.push_back(target_line.back());

                p1.predict(target_line_float);
                p1.predict_step2();

                double s1=0.5346*gammax[i]*2+std::sqrt(0.2166*4*gammax[i]*gammax[i]+sigmax[i]*sigmax[i]*8*0.6931);

                std::vector<int> kk;
                for (int k = 0; k < p1.posits.size(); k++)
                {
                    if(p1.posits[k]>=anchor_pos-s1*1.2 && p1.posits[k]<=anchor_pos+s1*1.2 && p1.intens[k]>noise_level*user_scale2)
                    {
                        kk.push_back(k);
                    }    
                }

                if(kk.size()==2) //
                {
                    int pl=std::min(p1.posits[kk[0]],p1.posits[kk[1]]);
                    int pr=std::max(p1.posits[kk[0]],p1.posits[kk[1]]);
                    // if(pl>anchor_pos+1 || pr<anchor_pos-1) continue; //both peak are both on the left (or right) of the old peak.
                    if(pl>anchor_pos || pr<anchor_pos) continue; //both peak are both on the left (or right) of the old peak.

                    if(p1.posits[kk[0]]<p1.posits[kk[1]])
                    {
                        std::swap(kk[0],kk[1]);
                    }

                    double x1=(p1.posits[kk[0]]-anchor_pos)*title_angle_x[j]+x_int;
                    double y1=(p1.posits[kk[0]]-anchor_pos)*title_angle_y[j]+y_int;
                    double x2=(p1.posits[kk[1]]-anchor_pos)*title_angle_x[j]+x_int;
                    double y2=(p1.posits[kk[1]]-anchor_pos)*title_angle_y[j]+y_int;
                    new_peak_tiltx.push_back(title_angle_x[j]);
                    new_peak1_x.push_back(x1);
                    new_peak1_y.push_back(y1);
                    new_peak2_x.push_back(x2);
                    new_peak2_y.push_back(y2);
                    new_peak1_inten.push_back(p1.intens[kk[0]]);
                    new_peak1_sigma.push_back(p1.sigmas[kk[0]]);
                    new_peak1_gamma.push_back(p1.gammas[kk[0]]);
                    new_peak2_inten.push_back(p1.intens[kk[1]]);
                    new_peak2_sigma.push_back(p1.sigmas[kk[1]]);
                    new_peak2_gamma.push_back(p1.gammas[kk[1]]);
                    
                }
                // std::cout<<"j="<<j<<" and size of add peak is "<<kk.size()<<" out of "<<p1.posits.size()<<std::endl;
            }

            std::cout<<"Potential new peak groups:"<<std::endl;
            for(int j=0;j<new_peak1_x.size();j++)
            {
                std::cout<<new_peak_tiltx[j]<<" "<<new_peak1_x[j]<<" "<<new_peak1_y[j]<<" "<<new_peak2_x[j]<<" "<<new_peak2_y[j]<<std::endl;
            }

            std::cout<<"finish check peak "<<i<<" at coor: "<<cx[i]+1<<" "<<cy[i]+1<<std::endl;

            if(new_peak1_x.size()>1)  //2 or more peaks.
            {
                std::vector<int> ndxs;
                ldw_math_dnn::find_best_from_peaks(new_peak1_x,new_peak1_y,new_peak2_x,new_peak2_y,ndxs);

                for(int m=0;m<ndxs.size();m++)
                {
                    bool b_add=false;
                    int pos=ndxs[m];
                    if(find_nearest_normal_peak(new_peak1_x[pos],new_peak1_y[pos],ndx_neighbors,i) && find_nearest_normal_peak(new_peak2_x[pos],new_peak2_y[pos],ndx_neighbors,i))
                    {
                        //check again along perpendicular direction
                        std::vector<double> perpendicular_line_x,perpendicular_line_y,perpendicular_line_v;

                        double td1=(new_peak1_x[pos]-cx[i])*(new_peak1_x[pos]-cx[i])+(new_peak1_y[pos]-cy[i])*(new_peak1_y[pos]-cy[i]);
                        double td2=(new_peak2_x[pos]-cx[i])*(new_peak2_x[pos]-cx[i])+(new_peak2_y[pos]-cy[i])*(new_peak2_y[pos]-cy[i]);

                        if(td1>td2)
                            ldw_math_dnn::get_perpendicular_line(new_peak1_x[pos],new_peak1_y[pos], cx[i], cy[i], perpendicular_line_x,perpendicular_line_y);
                        else
                            ldw_math_dnn::get_perpendicular_line(new_peak2_x[pos],new_peak2_y[pos], cx[i], cy[i], perpendicular_line_x,perpendicular_line_y);

                        interp2(perpendicular_line_x, perpendicular_line_y,perpendicular_line_v);

                        double max_ele=0.0;
                        int max_ele_pos=-1;
                        for(int k=19;k<26;k++)
                        {
                            if(perpendicular_line_v[k]>max_ele)
                            {
                                max_ele=perpendicular_line_v[k];
                                max_ele_pos=k;
                            }
                        }
                        
                        if(abs(max_ele_pos-22)<=2)
                        {
                            b_add=true;
                            std::cout<<"ADD new peak at: "<<new_peak1_x[pos]<<" "<<new_peak1_y[pos]<<" and "<<new_peak2_x[pos]<<" "<<new_peak2_y[pos]<<std::endl;
                            fnewpeak<<cx[i]<<" "<<cy[i]<<" "<<new_peak1_x[pos]<<" "<<new_peak1_y[pos]<<" "<<new_peak2_x[pos]<<" "<<new_peak2_y[pos]<<std::endl;
                        }
                    }
                    if(b_add==false)
                    {
                        std::cout<<"Potential new peak at: "<<new_peak1_x[pos]<<" "<<new_peak1_y[pos]<<" and "<<new_peak2_x[pos]<<" "<<new_peak2_y[pos]<<std::endl;
                    }
                }

                
            }
        }    
        fnewpeak.close();    
    }
    else
    {
        setup_peaks_from_p();
    }
    return true;
};


bool peak2d::cut_one_peak_v2(std::vector<int> & line_x,std::vector<int> & line_y,std::vector<int> & line_ndx,const int current_pos,const std::vector<int> ndx_neighbors)
{
    int anchor_pos=-1;
    int xx=cx[current_pos];
    int yy=cy[current_pos];

    for(int i=0;i<line_x.size();i++)
    {
        if(line_x[i]==xx && line_y[i]==yy)
        {
            anchor_pos=i;
            break;
        }
    }
    if(anchor_pos==-1)
    {
        std::cout<<"Something is wrong in cut_one_peak_v2"<<std::endl;
        return false;
    }

    int pos_start=0;
    int pos_end=line_x.size();

    for(int i=anchor_pos;i>=0;i--)
    {
        double x=line_x[i];
        double y=line_y[i];

        double z_current=inten[current_pos]*voigt(cx[current_pos]-x,sigmax[current_pos],gammax[current_pos])*voigt(cy[current_pos]-y,sigmay[current_pos],gammay[current_pos]);

        bool b_found=false;
        for(int j=0;j<ndx_neighbors.size();j++)
        {
            int test_pos=ndx_neighbors[j];
            double z_test=inten[test_pos]*voigt(cx[test_pos]-x,sigmax[test_pos],gammax[test_pos])*voigt(cy[test_pos]-y,sigmay[test_pos],gammay[test_pos]);
            if(z_test>z_current/3.0)
            {
                b_found=true;
                break;
            }
        }
        if(b_found)
        {
            pos_start=i;
            break;
        }
    }
    
    for(int i=anchor_pos;i<line_x.size();i++)
    {
        double x=line_x[i];
        double y=line_y[i];

        double z_current=inten[current_pos]*voigt(cx[current_pos]-x,sigmax[current_pos],gammax[current_pos])*voigt(cy[current_pos]-y,sigmay[current_pos],gammay[current_pos]);

        bool b_found=false;
        for(int j=0;j<ndx_neighbors.size();j++)
        {
            int test_pos=ndx_neighbors[j];
            double z_test=inten[test_pos]*voigt(cx[test_pos]-x,sigmax[test_pos],gammax[test_pos])*voigt(cy[test_pos]-y,sigmay[test_pos],gammay[test_pos]);
            if(z_test>z_current/3.0)
            {
                b_found=true;
                break;
            }
        }
        if(b_found)
        {
            pos_end=i;
            break;
        }
    }

    line_x=std::vector<int>(line_x.begin()+pos_start,line_x.begin()+pos_end);
    line_y=std::vector<int>(line_y.begin()+pos_start,line_y.begin()+pos_end);
    line_ndx=std::vector<int>(line_ndx.begin()+pos_start,line_ndx.begin()+pos_end);
    
    return true;
}
  

bool peak2d::cut_one_peak(std::vector<double> target_line_x,std::vector<double>  target_line_y,int current_pos,std::vector<int> ndx_neighbors, int anchor_pos,int &pos_start,int &pos_end)
{
    pos_start=0;
    pos_end=target_line_x.size();

    for(int i=anchor_pos;i>=0;i--)
    {
        double x=target_line_x[i];
        double y=target_line_y[i];

        double z_current=inten[current_pos]*voigt(cx[current_pos]-x,sigmax[current_pos],gammax[current_pos])*voigt(cy[current_pos]-y,sigmay[current_pos],gammay[current_pos]);

        bool b_found=false;
        for(int j=0;j<ndx_neighbors.size();j++)
        {
            int test_pos=ndx_neighbors[j];
            double z_test=inten[test_pos]*voigt(cx[test_pos]-x,sigmax[test_pos],gammax[test_pos])*voigt(cy[test_pos]-y,sigmay[test_pos],gammay[test_pos]);
            if(z_test>z_current)
            {
                b_found=true;
                break;
            }
        }
        if(b_found)
        {
            pos_start=i;
            break;
        }
    }

    for(int i=anchor_pos;i<target_line_x.size();i++)
    {
        double x=target_line_x[i];
        double y=target_line_y[i];

        double z_current=inten[current_pos]*voigt(cx[current_pos]-x,sigmax[current_pos],gammax[current_pos])*voigt(cy[current_pos]-y,sigmay[current_pos],gammay[current_pos]);

        bool b_found=false;
        for(int j=0;j<ndx_neighbors.size();j++)
        {
            int test_pos=ndx_neighbors[j];
            double z_test=inten[test_pos]*voigt(cx[test_pos]-x,sigmax[test_pos],gammax[test_pos])*voigt(cy[test_pos]-y,sigmay[test_pos],gammay[test_pos]);
            if(z_test>z_current)
            {
                b_found=true;
                break;
            }
        }
        if(b_found)
        {
            pos_end=i;
            break;
        }
    }
    return true;
}

bool peak2d::find_nearest_normal_peak(double x, double y,std::vector<int> ndxs, int p)
{

    double max_amp=0.0;
    double max_effect=0.0;
    for(int ii=0;ii<ndxs.size();ii++)
    {
        int i=ndxs[ii];
        double eff=voigt(cx[i]-x,sigmax[i],gammax[i])*voigt(cy[i]-y,sigmay[i],gammay[i]);
        double amp=inten[i]*eff;
        eff/=(voigt(0,sigmax[i],gammax[i])*voigt(0,sigmay[i],gammay[i]));
        if(amp>max_amp)
        {
            max_amp=amp;
        }
        if(eff>max_effect)
        {
            max_effect=eff;
        }
    }
    double amp_at_p=inten[p]*voigt(cx[p]-x,sigmax[p],gammax[p])*voigt(cy[p]-y,sigmay[p],gammay[p]);

    if(max_amp>amp_at_p/3.0 || max_effect>0.2)
        return false;
    else
        return true;
};

bool peak2d::interp2(std::vector<double> line_x, std::vector<double> line_y,std::vector<double> &line_v)
{
    int ndata=line_x.size();
    int min_x=std::max(int(floor(std::min(line_x[0],line_x.back())))-2,0);
    int max_x=std::min(int(ceil(std::max(line_x[0],line_x.back())))+2,xdim-1);
    int min_y=std::max(int(floor(std::min(line_y[0],line_y.back())))-2,0);
    int max_y=std::min(int(ceil(std::max(line_y[0],line_y.back())))+2,ydim-1);

    // if(max_x-min_x<4 || max_y-min_y<4) return false;
    
    //First version
    // std::vector< std::vector<double> > row_spe_at_x;
    // std::vector< std::vector<double> > column_spe_at_x;
    // std::vector<double> x_input;
    // for(int i=min_x;i<=max_x;i++) x_input.push_back(i);
    // for(int j=min_y;j<=max_y;j++)
    // {
    //     std::vector<double> y_input,t;
    //     y_input.clear();
    //     for(int i=min_x;i<=max_x;i++) y_input.push_back(spectrum_row[j*xdim+i]);
    //     tk::spline st(x_input,y_input);   
    //     for(int m=0;m<ndata;m++)
    //     {
    //         t.push_back(st(line_x[m]));  
    //     }
    //     row_spe_at_x.push_back(t);
    // }

    // //row_spe_at_x ==> column_spe_at_x. Size along x is ndata, along y is max_y-min_y+1
    // column_spe_at_x.resize(ndata,std::vector<double>(max_y-min_y+1,0.0));
    // for(int i=0;i<ndata;i++)
    // {
    //     for(int j=0;j<max_y-min_y+1;j++)
    //     {
    //         column_spe_at_x[i][j]=row_spe_at_x[j][i];
    //     }
    // }

    // //now x input is along y direction!
    // line_v.clear();
    // x_input.clear();
    // for(int j=min_y;j<=max_y;j++) x_input.push_back(j);
    // for(int i=0;i<ndata;i++)
    // {
    //     tk::spline st(x_input,column_spe_at_x[i]);       
    //     line_v.push_back(st(line_y[i]));
    // }

    //Version 2, share same code with interp3
    //data[i*ydim+j]

    std::vector<double> data;
    for (int i = min_x; i <= max_x; i++)
    {
        for (int j = min_y; j <= max_y; j++)
        {
            data.push_back(spectrum_row[j * xdim + i]);
        }
    }
    ldw_math_dnn::interp2(min_x,max_x,min_y,max_y,data,line_x,line_y,line_v);
    return true;
};

bool peak2d::setup_peaks_from_p()
{
    inten.clear();
    sigmax.clear();
    gammax.clear();
    shoulx.clear();
    sigmay.clear();
    gammay.clear();
    shouly.clear();
    confidencex.clear();
    confidencey.clear();

    //At this time, cx,cy,p_2_column_paras and p_2_row_paras all have same length and contain peaks in the same order!!
    for (int i = 0; i < p_2_column_paras.size(); i++)
    {
        int ndx1 = p_2_column_paras[i];
        int ndx2 = p_2_row_paras[i];
        inten.push_back(std::min(a_column[ndx1], a_row[ndx2]));
        sigmay.push_back(s_column[ndx1]);
        gammay.push_back(g_column[ndx1]);
        shouly.push_back(sh_column[ndx1]);
        confidencey.push_back(conf_column[ndx1]);
        sigmax.push_back(s_row[ndx2]);
        gammax.push_back(g_row[ndx2]);
        shoulx.push_back(sh_row[ndx2]);
        confidencex.push_back(conf_row[ndx2]);
    }
    return true;
}

/*
bool::peak2d::check_special_peaks_1()
{
    //Construct line to peak pointer from peak to line pointer
    int n_cline=column_line_segment.size(); // number of column line
    int n_rline=row_line_segment.size();   // number of row line
    std::vector<std::vector<int> > cline_2_peak; //1, column line index, 2: peak index belongs to this line
    std::vector<std::vector<int> > rline_2_peak; 
    cline_2_peak.resize(n_cline);
    rline_2_peak.resize(n_rline);
    for(int i=0;i<p_2_line_column.size();i++)
    {
        cline_2_peak[p_2_line_column[i]].push_back(i);
        rline_2_peak[p_2_line_row[i]].push_back(i);
    }


    //need inten for following special case 1 code to run:
    inten.clear();
    for(int i=0;i<cx.size();i++)
    {
        inten.push_back(std::min(a_column[p_2_column_paras[i]],a_row[p_2_row_paras[i]]));
    }

    int ncut;
    if(model_selection==2) ncut=5;
    else ncut=8;


    //find special case 1, tigtly overallped 4 peaks!!
    std::vector<int> peak_neighbor_c,peak_neighbor_r,peak_neighbor_rc,peak_neighbor_cr;

    peak_neighbor_c.resize(cx.size(),-1);
    peak_neighbor_r.resize(cx.size(),-1);
    peak_neighbor_rc.resize(cx.size(),-1);
    peak_neighbor_cr.resize(cx.size(),-1);
    
    for(int i=0;i<n_cline;i++)
    {
        std::vector<int> xcoors,ndx;
        for(int j1=0;j1<cline_2_peak[i].size();j1++)
        {
            xcoors.push_back(cx[cline_2_peak[i][j1]]);
        }
        ldw_math_dnn::sortArr(xcoors,ndx); 

        for(int j2=1;j2<ndx.size();j2++)
        {
            if(abs(cx[cline_2_peak[i][ndx[j2]]]-cx[cline_2_peak[i][ndx[j2-1]]])<=ncut)
            {
                peak_neighbor_c[cline_2_peak[i][ndx[j2]]]=cline_2_peak[i][ndx[j2-1]];
            }
        }
    }

    for(int i=0;i<n_rline;i++)
    {
        std::vector<int> ycoors,ndx;
        for(int j1=0;j1<rline_2_peak[i].size();j1++)
        {
            ycoors.push_back(cy[rline_2_peak[i][j1]]);
        }
        ldw_math_dnn::sortArr(ycoors,ndx); 

        for(int j2=1;j2<ndx.size();j2++)
        {
            if(abs(cy[rline_2_peak[i][ndx[j2]]]-cy[rline_2_peak[i][ndx[j2-1]]])<=ncut)
            {
                peak_neighbor_r[rline_2_peak[i][ndx[j2]]]=rline_2_peak[i][ndx[j2-1]];
            }
        }
    }

    std::vector<int> to_remove;
    to_remove.clear();
    to_remove.resize(cx.size(),0);

    for(int i=0;i<cx.size();i++)
    {
        if(peak_neighbor_r[i]>=0)
        {
            peak_neighbor_rc[i]=peak_neighbor_c[peak_neighbor_r[i]];
        }
        if(peak_neighbor_c[i]>=0)
        {
            peak_neighbor_cr[i]=peak_neighbor_r[peak_neighbor_c[i]];
        }
        if(peak_neighbor_cr[i]>=0 && peak_neighbor_cr[i]==peak_neighbor_rc[i])
        {
            // std::cout<<"found quater at peak "<<cx[i]<<" "<<cy[i]<<std::endl;
            //lable removal of 2 out of 4 here.
            double a1=inten[i];
            double a2=inten[peak_neighbor_r[i]];
            double a3=inten[peak_neighbor_c[i]];
            double a4=inten[peak_neighbor_cr[i]];
            if((a1>=a2 && a1>=a3) ||  (a4>=a2 && a4>=a3))
            {
                to_remove[peak_neighbor_r[i]]=1;
                to_remove[peak_neighbor_c[i]]=1;
            }
            else
            {
                to_remove[i]=1;
                to_remove[peak_neighbor_rc[i]]=1;
            }
        }
    }

    for(int i=cx.size()-1;i>=0;i--)
    {
        if(to_remove[i]==1)
        {
            cx.erase(cx.begin()+i);
            cy.erase(cy.begin()+i);
            to_remove.erase(to_remove.begin()+i);
            p_2_column_paras.erase(p_2_column_paras.begin()+i);
            p_2_row_paras.erase(p_2_row_paras.begin()+i); 
            p_2_line_column.erase(p_2_line_column.begin()+i); 
            p_2_line_row.erase(p_2_line_row.begin()+i); 
        }
    }
    //afer above remove, cline_2_peak and rline_2_peak is not valid any longer!!!
}
*/

bool peak2d::check_special_peaks_2()
{
    int n_cline=column_line_segment.size(); // number of column line
    int n_rline=row_line_segment.size();   // number of row line
    std::vector<std::vector<int> > cline_2_peak; //1, column line index, 2: peak index belongs to this line
    std::vector<std::vector<int> > rline_2_peak; 
    cline_2_peak.resize(n_cline);
    rline_2_peak.resize(n_rline);
    for(int i=0;i<p_2_line_column.size();i++)
    {
        cline_2_peak[p_2_line_column[i]].push_back(i);
        rline_2_peak[p_2_line_row[i]].push_back(i);
    }

    peak_exclude.resize(cx.size(),0);


    //need inten for following special case code to run:
    inten.clear();
    for(int i=0;i<cx.size();i++)
    {
        inten.push_back(std::min(a_column[p_2_column_paras[i]],a_row[p_2_row_paras[i]]));
    }  

    std::vector<int> peak_neighbor_c,peak_neighbor_r,peak_neighbor_rc,peak_neighbor_cr;

    peak_neighbor_c.resize(cx.size(),-1);
    peak_neighbor_r.resize(cx.size(),-1);
   
    
    for(int i=0;i<n_cline;i++)
    {
        std::vector<int> xcoors,ndx;
        for(int j1=0;j1<cline_2_peak[i].size();j1++)
        {
            xcoors.push_back(cx[cline_2_peak[i][j1]]);
        }
        ldw_math_dnn::sortArr(xcoors,ndx); 

        for(int j2=1;j2<ndx.size();j2++)
        {
            if(abs(cx[cline_2_peak[i][ndx[j2]]]-cx[cline_2_peak[i][ndx[j2-1]]])<=20)
            {   
                peak_neighbor_c[cline_2_peak[i][ndx[j2]]]=cline_2_peak[i][ndx[j2-1]];
                int k1=cline_2_peak[i][ndx[j2]];
                int k2=cline_2_peak[i][ndx[j2-1]];
                double a1=a_column[p_2_column_paras[k1]];
                double a2=a_column[p_2_column_paras[k2]];
                double r=fabs(double(cy[k1]-cy[k2])/double(cx[k1]-cx[k2]));

                if(a1/a2<=4.0 && a2/a1<=4.0 && r>=0.1)
                {
                    // std::cout<<"A are "<<a1<<" "<<a1<<std::endl;
                    int k3=p_2_line_column[k1];
                    int k4=p_2_line_column[k2];
                    // std::cout<<"they belong to column line "<<k3<<" and "<<k4<<std::endl;

                    int b,s;
                    if(k3==0) b=0;
                    else b=column_line_segment[k3-1];
                    s=column_line_segment[k3];

                    int ndx1=-1;
                    int ndx2=-1;
                    for(int i=b;i<s;i++)
                    {
                        if(cx[k1]==column_line_x[i] && cy[k1]==column_line_y[i]) {ndx1=i;}   
                        if(cx[k2]==column_line_x[i] && cy[k2]==column_line_y[i]) {ndx2=i;} 
                    }

                    if(ndx2<ndx1)
                    {
                        int t=ndx1;
                        ndx1=ndx2;
                        ndx2=t;
                    }
                    ndx2++;
                    
                    double min_a=a1;
                    for(int i=ndx1;i<ndx2;i++)
                    {
                       if(a_column[column_line_index[i]]<min_a)
                       {
                          min_a=a_column[column_line_index[i]];
                       }
                    }
                    if(min_a>0.5*std::min(a1,a2))
                    {
                        //exclude these two from special_case3!!
                        peak_exclude[k1]=1;
                        peak_exclude[k2]=1;
                    }
                    // std::cout<<std::endl;
                }
            }
        }
    }

    for(int i=0;i<n_rline;i++)
    {
        std::vector<int> ycoors,ndx;
        for(int j1=0;j1<rline_2_peak[i].size();j1++)
        {
            ycoors.push_back(cy[rline_2_peak[i][j1]]);
        }
        ldw_math_dnn::sortArr(ycoors,ndx); 

        for(int j2=1;j2<ndx.size();j2++)
        {
            if(abs(cy[rline_2_peak[i][ndx[j2]]]-cy[rline_2_peak[i][ndx[j2-1]]])<=20)
            {
                peak_neighbor_r[rline_2_peak[i][ndx[j2]]]=rline_2_peak[i][ndx[j2-1]];
                int k1=rline_2_peak[i][ndx[j2]];
                int k2=rline_2_peak[i][ndx[j2-1]];
                double a1=a_row[p_2_row_paras[k1]];
                double a2=a_row[p_2_row_paras[k2]];
                double r=fabs(double(cx[k1]-cx[k2])/double(cy[k1]-cy[k2]));
            
                if(a1/a2<=4.0 && a2/a1<=4.0 && r>=0.1)
                {
                    int k3=p_2_line_row[k1];
                    int k4=p_2_line_row[k2];  //k4 should == k3


                    int b,s;
                    if(k3==0) b=0;
                    else b=row_line_segment[k3-1];
                    s=row_line_segment[k3];

                    int ndx1=-1;
                    int ndx2=-1;
                    for(int i=b;i<s;i++)
                    {
                        if(cx[k1]==row_line_x[i] && cy[k1]==row_line_y[i]) {ndx1=i;}   
                        if(cx[k2]==row_line_x[i] && cy[k2]==row_line_y[i]) {ndx2=i;} 
                    }

                    if(ndx2<ndx1)
                    {
                        int t=ndx1;
                        ndx1=ndx2;
                        ndx2=t;
                    }
                    ndx2++;

                    double min_a=a1;
                    for(int i=ndx1;i<ndx2;i++)
                    {
                       if(a_row[row_line_index[i]]<min_a)
                       {
                          min_a=a_row[row_line_index[i]];
                       }
                    }
                    if(min_a>0.5*std::min(a1,a2))
                    {
                        //exclude these two from special_case3!!
                        peak_exclude[k1]=1;
                        peak_exclude[k2]=1;
                    }
                    // std::cout<<std::endl;
                }
            }
        }
    }  
    return true;
}

bool peak2d::check_special_peaks_3()
{
    int npeak_current=cx.size();

    std::vector<int> to_remove;

    to_remove.clear();
    to_remove.resize(npeak_current,0.0);


    for(int i=npeak_current-1;i>=0;i--)
    {
        // std::cout<<"check_special_case for peak "<<i<<" is started."<<std::endl;

        // if(peak_exclude[i]==1) continue;
        if( inten[i] < noise_level * user_scale ) continue;
        if( shoulx[i]==1 || shouly[i]==1 ) continue; 
        // if( confidencex[i]<0.7 || confidencey[i]<0.7) continue;

        int m=p_2_line_column[i];
        int bc,sc;

        if(m==0){
            bc=0;
        }
        else{
            bc=column_line_segment[m-1];
        }
        sc=column_line_segment[m];

        int n=p_2_line_row[i];
        int br,sr;

        if(n==0){
            br=0;
        }
        else{
            br=row_line_segment[n-1];
        }
        sr=row_line_segment[n];

        std::vector<int> cline_x(column_line_x.begin()+bc,column_line_x.begin()+sc);
        std::vector<int> cline_y(column_line_y.begin()+bc,column_line_y.begin()+sc);
        std::vector<int> cline_ndx(column_line_index.begin()+bc,column_line_index.begin()+sc);
        std::vector<int> rline_x(row_line_x.begin()+br,row_line_x.begin()+sr);
        std::vector<int> rline_y(row_line_y.begin()+br,row_line_y.begin()+sr);
        std::vector<int> rline_ndx(row_line_index.begin()+br,row_line_index.begin()+sr);
        std::vector<int> new_x,new_y,new_cline_ndx,new_rline_ndx;

        double fwhhx=1.0692*gammax[i]+sqrt(0.8664*gammax[i]*gammax[i]+5.5452*sigmax[i]*sigmax[i]);
        double fwhhy=1.0692*gammay[i]+sqrt(0.8664*gammay[i]*gammay[i]+5.5452*sigmay[i]*sigmay[i]);

        auto result_x = std::minmax_element (cline_x.begin(),cline_x.end());
        auto result_y = std::minmax_element (rline_y.begin(),rline_y.end());

        if( (cx[i]==1274 && (cy[i]==1521 || cy[i]==1522)) || (cx[i]==1076 && cy[i]==846))
        {
            std::cout<<"x is from "<<*result_x.first<<" to "<<*result_x.second<<std::endl;
            std::cout<<"y is from "<<*result_y.first<<" to "<<*result_y.second<<std::endl;
        }
        
        int n_min=6;
        if(model_selection==1)
        {
            n_min=10;
        }

        bool b11 = *result_x.first < cx[i]-n_min;  //line is too short on the left side of the testing peak
        bool b12 = a_column[cline_ndx[result_x.first-cline_x.begin()]]<noise_level*user_scale2*3.0; //end of line appraoch noise floor

        bool b21 = *result_x.second > cx[i]+n_min; //right side of the testing peak
        bool b22 = a_column[cline_ndx[result_x.second-cline_x.begin()]]<noise_level*user_scale2*3.0;

        bool b31 = *result_y.first < cy[i]-n_min;  //bottom side
        bool b32 = a_row[rline_ndx[result_y.second-rline_y.begin()]]<noise_level*user_scale2*3.0;

        bool b41 = *result_y.second < cy[i]+n_min; //top side
        bool b42 = a_row[rline_ndx[result_y.second-rline_y.begin()]]<noise_level*user_scale2*3.0;

        if( !(b11 || b12) || !(b21 || b22) || !(b31 || b32) || !(b41 || b42) ) 
        {
            continue;
        }
        

        check_special_case(i,fwhhx,fwhhy,cline_x,cline_y,cline_ndx,rline_x,rline_y,rline_ndx,new_x,new_y,new_cline_ndx,new_rline_ndx);
        // std::cout<<"check_special_case for peak "<<i<<" is done."<<std::endl;

        if(new_x.size()>0)
        {
            to_remove[i]=1;
            for(int j=0;j<new_x.size();j++)
            {
                cx.push_back(new_x[j]);
                cy.push_back(new_y[j]);
                p_2_column_paras.push_back(new_cline_ndx[j]);
                p_2_row_paras.push_back(new_rline_ndx[j]);
            }
        }
    }

    //p_2_line_column, p_2_line_row, etc are all lost here!!!!!
    for (int i = npeak_current - 1; i >= 0; i--)
    {
        if (to_remove[i] == 1)
        {
            cx.erase(cx.begin() + i);
            cy.erase(cy.begin() + i);
            p_2_column_paras.erase(p_2_column_paras.begin()+i);
            p_2_row_paras.erase(p_2_row_paras.begin()+i);
        }
    }
    return true;
}


bool peak2d::get_tilt_of_line(const int flag, const int x,const int y,const double w, const std::vector<int> &line_x,const std::vector<int> &line_y, const std::vector<int> &line_ndx, int &k1m, int &k2m, double &ratio)
{

    int ndx=-1;

    for(int i=0;i<line_x.size();i++)
    {
        if(line_x[i]==x && line_y[i]==y)
        {
            ndx=i;
            break;
        }
    }

    std::vector<double> *a;
    if(flag==1)
    {
        a=&a_column;
    }
    else
    {
        a=&a_row;
    }
    
    

    int cut2=100;
    int cut1=1;
    

    k1m=-1;
    k2m=-1;
    ratio=0.0;
    double ratio_abs=0.0;
    if(ndx>=cut1 && ndx+cut1<line_x.size())
    {
        double max_a=0;
        for(int i=ndx-1;i<=ndx+1;i++)
        {
            double t=a->at(line_ndx[i]);
            
            if(t>max_a)
            {
                max_a=t;
            }
        }

        std::vector<int> cline_x_left(line_x.begin()+std::max(ndx-cut2,0),line_x.begin()+ndx-cut1+1);
        std::vector<int> cline_y_left(line_y.begin()+std::max(ndx-cut2,0),line_y.begin()+ndx-cut1+1);
        std::vector<int> cline_ndx_left(line_ndx.begin()+std::max(ndx-cut2,0),line_ndx.begin()+ndx-cut1+1);
        std::vector<int> cline_x_right(line_x.begin()+ndx+cut1,line_x.begin()+std::min(ndx+cut2+1,int(line_x.size())));
        std::vector<int> cline_y_right(line_y.begin()+ndx+cut1,line_y.begin()+std::min(ndx+cut2+1,int(line_y.size())));
        std::vector<int> cline_ndx_right(line_ndx.begin()+ndx+cut1,line_ndx.begin()+std::min(ndx+cut2+1,int(line_ndx.size())));

        for(int k1=0;k1<cline_x_left.size();k1++)
        {
            double current_a=a->at(cline_ndx_left[k1]);
            if(current_a<0.3*max_a || current_a>max_a) continue; //at least 30% of max intensity to be considerted.
            for(int k2=0;k2<cline_x_right.size();k2++)
            {
                double current_a=a->at(cline_ndx_right[k2]);
                if(current_a<0.3*max_a || current_a>max_a) continue;
                double dx=cline_x_right[k2]-cline_x_left[k1];
                double dy=cline_y_right[k2]-cline_y_left[k1];
                if(fabs(dx)>=w*0.6 && fabs(dx)>=4 && fabs(dy/dx)>ratio_abs)
                {
                    ratio_abs=fabs(dy/dx);
                    ratio=dy/dx;
                    k1m=k1;
                    k2m=k2;
                }
            }
        }
        // std::cout<<std::endl;
    }

    k1m+=std::max(ndx-cut2,0);
    k2m+=ndx+cut1;

    return true;
}

bool peak2d::check_near_peak(const int x0, const int y0,const int x, const int y)
{
    bool b=true;
    for(int i=0;i<cx.size();i++)
    {
        if(abs(cx[i]-x0)+abs(cy[i]-y0)<=1)
        {
            continue;
        }
        if(abs(cx[i]-x)+abs(cy[i]-y)<=4)
        {
            b=false;
            break;
        }
    }
    return b;
}


bool peak2d::check_special_case(const int ndx,const double fx,const double fy,
                                std::vector<int> &cline_x,std::vector<int> &cline_y,std::vector<int> &cline_ndx,
                                std::vector<int> &rline_x,std::vector<int> &rline_y,std::vector<int> &rline_ndx,
                                std::vector<int> &xn,std::vector<int> &yn,std::vector<int> &cline_n, std::vector<int> &rline_n)
{
    //cline is from column prediciton, that is, cline is mainly horizental
    int k1,k2,m1,m2;
    double ratio1,ratio2;
    bool b1,b2;
    int potential_x1,potential_x2,potential_y1,potential_y2;
    int x=cx[ndx];
    int y=cy[ndx];

    std::vector<int> ndx_neighbors=ldw_math_dnn::find_neighboring_peaks(cx,cy,ndx);

    cut_one_peak_v2(cline_x,cline_y,cline_ndx,ndx,ndx_neighbors); //remove segment belongs to other peaks
    cut_one_peak_v2(rline_x,rline_y,rline_ndx,ndx,ndx_neighbors); //remove segment belongs to other peaks
    
    //TO DO: move volume check from get_tilt_of_line to cut_one_peak_v2

    get_tilt_of_line(1,x,y,std::min(fx,fy),cline_x,cline_y,cline_ndx,k1,k2,ratio1);
    get_tilt_of_line(2,y,x,std::min(fx,fy),rline_y,rline_x,rline_ndx,m1,m2,ratio2);

    b1= ratio1>=0.25 &&  ratio2>=0.25 &&  ratio1+ratio2>=0.6;
    b2=ratio1<=-0.25 && ratio2<=-0.25 && ratio1+ratio2<=-0.6;

    //debug output for spectra3\pseudo_3d\g12d_gtp_cpmg\first
    // if(x==1202 && y==348)
    //debug output for spectra3\protein\new_asynu,//debug output for spectra3\protein\new_asynu_a    
    // if( (x==1274 && (y==1521 || y==1522)) || (x==1076 && y==846))
    if( x==1390 && y==222 ) //pseudo/g12d_gtp_cpmg/first
    {
        std::cout<<"cline:"<<std::endl;
        for(int i=0;i<cline_x.size();i++)
        {
            std::cout<<cline_x[i]<<" "<<cline_y[i]<<std::endl;
        }
        std::cout<<"rline:"<<std::endl;
        for(int i=0;i<rline_x.size();i++)
        {
            std::cout<<rline_x[i]<<" "<<rline_y[i]<<std::endl;
        }
        std::cout<<"ratio1="<<ratio1<<" ratio2="<<ratio2<<std::endl;
    }

    if( (b1 || b2) && (check_near_peak(x,y,cline_x[k1],cline_y[k1]) && check_near_peak(x,y,rline_x[m1],rline_y[m1]) && check_near_peak(x,y,cline_x[k2],cline_y[k2]) && check_near_peak(x,y,rline_x[m2],rline_y[m2])) )
    {
        // std::cout<<"Add addtional peaks because ratio 1 is "<<ratio1<<" and ratio2 is"<<ratio2<<" from peak "<<x<<" "<<y<<std::endl;
        if(b1)
        {
            potential_x1=int(double(cline_x[k1]+rline_x[m1])/2.0);
            potential_y1=int(double(cline_y[k1]+rline_y[m1])/2.0);
            potential_x2=int(double(cline_x[k2]+rline_x[m2])/2.0);
            potential_y2=int(double(cline_y[k2]+rline_y[m2])/2.0);
            
            xn.push_back(potential_x1);
            yn.push_back(potential_y1); 
            cline_n.push_back(cline_ndx[k1]);
            rline_n.push_back(rline_ndx[m1]);

            xn.push_back(potential_x2);
            yn.push_back(potential_y2); 
            cline_n.push_back(cline_ndx[k2]);
            rline_n.push_back(rline_ndx[m2]);
        }
        if(b2)
        {
            potential_x1=int(double(cline_x[k1]+rline_x[m2])/2.0);
            potential_y1=int(double(cline_y[k1]+rline_y[m2])/2.0);
            potential_x2=int(double(cline_x[k2]+rline_x[m1])/2.0);
            potential_y2=int(double(cline_y[k2]+rline_y[m1])/2.0);
        
            xn.push_back(potential_x1);
            yn.push_back(potential_y1); 
            cline_n.push_back(cline_ndx[k1]);
            rline_n.push_back(rline_ndx[m2]);

            xn.push_back(potential_x2);
            yn.push_back(potential_y2); 
            cline_n.push_back(cline_ndx[k2]);
            rline_n.push_back(rline_ndx[m1]);
        }
    }
    
    return true;
}


bool peak2d::find_lines(int xd, int yd, std::vector<int> r, std::vector<int> &x0, std::vector<int> &y0, std::vector<int> &ndx0, std::vector<int> &s0)
{

    int min_line_length;
    int limit;

    if(model_selection==1 || model_selection==4 ) //peak is wide
    {
        min_line_length=5;
        limit=4;
    }
    else if(model_selection==2 || model_selection==3)
    {
        min_line_length=3;
        limit=2;
    }

    std::vector<int> x,y,ndx,s;
    x.clear();
    y.clear();
    ndx.clear();
    s.clear();

    for(int i=0;i<xd;i++)
    {
        for(int j=0;j<yd;j++)
        {
            int current_length=x.size();
            if(r[i*yd+j]>=0)
            {
                x.push_back(i);
                y.push_back(j);
                ndx.push_back(r[i*yd+j]);
                int cx=i;
                int cy=j;
                bool b=true;
                r[i*yd+j]=-1;
                while(b)
                {
                    b=false;
                    int dis=0;
                    while(dis<=limit && cx<xd-1 && cy-dis>=0 && cy+dis<=yd-1)
                    {
                        bool bfound=false;
                        if(r[(cx+1)*yd+cy+dis]>=0)
                        {
                            if(dis==1)
                            {
                                x.push_back(cx);
                                y.push_back(cy+1);
                                ndx.push_back(r[(cx+1)*yd+cy+1]);
                            }
                            else if(dis==2)
                            {
                                x.push_back(cx);
                                y.push_back(cy+1);
                                ndx.push_back(ndx[ndx.size()-1]);
                                x.push_back(cx+1);
                                y.push_back(cy+1);   
                                ndx.push_back(r[(cx+1)*yd+cy+dis]);
                            }
                            else if(dis==3)
                            {
                                x.push_back(cx);
                                y.push_back(cy+1);
                                ndx.push_back(ndx[ndx.size()-1]);
                                x.push_back(cx+1);
                                y.push_back(cy+1);   
                                ndx.push_back(r[(cx+1)*yd+cy+dis]);   
                                x.push_back(cx+1);
                                y.push_back(cy+2);   
                                ndx.push_back(r[(cx+1)*yd+cy+dis]);
                            }
                            else if(dis==4)
                            {
                                x.push_back(cx);
                                y.push_back(cy+1);
                                ndx.push_back(ndx[ndx.size()-1]);
                                x.push_back(cx);
                                y.push_back(cy+2);
                                ndx.push_back(ndx[ndx.size()-2]);
                                x.push_back(cx+1);
                                y.push_back(cy+2);   
                                ndx.push_back(r[(cx+1)*yd+cy+dis]);   
                                x.push_back(cx+1);
                                y.push_back(cy+3);   
                                ndx.push_back(r[(cx+1)*yd+cy+dis]);
                            }
                            cx=cx+1;
                            cy=cy+dis;
                            bfound=true;  
                        }
                        else if(r[(cx+1)*yd+cy-dis]>=0)
                        {
                            if(dis==1)
                            {
                                x.push_back(cx);
                                y.push_back(cy-1);
                                ndx.push_back(r[(cx+1)*yd+cy-1]);
                            }
                            else if(dis==2)
                            {
                                x.push_back(cx);
                                y.push_back(cy-1);
                                ndx.push_back(ndx[ndx.size()-1]);
                                x.push_back(cx+1);
                                y.push_back(cy-1);   
                                ndx.push_back(r[(cx+1)*yd+cy-dis]);
                            }
                            else if(dis==3)
                            {
                                x.push_back(cx);
                                y.push_back(cy-1);
                                ndx.push_back(ndx[ndx.size()-1]);
                                x.push_back(cx+1);
                                y.push_back(cy-1);   
                                ndx.push_back(r[(cx+1)*yd+cy-dis]);   
                                x.push_back(cx+1);
                                y.push_back(cy-2);   
                                ndx.push_back(r[(cx+1)*yd+cy-dis]);   
                            }
                            else if(dis==4)
                            {
                                x.push_back(cx);
                                y.push_back(cy-1);
                                ndx.push_back(ndx[ndx.size()-1]);
                                x.push_back(cx);
                                y.push_back(cy-2);
                                ndx.push_back(ndx[ndx.size()-2]);
                                x.push_back(cx+1);
                                y.push_back(cy-2);   
                                ndx.push_back(r[(cx+1)*yd+cy-dis]);   
                                x.push_back(cx+1);
                                y.push_back(cy-3);   
                                ndx.push_back(r[(cx+1)*yd+cy-dis]);   
                            }
                            cx=cx+1;
                            cy=cy-dis;
                            bfound=true;
                        }

                        if(bfound==true)
                        {
                            x.push_back(cx);
                            y.push_back(cy);
                            ndx.push_back(r[cx*yd+cy]);
                            r[cx*yd+cy]=-1;
                            b=true;
                            break;
                        }
                        dis++;
                    }
                }

                if(x.size()-current_length>=min_line_length)
                {
                    s.push_back(x.size());
                }
                else
                {
                    x.erase(x.begin()+current_length,x.end());
                    y.erase(y.begin()+current_length,y.end());   
                    ndx.erase(ndx.begin()+current_length,ndx.end());
                }
                // std::cout<<"current length of line is "<<current_length<<std::endl;
            }
        }
    }

    x0 = x;
    y0 = y;
    s0 = s;
    ndx0 = ndx;

    return true;
};

bool peak2d::predict()
{
    predict_step1();
    std::cout<<"Finished 1D prediction."<<std::endl;
    predict_step2();
    std::cout<<"Get lines from dots. Done."<<std::endl;
    predict_step3();
    std::cout<<"Finished ANN peak picking."<<std::endl;
    return true;
}

bool peak2d::print_prediction()
{
     std::ofstream fout;
    fout.open("predicted_peaks.txt");
    for(int i=0;i<cx.size();i++)
    {
        fout<<cx[i]<<" "<<cy[i]<<" ";
        fout<<inten[i]<<" ";
        fout<<sigmax[i]<<" "<<gammax[i]<<" "<<shoulx[i]<<" ";
        fout<<sigmay[i]<<" "<<gammay[i]<<" "<<shouly[i]<<" ";
        fout<<confidencex[i]<<" "<<confidencey[i];
        fout<<std::endl;
    }
    fout.close();

    return true;
}

bool peak2d::extract_result(std:: vector<double> &p1,std:: vector<double> &p2,std:: vector<double> &p_intensity,
        std:: vector<double> &sx,std:: vector<double> &sy,std:: vector<double> &gx,std:: vector<double> &gy,std::vector<int> &p_type, std::vector<double> &confx, std::vector<double> &confy)
{
    p1.clear();
    p2.clear();

    for(int i=0;i<cx.size();i++)
    {
        p1.push_back(double(cx[i]));
        p2.push_back(double(cy[i]));

        if(shoulx[i]==1 && shouly[i]==1) 
        {
            p_type.push_back(2); //shoulder peak
        }
        else
        {
            p_type.push_back(1); //normal peak
        }
        
    }

    p_intensity=inten;
    sx=sigmax;
    sy=sigmay;
    gx=gammax;
    gy=gammay;
    confx=confidencex;
    confy=confidencey;
    
    return true;
}