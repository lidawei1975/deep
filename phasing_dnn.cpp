#include <cmath>
#include "phasing_dnn.h"


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



dense::dense() {
    a=linear;
};
dense::~dense() {};

    void dense::set_size(int n, int k) 
    {
        ninput=n;
        nfilter=k;
    };

bool dense::set_act(enum activation_function a_)
{
    a=a_;
    return true;
}


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


//phase_estimator
phase_estimator::phase_estimator() {};
phase_estimator::~phase_estimator() {};

bool phase_estimator::init(std::vector<int> x, std::vector<int> y,int pos_,double w_, std::vector<double> ps_)
{
    ps=ps_;
    left_scores=x;
    right_scores=y;

    peak_pos=pos_;
    weight=w_;

    //find index of first element whose value is 1 in left_scores
    positive_edge=0;
    while(positive_edge<left_scores.size() && left_scores[positive_edge]!=1) positive_edge++;
    positive_edge=positive_edge>=ps.size()?180.0:ps[positive_edge];

    //find index of last element whose value is 1 in right_scores
    negative_edge=right_scores.size()-1;
    while(negative_edge>=0 && right_scores[negative_edge]!=1) negative_edge--;
    negative_edge=negative_edge<0?-180.0:ps[negative_edge];

    return true;
};

bool phase_estimator::print()
{
    std::cout<<"positive_edge from left  score is "<<positive_edge<<std::endl;
    std::cout<<"negative_edge from right score is "<<negative_edge<<std::endl;  
    return true;
}


double phase_estimator::get_cost(double phase)
{
    double cost=0.0;


    if(phase>=positive_edge)
    {
        cost+=1.0+phase-positive_edge;
    }
    else if(phase>=positive_edge-3.0)
    {
        cost+=(phase-(positive_edge-3.0))/10.0;
    }
    
    if(phase<=negative_edge)
    {
        cost+=1.0+negative_edge-phase;
    }
    else if(phase<=negative_edge+3.0)
    {
        cost+=(negative_edge+3.0-phase)/10.0;
    }
   
    return cost;
}