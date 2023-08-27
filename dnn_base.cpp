#include <cmath>
#include <algorithm>
#include "dnn_base.h"

pool1d::pool1d(){};
pool1d::~pool1d(){};

/**
 * @brief set the size of the max pooling layer
 * 
 * @param m number of datapoint of each input, output
 * @param n size of the pooling window
 */
void pool1d::set_size(int m, int n)
{
    nfilter = m;
    npool = n;
}

/**
 * @brief rum max pooling. Input and output are saved row by row as a 1D vector with size [nlen][nfilter]
 * Pooling is done along the first dimension
 * @param nlen: length of input
 * @param input length is nlen*nfilter. Data organization: input[nlen][nfilter]
 * @param output output length is same as input nlen*nfilter
 * @return true 
 * @return false 
 */
bool pool1d::predict(int nlen, std::vector<float> &input, std::vector<float> &output) const
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

/**
 * @brief matrix multiplication operation. All matrix is saved row by row as a 1D vector
 * 
 * @param in1 input matrix 1, size is m by n in1[m][n]
 * @param in2 input matrix 2, size is n by k in2[n][k]
 * @param out output matrix, size is m by k out[m][k]
 * @param m 
 * @param n 
 * @param k 
 * @return true 
 */
bool base1d::mat_mul(const std::vector<float> &in1, const std::vector<float> &in2, std::vector<float> &out, int m, int n, int k) const
{

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

/**
 * @brief print kernel and bias, for debug purpose
 * 
 * @return true 
 * @return false 
 */
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

/**
 * @brief Construct a new dense::dense object
 * default activation function is linear
 */
dense::dense()
{
    a = linear;
};
dense::~dense(){};

/**
 * @brief set size of kernel and bias
 * 
 * @param n: input size 
 * @param k: number of filters: output size
 */
void dense::set_size(int n, int k)
{
    ninput = n;
    nfilter = k;
};

/**
 * @brief set activation function
 * 
 * @param a_ activation function: linear, relu, softmax
 * @return true 
 * @return false 
 */
bool dense::set_act(enum activation_function a_)
{
    a=a_;
    return true;
}

/**
 * @brief read kernel and bias from a 1D vector
 * 
 * @param p pointer to 1D vector (float)
 * @return int: total number of elements read
 */
int dense::read(float *p)
{
    kernel.resize(ninput*nfilter); 
    bias.resize(nfilter);   
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
/**
 * @brief Run dense layer: nlen is number of input vectors (1D length of input vector is ninput, 1D length of output vector is nfilter)
 * All matrix is saved row by row as a 1D vector
 * @param nlen 
 * @param input: input[nlen][ninput], saved row by row as a 1D vector
 * @param output output[nlen][nfilter], saved row by row as a 1D vector
 * @return true 
 * @return false 
 */
bool dense::predict(int nlen, std::vector<float> &input, std::vector<float> &output) const
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
    if(a==relu)
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
    else if(a==selu)
    {
        for(int j=0;j<nlen*nfilter;j++)
        {
            if(output[j]>0.0f)
            {
                output[j]=1.05070098f*output[j];
            }
            else
            {
                output[j]=1.05070098f*1.67326324f*(exp(output[j])-1.0f);
            }
        }
    }
    //do nothing if it is linear activation

    return true;
};


/**
 * @brief Construct a new conv1d::conv1d object
 * 
 */
conv1d::conv1d(){};
conv1d::~conv1d(){};

/**
 * @brief set size of kernel and bias
 * 
 * @param m: size of kernels ( or number of kernels)
 * @param n: input size (how many channels, not length of 1D input)
 * @param k: number of filters (number of channels of output)
 */
void conv1d::set_size(int m, int n, int k)
{
    nkernel = m;
    ninput = n;
    nfilter = k;
}

/**
 * @brief read kernel and bias from a 1D vector
 * 
 * @param p: pointer to 1D vector (float)
 * @return int: total number of elements read
 */
int conv1d::read(float *p)
{
    kernel.resize(nkernel*ninput*nfilter); 
    bias.resize(nfilter); 
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

/**
 * @brief Run conv1d layer
 * 
 * @param nlen: length of input data
 * @param input [nlen][ninput]
 * @param output [nlen][nfilter]
 * @return true 
 * @return false 
 */
bool conv1d::predict(int nlen, std::vector<float> &input, std::vector<float> &output) const
{
    /**
     * @brief We have total nkernel kernels, each kernel block has ninput*nfilter elements
     * Important: nkernel is same thing as size of kernel in CNN 
     * because convolution operation can be done one by one for each input element (apply all kernels on it, like here)
     * or one by one for each output element (apply one kernel on all reqruied input elements)
     */
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

    //relu activation
    for(int j=0;j<nlen*nfilter;j++)
    {
        output[j]=std::max(output[j],0.0f);    
    }
    return true;
};