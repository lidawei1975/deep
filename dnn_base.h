#include <vector>
#include <string> 
#include <iostream>

#ifndef DNN_BASE
#define DNN_BASE

enum activation_function{linear, relu, selu, softmax};

//max pooling layer
class pool1d
{
private:
    int nfilter,npool;

public:    
    pool1d();
    ~pool1d();
    bool predict(int,std::vector<float> &, std::vector<float> &) const;
    void set_size(int m,int n);
};

//base class for both dense and conv
class base1d
{
protected:
    int nfilter;
    int ninput;

    std::vector<float> kernel;
    std::vector<float> bias;
    bool mat_mul(const std::vector<float> &, const std::vector<float> &, std::vector<float> &, int,int,int) const;

public:
    bool print();
    base1d();
    ~base1d();
};

//dense connected layer
class dense: public base1d
{
private:
    enum activation_function a;
public:
    dense();
    ~dense();

    bool predict(int, std::vector<float> &, std::vector<float> &) const;
    int read(float *);
    void set_size(int n, int k);
    bool set_act(enum activation_function a_);
};

//convalution layer class. Relu only for now
class conv1d: public base1d
{
private:
    int nkernel;

public:
    conv1d();
    ~conv1d();
    int read(float *);
    bool predict(int,std::vector<float> &, std::vector<float> &) const;
    void set_size(int m,int n, int k);
};

#endif