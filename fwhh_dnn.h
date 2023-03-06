#include <vector>
#include <string> 
#include <iostream>

#ifndef FWHH_DNN_H
#define FWHH_DNN_H

enum activation_function{linear, relo, selu, softmax};
//base class for both dense and conv
class base1d
{
protected:
    int nfilter;
    int ninput;

    std::vector<float> kernel;
    std::vector<float> bias;
    bool mat_mul(std::vector<float> &, std::vector<float> &, std::vector<float> &, int,int,int);

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

    bool predict(int, std::vector<float> &, std::vector<float> &);
    int read(float *);
    void set_size(int n, int k);
    bool set_activation_function(enum activation_function a_);
};

#endif

class fwhh_estimator
{
private:
    dense d1,d2,d3,d4,d5; //5 dense layers

public:
    
    fwhh_estimator();
    ~fwhh_estimator();

    float predict(std::vector<float> &);
};

