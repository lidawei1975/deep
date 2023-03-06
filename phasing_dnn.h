#include <vector>
#include <string> 
#include <iostream>

enum activation_function{linear, relo, softmax};
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
    bool set_act(enum activation_function a_);
};


//
class phase_estimator
{
private:
    std::vector<double> ps;
    std::vector<int> left_scores; //prevent positive phase error
    std::vector<int> right_scores; //prevent negative phase error
public:
    int positive_edge, negative_edge;
    int peak_pos;
    double weight;

    phase_estimator();
    ~phase_estimator();

    bool print();
    bool init(std::vector<int>, std::vector<int>,int,double,std::vector<double>);
    double get_cost(double);
};
