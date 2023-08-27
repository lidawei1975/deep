#include <vector>
#include <string> 
#include <iostream>
#include "dnn_base.h"

class fwhh_estimator
{
private:
    dense d1,d2,d3,d4,d5; //5 dense layers

public:
    
    fwhh_estimator();
    ~fwhh_estimator();

    float predict(std::vector<float> &);
};

