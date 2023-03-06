
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include "omp.h"

#include "json/json.h"

//will link to the C function in libcerf
extern "C"  
{
    double voigt(double x, double sigma, double gamma);
};

/**
 * This program will read in one peak from STDIN (3 numbers for each peak: sigma, gamma, grid size; plus one for drop cutoff)
 * grid size will be decided by this program if input is <=0
 * There is no need to input peak height. This program will normalize the peak to 1.0.
 * and output the peak shape to STDOUT (JSON format of an array, each element is 2 arrays: x and y)
 */

int main(int argc, char **argv)
{
    double center, sigma, gamma, grid_size, drop_cutoff;
    std::string line;
    //get line from STDIN
    std::getline(std::cin, line);

    //separate the line into 5 numbers
    std::istringstream iss(line);
    std::vector<double> inputs;
    double n;
    while (iss >> n)
    {
        inputs.push_back(n);
    }   

    //input size must be 4,7,10, ...
    if(inputs.size()>=4 && (inputs.size()-1)%3 == 0)
    {
        Json::Value root;
        drop_cutoff = inputs[0];

        for(int k=0;k<(inputs.size()-1)/3;k++)
        {
            sigma = fabs(inputs[1+k*3+0]);
            gamma = fabs(inputs[1+k*3+1]);
            grid_size = inputs[1+k*3+2];
        

            //make sure drop_cutoff is from 0.01 to 0.50
            if(drop_cutoff < 0.01)
            {
                drop_cutoff = 0.01;
            }
            else if(drop_cutoff > 0.50)
            {
                drop_cutoff = 0.50;
            }

            if(grid_size <= 0)
            {
                double fwhh = 0.5346 * sigma * 2 + std::sqrt(0.2166 * 4 * gamma * gamma + sigma * sigma * 8 * 0.6931);
                grid_size = fwhh / 10;
            }

            std::vector<double> xs, ys;
            double y_center = voigt(0, sigma, gamma);
            for(int i = 1; i < 100; i++) //100 is the maximal number of points along one side
            {
                double x = i*grid_size;
                double y = voigt(x, sigma, gamma) / y_center;
                if(y<drop_cutoff)
                {
                    break;
                }
                xs.push_back(x);
                ys.push_back(y);
            }
            
            //output the peak shape to STDOUT in JSON format
            root[k]["x"] = Json::Value(Json::arrayValue);
            root[k]["y"] = Json::Value(Json::arrayValue);
            for(int i = 0; i < xs.size(); i++)
            {
                root[k]["x"][i]=xs[i];
                root[k]["y"][i]=ys[i];
            }
        }
        std::cout << root << std::endl;
        return 0;
    }
    else
    {
        //return JSON error message
        Json::Value root;
        root["error"] = "input should be 4,7,10,13, ... numbers";
        std::cout << root << std::endl;
        return 1;
    }
}
