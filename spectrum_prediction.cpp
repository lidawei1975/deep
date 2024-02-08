

// #include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <valarray>
#include <string>
#include <cstring>
#include <vector>

#include "spectrum_prediction.h"

//will link to the C function in libcerf
extern "C"  
{
    double voigt(double x, double sigma, double gamma);
};


namespace spectrum_prediction_helper
{
    /**
     * Check whether two rectrange overlap or not. 
     * @param x1,y1,x2,y2 are the left bottom and right top corner of the first rectangle.
     * @param x3,y3,x4,y4 are the left bottom and right top corner of the second rectangle.
     * @return true if overlap, false if not.
    */
    bool doOverlap(int x1, int y1, int x2, int y2, int x3, int y3, int x4, int y4)
    {
        // If one rectangle is on left side of other
        if (x1 >= x4 || x3 >= x2)
            return false;
    
        // If one rectangle is above other
        if (y1 >= y4 || y3 >= y2)
            return false;
    
        return true;
    }
}


spectrum_prediction::spectrum_prediction()
{
    //ctor
}

spectrum_prediction::~spectrum_prediction()
{
    //dtor
}


/**
 * Load query result from a json file.
 * Example of query.json file
 *  "compound" : 
        [
                {
                        "c_rmsd" : 0.028970664495614095,
                        "peaks" : 
                        [
                                {
                                        "a" : 67.44639116435016,
                                        "exp_a" : 
                                        [
                                                184772.52619663297,
                                                286533.50078707747
                                        ],
                                        "exp_v" : 
                                        [
                                                846145.11736600066,
                                                1832465.4674080082
                                        ],
                                        "exp_x" : 
                                        [
                                                1.4772983776044117,
                                                1.4690000000000001
                                        ],
                                        "exp_y" : 18.850000000000001,
                                        "gamma" : 
                                        [
                                                0.29526054251001721,
                                                0.0049045489539087349
                                        ],
                                        "id_all" : 
                                        [
                                                57
                                        ],
                                        "sigma" : 
                                        [
                                                1.5840546629256331,
                                                2.5474379072578843
                                        ],
                                        "sigma_c" : 8.0,
                                        "x" : 1.466041078125297,
                                        "y" : 18.850999999999999
                                },
                        ]
                },
        ],
    "exp_information" : 
        {
                "b0" : 850.16400146484375,
                "begin" : 12.763794575401146,
                "begin_c" : 157.81873606332283,
                "log_detection_limit" : 10.473706760371538,
                "n_acquisition" : 2048,
                "n_zf" : 4,
                "noise_level" : 6431.4716796875,
                "spectral_width" : 13586.9560546875,
                "spectral_width_c" : 34246.57421875,
                "step" : -0.0019508753236572504,
                "step_c" : -0.078217142730302586,
                "stop" : -3.2158252006753925,
                "stop_c" : -2.2917551056065606,
                "x_dim" : 8192,
                "y_dim" : 2048
        },
*/
void spectrum_prediction::load_query_json(std::string query_file)
{
    std::ifstream fin(query_file.c_str());
    if (!fin)
    {
        std::cerr<<"Error: cannot open file "<<query_file<<std::endl;
        exit(1);
    }

    Json::Value root;
    fin >> root;

    /**
     * Load experimental information
    */
    b0 = root["exp_information"]["b0"].asDouble();
    begin = root["exp_information"]["begin"].asDouble();
    step = root["exp_information"]["step"].asDouble();
    stop = root["exp_information"]["stop"].asDouble();
    begin_c = root["exp_information"]["begin_c"].asDouble();
    step_c = root["exp_information"]["step_c"].asDouble();
    stop_c = root["exp_information"]["stop_c"].asDouble();
    x_dim = root["exp_information"]["x_dim"].asInt();
    y_dim = root["exp_information"]["y_dim"].asInt();
    log_detection_limit = root["exp_information"]["log_detection_limit"].asDouble();
    noise_level = root["exp_information"]["noise_level"].asDouble();
    n_acquisition = root["exp_information"]["n_acquisition"].asInt();
    n_zf = root["exp_information"]["n_zf"].asInt();
    spectral_width = root["exp_information"]["spectral_width"].asDouble();
    spectral_width_c = root["exp_information"]["spectral_width_c"].asDouble();

    /**
     * Load in compounds information. Loop through all compounds.
    */
    for (int i = 0; i < root["compound"].size(); i++)
    {
        /**
         * Define a compound object to store the information of one compound.
        */
        compound c;
        c.namne = root["compound"][i]["name"].asString();
        c.origin = root["compound"][i]["origin"].asString();
        c.origin_1d = root["compound"][i]["origin_1d"].asString();

        /**
         * Loop through all peaks of the compound.
        */
        for (int j = 0; j < root["compound"][i]["peaks"].size(); j++)
        {
            /**
             * Loop through all J splitted peaks from one proton atom.
            */
            
            for(int k=0;k<root["compound"][i]["peaks"][j]["exp_a"].size();k++)
            {
                c.exp_a.push_back(root["compound"][i]["peaks"][j]["exp_a"][k].asDouble());
                c.exp_v.push_back(root["compound"][i]["peaks"][j]["exp_v"][k].asDouble());
                c.exp_x.push_back(root["compound"][i]["peaks"][j]["exp_x"][k].asDouble());
                c.sigma_x.push_back(root["compound"][i]["peaks"][j]["sigma"][k].asDouble());
                c.gamma_x.push_back(root["compound"][i]["peaks"][j]["gamma"][k].asDouble());
            }
            c.peak_group.push_back(c.exp_a.size());
            
            /**
             * Same values for all peaks from the same proton atom.
            */
            c.a.push_back(root["compound"][i]["peaks"][j]["a"].asDouble());
            c.x.push_back(root["compound"][i]["peaks"][j]["x"].asDouble());
            c.y.push_back(root["compound"][i]["peaks"][j]["y"].asDouble());
            c.exp_y.push_back(root["compound"][i]["peaks"][j]["exp_y"].asDouble());
            c.sigma_y.push_back(root["compound"][i]["peaks"][j]["sigma_c"].asDouble());
            /**
             * gamma_y is not in the json file, so we need to push back 0.0.
             * We are using Gaussian shape for the indirect dimension but keep gamma_y for future use.
            */
            c.gamma_y.push_back(0.0);
        }
        compounds.push_back(c);
        std::cout<<"compound "<<i<<" loaded out of "<<root["compound"].size()<<std::endl;
    }

    return;
};

/**
 * This function will generate simulated spectrum based on the query result of one compound
 * and save it to simulated_spectrum.
*/
void spectrum_prediction::simulate_spectrum_of_one_compound(int compound_index)
{
    /**
     * Clear the signal region.
    */
    signal_region_x1.clear();
    signal_region_y1.clear();
    signal_region_x2.clear();
    signal_region_y2.clear();

    /**
     * Initialize simulated_spectrum.
    */
    simulated_spectrum.clear();
    simulated_spectrum.resize(x_dim,std::vector<double>(y_dim,0.0));

    /**
     * Calculate the simulated spectrum, loop all peak group
    */
    for(int i0=0;i0<compounds[compound_index].peak_group.size();i0++)
    {   
        int i_begin = 0;
        if(i0>0)
        {
            i_begin = compounds[compound_index].peak_group[i0-1];
        }
        int i_end = compounds[compound_index].peak_group[i0];

        /**
         * exp_y and sigma_y are the same for all peaks from the same proton atom.
        */
        double sigma_y = compounds[compound_index].sigma_y[i0]; //Unit is point
        double exp_y = compounds[compound_index].exp_y[i0];

        /**
         * Loop through all peaks of this peak group
        */
        for(int i=i_begin;i<i_end;i++)
        {

            /**
             * These are the peak parameters. Defined for code readability.
             * 
            */
            double exp_a = compounds[compound_index].exp_a[i];
            double exp_v = compounds[compound_index].exp_v[i];
            double exp_x = compounds[compound_index].exp_x[i];
            
            double sigma_x = compounds[compound_index].sigma_x[i]; //Unit is point
            double gamma_x = compounds[compound_index].gamma_x[i]; //Unit is point
            

    
            /**
             * Get peak position in points from ppm.
            */
            double pos_x = (exp_x-begin)/step;
            double pos_y = (exp_y-begin_c)/step_c;

            std::cout<<"Add a peak at "<<pos_x<<" "<<pos_y<<" cs_x "<<exp_x<<" cs_y "<<exp_y<<" exp_v "<<exp_v<<" sigma_x "<<sigma_x<<" gamma_x "<<gamma_x<<" sigma_y "<<sigma_y<<std::endl;
            /**
             * To speed up the calculation, we only calculate the peak within 10*sigma range.
            */
            double width_y=2.355*sqrt(sigma_y/2.0);
            int x_min_index = std::max(0,(int)floor(pos_x-2*sigma_x));
            int x_max_index = std::min(x_dim-1,(int)ceil(pos_x+2*sigma_x));
            int y_min_index = std::max(0,(int)floor(pos_y-2*width_y));
            int y_max_index = std::min(y_dim-1,(int)ceil(pos_y+2*width_y));

            for(int j=x_min_index;j<=x_max_index;j++)
            {
                for(int k=y_min_index;k<=y_max_index;k++)
                {
                    double height = exp_v;
                    height *= voigt(j-pos_x,sigma_x,gamma_x);
                    height *= exp(-(k-pos_y)*(k-pos_y)/sigma_y);  //sigma_y is actually 2*sigma_y*sigma_y
                    simulated_spectrum[j][k] += height;
                }
            }

            /**
             * Save the signal region for later use.
            */
            signal_region_x1.push_back(x_min_index);
            signal_region_y1.push_back(y_min_index);
            signal_region_x2.push_back(x_max_index);
            signal_region_y2.push_back(y_max_index);
        }
    }
    std::cout<<"compound "<<compound_index<<" simulated."<<std::endl;

    /**
     * Debug code
    */
    if(compound_index==0)
    {
        std::ofstream fout("simulated_spectrum.txt");
        for(int i=0;i<x_dim;i++)
        {
            for(int j=0;j<y_dim;j++)
            {
                fout<<simulated_spectrum[i][j]<<" ";
            }
            fout<<std::endl;
        }
        fout.close();
    }

    /**
     * Check for overlap of signal regions and merge them.
     * Loop through all pairs.
    */
    for(int i=0;i<signal_region_x1.size();i++)
    {
        for(int j=i+1;j<signal_region_x1.size();j++)
        {
            if(spectrum_prediction_helper::doOverlap(signal_region_x1[i],signal_region_y1[i],signal_region_x2[i],signal_region_y2[i],signal_region_x1[j],signal_region_y1[j],signal_region_x2[j],signal_region_y2[j]))
            {
                /**
                 * Merge the two rectangles.
                */
                signal_region_x1[i] = std::min(signal_region_x1[i],signal_region_x1[j]);
                signal_region_y1[i] = std::min(signal_region_y1[i],signal_region_y1[j]);
                signal_region_x2[i] = std::max(signal_region_x2[i],signal_region_x2[j]);
                signal_region_y2[i] = std::max(signal_region_y2[i],signal_region_y2[j]);
                signal_region_x1.erase(signal_region_x1.begin()+j);
                signal_region_y1.erase(signal_region_y1.begin()+j);
                signal_region_x2.erase(signal_region_x2.begin()+j);
                signal_region_y2.erase(signal_region_y2.begin()+j);
                j--;
            }
        }
    }

    return;
};

void spectrum_prediction::calcualte_contour(Json::Value &root_i,std::vector<float> &raw_data)
{   
    /**
     * Define a contour object to calculate the contour of the simulated spectrum.
    */
    class ccontour c;

    /**
     * Define contour leves. nosie_level*5.5 is 0st level, each level is 1.3 times of the previous level. Total 20 levels.
     * TODO: get it from how experimental spectrum.
    */
    std::vector<double> level;
    level.push_back(noise_level*5.5);
    for(int i=0;i<19;i++)
    {
        level.push_back(level[i]*1.3);
    }

    /**
     * Calculate the contour of the simulated spectrum using the contour object.
     * conrec is the matching square contour algorithm.
    */
    c.conrec(simulated_spectrum,level,signal_region_x1,signal_region_y1,signal_region_x2,signal_region_y2);

    /**
     * Group the contour lines to closed contour lines (except the edge of the spectrum, which are open contour lines).
    */
    c.group_line();

    /**
     * Save the calcualted contour. 
     * Informations are in root_i as a json object.
     * raw_data is the contour data.
    */
    raw_data.clear();
    c.save_result(root_i,raw_data);

    /**
     * Also need to track size of raw data of each compound.
    */
    root_i["size"] = raw_data.size();

    std::cout<<"contour calculated."<<std::endl;

    return;
};

void spectrum_prediction::work()
{
    root = Json::arrayValue;

    int n_current=0;
    for(int i=0;i<compounds.size();i++)
    {   
        /**
         * Simulate the spectrum of one compound. Save it to simulated_spectrum.
        */
        simulate_spectrum_of_one_compound(i);
        Json::Value root_i;
        std::vector<float> raw_data;
        /**
         * Calculate the contour of the simulated spectrum and save it to root_i and raw_data.
        */
        calcualte_contour(root_i,raw_data);

        /**
         * Track compound index for convenience.
        */
        root_i["compound_index"] = i;
        /**
         * Need to track the start and end of the raw data of each compound in the contour_data vector.
        */
        root_i["data_start"] = n_current;
        n_current += raw_data.size();
        root_i["data_end"] = n_current;

        /**
         * Append the root_i to root and append the raw_data to contour_data.
        */
        root.append(root_i);
        contour_data.insert(contour_data.end(),raw_data.begin(),raw_data.end());
    }
    return;
};

/**
 * Write contour information to a json file.
*/
void spectrum_prediction::save_simulated_spectrum(std::string output_file)
{
    std::ofstream fout(output_file.c_str());
    if (!fout)
    {
        std::cerr<<"Error: cannot open file "<<output_file<<std::endl;
        exit(1);
    }
    fout<<root;
    fout.close();
    return;
};

/**
 * Write contour raw data to a binary file.
 * We will need contour informaiton json file to interpret the binary file.
*/
void spectrum_prediction::save_simulated_spectrum_binary(std::string output_file)
{
    FILE *fp = fopen(output_file.c_str(),"wb");
    if (fp == NULL)
    {
        std::cerr<<"Error: cannot open file "<<output_file<<std::endl;
        exit(1);
    }
    fwrite(contour_data.data(),sizeof(float),contour_data.size(),fp);
    return;
};

