#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "json/json.h"
#include "spectrum_fwhh.h"

namespace fwhh
{
    template <typename T>
    void sortArr(std::vector<T> &arr, std::vector<int> &ndx)
    {
        std::vector<std::pair<T, int>> vp;

        for (int i = 0; i < arr.size(); ++i)
        {
            vp.push_back(std::make_pair(arr[i], i));
        }

        std::sort(vp.begin(), vp.end());

        for (int i = 0; i < vp.size(); i++)
        {
            ndx.push_back(vp[i].second);
        }
    };
} // namespace

spectrum_fwhh::spectrum_fwhh(){};
spectrum_fwhh::~spectrum_fwhh(){};

/**
 * @brief this is the main function of this class. It estimates the median peak width of the spectrum
 * @param median_width_direct median peak width of the direct dimension
 * @param median_width_indirect median peak width of the indirect dimension
 * @return true always for now
 */
bool spectrum_fwhh::get_median_peak_width(float &median_width_direct, float &median_width_indirect)
{
    // define DNN based fwhh estimation class
    fwhh_estimator fwhh;

    /**
     * get max intensity of the spectrum
     * spect is defined as float * in spectrum_io class
     */
    float max_intensity = 0.0f;
    for (int i = 0; i < xdim*ydim; i++)
    {
        max_intensity = std::max(max_intensity, spect[i]);
    }

    /**
     * Step 1: run simple peak picking on the spectrum
     * noise_level is estimated from base class read_spectrum function call
     * We define min_intensity as 5 times noise_level or 1/100 of the maximum intensity, whichever is larger
     */

    double min_intensity = std::max(5.0 * noise_level, 0.01 * max_intensity);

    std::cout << "Minimal peak intensity is set to " << min_intensity << std::endl;


    p1.clear(); //xdim peak position, direct dimension
    p2.clear(); //ydim peak position, indirect dimension
    p_intensity.clear(); //peak intensity

    /**
     * spect[i + j * xdim] means the intensity at (i,j) in the 2D spectrum
     * Data is stored in row major order, so, i is the direct dimension, j is the indirect dimension
    */

    for (unsigned int i = 0 + 50; i < xdim - 50; i++) //skip peaks too close to the spectral edge, because DNN requires 50 points on each side
    {
        for (unsigned int j = 0 + 50; j < ydim - 50; j++)
        {

            if (spect[i + j * xdim] > spect[i + (j - 1) * xdim] && spect[i + j * xdim] > spect[i + (j + 1) * xdim] && spect[i + j * xdim] > spect[(i - 1) + j * xdim] && spect[i + j * xdim] > spect[(i + 1) + j * xdim] && spect[i + j * xdim] > min_intensity)
            {
                int ndiag = 0;
                if (spect[i + j * xdim] > spect[(i + 1) + (j - 1) * xdim])
                    ndiag++;
                if (spect[i + j * xdim] > spect[(i - 1) + (j - 1) * xdim])
                    ndiag++;
                if (spect[i + j * xdim] > spect[(i + 1) + (j + 1) * xdim])
                    ndiag++;
                if (spect[i + j * xdim] > spect[(i - 1) + (j + 1) * xdim])
                    ndiag++;
                if (ndiag >= 4)
                {
                    p1.push_back(i); // index is from 0  direct dimension
                    p2.push_back(j);
                    p_intensity.push_back(spect[i + j * xdim]);
                }
            }
        }
    }

    std::cout << "Picked " << p1.size() << " peaks for peak width estimation." << std::endl;


    /**
     * Sort peaks by height. Keep track of original index in ndx
     * by default, sort is ascending. We want descending, so, we reverse the order below
    */

    std::vector<int> ndx;
    fwhh::sortArr(p_intensity, ndx);

    fwhh_wids_direct.clear();
    fwhh_pos_direct.clear();

    /**
     * We only check top 50 peaks even if more than 50 peaks are picked. 
     * This should be enough to get a good estimate of the median peak width
    */
    for (int i = ndx.size() - 1; i >= std::max(0, int(ndx.size()) - 50); i--) 
    {
        int peak_loc_direct = p1[ndx[i]];
        int peak_loc_indirect = p2[ndx[i]];


        /**
         * Working on direct dimension first
        */
        
        float wid = 20.0f;
        int stride = 1;

        while (wid > 16.00f && stride <= 16)
        {
            // run fwhh on peak
            std::vector<float> spectrum_part; // spectrum part to run fwhh on
            for (int j = peak_loc_direct - 50 * stride; j <= peak_loc_direct + 50 * stride; j += stride)
            {
                if (j < 0 || j >= xdim)
                {
                    spectrum_part.push_back(0.0f); // pad with zeros, not optimal, but works for the DNN
                }
                else
                {
                    //Data is stored in row major order
                    spectrum_part.push_back(std::max(0.0f, spect[j+ peak_loc_indirect * xdim]));
                }
            }


            /**
             * normalize spectrum part [0,1) before running DNN (this is important for DNN to work)
            */
            float max_val = 0.0f;
            for (int j = 0; j < spectrum_part.size(); j++)
            {
                max_val = std::max(max_val, spectrum_part[j]);
            }
            for (int j = 0; j < spectrum_part.size(); j++)
            {
                spectrum_part[j] /= max_val;
            }
            wid = fwhh.predict(spectrum_part);
            
            /**
             * in case fwhh is too large (>16, the DNN validaiton limit),
             * we increase stride to reduce the number of points to run DNN again.
            */
            stride = stride * 2;
        }

        /**
         * stride is now the last stride * 2 (that is smaller than 16) when fwhh falls below 16 (within DNN validation limit)
         * so we divide stride by 2 to get the last correct stride 
         * Real width is then stride * wid
        */
        stride = stride / 2; 
        wid *= stride;
        fwhh_wids_direct.push_back(wid);
        fwhh_pos_direct.push_back(peak_loc_direct);
    }

    /**
     * Now working on indirect dimension using the same procedure as above
     * except we get 1D traces along indirect dimension now instead of direct dimension
    */
    fwhh_wids_indirect.clear();
    fwhh_pos_indirect.clear();
    for (int i = ndx.size() - 1; i >= std::max(0, int(ndx.size()) - 50); i--)
    {
        int peak_loc_direct = p1[ndx[i]];
        int peak_loc_indirect = p2[ndx[i]];

        float wid = 20.0f;
        int stride = 1;

        while (wid > 16.00f && stride <= 16)
        {
            std::vector<float> spectrum_part;
            for (int j = peak_loc_indirect - 50 * stride; j <= peak_loc_indirect + 50 * stride; j += stride)
            {
                if (j < 0 || j >= ydim)
                {
                    spectrum_part.push_back(0.0f);
                }
                else
                {
                    //Data is stored in row major order
                    spectrum_part.push_back(std::max(0.0f, spect[peak_loc_direct + j * xdim]));
                }
            }

            float max_val = 0.0f;
            for (int j = 0; j < spectrum_part.size(); j++)
            {
                max_val = std::max(max_val, spectrum_part[j]);
            }
            for (int j = 0; j < spectrum_part.size(); j++)
            {
                spectrum_part[j] /= max_val;
            }
            wid = fwhh.predict(spectrum_part);
            stride = stride * 2;
        }
        stride = stride / 2;
        wid *= stride;
        fwhh_wids_indirect.push_back(wid);
        fwhh_pos_indirect.push_back(peak_loc_indirect);
    }


    // get median of wids
    std::vector<float> wids_sorted = fwhh_wids_direct;
    std::sort(wids_sorted.begin(), wids_sorted.end());
    median_width_direct = wids_sorted[wids_sorted.size() / 2];

    wids_sorted = fwhh_wids_indirect;
    std::sort(wids_sorted.begin(), wids_sorted.end());
    median_width_indirect = wids_sorted[wids_sorted.size() / 2];

    return true;
};

void spectrum_fwhh::print_result(std::string fname)
{
    std::ofstream out(fname);
    for (int i = 0; i < fwhh_pos_direct.size(); i++)
    {
        out << fwhh_pos_direct[i] << " ";  
        out << fwhh_pos_indirect[i] << " "; 
        out << fwhh_wids_direct[i] << " ";
        out << fwhh_wids_indirect[i] << " ";
        out << std::endl;
    }
    out.close();
};
