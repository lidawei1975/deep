#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "json/json.h"
#include "spectrum_fwhh_1d.h"

namespace fwhh_1d
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

spectrum_fwhh_1d::spectrum_fwhh_1d(){};
spectrum_fwhh_1d::~spectrum_fwhh_1d(){};

// predict whole line at once. not currently used anywhere
// not suitable for metabolomics (peaks can have huge intensity difference)
float spectrum_fwhh_1d::get_median_peak_width()
{
    //define and init our fwhh estimation class
    fwhh_estimator fwhh;


    // run peaks picking on spect, exclude 50 points from each end
    std::vector<int> peak_locations;
    std::vector<float> peak_heights;
    for (int i = 51; i < spect.size() - 51; i++)
    {
        if (spect[i] > spect[i - 1] && spect[i] > spect[i - 2] && spect[i] > spect[i + 1] && spect[i] > spect[i + 2])
        {
            peak_locations.push_back(i);
            peak_heights.push_back(spect[i]);
        }
    }

    // sort peaks by height. Keep track of original index in ndx
    // by default, sort is ascending. We want descending
    std::vector<int> ndx;
    fwhh_1d::sortArr(peak_heights, ndx);

    
    fwhh_1d_wids.clear();
    fwhh_1d_pos.clear();
    for (int i = ndx.size() - 1; i >= std::max(0, int(ndx.size()) - 50); i--) // only look at top 50 peaks
    {
        int peak_loc = peak_locations[ndx[i]];

        float wid=20.0f;
        int stride=1;

        while(wid>16.00f && stride<=16)
        {   
            // run fwhh on peak
            std::vector<float> spectrum_part; // spectrum part to run fwhh on
            for (int j = peak_loc - 50 * stride; j <= peak_loc + 50 * stride; j+=stride)
            {
                if(j<0 || j>=spect.size())
                {
                    spectrum_part.push_back(0.0f); // pad with zeros
                }
                else
                {
                    spectrum_part.push_back(std::max(0.0f,spect[j]));
                }
            }
            //normalize spectrum part
            float max_val=0.0f;
            for(int j=0;j<spectrum_part.size();j++)
            {
                max_val=std::max(max_val,spectrum_part[j]);
            }
            for(int j=0;j<spectrum_part.size();j++)
            {
                spectrum_part[j]/=max_val;
            }

            wid = fwhh.predict(spectrum_part);
            stride=stride*2;
        }

        stride=stride/2;
        wid*=stride;
        fwhh_1d_wids.push_back(wid);
        fwhh_1d_pos.push_back(peak_loc);
    }
    //get median of fwhh_1d_wids
    std::vector<float> wids_sorted = fwhh_1d_wids;
    std::sort(wids_sorted.begin(), wids_sorted.end());
    float median_wid = wids_sorted[wids_sorted.size() / 2];

    return median_wid;
};

void spectrum_fwhh_1d::print_result(std::string fname)
{
    std::ofstream out(fname);
    for(int i=0;i<fwhh_1d_pos.size();i++)
    {
        out<<fwhh_1d_pos[i]<<" "<<fwhh_1d_wids[i]<<std::endl;
    }
    out.close();
};
