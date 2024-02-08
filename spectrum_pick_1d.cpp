#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "json/json.h"
#include "commandline.h"

#include "spectrum_pick_1d.h"

// help function
// we have a list of peak_amplitudes and a ratio cutoff
// need to partitate them into pieces that fullfill
// largest amplitude ratio is less than the predefined ratio cutoff
// fewest pieces possible (try your best, not a hard requirement)
#ifndef ldw_math
#define ldw_math
namespace ldw_math
{
    std::vector<int> get_best_partition(std::vector<float> peak_amplitudes, float ratio_cutoff)
    {
        float v_min = 1e30f;
        float v_max = 0.0f;
        int v_max_ndx, v_min_ndx, p;
        std::vector<int> r, r2;

        for (int i = 0; i < peak_amplitudes.size(); i++)
        {
            if (peak_amplitudes[i] > v_max)
            {
                v_max = peak_amplitudes[i];
                v_max_ndx = i;
            }
            if (peak_amplitudes[i] < v_min)
            {
                v_min = peak_amplitudes[i];
                v_min_ndx = i;
            }
        }

        if (v_min / v_max > ratio_cutoff)
        {
            r.clear();
        }
        else
        {
            if (v_min_ndx > v_max_ndx)
            {
                p = v_max_ndx + 1;
                while (peak_amplitudes[p] / v_min > v_max / peak_amplitudes[p])
                {
                    p++;
                }
            }
            else
            {
                p = v_max_ndx - 1;
                while (peak_amplitudes[p] / v_min > v_max / peak_amplitudes[p])
                {
                    p--;
                }
                p = p + 1;
            }
            std::vector<float> peak_amplitudes_1(peak_amplitudes.begin(), peak_amplitudes.begin() + p);
            std::vector<float> peak_amplitudes_2(peak_amplitudes.begin() + p, peak_amplitudes.end());
            r = get_best_partition(peak_amplitudes_1, ratio_cutoff);
            r2 = get_best_partition(peak_amplitudes_2, ratio_cutoff);

            r.push_back(p);
            for (int k = 0; k < r2.size(); k++)
            {
                r.push_back(r2[k] + p);
            }
        }
        return r;
    };
};
#endif

/**
 * @brief spectrum_pick_1d class constructor and destructor
*/
spectrum_pick_1d::spectrum_pick_1d()
{
    /**
     * By deault, we do not run cubic spline interpolation
     * So the interpolation step is 1.0 pixel
    */
    interpolation_step = 1.0;
};
spectrum_pick_1d::~spectrum_pick_1d(){};

bool spectrum_pick_1d::init_mod(int mod)
{
    mod_selection = mod;
    if (mod_selection == 1)
    {
        p1.load();
    }
    else if (mod_selection == 2)
    {
        p1.load_m2();
    }
    else
    {
        p1.load_m3();
    }

    return true;
};

bool spectrum_pick_1d::peak_partition_1d()
{
    double boundary_cutoff = noise_level * user_scale2;
    if (spect[0] > boundary_cutoff)
    {
        signa_boudaries.push_back(0);
    }

    for (int j = 1; j < ndata; j++)
    {
        if (spect[j - 1] <= boundary_cutoff && spect[j] > boundary_cutoff)
        {
            signa_boudaries.push_back(std::max(j - 10, 0));
        }
        else if (spect[j - 1] > boundary_cutoff && spect[j] <= boundary_cutoff)
        {
            noise_boudaries.push_back(std::min(j + 10, ndata));
        }
    }
    if (noise_boudaries.size() < signa_boudaries.size())
    {
        noise_boudaries.push_back(ndata);
    }

    bool b = true;
    while (b)
    {
        b = false;
        for (int j = signa_boudaries.size() - 1; j >= 1; j--)
        {
            if (signa_boudaries[j] <= noise_boudaries[j - 1])
            {
                signa_boudaries.erase(signa_boudaries.begin() + j);
                noise_boudaries.erase(noise_boudaries.begin() + j - 1);
            }
        }
    }

    // combine noise_boudaries and signa_boudaries if too close
    for (int j = signa_boudaries.size() - 1; j >= 1; j--)
    {
        if (signa_boudaries[j] - noise_boudaries[j - 1] < 1)
        {
            signa_boudaries.erase(signa_boudaries.begin() + j);
            noise_boudaries.erase(noise_boudaries.begin() + j - 1);
        }
    }

    return true;
};

bool spectrum_pick_1d::peak_partition_step2()
{
    for (int j = 0; j < signa_boudaries.size(); j++)
    {
        int begin = signa_boudaries[j];
        int stop = noise_boudaries[j];

        if (stop - begin < 5)
            continue;

        // std::cout<<"Step 1, from "<<begin<<" to "<<stop<<std::endl;

        std::vector<int> peak_positions, min_positions;
        std::vector<float> peak_amplitudes;
        for (int k = begin + 2; k < stop - 2; k++)
        {
            if (spect[k] > noise_level * user_scale && spect[k] > spect[k - 2] && spect[k] > spect[k - 1] && spect[k] > spect[k + 1] && spect[k] > spect[k + 2])
            {
                peak_positions.push_back(k);
                peak_amplitudes.push_back(spect[k]);
                continue;
            }
        }
        if (peak_positions.size() == 0)
            continue;
        for (int k = 0; k < peak_positions.size() - 1; k++)
        {
            int p = peak_positions[k];
            float v = peak_amplitudes[k];
            for (int m = peak_positions[k]; m < peak_positions[k + 1]; m++)
            {
                if (spect[m] < v)
                {
                    v = spect[m];
                    p = m;
                }
            }
            min_positions.push_back(p);
        }
        min_positions.push_back(stop);

        // now we have N peaks (peak_positions) and N-1 minimal points (min_positions) between them; min_positions also have one addtional element = stop
        std::vector<int> bs = ldw_math::get_best_partition(peak_amplitudes, 0.2);
        bs.push_back(peak_amplitudes.size());

        for (int k = 0; k < bs.size(); k++)
        {
            int b, s;
            if (k == 0)
                b = begin;
            else
                b = min_positions[bs[k - 1] - 1];
            s = min_positions[bs[k] - 1];
            final_segment_begin.push_back(b);
            final_segment_stop.push_back(s);
            // std::cout<<"Cut into "<<b<<" - "<<s<<std::endl;
        }
    }

    return true;
}

bool spectrum_pick_1d::run_ann(bool b_neg)
{
    for (int j = 0; j < final_segment_begin.size(); j++)
    {
        int begin = final_segment_begin[j];
        int stop = final_segment_stop[j];
        int begin0 = begin;

        std::vector<float> t;

        int ndim = spect.size();
        int left_patch = 0;
        int right_patch = 0;
        int n = stop - begin + 40;

        left_patch = std::max(0, 20 - begin);
        begin = std::max(0, begin - 20);
        right_patch = std::max(0, 20 - (ndim - stop));
        stop = std::min(ndim, stop + 20);

        std::vector<float> data;
        data.clear();
        data.resize(n, 0.0f);
        std::copy(spect.begin() + begin, spect.begin() + stop, data.begin() + left_patch);

        p1.predict(data);
        p1.predict_step2();

        for (int k = 0; k < p1.posits.size(); k++)
        {
            if (b_neg)
            {
                a.push_back(-p1.intens[k]);
            }
            else
            {
                a.push_back(p1.intens[k]);
            }
            x.push_back(p1.posits[k] + begin0);
            confidence.push_back(p1.confidence[k]);
            sigmax.push_back(p1.sigmas[k]);
            gammax.push_back(p1.gammas[k]);
        }
    }
    return true;
}

bool spectrum_pick_1d::substract_baseline()
{
    for (int i = 0; i < ndata; i++)
    {
        spect[i] -= baseline[i];
    }
    return true;
}

// segment by segment prediction, as in peak2d
bool spectrum_pick_1d::spectrum_pick_1d_work(bool b_negative)
{
    // substract_baseline();
    p1.set_noise_level(noise_level);
    int ndim = spect.size();

    /**
     * @brief clear signal and noise boundaries
     * then partition the spectrum into segments in function call peak_partition() (from base class)
     * then we need to handle segment defined as [signal_boudaries[i], noise_boudaries[i]], for i=0,1,2, ...
     */
    signa_boudaries.clear();
    noise_boudaries.clear();
    peak_partition_1d();

    /**
     * @brief for each segment, we need to cut it into smaller segments, and then predict each segment
     * because our ANN model can only handle dynamic range of 0.01-1.0.
     */
    final_segment_begin.clear();
    final_segment_stop.clear();
    peak_partition_step2();

    run_ann(); // run ANN model to predict peaks on each final segment

    if (b_negative == true)
    {
        // for negative peaks, we need to flip the spectrum
        for (int k = 0; k < spect.size(); k++)
        {
            spect[k] = -spect[k];
        }
        signa_boudaries.clear();
        noise_boudaries.clear();
        peak_partition_1d();

        final_segment_begin.clear();
        final_segment_stop.clear();
        peak_partition_step2();

        run_ann(true); // run ANN model to predict peaks on each final segment. Negative peaks

        // flip back
        for (int k = 0; k < spect.size(); k++)
        {
            spect[k] = -spect[k];
        }
    }

    // fitler out  peaks below cutoff
    for (int k = a.size() - 1; k >= 0; k--)
    {
        if (fabs(a[k]) < noise_level * user_scale)
        {
            a.erase(a.begin() + k);
            x.erase(x.begin() + k);
            sigmax.erase(sigmax.begin() + k);
            gammax.erase(gammax.begin() + k);
            confidence.erase(confidence.begin() + k);
        }
    }

    return true;
};

bool spectrum_pick_1d::get_peak_pos(std::vector<int> &t)
{
    for (int k = 0; k < a.size(); k++)
    {
        t.push_back(std::round(x[k]));
    }
    return true;
}

bool spectrum_pick_1d::print_peaks(std::string outfname)
{
    FILE *fp = fopen(outfname.c_str(), "w");
    fprintf(fp, "VARS INDEX X_AXIS X_PPM XW HEIGHT CONFIDENCE\n");
    fprintf(fp, "FORMAT %%5d %%9.3f %%9.4f %%7.3f %%+e %%4.3f\n");

    for (int k = 0; k < a.size(); k++)
    {
        double cs = begin1 + step1 * x[k];
        /**
         * In case we have interpolated the spectrum, we need to convert peak position back to original spectrum in pixel
        */
        double peak_pos_pixel_original = x[k]*interpolation_step;
        float s1 = 0.5346 * sigmax[k] * 2 + std::sqrt(0.2166 * 4 * gammax[k] * gammax[k] + sigmax[k] * sigmax[k] * 8 * 0.6931);
        /**
         * s1 need to be converted back to original spectrum in pixel
        */
        s1 = s1*interpolation_step;
        fprintf(fp, "%5d %9.3f %9.4f %7.3f %+e %4.3f\n", k + 1, peak_pos_pixel_original, cs, s1, a[k], confidence[k]);
    }

    fclose(fp);
    return true;
}

bool spectrum_pick_1d::get_peaks(struct spectrum_1d_peaks &y)
{

    /**
     * Get ppm from x
    */
    std::vector<double> x_ppm;
    for (int k = 0; k < x.size(); k++)
    {
        x_ppm.push_back(begin1 + step1 * x[k]);
    }

    y.a = a;
    y.x = x;
    y.ppm = x_ppm;
    y.sigmax = sigmax;
    y.gammax = gammax;
    y.confidence = confidence;

    /**
     * This is the volume of each peak, a very rough estimation
    */
    std::vector<double> volume;
    for (int k = 0; k < a.size(); k++)
    {
        volume.push_back(a[k] * sigmax[k] * std::sqrt(3.1415926 * 2));
    }

    y.volume = volume;

    return true;
};


bool spectrum_pick_1d::interpolate_spectrum(const double inter_step)
{
    /**
     * Now we need to interpolate the spectrum
     * ndata_new is the new size of the spectrum
     */
    int ndata_new = int(std::round(ndata / inter_step));

    /**
     * define a cubic spline object, and calculate coefficients from old spectrum
     */
    cublic_spline cs,cs_for_image;
    cs.calculate_coefficients(spect);
    if(spe_image.size()>15)
    {
        cs_for_image.calculate_coefficients(spe_image);
    }

    /**
     * define a new spectrum, and calculate values from cubic spline
     */
    std::vector<float> spect_new,spe_image_new;

    /**
     * Now we can calculate new spectrum
     * x is the new x axis, x[0]=0.0, x[1]=inter_step, x[2]=2*inter_step, ...
     * old axis is from 0 to ndata-1.
     */
    for (int k = 0; k < ndata_new; k++)
    {
        spect_new.push_back(cs.calculate_value(k * inter_step));
    }

    if(spe_image.size()>15)
    {
        for (int k = 0; k < ndata_new; k++)
        {
            spe_image_new.push_back(cs_for_image.calculate_value(k * inter_step));
        }  
    }

    /**
     * update member variables
     */
    spect = spect_new;
    spe_image = spe_image_new;
    
    ndata = ndata_new;                  
    step1 = inter_step * step1; // unit of step1 is ppm, not points

    std::cout<<"After cubic spline interpolation, ";
    std::cout<<"Estimated FWHH is "<<get_median_peak_width()<<" pixel. ";
    std::cout<<"spectrum size is "<<ndata<<" and step1 is "<<step1<<" ppm"<<std::endl;

    return true;
}

/**
 * @brief adjust ppp of spectrum to a given value, using cubic spline interpolation
 * This function will change member variable spect,spe_image,step1,xdim (alias ndata)
 * begin1,stop1 will not be changed
 */
bool spectrum_pick_1d::adjust_ppp_of_spectrum(const double ppp)
{
    /**
     * First, get fwhh of spectrum
     */
    double fwhh = get_median_peak_width();
    std::cout<<"Estimated FWHH is "<<fwhh<<" pixel"<<std::endl;

    /**
     * To adjust ppp, we need to interpolate the spectrum
     */
    double current_interpolation_step = fwhh / ppp;
    std::cout<<"Interpolation step is "<<current_interpolation_step<<" pixel"<<std::endl;

    interpolate_spectrum(current_interpolation_step);    

    /**
     * Keep track of total interpolation step
     */
    interpolation_step = interpolation_step * current_interpolation_step;
    std::cout<<"Total interpolation step is "<<interpolation_step<<" pixel"<<std::endl;

    return true;
}