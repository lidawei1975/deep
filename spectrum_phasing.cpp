#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>
#include <limits>

#include <time.h>
#include <sys/time.h>

#include "kiss_fft.h"
#include "json/json.h"
#include "ldw_math.h"

#include "spectrum_phasing.h"

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        //  Handle error
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

namespace phase_2d_helper
{
    void sortArr(const std::vector<float> &arr, std::vector<int> &ndx, bool b_Descending = true)
    {
        ndx.clear(); // clear ndx

        std::vector<std::pair<float, int>> vp;

        for (int i = 0; i < arr.size(); ++i)
        {
            vp.push_back(std::make_pair(arr[i], i));
        }

        std::sort(vp.begin(), vp.end()); // ascending order

        if (b_Descending == true)
        {
            std::reverse(vp.begin(), vp.end()); // descending order
        }

        for (int i = 0; i < vp.size(); i++)
        {
            ndx.push_back(vp[i].second);
        }
    };

    /**
     * Get a unique set of integers from an array.
     * This function does NOT change the order of input array
    */
    std::vector<int> get_unique(const std::vector<int> &in)
    {
        std::vector<int> out;
        for (int i = 0; i < in.size(); i++)
        {
            if (std::find(out.begin(), out.end(), in[i]) == out.end())
            {
                out.push_back(in[i]);
            }
        }
        return out;
    }

    /**
     * Calculate entropy of a spectrum.
     * @param s_normalized: input spectrum, this function will normalize it to be between 0 and 1
     * @param ndim: size of spectrum
    */
    double calculate_entropy_of_spe(float *s_normalized, int ndim)
    {
        /**
         * Normalize s_normalized to be between 0 and 1
         */
        float max_s = -std::numeric_limits<float>::max();
        float min_s = std::numeric_limits<float>::max();
        int loc_min = 0;
        for (int i = 0; i < ndim; i++)
        {
            if (s_normalized[i] > max_s)
            {
                max_s = s_normalized[i];
            }
            if (s_normalized[i] < min_s)
            {
                min_s = s_normalized[i];
                loc_min = i;
            }
        }
        for (int i = 0; i < ndim; i++)
        {
            s_normalized[i] = (s_normalized[i] - min_s) / (max_s - min_s);
        }

        double sum = 0;
        for (int i = 0; i <ndim; i++)
        {
            sum += s_normalized[i];
        }

        double entropy = 0;
        for (int i = 0; i < ndim; i++)
        {
            if (s_normalized[i] > 0.0)
            {
                entropy += s_normalized[i] * log(s_normalized[i]);
            }
        }

        return -entropy / sum + log(sum);
    };

    /**
     * Select a subset of input array uniformly
     * @param in: input array
     * @param number_per_bin: number of elements to select from each bin
     * @param nbin: number of bins
     * @param max: max value of input array (used to define bin boundaries)
    */
    std::vector<int> selected_subset_uniformly(const std::vector<int> &in, int number_per_bin, int nbin, int max)
    {
        /**
         * Size of each bin
        */
        double bin_size = double(max) / nbin;

        /**
         * put input array into bins
        */
        std::vector<std::vector<int>> bins(nbin);

        for (int i = 0; i < in.size(); i++)
        {
            int bin_index = int(floor(in[i] / bin_size));
            /**
             * safety check. not sure if this is necessary
            */
            if (bin_index >= nbin)
            {
                bin_index = nbin - 1;
            }
            bins[bin_index].push_back(in[i]);
        }

        /**
         * Select number_per_bin elements from each bin, from the beginning of each bin (strongest peaks first because input is sorted)   
        */
        std::vector<int> out;
        for (int i = 0; i < nbin; i++)
        {
            for (int j = 0; j < number_per_bin; j++)
            {   
                /**
                 * If not enough elements in this bin, so be it
                */
                if (j < bins[i].size())
                {
                    out.push_back(bins[i][j]);
                }
            }
        }

        return out;
    }
}

/**
 * constructor
 * Will call spectrum_io_1d constructor
 * spectrum_fwhh constructor: initialize fwhh dnn
 * phase_dnn constructor: initialize phase correction dnn
 */

spectrum_phasing::spectrum_phasing()
{
}

spectrum_phasing::~spectrum_phasing()
{
}

/**
 * @brief fid_2d::phase_correction_worker: apply phase correction along dim2 for all rows in spectrum
 * @param n_dim1: size of first dimension.
 * @param n_dim2: size of second dimension. data are saved in [n_dim1][n_dim2] order
 * @param p0: phase correction degree 0-th order
 * @param p1: phase correction degree 1-st order
 * @param b_second: if true, apply phase correction to second dimension, otherwise apply to first dimension
 * @param spectrum_real: input data [n_dim1][n_dim2] real part
 * @param spectrum_imag: input data [n_dim1][n_dim2] imaginary part
 * @param spectrum_real_out: output data [n_dim1][n_dim2] real part. nullptr means write back to input array
 * @param spectrum_imag_out: output data [n_dim1][n_dim2] imaginary part. nullptr means write back to input array
 * 
 * General comments:
 * To speed up, we always run calculations along the major dimension (n_dim2).
 * when b_second is true, we pre-calculate cos(phase) and sin(phase) for all n_dim2 values to avoid repeated calculation (which is very slow)
 * This function need to be called many times in nested loops, so we try to make it as fast as possible.
*/
bool spectrum_phasing::phase_correction_worker(int n_dim1, int n_dim2, double p0, double p1, bool b_second,
                                               float *spectrum_real, float *spectrum_imag,
                                               float *spectrum_real_out, float *spectrum_image_out) const
{
    float phase;
    float cos_phase,sin_phase;

    if(b_second == false)
    {
        for (int i = 0; i < n_dim1; i++)
        {
            
            phase = p0 + p1 * i / n_dim1;
            cos_phase = cos(phase * M_PI / 180.0);
            sin_phase = sin(phase * M_PI / 180.0);
            for (int j = 0; j < n_dim2; j++)
            {
                float real = spectrum_real[j + i * n_dim2];
                float imag = spectrum_imag[j + i * n_dim2];

                if (spectrum_real_out == nullptr || spectrum_image_out == nullptr)
                {
                    /**
                     * Write back to input array
                     */
                    spectrum_real[j + i * n_dim2] = real * cos_phase - imag * sin_phase;
                    spectrum_imag[j + i * n_dim2] = real * sin_phase + imag * cos_phase;
                }
                else
                {
                    /**
                     * Write to output array
                     */
                    spectrum_real_out[j + i * n_dim2] = real * cos_phase - imag * sin_phase;
                    spectrum_image_out[j + i * n_dim2] = real * sin_phase + imag * cos_phase;
                }
            }
        }
    }
    else
    {
        std::vector<float> cos_phases(n_dim2),sin_phases(n_dim2);
        for (int j = 0; j < n_dim2; j++)
        {
            phase = p0 + p1 * j / n_dim2;
            cos_phase = cos(phase * M_PI / 180.0);
            sin_phase = sin(phase * M_PI / 180.0);
            cos_phases[j] = cos_phase;
            sin_phases[j] = sin_phase;
        }


        for (int i = 0; i < n_dim1; i++)
        {
            for (int j = 0; j < n_dim2; j++)
            {
                float real = spectrum_real[j + i * n_dim2];
                float imag = spectrum_imag[j + i * n_dim2];

                if (spectrum_real_out == nullptr || spectrum_image_out == nullptr)
                {
                    /**
                     * Write back to input array
                     */
                    spectrum_real[j + i * n_dim2] = real * cos_phases[j] - imag * sin_phases[j];
                    spectrum_imag[j + i * n_dim2] = real * sin_phases[j] + imag * cos_phases[j];
                }
                else
                {
                    /**
                     * Write to output array
                     */
                    spectrum_real_out[j + i * n_dim2] = real * cos_phases[j] - imag * sin_phases[j];
                    spectrum_image_out[j + i * n_dim2] = real * sin_phases[j] + imag * cos_phases[j];
                }
            }
        }
    }
    return true;
}



/**
 * Run a series of P0 only phase correction on spectra and get P0 at the minimal sum of entropy
 * @param ndim1: size of first dimension (how many spectra)
 * @param ndim2: size of second dimension (how many data points in each spectrum)
 * @param nstep: number of P0 to test on both size of P0=0
 * @param step_size: step size of P0, in degree
 * @param p0_center: center of P0, in degree
 * @param spectrum_real: input data [ndim1][ndim2] real part of input spectra
 * @param spectrum_imag: input data [ndim1][ndim2] imaginary part of input spectra
 * @param p0s: All tested P0 (this is output)
 * @param entropies: All entropies of tested P0 (this is output)
 * @return true
 */
float spectrum_phasing::entropy_based_p0_correction(int ndim1, int ndim2, int nstep, float step_size, float p0_center,
                                                    float *spectrum_real, float *spectrum_imag,
                                                    std::vector<float> &p0s, std::vector<float> &entropies) const
{
    float p0_at_min_entropy = p0_center;
    for (int i = -nstep; i < nstep; i++)
    {
        float p0 = p0_center + i * step_size;
        std::vector<float> spectrum_real_corrected(ndim1 * ndim2), spectrum_imag_corrected(ndim1 * ndim2);
        phase_correction_worker(ndim1, ndim2, p0, 0.0, true, spectrum_real, spectrum_imag, spectrum_real_corrected.data(), spectrum_imag_corrected.data());

        double entropy = 0.0;
        for (int j = 0; j < ndim1; j++)
        {
            entropy += phase_2d_helper::calculate_entropy_of_spe(spectrum_real_corrected.data() + j * ndim2, ndim2);
        }
        p0s.push_back(p0);
        entropies.push_back(entropy);
    }
    return true;
};

/**
 * Run a series of P0 and P1 phase correction on spectra and get P0 and P1 at the minimal sum of entropy
 * @param ndim1: size of first dimension (how many spectra)
 * @param ndim2: size of second dimension (how many data points in each spectrum)
 * @param nstep: number of P0 to test on both size of P0=0
 * @param step_size: step size of P0, in degree
 * @param p0_center: center of P0, in degree
 * @param nstep2: number of P1 to test on both size of P1=0
 * @param step_size2: step size of P1, in degree
 * @param p1_center: center of P1, in degree
 * @param spectrum_real: input data [ndim1][ndim2] real part of input spectra
 * @param spectrum_imag: input data [ndim1][ndim2] imaginary part of input spectra
 * @param p0s: All tested P0 (this is output)
 * @param p1s: All tested P1 (this is output)
 * @param entropies: All entropies of tested P0 and P1 (this is output)
 * @return true.
 */
bool spectrum_phasing::entropy_based_p0_p1_correction(int ndim1, int ndim2,
                                                      int nstep, float step_size, float p0_center,
                                                      int nstep2, float step_size2, float p1_center,
                                                      float *spectrum_real, float *spectrum_imag,
                                                      std::vector<float> &p0s, std::vector<float> &p1s, std::vector<float> &entropies) const
{
    for (int i = -nstep; i < nstep; i++)
    {
        float p0 = p0_center + i * step_size;
        for (int j = -nstep2; j < nstep2; j++)
        {
            float p1 = p1_center + j * step_size2;
            std::vector<float> spectrum_real_corrected(ndim1 * ndim2), spectrum_imag_corrected(ndim1 * ndim2);
            phase_correction_worker(ndim1, ndim2, p0, p1, true, spectrum_real, spectrum_imag, spectrum_real_corrected.data(), spectrum_imag_corrected.data());

            double entropy = 0.0;
            for (int k = 0; k < ndim1; k++)
            {
                entropy += phase_2d_helper::calculate_entropy_of_spe(spectrum_real_corrected.data() + k * ndim2, ndim2);
            }
            p0s.push_back(p0);
            p1s.push_back(p1);
            entropies.push_back(entropy);
        }
    }
    return true;
};

/**
 * This is the main working function
 * Entropy based algorithm only at this time. Seems more than enough for most cases.
 */
bool spectrum_phasing::auto_phase_correction()
{

    /**
     * Get a combined spectrum SQRT(real^2+imag^2) to find location of strong signal peaks
     */
    std::vector<float> combined_spectrum;
    for (int i = 0; i < xdim * ydim; i++)
    {
        combined_spectrum.push_back(sqrt(spectrum_real_real[i] * spectrum_real_real[i] + spectrum_real_imag[i] * spectrum_real_imag[i] + spectrum_imag_real[i] * spectrum_imag_real[i] + spectrum_imag_imag[i] * spectrum_imag_imag[i]));
    }

    /**
     * For debug, save combined spectrum to file
     */
    std::ofstream fout("combined_spectrum.txt");
    for (int i = 0; i < ydim; i++)
    {
        for (int j = 0; j < xdim; j++)
        {
            fout << combined_spectrum[j + i * xdim] << " ";
        }
        fout << std::endl;
    }
    fout.close();

    /**
     * Run simple peak picking on combined spectrum and selected the top 10 peaks
     */
    std::vector<int> px, py;
    std::vector<float> p_intensity;
    float max_intensity = 0.0;

    for (int i = 2; i < xdim - 2; i++)
    {
        for (int j = 2; j < ydim - 2; j++)
        {
            /**
             * Peak is defined as data point that is larger than all its 8 neighbors
             */
            if (combined_spectrum[i + j * xdim] > combined_spectrum[i - 1 + (j - 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + (j - 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + 1 + (j - 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i - 1 + j * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + 1 + j * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i - 1 + (j + 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + (j + 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + 1 + (j + 1) * xdim])
            {
                px.push_back(i);
                py.push_back(j);
                p_intensity.push_back(combined_spectrum[i + j * xdim]);
                if (combined_spectrum[i + j * xdim] > max_intensity)
                {
                    max_intensity = combined_spectrum[i + j * xdim];
                }
            }
        }
    }

    /**
     * Sort peaks by intensity and keep track of index
     */
    std::vector<int> ndx_sort;
    phase_2d_helper::sortArr(p_intensity, ndx_sort, true /** b_descend*/);

    /**
     * Define a sorted peak list that is near the diagonal line (abs(px/xdim-py/ydim)<=0.01)
    */
    std::vector<int> px_sort_on_diagonal, py_sort_on_diagonal;
    std::vector<float> p_intensity_sort_on_diagonal;

    /**
     * Define a sorted peak list that is not near the diagonal line (abs(px/xdim-py/ydim)>0.01)
     */
    std::vector<int> px_sort_off_diagonal, py_sort_off_diagonal;
    std::vector<float> p_intensity_sort_off_diagonal;

    for (int i = 0; i < ndx_sort.size(); i++)
    {
        /**
         * If peak is near the diagonal line, insert it to diagonal peak list
         */
        if (abs((float)px[ndx_sort[i]] / xdim - (float)py[ndx_sort[i]] / ydim) <= 0.01)
        {
            /**
             * Only keep top 100 peaks
            */
            if(px_sort_on_diagonal.size()<100)
            {
                px_sort_on_diagonal.push_back(px[ndx_sort[i]]);
                py_sort_on_diagonal.push_back(py[ndx_sort[i]]);
                p_intensity_sort_on_diagonal.push_back(p_intensity[ndx_sort[i]]);
            }
        }
        else
        {
            /**
             * Only keep top 500 peaks
            */
            if(px_sort_off_diagonal.size()<500)
            {
                px_sort_off_diagonal.push_back(px[ndx_sort[i]]);
                py_sort_off_diagonal.push_back(py[ndx_sort[i]]);
                p_intensity_sort_off_diagonal.push_back(p_intensity[ndx_sort[i]]);
            }
            /**
             * break if we have 500 peaks in the off-diagonal list
            */
            else
            {
                break;
            }
        }
    }

    /**
     * Define a combined peak list
     */
    std::vector<int> px_sort0 = px_sort_on_diagonal;
    std::vector<int> py_sort0 = py_sort_on_diagonal;
    std::vector<float> p_intensity_sort0 = p_intensity_sort_on_diagonal;

    px_sort0.insert(px_sort0.end(), px_sort_off_diagonal.begin(), px_sort_off_diagonal.end());
    py_sort0.insert(py_sort0.end(), py_sort_off_diagonal.begin(), py_sort_off_diagonal.end());
    p_intensity_sort0.insert(p_intensity_sort0.end(), p_intensity_sort_off_diagonal.begin(), p_intensity_sort_off_diagonal.end());

    /**
     * We need to resort the combined peak list by intensity
     */
    ndx_sort.clear();
    phase_2d_helper::sortArr(p_intensity_sort0, ndx_sort, true /** b_descend*/);

    std::vector<int> px_sort, py_sort;
    std::vector<float> p_intensity_sort;
    for(int m=0;m<ndx_sort.size();m++)
    {
        px_sort.push_back(px_sort0[ndx_sort[m]]);
        py_sort.push_back(py_sort0[ndx_sort[m]]);
        p_intensity_sort.push_back(p_intensity_sort0[ndx_sort[m]]);
    }

    /**
     * For debug, save peak list to file
     */
    fout.open("peak_list.txt");
    for (int i = 0; i < px_sort.size(); i++)
    {
        fout << px_sort[i] << " " << py_sort[i] << " " << p_intensity_sort[i] << std::endl;
    }
    fout.close();

    /**
     * Get a unique set of py_sort and px_sort. 
     * get_unique doesn't change the order of input array
     * They are still sorted by intensity from high to low
     */
    std::vector<int> py_sort_unique = phase_2d_helper::get_unique(py_sort);
    
    std::vector<int> px_sort_unique = phase_2d_helper::get_unique(px_sort);
    


    /**
     * For px_sort_unique, select a subset of them (40) to calculate entropy.
     * We want to have a good coverage of the whole spectrum
     * So, we will select 10 from 0-0.25, 10 from 0.25-0.5, 10 from 0.5-0.75, 10 from 0.75-1.0 of [0,x_dim]
     * Do the same for py_sort_unique
     */
    px_sort_unique = phase_2d_helper::selected_subset_uniformly(px_sort_unique, 10, 4, xdim);
    py_sort_unique = phase_2d_helper::selected_subset_uniformly(py_sort_unique, 10, 4, ydim);

    /**
     * Varibles for code readability. Compiler doesn't care about nrow or py_sort_unique.size()
    */
    int nrow = py_sort_unique.size();
    int ncol = px_sort_unique.size();

    float p0_direct;

    /**
     * If indirect_p0 is set (>-360), we apply phase correction along indirect dimension, there is no need to run phase correction along direct dimension
     */
    if (indirect_p0 > -360)
    {
        /**
         * Conversion from bruker definition to nmrPipe definition
         */
        indirect_p0 = -indirect_p0;
        indirect_p1 = -indirect_p1;
        std::cout << "Will apply phase correction along indirect dimension using pulse program provided P0: " << indirect_p0 << " P1: " << indirect_p1 << std::endl;

        /**
         * Apply phase correction along indirect dimension according to pulse program
         */
        phase_correction_worker(ydim, xdim, indirect_p0 /** P0 */, indirect_p1 /** P1*/, false, spectrum_real_real, spectrum_real_imag, nullptr, nullptr);
        phase_correction_worker(ydim, xdim, indirect_p0 /** P0 */, indirect_p1 /** P1*/, false, spectrum_imag_real, spectrum_imag_imag, nullptr, nullptr);

        /**
         * Update final phase correction values
         */
        final_p0_indirect = indirect_p0;
        final_p1_indirect = indirect_p1;

        /**
         * Now we need run phase correction along direct dimension.
         * We do this on selected rows only
         */
        std::vector<float> spectrum_real_real_selected(nrow * xdim), spectrum_image_real_selected(nrow * xdim);
        for (int i = 0; i < nrow; i++)
        {
            for (int j = 0; j < xdim; j++)
            {
                spectrum_real_real_selected[i * xdim + j] = spectrum_real_real[py_sort_unique[i] * xdim + j];
                spectrum_image_real_selected[i * xdim + j] = spectrum_imag_real[py_sort_unique[i] * xdim + j];
            }
        }

        /**
         * Rough tune of P0, from -180 to 180, step size 5
         */
        p0_direct = 0.0f;
        std::vector<float> p0s, entropies;
        p0s.clear();
        entropies.clear();

        entropy_based_p0_correction(nrow, xdim, 36 /** nstep*/, 5.0 /** step_size*/, p0_direct /** P0 center*/,
                                    spectrum_real_real_selected.data(), spectrum_image_real_selected.data(),
                                    p0s, entropies);
        p0_direct = p0s[0];
        float min_entropy = entropies[0];
        for (int i = 1; i < p0s.size(); i++)
        {
            if (entropies[i] < min_entropy)
            {
                p0_direct = p0s[i];
                min_entropy = entropies[i];
            }
        }
        std::cout << "Step 1, P0 direct: " << p0_direct << std::endl;
    }
    else
    {
        /**
         * Calculate phase correction on both dimension at the same time because they are not independent
         * Define a test list for indirect dimension P0,P1 pair
         */
        std::vector<std::pair<float, float>> possible_indirect_p0_p1_list = {
            {0.0f, 0.0f},
            {0.0f, 180.0f},
            {22.5f, 0.0f},
            {90.0f, 0.0f},
            {90.0f, -180.0f},
            {-90.0f, 0.0f},
            {-90.0f, 180.0f}};

        /**
         * Define a temporary spectrum to hold the indirect dimension only phase corrected spectrum
         */
        float *spectrum_real_real_test = new float[xdim * ydim];
        float *spectrum_real_imag_test = new float[xdim * ydim];
        float *spectrum_imag_real_test = new float[xdim * ydim];
        float *spectrum_imag_imag_test = new float[xdim * ydim];

        float *spectrum_real_real_test2 = new float[xdim * ydim];
        float *spectrum_imag_real_test2 = new float[xdim * ydim];
        float *spectrum_real_real_test2_transposed = new float[px_sort_unique.size() * ydim];

        /**
         * Loop over all possible P0,P1 pairs
         */
        std::vector<float> entropies, p0s; // p0s is for direct dimension
        for (int i = 0; i < possible_indirect_p0_p1_list.size(); i++)
        {
            /**
             * Apply indirect dimension phase correction, save new spectrum to spectrum_real_real_test, spectrum_real_imag_test, spectrum_imag_real_test, spectrum_imag_imag_test
             */
            phase_correction_worker(ydim, xdim, possible_indirect_p0_p1_list[i].first /** P0 */, possible_indirect_p0_p1_list[i].second /** P1*/, false /**along indirect*/, spectrum_real_real, spectrum_real_imag, spectrum_real_real_test, spectrum_real_imag_test);
            phase_correction_worker(ydim, xdim, possible_indirect_p0_p1_list[i].first /** P0 */, possible_indirect_p0_p1_list[i].second /** P1*/, false /**along indirect*/, spectrum_imag_real, spectrum_imag_imag, spectrum_imag_real_test, spectrum_imag_imag_test);
            /**
             * now varing P0 from [-180 to 180), step size 10.
             */
            for (int j = -18; j < 18; j++)
            {
                /**
                 * direct dimension phase correction, save new spectrum to spectrum_real_real_test2, spectrum_imag_real_test2
                 */
                float p0 = j * 10.0;
                // double t1=get_wall_time();
                phase_correction_worker(ydim, xdim, p0, 0.0 /**p1*/, true /**along direct dim*/, spectrum_real_real_test, spectrum_imag_real_test, spectrum_real_real_test2, spectrum_imag_real_test2);
                // double t2=get_wall_time();
                // std::cout<<"Spend "<<t2-t1<<" seconds to run phase correction along direct dimension"<<std::endl;
                /**
                 * Get a transposed version of spectrum_real_real_test2, selected columns only at px_sort_unique
                 */
                for (int k = 0; k < px_sort_unique.size(); k++)
                {
                    for (int m = 0; m < ydim; m++)
                    {
                        spectrum_real_real_test2_transposed[m + k * ydim] = spectrum_real_real_test2[px_sort_unique[k] + m * xdim];
                    }
                }

                double entropy = 0.0;
                /**
                 * Now calcualte entropy using spectrum_real_real_test2, for all rows at py_sort_unique
                 */
                for (int k = 0; k < nrow; k++)
                {
                    entropy += phase_2d_helper::calculate_entropy_of_spe(spectrum_real_real_test2 + py_sort_unique[k] * xdim, xdim);
                }
                /**
                 * now calcualte entropy using spectrum_real_real_test2, for all columns at px_sort_unique
                 */
                for (int k = 0; k < ncol; k++)
                {
                    entropy += phase_2d_helper::calculate_entropy_of_spe(spectrum_real_real_test2_transposed + k* ydim, ydim);
                }

                std::cout << "P0: " << p0 << " entropy: " << entropy << " at indirec P0,P1 pair: " << possible_indirect_p0_p1_list[i].first << " " << possible_indirect_p0_p1_list[i].second << std::endl;
                entropies.push_back(entropy);
                p0s.push_back(p0);
            }
        }
        /**
         * At this time, size of p0s and entropies should be possible_indirect_p0_p1_list.size()*18*2
         * 0-36: at possible_indirect_p0_p1_list 0
         * 37-72: at possible_indirect_p0_p1_list 1
         * etc.
         * We need to find the minimum entropy to locate best indirect dimension P0,P1 pair and direct dimension P0s
         */
        float min_entropy = entropies[0];
        int min_entropy_index = 0;
        for (int i = 1; i < entropies.size(); i++)
        {
            if (entropies[i] < min_entropy)
            {
                min_entropy = entropies[i];
                min_entropy_index = i;
            }
        }
        /**
         * Now we know the best indirect dimension P0,P1 pair is possible_indirect_p0_p1_list[min_entropy_index/36]
         * and the best direct dimension P0 is p0s[min_entropy_index%36]
         * Note that int division is used here
         * We then run direct dimension phase correction at fine interval and with P1 varied
         */
        int best_indirect_p0_p1_index = min_entropy_index / 36;
        int best_direct_p0_index = min_entropy_index % 36;

        indirect_p0 = possible_indirect_p0_p1_list[best_indirect_p0_p1_index].first;
        indirect_p1 = possible_indirect_p0_p1_list[best_indirect_p0_p1_index].second;

        p0_direct = p0s[best_direct_p0_index];

        std::cout << "Best indirect dimension P0,P1 pair: " << indirect_p0 << " " << indirect_p1 << std::endl;
        std::cout << "Best direct dimension P0: " << p0_direct << std::endl;

        /**
         * Update final phase correction values
         */
        final_p0_indirect = indirect_p0;
        final_p1_indirect = indirect_p1;

        /**
         * Release memory
        */   
        delete [] spectrum_real_real_test;
        delete [] spectrum_real_imag_test;
        delete [] spectrum_imag_real_test;
        delete [] spectrum_imag_imag_test;
        delete [] spectrum_real_real_test2;
        delete [] spectrum_imag_real_test2;
        delete [] spectrum_real_real_test2_transposed;

        /**
         * Do phase correction along indirect dimension using best indirect dimension P0,P1 pair
         * Notice we updated member varibles spectrum_real_real, spectrum_real_imag, spectrum_imag_real, spectrum_imag_imag
         */
        phase_correction_worker(ydim, xdim, indirect_p0 /** P0 */, indirect_p1 /** P1*/, false, spectrum_real_real, spectrum_real_imag, nullptr, nullptr);
        phase_correction_worker(ydim, xdim, indirect_p0 /** P0 */, indirect_p1 /** P1*/, false, spectrum_imag_real, spectrum_imag_imag, nullptr, nullptr);

    }

    /**
     * At this time, we finished indirect dimension phase correction
     * and we have a roughly tuned direct dimension P0_direct
     * Rough tune of P0: from -50 to 50 at 5 degree step size
     * and P1: from -50 to 50 at 5 degree step size
     */
    p0_direct = 0.0f;
    float p1_direct = 0.0f;
    std::vector<float> p1s, p0s, entropies;
    p0s.clear();
    p1s.clear();
    entropies.clear();

    /**
     * spectrum_real_real_selected and spectrum_image_real_selected are used to hold selected rows (at py_sort_unique) of spectrum_real_real and spectrum_imag_real
    */
    std::vector<float> spectrum_real_real_selected(nrow * xdim), spectrum_image_real_selected(nrow * xdim);
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < xdim; j++)
        {
            spectrum_real_real_selected[i * xdim + j] = spectrum_real_real[py_sort_unique[i] * xdim + j];
            spectrum_image_real_selected[i * xdim + j] = spectrum_imag_real[py_sort_unique[i] * xdim + j];
        }
    }

    entropy_based_p0_p1_correction(nrow, xdim, 9 /** nstep*/, 40.0 /** step_size*/, p0_direct /** P0 center*/,
                                   9 /** nstep*/, 40.0 /** step_size*/, p1_direct /** P1 center*/,
                                   spectrum_real_real_selected.data(), spectrum_image_real_selected.data(),
                                   p0s, p1s, entropies);
    p0_direct = p0s[0];
    p1_direct = p1s[0];
    float min_entropy = entropies[0];
    for (int i = 1; i < p0s.size(); i++)
    {
        if (entropies[i] < min_entropy)
        {
            p0_direct = p0s[i];
            p1_direct = p1s[i];
            min_entropy = entropies[i];
        }
    }
    std::cout << "Step 2, P0 direct: " << p0_direct << " P1 direct: " << p1_direct << std::endl;

    /**
     * Fine tune P0, around the rough tuned P0, -10 to 10, step size 0.5
     * and P1, around the rough tuned P1, -10 to 10, step size 0.5
     */
    p0s.clear();
    p1s.clear();
    entropies.clear();
    entropy_based_p0_p1_correction(nrow, xdim, 20 /** nstep*/,5.0 /** step_size*/, p0_direct /** P0 center*/,
                                   20 /** nstep*/,5.0 /** step_size*/, p1_direct /** P1 center*/,
                                   spectrum_real_real_selected.data(), spectrum_image_real_selected.data(),
                                   p0s, p1s, entropies);
    p0_direct = p0s[0];
    p1_direct = p1s[0];
    min_entropy = entropies[0];
    for (int i = 1; i < p0s.size(); i++)
    {
        if (entropies[i] < min_entropy)
        {
            p0_direct = p0s[i];
            p1_direct = p1s[i];
            min_entropy = entropies[i];
        }
    }
    std::cout << "Step 3, P0 direct: " << p0_direct << " P1 direct: " << p1_direct << std::endl;

    /**
     * Fine tune P0, around the rough tuned P0, -10 to 10, step size 0.5
     * and P1, around the rough tuned P1, -10 to 10, step size 0.5
     */
    p0s.clear();
    p1s.clear();
    entropies.clear();
    entropy_based_p0_p1_correction(nrow, xdim, 20 /** nstep*/,1.0 /** step_size*/, p0_direct /** P0 center*/,
                                   20 /** nstep*/,1.0 /** step_size*/, p1_direct /** P1 center*/,
                                   spectrum_real_real_selected.data(), spectrum_image_real_selected.data(),
                                   p0s, p1s, entropies);
    p0_direct = p0s[0];
    p1_direct = p1s[0];
    min_entropy = entropies[0];
    for (int i = 1; i < p0s.size(); i++)
    {
        if (entropies[i] < min_entropy)
        {
            p0_direct = p0s[i];
            p1_direct = p1s[i];
            min_entropy = entropies[i];
        }
    }
    std::cout << "Step 4, P0 direct: " << p0_direct << " P1 direct: " << p1_direct << std::endl;

    /**
     * Apply phase correction along direct dimension.
     */
    phase_correction_worker(ydim, xdim, p0_direct /** P0 */, p1_direct /** P1*/, true, spectrum_real_real, spectrum_imag_real, nullptr, nullptr);
    phase_correction_worker(ydim, xdim, p0_direct /** P0 */, p1_direct /** P1*/, true, spectrum_real_imag, spectrum_imag_imag, nullptr, nullptr);

    /**
     * Update final phase correction values
     */
    final_p0_direct = p0_direct;
    final_p1_direct = p1_direct;

    return true;
}

/**
 * Run phase correction on the spectrum according to the user input
 * @param p0: phase correction degree 0-th order
 * @param p1: phase correction degree 1-st order
 * @param p0_indirect: phase correction degree 0-th order along indirect dimension
 * @param p1_indirect: phase correction degree 1-st order along indirect dimension
 */
bool spectrum_phasing::set_user_phase_correction(double p0_direct, double p1_direct, double p0_indirect, double p1_indirect)
{
    /**
     * Apply phase correction along indirect dimension.
     */
    phase_correction_worker(ydim, xdim, p0_indirect /** P0 */, p1_indirect /** P1*/, false, spectrum_real_real, spectrum_real_imag, nullptr, nullptr);
    phase_correction_worker(ydim, xdim,  p0_indirect /** P0 */, p1_indirect /** P1*/, false, spectrum_imag_real, spectrum_imag_imag, nullptr, nullptr);

    /**
     * Apply phase correction along direct dimension.
     */
    phase_correction_worker(ydim, xdim, p0_direct /** P0 */, p1_direct /** P1*/, true, spectrum_real_real, spectrum_imag_real, nullptr, nullptr);
    phase_correction_worker(ydim, xdim, p0_direct /** P0 */, p1_direct /** P1*/, true, spectrum_real_imag, spectrum_imag_imag, nullptr, nullptr);

    /**
     * In case user want to save the final phase correction values
     */
    final_p0_direct = p0_direct;
    final_p1_direct = p1_direct;
    final_p0_indirect = p0_indirect;
    final_p1_indirect = p1_indirect;
    
    return true;
}


/**
 * Save the final result to file
 * 4 numbers seperated by space in one line
 * p0_direct p1_direct p0_indirect p1_indirect
 * @param filename: output file name
 */
bool spectrum_phasing::save_phase_correction_result(std::string filename) const
{
    std::ofstream fout(filename);
    fout << final_p0_direct << " " << final_p1_direct << " " << final_p0_indirect << " " << final_p1_indirect << std::endl;
    fout.close();
    return true;
}
    /**
     * User segment by segment variance method to estimate the noise level of the spectrum.
     * size of spectrum is ndim1 * ndim2. dim2 is major dimension.
    */
float spectrum_phasing::estimate_noise_level(int ydim, int xdim, float *spect) const
{
    std::cout << "In noise estimation, xdim*ydim is " << xdim * ydim << std::endl;

    int n_segment_x = xdim / 32;
    int n_segment_y = ydim / 32;

    std::vector<float> variances;      // variance of each segment
    std::vector<float> maximal_values; // maximal value of each segment

    /**
     * loop through each segment, and calculate variance
     */
    for (int i = 0; i < n_segment_x; i++)
    {
        for (int j = 0; j < n_segment_y; j++)
        {
            std::vector<float> t;
            for (int m = 0; m < 32; m++)
            {
                for (int n = 0; n < 32; n++)
                {
                    t.push_back(spect[(j * 32 + m) * xdim + i * 32 + n]);
                }
            }

            /**
             * calculate variance of this segment. Substract the mean value of this segment first
             * also calculate the max value of this segment
             */
            float max_of_t = 0.0f;
            float mean_of_t = 0.0f;
            for (int k = 0; k < t.size(); k++)
            {
                mean_of_t += t[k];
                if (fabs(t[k]) > max_of_t)
                {
                    max_of_t = fabs(t[k]);
                }
            }
            mean_of_t /= t.size();

            float variance_of_t = 0.0f;
            for (int k = 0; k < t.size(); k++)
            {
                variance_of_t += (t[k] - mean_of_t) * (t[k] - mean_of_t);
            }
            variance_of_t /= t.size();
            variances.push_back(variance_of_t);
            maximal_values.push_back(max_of_t);
        }
    }

    /**
     * sort the variance, and get the median value
     */
    std::vector<float> variances_sorted = variances;
    sort(variances_sorted.begin(), variances_sorted.end());
    float noise_level = sqrt(variances_sorted[variances_sorted.size() / 2]);
    std::cout << "Noise level is " << noise_level << " using variance estimation in step 1" << std::endl;

    /**
     * loop through maximal_values, remove the ones that are larger than 10.0*noise_level
     * remove the corresponding variance as well
     */
    for (int i = maximal_values.size() - 1; i >= 0; i--)
    {
        if (maximal_values[i] > 10.0 * noise_level)
        {
            maximal_values.erase(maximal_values.begin() + i);
            variances.erase(variances.begin() + i);
        }
    }

    /**
     * sort the variance, and get the median value
     */
    variances_sorted = variances;
    sort(variances_sorted.begin(), variances_sorted.end());
    noise_level = sqrt(variances_sorted[variances_sorted.size() / 2]);

    std::cout << "Final noise level is estiamted to be " << noise_level << std::endl;

    return noise_level;
}



/**
 * This is the main working function, version 2
 * Still entropy based algorithm only at this time. Seems more than enough for most cases.
 */
bool spectrum_phasing::auto_phase_correction_v2()
{

    /**
     * Get a combined spectrum SQRT(real^2+imag^2) to find location of strong signal peaks
     */
    std::vector<float> combined_spectrum;
    for (int i = 0; i < xdim * ydim; i++)
    {
        combined_spectrum.push_back(sqrt(spectrum_real_real[i] * spectrum_real_real[i] + spectrum_real_imag[i] * spectrum_real_imag[i] + spectrum_imag_real[i] * spectrum_imag_real[i] + spectrum_imag_imag[i] * spectrum_imag_imag[i]));
    }

    /**
     * For debug, save combined spectrum to file
     */
    std::ofstream fout("combined_spectrum.txt");
    for (int i = 0; i < ydim; i++)
    {
        for (int j = 0; j < xdim; j++)
        {
            fout << combined_spectrum[j + i * xdim] << " ";
        }
        fout << std::endl;
    }
    fout.close();

    /**
     * Estimate noise level of the spectrum
    */
    std::cout<<"Estimating noise level of the combined spectrum"<<std::endl;
    float estimate_noise_level_combined_spectrum = estimate_noise_level(ydim, xdim, combined_spectrum.data());
    std::cout<<std::endl; //function will print noise level to screen

    /**
     * Run simple peak picking on combined spectrum and selected the top 10 peaks
     */
    std::vector<int> px, py;
    std::vector<float> p_intensity;
    float max_intensity = 0.0;

    for (int i = 2; i < xdim - 2; i++)
    {
        for (int j = 2; j < ydim - 2; j++)
        {
            /**
             * Peak is defined as data point that is larger than all its 8 neighbors
             * and is larger than 10 times of noise level
             */
            if (combined_spectrum[i + j * xdim] > combined_spectrum[i - 1 + (j - 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + (j - 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + 1 + (j - 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i - 1 + j * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + 1 + j * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i - 1 + (j + 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + (j + 1) * xdim] &&
                combined_spectrum[i + j * xdim] > combined_spectrum[i + 1 + (j + 1) * xdim] &&
                combined_spectrum[i + j * xdim] > estimate_noise_level_combined_spectrum*10.0f
                )
            {
                px.push_back(i);
                py.push_back(j);
                p_intensity.push_back(combined_spectrum[i + j * xdim]);
                if (combined_spectrum[i + j * xdim] > max_intensity)
                {
                    max_intensity = combined_spectrum[i + j * xdim];
                }
            }
        }
    }

    /**
     * Sort peaks by intensity and keep track of index
     */
    std::vector<int> ndx_sort;
    phase_2d_helper::sortArr(p_intensity, ndx_sort, true /** b_descend*/);

    std::vector<int> px_sort, py_sort;
    std::vector<float> p_intensity_sort;

    constexpr int n_peak_to_keep = 20;

    for (int i = 0; i < n_peak_to_keep; i++)
    {
        p_intensity_sort.push_back(p_intensity[ndx_sort[i]]);
        px_sort.push_back(px[ndx_sort[i]]);
        py_sort.push_back(py[ndx_sort[i]]);
    }

    px_sort.clear();
    py_sort.clear();
    p_intensity_sort.clear();

    px_sort.push_back(775);py_sort.push_back(291);p_intensity_sort.push_back(combined_spectrum[775+291*xdim]);
    px_sort.push_back(1177);py_sort.push_back(231);p_intensity_sort.push_back(combined_spectrum[1177+231*xdim]);
    px_sort.push_back(2051);py_sort.push_back(213);p_intensity_sort.push_back(combined_spectrum[2051+213*xdim]);
    px_sort.push_back(2324);py_sort.push_back(99);p_intensity_sort.push_back(combined_spectrum[2324+99*xdim]);
    px_sort.push_back(3361);py_sort.push_back(350);p_intensity_sort.push_back(combined_spectrum[3361+350*xdim]);

    /**
     * For top n_peak_to_keep peaks from the combined spectrum, get a row and column profile, extend to both directions
     * until amplitude is smaller than peak_intensity*0.05. Skip it if the peak is too close to the edge
     * or the row/column profile is too long (>100 points)
     * @param extend_left: used to hold the left end of the row profile, relative to the peak location (-x is left)
     * @param extend_right: used to hold the right end of the row profile, relative to the peak location (+x is right)
     * @param extend_up: used to hold the up end of the column profile, relative to the peak location (+y is up)
     * @param extend_down: used to hold the down end of the column profile, relative to the peak location (-y is down)
     */
    std::vector<int> extend_left, extend_right, extend_up, extend_down;

    for(int i=0;i<px_sort.size();i++)
    {   
        /**
         * Peak centers are px_sort[i], py_sort[i]
        */
        int peak_x = px_sort[i];
        int peak_y = py_sort[i];
        float peak_intensity = p_intensity_sort[i];

        /**
         * Skip if peak is too close to the edge
        */
        if(peak_x<80 || peak_x>xdim-80 || peak_y<80 || peak_y>ydim-80)
        {
            continue;
        }


        int left=0, right=0, up=0, down=0;
        for(int j=1;j<=peak_x;j++)
        {
            if(combined_spectrum[peak_x-j+peak_y*xdim]<peak_intensity*0.01f)
            {
                /**
                 * Once we find a point that is smaller than 40 times of noise level, we stop
                 * left is the distance from peak_x to this point
                */
                left = j;
                break;
            }
        }
       

        /**
         * Do the same for right, up, down
        */
        for(int j=1;j<xdim-peak_x;j++)
        {
            if(combined_spectrum[peak_x+j+peak_y*xdim]<peak_intensity*0.01f)
            {
                right = j;
                break;
            }
        }
       
        for(int j=1;j<=peak_y;j++)
        {
            if(combined_spectrum[peak_x+(peak_y-j)*xdim]<peak_intensity*0.01f)
            {
                down = j;
                break;
            }
        }
       

        for(int j=1;j<ydim-peak_y;j++)
        {
            if(combined_spectrum[peak_x+(peak_y+j)*xdim]<peak_intensity*0.01f)
            {
                up = j;
                break;
            }
        }

        left=90;
        right=90;
        up=90;
        down=90;
       
        extend_left.push_back(left);
        extend_right.push_back(right);
        extend_up.push_back(up);
        extend_down.push_back(down);
    }

    /**
     * If any of the extend_left, extend_right, extend_up, extend_down > 100 points, we skip this peak
    */
    for(int i=0;i<extend_left.size();i++)
    {
        if(extend_left[i]>400 || extend_right[i]>400 || extend_up[i]>100 || extend_down[i]>100)
        {
            extend_left.erase(extend_left.begin()+i);
            extend_right.erase(extend_right.begin()+i);
            extend_up.erase(extend_up.begin()+i);
            extend_down.erase(extend_down.begin()+i);
            /**
             * need to remove the corresponding px_sort, py_sort, p_intensity_sort as well
            */
            px_sort.erase(px_sort.begin()+i);
            py_sort.erase(py_sort.begin()+i);
            p_intensity_sort.erase(p_intensity_sort.begin()+i);
            i--;
        }
    }
    
    

    /**
     * double loop over all possible phase correction of both direct and indirect dimension
     * -180 to 180, step size 10. 
     */
    float *spectrum_real_real_test = new float[xdim * ydim];
    float *spectrum_real_imag_test = new float[xdim * ydim];
    float *spectrum_imag_real_test = new float[xdim * ydim];
    float *spectrum_imag_imag_test = new float[xdim * ydim];

    /**
     * @param entropies_direct: used to hold entropy of each peak, row profile
     * @param entropies_indirect: used to hold entropy of each peak, column profile
     * @param entropies_combined: used to hold entropy of each peak, combined profile
     * entropies_direct[i][j] is the entropy of peak i, row profile j ( j is the index of phase correction)
     * Because direct and indirect phase correction affect each other. We need a combined optimization problem. So, we also add themn together
    */
    std::vector<std::vector<float>> entropies_direct(px_sort.size()), entropies_indirect(px_sort.size()),entropies_combined(px_sort.size());

    for(int ndx_phase_direct = -18; ndx_phase_direct < 18; ndx_phase_direct++)
    {
        for(int ndx_phase_indirect = -18; ndx_phase_indirect < 18; ndx_phase_indirect++)
        {
            float phase_direct = ndx_phase_direct * 10.0f;
            float phase_indirect = ndx_phase_indirect * 10.0f;

            /**
             * Apply phase correction along direct dimension
            */
            phase_correction_worker(ydim, xdim, phase_direct /** P0 */, 0.0 /** P1*/, true, spectrum_real_real, spectrum_imag_real, spectrum_real_real_test, spectrum_imag_real_test);
            phase_correction_worker(ydim, xdim, phase_direct /** P0 */, 0.0 /** P1*/, true, spectrum_real_imag, spectrum_imag_imag, spectrum_real_imag_test, spectrum_imag_imag_test);

            /**
             * Apply phase correction along indirect dimension. We only need spectrum_real_real_test for entropy calculation 
            */
            phase_correction_worker(ydim, xdim, phase_indirect /** P0 */, 0.0 /** P1*/, false, spectrum_real_real_test, spectrum_real_imag_test, nullptr, nullptr);

            /**
             * Calculate entropy of each peak, both row (from extend_left to extend_right) and column (from extend_down to extend_up) profiles.
            */
            for(int k=0;k<px_sort.size();k++)
            {
                /**
                 * Get a row profile
                */
                std::vector<float> row_profile;
                for(int m=0;m<extend_left[k]+extend_right[k];m++)
                {
                    row_profile.push_back(spectrum_real_real_test[px_sort[k]-extend_left[k]+m+py_sort[k]*xdim]);
                }
                /**
                 * Get a column profile
                */
                std::vector<float> column_profile;
                for(int m=0;m<extend_up[k]+extend_down[k];m++)
                {
                    column_profile.push_back(spectrum_real_real_test[px_sort[k]+(py_sort[k]-extend_down[k]+m)*xdim]);
                }

                /**
                 * Calculate entropy of row profile
                */
                float s1=phase_2d_helper::calculate_entropy_of_spe(row_profile.data(), row_profile.size());
                entropies_direct[k].push_back(s1);
                /**
                 * Calculate entropy of column profile
                */
                float s2=phase_2d_helper::calculate_entropy_of_spe(column_profile.data(), column_profile.size());
                entropies_indirect[k].push_back(s2);
                /**
                 * Get sum of entropy of the two profiles
                */
                entropies_combined[k].push_back(s1+s2);
                
            }
        }
    }
    
    /**
     * Because direct and indirect phase correction affect each other. We need a combined optimization problem. 
     * For each peak, find best direct and indirect phase correction pair. Best means the smallest entropy
    */
    std::vector<float> best_direct_phase_correction(px_sort.size()), best_indirect_phase_correction(px_sort.size());
    for(int i=0;i<px_sort.size();i++)
    {
        float min_entropy = entropies_combined[i][0];
        int min_entropy_index = 0;
        for(int j=1;j<entropies_combined[i].size();j++)
        {
            if(entropies_combined[i][j]<min_entropy)
            {
                min_entropy = entropies_combined[i][j];
                min_entropy_index = j;
            }
        }
        best_indirect_phase_correction[i] = (min_entropy_index%36 -18) * 10.0f;
        best_direct_phase_correction[i] = (min_entropy_index/36-18) * 10.0f; // int division is used here
    }

    /**
     * Debug print px_sort, py_sort, best_direct_phase_correction, best_indirect_phase_correction
    */
    std::ofstream fout3("phase_correction_result.txt");
    for(int i=0;i<px_sort.size();i++)
    {
        fout3<<px_sort[i]<<" "<<py_sort[i]<<" "<<best_direct_phase_correction[i]<<" "<<best_indirect_phase_correction[i]<<std::endl;
    }
    fout3.close();
   
    return true;
}
