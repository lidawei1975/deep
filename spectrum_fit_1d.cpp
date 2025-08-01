#include <vector>
#include <array>
#include <valarray>
#include <fstream>
#include <iostream>
#include <random>
#include <limits>

#include "kiss_fft.h"
#include "json/json.h"
#include "commandline.h"
#include "fid_1d.h"
#include "cost_functors.h"
#include "spectrum_fit_1d.h"

#ifdef WEBASSEMBLY
/**
* Bind to emscripten with exposed class methods.
*/
#include <emscripten/bind.h>
using namespace emscripten;
#endif

#define CONVOLUTION_RANGE 20.0 



namespace ldw_math_1d
{
    double calcualte_median(std::vector<double> scores)
    {
        size_t size = scores.size();

        if (size == 0)
        {
            return 0; // Undefined, really.
        }
        else
        {
            sort(scores.begin(), scores.end());
            if (size % 2 == 0)
            {
                return (scores[size / 2 - 1] + scores[size / 2]) / 2;
            }
            else
            {
                return scores[size / 2];
            }
        }
    };

    void sortArr(std::vector<double> &arr, std::vector<int> &ndx)
    {
        std::vector<std::pair<double, int>> vp;

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

    bool gaussian_convolution(double a, double x, double sigmax, std::vector<double> *kernel, int &i0, int &i1, int xdim, double scale)
    {

        double wx = 2.355 * sigmax;

        i0 = std::max(0, int(x - wx * scale + 0.5));
        i1 = std::min(xdim, int(x + wx * scale + 0.5));

        kernel->clear();
        kernel->resize((i1 - i0));

        double sigmax2 = sigmax * sigmax * 2.0;

        for (int i = i0; i < i1; i++)
        {
            double t1 = x - i;
            kernel->at(i - i0) = a * exp(-(t1 * t1) / sigmax2);
        }
        return true;
    };

    bool lorentz_convolution(double a, double x, double gammax, std::vector<double> *kernel, int &i0, int &i1, int xdim, double scale)
    {

        double wx = 2.0 * gammax;

        i0 = std::max(0, int(x - wx * scale + 0.5));
        i1 = std::min(xdim, int(x + wx * scale + 0.5));

        kernel->clear();
        kernel->resize((i1 - i0));

        for (int i = i0; i < i1; i++)
        {
            double t1 = x - i;
            kernel->at(i - i0) = a / (1 + (t1 / gammax) * (t1 / gammax));
        }
        return true;
    };

    bool voigt_convolution(double a, double x, double sigmax, double gammax, std::vector<double> *kernel, int &i0, int &i1, int xdim, double scale)
    {
        double wx = 0.5346 * gammax * 2 + std::sqrt(0.2166 * 4 * gammax * gammax + sigmax * sigmax * 8 * 0.6931);

        i0 = std::max(0, int(x - wx * scale + 0.5));
        i1 = std::min(xdim, int(x + wx * scale + 0.5));

        if (i1 <= i0)
            return false;

        kernel->clear();
        kernel->resize(i1 - i0);

        for (int i = i0; i < i1; i++)
        {
            double z1 = voigt(i - x, sigmax, gammax);
            kernel->at(i - i0) = a * z1;
        }
        return true;
    };
}


// mixed guassian fitter
gaussian_fit_1d::gaussian_fit_1d()
{
    wx = 30;
    n_patch = 20;

#ifndef LMMIN
    // default fitting parameters
    options.max_num_iterations = 250;
    options.function_tolerance = 1e-12;
    options.parameter_tolerance = 1e-12;
    options.initial_trust_region_radius = 15.0;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
#endif

    return;
};
gaussian_fit_1d::~gaussian_fit_1d(){};

bool gaussian_fit_1d::gaussian_fit_init(std::vector<std::vector<float>> &d, std::vector<double> p1, std::vector<double> sigma, std::vector<double> gamma, std::vector<std::vector<double>> inten, std::vector<int> index)
{
    spectrum_scale = 0.0;

    // surfaces=d, convert from float to double
    // find maxiaml abs value and save it as spectrum_scale
    for (int i = 0; i < d.size(); i++)
    {
        std::vector<double> tmp;
        for (int j = 0; j < d[i].size(); j++)
        {
            tmp.push_back(d[i][j]);
            if (fabs(d[i][j]) > spectrum_scale)
            {
                spectrum_scale = fabs(d[i][j]);
            }
        }
        surfaces.push_back(tmp);
        if(i==0)
        {
            surface=tmp;
        }
    }

    xdim = surface.size();
    nspect = surfaces.size();

    a = inten;
    x = p1;
    x_as_input = p1; // save original peak positions
    sigmax = sigma;
    gammax = gamma;
    original_ndx = index;

    original_peak_pos = x;
    original_spectral_height.clear();
    for (int i = 0; i < x.size(); i++)
    {
        original_spectral_height.push_back(surface[round(x[i])]/spectrum_scale);
    }

    // Sort all peaks from left to right to set x_range_left(right) as peak pos on the left(right)
    std::vector<int> ndx;
    ldw_math_1d::sortArr(p1, ndx);
    x_range_left.resize(p1.size());
    x_range_right.resize(p1.size());

    for (int i = 0; i < ndx.size(); i++)
    {
        int n = ndx[i];
        if (ndx.size() == 1)
        {
            x_range_left[n] = 0;
            x_range_right[n] = xdim;
        }
        else if (i == 0)
        {
            x_range_left[n] = 0;
            x_range_right[n] = x[ndx[i + 1]];
        }
        else if (i == ndx.size() - 1)
        {
            x_range_left[n] = x[ndx[i - 1]];
            x_range_right[n] = xdim;
        }
        else
        {
            x_range_left[n] = x[ndx[i - 1]];
            x_range_right[n] = x[ndx[i + 1]];
        }
    }

    if (type == gaussian_type) // if gaussian fit, 1 convert sigma and gamma to fwhh, 2 get sigma from fwhh
    {
        float s;
        for (int k = 0; k < p1.size(); k++)
        {
            s = 0.5346 * sigmax[k] * 2 + std::sqrt(0.2166 * 4 * gammax[k] * gammax[k] + sigmax[k] * sigmax[k] * 8 * 0.6931); // fwhh
            s = s / 2.355f;                                                                                                  // sigma
            sigmax[k] = s;
            gammax[k] = 1e-10;
        }
    }
    else if (type == lorentz_type) // if lorentz fit, 1 convert sigma and gamma to fwhh, 2 get gamma from fwhh
    {
        float s;
        for (int k = 0; k < p1.size(); k++)
        {
            s = 0.5346 * sigmax[k] * 2 + std::sqrt(0.2166 * 4 * gammax[k] * gammax[k] + sigmax[k] * sigmax[k] * 8 * 0.6931); // fwhh
            gammax[k] = s * 0.5;
            sigmax[k] = 1e-10;
        }
    }
    else if (type == voigt_type) // if Voigt fit, correct a
    {
        for (int k = 0; k < p1.size(); k++)
        {
            for(int m=0; m < nspect; m++)
            {
                a[k][m] /= voigt(0, sigmax[k], gammax[k]);
            }
        }
    }

    return true;
};

void gaussian_fit_1d::set_up(fit_type t_, int r_, double near, double scale_, double noise_, bool b_,double mw_)

{
    type = t_;
    rmax = r_;
    too_near_cutoff = near;
    minimal_height = scale_ * noise_;
    noise_level = noise_;
    b_negative=b_;
    median_width_x = mw_;
};

bool gaussian_fit_1d::save_postion_informations(int begin_, int stop_, int left_patch_, int right_patch_, int n)
{
    begin = begin_;
    stop = stop_;
    left_patch = left_patch_;
    right_patch = right_patch_;
    n_initial = n;
    return true;
}

int gaussian_fit_1d::get_nround()
{
    return nround;
}

int gaussian_fit_1d::test_possible_removal(double x1, double a1, double s1, double g1, double x2, double a2, double s2, double g2)
{
    double fwhh1 = 0.5346 * s1 * 2 + std::sqrt(0.2166 * 4 * g1 * g1 + s1 * s1 * 8 * 0.6931);
    double fwhh2 = 0.5346 * s2 * 2 + std::sqrt(0.2166 * 4 * g2 * g2 + s2 * s2 * 8 * 0.6931);

    double aa1 = a1 * voigt(0, s1, g1);
    double aa2 = a2 * voigt(0, s2, g2);

    double a, x, s, g;

    int p1 = int(round(x1));
    int p2 = int(round(x2));

    int pp1 = p1 - int(round(fwhh1 / 2));
    int pp2 = p2 + int(round(fwhh2 / 2));

    if (aa1 >= aa2)
    {
        x = x1;
        s = s1;
        g = g1;
    }
    else
    {
        x = x2;
        s = s2;
        g = g2;
    }
    std::vector<double> profile;

    double max_v = 0;
    int max_ndx = 0;
    for (int i = pp1; i <= pp2; i++)
    {
        double v = a1 * voigt(i - x1, s1, g1) + a2 * voigt(i - x2, s2, g2);
        if (v > max_v)
        {
            max_v = v;
            max_ndx = i;
        }
        profile.push_back(v);
    }

    // print out profile to file debug1.txt
    //  std::ofstream outfile;
    //  outfile.open("debug1.txt");
    //  for(int i=0;i<profile.size();i++)
    //  {
    //      outfile << i << " " << profile[i] << std::endl;
    //  }
    //  outfile.close();

    // search for peak in profile
    int npeak_t1 = 0;
    for (int i = 1; i < profile.size() - 1; i++)
    {
        if (profile[i] > profile[i - 1] && profile[i] > profile[i + 1])
        {
            npeak_t1++;
        }
    }
    if (npeak_t1 >= 2)
    {
        return 0;
    }

    // apply laplacian filter to profile
    std::vector<double> profile_filtered;
    profile_filtered.clear();
    for (int i = 1; i < profile.size() - 1; i++)
    {
        profile_filtered.push_back(-profile[i - 1] + 2 * profile[i] - profile[i + 1]);
    }

    // print out profile_filtered to file debug2.txt
    //  outfile.open("debug2.txt");
    //  for(int i=0;i<profile_filtered.size();i++)
    //  {
    //      outfile << i << " " << profile_filtered[i] << std::endl;
    //  }
    //  outfile.close();

    // search for peak in filtered profile
    int npeak_t2 = 0;
    for (int i = 1; i < profile_filtered.size() - 1; i++)
    {
        if (profile_filtered[i] > profile_filtered[i - 1] && profile_filtered[i] > profile_filtered[i + 1])
        {
            npeak_t2++;
        }
    }
    if (npeak_t2 >= 2)
    {
        return 0;
    }

    double e;
    x -= pp1;
    a = std::max(a1, a2);
    one_fit_voigt_core(pp2 - pp1 + 1, &profile, x, a, s, g, e);
    x += pp1;

    double max_e = 0;
    for (int i = pp1; i <= pp2; i++)
    {
        double e = fabs(profile[i - pp1] - a * voigt(i - x, s, g));
        if (e > max_e)
        {
            max_e = e;
        }
    }

    if (max_e / max_v > peak_combine_cutoff)
    {
        return 0;
    }
    else if (aa1 > aa2)
    {
        return 1;
    }
    else
    {
        return -1;
    }
};

bool gaussian_fit_1d::generate_random_noise(int m, int m2, std::vector<float> &noise_spectrum)
{
    noise_spectrum.clear();
    noise_spectrum.resize(m2, 0.0f);

    // init random engine
    std::random_device dev;
    std::mt19937 gen(dev());
    std::normal_distribution<float> distribution(0.0, 1.0);
    float scale = 1 / sqrtf(float(m)) / 0.371;

    float sp;
    kiss_fft_cfg cfg;
    kiss_fft_cpx *in, *out;

    in = new kiss_fft_cpx[m2];
    out = new kiss_fft_cpx[m2];
    for (int i = 0; i < m; i++)
    {
        if (i == 0)
            sp = 0.5;
        else
            sp = pow(sin(M_PI * 0.5 + M_PI * 0.896 / 2 / float(m) * i), 3.684);
        in[i].r = distribution(gen) * sp;
        in[i].i = distribution(gen) * sp;
    }
    for (int i = m; i < m2; i++)
    {
        in[i].r = 0.0f;
        in[i].i = 0.0f;
    }
    if ((cfg = kiss_fft_alloc(m2, 0, NULL, NULL)) != NULL)
    {
        kiss_fft(cfg, in, out);
        free(cfg);
    }
    for (int i = 0; i < m2; i++)
    {
        noise_spectrum[i] = out[i].r * scale;
    }

    return true;
}

bool gaussian_fit_1d::run_with_error_estimation(int zf1, int n_error_round)
{
    std::vector<double> good_x, good_sigmax, good_gammax;
    std::vector<std::vector<double>> good_a,good_num_sum;
    std::vector<double> good_err;
    std::vector<double> good_surface;
    std::vector<std::vector<double>> good_surfaces;
    std::vector<double> good_b_spline_values;
    int good_nround;

    run_peak_fitting(true); // run once without noise
    std::cout << "Finished fitting round without noise." << std::endl;

    if (a.size() == 0) // no peak left after fitting.
    {
        return false;
    }

    // save good result from noise free fitting.
    good_x = x;
    good_a = a;
    good_sigmax = sigmax;
    good_gammax = gammax;
    good_err = err;
    good_num_sum = num_sum;
    good_nround = nround;
    good_surface = surface;
    good_surfaces = surfaces; //keep in mind surface is nothing but surfaces[0]

    // generate theorical spectrum for error estimation
    std::vector<double> theortical_surface(xdim, 0.0);
    for (unsigned int i = 0; i < x.size(); i++)
    {

        int i0, i1;
        std::vector<double> temp_spectrum;
        temp_spectrum.clear();
        if (type == gaussian_type)
        {
            gaussain_convolution(a[i][0], x[i], sigmax[i], &(temp_spectrum), i0, i1, CONVOLUTION_RANGE);
        }
        else if (type == voigt_type)
        {
            voigt_convolution(a[i][0], x[i], sigmax[i], gammax[i], &(temp_spectrum), i0, i1, CONVOLUTION_RANGE);
        }
        else if (type == lorentz_type)
        {
            lorentz_convolution(a[i][0], x[i], gammax[i], &(temp_spectrum), i0, i1, CONVOLUTION_RANGE);
        }

        for (int ii = i0; ii < i1; ii++)
        {
            theortical_surface[ii] += temp_spectrum[ii - i0];
        }
    }

    for (int i = 0; i < n_error_round; i++)
    {
        // add noise_spectrum to surface
        
        int xdim1 = xdim;
        int xdim0 = ceil(double(xdim1) / double(zf1));

        for(int k=0;k<nspect;k++)
        {
            std::vector<float> noise_spectrum;
            generate_random_noise(xdim0, xdim1, noise_spectrum);
            if(k==0)
            {
                for (int j = 0; j < xdim; j++)
                {
                    surface[j] = good_surface[j] + noise_spectrum[j] * noise_level;
                }
            }
            for(int j=0;j<xdim;j++)
            {
                surfaces[k][j] = good_surfaces[k][j] + noise_spectrum[j] * noise_level;
            }
        }

        /**
         * @brief If b_negavie is false. We don't allow negative data point.
         * 
         */
        if (b_negative == false)
        {
            for (int j = 0; j < xdim; j++)
            {
                if (surface[j] < 0.0)
                {
                    surface[j] = 0.0;
                }
            }
            for(int k=0;k<nspect;k++)
            {
                for (int j = 0; j < xdim; j++)
                {
                    if (surfaces[k][j] < 0.0)
                    {
                        surfaces[k][j] = 0.0;
                    }
                }
            }
        }

        x = good_x;
        a = good_a;
        sigmax = good_sigmax;
        gammax = good_gammax;

        run_peak_fitting(false); // run again with noise
        std::cout << "Finished " << i + 1 << "th round of error estimation." << std::endl;

        // push_back x to batch_x, a to batch_a, etc.
        batch_x.push_back(x);
        batch_a.push_back(a);
        batch_sigmax.push_back(sigmax);
        batch_gammax.push_back(gammax);
    }

    // copy good_x, etc back from saved values
    x = good_x;
    a = good_a;
    sigmax = good_sigmax;
    gammax = good_gammax;
    err = good_err;
    num_sum = good_num_sum;
    nround = good_nround;
    surface = good_surface;

    return true;
}

bool gaussian_fit_1d::run_peak_fitting(bool flag_first)
{
    int npeak = x.size();
    bool b = false;

    for (int k = 0; k < surfaces.size(); k++)
    {
        for (int i = 0; i < xdim; i++)
        {
            surfaces[k][i] /= spectrum_scale;
        }
    }
    for(int i = 0; i < xdim; i++)
    {
        surface[i] /= spectrum_scale;
    }

    minimal_height /= spectrum_scale;
    noise_level /= spectrum_scale;


    for (int i = 0; i < npeak; i++)
    {
        for (int k = 0; k < nspect; k++)
        {
            a[i][k] /= spectrum_scale;
        }
    }

    /**
     * @brief Set diffusion coefficient for each peak.
     * This varible is only useful for doesy experiment. But it is filled for all cases for simplicity.
     */
    diffusion_coefficient.resize(npeak, 0.0025);

    /**
     * @brief init guess of a[i][1,2,3,...,nspect-1] from a[i][0] and diffusion_coefficient if b_dosy is true.
     * 
     */
    if(b_dosy==true)
    {
        double z_gradient_squared_0 = z_gradients[0]*z_gradients[0];
        for(int i=0;i<npeak;i++)
        {
            for(int k=1;k<nspect;k++)
            {
                double z_gradient_squared = z_gradients[k]*z_gradients[k];
                a[i][k] = a[i][0] * exp(-diffusion_coefficient[i] * z_gradient_squared)/exp(-diffusion_coefficient[i] * z_gradient_squared_0);
            }
            a_at_time_zero.push_back(a[i][0]/exp(-diffusion_coefficient[i] * z_gradient_squared_0));
        }
    }

    if (flag_first == true)
    {
        if (npeak > 1)
        {
            limit_fitting_region_of_each_peak();
        }
        else
        {
            valid_fit_region.resize(1, std::array<int, 2>{{0, xdim - 1}});
        }
    }

    if (nspect == 1) // only one spectrum
    {
        if (npeak == 1)
        {
            b = run_single_peak();
        }
        else
        {
            b = run_multi_peaks();
        }
    }
    else // multiple spectra from pseudo-2D experiment
    {
        if (npeak == 1)
        {
            b = run_single_peak_multi_spectra();
        }
        else
        {
            b = run_multi_peaks_multi_spectra();
        }
    }

    // restore scale of spectrum by scale fitted intensity
    for (int i = 0; i < a.size(); i++)
    {
        for (int k = 0; k <nspect; k++)
        {
            a[i][k] *= spectrum_scale;
            num_sum[i][k] *= spectrum_scale;
        }
    }

    for (int k = 0; k < surfaces.size(); k++)
    {
        for (int i = 0; i < xdim; i++)
        {
            surfaces[k][i] *= spectrum_scale;
        }
    }

    for(int i = 0; i < xdim; i++)
    {
        surface[i] *= spectrum_scale;
    }

    minimal_height *= spectrum_scale;
    noise_level *= spectrum_scale;


    /**
     * Only remove peaks in the first round of fitting (with error estimation)
    */
    if (flag_first == true)
    {
        // actual remove peaks
        if(b_remove_failed_peaks == true)
        {
            for (int i = to_remove.size() - 1; i >= 0; i--)
            {
                if (to_remove[i] == 1)
                {
                    a.erase(a.begin() + i);
                    x.erase(x.begin() + i);
                    x_as_input.erase(x_as_input.begin() + i);
                    sigmax.erase(sigmax.begin() + i);
                    gammax.erase(gammax.begin() + i);
                    num_sum.erase(num_sum.begin() + i);
                    err.erase(err.begin() + i);
                    original_ndx.erase(original_ndx.begin() + i);
                    valid_fit_region.erase(valid_fit_region.begin() + i);
                    x_range_right.erase(x_range_right.begin() + i);
                    x_range_left.erase(x_range_left.begin() + i);
                    to_remove.erase(to_remove.begin() + i);
                    diffusion_coefficient.erase(diffusion_coefficient.begin() + i); //not sure this is needed. won't hurt
                }
            }
        }
        else
        {
            // if not remove failed peaks, make sure a is set to 0
            for (int i = 0; i < to_remove.size(); i++)
            {
                if (to_remove[i] == 1)
                {
                    for(int k = 0; k < nspect; k++)
                    {
                        a[i][k] = 0.0;
                        num_sum[i][k] = 0.0;
                    }
                    sigmax[i] = 1.0; //prevent division by zero in following calculations
                    gammax[i] = 1.0; //prevent division by zero in following calculations
                    err[i] = std::numeric_limits<double>::max();
                }
            }
        }

        for (int i = a.size() - 1; i >= 0; i--)
        {
            /**
             * @brief Remove peaks at the edge of the spectrum (mostlikely this peak is a main peak in a neighboring segment)
             * use final fitted peak position x[i] and n_patch to determine if the peak is at the edge if we can remove failed peaks.
             * If we don't remove failed peaks, we just remove peaks at the edge of the spectrum from the input (before fitting) because
             * peak can move from and to the edge of the spectrum during fitting.
            */
            if(b_remove_failed_peaks == true)
            {
                if (x_as_input[i] < n_patch || x_as_input[i] > surface.size() - n_patch)
                {
                    a.erase(a.begin() + i);
                    x.erase(x.begin() + i);
                    x_as_input.erase(x_as_input.begin() + i);
                    sigmax.erase(sigmax.begin() + i);
                    gammax.erase(gammax.begin() + i);
                    num_sum.erase(num_sum.begin() + i);
                    err.erase(err.begin() + i);
                    original_ndx.erase(original_ndx.begin() + i);
                    valid_fit_region.erase(valid_fit_region.begin() + i);
                    x_range_right.erase(x_range_right.begin() + i);
                    x_range_left.erase(x_range_left.begin() + i);
                    to_remove.erase(to_remove.begin() + i);
                    diffusion_coefficient.erase(diffusion_coefficient.begin() + i); //not sure this is needed. won't hurt
                }
            }
            else
            {
                if (x[i] < n_patch || x[i] > surface.size() - n_patch)
                {
                    a.erase(a.begin() + i);
                    x.erase(x.begin() + i);
                    x_as_input.erase(x_as_input.begin() + i);
                    sigmax.erase(sigmax.begin() + i);
                    gammax.erase(gammax.begin() + i);
                    num_sum.erase(num_sum.begin() + i);
                    err.erase(err.begin() + i);
                    original_ndx.erase(original_ndx.begin() + i);
                    valid_fit_region.erase(valid_fit_region.begin() + i);
                    x_range_right.erase(x_range_right.begin() + i);
                    x_range_left.erase(x_range_left.begin() + i);
                    to_remove.erase(to_remove.begin() + i);
                    diffusion_coefficient.erase(diffusion_coefficient.begin() + i); //not sure this is needed. won't hurt
                }
            }
        }
    }

    return b;
};

bool gaussian_fit_1d::find_highest_neighbor(int xx, int &mm)
{
    bool b_already_at_max = true;
    double current_a = surface[xx];
    double a_difference = 0.0;
    mm = 0;

    for (int m = -1; m <= 1; m++)
    {
        if (surface[xx] - current_a > a_difference)
        {
            a_difference = surface[xx + m] - current_a;
            mm = m;
            b_already_at_max = false;
        }
    }

    return b_already_at_max;
}

bool gaussian_fit_1d::limit_fitting_region_of_each_peak()
{
    valid_fit_region.clear();
    int npeak = x.size();
    for (int ndx = 0; ndx < npeak; ndx++)
    {
        std::array<int, 2> region = {0, xdim};
        int xx = round(x[ndx]);
        double aa = std::max(fabs(surface[xx]), fabs(surface[xx - 1]));
        aa = std::max(aa, fabs(surface[xx + 1]));
        bool b = true;
        for (int i = xx - 1; i >= std::max(0, xx - int(wx * 10)); i--)
        {
            region[0] = i + 1;
            double aa2 = aa;
            aa2 = std::max(aa2, fabs(surface[i]));
            if (fabs(surface[i]) > fabs(surface[i + 1]) && fabs(surface[i + 2]) > fabs(surface[i + 1]) && fabs(surface[i + 1]) < 0.8 * aa2)
            {
                b = false;
                break;
            }
        }
        if (b)
        {
            region[0] = 0;
        }

        b = true;
        for (int i = xx + 1; i < std::min(xdim, xx + int(wx * 10)); i++)
        {
            region[1] = i;
            double aa2 = aa;
            aa2 = std::max(aa2, fabs(surface[i]));
            if (fabs(surface[i]) > fabs(surface[i - 1]) && fabs(surface[i - 2]) > fabs(surface[i - 1]) && fabs(surface[i - 1]) < 0.8 * aa2)
            {
                b = false;
                break;
            }
        }
        if (b)
        {
            region[1] = xdim;
        }
        // expand by 1 point
        if (region[0] > 0)
            region[0] -= 1;
        if (region[1] < xdim - 1)
            region[1] += 1;

        valid_fit_region.push_back(region);
    }
    return true;
}

bool gaussian_fit_1d::run_single_peak()
{
    int npeak = 1;
    num_sum.resize(npeak, std::vector<double>(1, 0));
    err.resize(npeak, 0.0);
    to_remove.resize(npeak, 0);

    std::vector<double> zz;
    for (int ii = 0; ii < xdim; ii++)
    {
        double temp = surface[ii];
        zz.push_back(temp);
        num_sum[0][0] += temp;
    }

    //[peak index][spe index] for each peak and num_sum

    if (type == gaussian_type) // gaussian
    {
        one_fit_gaussian(xdim, &zz, x[0], a[0][0], sigmax[0], err[0]);
    }
    else if (type == lorentz_type) // lorentz
    {
        one_fit_lorentz(xdim, &zz, x[0], a[0][0], gammax[0], err[0]);
    }
    else if (type == voigt_type) // voigt
    {
        one_fit_voigt(xdim, &zz, x[0], a[0][0], sigmax[0], gammax[0], err[0], 0, 0);
    }
    nround = 1;

    return true;
};

bool gaussian_fit_1d::run_single_peak_multi_spectra()
{
    int npeak = 1;
    num_sum.resize(npeak, std::vector<double>(nspect, 0));
    err.resize(npeak, 0.0);
    to_remove.resize(npeak, 0);

    std::vector<std::vector<double>> zz; //zz[spectra index][x index]

    for(int k=0;k<nspect;k++)
    {
        double sum=0.0;
        for(int m=0;m<surfaces[k].size();m++)
        {
            sum+=surfaces[k][m];
        }
        num_sum[0][k]=sum;
        zz.push_back(surfaces[k]);
    }

    //[peak index][spe index] for each peak and num_sum
    

    if (type == gaussian_type) // gaussian
    {
        multi_fit_gaussian(xdim, zz, x[0], a[0], sigmax[0], err[0]);
    }
    else if (type == lorentz_type) // lorentz
    {
        multi_fit_lorentz(xdim, zz, x[0], a[0], gammax[0], err[0]);
    }
    else if (type == voigt_type && b_dosy==false) // voigt for pseudo 2d
    {
        multi_fit_voigt(xdim, zz, x[0], a[0], sigmax[0], gammax[0], err[0], 0, 0);
    }
    else if (type == voigt_type && b_dosy==true) // voigt for doesy
    {
        multi_fit_voigt_doesy(xdim, zz, x[0], a_at_time_zero[0], diffusion_coefficient[0], sigmax[0], gammax[0], err[0], 0, 1);
        /**
         * @brief fill a[0][1,2,3,...,nspec-1] using diffusion coefficient and a[0][0]
         * 
         */
        for(int k=0;k<nspect;k++)
        {
            double z_gradient_squared = z_gradients[k]*z_gradients[k];
            a[0][k]=a_at_time_zero[0]*exp(-diffusion_coefficient[0]*z_gradient_squared);
        }
    }
    nround = 1;

    return true;
}

bool gaussian_fit_1d::run_multi_peaks()
{
    int npeak = x.size();
    nround = 0;
    num_sum.resize(npeak, std::vector<double>(nspect, 0.0));
    err.resize(npeak, 0.0);

    double e;

    x_old.clear();
    to_remove.clear();
    to_remove.resize(x.size(), 0);

    bool flag_break = false;

    int loop;
    int loop2 = -1;
    for (loop = 0; loop < rmax; loop++)
    {
        loop2++;
        std::vector<double> analytical_spectra; // analytical spectrum for each peak
        std::vector<double> peaks_total;

        peaks_total.clear();
        peaks_total.resize(xdim, 0.0);

        for (unsigned int i = 0; i < x.size(); i++)
        {
            if (to_remove[i] == 1) // peak has been removed.
            {
                continue;
            }

            int i0, i1;
            analytical_spectra.clear();
            if (type == gaussian_type)
                gaussain_convolution(a[i][0], x[i], sigmax[i], &(analytical_spectra), i0, i1, CONVOLUTION_RANGE);
            else if (type == voigt_type)
                voigt_convolution(a[i][0], x[i], sigmax[i], gammax[i], &(analytical_spectra), i0, i1, CONVOLUTION_RANGE);
            else if (type == lorentz_type)
                lorentz_convolution(a[i][0], x[i], gammax[i], &(analytical_spectra), i0, i1, CONVOLUTION_RANGE);

            {
                for (int ii = i0; ii < i1; ii++)
                {
                    peaks_total[ii] += analytical_spectra[ii - i0];
                }
            }
        }

        // save old values so that we can check for convergence!
        x_old.push_back(x);

        // fit peaks one by one
        std::vector<int> peak_remove_flag;
        peak_remove_flag.resize(x.size(), 0);

        bool b_some_peak_removed = 0;
        for (int i = 0; i < x.size(); i++)
        {
            if (to_remove[i] == 1)
                continue; // peak has been removed.

            std::vector<double> zz;
            double total_z = 0.0;

            int i0, i1;
            analytical_spectra.clear();

            if (type == gaussian_type)
                gaussain_convolution_with_limit(i, a[i][0], x[i], sigmax[i], &(analytical_spectra), i0, i1, CONVOLUTION_RANGE);
            else if (type == voigt_type)
                voigt_convolution_with_limit(i, a[i][0], x[i], sigmax[i], gammax[i], &(analytical_spectra), i0, i1, CONVOLUTION_RANGE);
            else if (type == lorentz_type)
                lorentz_convolution_with_limit(i, a[i][0], x[i], gammax[i], &(analytical_spectra), i0, i1, CONVOLUTION_RANGE);

            for (int ii = i0; ii < i1; ii++)
            {
                double inten1 = analytical_spectra[ii - i0];
                double inten2 = peaks_total[ii];
                double scale_factor;
                if (fabs(inten2) > 1e-100)
                    scale_factor = inten1 / inten2;
                else
                    scale_factor = 0.0;

                scale_factor = std::min(scale_factor, 1.0);
                double temp = scale_factor * surface[ii];
                zz.push_back(temp);
                total_z += temp;
            }
            num_sum[i][0] = total_z;

            x[i] -= i0;

            if (type == gaussian_type)
            {
                one_fit_gaussian(i1 - i0, &zz, x[i], a[i][0], sigmax[i], err[i]);
            }
            else if (type == voigt_type)
            {
                one_fit_voigt(i1 - i0, &zz, x[i], a[i][0], sigmax[i], gammax[i], err[i], loop, loop2);
            }
            else if (type == lorentz_type)
            {
                one_fit_lorentz(i1 - i0, &zz, x[i], a[i][0], gammax[i], err[i]);
            }

            if (fabs(sigmax.at(i)) + fabs(gammax.at(i)) < 0.2 || fabs(sigmax.at(i)) + fabs(gammax.at(i)) > 10.0 * median_width_x)
            {
                if(n_verbose>1)
                {
                    std::cout << original_ndx[i] << " will be removed because too wide or narrow x=" << x.at(i) + i0 << " a=" << a[i][0] << " sigma=" << sigmax.at(i) << " gamma=" << gammax.at(i) << " total_z=" << total_z << std::endl;
                }
                peak_remove_flag[i] = 1;
                b_some_peak_removed = 1;
                continue; //no need to check other conditions
            }

            // if (x.at(i) < 0 || x.at(i) >= i1 - i0)
            // {
            //     if(n_verbose>1)
            //     {
            //         std::cout << original_ndx[i] << " will be removed because moved out of area  x=" << x.at(i) + i0 << " a=" << a[i][0] << " sigma=" << sigmax.at(i) << " gamma=" << gammax.at(i) << " total_z=" << total_z << std::endl;
            //     }
            //     peak_remove_flag[i] = 1;
            //     b_some_peak_removed = 1;
            //     continue; //no need to check other conditions
            // }

            //peah height check. if the peak height is too small, remove it.
            //height=a[i][0] for gaussian and lorentz but for voigt, we need to calculate it.
            double temp_peak_height = a[i][0];
            if (type == voigt_type)
            {
                temp_peak_height = a[i][0]*voigt(0.0,sigmax[i],gammax[i]);
            }
            
            if(fabs(temp_peak_height)<minimal_height)
            {
                if(n_verbose>1)
                {
                    std::cout << original_ndx[i] << " will be removed because low intensity x=" << x.at(i) + i0 << " height=" << temp_peak_height << " sigma=" << sigmax.at(i) << " gamma=" << gammax.at(i) << " total_z=" << total_z << std::endl;
                }
                peak_remove_flag[i] = 1;
                b_some_peak_removed = 1;
                continue; //no need to check other conditions
            }

            // if(x.at(i)+i0<x_range_left[i] || x.at(i)+i0>x_range_right[i])
            // {
            //     if(n_verbose>1)
            //     {
            //         std::cout << original_ndx[i] << " will be removed because moved out of range x=" << x.at(i) + i0 << " height=" << temp_peak_height << " sigma=" << sigmax.at(i) << " gamma=" << gammax.at(i) << " total_z=" << total_z << std::endl;
            //     }
            //     peak_remove_flag[i] = 1;
            //     b_some_peak_removed=1;
            //     continue; //no need to check other conditions
            // }

            
            x[i] += i0;

            // track peak movement
            int p_start = int(round(original_peak_pos[i]));
            int p_end = int(round(x[i]));
            int p_step = 1;
            if (p_end < p_start)
            {
                p_step = -1;
            }

            double min_of_trace, max_of_trace; //both are absolute values
            min_of_trace = std::numeric_limits<double>::max();
            max_of_trace = 0.0;
            for (int k = p_start; k <= p_end; k += p_step)
            {
                double temp = fabs(surface[k]);
                if (temp < min_of_trace)
                {
                    min_of_trace = temp;
                }
                if (temp > max_of_trace)
                {
                    max_of_trace = temp;
                }
            }

            if (min_of_trace / fabs(original_spectral_height[i]) < 1 / 3.0 || max_of_trace / fabs(original_spectral_height[i]) > 3.0)
            {
                if(n_verbose>1)
                {
                    std::cout << original_ndx[i] << " will be removed because moved too far away." << std::endl;
                }
                peak_remove_flag[i] = 1;
                b_some_peak_removed = 1;
            }
        } // end of parallel for(int i = 0; i < x.size(); i++)

        // also remove peaks if two peaks become very close.
        // that is, the program fit one peak with two overallped peaks, which can happen occasionally
        if (loop > 5 && loop2 > 5)
        {
            for (int k1 = 0; k1 < a.size(); k1++)
            {
                if (to_remove[k1] == 1)
                    continue; // peak has already been removed.
                if (peak_remove_flag[k1] == 2)
                    continue; // moved out of region peaks, do not use it too near calculation
                for (int k2 = k1 + 1; k2 < a.size(); k2++)
                {
                    if (to_remove[k2] == 1)
                        continue; // peak has already been removed.
                    if (peak_remove_flag[k2] == 2)
                        continue; // moved out of region peaks, do not use it too near calculation
                    double dx = x[k1] - x[k2];
                    double cut_near = wx * too_near_cutoff;
                    if (fabs(dx) < cut_near || fabs(dx) < 1.6) // too close peaks
                    {
                        if(n_verbose>1)
                        {
                            std::cout << "too_near_cutoff is " << too_near_cutoff << std::endl;
                        }

                        double height1 = fabs(a[k1][0]);
                        double height2 = fabs(a[k2][0]);

                        if (type == voigt_type)
                        {
                            height1 = fabs(a[k1][0]) * (voigt(0.0, sigmax[k1], gammax[k1]));
                            height2 = fabs(a[k2][0]) * (voigt(0.0, sigmax[k2], gammax[k2]));
                        }

                        if (height1 > height2)
                        {
                            // a[k2] = 0.0;
                            peak_remove_flag[k2] = 1;
                            if(n_verbose>1)
                            {
                                std::cout << original_ndx[k2] << " will be removed because too near " << original_ndx[k1] << ", distance is " << fabs(dx) << "<" << cut_near << std::endl;
                            }
                            if (peak_remove_flag[k1] == 1)
                            {
                                peak_remove_flag[k1] = 0; // restore k1, because of effect of k2 on it.
                            }
                        }
                        else
                        {
                            // a[k1] = 0.0;
                            peak_remove_flag[k1] = 1;
                            if(n_verbose>1)
                            {
                                std::cout << original_ndx[k1] << " will be removed because too near " << original_ndx[k2] << ", distance is " << fabs(dx) << "<" << cut_near << std::endl;
                            }
                            if (peak_remove_flag[k2] == 1)
                            {
                                peak_remove_flag[k2] = 0; // restore k2, because of effect of k1 on it.
                            }
                        }
                        flag_break = false;
                    }
                }
            }
        }

        // remove unvisible peak
        if (loop2 > 5)
        {
            std::vector<int> ndx;
            ldw_math_1d::sortArr(x, ndx);

            for (int i = ndx.size() - 1; i >= 0; i--)
            {
                if (to_remove[ndx[i]] == 1)
                {
                    ndx.erase(ndx.begin() + i);
                }
            }

            for (int i = 0; i < ndx.size() - 1; i++)
            {
                int ii = ndx[i];
                int jj = ndx[i + 1];
                int n = test_possible_removal(x[ii], a[ii][0], sigmax[ii], gammax[ii], x[jj], a[jj][0], sigmax[jj], gammax[jj]);
                if (n == 1)
                {
                    to_remove[jj] = 1;
                }
                else if (n == -1)
                {
                    to_remove[ii] = 1;
                }
            }
        }

        // lable peak to remove!!
        for (int i = peak_remove_flag.size() - 1; i >= 0; i--)
        {
            if (peak_remove_flag.at(i) != 0)
            {
                to_remove[i] = 1;
                b_some_peak_removed = 1;
            }
        }

        if (b_some_peak_removed == 1)
        {
            loop2 = -1;
        }

        if (flag_break)
        {
            break;
        }

        // test convergence. If so, we can break out of loop early
        bool bcon = false;
        for (int i = x_old.size() - 1; i >= std::max(int(x_old.size()) - 2, 0); i--)
        {
            if (x.size() != x_old[i].size())
            {
                continue;
            }

            bool b = true;
            for (int j = 0; j < x.size(); j++)
            {
                if (fabs(x[j] - x_old[i][j]) > 0.01)
                {
                    b = false;
                    break;
                }
            }
            if (b == true)
            {
                bcon = true;
                break;
            }
        }

        if (bcon == true || a.size() == 0)
        // if(a.size()==0)
        {
            flag_break = true;
        }
        if(n_verbose>1)
        {
            std::cout << "\r" << "Iteration " << loop + 1 << " " << loop2 << std::flush;
        }
    } // loop
    if(n_verbose>1)
    {
        std::cout << std::endl;
    }

    nround = loop;

    for (int i = sigmax.size() - 1; i >= 0; i--)
    {
        sigmax[i] = fabs(sigmax[i]);
        gammax[i] = fabs(gammax[i]);
    }

    for (int i = 0; i < to_remove.size(); i++)
    {
        if (to_remove[i] == 1)
        {
            a[i][0] = 0.0;
            sigmax[i] = 0.0;
            gammax[i] = 0.0;
            x[i] = xdim / 2.0;
        }
    }

    return true;
};



bool gaussian_fit_1d::run_multi_peaks_multi_spectra()
{
    to_remove.clear();
    to_remove.resize(a.size(), 0); //we will not use this in this function. But it need to have the same size as a.

    int npeak = x.size();
    nround = 0;
    num_sum.resize(npeak, std::vector<double>(nspect, 0.0));
    err.resize(npeak, 0.0);

    double e;

    x_old.clear();

    bool flag_break = false;

    int loop;
    for (loop = 0; loop < rmax; loop++)
    {
        std::vector<double> analytical_spectra; // analytical spectrum for each peak
        std::vector<std::vector<double>> peaks_total; //[spectra][x]

        peaks_total.clear();
        peaks_total.resize(nspect, std::vector<double>(xdim, 0.0));

        /**
         * Note: For doesy experiment, a[i][1,2,3 ...] is not fitted directly but calculated from a[i][0] and diffusion coefficient[i]
         * But they are still good for spectral deconvolution here. 
         */
        for(int k=0;k<nspect;k++)
        {
            for (unsigned int i = 0; i < x.size(); i++)
            {
                /**
                 * If all a[i][k] are 0.0, we skip this peak.
                 */
                bool b_all_zero=true;
                for(int kk=0;kk<nspect;kk++)
                {
                    if(a[i][kk]>std::numeric_limits<double>::epsilon() || a[i][kk]<-std::numeric_limits<double>::epsilon())
                    {
                        b_all_zero=false;
                        break;
                    }
                }
                if(b_all_zero)
                {
                    continue;
                }

                int i0, i1;
                analytical_spectra.clear();
                if (type == gaussian_type)
                    gaussain_convolution(a[i][k], x[i], sigmax[i], &(analytical_spectra), i0, i1, CONVOLUTION_RANGE);
                else if (type == voigt_type)
                    voigt_convolution(a[i][k], x[i], sigmax[i], gammax[i], &(analytical_spectra), i0, i1, CONVOLUTION_RANGE);
                else if (type == lorentz_type)
                    lorentz_convolution(a[i][k], x[i], gammax[i], &(analytical_spectra), i0, i1, CONVOLUTION_RANGE);

                for (int ii = i0; ii < i1; ii++)
                {
                    peaks_total[k][ii] += analytical_spectra[ii - i0];
                }
            }
        }

        // save old values so that we can check for convergence!
        x_old.push_back(x);

        # pragma omp parallel for
        for (int i = 0; i < x.size(); i++)
        {
            /**
             * If all a[i][k] are 0.0, we skip this peak.
             */
            bool b_all_zero = true;
            for (int kk = 0; kk < nspect; kk++)
            {
                if (a[i][kk] > std::numeric_limits<double>::epsilon() || a[i][kk] < -std::numeric_limits<double>::epsilon())
                {
                    b_all_zero = false;
                    break;
                }
            }
            if (b_all_zero)
            {
                continue;
            }

            int i0, i1;
            std::vector<std::vector<double>> zzz; //[spectra][x]
            for(int k=0;k<nspect;k++)
            {
                std::vector<double> zz; // deconvoluted spectrum for each peak
                double total_z = 0.0;

                std::vector<double> analytical_spectra_peak; // analytical spectrum for each peak
                analytical_spectra_peak.clear();

                if (type == gaussian_type)
                    gaussain_convolution_with_limit(i, a[i][k], x[i], sigmax[i], &(analytical_spectra_peak), i0, i1, CONVOLUTION_RANGE);
                else if (type == voigt_type)
                    voigt_convolution_with_limit(i, a[i][k], x[i], sigmax[i], gammax[i], &(analytical_spectra_peak), i0, i1, CONVOLUTION_RANGE);
                else if (type == lorentz_type)
                    lorentz_convolution_with_limit(i, a[i][k], x[i], gammax[i], &(analytical_spectra_peak), i0, i1, CONVOLUTION_RANGE);

                for (int ii = i0; ii < i1; ii++)
                {
                    double inten1 = analytical_spectra_peak[ii - i0];
                    double inten2 = peaks_total[k][ii];
                    double scale_factor;
                    if (fabs(inten2) > 1e-100)
                        scale_factor = inten1 / inten2;
                    else
                        scale_factor = 0.0;

                    scale_factor = std::min(scale_factor, 1.0);
                    double temp = scale_factor * surfaces[k][ii];
                    zz.push_back(temp);
                    total_z += temp;
                }
                num_sum[i][k] = total_z;
                zzz.push_back(zz);
            }
        
            x[i] -= i0;
            if (type == gaussian_type)
            {
                multi_fit_gaussian(i1 - i0, zzz, x[i], a[i], sigmax[i], err[i]);
            }
            else if (type == voigt_type && b_dosy==false)
            {
               multi_fit_voigt(i1 - i0, zzz, x[i], a[i], sigmax[i], gammax[i], err[i], loop, 100);
            }
            else if (type == voigt_type && b_dosy==true)
            {
                
                multi_fit_voigt_doesy(i1 - i0, zzz, x[i], a_at_time_zero[i], diffusion_coefficient[i], sigmax[i], gammax[i], err[i], loop, 100);
               /**
                 * @brief fill a[0][1,2,3,...,nspec-1] using diffusion coefficient and a[0][0]
                 * 
                 */
                for(int k=0;k<nspect;k++)
                {
                    double z_gradient_squared = z_gradients[k]*z_gradients[k];
                    a[i][k]=a_at_time_zero[i]*exp(-diffusion_coefficient[i]*z_gradient_squared);
                }
            }
            else if (type == lorentz_type)
            {
                multi_fit_lorentz(i1 - i0, zzz, x[i], a[i], gammax[i], err[i]);
            }

            if (fabs(sigmax.at(i)) + fabs(gammax.at(i)) < 0.2 || fabs(sigmax.at(i)) + fabs(gammax.at(i)) > 10.0 * median_width_x)
            {
                if(n_verbose>1)
                {
                    std::cout << original_ndx[i] << " will be removed because too wide or narrow x=" << x.at(i) + i0 << " a=" << a[i][0] << " sigma=" << sigmax.at(i) << " gamma=" << gammax.at(i) << std::endl;
                }
                /**
                 * Set a[i] to all 0.0 as flag. We keep them to keep original size and order of peaks
                 */
                for(int k=0;k<nspect;k++)
                {
                    a[i][k]=0.0;
                    to_remove[i]=1; // mark this peak to be removed
                }
            }

            
            if (x.at(i) < 0 || x.at(i) >= i1 - i0)
            {
                if(n_verbose>1)
                {
                    std::cout << original_ndx[i] << " will be removed because moved out of area  x=" << x.at(i) << " i0="<<i0<<" i1="<<i1<<" a=" << a[i][0] << " sigma=" << sigmax.at(i) << " gamma=" << gammax.at(i) << std::endl;
                }
                /**
                 * Set a[i] to all 0.0 as flag. We keep them to keep original size and order of peaks
                 */
                for(int k=0;k<nspect;k++)
                {
                    a[i][k]=0.0;
                    to_remove[i]=1; // mark this peak to be removed
                }
            }

            if(x.at(i)+i0<x_range_left[i] || x.at(i)+i0>x_range_right[i])
            {
                if(n_verbose>1)
                {
                    std::cout << original_ndx[i] << " will be removed because moved out of range x=" << x.at(i) + i0 << " sigma=" << sigmax.at(i) << " gamma=" << gammax.at(i) << std::endl;
                }
                /**
                 * Set a[i] to all 0.0 as flag. We keep them to keep original size and order of peaks
                 */
                for(int k=0;k<nspect;k++)
                {
                    a[i][k]=0.0;
                    to_remove[i]=1; // mark this peak to be removed
                }
            }

            x[i] += i0;
        } // end of parallel for(int i = 0; i < x.size(); i++)

        if(n_verbose>1)
        {
            std::cout << "\r"  << "Iteration " << loop + 1 << std::flush;
        }
    } // loop
    if(n_verbose>1)
    {
        std::cout << std::endl;
    }

    nround = loop;

    for (int i = sigmax.size() - 1; i >= 0; i--)
    {
        sigmax[i] = fabs(sigmax[i]);
        gammax[i] = fabs(gammax[i]);
    }
    return true;
}

bool gaussian_fit_1d::one_fit_gaussian(int xdim, std::vector<double> *zz, double &x0, double &a, double &sigmax, double &e)
{

#ifdef LMMIN



#else

    ceres::Solver::Summary summary;
    ceres::Problem problem;

    sigmax = 2 * sigmax * sigmax; // important: sigma is actually 2*sigma*sigma in mycostfunction_gaussian1d

    mycostfunction_gaussian1d *cost_function = new mycostfunction_gaussian1d(xdim, zz->data());
    cost_function->set_n_residuals(zz->size());
    for (int m = 0; m < 3; m++)
        cost_function->parameter_block_sizes()->push_back(1);
    problem.AddResidualBlock(cost_function, NULL, &a, &x0, &sigmax);

    ceres::Solve(options, &problem, &summary);
    e = sqrt(summary.final_cost / zz->size());

    sigmax = sqrt(fabs(sigmax) / 2.0); // important: sigma is actually 2*sigma*sigma in mycostfunction_gaussian1d

#endif
    return true;
};

//fit one Gaussian peak in pseudo-2D spectra
bool gaussian_fit_1d::multi_fit_gaussian(int xdim, std::vector<std::vector<double>> &zz, double &x0, std::vector<double> &a, double &sigmax, double &e)
{
#ifdef LMMIN



#else
    // std::vector<mycostfunction_gaussian1d *> cost_functions;
    ceres::Solver::Summary summary;
    ceres::Problem problem;
    sigmax = 2 * sigmax * sigmax; // important: sigma is actually 2*sigma*sigma in mycostfunction_gaussian1d

    for(int k=0;k<zz.size();k++)
    {
        mycostfunction_gaussian1d *cost_function = new mycostfunction_gaussian1d(xdim, zz[k].data());
        // cost_functions.emplace_back(cost_function);
        cost_function->set_n_residuals(zz[k].size());
        for (int m = 0; m < 3; m++)
            cost_function->parameter_block_sizes()->push_back(1);
        problem.AddResidualBlock(cost_function, NULL, &a[k], &x0, &sigmax);
    }

    ceres::Solve(options, &problem, &summary);
    e = sqrt(summary.final_cost / zz[0].size());

    sigmax = sqrt(fabs(sigmax) / 2.0); // important: sigma is actually 2*sigma*sigma in mycostfunction_gaussian1d
#endif
    return true;
};


bool gaussian_fit_1d::one_fit_lorentz(int xdim, std::vector<double> *zz, double &x0, double &a, double &gammax, double &e)
{
#ifdef LMMIN



#else
    ceres::Solver::Summary summary;
    ceres::Problem problem;

    mycostfunction_lorentz1d *cost_function = new mycostfunction_lorentz1d(xdim, zz->data());
    cost_function->set_n_residuals(zz->size());
    for (int m = 0; m < 3; m++)
        cost_function->parameter_block_sizes()->push_back(1);
    problem.AddResidualBlock(cost_function, NULL, &a, &x0, &gammax);

    ceres::Solve(options, &problem, &summary);
    e = sqrt(summary.final_cost / zz->size());

    gammax = fabs(gammax);
#endif
    return true;
};

bool gaussian_fit_1d::multi_fit_lorentz(int xdim, std::vector<std::vector<double>> &zz, double &x0, std::vector<double> &a, double &gammax, double &e)
{
#ifdef LMMIN



#else
    ceres::Solver::Summary summary;
    ceres::Problem problem;

    for(int k=0;k<zz.size();k++)
    {
        mycostfunction_lorentz1d *cost_function = new mycostfunction_lorentz1d(xdim, zz[k].data());
        cost_function->set_n_residuals(zz[k].size());
        for (int m = 0; m < 3; m++)
            cost_function->parameter_block_sizes()->push_back(1);
        problem.AddResidualBlock(cost_function, NULL, &a[k], &x0, &gammax);
    }

    ceres::Solve(options, &problem, &summary);
    e = sqrt(summary.final_cost / zz[0].size());

    gammax = fabs(gammax);
#endif
    return true;
};

bool gaussian_fit_1d::one_fit_voigt(int xdim, std::vector<double> *zz, double &x0, double &a, double &sigmax, double &gammax, double &e, int n, int n2)
{
    double scale = 0.0;
    for (int i = 0; i < zz->size(); i++)
    {
        if (zz->at(i) > scale)
        {
            scale = zz->at(i);
        }
    }
    scale *= 0.1;
    for (int i = 0; i < zz->size(); i++)
    {
        zz->at(i) /= scale;
    }
    a /= scale;

    if (n <= 3 || n2 <= 0)
    {
        double fwhh = 0.5346 * gammax * 2 + std::sqrt(0.2166 * 4 * gammax * gammax + sigmax * sigmax * 8 * 0.6931);
        double min_e = 1e20;
        double x0_in, a_in, sigmax_in, gammax_in, ee;

        double gammas[4] = {10, 1.0, 1e-5, 1e-10};
        gammas[0] = fwhh / 4.0;

        x0_in = x0;
        a_in = a;

        for (int i = 0; i < 4; i++)
        {
            gammax_in = gammas[i];
            sigmax_in = sqrt(((fwhh - 1.0692 * gammax_in) * (fwhh - 1.0692 * gammax_in) - 0.2166 * 4 * gammax_in * gammax_in) / (8 * 0.6931));
            one_fit_voigt_core(xdim, zz, x0_in, a_in, sigmax_in, gammax_in, ee);
            if (ee < min_e)
            {
                min_e = ee;
                a = a_in;
                x0 = x0_in;
                sigmax = sigmax_in;
                gammax = gammax_in;
                e = ee;
            }
        }
    }
    else
    {
        one_fit_voigt_core(xdim, zz, x0, a, sigmax, gammax, e);
    }

    a *= scale;

    return true;
};

/**
 * @brief voigt fitting for doesy type pseudo-2D fitting, where amplitude scale as exp(-D*t^2)
 * and t is usually = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14, ...}
 * @param xdim size of the spectrum
 * @param zz  pseudo-2D spectrum zz[spectral_index][spatial_index]
 * @param x0 peak position
 * @param a peak amplitude at current spectral index
 * @param sigmax 
 * @param gammax 
 * @param e fitting residual
 * @param n loop number
 * @param n2 loop number after big change. n and n2 decide when to try different initial values again.
 * @return true 
 * @return false 
 */
bool gaussian_fit_1d::multi_fit_voigt_doesy(const int xdim,std::vector<std::vector<double>> zz, double &x0, double &a0, double &d, double &sigmax, double &gammax, double &e, int n, int n2)
{
    double scale = 0.0;
    for(int j=0;j<zz.size();j++)
    {
        for (int i = 0; i < zz[j].size(); i++)
        {
            if (zz[j][i] > scale)
            {
                scale = zz[j][i];
            }
        }
    }
    scale *= 0.1;
    for(int j=0;j<zz.size();j++)
    {
        for (int i = 0; i < zz[j].size(); i++)
        {
            zz[j][i] /= scale;
        }
    }
    a0 /= scale;

    
    if (n <= 3 || n2 <= 0)
    {
        double fwhh = 0.5346 * gammax * 2 + std::sqrt(0.2166 * 4 * gammax * gammax + sigmax * sigmax * 8 * 0.6931);
        double min_e = 1e20;
        double x0_in, sigmax_in, gammax_in, ee;
        double a0_in;
        double d_in;

        double gammas[4] = {10, 1.0, 1e-5, 1e-10};
        gammas[0] = fwhh / 4.0;

        x0_in = x0;
        a0_in = a0;
        d_in = d;

        for (int i = 0; i < 4; i++)
        {
            gammax_in = gammas[i];
            sigmax_in = sqrt(((fwhh - 1.0692 * gammax_in) * (fwhh - 1.0692 * gammax_in) - 0.2166 * 4 * gammax_in * gammax_in) / (8 * 0.6931));
            multi_fit_voigt_core_doesy(xdim, zz, x0_in, a0_in, d_in, sigmax_in, gammax_in, ee);
            if (ee < min_e)
            {
                min_e = ee;
                a0 = a0_in;
                x0 = x0_in;
                sigmax = sigmax_in;
                gammax = gammax_in;
                d = d_in;
                e = ee;
            }
        }
    }
    else
    {
        multi_fit_voigt_core_doesy(xdim, zz, x0, a0, d, sigmax, gammax, e);
    }

    a0 *= scale; 

    return true;
};

/**
 * @brief voigt fitting for pseduo-2D
 * 
 * @param xdim size of the spectrum
 * @param zz  pseudo-2D spectrum zz[spectral_index][spatial_index]
 * @param x0 peak position
 * @param a peak amplitude at current spectral index
 * @param sigmax 
 * @param gammax 
 * @param e fitting residual
 * @param n loop number
 * @param n2 loop number after big change. n and n2 decide when to try different initial values again.
 * @return true 
 * @return false 
 */
bool gaussian_fit_1d::multi_fit_voigt(int xdim, std::vector<std::vector<double>> zz, double &x0, std::vector<double> &a, double &sigmax, double &gammax, double &e, int n, int n2)
{
    double scale = 0.0;
    for(int j=0;j<zz.size();j++)
    {
        for (int i = 0; i < zz[j].size(); i++)
        {
            if (zz[j][i] > scale)
            {
                scale = zz[j][i];
            }
        }
    }
    scale *= 0.1;
    for(int j=0;j<zz.size();j++)
    {
        for (int i = 0; i < zz[j].size(); i++)
        {
            zz[j][i] /= scale;
        }
        a[j] /= scale;
    }

    
    if (n <= 3 || n2 <= 0)
    {
        double fwhh = 0.5346 * gammax * 2 + std::sqrt(0.2166 * 4 * gammax * gammax + sigmax * sigmax * 8 * 0.6931);
        double min_e = 1e20;
        double x0_in, sigmax_in, gammax_in, ee;
        std::vector<double> a_in;

        double gammas[4] = {10, 1.0, 1e-5, 1e-10};
        gammas[0] = fwhh / 4.0;

        x0_in = x0;
        a_in = a;

        for (int i = 0; i < 4; i++)
        {
            gammax_in = gammas[i];
            sigmax_in = sqrt(((fwhh - 1.0692 * gammax_in) * (fwhh - 1.0692 * gammax_in) - 0.2166 * 4 * gammax_in * gammax_in) / (8 * 0.6931));
            multi_fit_voigt_core(xdim, zz, x0_in, a_in, sigmax_in, gammax_in, ee);
            if (ee < min_e)
            {
                min_e = ee;
                a = a_in;
                x0 = x0_in;
                sigmax = sigmax_in;
                gammax = gammax_in;
                e = ee;
            }
        }
    }
    else
    {
        multi_fit_voigt_core(xdim, zz, x0, a, sigmax, gammax, e);
    }

    for(int j=0;j<a.size();j++)
    {
        a[j] *= scale;
    }

    return true;
};


bool gaussian_fit_1d::one_fit_voigt_core(int xdim, std::vector<double> *zz, double &x0, double &a, double &sigmax, double &gammax, double &e)
{
#ifdef LMMIN

    double **par;
    par = new double *[4];
    for (int i = 0; i < 4; i++)
        par[i] = new double[1];
    par[0][0] = a;
    par[1][0] = x0;
    par[2][0] = sigmax;
    par[3][0] = gammax;

    mycostfunction_voigt1d  cost_function(xdim, zz->data());

    class levmarq minimizer;
    minimizer.solve(4,par,zz->size(),NULL/**weight*/,cost_function);

    a = par[0][0];
    x0 = par[1][0];
    sigmax = fabs(par[2][0]);
    gammax = fabs(par[3][0]);

    e = minimizer.error_func(par, zz->size(), NULL /**weight*/, cost_function);

    for (int i = 0; i < 4; i++)
        delete[] par[i];
    delete[] par;   

#else
    ceres::Solver::Summary summary;
    ceres::Problem problem;
    mycostfunction_voigt1d *cost_function = new mycostfunction_voigt1d(xdim, zz->data());
    cost_function->set_n_residuals(zz->size());
    for (int m = 0; m < 4; m++)
        cost_function->parameter_block_sizes()->push_back(1);
    problem.AddResidualBlock(cost_function, NULL, &a, &x0, &sigmax, &gammax);

    ceres::Solve(options, &problem, &summary);
    e = sqrt(summary.final_cost / zz->size());
    // a = fabs(a);  a is not always positive
    sigmax = fabs(sigmax);
    gammax = fabs(gammax);
#endif
    return true;
};

/**
 * @brief true voigt fitting for 1D. This is the core function, which will be called by multi_fit_voigt
 * see multi_fit_voigt for the meaning of parameters
 * @return true 
 * @return false 
 */
bool gaussian_fit_1d::multi_fit_voigt_core(int xdim, std::vector<std::vector<double>> &zz, double &x0, std::vector<double> &a, double &sigmax, double &gammax, double &e)
{
#ifdef LMMIN



#else
    ceres::Solver::Summary summary;
    ceres::Problem problem;

    for(int k=0;k<zz.size();k++)
    {
        mycostfunction_voigt1d *cost_function = new mycostfunction_voigt1d(xdim, zz[k].data());
        cost_function->set_n_residuals(zz[k].size());
        for (int m = 0; m < 4; m++)
            cost_function->parameter_block_sizes()->push_back(1);
        problem.AddResidualBlock(cost_function, NULL, &a[k], &x0, &sigmax, &gammax);
    }

    ceres::Solve(options, &problem, &summary);
    e = sqrt(summary.final_cost / zz[0].size());

    // for(int k=0;k<a.size();k++)
    //     a[k] = fabs(a[k]);

    sigmax = fabs(sigmax);
    gammax = fabs(gammax);
#endif
    return true;
};

/**
 * @brief true voigt fitting for doesy type pseudo 2D. This is the core function, which will be called by multi_fit_voigt_doesy
 * see multi_fit_voigt_doesy for the meaning of parameters
 * @return true 
 * @return false 
 */
bool gaussian_fit_1d::multi_fit_voigt_core_doesy(const int xdim,std::vector<std::vector<double>> &zz, double &x0, double &a, double &d,double &sigmax, double &gammax, double &e)
{
#ifdef LMMIN



#else
    ceres::Solver::Summary summary;
    ceres::Problem problem;

    double d_sqrt=sqrt(fabs(d));

    for(int k=0;k<zz.size();k++)
    {
        double z_gradient_squared = z_gradients[k]*z_gradients[k];
        mycostfunction_voigt1d_doesy *cost_function = new mycostfunction_voigt1d_doesy(z_gradient_squared,xdim, zz[k].data());
        cost_function->set_n_residuals(zz[k].size());
        for (int m = 0; m < 5; m++)
            cost_function->parameter_block_sizes()->push_back(1);
        problem.AddResidualBlock(cost_function, NULL, &a, &x0, &sigmax, &gammax,&d_sqrt); // d_sqrt is used to make sure d is positive
    }

    ceres::Solve(options, &problem, &summary);
    e = sqrt(summary.final_cost / zz[0].size());

    sigmax = fabs(sigmax);
    gammax = fabs(gammax);
    d = d_sqrt*d_sqrt;
#endif
    return true;
};


bool gaussian_fit_1d::gaussain_convolution(double a, double x, double sigmax, std::vector<double> *kernel, int &i0, int &i1, double scale)
{

    double wx = 2.355 * sigmax;

    i0 = std::max(0, int(x - wx * scale + 0.5));
    i1 = std::min(xdim, int(x + wx * scale + 0.5));

    if(i1-i0>xdim)
    {
        return false;
    }

    kernel->clear();
    kernel->resize((i1 - i0));

    double sigmax2 = sigmax * sigmax * 2.0;

    for (int i = i0; i < i1; i++)
    {
        double t1 = x - i;
        kernel->at(i - i0) = a * exp(-(t1 * t1) / sigmax2);
    }
    return true;
};

bool gaussian_fit_1d::gaussain_convolution_with_limit(int ndx, double a, double x, double sigmax, std::vector<double> *kernel, int &i0, int &i1, double scale)
{

    double wx = 2.355 * sigmax;

    std::array<int, 2> add_limit = valid_fit_region.at(ndx);
    i0 = std::max(std::max(0, int(x - wx * scale + 0.5)), add_limit[0]);
    i1 = std::min(std::min(xdim, int(x + wx * scale + 0.5)), add_limit[1]);

    kernel->clear();
    kernel->resize((i1 - i0));

    double sigmax2 = sigmax * sigmax * 2.0;

    for (int i = i0; i < i1; i++)
    {
        double t1 = x - i;
        kernel->at(i - i0) = a * exp(-(t1 * t1) / sigmax2);
    }
    return true;
};

bool gaussian_fit_1d::lorentz_convolution(double a, double x, double gammax, std::vector<double> *kernel, int &i0, int &i1, double scale)
{

    double wx = 2.0 * gammax;

    i0 = std::max(0, int(x - wx * scale + 0.5));
    i1 = std::min(xdim, int(x + wx * scale + 0.5));

    kernel->clear();
    kernel->resize((i1 - i0));

    for (int i = i0; i < i1; i++)
    {
        double t1 = x - i;
        kernel->at(i - i0) = a / (1 + (2 * t1 / gammax) * (2 * t1 / gammax));
    }
    return true;
};

bool gaussian_fit_1d::lorentz_convolution_with_limit(int ndx, double a, double x, double gammax, std::vector<double> *kernel, int &i0, int &i1, double scale)
{

    double wx = 2.0 * gammax;

    std::array<int, 2> add_limit = valid_fit_region.at(ndx);
    i0 = std::max(std::max(0, int(x - wx * scale + 0.5)), add_limit[0]);
    i1 = std::min(std::min(xdim, int(x + wx * scale + 0.5)), add_limit[1]);

    kernel->clear();
    kernel->resize((i1 - i0));

    for (int i = i0; i < i1; i++)
    {
        double t1 = x - i;
        kernel->at(i - i0) = a / (1 + ( t1 / gammax) * ( t1 / gammax));
    }
    return true;
};

bool gaussian_fit_1d::voigt_convolution(double a, double x, double sigmax, double gammax, std::vector<double> *kernel, int &i0, int &i1, double scale)
{
    double wx = 0.5346 * gammax * 2 + std::sqrt(0.2166 * 4 * gammax * gammax + sigmax * sigmax * 8 * 0.6931);

    i0 = std::max(0, int(x - wx * scale + 0.5));
    i1 = std::min(xdim, int(x + wx * scale + 0.5));

    if (i1 <= i0)
        return false;

    kernel->clear();
    kernel->resize(i1 - i0);

    for (int i = i0; i < i1; i++)
    {
        double z1 = voigt(i - x, sigmax, gammax);
        kernel->at(i - i0) = a * z1;
    }
    return true;
};

bool gaussian_fit_1d::voigt_convolution_with_limit(int ndx, double a, double x, double sigmax, double gammax, std::vector<double> *kernel, int &i0, int &i1, double scale)
{
    double wx = 0.5346 * gammax * 2 + std::sqrt(0.2166 * 4 * gammax * gammax + sigmax * sigmax * 8 * 0.6931);

    // wx=40.0;
    std::array<int, 2> add_limit = valid_fit_region.at(ndx);
    i0 = std::max(std::max(0, int(x - wx * scale + 0.5)), add_limit[0]);
    i1 = std::min(std::min(xdim, int(x + wx * scale + 0.5)), add_limit[1]);

    if (i1 <= i0)
        return false;

    kernel->clear();
    kernel->resize(i1 - i0);

    for (int i = i0; i < i1; i++)
    {
        double z1 = voigt(i - x, sigmax, gammax);
        kernel->at(i - i0) = a * z1;
    }
    return true;
};

spectrum_fit_1d::spectrum_fit_1d()
{
    zf = 1;
    error_nround = 0;
    n_patch = 20;
};
spectrum_fit_1d::~spectrum_fit_1d(){};

bool spectrum_fit_1d::init_all_spectra(std::vector<std::string> fnames_,bool b_negative_)
{
    b_negative = b_negative_;

    fnames = fnames_;
    int i = fnames.size() - 1;
    if (fid_1d::read_spectrum(fnames[i],b_negative))
    {
        spects.push_back(spectrum_real);
    }

    for (int i = fnames.size() - 2; i >= 0; i--)
    {
        if (fid_1d::read_spectrum(fnames[i],b_negative))
        {
            spects.push_back(spectrum_real);
        }
    }
    std::reverse(spects.begin(), spects.end());
    nspect = spects.size();
    if (nspect > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
};

bool spectrum_fit_1d::prepare_to_read_additional_spectrum_from_buffer(bool b_negative_)
{
    /**
     * This function is used to prepare the spectrum_fit_1d object to read additional spectrum from buffer
     * It will clear the previous spectrum and set the b_negative flag
     */
    b_negative = b_negative_;

    if(b_negative == false)
    {
        /**
         * Let set all negative points to 0.0 for the first spectrum
         */
        for (int i = 0; i < ndata_frq; i++)
        {
            if (spectrum_real[i] < 0.0)
            {
                spectrum_real[i] = 0.0;
            }
        }
    }
    /**
     * Push the first spectrum to spects
     * This is the first spectrum that will be read from buffer
     */
    spects.clear();
    spects.push_back(spectrum_real);
    nspect = 1;

    fnames.push_back("buffer_spectrum_0.ft1");
  
    return true;
};


bool spectrum_fit_1d::read_additonal_spectrum_from_buffer(std::vector<float> &buffer){
    /**
     * We suppose the first spectrum is already read
     * and the buffer contains the next spectrum, real part only
    */   
    if (buffer.size() != ndata_frq)
    {
        std::cout << "Buffer size is not equal to ndata_frq. Buffer size=" << buffer.size() << " ndata_frq=" << ndata_frq << std::endl;
        return false;
    }
    /**
     * If b_negative is false, we will set all negative points to 0.0
     */
    if (b_negative == false)
    {
        for (int i = 0; i < ndata_frq; i++)
        {
            if (buffer[i] < 0.0)
            {
                buffer[i] = 0.0;
            }
        }
    }
    spects.push_back(buffer);
    nspect = spects.size();

    fnames.push_back("buffer_spectrum_" + std::to_string(nspect - 1) + ".ft1");


    return nspect > 0;
}


/**
 * @brief find signal free region and signal region
 * we may have both positive and negative peaks
 * Currently. Positive peaks and negative peaks will be put into different groups because 0 crossing between them
 * The crossing point will be considered as non-signal region
 * @return true always at the moment
 */
bool spectrum_fit_1d::peak_partition_1d_for_fit(double spectrum_begin,double spectrum_end)
{
    double boundary_cutoff = noise_level * user_scale2;
    /**
     * Convert spectrum_begin and spectrum_end to index
     */
    int spectrum_begin_index = (spectrum_begin - begin1) / step1;
    int spectrum_end_index = (spectrum_end - begin1) / step1;

    std::cout<<"spectrum_begin="<<spectrum_begin<<" spectrum_end="<<spectrum_end<<std::endl;
    std::cout<<"Step 1="<<step1<<" begin1="<<begin1<<std::endl;
    std::cout<<"Cut spectrum from "<<spectrum_begin_index<<" to "<<spectrum_end_index<<std::endl;

    for (int j = std::max(1,spectrum_begin_index); j < std::min(ndata_frq,spectrum_end_index); j++)
    {
        if(j==std::max(1,spectrum_begin_index) && fabs(spectrum_real[j]) > boundary_cutoff)
        {
            signa_boudaries.push_back(std::max(j - 10, 0));
        }
        else if (fabs(spectrum_real[j - 1]) <= boundary_cutoff && fabs(spectrum_real[j]) > boundary_cutoff)
        {
            signa_boudaries.push_back(std::max(j - 10, 0));
        }
        else if (fabs(spectrum_real[j - 1]) > boundary_cutoff && fabs(spectrum_real[j]) <= boundary_cutoff)
        {
            noise_boudaries.push_back(std::min(j + 10, ndata_frq));
        }
    }
    if (noise_boudaries.size() < signa_boudaries.size())
    {
        noise_boudaries.push_back(std::min(ndata_frq,spectrum_end_index));
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
}

bool spectrum_fit_1d::set_for_one_spectrum()
{
    spects.clear();
    spects.push_back(spectrum_real);
    nspect = 1;

    return true;
}

bool spectrum_fit_1d::peak_fitting(double spectrum_begin,double spectrum_end)
{

    /**
     * If z_gradients is not set, we will turn off the doesy fitting
    */
    if(b_dosy==true && z_gradients.size()!=nspect)
    {
        std::cout<<"z_gradients.size()="<<z_gradients.size()<<" is not equal to nspect="<<nspect<<". Doesy fitting will be turned off."<<std::endl;
        b_dosy=false;
    }

    double wx;

    wx = std::max(median_width_x * 1.6, 15.0);

    peak_partition_1d_for_fit(spectrum_begin,spectrum_end);

    // fit all spectrum at once
    //  signa_boudaries.clear();
    //  noise_boudaries.clear();
    //  signa_boudaries.push_back(1820);
    //  noise_boudaries.push_back(1980);

    for (int j = 0; j < signa_boudaries.size(); j++)
    {
        int begin = signa_boudaries[j];
        int stop = noise_boudaries[j];
        int begin0 = begin;
        int stop0 = stop; // only mentioned in comment. Compiler will skip it anyway

        int ndim = spectrum_real.size();
        int left_patch = 0;
        int right_patch = 0;
        int n = stop - begin + 40;

        // collect peaks that belong to this spectru part
        std::vector<double> part_p1, part_sigmax, part_gammax, part_p_intensity;
        std::vector<std::vector<double>> part_p_intensity_all_spectra;
        std::vector<int> part_peak_index;

        left_patch = std::max(0, n_patch - begin);
        begin = std::max(0, begin - n_patch);
        right_patch = std::max(0, n_patch - (ndim - stop));
        stop = std::min(ndim, stop + n_patch);

        std::vector<std::vector<float>> part_spects;
        for (int k = 0; k < nspect; k++)
        {
            std::vector<float> data;
            data.clear();
            data.resize(n, 0.0f);
            std::copy(spects[k].begin() + begin, spects[k].begin() + stop, data.begin() + left_patch);
            part_spects.push_back(data);
        }

        for (int i = 0; i < p1.size(); i++)
        {
            if (p1[i] >= begin && p1[i] < stop)
            {
                part_p1.push_back(p1[i] - begin + left_patch);
                part_sigmax.push_back(sigmax[i]);
                part_gammax.push_back(gammax[i]);
                part_p_intensity.push_back(p_intensity[i]);
                part_p_intensity_all_spectra.push_back(p_intensity_all_spectra[i]);
                part_peak_index.push_back(i);
            }
        }

        if (part_p1.size() == 0)
            continue;
        if(n_verbose>0)
        {
            std::cout << "Working on fitting region " << j << " of "<< signa_boudaries.size() <<" ( from " << begin << " to " << stop << ", " << part_p1.size() << " peaks before fitting)." << std::endl;
        }

        gaussian_fit_1d f1;
        f1.set_up(peak_shape, rmax, to_near_cutoff, user_scale, noise_level,b_negative,median_width_x);
        f1.gaussian_fit_init(part_spects, part_p1, part_sigmax, part_gammax, part_p_intensity_all_spectra, part_peak_index);
        f1.save_postion_informations(begin, stop, left_patch, right_patch, n);
        
        if (error_nround > 0)
        {
            f1.run_with_error_estimation(zf, error_nround);
        }
        else
        {
            f1.run_peak_fitting(true);
        }

        if(n_verbose>0)
        {
            std::cout << "Finished fitting region " << j << " of "<< signa_boudaries.size() <<" ( from " << begin << " to " << stop << ", " << f1.x.size() << " peaks after fitting)." << std::endl;
        }

        if (f1.a.size() > 0)
        {
            fits.emplace_back(f1);
        }
    }

    return true;
};

bool spectrum_fit_1d::gather_result()
{
    fit_p_intensity_all_spectra.clear();
    fit_p_intensity.clear();
    fit_p1.clear();
    fit_p1_ppm.clear();
    fit_sigmax.clear();
    fit_gammax.clear();
    fit_err.clear();
    fit_num_sum.clear();
    fit_nround.clear();
    fit_peak_index.clear();

    for (int i = 0; i < fits.size(); i++)
    {
        // gather result
        for (unsigned int ii = 0; ii < fits[i].x.size(); ii++)
        {
            fit_p1.push_back(fits[i].x.at(ii) + fits[i].begin - fits[i].left_patch);
            fit_nround.push_back(fits[i].get_nround());
        }

        for(int ii=0;ii<fits[i].a.size();ii++)
        {
            fit_p_intensity.push_back(fits[i].a.at(ii).at(0)); //fit_p_intensity keep the intensity of the first spectrum
        }

        fit_p_intensity_all_spectra.insert(fit_p_intensity_all_spectra.end(), fits[i].a.begin(), fits[i].a.end());
        fit_sigmax.insert(fit_sigmax.end(), fits[i].sigmax.begin(), fits[i].sigmax.end());
        fit_peak_index.insert(fit_peak_index.end(), fits[i].original_ndx.begin(), fits[i].original_ndx.end());
        fit_err.insert(fit_err.end(), fits[i].err.begin(), fits[i].err.end());
        fit_num_sum.insert(fit_num_sum.end(), fits[i].num_sum.begin(), fits[i].num_sum.end());
        fit_gammax.insert(fit_gammax.end(), fits[i].gammax.begin(), fits[i].gammax.end());

        // gather error estimation result
    }

    // point to ppm for fitted peaks
    if(n_verbose>0)
    {
        std::cout << "Total fitted " << fit_p1.size() << " peaks." << std::endl;
    }

    return true;
};

bool spectrum_fit_1d::gather_result_with_error_estimation(int c)
{
    // in error fitting step, fit_peak_index won't change
    // skip fit_err,fit_num_sum and fit_nround too. They will change, but we won't use them in the output. WE still print old values.
    fit_p_intensity.clear();
    fit_p_intensity_all_spectra.clear();
    fit_p1.clear();
    fit_sigmax.clear();
    fit_gammax.clear();
    // fit_err.clear();
    // fit_num_sum.clear();
    // fit_nround.clear();

    for (int i = 0; i < fits.size(); i++)
    {
        // gather result
        for (unsigned int ii = 0; ii < fits[i].x.size(); ii++)
        {
            fit_p1.push_back(fits[i].batch_x.at(c).at(ii) + fits[i].begin - fits[i].left_patch);
        }

        for(int ii=0;ii<fits[i].batch_a.at(c).size();ii++)
        {
            fit_p_intensity.push_back(fits[i].batch_a.at(c).at(ii).at(0)); //fit_p_intensity keep the intensity of the first spectrum
        }

        fit_p_intensity_all_spectra.insert(fit_p_intensity_all_spectra.end(), fits[i].batch_a.at(c).begin(), fits[i].batch_a.at(c).end());
        fit_sigmax.insert(fit_sigmax.end(), fits[i].batch_sigmax.at(c).begin(), fits[i].batch_sigmax.at(c).end());
        fit_gammax.insert(fit_gammax.end(), fits[i].batch_gammax.at(c).begin(), fits[i].batch_gammax.at(c).end());
    }
    return true;
};

/**
 * save fitting results to a varible of type spectrum_1d_peaks
 * to be used by other programs without using output function (file IO)
*/
bool spectrum_fit_1d::get_fitted_peaks(spectrum_1d_peaks &peaks)
{
    /**
     * Gather all fitting results from all regions
    */
    gather_result();

    /**
     * Get peak amplitudes (maximal intensity of the peak)
     * This depends on peak shape (See function output for more details)
    */
    std::vector<double> amplitudes;
    std::vector<double> fitted_volume;
    if (peak_shape == gaussian_type)
    {
        amplitudes = fit_p_intensity;
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            fitted_volume.push_back(fit_p_intensity[i] * sqrt(fabs(fit_sigmax[i]) * 3.14159265358979));
        }
    }
    else if (peak_shape == voigt_type)
    {
        fitted_volume = fit_p_intensity;
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            amplitudes.push_back(fit_p_intensity[i] * voigt(0.0, fit_sigmax[i], fit_gammax[i]));
        }
    }
    else if (peak_shape == lorentz_type)
    {
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            fitted_volume.push_back(fit_p_intensity[i] * fabs(fit_gammax[i]) * 3.14159265358979);
        }
        amplitudes = fit_p_intensity;
    }

    // get ppm from peak postion
    std::vector<double> ppm;
    for (unsigned int i = 0; i < fit_p1.size(); i++)
    {
        ppm.push_back(begin1 + step1 * (fit_p1[i]));
    }

    std::vector<int> ndx;

    if(b_remove_failed_peaks == false)
    {
        /**
         * Do not changed the order of peaks, just use the original index
         * if we do not remove failed peaks, we will use the original index
         */
        for (unsigned int i = 0; i < ppm.size(); i++)
        {
            ndx.push_back(i);
        }
    }
    else
    {
        ldw_math_1d::sortArr(ppm, ndx);
        std::reverse(ndx.begin(), ndx.end()); // flip ndx so that the ppm is in descending order
    }
    
    for (int i = 0; i < ndx.size(); i++)
    {
        peaks.x.push_back(fit_p1[ndx[i]]);
        peaks.ppm.push_back(ppm[ndx[i]]);
        peaks.sigmax.push_back(fit_sigmax[ndx[i]]);
        peaks.gammax.push_back(fit_gammax[ndx[i]]);
        peaks.a.push_back(amplitudes[ndx[i]]);
        peaks.volume.push_back(fitted_volume[ndx[i]]);
    }

    peaks.confidence.clear(); //in case we have old data, we clear it first
    peaks.confidence=std::vector<double>(fit_p1.size(),1.0); // no confidence information from fitting

    return true;
};

/**
 * @brief Try to identify baseline peaks and label them as un-reliable peaks
 * Must be called after gather_result() or gather_result_with_error_estimation()
*/
bool spectrum_fit_1d::label_baseline_peaks()
{
     /**
     * Get median experimental peak width. 
     * unit is point
    */
    std::vector<double> fwhhs;
    for(int i=0;i<fit_sigmax.size();i++)
    {
        fwhhs.push_back(0.5346 * fit_gammax[i] * 2 + std::sqrt(0.2166 * 4 * fit_gammax[i] * fit_gammax[i] + fit_sigmax[i] * fit_sigmax[i] * 8 * 0.6931));
    }
    /**
     * Median experimental peak width, in unit of point
    */
    double median_exp_peak_width = ldw_math_1d::calcualte_median(fwhhs);

    /**
     * Get peak amplitudes (maximal intensity of the peak) and spectrum amplitudes (sum of all peaks) at the exact peak location
    */
    std::vector<double> amplitudes;
    std::vector<double> spectrum_amplitudes;
    
    if (peak_shape == gaussian_type || peak_shape == lorentz_type)
    {
        amplitudes = fit_p_intensity;
    }
    else // (peak_shape == voigt_type)
    { 
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            amplitudes.push_back(fit_p_intensity[i] * voigt(0.0, fit_sigmax[i], fit_gammax[i]));
        }
    }

    for(int i=0;i<fit_p1.size();i++)
    {
        int loc=int(round(fit_p1[i]));
        spectrum_amplitudes.push_back(spectrum_real[loc]);
    }
    

    /**
     * Define un-reliable peaks, which will be excluded from subsequent database matching.
    */
    background_peak_flag.clear();
    background_peak_flag.resize(fit_p1.size(),0); // 1: background peak, 0: not background peak


    /**
     * Sort peaks by point in ascending order (ppm in descending order)
    */
    std::vector<int> ndx;
    ldw_math_1d::sortArr(fit_p1, ndx);


    /**
     * Step 1. Define connected peaks (neighboring peaks fullfill: 1,spectrum  amplitude ratio < 1.2; 2, ppm difference < 2 * fwhh; 3. no big dip between them)
     * We use ndx to sort peaks by ppm.
     * Connected peaks are also defined in the order of ndx.
     * In connected_peaks, -1 means not connected and 0 means connected to previous peak
    */
    std::vector<int> connected_peaks(fit_p1.size(), -1); 
    for (int i0 = 0; i0 < ndx.size() -1 ; i0++)
    {
        int j0 = i0 + 1;

        /**
         * i and j are the original index of peaks
        */
        int i = ndx[i0];
        int j = ndx[j0];

        if ( fabs(fit_p1[j] - fit_p1[i]) < 2 * median_exp_peak_width ) 
        {
            double intensity_ratio = spectrum_amplitudes[i] / spectrum_amplitudes[j];

            if(intensity_ratio<1.0)
            {
                intensity_ratio = 1.0/intensity_ratio;
            }

            if (intensity_ratio < 1.2)
            {
                /**
                 * Get the lowest spect point from fit_p1[i] to fit_p1[j]
                */
                float lowest_spect_point = std::numeric_limits<float>::max();
                for (int k = int(fit_p1[i]); k <= int(fit_p1[j]); k++)
                {
                    lowest_spect_point = std::min(lowest_spect_point, spectrum_real[k]);
                }

                /**
                 * If lowest_spect_point is >= either of 0.9 * spectrum_amplitudes[i] or 0.9*spectrum_amplitudes[i], then we consider it as a connected peak
                 * i and j are connected
                */
                if (lowest_spect_point >= 0.8 * spectrum_amplitudes[i] || lowest_spect_point >= 0.8 * spectrum_amplitudes[j])
                {
                    connected_peaks[j0] = 0;
                }
            }
        }   
    }

   /**
    * STep 2. If 4 or more peaks are connected, then they are all considered as non-reliable peaks.
    * i0 and j0 are the index of peaks in ndx
    * j is the original index of peaks
   */
    int n_connected = 0;
    for (int i0 = 0; i0 < ndx.size(); i0++)
    {
        if (connected_peaks[i0] == 0)
        {
            n_connected++;
        }
        else
        {
            if (n_connected >= 4)
            {
                for (int j0 = i0 - n_connected; j0 < i0; j0++)
                {
                    int j=ndx[j0];
                    background_peak_flag[j] = 1;
                }
            }
            n_connected = 0;
        }
    }

    return true;
}

/**
 * @brief Output fitting results to a text file
 * @param outfname output file name
 */
bool spectrum_fit_1d::output(std::string outfname)
{
    FILE *fp;
    for (int c = -1; c < error_nround; c++)
    {
        if (c == -1)
        {
            fp = fopen(outfname.c_str(), "w");
        }
        else
        {
            std::string thead = "err_";
            std::string tname = std::to_string(c);
            std::string tname2 = std::string(3 - std::min(3, int(tname.length())), '0') + tname;
            fp = fopen((thead + tname2 + outfname).c_str(), "w");
        }
        std::string output_string = output_as_string(c);
        if (fp == NULL)
        {
            std::cout << "Error opening file " << outfname << std::endl;
            return false;
        }
        fprintf(fp, "%s", output_string.c_str());
        fclose(fp);
    }
    return true;
}

std::string spectrum_fit_1d::output_as_string(int c = -1)
{
    std::string outstr;
    std::vector<int> ndx;

    if (c == -1)
    {
        gather_result();
    }
    else
    {
        gather_result_with_error_estimation(c);
    }

    label_baseline_peaks();

    std::vector<double> amplitudes, fitted_volume;
    std::vector<std::vector<double>> amplitudes_all_spectra;

    if (peak_shape == gaussian_type)
    {
        amplitudes = fit_p_intensity;
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            fitted_volume.push_back(fit_p_intensity[i] * sqrt(fabs(fit_sigmax[i]) * 3.14159265358979));
        }
    }
    else if (peak_shape == voigt_type)
    {
        fitted_volume = fit_p_intensity;
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            amplitudes.push_back(fit_p_intensity[i] * voigt(0.0, fit_sigmax[i], fit_gammax[i]));
        }
    }
    else if (peak_shape == lorentz_type)
    {
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            fitted_volume.push_back(fit_p_intensity[i] * fabs(fit_gammax[i]) * 3.14159265358979);
        }
        amplitudes = fit_p_intensity;
    }

    if (spects.size() > 1)
    {
        // rescale amplitudes_all_spectra using first spectrum. The final values are relative to the first spectrum.
        // this doesn't depend on peak_shape because all peak parameters are the same for all spectra, except for amplitudes
        amplitudes_all_spectra = fit_p_intensity_all_spectra;
        for (int i = 0; i < amplitudes_all_spectra.size(); i++)
        {
            for (int k = 1; k < amplitudes_all_spectra[i].size(); k++)
            {
                /**
                 * Add a small number to avoid division by zero
                 */
                amplitudes_all_spectra[i][k] /= (amplitudes_all_spectra[i][0] + std::numeric_limits<double>::epsilon());
            }
            amplitudes_all_spectra[i][0] = 1.0;
        }
    }

    // get ppm from peak postion
    fit_p1_ppm.clear();
    for (unsigned int i = 0; i < fit_p1.size(); i++)
    {
        fit_p1_ppm.push_back(begin1 + step1 * (fit_p1[i]));
    }
    if (c == -1)
    {
        ldw_math_1d::sortArr(fit_p1_ppm, ndx);
        // flip ndx so that the ppm is in descending order
        std::reverse(ndx.begin(), ndx.end());
    }

    char buffer[1024];

    sprintf(buffer, "VARS INDEX X_AXIS X_PPM XW HEIGHT DHEIGHT ASS INTEGRAL VOL SIMGAX GAMMAX CONFIDENCE NROUND BACKGROUND");
    outstr += buffer;
    if (spects.size() > 1)
    {
        for (int i = 0; i < spects.size(); i++)
        {
            sprintf(buffer, " Z_A%d", i);
            outstr += buffer;
        }
    }
    sprintf(buffer, "\n");
    outstr += buffer;

    sprintf(buffer, "FORMAT %%5d %%9.4f %%8.4f %%7.3f %%+e %%+e %%s %%+e %%+e %%f %%f %%f %%4d %%1d");
    outstr += buffer;
    if (spects.size() > 1)
    {
        for (int i = 0; i < spects.size(); i++)
        {
            sprintf(buffer, " %%7.4f");
            outstr += buffer;
        }
    }
    sprintf(buffer, "\n");
    outstr += buffer;

    for (unsigned int ii = 0; ii < ndx.size(); ii++)
    {
        int i = ndx[ii];
        float s1 = 0.5346 * fit_gammax[i] * 2 + std::sqrt(0.2166 * 4 * fit_gammax[i] * fit_gammax[i] + fit_sigmax[i] * fit_sigmax[i] * 8 * 0.6931);
        sprintf(buffer, "%5d %9.4f %8.4f %7.3f %+e %+e %s %+e %+e %f %f %f %4d %1d",
                fit_peak_index[i], fit_p1[i] + 1, fit_p1_ppm[i], s1, amplitudes[i], fit_err[i],
                user_comments[fit_peak_index[i]].c_str(), fit_num_sum[i][0], fitted_volume[i],
                fit_sigmax[i], fit_gammax[i], confident_level[fit_peak_index[i]], fit_nround[i], background_peak_flag[i]);
        outstr += buffer;
        if (spects.size() > 1)
        {
            for (int j = 0; j < spects.size(); j++)
            {
                sprintf(buffer, " %7.4f", amplitudes_all_spectra[i][j]);
                outstr += buffer;
            }
        }
        sprintf(buffer, "\n");
        outstr += buffer;
    }

    return outstr;
};

/**
 * @brief output fitting result to a json file. This mainly for the web server
 * @param outfname output file name of json file.
 * @param b_individual_peaks if true, output individual peaks reconstrution
 */
bool spectrum_fit_1d::output_json(std::string outfname,bool b_individual_peaks)
{
    std::string json_str = output_json_as_string(b_individual_peaks);
    std::ofstream ofs(outfname);
    if (!ofs.is_open())
    {
        std::cerr << "Error: Cannot open file " << outfname << " for writing." << std::endl;
        return false;
    }
    ofs << json_str;
    ofs.close();
    return true;
}


/**
 * @brief output fitting result to a json string. This mainly for the web server
 * @param b_individual_peaks if true, output individual peaks reconstrution
 * @return json string
 */
std::string spectrum_fit_1d::output_json_as_string(bool b_individual_peaks)
{
    std::vector<int> ndx;
    gather_result();
    label_baseline_peaks();

    fit_p1_ppm.clear();
    for (unsigned int i = 0; i < fit_p1.size(); i++)
    {
        fit_p1_ppm.push_back(begin1 + step1 * (fit_p1[i]));
    }
    ldw_math_1d::sortArr(fit_p1_ppm, ndx);
    std::reverse(ndx.begin(), ndx.end());

    std::vector<double> amplitudes, fitted_volume;
    std::vector<std::vector<double>> amplitudes_all_spectra;

    if (peak_shape == gaussian_type)
    {
        amplitudes = fit_p_intensity;
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            fitted_volume.push_back(fit_p_intensity[i] * sqrt(fabs(fit_sigmax[i]) * 3.14159265358979));
        }
    }
    else if (peak_shape == voigt_type)
    {
        fitted_volume = fit_p_intensity;
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            amplitudes.push_back(fit_p_intensity[i] * voigt(0.0, fit_sigmax[i], fit_gammax[i]));
        }
    }
    else if (peak_shape == lorentz_type)
    {
        for (int i = 0; i < fit_p_intensity.size(); i++)
        {
            fitted_volume.push_back(fit_p_intensity[i] * fabs(fit_gammax[i]) * 3.14159265358979);
        }
        amplitudes = fit_p_intensity;
    }

    if (spects.size() > 1)
    {
        // rescale amplitudes_all_spectra using first spectrum. The final values are relative to the first spectrum.
        // this doesn't depend on peak_shape because all peak parameters are the same for all spectra, except for amplitudes
        amplitudes_all_spectra = fit_p_intensity_all_spectra;
        for (int i = 0; i < amplitudes_all_spectra.size(); i++)
        {
            for (int k = 1; k < amplitudes_all_spectra[i].size(); k++)
            {
                /**
                 * Add a small number to avoid division by zero
                 */
                amplitudes_all_spectra[i][k] /= (amplitudes_all_spectra[i][0] + std::numeric_limits<double>::epsilon());
            }
            amplitudes_all_spectra[i][0] = 1.0;
        }
    }

        Json::Value root, peaks, peak_params;
    std::vector<double> spe_recon;
    std::vector<double> spe_recon_ppm;

    for (int j = 0; j < ndata_frq; j++)
    {
        spe_recon_ppm.push_back(begin1 + j * step1);
    }

    spe_recon.clear();
    spe_recon.resize(ndata_frq, 0.0);

    for (int i = 0; i < fit_p1.size(); i++)
    {
        std::vector<double> data;
        int i0, i1;
        if (peak_shape == voigt_type)
        {
            ldw_math_1d::voigt_convolution(fit_p_intensity[i], fit_p1[i], fit_sigmax[i], fit_gammax[i], &data, i0, i1, spectrum_real.size(), CONVOLUTION_RANGE);
        }
        else if (peak_shape == lorentz_type)
        {
            ldw_math_1d::lorentz_convolution(fit_p_intensity[i], fit_p1[i], fit_gammax[i], &data, i0, i1, spectrum_real.size(), CONVOLUTION_RANGE);
        }
        else if (peak_shape == gaussian_type)
        {
            ldw_math_1d::gaussian_convolution(fit_p_intensity[i], fit_p1[i], fit_sigmax[i], &data, i0, i1, spectrum_real.size(), CONVOLUTION_RANGE);
        }
        // add data to spe_recon[i0:i1]
        for (int j = i0; j < i1; j++)
        {
            spe_recon[j] += data[j - i0];
        }
    }
    Json::Value data;
    for (int j = 0; j < ndata_frq; j++)
    {
        data[j][0] = spe_recon_ppm[j];
        data[j][1] = spe_recon[j];
    }
    root["spectrum_recon"] = data;

    if(b_individual_peaks==true)
    {
        for (int m = 0; m < fit_p1.size(); m++)
        {
            int i=ndx[m];
            std::vector<double> data;
            int i0, i1;
            if (peak_shape == voigt_type)
            {
                ldw_math_1d::voigt_convolution(fit_p_intensity[i], fit_p1[i], fit_sigmax[i], fit_gammax[i], &data, i0, i1, spectrum_real.size(), 3.0);
            }
            else if (peak_shape == lorentz_type)
            {
                ldw_math_1d::lorentz_convolution(fit_p_intensity[i], fit_p1[i], fit_gammax[i], &data, i0, i1, spectrum_real.size(), 3.0);
            }
            else if (peak_shape == gaussian_type)
            {
                ldw_math_1d::gaussian_convolution(fit_p_intensity[i], fit_p1[i], fit_sigmax[i], &data, i0, i1, spectrum_real.size(), 3.0);
            }

            for (int ii = i0; ii < i1; ii++)
            {
                peaks[m][ii - i0][0] = spe_recon_ppm[ii];
                peaks[m][ii - i0][1] = data[ii - i0];
            }
        }
        root["peaks_recon"] = peaks;
    }

    //add peak parameters to json root
    for (int ii = 0; ii < fit_p1.size(); ii++)
    {
        int i = ndx[ii];
        peak_params[ii]["intensity"] = amplitudes[i];
        peak_params[ii]["position"] = fit_p1_ppm[i]; //use ppm instead of points (fit_p1)
        peak_params[ii]["sigma"] = fit_sigmax[i]*fabs(step1); //convert from points to ppm
        peak_params[ii]["gamma"] = fit_gammax[i]*fabs(step1); //convert from points to ppm. step is negative
        peak_params[ii]["background"] = background_peak_flag[i]; // 1: background peak, 0: not background peak
    }
    root["peak_params"] = peak_params;

    std::ostringstream oss;
    oss << root;
    return oss.str();
}


int spectrum_fit_1d::get_size_of_recon()
{
    return spectrum_real.size();
}

/**
 * Write the first reconstructed spectrum to memory.
 * This is used for web assembly to return the reconstructed spectrum.
 * @param n number of index of the spectrum to write. (0 based). We will have only one spectrum unless this is a pseudo-2D spectrum fit
 */
uintptr_t spectrum_fit_1d::get_data_of_recon(int file_ndx)
{
    std::vector<int> ndx;
    gather_result();

    std::vector<double> intens;
    for (int i = 0; i < fit_p_intensity_all_spectra.size(); i++)
    {
        intens.push_back(fit_p_intensity_all_spectra[i][file_ndx]);
    }
    spe_recon.clear();
    spe_recon.resize(spectrum_real.size(), 0.0f);
    for (int i = 0; i < fit_p1.size(); i++)
    {
        std::vector<double> data;
        int i0, i1;
        if (peak_shape == voigt_type)
        {
            ldw_math_1d::voigt_convolution(intens[i], fit_p1[i], fit_sigmax[i], fit_gammax[i], &data, i0, i1, spectrum_real.size(), CONVOLUTION_RANGE);
        }
        else if (peak_shape == lorentz_type)
        {
            ldw_math_1d::lorentz_convolution(intens[i], fit_p1[i], fit_gammax[i], &data, i0, i1, spectrum_real.size(), CONVOLUTION_RANGE);
        }
        else if (peak_shape == gaussian_type)
        {
            ldw_math_1d::gaussian_convolution(intens[i], fit_p1[i], fit_sigmax[i], &data, i0, i1, spectrum_real.size(), CONVOLUTION_RANGE);
        }
        // add data to spe_recon[i0:i1]
        for (int j = i0; j < i1; j++)
        {
            spe_recon[j] += data[j - i0];
        }
    }

    return  reinterpret_cast<uintptr_t>(spe_recon.data());
}



/**
 * Write reconstructed spectrum to a folder. Add "_recon" to the file name.
 * Also write the difference spectrum (recon - original) to a file with "_diff" added to the file name.
 * And add gaussian_, voigt_ or lorentz_ to the file name depending on the peak shape, at the beginning of the file name.
 * @param folder_name folder to save the reconstructed spectrum
 */
bool spectrum_fit_1d::write_recon(std::string folder_name)
{
    std::vector<int> ndx;
    gather_result();

    for(int file_ndx=0;file_ndx<fnames.size();file_ndx++)
    {
        std::string path_name, file_name, file_name_ext;
        ldw_math_spectrum_1d::SplitFilename(fnames[file_ndx], path_name, file_name, file_name_ext);

        if (file_name_ext != "ft1")
        {
            std::cout << "Can't write recon spectrum in ft1 format: input file is not .ft1 file." << std::endl;
            continue;
        }

        if (peak_shape == gaussian_type)
        {
            file_name = "gaussian_" + file_name;
        }
        else if (peak_shape == voigt_type)
        {
            file_name = "voigt_" + file_name;
        }
        else // lorentz
        {
            file_name = "lorentz_" + file_name;
        }

        std::vector<double> intens;
        for (int i = 0; i < fit_p_intensity_all_spectra.size(); i++)
        {
            intens.push_back(fit_p_intensity_all_spectra[i][file_ndx]);
        }

        std::vector<float> spe_recon(spectrum_real.size(), 0.0f);
        for (int i = 0; i < fit_p1.size(); i++)
        {
            std::vector<double> data;
            int i0, i1;
            if (peak_shape == voigt_type)
            {
                ldw_math_1d::voigt_convolution(intens[i], fit_p1[i], fit_sigmax[i], fit_gammax[i], &data, i0, i1, spectrum_real.size(), CONVOLUTION_RANGE);
            }
            else if (peak_shape == lorentz_type)
            {
                ldw_math_1d::lorentz_convolution(intens[i], fit_p1[i], fit_gammax[i], &data, i0, i1, spectrum_real.size(), CONVOLUTION_RANGE);
            }
            else if (peak_shape == gaussian_type)
            {
                ldw_math_1d::gaussian_convolution(intens[i], fit_p1[i], fit_sigmax[i], &data, i0, i1, spectrum_real.size(), CONVOLUTION_RANGE);
            }
            // add data to spe_recon[i0:i1]
            for (int j = i0; j < i1; j++)
            {
                spe_recon[j] += data[j - i0];
            }
        }

        FILE *fp = fopen(( folder_name + '/' + file_name + "_recon.ft1").c_str(), "w");

        if(fp!=NULL)
        {
            // write header saved from input file
            fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);
            // write recon spectrum
            fwrite(spe_recon.data(), sizeof(float), spe_recon.size(), fp);
            fclose(fp);
        }
        else
        {
            std::cerr<<"Can't write recon spectrum in ft1 format: can't open file "<< folder_name + '/' + file_name + "_recon.ft1"<<std::endl;
        }

        // write diff spectrum
        FILE *fp2 = fopen((folder_name + '/' + file_name + "_diff.ft1").c_str(), "w");
        if(fp2!=NULL)
        {
            fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);
            for(int i=0;i<spectrum_real.size();i++)
            {
                spe_recon[i] = spe_recon[i] -spectrum_real[i] ;
            }
            fwrite(spe_recon.data(), sizeof(float), spe_recon.size(), fp);
            fclose(fp);
        }
    }

    return true;
}


bool spectrum_fit_1d::init_fit(int t, int r, double c)
{
    if (t == 1)
        peak_shape = gaussian_type;
    else if (t == 3)
        peak_shape = lorentz_type;
    else
        peak_shape = voigt_type;

    rmax = r;
    to_near_cutoff = c;

    return true;
}

bool spectrum_fit_1d::init_error(int m, int n)
{
    zf = m;
    error_nround = n;
    return true;
}

// for check
bool spectrum_fit_1d::assess_size()
{
    bool b = true;

    if (fit_p_intensity.size() != fit_p1.size() || fit_p_intensity.size() != fit_sigmax.size()
        || fit_p_intensity.size() != fit_num_sum.size() || fit_p_intensity.size() != fit_err.size()
        || fit_p_intensity.size() != fit_gammax.size()
        || fit_p_intensity.size() != fit_p_intensity_all_spectra.size() )
    {
        std::cout << "ERROR:  vector size is not consistent in spectrum_fit." << std::endl;
        std::cout << "size of p_intensity is " << fit_p_intensity.size() << std::endl;
        std::cout << "size of p1 is " << fit_p1.size() << std::endl;
        std::cout << "size of simgax is " << fit_sigmax.size() << std::endl;
        std::cout << "size of num_sum is " << fit_num_sum.size() << std::endl;
        std::cout << "size of err is " << fit_err.size() << std::endl;

        b = false;
    }
    return b;
}



/**
 * @brief read in peaks
 * 
 * @param infname filename of peak list
 * @return true 
 * @return false 
 */
bool spectrum_fit_1d::peak_reading(std::string infname)
{

    std::ifstream file(infname);
    std::stringstream buffer;
    buffer << file.rdbuf();  // reads entire file including newlines
    std::string content = buffer.str();


    std::string stab(".tab");
    std::string slist(".list");
    std::string sjson(".json");

    if (std::equal(stab.rbegin(), stab.rend(), infname.rbegin()))
    {
        return peak_reading_from_string(content, 0);
    }
    else if (std::equal(slist.rbegin(), slist.rend(), infname.rbegin()))
    {
        return peak_reading_from_string(content, 1);
    }
    else if (std::equal(sjson.rbegin(), sjson.rend(), infname.rbegin()))
    {
        return peak_reading_from_string(content, 2);
    }
    else
    {
        std::cout << "ERROR: unknown peak list file format. Skip peaks reading." << std::endl;
        return false;
    }
    return false; // We can not reach here, but just to avoid warning.
}


bool spectrum_fit_1d::peak_reading_from_string(const std::string &content, int read_type)
{
    bool b_read = false;
    if( read_type == 0)
    {
        b_read = peak_reading_pipe(content);
    }
    else if (read_type == 1)
    {
        b_read = peak_reading_sparky(content);
    }
    else if (read_type == 2)
    {
        b_read = peak_reading_json(content);
    }

    // set gamma if it is not readed in.
    gammax.resize(p1.size(), 1e-20);

    /**
     * @brief remove out of range peaks and negative peaks (only if b_negative is flase)
     * 
     */
    for (int i = p_intensity.size() - 1; i >= 0; i--)
    {
        if ((p1[i] < 1 || p1[i] > ndata_frq - 2) || (b_negative == false && p_intensity[i] < 0.0))
        {
            p1.erase(p1.begin() + i);
            p1_ppm.erase(p1_ppm.begin() + i);
            p_intensity.erase(p_intensity.begin() + i);
            sigmax.erase(sigmax.begin() + i);
            gammax.erase(gammax.begin() + i);
            user_comments.erase(user_comments.begin() + i);
        }
    }
    if(b_negative==true)
    {
        std::cout << "Remove out of bound peaks done." << std::endl;
    }
    else
    {
        std::cout << "Remove out of bound peaks and negative peaks done." << std::endl;
    }

    // fill p_intensity_all_spectra using spectra data if needed.
    for (int i = 0; i < p1.size(); i++)
    {
        std::vector<double> temp;
        temp.push_back(p_intensity[i]); // intensity of first spec, either from peak list or from spects[0]
        for (int n = 1; n < spects.size(); n++)
        {
            int n1 = int(p1[i] + 0.5) - 1; // -1 because start at 0 in this program but start from 1 in pipe's tab file.
            if (n1 < 0)
                n1 = 0;
            if (n1 > ndata_frq - 1)
                n1 = ndata_frq - 1;
            temp.push_back(spects[n][n1]); // n1 is direct dimension location
        }
        p_intensity_all_spectra.push_back(temp);
    }

    // required for fitting stage because we need to set up wx, which are important
    // AFter peak reading, sigma is real sigma for both GAussian and voigt
    std::vector<double> sx;
    sx.clear();
    for (unsigned int i = 0; i < p1.size(); i++)
    {
        float s1 = 0.5346 * gammax[i] * 2 + std::sqrt(0.2166 * 4 * gammax[i] * gammax[i] + sigmax[i] * sigmax[i] * 8 * 0.6931);
        sx.push_back(s1); // sx is fwhh
    }
    median_width_x = ldw_math_1d::calcualte_median(sx);
    std::cout << "Median peak width is estimated to be " << median_width_x << " points from picking." << std::endl;
    if (median_width_x < 5.0)
    {
        median_width_x = 5.0;
        std::cout << "Set median peak width along x to 5.0" << std::endl;
    }

    // print infor and finish
    std::cout << "loaded in " << p1.size() << " peaks." << std::endl;
    if (p1.size() == 0)
        b_read = false;
    return b_read;
};

bool spectrum_fit_1d::peak_reading_sparky(const std::string &content)
{
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;

    int xpos = -1;
    int ypos = -1;
    int ass = -1;

    std::istringstream fin(content);
    getline(fin, line);
    iss.str(line);
    while (iss >> p)
    {
        ps.push_back(p);
    }

    for (int i = 0; i < ps.size(); i++)
    {
        if (ps[i] == "w2")
        {
            xpos = i;
        } // in sparky, w2 is direct dimension
        else if (ps[i] == "Assignment")
        {
            ass = i;
        }
    }

    if (xpos == -1)
    {
        std::cout << "Required varible xpos is missing." << std::endl;
        return false;
    }

    int c = 0;
    while (getline(fin, line))
    {
        iss.clear();
        iss.str(line);
        ps.clear();
        while (iss >> p)
        {
            ps.push_back(p);
        }

        if (ps.size() < 3)
            continue; // empty line??

        c++;
        p1_ppm.push_back(atof(ps[xpos].c_str()));
        p_intensity.push_back(0.0);
        sigmax.push_back(3.0);

        if (ass != -1)
        {
            user_comments.push_back(ps[ass]);
        }
        else
        {
            user_comments.push_back("peaks" + std::to_string(c));
        }
    }

    // get points from ppm
    p1.clear();

    for (unsigned int i = 0; i < p1_ppm.size(); i++)
    {
        p1.push_back((p1_ppm[i] - begin1) / step1); // direct dimension
    }

    for (int i = 0; i < p1.size(); i++)
    {
        int n1 = int(p1[i] + 0.5) - 1; // -1 because start at 0 in this program but start from 1 in pipe's tab file.
        if (n1 < 0)
            n1 = 0;
        if (n1 > ndata_frq - 1)
            n1 = ndata_frq - 1;
        p_intensity[i] = spectrum_real[n1];
    }
    return true;
}

bool spectrum_fit_1d::peak_reading_pipe(const std::string &content)
{
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;

    int index = -1;
    int xpos = -1;
    int xpos_ppm = -1;
    int xw = -1;
    int height = -1;
    int ass = -1;
    int confidence = -1;
    bool b_format = false;
    user_comments.clear();
    int c=0;

    std::istringstream fin(content);

    while(getline(fin,line))
    {
        /**
         * Skit REMARK,DATA and FORMAT line or empty line (less or equal than 4 charactors)
        */
        if(line.find("REMARK")==0 || line.length()<=4 ) continue;
        if(line.find("DATA")==0) continue;
        if(line.find("FORMAT")==0) continue;
    
        /**
         * VARS line is required
        */
        if(line.find("VARS")==0)
        {
    
            iss.str(line);
            while (iss >> p)
            {
                ps.push_back(p);
            }
            ps.erase(ps.begin()); // remove first words (VARS)
            for (int i = 0; i < ps.size(); i++)
            {
                if (ps[i] == "INDEX")
                {
                    index = i;
                }
                else if (ps[i] == "X_AXIS")
                {
                    xpos = i;
                }
                else if (ps[i] == "X_PPM")
                {
                    xpos_ppm = i;
                }
                else if (ps[i] == "XW")
                {
                    xw = i;
                }
                else if (ps[i] == "HEIGHT")
                {
                    height = i;
                }
                else if (ps[i] == "ASS")
                {
                    ass = i;
                }
                else if (ps[i] == "CONFIDENCE")
                {
                    confidence = i;
                }
            }
            
            b_format = true;
            if (xpos == -1 && xpos_ppm == -1)
            {
                std::cout << "One or more required varibles are missing." << std::endl;
                b_format = false;
            }
            continue;
        }//enf of if(line.find("VARS")==0)

        if(b_format==false)
        {
            continue; // do not process until we get format line.
        }


        iss.clear();
        iss.str(line);
        ps.clear();
        while (iss >> p)
        {
            ps.push_back(p);
        }

        if (ps.size() < 4)
            continue; // less than 4 elements, empty line??

        c++;

        if (xpos != -1)
        {
            p1.push_back(atof(ps[xpos].c_str()) - 1);
        }
        if (xpos_ppm != -1)
        {
            p1_ppm.push_back(atof(ps[xpos_ppm].c_str()));
        }

        if (height != -1)
        {
            p_intensity.push_back(atof(ps[height].c_str()));
        }
        else
            p_intensity.push_back(0.0);

        if (xw != -1)
        {
            float s = atof(ps[xw].c_str());
            sigmax.push_back(s * 0.425); // from FWHH to sigma, suppose GAussian shape
        }
        else
            sigmax.push_back(3.0);

        if (ass != -1)
        {
            user_comments.push_back(ps[ass]);
        }
        else
        {
            user_comments.push_back("peaks" + std::to_string(c));
        }

        if (confidence != -1)
        {
            confident_level.push_back(std::stod(ps[confidence]));
        }
        else
        {
            confident_level.push_back(1.0);
        }
    } // end of while(getline(fin,line))

    if (p1_ppm.size() > 0) // fill in point from ppm.
    {
        p1.clear();
        for (unsigned int i = 0; i < p1_ppm.size(); i++)
        {
            p1.push_back((p1_ppm[i] - begin1) / step1); // direct dimension
        }
    }
    else // fill in ppm from points
    {
        p1_ppm.clear();
        for (unsigned int i = 0; i < p1.size(); i++)
        {
            double f1 = begin1 + step1 * (p1[i]); // direct dimension
            p1_ppm.push_back(f1);
        }
    }
    
    // fill in intensity information from spectrum
    if (height == -1)
    {
        for (int i = 0; i < p1.size(); i++)
        {
            int n1 = int(p1[i] + 0.5) - 1; // -1 because start at 0 in this program but start from 1 in pipe's tab file.
            if (n1 < 0)
                n1 = 0;
            if (n1 > ndata_frq - 1)
                n1 = ndata_frq - 1;
            p_intensity[i] = spectrum_real[n1]; // n1 is direct dimension; n2 is indirect
        }
    }

    return true;
}

bool spectrum_fit_1d::peak_reading_json(const std::string &content)
{
    Json::Value root, peaks;
    std::istringstream fin(content);

    fin >> root;
    peaks = root["picked_peaks"];

    user_comments.clear();
    for (int i = 0; i < peaks.size(); i++)
    {
        p1_ppm.push_back(peaks[i]["cs_x"].asDouble());
        p_intensity.push_back(peaks[i]["index"].asDouble());
        sigmax.push_back(peaks[i]["sigmax"].asDouble());
        gammax.push_back(peaks[i]["gammax"].asDouble());
        user_comments.push_back("peaks" + std::to_string(i));
    }

    p1.clear();

    for (unsigned int i = 0; i < p1_ppm.size(); i++)
    {
        p1.push_back((p1_ppm[i] - begin1) / step1); // direct dimension
    }

    return true;
};

bool spectrum_fit_1d::set_peaks(const spectrum_1d_peaks p)
{
    p1 = p.x;
    p_intensity = p.a;
    sigmax = p.sigmax;
    gammax = p.gammax;

    /**
     * Set user_comments to "peak" for all peaks
    */
    user_comments.resize(p1.size(), "peak");

    p1_ppm=p.ppm;



    /**
     * Fill p_intensity_all_spectra 
     * This is required even for single spectrum fitting.
    */ 
    for (int i = 0; i < p1.size(); i++)
    {
        std::vector<double> temp;
        temp.push_back(p_intensity[i]); 
        p_intensity_all_spectra.push_back(temp);
    }


    std::vector<double> sx;
    sx.clear();
    for (unsigned int i = 0; i < p1.size(); i++)
    {
        float s1 = 0.5346 * gammax[i] * 2 + std::sqrt(0.2166 * 4 * gammax[i] * gammax[i] + sigmax[i] * sigmax[i] * 8 * 0.6931);
        sx.push_back(s1); // sx is fwhh
    }
    median_width_x = ldw_math_1d::calcualte_median(sx);
    if(n_verbose>=1) std::cout << "Median peak width is estimated to be " << median_width_x << " points from picking." << std::endl;
    if (median_width_x < 5.0)
    {
        median_width_x = 5.0;
        if(n_verbose>=1) std::cout << "Set median peak width along x to 5.0" << std::endl;
    }
    return true;
}

#ifdef WEBASSEMBLY
/**
 * Exposed functions
*/
EMSCRIPTEN_BINDINGS(dp_1d_module_fit) {

    /**
     * Note: base class function init is exposed in spectrum_pick_1d.cpp, so it is not included here.
    */


    class_<spectrum_fit_1d,base<fid_1d>>("spectrum_fit_1d")
        .constructor()
        .class_property("n_verbose", &shared_data_1d::n_verbose)
        .function("init", &spectrum_fit_1d::init)
        .function("init_fit", &spectrum_fit_1d::init_fit)
        .function("init_error", &spectrum_fit_1d::init_error)
        .function("read_first_spectrum_from_buffer",&spectrum_fit_1d::read_first_spectrum_from_buffer)
        .function("peak_reading_from_string", &spectrum_fit_1d::peak_reading_from_string)
        .function("prepare_to_read_additional_spectrum_from_buffer", &spectrum_fit_1d::prepare_to_read_additional_spectrum_from_buffer)
        .function("peak_fitting", &spectrum_fit_1d::peak_fitting)
        .function("output_as_string", &spectrum_fit_1d::output_as_string)
        .function("output_json_as_string", &spectrum_fit_1d::output_json_as_string)
        .function("get_size_of_recon", &spectrum_fit_1d::get_size_of_recon)
        .function("get_data_of_recon", &spectrum_fit_1d::get_data_of_recon)
        .property("spe_recon", &spectrum_fit_1d::spe_recon);
        ;

    
    /**
     * float vector registration is also done in spectrum_pick_1d.cpp, not here.
    */
}
#endif // WEBASSEMBLY

