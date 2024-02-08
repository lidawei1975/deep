#include "spectrum_simulation.h"

spectrum_simulation::spectrum_simulation()
{
}

spectrum_simulation::~spectrum_simulation()
{
}


bool spectrum_simulation::run(int ndata, int ndata_frq, const std::vector<double> &ppm, const std::vector<float> &fid_real, const std::vector<float> &fid_imag, spectrum_1d_peaks &peak_list,double &step2,int n)
{
    kiss_fft_cfg cfg;
    kiss_fft_cpx *in, *out;


    in = new kiss_fft_cpx[ndata_frq];
    out = new kiss_fft_cpx[ndata_frq];
    for (int i = 0; i < ndata; i++)
    {
        in[i].r = fid_real[i];
        in[i].i = fid_imag[i];
    }

    /**
     * fill remaining of in with 0, i.e., zero filling according to nzf
     */
    for (int i = ndata; i < ndata_frq; i++)
    {
        in[i].r = 0.0;
        in[i].i = 0.0;
    }

    if ((cfg = kiss_fft_alloc(ndata_frq, 0, NULL, NULL)) != NULL)
    {
        kiss_fft(cfg, in, out);
        free(cfg);
    }
    else
    {
        std::cerr << "Error: cannot allocate memory for fft" << std::endl;
        return false;
    }

    /**
     * @var spectrum_real, spectrum_imag: real and imaginary part of the simulated spectrum in frequency domain
     * We don't need swap (fftshift) because how we generate ppm above
     */
    std::vector<float> spectrum_real, spectrum_imag, spectrum_real_temp, spectrum_imag_temp;
    spectrum_real_temp.resize(ndata_frq);
    spectrum_imag_temp.resize(ndata_frq);

    for (int i = 0; i < ndata_frq; i++)
    {
        spectrum_real_temp[i] = out[i].r / sqrt(float(ndata_frq));
        spectrum_imag_temp[i] = out[i].i / sqrt(float(ndata_frq)); // scale by sqrt(ndata_frq) to be consistent with standard fft
    }

    /**
     * move the last n data points to the beginning of the vector for spectrum_real and spectrum_imag to be consistent with ppm
     */
    for (int i = 0; i < n; i++)
    {
        spectrum_real.push_back(spectrum_real_temp[ndata_frq - n + i]);
        spectrum_imag.push_back(spectrum_imag_temp[ndata_frq - n + i]);
    }
    for (int i = n; i < ndata_frq; i++)
    {
        spectrum_real.push_back(spectrum_real_temp[i - n]);
        spectrum_imag.push_back(spectrum_imag_temp[i - n]);
    }

    // #ifdef DEBUG
    // std::ofstream fdebug("debug-spectrum-real.txt");
    // for (int i = 0; i < ndata_frq; i++)
    // {
    //     fdebug << ppm[i] << " " << spectrum_real[i] << std::endl;
    // }
    // fdebug.close();
    // #endif

    /**
     * Run Deep Picker 1D to get peak list (with assignment to each protons). Peaks belong to same proton form a group.
     * There is no need to run spectrum_pick_1d, because simulated spectrum is very clean with less dynamic range.
     * At r2=6.0, FWHH is about 45 data points, when ndata_frq=262144. So we can estiamte fwhh analytically.
     */
    double fwhh = 45.0 / 262144.0 * ndata_frq;

    /**
     * Optimal DP model 2 fwhh is 6.0.
     */
    int nstride_database_spectrum = int(fwhh / 6.0 + 0.5); // stride is 1/6 of fwhh

    /**
     * nstride_database_spectrum must >=1
     */
    if (nstride_database_spectrum < 1)
    {
        nstride_database_spectrum = 1;
    }

    /**
     * Get strided real spectrum and corresponding ppm
     */
    std::vector<float> spectrum_real_strided, ppm_strided;
    for (int i = 0; i < ndata_frq; i += nstride_database_spectrum)
    {
        spectrum_real_strided.push_back(spectrum_real[i]);
        ppm_strided.push_back(ppm[i]);
    }

    /**
     * begin, step and stop of simulated spectrum, not to be confused with global variables begin,step,step
     * which are for the experimental spectrum
     */
    double simualted_spectrum_begin, simualted_spectrum_step, simualted_spectrum_stop; //

    simualted_spectrum_begin = ppm_strided[0];
    simualted_spectrum_step = (ppm_strided[spectrum_real_strided.size() - 1] - ppm_strided[0]) / double(spectrum_real_strided.size() - 1);
    simualted_spectrum_stop = ppm_strided[spectrum_real_strided.size() - 1] + simualted_spectrum_step;

    /**
     * step2 is used to convert from points to ppm when write sigma and gamma of database peaks to json file
     * step2 is set by user when loading database peak directly.
     * step2 need to be set to simualted_spectrum_step when simulating database peaks
     */
    step2 = simualted_spectrum_step;

    /**
     * peak1d is a class defined in dnn_picker.h
     * We will use it to get peak list, using model 2
     */
    class spectrum_pick_1d peak_picker;

    peak_picker.n_verbose = -1; // suppress output

    /**
     * set parameters for peak_picker.
     * In our simulated spectrum, a single peak from one proton has a peak height of 1.0
     */
    peak_picker.init(10 /*peak height cutoff*/, 3.0 /* noise floor*/, 0.001 /*noise level*/);
    peak_picker.init_mod(2); // DNN model 2, optimal fwhh is 6.0.


    /**
     * set spectrum for peak_picker
     */
    peak_picker.set_spectrum_from_data(spectrum_real_strided, simualted_spectrum_begin, simualted_spectrum_step, simualted_spectrum_stop);

    /**
     * run the peak picker, without negative peaks
     */
    peak_picker.spectrum_pick_1d_work(false /*b_negative*/);

    /**
     * Get the peak list form picker
     */
    
    peak_picker.get_peaks(peak_list);


    /**
     * Water proof the peak list. When fwhh is too narrow, the peak list may be empty.
     * Fall back to a simnple peak picker.
     */
    if(peak_list.a.size() == 0 || ndata_frq < 8192)
    {
        /**
         * Simple picker
        */
        peak_list.a.clear();
        peak_list.x.clear();
        peak_list.ppm.clear();
        peak_list.volume.clear();
        peak_list.sigmax.clear();
        peak_list.gammax.clear();
        peak_list.confidence.clear();


        for(int i=2;i<spectrum_real_strided.size()-2;i++)
        {
            if(spectrum_real_strided[i]>spectrum_real_strided[i-1] && spectrum_real_strided[i]>spectrum_real_strided[i-2] && spectrum_real_strided[i]>spectrum_real_strided[i+1] && spectrum_real_strided[i]>spectrum_real_strided[i+2])
            {
                peak_list.a.push_back(spectrum_real_strided[i]);
                peak_list.x.push_back(i);
                peak_list.ppm.push_back(ppm_strided[i]);
                peak_list.volume.push_back(2.0);
                peak_list.sigmax.push_back(1.0);
                peak_list.gammax.push_back(0.01);
                peak_list.confidence.push_back(1.0);
            }
        }
    }


    class spectrum_fit_1d peak_fitter;
    /**
     * save initial values as picker
     */
    peak_fitter.init(10 /*peak height cutoff*/, 3.0 /* noise floor*/, 0.001 /*noise level*/);
    peak_fitter.init_fit(2 /** voigt*/, 50 /*maxround*/, 0.0000001 /* to near cutoff*/);
    peak_fitter.set_spectrum_from_data(spectrum_real_strided, simualted_spectrum_begin, simualted_spectrum_step, simualted_spectrum_stop);
    peak_fitter.set_for_one_spectrum();
    peak_fitter.set_peaks(peak_list);

    /**
     * run peak fitting to optimize peak position, height, width, etc.
     * Set n_verbose to 0 to suppress output
     */
    peak_fitter.peak_fitting();

    /**
     * clear peak_list before we get fitted peaks
     */
    peak_list.a.clear();
    peak_list.x.clear();
    peak_list.ppm.clear();
    peak_list.volume.clear();
    peak_list.sigmax.clear();
    peak_list.gammax.clear();
    peak_list.confidence.clear();

    /**
     * Get peak list after fitting
     */
    peak_fitter.get_fitted_peaks(peak_list);



    delete[] in;
    delete[] out;

    return true;
}
