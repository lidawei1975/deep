// #include <omp.h>
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


#include "kiss_fft.h"
#include "json/json.h"
#include "ldw_math.h"
#include "commandline.h"
#include "spectrum_fwhh_1d.h" //spectrum_io_1d is included in spectrum_fwhh_1d.h

#include "phase_data.h"

template <class myType>
void sortArr(std::vector<myType> &arr, std::vector<int> &ndx)
{
    std::vector<std::pair<myType, int>> vp;

    for (int i = 0; i < arr.size(); ++i)
    {
        vp.push_back(std::make_pair(arr[i], i));
    }

    std::sort(vp.begin(), vp.end()); // sort pairs by first element in ascending order

    // sort indices based on sorted pairs but in descending order
    for (int i = vp.size() - 1; i >= 0; i--)
    {
        ndx.push_back(vp[i].second);
    }
};

class spectrum_phasing_1d : public spectrum_fwhh_1d // spectrum_fwhh_1d is a derived class of spectrum_io_1d
{
private:
    std::array<double, 2> phase_correction; // phase_correction[0] is total left phase, phase_correction[1] is total right phase

    float fwhh_1d_median; // median of peak widthes

    int max_loop;
    int max_peak;
    int max_dist;
    bool b_end;

    /**
     * @brief Define validation_dnn here
     * The order or layers are same as order defination here
     */
    conv1d validation_dnn_c1, validation_dnn_c2, validation_dnn_c3;
    pool1d validation_pool1;
    conv1d validation_dnn_c4;
    pool1d validation_pool2;
    dense validation_dnn_d1, validation_dnn_d2, validation_dnn_d3;

    /**
     * @brief Define phasing_dnn here
     * The order of layers are same as order defination here
     */
    conv1d phase_dnn_c1, phase_dnn_c2, phase_dnn_c3;
    pool1d phase_pool1;
    conv1d phase_dnn_c4;
    pool1d phase_pool2;
    dense phase_dnn_d1, phase_dnn_d2, phase_dnn_d3;

protected:
    std::vector<int> p1;            // peak position
    std::vector<float> p_intensity; // peak intensity
    double max_intensity;           // maximum of p_intensity (used to normalize p_intensity)
    int water_pos;                  // water peak position

    bool normalize_spectrum(std::vector<float> &y) const;         // normalize spectrum to be between 0 and 1. This is required by DNN
    bool is_valid_from_all_peaks(int p_inten,int start, int end, int direction) const; // return false if phase cannot be determined, using peaks information only
    bool is_valid(std::vector<float> spectrum_part) const;        // return false if phase cannot be determined
    bool has_phase_error(std::vector<float> spectrum_part) const; // return true if phase is incorrect

    bool simple_peak_picking();
    double calculate_entropy_of_spe(const std::vector<float> &s) const;
    bool calculate_phased_spectrum(const std::vector<float> &spe_real, const std::vector<float> &spe_imag, const double phase_left, const double phase_right, std::vector<float> &spe_real_phased, std::vector<float> &spe_image_phased) const;
    bool entropy_minimization(double &, double &) const;
    bool assess_phase_at_peaks(std::vector<int> &left_cross, std::vector<int> &right_cross, int &npeak_tested, int stride) const;
    bool assess_two_end_phase_error(const int,const int,int &left_end,int &right_end) const;
    bool gd_optimization_from_cross(int,int,const std::vector<int> left_cross, const std::vector<int> right_cross, const int npeak_tested, double &, double &) const;
    bool phase_spectrum(const double phase_left, const double phase_right);
    double test_consistence_with_peak_based_method(const double left_end,const double right_end,const std::vector<int> left_cross, const std::vector<int> right_cross,const int npeak_tested) const;

public:
    spectrum_phasing_1d();
    ~spectrum_phasing_1d();
    bool set_up_parameters(const int max_loop, const int max_peak, const int max_dist,const bool b_end);
    bool auto_phase_correction();                                                    // main working function
    std::array<double, 2> get_phase_correction() const { return phase_correction; }; // phase_correction[0] is total left phase, phase_correction[1] is total right phase
};

spectrum_phasing_1d::spectrum_phasing_1d()
{
    /**
     * @brief initialize validation_dnn here
     * First one always have 1 input channel (from 1D spectrum)
     */
    int n = 0;
    validation_dnn_c1.set_size(21, 1, 10);                // nkernel, ninput, nfilter
    n += validation_dnn_c1.read(validation_dnn_data + n); // 220

    validation_dnn_c2.set_size(11, 10, 6);
    n += validation_dnn_c2.read(validation_dnn_data + n); // 666

    validation_dnn_c3.set_size(11, 6, 4);
    n += validation_dnn_c3.read(validation_dnn_data + n); // 268

    validation_pool1.set_size(4, 3); // ninput,npool

    validation_dnn_c4.set_size(11, 4, 4);
    n += validation_dnn_c4.read(validation_dnn_data + n); // 180

    validation_pool2.set_size(4, 3); // ninput,npool

    /**
     * There is a flatten layer here. Which is not implemented explicitly
     * We need to reshape the output from previous layer to a 1D array
     */

    validation_dnn_d1.set_size(2000, 10); // ninput,noutput
    validation_dnn_d1.set_act(activation_function::relu);
    n += validation_dnn_d1.read(validation_dnn_data + n); // 20010

    validation_dnn_d2.set_size(10, 5);
    validation_dnn_d2.set_act(activation_function::relu);
    n += validation_dnn_d2.read(validation_dnn_data + n); // 55

    validation_dnn_d3.set_size(5, 2);
    validation_dnn_d3.set_act(activation_function::softmax);
    n += validation_dnn_d3.read(validation_dnn_data + n); // 12

    /**
     * @brief initialize phase_dnn here
     * The two DNNs have identical structure (with different weights and biases)
     * First one always have 1 input channel (from 1D spectrum)
     */
    n = 0;
    phase_dnn_c1.set_size(21, 1, 10);           // nkernel, ninput, nfilter
    n += phase_dnn_c1.read(phase_dnn_data + n); // 220

    phase_dnn_c2.set_size(11, 10, 6);
    n += phase_dnn_c2.read(phase_dnn_data + n); // 666

    phase_dnn_c3.set_size(11, 6, 4);
    n += phase_dnn_c3.read(phase_dnn_data + n); // 268

    phase_pool1.set_size(4, 3); // ninput,npool

    phase_dnn_c4.set_size(11, 4, 4);
    n += phase_dnn_c4.read(phase_dnn_data + n); // 180

    phase_pool2.set_size(4, 3); // ninput,npool

    /**
     * There is a flatten layer here. Which is not implemented explicitly
     * We need to reshape the output from previous layer to a 1D array
     */

    phase_dnn_d1.set_size(2000, 10); // ninput,noutput
    phase_dnn_d1.set_act(activation_function::relu);
    n += phase_dnn_d1.read(phase_dnn_data + n); // 20010

    phase_dnn_d2.set_size(10, 5);
    phase_dnn_d2.set_act(activation_function::relu);
    n += phase_dnn_d2.read(phase_dnn_data + n); // 55

    phase_dnn_d3.set_size(5, 2);
    phase_dnn_d3.set_act(activation_function::softmax);
    n += phase_dnn_d3.read(phase_dnn_data + n); // 12

    /**
     * by default, set water peak pos at far far away
     */
    water_pos = -1000000;

    return;
};

spectrum_phasing_1d::~spectrum_phasing_1d(){

};

/**
 * @brief set up some used defined parameters
 * @param _max_loop: maximal number of loops for phase correction
 * @param _max_peak: maximal number of peaks to be used for phase correction
*/
bool spectrum_phasing_1d::set_up_parameters(const int _max_loop, const int _max_peak, int _max_dis,bool _b_end)
{
    max_loop = _max_loop;
    max_peak = _max_peak;
    max_dist=_max_dis;
    b_end=_b_end;
    return true;
};


/**
 * @brief simple peak picking (find local maximum, using nearest 3 neighbors)
 * Will update p1 and p_intensity (class members)
 * p1 and p_intensity are sorted by p_intensity
 * @return true
 * @return false
 */
bool spectrum_phasing_1d::simple_peak_picking()
{
    // to be safe. Clear p_intensity and p1
    p1.clear();
    p_intensity.clear();

    for (int i = 103; i < xdim - 103; i++)
    {
        if (spect[i] > spect[i + 1] && spect[i] > spect[i + 2] && spect[i] > spect[i + 3] && spect[i] > spect[i - 1] && spect[i] > spect[i - 2] && spect[i] > spect[i - 3])
        {
            p1.push_back(i);
            p_intensity.push_back(spect[i]);
        }
    }

    // sort p_intensity in descending order, also save index
    std::vector<int> ndx;
    sortArr(p_intensity, ndx);

    std::vector<int> p1_sorted;
    std::vector<float> p_intensity_sorted;
    for (int i = 0; i < ndx.size(); i++)
    {
        p1_sorted.push_back(p1[ndx[i]]);
        p_intensity_sorted.push_back(p_intensity[ndx[i]]);
    }

    // copy p1_sorted to p1, p_intensity_sorted to p_intensity
    p1 = p1_sorted;
    p_intensity = p_intensity_sorted;

    // get max intensity
    max_intensity = p_intensity[0];

    return true;
};

/**
 * @brief normalize spectrum to be between 0 and 1
 *
 */
bool spectrum_phasing_1d::normalize_spectrum(std::vector<float> &y) const
{
    float max_y = -std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    for (int i = 0; i < y.size(); i++)
    {
        if (y[i] > max_y)
        {
            max_y = y[i];
        }
        if (y[i] < min_y)
        {
            min_y = y[i];
        }
    }
    for (int i = 0; i < y.size(); i++)
    {
        y[i] = (y[i] - min_y) / (max_y - min_y);
    }
    return true;
}

/**
 * @brief check if phase can be determined for each peak. This function will check region just outside is_valid and has_phase_error woring region
 * In details, it will check peaks in the region from start to end (inclusive) to see if there is any very high peaks.
 * If there is, return false (phase cannot be determined)
 * If there is no very high peaks, return true (phase can be determined)
 * @param p_inten: peak intensity of the peak to be checked
 * @param start: start position of the region to be checked, we will check from start to the end of the spectrum
 * @param len: length of the region to be checked, to be used to calculate allowed peak height
 * @param direction: 1 means check right side, -1 means check left side
 * 
*/
bool spectrum_phasing_1d::is_valid_from_all_peaks(int p_inten,int start, int len, int direction) const
{
    //p1 and p_intensity are sorted by p_intensity in descending order
    for(int i=0;i<p1.size();i++)
    {
        //we can break out when p_intensity[i] is smaller than p_inten*0.4
        if(p_intensity[i]<p_inten*0.4)
        {
            break;
        }
        //calculate the distance between p1[i] and start. It must >0 and < end-start, otherwise we can skip this peak
        int dist1=p1[i]-start;
        if(direction==-1)
        {
            dist1=-dist1;
        }

        if(dist1<0) //skip this peak
        {
            continue;
        }
        
        /**
         * define a allowed peak height for this peak. It is 0.6*p_inten+dist1/(end-start)*0.4*p_inten 
         * i.e., linearly increase from 0.6*p_inten to p_inten when dist1 increases from 0 to end-start
        */
        float allowed_peak_height=0.6*p_inten+0.4*p_inten*dist1*10.0/len;
        if(p_intensity[i]>allowed_peak_height)
        {
            return false;
        }
    }
    /**
     * @brief if we reach here, it means all peaks in the region from start to end (inclusive) are not very high
    */
    return true;
}



/**
 * @brief Apply DNN to check spectral data of 500 pixels on the left and right side of the peak
 *
 * @param y: input spectrum to be checked
 * @return true: if the phase error can be detected (input spectrum is valid)
 * @return false: if the phase error can not be detected (input spectrum is invalid)
 */
bool spectrum_phasing_1d::is_valid(std::vector<float> y) const
{

    bool b_phase_error = true;
    /**
     * @brief y[0] should have the largest value
     * Any peak after y[0] should < 0.7*y[0], if not, return false (cannot determine whether phase is correct or not)
     */
    for (int i = 4; i < y.size(); i++)
    {
        // if y[i] is a peak (local maximum) and y[i] > 0.4*y[0], return false
        if (y[i] > y[i - 1] && y[i] > y[i + 1] && y[i] > 0.4 * y[0])
        {
            b_phase_error = false;
            break;
        }
    }

    if (b_phase_error == false) // no need to check phase
    {
        return false;
    }

    y.resize(500); // only check 500 pixels for DNN

    /**
     * @brief  apply validation DNN to check if the phase can be determined or not
     * false means cannot determine the phase
     * See spectrum_phasing_1d constructor for details of the DNN
     */
    std::vector<float> t1, t2, t3, t4, t5, t6, t7, t8, output;

    // y is 1*500*1

    validation_dnn_c1.predict(500, y, t1);  // 1*500*10
    validation_dnn_c2.predict(500, t1, t2); // 1*500*6
    validation_dnn_c3.predict(500, t2, t3); // 1*500*4
    validation_pool1.predict(500, t3, t4);  // 1*500*4
    validation_dnn_c4.predict(500, t4, t5); // 1*500*4
    validation_pool2.predict(500, t5, t6);  // 1*500*4

    // flattern t6 from 1*500*4 to 1*2000 implicitly without any operation

    validation_dnn_d1.predict(1, t6, t7);     // 1*10
    validation_dnn_d2.predict(1, t7, t8);     // 1*5
    validation_dnn_d3.predict(1, t8, output); // size of t9 is 2

    if (output[1] > output[0]) // valid
    {
        return true;
    }
    else
    {
        return false;
    }
};

/**
 * @brief Apply DNN to check spectral data of 500 pixels on the left and right side of the peak
 *
 * @param y input spectrum to be checked
 * @return true if phase is incorrect
 * @return false if phase is correct
 */

bool spectrum_phasing_1d::has_phase_error(std::vector<float> y) const
{
    /**
     * @brief apply phase DNN to determine the phase is correct or not
     * false means phase is correct
     */
    std::vector<float> t1, t2, t3, t4, t5, t6, t7, t8, output;

    // y is 1*500*1

    phase_dnn_c1.predict(500, y, t1);  // 1*500*10
    phase_dnn_c2.predict(500, t1, t2); // 1*500*6
    phase_dnn_c3.predict(500, t2, t3); // 1*500*4
    phase_pool1.predict(500, t3, t4);  // 1*500*4
    phase_dnn_c4.predict(500, t4, t5); // 1*500*4
    phase_pool2.predict(500, t5, t6);  // 1*500*4

    // flattern t6 from 1*500*4 to 1*2000 implicitly without any operation

    phase_dnn_d1.predict(1, t6, t7);     // 1*10
    phase_dnn_d2.predict(1, t7, t8);     // 1*5
    phase_dnn_d3.predict(1, t8, output); // size of t9 is 2

    if (output[1] > output[0]) // has phase error
    {
        return true;
    }
    else
    {
        return false;
    }
};

/**
 * @brief calculate phased spectrum
 * @param spe_real vector of spectrum, real part
 * @param spe_imag vector of spectrum, imaginary part
 * @param phase_left phase in degree at the left end of the spectrum. Inclusive
 * @param phase_right phase in degree at the right end of the spectrum. Exclusive
 * @param spe_real_phased vector of phased spectrum, real part
 * @param spe_imag_phased vector of phased spectrum, imaginary part
 * We suppose linear phase change from left to right
 */

bool spectrum_phasing_1d::calculate_phased_spectrum(
    const std::vector<float> &spe_real,
    const std::vector<float> &spe_imag,
    const double phase_left,
    const double phase_right,
    std::vector<float> &spe_real_phased,
    std::vector<float> &spe_imag_phased) const
{
    spe_real_phased.clear();
    spe_real_phased.resize(spe_real.size());
    spe_imag_phased.clear();
    spe_imag_phased.resize(spe_imag.size());

    double phase_step = (phase_right - phase_left) / spe_real.size();

    for (int i = 0; i < spe_real.size(); i++)
    {
        double phase = phase_left + phase_step * i;
        /**
         * @brief We are using degree as default unit for phase
         * That is, phase_left and phase_right are in degree and need to be converted to radian in the following calculation
         */
        phase = phase / 180 * M_PI;
        spe_real_phased[i] = spe_real[i] * cos(phase) + spe_imag[i] * sin(phase);
        spe_imag_phased[i] = spe_imag[i] * cos(phase) - spe_real[i] * sin(phase);
    }

    return true;
};

/**
 * @brief calculate the entropy of a spectrum (after abs)
 *
 * @param s vector of spectrum
 * @return double: entropy
 */
double spectrum_phasing_1d::calculate_entropy_of_spe(const std::vector<float> &s) const
{
    /**
     * calculate absolute value of s
     */
    std::vector<float> absolute_s;
    for (int i = 0; i < s.size(); i++)
    {
        absolute_s.push_back(fabs(s[i]));
    }

    /**
     * renormalize absolute_s to be between 0 and 1
     */
    float max_s = -std::numeric_limits<float>::max();
    float min_s = std::numeric_limits<float>::max();
    for (int i = 0; i < absolute_s.size(); i++)
    {
        if (absolute_s[i] > max_s)
        {
            max_s = absolute_s[i];
        }
        if (absolute_s[i] < min_s)
        {
            min_s = absolute_s[i];
        }
    }
    for (int i = 0; i < absolute_s.size(); i++)
    {
        absolute_s[i] = (absolute_s[i] - min_s) / (max_s - min_s);
    }

    double sum = 0;
    for (int i = 0; i < absolute_s.size(); i++)
    {
        sum += absolute_s[i];
    }

    double entropy = 0;
    for (int i = 0; i < absolute_s.size(); i++)
    {
        if (absolute_s[i] > 0)
        {
            entropy += absolute_s[i] * log(absolute_s[i]);
        }
    }

    return -entropy / sum + log(sum);
}

/**
 * @brief minimize the entropy of the spectrum, using gradient descent
 *
 * @return true on success
 * @return false on failure
 */
bool spectrum_phasing_1d::entropy_minimization(double &phase_left, double &phase_right) const
{
    phase_left = 0.0;
    phase_right = 0.0;
    double step = 20.0;            // step size in degree
    double derivative_step = 0.01; // step size in degree for derivative calculation. Analytical derivative is not used at this moment.

    std::vector<float> stribed_spect, stribed_spe_image;
    int nstep = ndata/4096; //stribe to have only 4k points for entropy calculation
    int n_skip=ndata/10; //exclude the first and last 10% of the spectrum

    for (int i = n_skip; i < spect.size()-n_skip; i+=nstep)
    {
        stribed_spect.push_back(spect[i]);
        stribed_spe_image.push_back(spe_image[i]);
    }


    // maximal number of iterations is 500
    for (int i = 0; i < 2500; i++)
    {
        // calculate base entropy
        std::vector<float> spe_real_phased;
        std::vector<float> spe_image_phased; // not used
        calculate_phased_spectrum(stribed_spect, stribed_spe_image, phase_left, phase_right, spe_real_phased, spe_image_phased);
        double entropy = calculate_entropy_of_spe(spe_real_phased);

        // calculate the derivative of entropy numerically
        std::vector<float> spe_real_phased_left;
        calculate_phased_spectrum(stribed_spect, stribed_spe_image, phase_left + derivative_step, phase_right, spe_real_phased_left, spe_image_phased);
        double entropy_left = calculate_entropy_of_spe(spe_real_phased_left);

        std::vector<float> spe_real_phased_right;
        calculate_phased_spectrum(stribed_spect, stribed_spe_image, phase_left, phase_right + derivative_step, spe_real_phased_right, spe_image_phased);
        double entropy_right = calculate_entropy_of_spe(spe_real_phased_right);

        double derivative_left = (entropy_left - entropy) / derivative_step;
        double derivative_right = (entropy_right - entropy) / derivative_step;

        // update phase
        phase_left -= step * derivative_left;
        phase_right -= step * derivative_right;

        // exit if both derivatives are smaller than 1e-7
        if (fabs(derivative_left) <1e-7 && fabs(derivative_right) <1e-7)
        {
            break;
        }
        if((i+1)%100==0)
        {
            std::cout<<"entropy_minimization, i="<<i<<", phase_left="<<phase_left<<", phase_right="<<phase_right;
            std::cout<<", derivative_left="<<derivative_left;
            std::cout<<", derivative_right="<<derivative_right;
            std::cout<<std::endl;
        }

        // scale step size
        step *= 0.99999;
    }

    return true;
}

bool spectrum_phasing_1d::phase_spectrum(const double phase_left, const double phase_right)
{
    // Actually updating the spectrum (spect and spe_image)
    std::vector<float> spe_real_phased, spe_image_phased;
    calculate_phased_spectrum(spect, spe_image, phase_left, phase_right, spe_real_phased, spe_image_phased);

    // if sum of spectrum is negative, flip the spectrum
    double sum = 0;
    for (int i = 0; i < spe_real_phased.size(); i++)
    {
        sum += spe_real_phased[i];
    }
    if (sum < 0)
    {
        for (int i = 0; i < spe_real_phased.size(); i++)
        {
            spe_real_phased[i] = -spe_real_phased[i];
            spe_image_phased[i] = -spe_image_phased[i];
        }
    }

    /**
     * @brief copy the phased spectrum to class member variable
     *
     */
    spect = spe_real_phased;
    spe_image = spe_image_phased;

    return true;
};

/**
 * @brief using min std at two ends to determine the phase error
 * @param from: start phase, inclusive
 * @param to: end phase, inclusive
 * @param left_end: best phase at the left end.
 * @param right_end: best phase at the right end.
*/
bool spectrum_phasing_1d::assess_two_end_phase_error(const int from, const int to, int &left_end,int &right_end) const
{
    int n_end_size=ndata/200; //ndata is the number of pixels in the spectrum
    float left_min_std=std::numeric_limits<float>::max();
    float right_min_std=std::numeric_limits<float>::max();

    
    for (int j = from; j <= to; j++)
    {
        std::vector<float> left_end_at_phase, right_end_at_phase;
        left_end_at_phase.clear();
        right_end_at_phase.clear(); //probably not necessary

        float phase = j * M_PI / 180 * 1;
        float cos_phase = cos(phase);
        float sin_phase = sin(phase);
        for(int i = 0; i < n_end_size; i++)
        {
             left_end_at_phase.push_back(spect[i] * cos_phase + spe_image[i] * sin_phase);
        }
        for(int i=ndata-n_end_size;i<ndata;i++)
        {
            right_end_at_phase.push_back(spect[i] * cos_phase + spe_image[i] * sin_phase);
        }

        /**
         * calculate standard deviation of left_end_at_phase and right_end_at_phase
         * Step1: calculate mean
         * Step2: calculate standard deviation
         */
        float left_mean=0;
        float right_mean=0;
        for(int i=0;i<n_end_size;i++)
        {
            left_mean+=left_end_at_phase[i];
            right_mean+=right_end_at_phase[i];
        }
        left_mean/=n_end_size;
        right_mean/=n_end_size;

        float left_std=0;
        float right_std=0;
        for(int i=0;i<n_end_size;i++)
        {
            left_std+=(left_end_at_phase[i]-left_mean)*(left_end_at_phase[i]-left_mean);
            right_std+=(right_end_at_phase[i]-right_mean)*(right_end_at_phase[i]-right_mean);
        }
        left_std=sqrt(left_std/n_end_size);
        right_std=sqrt(right_std/n_end_size);

        if(left_std<left_min_std)
        {
            left_min_std=left_std;
            left_end=j;
        }
        if(right_std<right_min_std)
        {
            right_min_std=right_std;
            right_end=j;
        }
        
    }
    return true;
}

/**
 * @brief assess the phase at peaks, using DNNs
 * Save the result in left_cross and right_cross. Both are in degrees at an interget and shifted by 10 degrees
 * because we pre-calcualte the spectrum at each 21 phases (from -10 to 10 degrees)
 * npeak_tested is the number of peaks tested (at most 100)
 * stride is the step size when checking both sides of each peak using the DNNs
 * @return true
 * @return false
 */
bool spectrum_phasing_1d::assess_phase_at_peaks(std::vector<int> &left_cross, std::vector<int> &right_cross, int &npeak_tested, int stride) const
{
    /**
     * @brief To speed up calculation, we pre-calcualte the spectrum at each 21 phases (from -10 to 10 degrees)
     * From experiences, entropy based methods above won't have phase error larger than 10 degrees.
     * This part of code will update class member varible spes_at_each_phase
     *
     */
    int from = -10;
    int to = 10;

    // this is used to speed up calculation (pre-calcualte the spectrum at each 21 phases (from -10 to 10 degrees))
    std::vector<std::vector<float>> spes_at_each_phase;

    spes_at_each_phase.clear();
    spes_at_each_phase.resize(to - from + 1, std::vector<float>(spect.size()));

    for (int j = from; j <= to; j++)
    {
        float phase = j * M_PI / 180 * 1;
        float cos_phase = cos(phase);
        float sin_phase = sin(phase);
        for (int i = 0; i < spect.size(); i++)
        {
            spes_at_each_phase[j - from][i] = spect[i] * cos_phase + spe_image[i] * sin_phase;
        }
    }

    npeak_tested = std::min(max_peak, int(p1.size())); // check at most 100 peaks

    /**
     * @brief left_corss size is min(npeak,100), right_cross size is min(npeak,100). index is same as index of spes_at_each_phase, from -10 to 10 degrees
     * In both left_cross and right_cross, each element is the crossing from having to not having error
     * For left, possible values are from 20 to -1 (-1 means no crossing at all, all have phase error) (20 means none of the 21 phases have phase error)
     * For right, possible values are from 0 to 21 (21 means no crossing at all, all have phase error) (0 means none of the 21 phases have phase error)
     */
    left_cross.clear();
    left_cross.resize(npeak_tested, 20);

    right_cross.clear();
    right_cross.resize(npeak_tested, 0);

    /**
     * @brief step size is 1,2 and 3 pixels when checking both sides of each peak using the DNNs if stride == 1
     * step=1: at most look 500 points away: 0,1,2,....,499
     * step=2: at most look 999 pixels away, 0,2,4,....,998
     * step=3: at most look 1499 pixels away, 0,3,6,....,1497
     * When stride !=1, step ==> step*stride
     */
    std::vector<int> steps;
    
    for(int i=1;i<=max_dist;i++)
    {
        steps.push_back(i*stride);
    }

    /**
     * @brief Obtain corrected phase correction for each peak, check at most 100 peaks.
     *
     */
    for (int i = 0, c = 0; i < npeak_tested; i++)
    {
        for (int i_step = 0; i_step < steps.size(); i_step++)
        {
            int step = steps[i_step]; // step size when checking both sides of each peak using the DNNs

            std::vector<int> left_flags, right_flags;

            /**
             * @brief Step 1. Check if phase can be determined for each peak, using function is_valid()
             * A peak's left or right side is valid only if at all 21 phases, the phase can be determined (is_valid() returns true)
             */

            bool b_left_valid = true;
            bool b_right_valid = true;

            for (int j = 0; j < spes_at_each_phase.size(); j++)
            {

                // get partial spectrum on the right side of p1[i] at length of 500 pixels
                std::vector<float> partial_spe_right;
                partial_spe_right.clear();

                for (int k = 0; k < 500+100; k++)
                {
                    if (p1[i] + k * step >= spes_at_each_phase[j].size()) // out of range
                    {
                        break;
                    }
                    if (abs(p1[i] + k * step - water_pos) < 100) // water signal is too close
                    {
                        break;
                    }

                    partial_spe_right.push_back(spes_at_each_phase[j][p1[i] + k * step]);
                }

                // in case the peak is too close to the edge of the spectrum. It is not valid to check phase error at this peak with this step size
                if (partial_spe_right.size() < 500)
                {
                    b_right_valid = false;
                    break;
                }

                //check peaks from p1[i]+499*step to p1[i]+499*step+499*step to see if they are valid. See comments of is_valid_from_all_peaks() for details
                // if(is_valid_from_all_peaks(p_intensity[i],p1[i]+499*step,500*step,1)==false)
                // {
                //     b_right_valid = false;
                //     break;
                // }

                // normalise the partial spectrum, which is required by is_valid()
                normalize_spectrum(partial_spe_right);

                // Apply vilation DNN to check if the phase can be determined or not
                if (is_valid(partial_spe_right) == false)
                {
                    b_right_valid = false;
                    break;
                }
            }

            for (int j = 0; j < spes_at_each_phase.size(); j++)
            {
                // get partial spectrum on the left side of p2[i] at length of 500 pixels
                std::vector<float> partial_spe_left;
                partial_spe_left.clear();

                for (int k = 0; k < 500+100; k++)
                {
                    if (p1[i] - k * step < 0) // out of range
                    {
                        break;
                    }
                    if (abs(p1[i] - k * step - water_pos) < 100) // water signal is too close
                    {
                        break;
                    }
                    partial_spe_left.push_back(spes_at_each_phase[j][p1[i] - k * step]);
                }

                // in case the peak is too close to the edge of the spectrum. It is not valid to check phase error at this peak with this step size
                if (partial_spe_left.size() < 500)
                {
                    b_left_valid = false;
                    break;
                }

                //check peaks from p1[i]+499*step to p1[i]+499*step+499*step to see if they are valid. See comments of is_valid_from_all_peaks() for details
                // if(is_valid_from_all_peaks(p_intensity[i],p1[i]-499*step,500*step,-1)==false)
                // {
                //     b_left_valid = false;
                //     break;
                // }

                // normalise the partial spectrum, which is required by is_valid()
                normalize_spectrum(partial_spe_left);

                if (is_valid(partial_spe_left) == false)
                {
                    b_left_valid = false;
                    break;
                }
            }

            int current_detected_left_phase_error; // at this step size.
            if (b_left_valid == true)
            {
                /**
                 * @brief for left side run has_phase_error to check if phase is correct or not
                 * Positive phase error will only be detected by left side spectrum
                 * We start from phase error of 10 degrees, and decrease by 1 degree each time.
                 * Once phase error is NOT detected, we stop and save the 1st cross (from having phase error to not having phase error)
                 * if the last one is still having phase error, we save the last one -1 (which is -10-1=-11 degrees)
                 */
                current_detected_left_phase_error = -1; // default value is -1, which means no crossing at all, all have phase error
                for (int i_phase = 20; i_phase >= 0; i_phase--)
                {
                    std::vector<float> partial_spe_left;
                    partial_spe_left.clear();
                    for (int k = 0; k < 500; k++)
                    {
                        // note: no need to check out of range, b_left_valid=false when out of range
                        partial_spe_left.push_back(spes_at_each_phase[i_phase][p1[i] - k * step]);
                    }
                    // normalize the partial spectrum, which is required by has_phase_error()
                    normalize_spectrum(partial_spe_left);
                    if (has_phase_error(partial_spe_left) == false)
                    {
                        current_detected_left_phase_error = i_phase;
                        break;
                    }
                }
            }
            else
            {
                /**
                 * @brief label none has phase error
                 * to be accurate, it is not possible to detect phase error. But we don't distinguish between not valid for detection and not having phase error
                 */
                current_detected_left_phase_error = 20;
            }

            /**
             * @brief When detecting at different steps (look away 500,1000 and 1500 pixels), we save the smallest phase error detected
             */
            if (current_detected_left_phase_error < left_cross[i])
            {
                left_cross[i] = current_detected_left_phase_error;
            }

            int current_detected_right_phase_error; // at this step size.
            if (b_right_valid == true)
            {
                /**
                 * @brief for right side run has_phase_error to check if phase is correct or not
                 * Negative phase error will only be detected by right side spectrum
                 * We start from phase error of -10 degrees, and increase by 1 degree each time.
                 * Once phase error is NOT detected, we stop and save the 1st cross (from having phase error to not having phase error)
                 * if the last one is still having phase error, we save the last one +1 (which is 10+1=11 degrees)
                 */
                current_detected_right_phase_error = 21; // default value is 21, which means no crossing at all, all have phase error
                for (int i_phase = 0; i_phase <= 20; i_phase++)
                {
                    std::vector<float> partial_spe_right;
                    partial_spe_right.clear();

                    for (int k = 0; k < 500; k++)
                    {
                        // note: no need to check out of range, b_right_valid=false when out of range
                        partial_spe_right.push_back(spes_at_each_phase[i_phase][p1[i] + k * step]);
                    }

                    // normalize the partial spectrum, which is required by has_phase_error()
                    normalize_spectrum(partial_spe_right);

                    if (has_phase_error(partial_spe_right) == false)
                    {
                        current_detected_right_phase_error = i_phase;
                        break;
                    }
                }
            }
            else
            {
                /**
                 * @brief label none has phase error
                 * to be accurate, it is not possible to detect phase error. But we don't distinguish between not valid for detection and not having phase error
                 */
                current_detected_right_phase_error = 0;
            }

            /**
             * @brief When detecting at different steps (look away 500,1000 and 1500 pixels), we save the largest phase error detected
             */
            if (current_detected_right_phase_error > right_cross[i])
            {
                right_cross[i] = current_detected_right_phase_error;
            }

        } // end of for(int i_step=0;i_step<steps.size();i_step++)
    }     // end of for (int i = 0, c = 0; i < nn; i++)

    return true;
};

/**
 * @brief This is the main function to run phase correction.
 * It will update member variables spect,spe_image and phase_correction
 *
 * @return true
 * @return false
 */

bool spectrum_phasing_1d::auto_phase_correction()
{

    double phase_left1 = 0.0;
    double phase_right1 = 0.0;

#ifdef DEBUG
    std::cout << "Messup spectrum by -120.0 and 34.8 degrees." << std::endl;
    phase_spectrum(-120.0, 34.8);
#endif

    
    /**
     * @brief initial phase correction using an entropy minimization method.
     * This is a const function, so it will not change any member variables
     */

    phase_left1 = 0.0;
    phase_right1 = 0.0;
    entropy_minimization(phase_left1, phase_right1);
    std::cout << "Finished entropy minimization. Phase left: " << phase_left1 << " Phase right: " << phase_right1 << std::endl;

    phase_correction[0] = phase_left1;
    phase_correction[1] = phase_right1;

    /**
     * @brief update member variables spect and spe_image
     */
    phase_spectrum(phase_left1, phase_right1);

    /**
     * @brief Update class member variables p1 and p_intensity
     * p1 is peak position and p1_intensity is peak intensity, both are sorted by intensity
     *
     */
    simple_peak_picking();

    /**
     * @brief calculate FWHH of the spectrum
     * using base class spectrum_fwhh_1d function get_median_peak_width()
     * The function call will only update member variable fwhh_1d_pos,fwhh_1d_wid base class member variables
     */
    fwhh_1d_median = get_median_peak_width();
    std::cout << "Median peak width: " << fwhh_1d_median << std::endl;

    /**
     * Because below valid and phase_error DNN were both trained, using peak width from 3 to 15 pixels,
     * we need to make sure the peak width is in this range.
     * Without using spline, we will only do data reduction by stride=2,3,4,... until
     * the peak width is most near 10.
     */
    float min_r = 100000.0;
    int best_stride = 1;
    for (int stride = 1; stride < fwhh_1d_median / 5.0f; stride++)
    {
        float r = fwhh_1d_median / stride / 8;
        if (r < 1.0)
        {
            r = 1.0 / r;
        }
        if (r < min_r)
        {
            min_r = r;
            best_stride = stride;
        }
    }
    std::cout << "Best stride: " << best_stride << std::endl;

    /**
     * Set water peak at the middle of the spectrum
     */
    water_pos = spect.size() / 2;

    std::vector<int> left_cross;
    std::vector<int> right_cross;
    int npeak_tested; // number of peaks tested. nn=min(p1.size(),100)

    /**
     * we will run following code multiple times to get better phase correction
     */
    bool b_use_end=false; //if true, we will use two ends method to assess phase error
    for (int loop = 0; loop < max_loop; loop++)
    {

        /**
         * @brief Update class member variables p1 and p_intensity
         * p1 is peak position and p1_intensity is peak intensity, both are sorted by intensity
         * After each iteration, we will update p1 and p_intensity to be as accurate as possible. 
         * This is very fast anyway. 
         */
        simple_peak_picking();


        left_cross.clear();
        right_cross.clear();
        /**
         * @brief assess phase error at each peak. Save the first location where phase error vanishes
         * This function does not update any member variables
         */
        assess_phase_at_peaks(left_cross, right_cross, npeak_tested, best_stride);
        std::cout << "Finished assessing phase error using DNN at each peak." << std::endl;

        /** 
         * Shift left_cross [0:21) and right_cross by -10 degrees, so that they are in the same range as spes_at_each_phase [-10:11)
        */
        for (int i = 0; i < left_cross.size(); i++)
        {
            left_cross[i] -= 10;
        }
        for (int i = 0; i < right_cross.size(); i++)
        {
            right_cross[i] -= 10;
        }

        /**
         * For debug purpose, save left_cross and right_cross to file, together with peak position and intensity
         */
        #ifdef DEBUG
            std::ofstream fout("peak_cross_"+std::to_string(loop)+".txt");
            for (int i = 0; i < left_cross.size(); i++)
            {
                fout << p1[i] << " " << p_intensity[i] << " " << left_cross[i] << " " << right_cross[i] << std::endl;
            }
            fout.close();
        #endif


        /**
         * -100 is a flag, means not usable. Otherwise, it is the index of best phase correction at the two ends of the spectrum
        */
        int left_end=-10000;
        int right_end=-10000; 

        /**
         * we only use two ends of the spectrum to assess phase error after the first loop, so that 
         * if ends based method is consistent with peak based method, we will know and not to include them
         * this is probably caused by Bruker's digital filtering method
        */
        if(b_end && loop==2)
        {
            assess_two_end_phase_error(-80,80,left_end, right_end); //from -80 to 80 degrees
            double cost1=test_consistence_with_peak_based_method(left_end,right_end,left_cross,right_cross,npeak_tested);  //cost if we use two ends method
            double cost2=test_consistence_with_peak_based_method(0.0,0.0,left_cross,right_cross,npeak_tested); //current cost
            // std::cout<<"At loop "<<loop<<" use two ends method. Left end: "<<left_end<<" Right end: "<<right_end;
            // std::cout<<", cost1 = " << cost1;           
            // std::cout<<", cost2 = " << cost2;
            // std::cout<<std::endl;
            if(cost1>cost2*10.0) //if two ends method is not consistent with peak based method, we will not use two ends method
            {
                b_use_end=false;
                std::cout<<"Turn off two ends method."<<std::endl;
            }
            else
            {
                b_use_end=true;
                std::cout<<"Turn on two ends method."<<std::endl;
            }
        }

        

        /**
         * @brief Run gradient descent to find the best linear phase correction
         * This is a const function, so it will not change any member variables
         *
         * There are two fitting parameters: phase_left (at 0) and phase_right (at spect.size(), exclusive)
         * effective phase correction for each peak is linearly interpolated from phase_left to phase_right phase_at_peak=(phase_left+(phase_right-phase_left)*i/spect.size())
         * For each peak:
         *
         * Note: for left, possible values are from 20 to -1 (-1 means no crossing at all, all have phase error) (20 means none of the 21 phases have phase error)
         * cost_left=(phase_at_peak-left_cross)^2 if phase_at_peak>left_cross
         * cost_left=(phase_at_peak-left_cross)*0.1 if phase_at_peak<=left_cross
         *
         * Note: for right, possible values are from 0 to 21 (21 means no crossing at all, all have phase error) (0 means none of the 21 phases have phase error)
         * cost_right= (phase_at_peak-right_cross)^2 if phase_at_peak<right_cross
         * cost_right=-(phase_at_peak-right_cross)*0.1 if phase_at_peak>=right_cross
         */
        double phase_left2, phase_right2; // to be optimized by gradient descent function gd_optimization_from_cross
        if(b_use_end==true && loop>2)
        {
            assess_two_end_phase_error(-80,80,left_end, right_end); //from -80 to 80 degrees    
            gd_optimization_from_cross(left_end,right_end,left_cross, right_cross, npeak_tested, phase_left2, phase_right2);
        }
        else
        {
            gd_optimization_from_cross(-10000,-10000,left_cross, right_cross, npeak_tested, phase_left2, phase_right2);
        }

        phase_correction[0] += phase_left2;
        phase_correction[1] += phase_right2;

        std::cout << "Finished gradient descent optimization at loop "<< loop<<". Phase left: " << phase_correction[0] << " Phase right: " << phase_correction[1] << std::endl;

        /**
         * @brief update member variables spect and spe_image
         */
        phase_spectrum(phase_left2, phase_right2);
    }
    return true;
};

/**
 * @brief This function is used to test if two ends method is consistent with peak based method
 * @param left_end: best phase at the left end, according to two ends method
 * @param right_end: best phase at the right end, according to two ends method
 * @param left_cross: left_cross[i] is the first phase where phase error vanishes for peak i
 * @param right_cross: right_cross[i] is the first phase where phase error vanishes for peak i
 * @param npeak_tested: number of peaks tested. nn=min(p1.size(),100)
 * @return weighted cost of phase error at each peak, if phase correction from two end method is applied.
*/
double spectrum_phasing_1d::test_consistence_with_peak_based_method(const double left_end,const double right_end,const std::vector<int> left_cross, const std::vector<int> right_cross,const int npeak_tested) const
{
    /**
     * @brief weight of each peak is proportional to its intensity
     * Normalized to have a maximal of 1.0
     * Weight of the ends will be set to 0.5
     */
    std::vector<double> weights;
    weights.clear();
    for (int i = 0; i < p_intensity.size(); i++)
    {
        weights.push_back(p_intensity[i] / max_intensity);
    }

    // loop over all peaks
    double total_cost=0.0;
    double total_weight=0.0;
    for (int j = 0; j < npeak_tested; j++)
    {
        // calculate phase correction for each peak. Note: all input are intergers, so an explicit cast is needed
        double phase_at_peak = left_end + double(right_end - left_end) * p1[j] / spect.size();

        if (left_cross[j] < 20) //informative.
        {
            if (phase_at_peak > left_cross[j])
            {
                total_cost += (phase_at_peak - left_cross[j]) * (phase_at_peak - left_cross[j]) * weights[j];
                total_weight += weights[j];
            }
        }

        if(right_cross[j]>0) //informative
        {
            if (phase_at_peak < right_cross[j])
            {
                total_cost += (phase_at_peak - right_cross[j]) * (phase_at_peak - right_cross[j]) * weights[j];
                total_weight += weights[j];
            }
        }
    }

    //get weighted average
    if(total_weight>0.0)
    {
        return total_cost/total_weight;
    }
    else
    {
        return 0.0;
    }
}


/**
 * @brief This function is used to run gradient descent to find the best linear phase correction
 * This is a const function, so it will not change any member variables
 * There are two fitting parameters: phase_left (at 0) and phase_right (at spect.size(), exclusive)
 * effective phase correction for each peak is linearly interpolated from phase_left to phase_right phase_at_peak=(phase_left+(phase_right-phase_left)*i/spect.size())
 * For each peak:
 * Note: for left, possible values are from 10 to -11 (-11 means no crossing at all, all have phase error) (10 means none of the 21 phases have phase error)
 * cost_left=(phase_at_peak-left_cross)^2 if phase_at_peak>left_cross
 * Note: for right, possible values are from -10 to 11 (11 means no crossing at all, all have phase error) (-10 means none of the 21 phases have phase error)
 * cost_right= (phase_at_peak-right_cross)^2 if phase_at_peak<right_cross
 * @param left_end: best phase at the left end, according to two ends method
 * @param right_end: best phase at the right end, according to two ends method
 * @param left_cross: left_cross[i] is the first phase where phase error vanishes for peak i
 * @param right_cross: right_cross[i] is the first phase where phase error vanishes for peak i
 * @param npeak_tested: number of peaks tested. nn=min(p1.size(),100)
 * @param phase_left: best phase at the left end. this is the output
 * @param phase_right: best phase at the right end. This is the output
*/
bool spectrum_phasing_1d::gd_optimization_from_cross(int left_end, int right_end,const std::vector<int> left_cross, const std::vector<int> right_cross, const int npeak_tested, double &phase_left, double &phase_right) const
{

    phase_left = 0.0;
    phase_right = 0.0;

    double step = 0.05; // step size in degree

    /**
     * @brief weight of each peak is proportional to its intensity
     * Normalized to have a maximal of 1.0
     * Weight of the ends will be set to 0.5
     */
    std::vector<double> weights;
    weights.clear();
    for (int i = 0; i < p_intensity.size(); i++)
    {
        weights.push_back(p_intensity[i] / max_intensity);
    }

    // maximal number of iterations is 10000. This is a overkill, but it is very fast.
    for (int i = 0; i < 10000; i++)
    {
        /**
         * @brief analytical derivative method
         *
         */
        double total_derivative_wrt_phase_left = 0.0;
        double total_derivative_wrt_phase_right = 0.0;
        double total_cost = 0.0;

        double total_weight=0.0;
        // loop over all peaks
        for (int j = 0; j < npeak_tested; j++)
        {
            // calculate phase correction for each peak
            double phase_at_peak = phase_left + (phase_right - phase_left) * p1[j] / spect.size();

            /**
             * @brief Get derivative of phase_at_peak with respect to phase_left and phase_right
             */
            double delta_phase_left = 1 - double(p1[j]) / spect.size();
            double delta_phase_right = double(p1[j]) / spect.size();

            /**
             * @brief Notice meaning of delta_phase_left and delta_phase_right are now different
             * They both include the weight of the peak
             */
            delta_phase_left *= weights[j];
            delta_phase_right *= weights[j];

            /**
             * @brief left_cross >=20 means none has phase error or is not valid for detection, skip
             *
             */
            if (left_cross[j] < 10)
            {
                /**
                 * @brief derivative of cost_left with respect to phase_at_peak
                 * when phase_at_peak <= left_cross[j], cost_left=(phase_at_peak-left_cross)*0.1
                 * when phase_at_peak > left_cross[j], cost_left=(phase_at_peak-left_cross)^2
                 */
                double delta = 0.1;
                delta=0.0;
                if (phase_at_peak > left_cross[j])
                {
                    delta = 2.0 * (phase_at_peak - left_cross[j]);
                    total_cost += (phase_at_peak - left_cross[j]) * (phase_at_peak - left_cross[j]) * weights[j];
                    total_weight+=weights[j];
                }
                /**
                 * @brief derivative of cost_left with respect to phase_left and phase_right
                 * using derivation chain rule
                 */
                total_derivative_wrt_phase_left += delta * delta_phase_left;
                total_derivative_wrt_phase_right += delta * delta_phase_right;
                
            }

            /**
             * @brief right_cross <=0 means none has phase error or is not valid for detection, skip
             *
             */
            if (right_cross[j] > -10)
            {
                /**
                 * @brief derivative of cost_right with respect to phase_at_peak
                 *
                 */
                double delta = -0.1; // when phase_at_peak >= right_cross[j], cost_right=-(phase_at_peak-right_cross)*0.1
                delta=0.0;
                if (phase_at_peak < right_cross[j])
                {
                    delta = 2.0 * (phase_at_peak - right_cross[j]);
                    total_cost += (phase_at_peak - right_cross[j]) * (phase_at_peak - right_cross[j]) * weights[j];
                    total_weight+=weights[j];
                }
                total_derivative_wrt_phase_left += delta * delta_phase_left;
                total_derivative_wrt_phase_right += delta * delta_phase_right;
                
            }
        }

        /**
         * if they are at -10000, it means not usable
        */
        if(left_end>-1000 && right_end>-1000) 
        {
            /**
             * cost is (phase_left-left_end)^2 + (phase_right-right_end)^2. Weights are total_weight*0.01
             */
            total_derivative_wrt_phase_left += 0.2 * total_weight * 2.0 * (phase_left - left_end );
            total_derivative_wrt_phase_right += 0.2 * total_weight * 2.0 * (phase_right - right_end);
        }

        /**
         * @brief update phase_left and phase_right using gradient descent
         *
         */
        phase_left -= step * total_derivative_wrt_phase_left;
        phase_right -= step * total_derivative_wrt_phase_right;

        // if (i % 1000 == 0)
        // {
        //     std::cout << "Iteration: " << i << " Phase left: " << phase_left << " Phase right: " << phase_right << " step = " << step;
        //     std::cout << " cost: " << total_cost/total_weight;
        //     std::cout << std::endl;
        // }

        step *= 0.99999999; // scale step size
    }

    return true;
}

int main(int argc, char **argv)
{

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-f");
    args2.push_back("arguments_phase.txt");
    args3.push_back("input arguments file name.");

    args.push_back("-in");
    args2.push_back("test.ft1");
    args3.push_back("input spectral file names.");

    args.push_back("-out");
    args2.push_back("test-phased.ft1");
    args3.push_back("output spectral file names.");

    args.push_back("-n_loop");
    args2.push_back("5");
    args3.push_back("number of iterations for phase correction.");

    args.push_back("-n_peak");
    args2.push_back("100");
    args3.push_back("number of peaks to be used for phase correction.");

    args.push_back("-n_dist");
    args2.push_back("3");
    args3.push_back("Check phase error of each peak at most 500(1),1000(2),1500(3) pixels away from the peak.");

    args.push_back("-b_end");
    args2.push_back("yes");
    args3.push_back("use both ends of the spectrum to assess phase correction.");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    std::string infname = cmdline.query("-in");
    std::string outfname = cmdline.query("-out");

    if (cmdline.query("-h") != "yes")
    {
        /**
         * Some user defined parameters
        */
        int n_dist = std::stoi(cmdline.query("-n_dist"));
        int n_loop = std::stoi(cmdline.query("-n_loop"));
        int n_peak = std::stoi(cmdline.query("-n_peak"));
        bool b_end = cmdline.query("-b_end") == "yes" || cmdline.query("-b_end") == "Yes" || cmdline.query("-b_end") == "YES" || cmdline.query("-b_end") == "y" || cmdline.query("-b_end") == "Y";
        
        class spectrum_phasing_1d x;
        x.init(10.0, 3.0, 0.0); // this function is required by spectrum_io, but not used in this program
        x.read_spectrum(infname);
        x.set_up_parameters(n_loop, n_peak, n_dist,b_end);
        x.auto_phase_correction();
        x.write_spectrum(outfname);

        std::array<double, 2> phase_correction = x.get_phase_correction();
        std::cout << "phase correction left: " << phase_correction[0] << " right: " << phase_correction[1] << std::endl;
    }

    return 0;
}