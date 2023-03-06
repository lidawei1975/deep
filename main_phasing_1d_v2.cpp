//#include <omp.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>

#include "spline.h"
#include "kiss_fft.h"
#include "json/json.h"
#include "ldw_math.h"
#include "commandline.h"
#include "spectrum_io_1d.h"

#include "phase_data.h"
#include "phasing_dnn.h"

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
    for (int i = vp.size()-1; i >=0 ; i--)
    {
        ndx.push_back(vp[i].second);
    }
};

class spectrum_phasing_1d : public spectrum_io_1d
{
private:
    class dense d1, d2, d3, d4, d5;

    std::vector<std::vector<float>> spes_at_each_phase;
    std::vector<double> each_phase;

protected:
    int median_width;
    double a, b, r; // a,b,r are the parameters of the phase correction.
    std::vector<double> p1, p_intensity;
    std::vector<double> p2, p_intensity2;

    std::vector<phase_estimator> phase_estimator_vector;

    bool simple_peak_picking2(bool b_negative = false);
    int simple_peak_width(int p, double height);
    double calculate_entropy(std::vector<double> s);
    template <typename T>
    T calculate_mad(std::vector<T> s);
    double calculate_rmsd(std::vector<double> s);
    bool estimate_noise_level();
    bool move_median(std::vector<double> in, std::vector<double> &out, int width);
    bool move_mean(std::vector<double> in, std::vector<double> &out, int width);

    bool check_peak(std::vector<double> spectrum_part, std::vector<float> &scores);
    int check_score(std::vector<float> scores);

    double entropy(std::vector<double> s);

    bool entropy_minimization();

public:
    spectrum_phasing_1d();
    ~spectrum_phasing_1d();
    bool work(int n_select);
};

spectrum_phasing_1d::spectrum_phasing_1d()
{
    // load in a very simple 3 dense connected layers ANN
    int n = 0;
    d1.set_act(activation_function::relo);
    d1.set_size(200, 40); // ninput, nfilter
    n += d1.read(ann_data + n);

    d2.set_act(activation_function::relo);
    d2.set_size(40, 30); // ninput, nfilter
    n += d2.read(ann_data + n);

    d3.set_act(activation_function::relo);
    d3.set_size(30, 20); // ninput, nfilter
    n += d3.read(ann_data + n);

    d4.set_act(activation_function::relo);
    d4.set_size(20, 10); // ninput, nfilter
    n += d4.read(ann_data + n);

    d5.set_act(activation_function::softmax);
    d5.set_size(10, 2); // ninput, nfilter
    n += d5.read(ann_data + n);

    std::cout << "n=" << n << ", which shoul be 10122." << std::endl;
};
spectrum_phasing_1d::~spectrum_phasing_1d(){

};

template <typename T>
T spectrum_phasing_1d::calculate_mad(std::vector<T> s)
{
    T mean = 0.0;
    for (unsigned int i = 0; i < s.size(); i++)
    {
        mean += s[i];
    }
    mean /= s.size();

    T mad = 0.0;
    std::vector<T> temp;
    for (unsigned int i = 0; i < s.size(); i++)
    {
        mad += fabs(s[i] - mean);
        temp.push_back(fabs(s[i] - mean));
    }
    mad /= s.size();

    std::sort(temp.begin(), temp.end());
    mad = temp[(int)(s.size() / 2)];

    return mad;
};

double spectrum_phasing_1d::calculate_rmsd(std::vector<double> s)
{
    double rmsd = 0.0;
    for (unsigned int i = 0; i < s.size(); i++)
    {
        rmsd += s[i] * s[i];
    }
    rmsd /= s.size();
    rmsd = sqrt(rmsd);

    return rmsd;
};

double spectrum_phasing_1d::calculate_entropy(std::vector<double> s)
{

    // normalize final_spectrum to 10.0
    double max_spe = 0.0;
    int max_pos = -1;
    for (int k = 0; k < s.size(); k++)
    {
        if (fabs(s[k]) > max_spe)
        {
            max_spe = fabs(s[k]);
            max_pos = k;
        }
    }

    for (int k = 0; k < s.size(); k++)
    {
        s[k] /= max_spe;
    }

    double sum = 0.0;
    double negative_sum = 0.0;
    for (int i = 0; i < s.size(); i++)
    {
        if (s[i] < 0.0)
        {
            s[i] = -s[i];
            negative_sum += 1.0;
        }
        sum += s[i];
    }

    double entropy = 0.0;
    for (int i = 0; i < s.size(); i++)
    {
        s[i] /= sum;
        entropy -= s[i] * log(s[i]);
    }

    // entropy+=negative_sum/s.size();

    return entropy;
}

int spectrum_phasing_1d::simple_peak_width(int p, double center_height)
{
    // count number of elements of spe on the left of spe[p], whose height is larger than center_height/2
    int nleft = 0;
    for (int i = p - 1; i >= 0; i--)
    {
        if (spect[i] < center_height / 2)
            break;
        if (spect[i] > spect[i + 1])
            break;
        nleft++;
    }
    // count number of elements of spe on the right of spe[p], whose height is larger than center_height/2
    int nright = 0;
    for (int i = p + 1; i < spect.size(); i++)
    {
        if (spect[i] < center_height / 2)
            break;
        if (spect[i] > spect[i - 1])
            break;
        nright++;
    }
    return nleft + nright;
};

bool spectrum_phasing_1d::simple_peak_picking2(bool b_negative)
{
    if (b_negative == true)
    {
        for (int i = 1; i < xdim - 1; i++)
        {
            if (fabs(spect[i]) > noise_level * user_scale && fabs(spect[i]) > fabs(spect[i + 1]) && fabs(spect[i]) > fabs(spect[i + 2])

                && fabs(spect[i]) > fabs(spect[i - 1]) && fabs(spect[i] > spect[i - 2]))
            {
                p1.push_back(i);
                p_intensity.push_back(spect[i]);
            }
        }
    }
    else
    {
        for (int i = 1; i < xdim - 1; i++)
        {
            if (spect[i] > noise_level * user_scale && spect[i] > spect[i + 1] && spect[i] > spect[i + 2]

                && spect[i] > spect[i - 1] && spect[i] > spect[i - 2])
            {
                p1.push_back(i);
                p_intensity.push_back(spect[i]);
            }
        }
    }
    int npeak = p1.size();

    // get min and max of p1
    double min_p1 = p1[0];
    double max_p1 = p1[0];
    for (int i = 1; i < npeak; i++)
    {
        if (p1[i] < min_p1)
            min_p1 = p1[i];
        if (p1[i] > max_p1)
            max_p1 = p1[i];
    }

    // sort p_intensity2 in descending order, also save index
    std::vector<int> ndx;
    sortArr(p_intensity, ndx);

    p2.clear();
    p_intensity2.clear();

    // get the n_select largest peaks
    for (int i = 0; i < ndx.size(); i++)
    {
        p2.push_back(p1[ndx[i]]);
        p_intensity2.push_back(p_intensity[ndx[i]]);
    }

    return true;
};

bool spectrum_phasing_1d::move_median(std::vector<double> in, std::vector<double> &out, int width)
{
    out.clear();
    for (int i = 0; i < in.size(); i++)
    {
        int b = std::max(0, i - width / 2);
        int s = std::min(i + width / 2 + 1, int(in.size()));
        std::vector<double> temp(in.begin() + b, in.begin() + s);

        int n = temp.size() / 2;
        nth_element(temp.begin(), temp.begin() + n, temp.end());
        out.push_back(temp[n]);
    }
    return true;
}

bool spectrum_phasing_1d::move_mean(std::vector<double> in, std::vector<double> &out, int width)
{
    out.clear();
    for (int i = 0; i < in.size(); i++)
    {
        int b = std::max(0, i - width / 2);
        int s = std::min(i + width / 2 + 1, int(in.size()));
        std::vector<double> temp(in.begin() + b, in.begin() + s);
        double sum = 0.0;
        for (int j = 0; j < temp.size(); j++)
        {
            sum += temp[j];
        }
        out.push_back(sum / temp.size());
    }
    return true;
}

bool spectrum_phasing_1d::estimate_noise_level()
{
    std::vector<double> t;

    for (int i = 0; i < spect.size(); i++)
    {
        if (spect[i] > 0.0)
        {
            t.push_back(spect[i]);
        }
        else
        {
            t.push_back(-spect[i]);
        }
    }

    std::vector<double> scores = t;
    sort(scores.begin(), scores.end());
    noise_level = scores[scores.size() / 2] * 1.4826;
    if (noise_level <= 0.0)
        noise_level = 0.1; // artificail spectrum w/o noise
    std::cout << "First round, noise level is " << noise_level << std::endl;

    std::vector<int> flag(xdim, 0); // flag

    for (int i = 0; i < xdim; i++)
    {
        if (t[i] > 5.5 * noise_level)
        {
            int xstart = std::max(i - 5, 0);
            int xend = std::min(i + 6, xdim);

            for (int n = xstart; n < xend; n++)
            {
                flag[n] = 1;
            }
        }
    }
    scores.clear();

    for (int i = 0; i < xdim; i++)
    {
        if (flag[i] == 0)
        {
            scores.push_back(t[i]);
        }
    }

    sort(scores.begin(), scores.end());
    noise_level = scores[scores.size() / 2] * 1.4826;
    std::cout << "Final noise level is estiamted to be " << noise_level << std::endl;

    return true;
};

bool spectrum_phasing_1d::check_peak(std::vector<double> y, std::vector<float> &scores)
{
    std::vector<double> x;

    int n = y.size();  // n=400 in this case

    for (int i = 0; i < n; i++)
    {
        x.push_back(i);
    }
    tk::spline st(x, y);

    std::vector<int> locs;
    for (int i = 100; i <=n; i += 5)
    {
        locs.push_back(i);
    }
    //locs = [100:5:n]; n=400 in this case

    // locs.clear();
    // locs.push_back(340);

    int n_input_size=200; //input size for the neural network

    std::vector<float> y_spline_float;
    for (int i = 0; i < locs.size(); i++)
    {
        std::vector<double> y_spline;
        double step = locs[i] / double(n_input_size);
        for (int j = 0; j < n_input_size; j++)
        {
            y_spline.push_back(st(j * step));
        }
        double min_y_spline = y_spline[0];
        double max_y_spline = y_spline[0];
        for (int j = 1; j < n_input_size; j++)
        {
            if (y_spline[j] < min_y_spline)
                min_y_spline = y_spline[j];
            if (y_spline[j] > max_y_spline)
                max_y_spline = y_spline[j];
        }
        double y_spline_range = max_y_spline - min_y_spline;
        for (int j = 0; j < n_input_size; j++)
        {
            y_spline[j] = y_spline[j] - min_y_spline;
            y_spline[j] = y_spline[j] / y_spline_range;
        }

        for (int j = 0; j < n_input_size; j++)
        {
            y_spline_float.push_back(log(y_spline[j]+1e-5)/12.0);
        }
    }
    // run dnn on y_spline to get the score

    std::vector<float> temp1, temp2, temp3, temp4;

    //size of y_spline_float is 200*locs.size()=200*61=12200
    //size of score is 61*2=122
    d1.predict(locs.size(), y_spline_float, temp1);
    d2.predict(locs.size(), temp1, temp2);
    d3.predict(locs.size(), temp2, temp3);
    d4.predict(locs.size(), temp3, temp4);
    d5.predict(locs.size(), temp4, scores);

    // std::ofstream outfile("scores.txt");
    // for(int i=0;i<scores.size();i++)
    // {
    //     outfile << scores[i] << " ";
    //     if(i%3==2)
    //         outfile << std::endl;
    // }
    // outfile.close();

    // std::ofstream outfile2("y_spline.txt");
    // for(int i=0;i<y_spline_float.size();i++)
    // {
    //     outfile2 << y_spline_float[i] << " ";
    //     if(i%50==49)
    //         outfile2 << std::endl;
    // }
    // outfile2.close();

    return true;
};

int spectrum_phasing_1d::check_score(std::vector<float> scores)
{
    int n_data = scores.size() / 2;

    std::vector<float> scores_0, scores_1;

    // scores_0=scores[0:3:6:9:.....]
    for (int i = 0; i < n_data; i++)
    {
        scores_0.push_back(scores[i * 2]);      // without phase error scores
        scores_1.push_back(scores[i * 2 + 1]);  // with phase error scores
    }

    // find longest stretch when scores_0 is the largest
    int max_len0 = 0;
    int len = 0;
    for (int i = 0; i < n_data; i++)
    {
        if (scores_0[i] > scores_1[i] )
        {
            len++;
        }
        else
        {
            if (len > max_len0)
            {
                max_len0 = len;
            }
            len = 0;
        }
    }
    if (len > max_len0)
    {
        max_len0 = len;
    }

    // find longest stretch when scores_1 is the largest
    int max_len1 = 0;
    len = 0;
    for (int i = 0; i < n_data; i++)
    {
        if (scores_1[i] > scores_0[i] )
        {
            len++;
        }
        else
        {
            if (len > max_len1)
            {
                max_len1 = len;
            }
            len = 0;
        }
    }
    if (len > max_len1)
    {
        max_len1 = len;
    }

    if (max_len1 >= 6)
    {
        return 1;
    }
    else
    {
        return 0;
    }
};


double spectrum_phasing_1d::entropy(std::vector<double> s)
{
    std::vector<double> absolute_s(s.size());

    for (int i = 0; i < s.size(); i++)
    {
        absolute_s[i]=abs(s[i]);
    }

    double sum = 0;
    for (int i = 0; i < s.size(); i++)
    {
        sum += absolute_s[i];
    }

    double entropy = 0;
    for (int i = 0; i < s.size(); i++)
    {
        entropy += absolute_s[i] * log(absolute_s[i]);
    }

    return -entropy/sum+log(sum);

}

bool spectrum_phasing_1d::entropy_minimization()
{
    return true;
}



bool spectrum_phasing_1d::work(int n_select)
{

    // entropy_minimization();


    estimate_noise_level(); // for unphased spe with baseline, noise estimation is spectrum_io is not accurate at all.

    // under sampling spe, keep 1 data points per 1 data points
    {
        std::vector<float> spe_image_under;
        for (int i = 0; i < spe_image.size(); i += 1)
        {
            spe_image_under.push_back(spe_image[i]);
        }

        std::vector<float> spe_under;
        for (int i = 0; i < spect.size(); i += 1)
        {
            spe_under.push_back(spect[i]);
        }

        spect=spe_under;
        spe_image=spe_image_under;
        xdim=spect.size();
    }

    // prepare the spectrum at the range of phase
    {
        int from=-5;
        int to=5;

        spes_at_each_phase.clear();
        spes_at_each_phase.resize(to-from+1,std::vector<float>(spect.size()));
        each_phase.clear();

        for (int j = from; j <= to; j++)
        {
            float phase = j * M_PI / 180 * 1;
            float cos_phase = cos(phase);
            float sin_phase = sin(phase);
            for(int i=0;i<spect.size();i++)
            {
                spes_at_each_phase[j-from][i]=spect[i]*cos_phase+spe_image[i]*sin_phase;
            }
            each_phase.push_back(j);
        }
    }

    //p2 is peak position and p2_intensity is peak intensity, both are sorted by intensity
    //false means no negative peaks.
    simple_peak_picking2(false); 


    std::ofstream outfile("phase.txt"); //this is for debug

    for (int i = 0, c = 0; i < p2.size() && c < n_select; i++)
    {
        // std::cout << "peak " << i << " is " << p2[i] << " with intensity " << p_intensity2[i] << std::endl;

        std::vector<int> left_flags, right_flags;

        for (int j = 0; j < spes_at_each_phase.size(); j++)
        {

            std::vector<float> scores_right, scores_left;

            // get partial spectrum on the right side of p2[i] at length of 340 pixels
            std::vector<double> partial_spe_right;
            for (int k = p2[i]; k < p2[i] + 400; k++)
            {
                partial_spe_right.push_back(spes_at_each_phase[j][k]);
            }


            check_peak(partial_spe_right, scores_right);
            right_flags.push_back(check_score(scores_right));

            // get partial spectrum on the left side of p2[i] at length of 340 pixels
            std::vector<double> partial_spe_left;
            for (int k = p2[i]; k > p2[i] - 400; k--)
            {
                partial_spe_left.push_back(spes_at_each_phase[j][k]);
            }
            check_peak(partial_spe_left, scores_left);
            left_flags.push_back(check_score(scores_left));

            // std::cout << "Phase = " << phase << " left=" << left_flag << " right=" << right_flag << std::endl;
        }

        // if both left_flags and right_flags contain only 0, skip this peak
        bool b_skip = true;
        for (int j = 0; j < left_flags.size(); j++)
        {
            if (left_flags[j] != 0)
            {
                b_skip = false;
            }
            if (right_flags[j] != 0)
            {
                b_skip = false;
            }
        }

        if (b_skip == false)
        {

            std::cout << std::setw(10) << p2[i] << " " << std::setw(20) << p_intensity2[i] / 1e9 << " ";
            for (int j = 0; j < left_flags.size(); j++)
            {
                std::cout << std::setw(1) << left_flags[j];
                outfile << p2[i] - 0.4 << " " << j << " " << left_flags[j] << std::endl;
            }
            std::cout << std::endl;

            std::cout << std::setw(10) << p2[i] << " " << std::setw(20) << p_intensity2[i] / 1e9 << " ";
            for (int j = 0; j < right_flags.size(); j++)
            {
                std::cout << std::setw(1) << right_flags[j];
                 outfile << p2[i] + 0.4 << " " << j << " " << right_flags[j] + 10 << std::endl;
            }
            std::cout << std::endl;
            c++;
            phase_estimator pe;
            pe.init(left_flags, right_flags, p2[i], p_intensity2[i]/1e9,each_phase);
            pe.print();
            std::cout << std::endl << std::endl;
            phase_estimator_vector.emplace_back(pe);
        }
    }
    outfile.close();

    double min_cost=std::numeric_limits<double>::max();
    double min_phase_i, min_phase_j;
    for(int i=0;i<each_phase.size();i++)
    {
        // for(int j=30;j<each_phase.size();j++)
        int j=i;
        {
            double cost=0.0;
            for(int k=0;k<phase_estimator_vector.size();k++)
            {
                double phase=each_phase[i]+(each_phase[j]-each_phase[i])*double(phase_estimator_vector[k].peak_pos);
                cost+=phase_estimator_vector[k].get_cost(phase)*phase_estimator_vector[k].weight; //unit is degree for phase_estimator_vector
                // std::cout<<phase<<" "<<phase_estimator_vector[k].get_cost(phase)<<std::endl;
            }
            // std::cout<<std::endl;
            if(cost<min_cost)
            {
                min_cost=cost;
                min_phase_i=each_phase[i];
                min_phase_j=each_phase[j];
            }
        }
    }

    std::cout<<"min cost="<<min_cost<<" min phase="<<min_phase_i<<" "<<min_phase_j<<std::endl;

    return true;
}

int main(int argc, char **argv)
{

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-in");
    args2.push_back("test.ft1");
    args3.push_back("input spectral file names.");

    args.push_back("-n");
    args2.push_back("4");
    args3.push_back("# of peaks to check");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    std::string infname = cmdline.query("-in");

    if (cmdline.query("-h") != "yes")
    {
        class spectrum_phasing_1d x;

        int npeak = std::stoi(cmdline.query("-n"));

        x.init(10.0, 3.0, 0.0);
        x.read_spectrum(infname);
        x.work(npeak);
    }

    return 0;
}