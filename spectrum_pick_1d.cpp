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
        std::vector<int> r,r2;

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
                while(peak_amplitudes[p]/v_min>v_max/peak_amplitudes[p])
                {
                    p++;
                }
            }
            else
            {
                p = v_max_ndx -1 ;
                while(peak_amplitudes[p]/v_min>v_max/peak_amplitudes[p])
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


spectrum_pick_1d::spectrum_pick_1d() {};
spectrum_pick_1d::~spectrum_pick_1d() {};


bool spectrum_pick_1d::init_mod(int mod)
{
    mod_selection=mod;
    if(mod_selection==1){
        p1.load();
    }
    else if(mod_selection==2)
    {
        p1.load_m2();
    }
    else
    {
        p1.load_m3();
    }
    
    return true;
};


//predict whole line at once. not currently used anywhere
//not suitable for metabolomics (peaks can have huge intensity difference)
bool spectrum_pick_1d::work(std::string outfname)
{
    substract_baseline();
    p1.set_noise_level(noise_level);
    p1.predict(spect);
    p1.predict_step2();

    std::ofstream foutd(outfname+".debug");
    std::ofstream fout(outfname);

    int ndim = p1.output1.size() / 3;
    for (int k = 0; k < ndim; k++)
    {
        foutd << k-20 << " " << p1.output1[k * 3] << " " << p1.output1[k * 3 + 1] << " " << p1.output1[k * 3 + 2] << std::endl;
    }

    for (int k = 0; k < p1.posits.size(); k++)
    {
        fout << k << " " << p1.posits[k] << " " << p1.centes[k] << " " << p1.intens[k] << " " ;
        fout << p1.sigmas[k] << " " << p1.gammas[k] << " " << p1.confidence[k] << std::endl;    
    }

    fout.close();
    foutd.close();

    return true;
};

bool spectrum_pick_1d::substract_baseline()
{
    for(int i=0;i<ndata;i++)
    {
        spect[i]-=baseline[i];
    }
    return true;
}

//segment by segment prediction, as in peak2d
bool spectrum_pick_1d::work2()
{
    // substract_baseline();
    p1.set_noise_level(noise_level);
    int ndim = spect.size();

    std::vector<int> final_segment_begin,final_segment_stop;
    peak_partition();

    for (int j = 0; j < signa_boudaries.size(); j++)
    {
        int begin = signa_boudaries[j];
        int stop = noise_boudaries[j];

        if (stop - begin < 5)
            continue;

        // std::cout<<"Step 1, from "<<begin<<" to "<<stop<<std::endl;

        std::vector<int> peak_positions,min_positions;
        std::vector<float> peak_amplitudes;
        for(int k=begin+2;k<stop-2;k++)
        {   
            if(spect[k] > noise_level*user_scale && spect[k]>spect[k-2] &&spect[k]>spect[k-1] && spect[k]>spect[k+1]&& spect[k]>spect[k+2])
            {
                peak_positions.push_back(k);
                peak_amplitudes.push_back(spect[k]);
                continue;
            }
        }
        if(peak_positions.size()==0) continue;
        for(int k=0;k<peak_positions.size()-1;k++)
        {
            int p=peak_positions[k];
            float v=peak_amplitudes[k];
            for(int m=peak_positions[k];m<peak_positions[k+1];m++)
            {
                if(spect[m]<v)
                {
                    v=spect[m];
                    p=m;
                }
            }
            min_positions.push_back(p);
        }
        min_positions.push_back(stop);

        //now we have N peaks (peak_positions) and N-1 minimal points (min_positions) between them; min_positions also have one addtional element = stop
        std::vector<int> bs=ldw_math::get_best_partition(peak_amplitudes,0.2);
        bs.push_back(peak_amplitudes.size());

        for(int k=0;k<bs.size();k++)
        {
            int b,s;
            if(k==0) b=begin;
            else b=min_positions[bs[k-1]-1];
            s=min_positions[bs[k]-1];
            final_segment_begin.push_back(b);
            final_segment_stop.push_back(s);
            // std::cout<<"Cut into "<<b<<" - "<<s<<std::endl;
        }

    }

#ifdef LDW_DEBUG
    std::ofstream foutd("picked_peaks_1d.tab.debug");
#endif
    for (int j = 0; j < final_segment_begin.size(); j++)
    {
        int begin = final_segment_begin[j];
        int stop = final_segment_stop[j];
        int begin0=begin;

        std::vector<float> t;
        
        int ndim=spect.size();
        int left_patch=0;
        int right_patch=0;
        int n=stop-begin+40;

        left_patch=std::max(0,20-begin);
        begin=std::max(0,begin-20);
        right_patch=std::max(0,20-(ndim-stop));
        stop=std::min(ndim,stop+20);

        std::vector<float> data;
        data.clear();
        data.resize(n,0.0f);
        std::copy(spect.begin()+begin, spect.begin()+stop, data.begin()+left_patch);

        p1.predict(data);
        p1.predict_step2();

#ifdef LDW_DEBUG
        int nn = p1.output1.size() / 3;
        for (int k = 20; k < nn -20 ; k++)
        {
            foutd << begin0 + k - 20 << " " << p1.output1[k * 3] << " " << p1.output1[k * 3 + 1] << " " << p1.output1[k * 3 + 2] << std::endl;
        }
#endif

        for (int k = 0; k < p1.posits.size(); k++)
        {
            a.push_back(p1.intens[k]);
            x.push_back(p1.posits[k]+begin0);
            confidence.push_back(p1.confidence[k]);
            sigmax.push_back(p1.sigmas[k]);
            gammax.push_back(p1.gammas[k]);
        }
        
    }

#ifdef LDW_DEBUG
    foutd.close();
#endif

    //fitler out  peaks below cutoff
    for (int k = a.size() - 1; k >= 0; k--)
    {
        if (a[k] < noise_level * user_scale)
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
    for(int k=0;k<a.size();k++)
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
        float s1 = 0.5346 * sigmax[k] * 2 + std::sqrt(0.2166 * 4 * gammax[k] * gammax[k] + sigmax[k] * sigmax[k] * 8 * 0.6931);
        fprintf(fp, "%5d %9.3f %9.4f %7.3f %+e %4.3f\n", k + 1, x[k], cs, s1, a[k], confidence[k]);
    }

    fclose(fp);
    return true;
}

bool spectrum_pick_1d::get_peaks(struct spectrum_1d_peaks &y )
{
    y.a=a;
    y.x=x;
    y.sigmax=sigmax;
    y.gammax=gammax;
    y.confidence=confidence;
    y.intens=a;
    return true;
};