
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>



#include "commandline.h"
#include "dnn_picker.h"


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

class spectrum_picking_1d
{
private:
    double user_scale,user_scale2;
    double noise_level;
    int mod_selection;
    std::vector<float> spe;  //1d spectrum
    class peak1d p1;

public:
    spectrum_picking_1d();
    ~spectrum_picking_1d();
    bool init(double,double,double,int);
    bool read_spectrum(std::string);
    bool work(std::string outfname);
    bool work2(std::string outfname);
};

spectrum_picking_1d::spectrum_picking_1d() {
    noise_level=0.0;
    mod_selection=2;
};
spectrum_picking_1d::~spectrum_picking_1d() {};


bool spectrum_picking_1d::init(double user_,double user2_,double noise_, int mod)
{
    user_scale=user_;
    user_scale2=user_;
    
    mod_selection=mod;

    if(noise_>1e-20)
    {
        noise_level=noise_;
    }


    if(mod_selection==1){
        p1.load();
    }
    else
    {
        p1.load_m2();
    }
    

    return true;
};

bool spectrum_picking_1d::read_spectrum(std::string infname)
{
    std::ifstream fin(infname);

    float data;
    while(fin>>data)
    {
        spe.push_back(data);
    }
    fin.close();

    int ndim=spe.size();

    if(noise_level<1e-20)  //estimate noise level
    {
        std::vector<float> t=spe;
        for(unsigned int i = 0; i < t.size(); i++)
        {
            if(t[i] < 0)
                t[i] *= -1;
        }

        std::vector<float> scores=t;
        sort(scores.begin(), scores.end());
        noise_level = scores[scores.size() / 2]*1.4826;
        if(noise_level<=0.0) noise_level=0.1; //artificail spectrum w/o noise
        std::cout<<"First round, noise level is "<<noise_level<<std::endl;

        std::vector<int> flag(ndim,0); //flag

        for (int i = 0; i < ndim; i++)
        {
            if (t[i] > 5.5 * noise_level)
            {
                int xstart = std::max(i - 5, 0);
                int xend = std::min(i + 6, ndim);

                for (int n = xstart; n < xend; n++)
                {
                    flag[n] = 1;
                }
            }
        }
        scores.clear();

        for (int i = 0; i < ndim; i++)
        {
            if (flag[i] == 0)
            {
                scores.push_back(t[i]);
            }
        }

        sort(scores.begin(), scores.end());
        noise_level = scores[scores.size() / 2] * 1.4826;
        if (noise_level <= 0.0)
            noise_level = 0.1; //artificail spectrum w/o noise
        std::cout << "Final noise level is estiamted to be " << noise_level << std::endl;
    }

    // for (unsigned int i = 0; i < spe.size(); i++)
    // {
    //     if (spe[i] < 0)
    //         spe[i] = 0.0f;
    // }

    return true;
};

//predict whole line at once. not currently used anywhere
//not suitable for metabolomics (peaks can have huge intensity difference)
bool spectrum_picking_1d::work(std::string outfname)
{
    p1.set_noise_level(noise_level);
    p1.predict(spe);
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

//segment by segment prediction, as in peak2d
bool spectrum_picking_1d::work2(std::string outfname)
{
    p1.set_noise_level(noise_level);
    int ndim=spe.size();
    double boundary_cutoff=noise_level*user_scale2;

    std::ofstream fout(outfname);
    std::ofstream foutd(outfname+".debug");

    std::vector<int> signa_boudaries;
    std::vector<int> noise_boudaries;

    std::vector<int> final_segment_begin,final_segment_stop;

    int c=0;

    if (spe[0] > boundary_cutoff)
    {
        signa_boudaries.push_back(0);
    }
    for (int j = 1; j < ndim; j++)
    {
        if (spe[j - 1] <= boundary_cutoff && spe[j] > boundary_cutoff)
        {
            signa_boudaries.push_back(std::max(j - 10, 0));
        }
        else if (spe[j - 1] > boundary_cutoff && spe[j] <= boundary_cutoff)
        {
            noise_boudaries.push_back(std::min(j + 10, ndim));
        }
    }
    if (noise_boudaries.size() < signa_boudaries.size())
    {
        noise_boudaries.push_back(ndim);
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

    for (int j = 0; j < signa_boudaries.size(); j++)
    {
        int begin = signa_boudaries[j];
        int stop = noise_boudaries[j];

        if (stop - begin < 5)
            continue;

        std::cout<<"Step 1, from "<<begin<<" to "<<stop<<std::endl;

        std::vector<int> peak_positions,min_positions;
        std::vector<float> peak_amplitudes;
        for(int k=begin+2;k<stop-2;k++)
        {   
            if(spe[k] > noise_level*user_scale && spe[k]>spe[k-2] &&spe[k]>spe[k-1] && spe[k]>spe[k+1]&& spe[k]>spe[k+2])
            {
                peak_positions.push_back(k);
                peak_amplitudes.push_back(spe[k]);
                continue;
            }
        }

        for(int k=0;k<peak_positions.size()-1;k++)
        {
            int p=peak_positions[k];
            float v=peak_amplitudes[k];
            for(int m=peak_positions[k];m<peak_positions[k+1];m++)
            {
                if(spe[m]<v)
                {
                    v=spe[m];
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
            std::cout<<"Cut into "<<b<<" - "<<s<<std::endl;
        }

    }

    for (int j = 0; j < final_segment_begin.size(); j++)
    {
        int begin = final_segment_begin[j];
        int stop = final_segment_stop[j];
        int begin0=begin;

        // std::cout<<"Step 2, from "<<begin<<" to "<<stop<<std::endl;

        std::vector<float> t;
        

        int ndim=spe.size();
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
        std::copy(spe.begin()+begin, spe.begin()+stop, data.begin()+left_patch);

        p1.predict(data);
        p1.predict_step2();

        int nn = p1.output1.size() / 3;
        for (int k = 20; k < nn -20 ; k++)
        {
            foutd << begin0 + k - 20 << " " << p1.output1[k * 3] << " " << p1.output1[k * 3 + 1] << " " << p1.output1[k * 3 + 2] << std::endl;
        }

        for (int k = 0; k < p1.posits.size(); k++)
        {
            fout<<c<<" "<<p1.posits[k]+begin0<<" "<<p1.centes[k]+begin0<<" "<<p1.intens[k]<<" "<<p1.sigmas[k]<<" "<<p1.gammas[k]<<" "<<p1.confidence[k]<<std::endl;  
            c++;
        }
    }

    fout.close();
    foutd.close();
    return true;
};



int main(int argc, char **argv)
{
 
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-scale");
    args2.push_back("5.5");
    args3.push_back("user defined noise level scale factor for peak picking (5.5)");

    args.push_back("-scale2");
    args2.push_back("3.0");
    args3.push_back("user defined noise level scale factor for peak fitting(3.0)");

    args.push_back("-noise_level");
    args2.push_back("164776");
    args3.push_back("Direct set noise level to this value, estimate from sepctrum if input is 0.0 (0.0)");

    args.push_back("-in");
    args2.push_back("input.ft");
    args3.push_back("input file name (input.ft)");

    args.push_back("-out");
    args2.push_back("peaks_1d.txt");
    args3.push_back("output file name (peaks_1d.txt)");

    args.push_back("-model");
    args2.push_back("1");
    args3.push_back("Model selection for ANN picker, 1: FWHH 6-20, 2: FWHH 4-12 (1)");


    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    std::string infname,outfname;
    double user,user2;
    double max_width;
    int model_selection;


    model_selection=atoi(cmdline.query("-model").c_str());
    if(model_selection!=1) model_selection=2;

    double noise_level=atof(cmdline.query("-noise_level").c_str());
    infname = cmdline.query("-in");
    outfname = cmdline.query("-out");
    user=atof(cmdline.query("-scale").c_str());
    user2=atof(cmdline.query("-scale2").c_str());
   
    
    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_picking_1d x;
        x.init(user,user2,noise_level,model_selection);
        if(x.read_spectrum(infname)) //read
        {
            x.work2(outfname); //picking and output
        }
    }
    return 0;
};