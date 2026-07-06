#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <set>

#include "json/json.h"
#include "commandline.h"
#include "spectrum_simulation.h"
#include "DeepConfig.h"

int main(int argc, char ** argv)
{
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-stdin");
    args2.push_back("yes");
    args3.push_back("stdin mode (yes), read all input from stdin and all output to stdout");

    args.push_back("-f");
    args2.push_back("arguments_spin_simulation.txt");
    args3.push_back("read arguments from file");

    args.push_back("-in");
    args2.push_back("bmse000408_1.txt");
    args3.push_back("input spin system simulation intermediate file");

    args.push_back("-rev");
    args2.push_back("revision.json");
    args3.push_back("CS revision file");

    args.push_back("-out");
    args2.push_back("simulated-spectrum.txt");
    args3.push_back("output simulated spectrum file");

    args.push_back("-r2");
    args2.push_back("4.0");
    args3.push_back("r2 relaxation rate");

    args.push_back("-b0");
    args2.push_back("850.0");
    args3.push_back("magnetic field strength in MHz");

    args.push_back("-n_acq");
    args2.push_back("32k");
    args3.push_back("number of complex data points in FID");

    args.push_back("-zf");
    args2.push_back("2");
    args3.push_back("zero filling fold (2,4,8,16)");

    args.push_back("-width");
    args2.push_back("16.0");
    args3.push_back("Spectrum width in ppm");

    args.push_back("-apod");
    args2.push_back("kaiser 0.5 0.896 3.684");
    args3.push_back("Apodization function and parameters");


    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    bool stdin_mode = cmdline.query("-stdin")[0] == 'Y' || cmdline.query("-stdin")[0] == 'y';

    double r2,b0,spectral_width_ppm,delta_t;
    int n_acq,n_zf,n_acq2,ndim;
    std::string n_acq_string;
    std::string apod;
    std::string input_file;
    std::vector<double> cs, height;
    std::vector<int> proton_index;
    std::vector<double> cs_revision;
    std::vector<int> proton_index_new;

    if (!stdin_mode)
    {
        std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;
        std::cout<<"This program will read spin simulation intermediate file and genrate theoritical spectrum"<<std::endl;
        cmdline.print();
    }
    
    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    input_file = cmdline.query("-in");
    r2 = std::stod(cmdline.query("-r2"));
    b0 = std::stod(cmdline.query("-b0"));
    spectral_width_ppm = std::stod(cmdline.query("-width"));

    /**
     * Number of complex data points in FID.
     * Could be
     * 1. a number
     * 2. a number followed by k (kilo), m (mega)
     */
    n_acq_string = cmdline.query("-n_acq");
    n_zf = std::stoi(cmdline.query("-zf"));
    apod = cmdline.query("-apod");

    if (n_acq_string.find("k") != std::string::npos)
    {
        n_acq_string = n_acq_string.substr(0, n_acq_string.size() - 1);
        n_acq = std::stoi(n_acq_string) * 1024;
    }
    else if (n_acq_string.find("m") != std::string::npos)
    {
        n_acq_string = n_acq_string.substr(0, n_acq_string.size() - 1);
        n_acq = std::stoi(n_acq_string) * 1024 * 1024;
    }
    else
    {
        n_acq = std::stoi(n_acq_string);
    }

    /**
     * Unit of spectral width is ppm, unit of b0 is MHz
     * so, delta_t has a unit of second
    */
    delta_t = 1.0 / (b0 * spectral_width_ppm);

    /**
     * Read spin simulation intermediate file
     *
     * Example spin simulation intermediate file.
     * 3 columns: ppm, height, proton index (integer in float format)
     * NOTE: magnetic equivalent protons have the same proton index in this file.
     *  2.2559072e+00   9.8348424e-02   1.5000000e+01
     *  2.2463358e+00   9.8159740e-02   1.5000000e+01
     *  2.2159291e+00   9.7966879e-02   1.4000000e+01
     */
    std::ifstream fin(input_file);
    if (!fin.is_open())
    {
        std::cerr << "Error: cannot open file " << input_file << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(fin, line))
    {
        std::istringstream iss(line);
        double c, h;
        double p; 
        iss >> c >> h >> p;
        cs.push_back(c);
        height.push_back(h);
        proton_index.push_back(int(p));
    }
    fin.close();


    Json::Value root;
    std::string revision_string;
    if(stdin_mode)
    {
        /**
         * Read revision file from stdin as a json string
         * root will be a array of [cs_rev,proton_index]
        */
        std::string revision_string;
        std::cin>>revision_string;
        std::istringstream iss(revision_string);

        Json::CharReaderBuilder jsonReader;
        std::string errs;
        Json::parseFromStream(jsonReader, iss, &root, &errs);
    }
    else 
    {
        /**
         * Read revision file, also a json string
        */
        std::string revision_file = cmdline.query("-rev");
        std::ifstream fin2(revision_file);
        fin2>>root;
        fin2.close();
    }


    for(int i=0;i<root.size();i++)
    {
        cs_revision.push_back(root[i][0].asDouble());
        proton_index_new.push_back(root[i][1].asInt());
    }

    /**
     * For each cs_revision, proton_index_new pair, find the corresponding cs, proton_index pair 
     * in the original spin simulation intermediate file (could be multiple lines for the same proton index)
     * and add cs_revision to cs
    */
    for(int i=0;i<cs_revision.size();i++)
    {
        /**
         * Skip if cs_revision is 0
        */
        if(fabs(cs_revision[i])<1e-6)
        {
            continue;
        }
        for(int j=0;j<cs.size();j++)
        {
            if(proton_index[j]==proton_index_new[i])
            {
                cs[j] += cs_revision[i];
            }
        }
    }


    /**
     * Get nearest 2^n for n_acq
     */
    n_acq2 = 1;
    while (n_acq2 < n_acq)
    {
        n_acq2 *= 2;
    }
    ndim = n_acq2 * n_zf;

    /**
     * Generate ppm scale form 0 to spectral_width_ppm.
     * IMPORTANT: If want inverse ppm scale, just reverse the ppm vector and negative fid_imag before run_fft
    */
    std::vector<float> ppm0; 
    for (int i = 0; i < ndim; i++)
    {
        ppm0.push_back(i * spectral_width_ppm / ndim); // [0 spectral_width_ppm] 
    }

    /**
     * Find a unique set of proton index
    */
    std::set<int> proton_index_set(proton_index.begin(), proton_index.end());

    std::vector<std::vector<float>> spectrum_real_all, spectrum_imag_all;
    std::vector<std::vector<float>> ppm_all;

    /**
     * Run simulation for each unique proton index
    */
    for (auto it = proton_index_set.begin(); it != proton_index_set.end(); it++)
    {
        int cuurent_proton_index = *it;
        std::vector<double> current_cs, current_height;
        for(int i=0;i<proton_index.size();i++)
        {
            if(proton_index[i]==cuurent_proton_index)
            {
                current_cs.push_back(cs[i]);
                current_height.push_back(height[i]);
            }
        }
        
        spectrum_simulation sim;
        std::vector<float> fid_real, fid_imag;
        std::vector<float> spectrum_real, spectrum_imag;
        std::vector<float> ppm = ppm0;
        sim.simualte_fid(n_acq, current_cs, current_height, b0, delta_t, r2, apod, fid_real, fid_imag);
        sim.run_fft(n_acq, ndim, spectrum_real, spectrum_imag, fid_real, fid_imag, 0);

        spectrum_real_all.push_back(spectrum_real);
        spectrum_imag_all.push_back(spectrum_imag);
        ppm_all.push_back(ppm);
    }

    /**
     * Create a total spectrum by adding all spectra together
    */
    std::vector<float> spectrum_real_total(ndim, 0);
    std::vector<float> spectrum_imag_total(ndim, 0);
    for(int k=0;k<spectrum_real_all.size();k++)
    {
        for(int i=0;i<spectrum_real_all[k].size();i++)
        {
            spectrum_real_total[i] += spectrum_real_all[k][i];
            spectrum_imag_total[i] += spectrum_imag_all[k][i];
        }
    }
    spectrum_real_all.push_back(spectrum_real_total);
    spectrum_imag_all.push_back(spectrum_imag_total);
    ppm_all.push_back(ppm0);


    /**
     * Get the max amplitudes of spectrum_real_all
    */
    float max_real_total = *std::max_element(spectrum_real_total.begin(), spectrum_real_total.end());


    /**
     * For each spectrum, remove part of the spectrum with real value less than 0.1% of max_real_total
    */
    for(int k=0;k<spectrum_real_all.size();k++)
    {
        for (int i = spectrum_real_all[k].size() - 1; i >= 0; i--)
        {
            if (spectrum_real_all[k][i] < 0.001 * max_real_total)
            {
                spectrum_real_all[k].erase(spectrum_real_all[k].begin() + i);
                spectrum_imag_all[k].erase(spectrum_imag_all[k].begin() + i);
                ppm_all[k].erase(ppm_all[k].begin() + i);
            }
        }
    }


    /**
     * Construct a json object for output
     * root["spectra"][i][j] = [ppm, real, imag], where i is the index of proton index, j is the index of ppm
     * root["proton_index"] = [proton_index]
    */
    Json::Value root_output;
    root_output["spectra"] = Json::Value(Json::arrayValue);
    for (int k = 0; k < spectrum_real_all.size(); k++)
    {
        root_output["spectra"][k] = Json::Value(Json::arrayValue);
        for (int i = 0; i < spectrum_real_all[k].size(); i++)
        {
            Json::Value item = Json::Value(Json::arrayValue);
            item[0] = ppm_all[k][i];
            item[1] = spectrum_real_all[k][i];
            item[2] = spectrum_imag_all[k][i];
            root_output["spectra"][k].append(item);
        }
    }
    for (auto it = proton_index_set.begin(); it != proton_index_set.end(); it++)
    {
        root_output["proton_index"].append(*it);
    }
    root_output["proton_index"].append(-1); // total spectrum index


    /**
     * Output simulated spectrum, may contains mutiple file names, separated by space
    */
    if(stdin_mode==true)
    {
       
        std::cout << root_output;
    }
    else
    {
        std::string output_file_string = cmdline.query("-out");
        std::vector<std::string> output_files;

        spectrum_simulation_helper::split(output_file_string, output_files, ' ');

        for(int i=0;i<output_files.size();i++)
        {
            /**
             * Skip of output file is none or null or NONE or NULL
            */
            if(output_files[i]=="none" || output_files[i]=="null" || output_files[i]=="NONE" || output_files[i]=="NULL")
            {
                continue;
            }

            std::string output_file = output_files[i];
            std::ofstream fout(output_file);
        
            /**
             * Get file extension of output_file
            */
            std::string file_extension = output_file.substr(output_file.find_last_of(".") + 1);
            if(file_extension=="json")
            {
                fout << root_output;
            }
            /**
             * All other cases, write a text file with 3 columns: ppm, real, imag
             * block by block for each proton index
            */
            else
            {   
                for(int k=0;k<ppm_all.size();k++)
                {
                    for (int i = 0; i < ndim; i++)
                    {
                        fout << ppm_all[k][i] << " " << spectrum_real_all[k][i] << " " << spectrum_imag_all[k][i] << std::endl;
                    }
                    fout << std::endl; // empty line as block separator
                }
            }
            fout.close();
        }
    }

    return 0;
}