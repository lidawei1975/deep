#include <string>
#include <vector>
#include <algorithm>

#include "json/json.h"
#include "commandline.h"
#include "spectrum_phasing_1d.h"

#include "fid_1d.h"
#include "DeepConfig.h"

namespace phasing_1d_helper
{
    size_t split(const std::string &txt, std::vector<std::string> &strs, char ch)
    {
        size_t pos = txt.find(ch);
        size_t initialPos = 0;
        strs.clear();

        // Decompose statement
        while (pos != std::string::npos)
        {
            strs.push_back(txt.substr(initialPos, pos - initialPos));
            initialPos = pos + 1;

            pos = txt.find(ch, initialPos);
        }

        // Add the last one
        strs.push_back(txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1));

        return strs.size();
    }
};


int main(int argc, char **argv)
{ 
    std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;

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
    args3.push_back("input spectral file names. none or null means we will use -fid_in to read fid file instead.");

    args.push_back("-fid_in");
    args2.push_back("fid acqus");
    args3.push_back("input fid file name and acqus file name, separated by space.");

    args.push_back("-out");
    args2.push_back("test-phased.ft1");
    args3.push_back("output spectral file names.");

    args.push_back("-out-json");
    args2.push_back("fid-information.json");
    args3.push_back("output json file name for spectral information");

    args.push_back("-user");
    args2.push_back("no");
    args3.push_back("use user defined phase correction.");

    args.push_back("-user_phase");
    args2.push_back("0.0 0.0");
    args3.push_back("user defined phase correction at the left and right ends, separated by space.");

    args.push_back("-flip");
    args2.push_back("no");
    args3.push_back("flip spectrum (yes,no,auto). auto means we will decide based on the positivity of spectrum.");

    args.push_back("-n_loop");
    args2.push_back("10");
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

    args.push_back("-b_smooth_baseline");
    args2.push_back("yes");
    args3.push_back("at last, adjust phase correction to make baseline smooth (minimal entropy).");

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
        bool b_smooth_baseline = cmdline.query("-b_smooth_baseline") == "yes" || cmdline.query("-b_smooth_baseline") == "Yes" || cmdline.query("-b_smooth_baseline") == "YES" || cmdline.query("-b_smooth_baseline") == "y" || cmdline.query("-b_smooth_baseline") == "Y";

        /**
         * b_smooth_baseline will be set to off when b_end is off
        */
        if(b_end == false)
        {
            b_smooth_baseline = false;
        }

        class spectrum_phasing_1d x;
        x.init(10.0, 3.0, 0.0); // this function is required by spectrum_io, but not used in this program

        /**
         * check whether we need to read fid file. We do if infname is "null" or "none" or their capital letters
        */
        std::transform(infname.begin(), infname.end(), infname.begin(), ::tolower);

        if(infname == "null" || infname == "none")
        {
            std::string fid_infname = cmdline.query("-fid_in");
            std::vector<std::string> fid_infname_split;
            if(phasing_1d_helper::split(fid_infname,fid_infname_split,' ') < 2)
            {
                std::cout << "Error: -fid_in should be followed by at least two file names, separated by space." << std::endl;
                return 1;
            }

            /**
             * Last file name is acqus file name
            */
            std::string acqus_file_name = fid_infname_split.back();    

            /**
             * Remove the last file name from the vector. Others are fid file names 
            */  
            fid_infname_split.pop_back();     


            /**
             * define a fid_1d object to read fid file
            */
            apodization apod("kaiser 0.5 0.896 3.684");
            
            fid_1d fid;
            fid.set_up_apodization(&apod);
            fid.read_bruker_acqus_and_fid(acqus_file_name, fid_infname_split);
            fid.run_zf(2);
            fid.run_fft_and_rm_bruker_filter();

            /**
             * need to run this function to get the header 
             * without writing any file to disk
            */
            fid.write_nmrpipe_ft1(""); 

            /**
             * write some spectrum infor to a json file
             * This is mainly for the webserver to use
            */
            std::string outfname_json=cmdline.query("-out-json");
            std::cout<<"Write json file: " << outfname_json << std::endl;
            fid.write_json(outfname_json);

            /**
             * get header and spectrum_real, spectrum_imag from fid_1d object and send them to spectrum_phasing_1d object
            */
           x.direct_set_spectrum_from_nmrpipe(fid.get_spectrum_header(), fid.get_spectrum_real(), fid.get_spectrum_imag());

        }
        else
        {
            x.read_spectrum(infname);  //allow negative spectrum by default
        }

        /**
         * if user defined phase correction is used, we will use it and skip auto phase correction
        */
        bool b_user= cmdline.query("-user") == "yes" || cmdline.query("-user") == "Yes" || cmdline.query("-user") == "YES" || cmdline.query("-user") == "y" || cmdline.query("-user") == "Y";
        
        if(b_user)
        {
            std::vector<std::string> user_phase_string_split;
            std::string user_phase_string = cmdline.query("-user_phase");
            if(phasing_1d_helper::split(user_phase_string,user_phase_string_split,' ') != 2)
            {
                std::cout << "Error: -user_phase should be followed by two numbers, separated by space." << std::endl;
                return 1;
            }
            double user_phase_left = std::stod(user_phase_string_split[0]);
            double user_phase_right = std::stod(user_phase_string_split[1]);
            x.phase_spectrum(user_phase_left, user_phase_right);  //unit is degree !!

            std::string flip=cmdline.query("-flip");
            /**
             * Get first element of flip 
            */
            char flip_char=flip[0];
            if(flip_char == 'y' || flip_char == 'Y')
            {
                x.flip_spectrum();
            }
            else if(flip_char == 'a' || flip_char == 'A')
            {
                x.auto_flip_spectrum();
            }
            else
            {
                //do nothing
            }

        }
        else
        {
            x.set_up_parameters(n_loop, n_peak, n_dist,b_end,b_smooth_baseline);
            x.auto_phase_correction();
            std::array<double, 2> phase_correction = x.get_phase_correction();
            std::cout << "phase correction left: " << phase_correction[0] << " right: " << phase_correction[1] << std::endl;
        }
        x.write_spectrum(outfname);


    }

    return 0;
}