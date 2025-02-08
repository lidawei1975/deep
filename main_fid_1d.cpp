#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "commandline.h"
#include "fid_1d.h"


#include "DeepConfig.h"

int main(int argc, char **argv)
{ 
    std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;
 
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-f");
    args2.push_back("arguments_fid_1d.txt");
    args3.push_back("Arguments file");

    args.push_back("-in");
    args2.push_back("2");
    args3.push_back("input spectral experiment folder name; none or null means we will use -fid_in to read fid and acqus files instead.");

    args.push_back("-fid_in");
    args2.push_back("fid acqus");
    args3.push_back("input fid file name and acqus file name, separated by space(s).");

    args.push_back("-out");
    args2.push_back("test.ft1"); 
    args3.push_back("output frq domain file name");

    args.push_back("-out-json");
    args2.push_back("fid-information.json");
    args3.push_back("output json file name for spectral information");

    args.push_back("-apod");
    args2.push_back("sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 0.5");
    args3.push_back("apodization function name and parameters, separated by space(s).");

    args.push_back("-zf");
    args2.push_back("2");
    args3.push_back("zero filling factor(2,4,8,16).");  


    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        fid_1d fid;
        std::string infname = cmdline.query("-in");
        std::string infname_lower = infname;
        std::transform(infname_lower.begin(), infname_lower.end(), infname_lower.begin(), ::tolower);
        bool b_read_fid = false;

        if(infname_lower == "null" || infname_lower == "none")
        {
            std::string fid_infname = cmdline.query("-fid_in");
            std::vector<std::string> fid_infname_split;
            if(fid_1d_helper::split(fid_infname,fid_infname_split,' ') < 2)
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
            b_read_fid=fid.read_bruker_acqus_and_fid(acqus_file_name, fid_infname_split);
        }
        else
        {
            b_read_fid=fid.read_bruker_folder(infname);
        }

        if(!b_read_fid)
        {
            std::cout << "Error: read fid file failed." << std::endl;
            return 1;
        }

        //for debug. write fid to a file
        // fid.write_nmrpipe_fid("test.fid");

        std::string apodization_string=cmdline.query("-apod");

        apodization apod(apodization_string);
        fid.set_up_apodization(&apod);


        int n_zf=std::stoi(cmdline.query("-zf"));
        /**
         * make sure n_zf is 2,4,8,16
         */
        if(n_zf != 2 && n_zf != 4 && n_zf != 8 && n_zf != 16)
        {
            std::cout << "Error: n_zf must be 2,4,8,16." << std::endl;
            return 1;
        }

        fid.run_zf(n_zf);
        std::cout<<"Apply zero filling factor: " << n_zf << std::endl;

        fid.run_fft_and_rm_bruker_filter(); //this step will also call remove_bruker_digitizer_filter()
        std::cout<<"Run FFT and remove Bruker digitizer filter." << std::endl;

        std::string outfname=cmdline.query("-out");
        std::cout<<"Write FT1 file: " << outfname << std::endl;
        fid.write_nmrpipe_ft1(outfname);

        std::string outfname_json=cmdline.query("-out-json");
        std::cout<<"Write json file: " << outfname_json << std::endl;
        fid.write_json(outfname_json);
    }

    return 0;
}