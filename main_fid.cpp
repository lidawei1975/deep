#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "commandline.h"
#include "fid_2d.h"


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
    args2.push_back("arguments_fid_2d.txt");
    args3.push_back("Arguments file");

    args.push_back("-in");
    args2.push_back("ser acqus acqu2s pulseprogram");
    args3.push_back("input fid,acqus,acqus2 and pulseprogram file names, separated by space(s) or *.ft1 or a folder name.");

    args.push_back("-out");
    args2.push_back("test.ft2"); 
    args3.push_back("output frq domain file name");

    args.push_back("-out-json");
    args2.push_back("fid-information.json");
    args3.push_back("output json file name for spectral information");

    args.push_back("-apod");
    args2.push_back("kaiser 0.5 0.896 3.684");
    args3.push_back("apodization function name and parameters, separated by space(s).");

    args.push_back("-apod-indirect");
    args2.push_back("kaiser 0.5 0.896 3.684");
    args3.push_back("apodization function name and parameters for indirect dimension, separated by space(s).");

    args.push_back("-zf");
    args2.push_back("2");
    args3.push_back("zero filling factor(2,4,8,16).");  

    args.push_back("-zf_indirect");
    args2.push_back("2");
    args3.push_back("zero filling factor(2,4,8,16) for indirect dimension.");

    args.push_back("-phase_in");
    args2.push_back("phase-correction.txt");
    args3.push_back("input phase correction file name. none means no phase correction.");

    args.push_back("-real_only");
    args2.push_back("no");
    args3.push_back("only save real spectrum.");


    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        fid_2d fid;
        std::string fid_infname = cmdline.query("-in");
        bool b_read_fid = false;

        std::vector<std::string> fid_infname_split;
        int n_fileds = fid_1d_helper::split(fid_infname, fid_infname_split, ' ');
        if (n_fileds >= 4)
        {
            /**
             * Last file name is pulseprogram
             * second last file name is acqu2s
             * 3rd last file name is acqus
             * others are fid file name(s)
             */
            std::string pulseprogram_file_name = fid_infname_split.back();
            fid_infname_split.pop_back();
            std::string acqus_file2_name = fid_infname_split.back();
            fid_infname_split.pop_back();
            std::string acqus_file_name = fid_infname_split.back();
            fid_infname_split.pop_back();
            b_read_fid = fid.read_bruker_files(pulseprogram_file_name,acqus_file2_name, acqus_file_name, fid_infname_split);
        }
        else if (n_fileds == 1)
        {
            /**
             * If ends with .fid, read nmrPipe .ft1 file, 
             * if there is no ., read bruker folder
             * else do nothing
             */
            std::string ft1(".fid");
            if(std::equal(ft1.rbegin(), ft1.rend(), fid_infname_split[0].rbegin()))
            {
                b_read_fid = fid.read_nmrpipe_fid(fid_infname_split[0]);
            }
            else if(fid_infname_split[0].find(".") == std::string::npos)
            {
                b_read_fid = fid.read_bruker_folder(fid_infname_split[0]);
            }
            else
            {
                b_read_fid = false;
            }
        }

        if(!b_read_fid)
        {
            std::cout << "Error: read fid file failed." << std::endl;
            return 1;
        }


        std::string apodization_string=cmdline.query("-apod");
        std::string apodization_string_indirect=cmdline.query("-apod-indirect");

        apodization apod1(apodization_string);
        apodization apod2(apodization_string_indirect);

        fid.set_up_apodization(&apod1,&apod2);
       

        int n_zf=std::stoi(cmdline.query("-zf"));
        int n_zf_indirect=std::stoi(cmdline.query("-zf_indirect"));
        /**
         * make sure n_zf is 2,4,8,16
         */
        if(n_zf != 2 && n_zf != 4 && n_zf != 8 && n_zf != 16)
        {
            std::cout << "Error: n_zf must be 2,4,8,16." << std::endl;
            return 1;
        }
        if(n_zf_indirect != 2 && n_zf_indirect != 4 && n_zf_indirect != 8 && n_zf_indirect != 16)
        {
            std::cout << "Error: n_zf_indirect must be 2,4,8,16." << std::endl;
            return 1;
        }

        fid.run_zf(n_zf,n_zf_indirect);
        std::cout<<"Apply zero filling factor: " << n_zf << std::endl;

        if(cmdline.query("-phase_in") != "none" && cmdline.query("-phase_in") != "null")
        {
            std::string phase_infname=cmdline.query("-phase_in");
            std::cout<<"Read phase correction file: " << phase_infname << std::endl;
            fid.read_phase_correction(phase_infname);
        }

        fid.run_fft_and_rm_bruker_filter(); //this step will also call remove_bruker_digitizer_filter()
        std::cout<<"Run FFT and remove Bruker digitizer filter." << std::endl;

        bool b_real_only=cmdline.query("-real_only")[0] == 'y' || cmdline.query("-real_only")[0] == 'Y';

        std::string outfname=cmdline.query("-out");
        std::cout<<"Write FT2 file: " << outfname << std::endl;
        fid.write_nmrpipe_ft2(outfname,b_real_only);

        // std::string outfname_json=cmdline.query("-out-json");
        // std::cout<<"Write json file: " << outfname_json << std::endl;
        // fid.write_json(outfname_json);
    }

    return 0;
}