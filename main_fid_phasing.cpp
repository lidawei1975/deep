/**
 * This program will run FID processing and phase correction to generate a .ft2 file 
*/

#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "commandline.h"
#include "fid_2d.h"
#include "spectrum_phasing.h"

#include "DeepConfig.h"

namespace phasing_helper
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
    args2.push_back("arguments_fid_phasing.txt");
    args3.push_back("Arguments file");

    args.push_back("-in");
    args2.push_back("ser acqus acqu2s");
    args3.push_back("input fid,acqus, and acqus2, separated by space(s)");

    args.push_back("-aqseq");
    args2.push_back("321");
    args3.push_back("acquisition sequence in pseudo-3D exp, 321 or 312 only. N.A. if not pseudo-3D exp.");

    args.push_back("-negative");
    args2.push_back("no");
    args3.push_back("negative image data along indirect dimension?");
    
    args.push_back("-out");
    args2.push_back("test.ft2"); 
    args3.push_back("output frq domain file name");

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

    args.push_back("-user");
    args2.push_back("no");
    args3.push_back("use user defined phase correction instead of automatic phase correction.");

    args.push_back("-user_phase");
    args2.push_back("0.0 0.0 0.0 0.0");
    args3.push_back("user defined phase correction in degree P0,P1 (direct) P0,P1 (indirect).");

    args.push_back("-out_phase");
    args2.push_back("phase-correction.txt");
    args3.push_back("output file name phase correction values.");

    args.push_back("-real_only");
    args2.push_back("yes");
    args3.push_back("only save real spectrum.");


    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        fid_2d fid;
        std::string fid_infname = cmdline.query("-in");
        std::string aqseq = cmdline.query("-aqseq");
        
        if(fid.set_aqseq(aqseq) == false)
        {
            return 1;
        }

        bool b_negative = cmdline.query("-negative")[0] == 'y' || cmdline.query("-negative")[0] == 'Y';
        fid.set_negative(b_negative);

        /**
         * To run phasing, we only need to process the first spectrum in pseudo-3D experiment.
        */
        fid.set_first_only(true);

        bool b_read_fid = false;

        std::vector<std::string> fid_infname_split;
        int n_fileds = fid_1d_helper::split(fid_infname, fid_infname_split, ' ');
        if (n_fileds >= 3)
        {
            /**
             * Last file name is acqu2s
             * 2rd last file name is acqus
             * others are fid file name(s)
             */
            std::string acqus_file2_name = fid_infname_split.back();
            fid_infname_split.pop_back();
            std::string acqus_file_name = fid_infname_split.back();
            fid_infname_split.pop_back();
            b_read_fid = fid.read_bruker_files("none",acqus_file2_name, acqus_file_name, fid_infname_split);
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


        fid.full_process(); //this step will also call remove_bruker_digitizer_filter()
        std::cout<<"Run FFT and remove Bruker digitizer filter." << std::endl;


        /**
         * Extract ft2 file as a virtual file
        */
        std::array<float,512> header;
        std::vector<float> data;
        fid.write_nmrpipe_ft2_virtual(header, data);
        // fid.write_nmrpipe_ft2("temp.ft2",false);

        
        class spectrum_phasing phase_obj;
        /**
         * Baseclass spectrum_io function to read spectrum, without noise estimation (which is not good for un-phased spectrum)
        */
        phase_obj.read_nmr_ft2_virtual(header,data);
        
        // class spectrum_phasing phase_obj2;
        // phase_obj2.init("temp.ft2",false);
        // phase_obj.assess_match(phase_obj2);

        if (cmdline.query("-user").substr(0, 1) == "y" || cmdline.query("-user").substr(0, 1) == "Y")
        {
            std::vector<std::string> strs;
            std::string str = cmdline.query("-user_phase");
            phasing_helper::split(str, strs, ' ');
            if (strs.size() != 4)
            {
                std::cout << "Error: user phase correction must have 4 values!" << std::endl;
                return -1;
            }
            if(std::stod(strs[3]) != 0.0)
            {
                std::cout << "Warning: 1st order phase correction for indirect dimension is usually 0." << std::endl;
            }
            phase_obj.set_user_phase_correction(std::stod(strs[0]), std::stod(strs[1]), std::stod(strs[2]), std::stod(strs[3]));
        }
        else
        {
            /**
             * main working function
            */
            phase_obj.auto_phase_correction_v2();
            std::cout<<"Auto phase correction done!"<<std::endl;
        }

        /**
         * Save phased spectrum to a file
        */
        std::string outfname=cmdline.query("-out");
        if(outfname != "none" && outfname != "null")
        {
            std::cout<<"Write FT2 file: " << outfname << std::endl;
            phase_obj.write_pipe(outfname,cmdline.query("-real_only")[0] == 'y' || cmdline.query("-real_only")[0] == 'Y');
            std::cout<<"Phased spectrum saved to "<<outfname<<std::endl;
        }

        /**
         * Save phase correction values to a file
        */
        std::string out_phase_fname = cmdline.query("-out_phase");
        if(out_phase_fname != "none" && out_phase_fname != "null")
        {
            phase_obj.save_phase_correction_result(out_phase_fname);
            std::cout<<"Phase correction values saved to "<<out_phase_fname<<std::endl;
        }

    }

    return 0;
}