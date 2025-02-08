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

    args.push_back("-nus");
    args2.push_back("nuslist");
    args3.push_back("non-uniform sampling list file name. None means no nus (fully sampled).");

    args.push_back("-aqseq");
    args2.push_back("321");
    args3.push_back("acquisition sequence in pseudo-3D exp, 321 or 312 only. N.A. if not pseudo-3D exp.");

    args.push_back("-negative");
    args2.push_back("no");
    args3.push_back("negative image data along indirect dimension?");
    
    args.push_back("-first-only");
    args2.push_back("no");
    args3.push_back("only process the first spectrum in pseudo-3D experiment.");

    args.push_back("-process");
    args2.push_back("full");
    args3.push_back("full: full process, direct: only process direct dimension, indirect: only process indirect dimension.");

    args.push_back("-out");
    args2.push_back("test.ft2"); 
    args3.push_back("output frq domain file name");

    args.push_back("-out-json");
    args2.push_back("fid-information.json");
    args3.push_back("output json file name for spectral information");

    args.push_back("-out-pseudo3d-json");
    args2.push_back("pseudo3d.json");
    args3.push_back("output json file name for pseudo 3D information");

    args.push_back("-ext");
    args2.push_back("0.00 1.00");
    args3.push_back("extract region from and to numbers, full is 0.0 1.0");


    args.push_back("-apod");
    args2.push_back("sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 0.5");
    args3.push_back("apodization function name and parameters, separated by space(s).");

    args.push_back("-apod-indirect");
    args2.push_back("sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 1.0");
    args3.push_back("apodization function name and parameters for indirect dimension, separated by space(s).");

    args.push_back("-zf");
    args2.push_back("2");
    args3.push_back("zero filling factor(2,4,8,16).");  

    args.push_back("-zf-indirect");
    args2.push_back("2");
    args3.push_back("zero filling factor(2,4,8,16) for indirect dimension.");

    args.push_back("-phase-in");
    args2.push_back("phase-correction.txt");
    args3.push_back("input phase correction file name. none means no phase correction.");

    args.push_back("-di");
    args2.push_back("yes");
    args3.push_back("delete imaginary part for direct dimension?");

    args.push_back("-di-indirect");
    args2.push_back("yes");
    args3.push_back("delete imaginary part for indirect dimension?");

    args.push_back("-water");
    args2.push_back("no");
    args3.push_back("remove water signal at carrier frequency?");

    args.push_back("-poly");
    args2.push_back("-1");
    args3.push_back("polynomial order for baseline correction. < 0 means no baseline correction.");


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

        /**
         * Negative image data along indirect dimension? effectively flip the indirect dimension
        */
        bool b_negative = cmdline.query("-negative")[0] == 'y' || cmdline.query("-negative")[0] == 'Y';
        fid.set_negative(b_negative);

        /**
         * For pseudo-3D experiment, only process the first spectrum in pseudo-3D experiment.
        */
        bool b_first_only = cmdline.query("-first-only")[0] == 'y' || cmdline.query("-first-only")[0] == 'Y';\
        fid.set_first_only(b_first_only);

        /**
         * Set up apoization
        */
        std::string apodization_string=cmdline.query("-apod");
        std::string apodization_string_indirect=cmdline.query("-apod-indirect");
        apodization apod1(apodization_string);
        apodization apod2(apodization_string_indirect);
        fid.set_up_apodization(&apod1,&apod2);

        /**
         * Set up delete imaginary data
        */
        bool b_di_direct = cmdline.query("-di")[0] == 'y' || cmdline.query("-di")[0] == 'Y';
        bool b_di_indirect = cmdline.query("-di-indirect")[0] == 'y' || cmdline.query("-di-indirect")[0] == 'Y';
        /**
         * Set up water signal removal
        */
        bool b_water = cmdline.query("-water")[0] == 'y' || cmdline.query("-water")[0] == 'Y';
        
       
        /**
         * Set up ZF
        */
        int n_zf=std::stoi(cmdline.query("-zf"));
        int n_zf_indirect=std::stoi(cmdline.query("-zf-indirect"));
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

        /**
         * Setup extraction
        */
        std::string ext_string=cmdline.query("-ext");
        std::vector<std::string> ext_string_split;
        int n_ext = fid_1d_helper::split(ext_string, ext_string_split, ' ');
        if(n_ext != 2)
        {
            std::cout << "Error: ext must have two numbers. Skip user input" << std::endl;
        }
        else
        {
            double ext_from=std::stod(ext_string_split[0]);
            double ext_to=std::stod(ext_string_split[1]);
            fid.extract_region(ext_from,ext_to);
            std::cout<<"Extract region from: " << ext_from << " to: " << ext_to << " along direct dimension." << std::endl;
        }

        /**
         * Set up phase correction
        */
        if(cmdline.query("-phase-in") != "none" && cmdline.query("-phase-in") != "null")
        {
            std::string phase_infname=cmdline.query("-phase-in");
            std::cout<<"Read phase correction file: " << phase_infname << std::endl;
            fid.read_phase_correction(phase_infname);
        }

        /**
         * Set up nus
        */
        std::string nus_file_name=cmdline.query("-nus");
        std::string nus_file_name_lower = nus_file_name;
        std::transform(nus_file_name_lower.begin(), nus_file_name_lower.end(), nus_file_name_lower.begin(), ::tolower);
        if(nus_file_name_lower != "none" && nus_file_name_lower != "no" && nus_file_name_lower != "null")
        {
            std::cout<<"Read nus file: " << nus_file_name << std::endl;
            fid.read_nus_list(nus_file_name);
        }


        /**
         * IMPORTANT: 
         * When read half-processed data or nmrPipe fid file, some flags set above might be overwritten
         * by values in the data file.
        */
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
             * If ends with .fid or .ft2 read nmrPipe file, 
             * if there is no ., read bruker folder
             * else do nothing
             */
            std::string pipe_file1(".fid");
            std::string pipe_file2(".ft2");
            if(std::equal(pipe_file1.rbegin(), pipe_file1.rend(), fid_infname_split[0].rbegin()))
            {
                b_read_fid = fid.read_nmrpipe_file(fid_infname_split[0]);
            }
            else if(std::equal(pipe_file2.rbegin(), pipe_file2.rend(), fid_infname_split[0].rbegin()))
            {
                b_read_fid = fid.read_nmrpipe_file(fid_infname_split[0]);
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

        if(b_water)
        {
            fid.water_suppression();
            std::cout<<"Remove water signal from FID." << std::endl;
        }


        std::string process_flag=cmdline.query("-process");
        if(process_flag == "none")
        {
            std::cout<<"No process." << std::endl;
        }
        else if(process_flag == "direct")
        {
            fid.direct_only_process(b_di_direct);
            std::cout<<"Only process direct dimension." << std::endl;
        }
        else if(process_flag == "indirect")
        {
            fid.indirect_only_process(b_di_indirect);
            std::cout<<"Only process indirect dimension." << std::endl;
        }
        else if(process_flag == "full")
        {
            fid.full_process(b_di_direct,b_di_indirect);
            std::cout<<"Full process." << std::endl;
        }
        else
        {
            fid.other_process(b_di_direct,b_di_indirect);
            std::cout<<"Other process." << std::endl;
        }

        int n_poly=std::stoi(cmdline.query("-poly"));
        if(n_poly >= 0)
        {
            if(n_poly>3)
            {
                n_poly=3; /** only support 0,1,2,3 at this time */
            }
            fid.polynorminal_baseline(n_poly);
            std::cout<<"Apply polynomial baseline correction with order: " << n_poly << std::endl;
        }
        else
        {
            std::cout<<"Skip baseline correction." << std::endl;
        }


        /**
         * Write FT2 file and json file (for web server to display the spectrum information)
        */
        std::string outfname=cmdline.query("-out");
        if(outfname != "no" && outfname != "null" && outfname != "none")
        {
            std::cout<<"Write file: " << outfname << std::endl;
            if(process_flag == "direct")
            {
                fid.write_nmrpipe_intermediate(outfname);
            }
            else if(process_flag == "none")
            {
                fid.write_nmrpipe_fid(outfname);
            }
            else
            {
                fid.write_nmrpipe_ft2(outfname);
            }
        }

        std::string outfname_json=cmdline.query("-out-json");
        std::cout<<"Write json information file: " << outfname_json << std::endl;
        fid.write_json(outfname_json);

        std::string outfname_pseudo3d_json=cmdline.query("-out-pseudo3d-json");
        std::cout<<"Write pseudo 3D json information file: " << outfname_pseudo3d_json << std::endl;
        fid.write_pseudo3d_json(outfname_pseudo3d_json);
    }

    return 0;
}