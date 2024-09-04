#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>

#include "json/json.h"
#include "commandline.h"
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
    args2.push_back("arguments_phase_2d.txt");
    args3.push_back("input arguments file name.");

    args.push_back("-in");
    args2.push_back("test.ft2");
    args3.push_back("input spectral file names. none or null means we will use -fid_in to read fid file instead.");

    args.push_back("-out");
    args2.push_back("test-phased.ft2");
    args3.push_back("output spectral file names.");

    args.push_back("-real_only");
    args2.push_back("no");
    args3.push_back("only save real spectrum.");

    args.push_back("-user");
    args2.push_back("no");
    args3.push_back("use user defined phase correction.");

    args.push_back("-user_phase");
    args2.push_back("0.0 0.0 0.0 0.0");
    args3.push_back("user defined phase correction in degree P0,P1 (direct) P0,P1 (indirect).");

    args.push_back("-out_phase");
    args2.push_back("phase-correction.txt");
    args3.push_back("output file name phase correction values.");

    
    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    std::string infname = cmdline.query("-in");
    std::string outfname = cmdline.query("-out");

    if (cmdline.query("-h") != "yes")
    {
        
        class spectrum_phasing x;

        /**
         * Baseclass spectrum_io function to read spectrum, without noise estimation (which is not good for un-phased spectrum)
        */
        x.init(infname,false);

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
            x.set_user_phase_correction(std::stod(strs[0]), std::stod(strs[1]), std::stod(strs[2]), std::stod(strs[3]));
        }
        else
        {
            /**
             * main working function
            */
            x.auto_phase_correction_v2();
            std::cout<<"Auto phase correction done!"<<std::endl;
        }

        /**
         * Save phased spectrum to a file
        */
        bool b_real_only=cmdline.query("-real_only")[0] == 'y' || cmdline.query("-real_only")[0] == 'Y';
        x.write_pipe(outfname,b_real_only);
        std::cout<<"Phased spectrum saved to "<<outfname<<std::endl;

        /**
         * Save phase correction values to a file
        */
        std::string out_phase_fname = cmdline.query("-out_phase");
        x.save_phase_correction_result(out_phase_fname);
        std::cout<<"Phase correction values saved to "<<out_phase_fname<<std::endl;

    }

    return 0;
}