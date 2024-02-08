
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>



#include "commandline.h"
#include "json/json.h"
#include "spectrum_io_1d.h"




int main(int argc, char **argv)
{
 
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");


    args.push_back("-in");
    args2.push_back("test.ft1");
    args3.push_back("input file name (test.ft1)");


    args.push_back("-out");
    args2.push_back("experiments.json");
    args3.push_back("output file name");

    args.push_back("-out-json");
    args2.push_back("fid-information.json");
    args3.push_back("output json file name for spectral information");


    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    std::string infname,outfname;

    infname = cmdline.query("-in");
    outfname = cmdline.query("-out");

    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class spectrum_io_1d x;
        if(x.read_spectrum(infname)) //read
        {
            x.write_spectrum(outfname); //write in json format

            /**
             * Write some userful information to a json file for web server
            */
            std::string outfname_json=cmdline.query("-out-json");
            std::cout<<"Write json file: " << outfname_json << std::endl;
            x.write_json(outfname_json);

        }
        std::cout<<"Done!"<<std::endl;
    }
    return 0;
};
