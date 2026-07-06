#include <vector>
#include <iostream>
#include <fstream>

#include "commandline.h"
#include "json/json.h"
#include "fid_2d.h"


int main(int argc, char **argv)
{


    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;
    double s;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-f");
    args2.push_back("arguments_plot_2d.txt");
    args3.push_back("Arguments file");
    
    args.push_back("-in");
    args2.push_back("input.ft2");
    args3.push_back("input hsqc spectrum file name");


    args.push_back("-out-json");
    args2.push_back("spectrum-information.json");
    args3.push_back("output information file name");

    args.push_back("-out-spectrum");
    args2.push_back("input.bin");
    args3.push_back("output spectrum as float32 array file name");



    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);


    std::string infname=cmdline.query("-in");
    std::string outfname_spectrum=cmdline.query("-out-spectrum");
    std::string outfname_json=cmdline.query("-out-json");


    
    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        class fid_2d spe;
        spe.init(infname,1); //read spectrum and estimate noise level
        std::cout<<"Read spectrum from "<<infname<<" done."<<std::endl;

        int n_direct,n_indirect;
        spe.get_dim(&n_direct,&n_indirect);
        
        spe.write_json(outfname_json);
        std::cout<<"Write spectrum information to "<<outfname_json<<std::endl;

        /**
         * Write spectrum to binary file
        */
        float *data=spe.get_spect_data();
        std::ofstream ofs2(outfname_spectrum.c_str(),std::ios::binary);
        ofs2.write((char *)data,sizeof(float)*n_direct*n_indirect);
        ofs2.close();
        std::cout<<"Write spectrum to "<<outfname_spectrum<<std::endl;
        
    }
    
    return 0;
}