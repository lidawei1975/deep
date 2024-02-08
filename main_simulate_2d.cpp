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
#include "contour.h"
#include "spectrum_prediction.h"
#include "DeepConfig.h"

int main(int argc, char ** argv)
{
    std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;
    std::cout<<"This program will generate simulated spectrum contour from 2D databse query result."<<std::endl;


    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-v");
    args2.push_back("0");
    args3.push_back("verbose level (0: minimal, 1:normal)");

    args.push_back("-f");
    args2.push_back("arguments_simulated_2d.txt");
    args3.push_back("read arguments from file");

    args.push_back("-in");
    args2.push_back("query.json");
    args3.push_back("input query result file");

    args.push_back("-out");
    args2.push_back("simulated-spectrum.json");
    args3.push_back("output simulated spectrum information file");

    args.push_back("-out-bin");
    args2.push_back("simulated-spectrum.bin");
    args3.push_back("output simulated spectrum contour binary file");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    /**
     * Define a spectrum_simulation object.
    */
    spectrum_prediction spe_prediction;

    spe_prediction.load_query_json(cmdline.query("-in"));
    spe_prediction.work();
    spe_prediction.save_simulated_spectrum(cmdline.query("-out"));
    spe_prediction.save_simulated_spectrum_binary(cmdline.query("-out-bin"));

    return 0;
}