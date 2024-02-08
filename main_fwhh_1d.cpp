
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>
#include <time.h>
#include <sys/time.h>


#include "json/json.h"
#include "commandline.h"
#include "spectrum_fwhh_1d.h"

#include "DeepConfig.h"

int main(int argc, char **argv)
{ 
    std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;
 
    struct timeval time;
    gettimeofday(&time,NULL);
    double time_1= (double)time.tv_sec + (double)time.tv_usec * .000001;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-f");
    args2.push_back("arguments_fwhh_1d.txt");
    args3.push_back("Arguments file");

    args.push_back("-in");
    args2.push_back("test.ft1"); 
    args3.push_back("input file name (test.ft1)");

    args.push_back("-out");
    args2.push_back("fwhh.txt"); 
    args3.push_back("output file name (fwhh.txt)");


    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    cmdline.print();
    if (cmdline.query("-h") != "yes")
    {
        spectrum_fwhh_1d fwhh;
        fwhh.read_spectrum(cmdline.query("-in"));
        std::cout<<"Estimated median peak width: "<<fwhh.get_median_peak_width()<<std::endl;
        fwhh.print_result(cmdline.query("-out"));
    }

    return 0;
}
