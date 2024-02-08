
#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>
#include <time.h>
#include <sys/time.h>


#include "json/json.h"
#include "commandline.h"
#include "spectrum_fwhh.h"

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
    args2.push_back("arguments_fwhh.txt");
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
        spectrum_fwhh fwhh;
        fwhh.init(cmdline.query("-in"),1); // read spectrum and estimate noise level (defined in base class spectrum_io)
        float median_width_direct, median_width_indirect;
        fwhh.get_median_peak_width(median_width_direct, median_width_indirect);
        std::cout<<"Estimated median peak width: "<<std::endl;
        std::cout<<median_width_direct<<" along direct dimension"<<std::endl;
        std::cout<<median_width_indirect<<" along indirect dimension"<<std::endl;
        fwhh.print_result(cmdline.query("-out"));
    }

    return 0;
}
