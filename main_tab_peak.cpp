#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <set>

#include "commandline.h"
#include "DeepConfig.h"

#include "peak_manipulation.h"

int main(int argc, char **argv)
{
    std::cout << "DEEP Picker package Version " << deep_picker_VERSION_MAJOR << "." << deep_picker_VERSION_MINOR << std::endl;
    std::cout << "This program will create a new peak list file in .tab format from an old one. " << std::endl;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("Print this help message then quit");

    args.push_back("-f");
    args2.push_back("arguments_peak_manipulation.txt");
    args3.push_back("A file containing all the arguments");

    args.push_back("-in");
    args2.push_back("fitted.tab");
    args3.push_back("input file in .tab format");

    args.push_back("-out");
    args2.push_back("output.tab");
    args3.push_back("output file in .tab format");
    
    args.push_back("-revise");
    args2.push_back("X_PPM 0.0");
    args3.push_back("Change column (X_PPM) by a value (0.0)");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    std::string fname_in=cmdline.query("-in");
    std::string fname_out=cmdline.query("-out");

    std::string revise=cmdline.query("-revise");
    std::istringstream iss(revise);
    std::string revise_col;
    double revise_value;
    iss >> revise_col >> revise_value;

    peak_manipulation pm;
    pm.read_file(fname_in);
    pm.operate_on_column(pm.get_column_index(revise_col), column_operation::ADD, revise_value);
    pm.write_file(fname_out);

    return 0;
}