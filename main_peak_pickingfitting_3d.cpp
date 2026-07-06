#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

#include "commandline.h"
#include "spectrum_fit_3d.h"
#include "DeepConfig.h" // Assuming this exists and is needed for version info

int main(int argc, char **argv)
{
    std::cout << "DEEP Picker 3D package" << std::endl;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit");

    args.push_back("-f");
    args2.push_back("arguments_deconv_3d.txt");
    args3.push_back("Arguments file");

    args.push_back("-v");
    args2.push_back("2");
    args3.push_back("verbose level (0: no output, 1: status only, 2: all)");

    args.push_back("-in");
    args2.push_back("test001.ft3"); // Default example
    args3.push_back("input either one stacked .ft3 file or two file names (first and last file of the 3D series). Separated by space.");

    args.push_back("-peak-in");
    args2.push_back("3d_peaks.txt");
    args3.push_back("input file containing picked peaks");

    args.push_back("-noise_level");
    args2.push_back("0");
    args3.push_back("Direct set noise level to this value. Noise level will be estimated from spectrum if input is 0.0");

    args.push_back("-nus");
    args2.push_back("no");
    args3.push_back("Estimate noise level for NUS simulation (yes/no)");

    args.push_back("-top-n");
    args2.push_back("10");
    args3.push_back("Fit only top N partitions sorted by max intensity");

    args.push_back("-scale");
    args2.push_back("6.0");
    args3.push_back("User defined minimal peak intensity scale for peak picking");

    args.push_back("-scale2");
    args2.push_back("3.5");
    args3.push_back("User defined noise floor scale factor for signal/region threshold");

    args.push_back("-new-peak-cutoff");
    args2.push_back("0.4");
    args3.push_back("Normalized residual cutoff for adding new peaks");

    cmdline.init(args, args2, args3);
    if(!cmdline.pharse(argc, argv)) return 1;

    if (cmdline.query("-h") == "yes")
    {
        cmdline.print();
        return 0;
    }

    std::string infname = cmdline.query("-in");
    double noise_level = atof(cmdline.query("-noise_level").c_str());
    double scale1 = atof(cmdline.query("-scale").c_str());
    double scale2 = atof(cmdline.query("-scale2").c_str());
    double new_peak_cutoff = atof(cmdline.query("-new-peak-cutoff").c_str());
    int top_n = atoi(cmdline.query("-top-n").c_str());
    int verbose = atoi(cmdline.query("-v").c_str());
    bool b_noise_level_for_nus = (cmdline.query("-nus").at(0) == 'y' || cmdline.query("-nus").at(0) == 'Y');

    if (verbose >= 1) std::cout << "Input files string: " << infname << std::endl;

    // Parse input files: either one stacked .ft3 file or a first/last range.
    std::istringstream iss(infname);
    std::string p;
    std::vector<std::string> file_names;
    while (iss >> p)
    {
        file_names.push_back(p);
    }

    if (file_names.empty() || file_names.size() > 2)
    {
        std::cerr << "Error: Please provide either one stacked .ft3 file or two file names (first and last file of the series) for -in argument." << std::endl;
        return 1;
    }

    const bool single_ft3_input = (file_names.size() == 1);
    std::string first_file = file_names[0];
    std::string last_file = single_ft3_input ? file_names[0] : file_names[1];

    if (single_ft3_input)
    {
        if (verbose >= 1) std::cout << "Loading 3D Spectrum from stacked FT3 file: " << first_file << std::endl;
    }
    else
    {
        if (verbose >= 1) std::cout << "Loading 3D Spectrum Range: " << first_file << " to " << last_file << std::endl;
    }

    spectrum_fit_3d app;
    app.set_verbose(verbose);

    if (noise_level > 1e-20)
    {
        if (verbose >= 1) std::cout << "User set noise level: " << noise_level << std::endl;
    }
    if (scale1 > 0.0)
    {
        if (verbose >= 1) std::cout << "User set scale1: " << scale1 << std::endl;
    }
    if (scale2 > 0.0)
    {
        if (verbose >= 1) std::cout << "User set scale2: " << scale2 << std::endl;
    }
    if (new_peak_cutoff > 0.0)
    {
        if (verbose >= 1) std::cout << "User set new peak cutoff: " << new_peak_cutoff << std::endl;
    }

    /**
     *  If b_noise_level_for_nus == true, app will use 0.001 * maximal sepctral intensity as noise level
     * else if noise_level is 0.0, app will estimate noise level from spectrum using median 16x16x16 segment RMSD.
     * else app will use user provided noise level.
     */
    app.set_noise_scales(noise_level, scale1, scale2);
    app.set_new_peak_cutoff(static_cast<float>(new_peak_cutoff));
    app.set_noise_level_for_nus(b_noise_level_for_nus);

    bool read_ok = false;

    if(single_ft3_input == true){
        std::ifstream ft3_stream(first_file, std::ios::binary);
        if (!ft3_stream.good())
        {
            std::cerr << "Error: cannot open ft3 file: " << first_file << std::endl;
            return 1;
        }

        std::vector<unsigned char> ft3_bytes((std::istreambuf_iterator<char>(ft3_stream)), std::istreambuf_iterator<char>());
        read_ok = app.read_ft3_from_buffer(ft3_bytes);
    }
    else
    {
        read_ok = app.read(first_file, last_file);
    }

    if (!read_ok)
    {
        std::cerr << "Failed to read spectrum." << std::endl;
        return 1;
    }

    if (verbose >= 1) std::cout << "Successfully read spectrum." << std::endl;

    std::string peak_in_fname = cmdline.query("-peak-in");

    if(peak_in_fname != "none" && peak_in_fname != "null" && peak_in_fname != "no")
    {
        app.load_picked_peaks_file(peak_in_fname);
    }
    else
    {
        if (verbose >= 1) std::cout << "No picked peaks file provided. Run peaks picking first." << std::endl;
        app.peak_picking("3d_peaks.txt"); //also save a peak file.
    }

    app.partition_signal_regions("3d_signal_regions.txt");

    // app.dump_partition_voxels(775, "partition_775_voxels.txt");

    // app.iterative_fit_one_partition(11);
    // app.iterative_fit_one_partition(16);
    // app.iterative_fit_one_partition(6);
    app.iterative_fit_all_partitions(top_n);

    std::string prefit_result = app.get_fitted_peaks_string();
    std::ofstream prefit_out("3d_fit_partition_prefit.txt");
    if (prefit_out.is_open()) {
        prefit_out << prefit_result;
        prefit_out.close();
    }
    
    return 0;
}
