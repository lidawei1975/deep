#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "DeepConfig.h"
#include "commandline.h"
#include "fid_3d.h"

namespace
{

    bool parse_indexed_filename(const std::string &in,
                                std::string &path_name,
                                std::string &prefix,
                                std::string &extension,
                                int &idx,
                                int &digit_width)
    {
        std::string stem;
        if (!ldw_math_spectrum_1d::SplitFilename(in, path_name, stem, extension))
        {
            return false;
        }

        const size_t last_nondigit = stem.find_last_not_of("0123456789");
        std::string digits;
        if (last_nondigit == std::string::npos)
        {
            prefix = "";
            digits = stem;
        }
        else
        {
            prefix = stem.substr(0, last_nondigit + 1);
            digits = stem.substr(last_nondigit + 1);
        }

        if (digits.empty())
        {
            return false;
        }

        idx = std::stoi(digits);
        digit_width = static_cast<int>(digits.size());
        return true;
    }

} // namespace

int main(int argc, char **argv)
{
    std::cout << "DEEP Picker package Version " << deep_picker_VERSION_MAJOR << "." << deep_picker_VERSION_MINOR << std::endl;

    CCommandline cmdline;
    std::vector<std::string> args, defaults, infos;

    args.push_back("-h");
    defaults.push_back("no");
    infos.push_back("print help message then quit");

    args.push_back("-f");
    defaults.push_back("arguments_fid_3d.txt");
    infos.push_back("Arguments file");

    args.push_back("-fid-in");
    defaults.push_back("test001.fid test128.fid");
    infos.push_back("input first/last NMRPipe 3D plane file names, or a single stacked input file name");

    args.push_back("-ft3-in");
    defaults.push_back("none");
    infos.push_back("input a stacked .ft3 file, post-process it, and write an updated .ft3 file");

    args.push_back("-in");
    defaults.push_back("none");
    infos.push_back("Bruker direct input: acqus acqu2s acqu3s ser");

    args.push_back("-nus");
    defaults.push_back("none");
    infos.push_back("optional 3D nuslist file name; use with -in");

    args.push_back("-nus-ser-inflated");
    defaults.push_back("no");
    infos.push_back("when -nus is set: yes means ser is already inflated to NusTD grid");

    args.push_back("-out");
    defaults.push_back("none");
    infos.push_back("single output .ft3 file name (stacked planes)");

    args.push_back("-out-ft2-first");
    defaults.push_back("none");
    infos.push_back("first output .ft2 plane file name, e.g. proc001.ft2");

    args.push_back("-out-direct");
    defaults.push_back("none");
    infos.push_back("optional direct-only intermediate output .ft3 after direct processing, reordered to y,z,x");

    args.push_back("-indirect-only");
    defaults.push_back("no");
    infos.push_back("continue from direct-only intermediate input and finish remaining indirect processing (yes/no)");

    args.push_back("-ext");
    defaults.push_back("0.00 1.00");
    infos.push_back("extract direct dimension region: use normalized range (0.0 1.0) or ppm range (e.g. 8.8ppm 6.0ppm)");

    args.push_back("-apod");
    defaults.push_back("sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 0.5");
    infos.push_back("apodization for direct dimension");

    args.push_back("-apod-indirect1");
    defaults.push_back("sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 1.0");
    infos.push_back("apodization for indirect1 dimension");

    args.push_back("-apod-indirect2");
    defaults.push_back("sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 1.0");
    infos.push_back("apodization for indirect2 dimension");

    args.push_back("-zf");
    defaults.push_back("2");
    infos.push_back("zero filling factor for direct dimension");

    args.push_back("-zf-indirect1");
    defaults.push_back("2");
    infos.push_back("zero filling factor for indirect1 dimension");

    args.push_back("-zf-indirect2");
    defaults.push_back("2");
    infos.push_back("zero filling factor for indirect2 dimension");

    args.push_back("-phase-in");
    defaults.push_back("phase-correction.txt");
    infos.push_back("input phase correction file name. none means no phase correction.");

    args.push_back("-td-poly-order");
    defaults.push_back("-1 -1 -1");
    infos.push_back("time-domain polynorminal order triplet for F2 F1 F3, each in {-1,0..4}; -1 means disabled");

    args.push_back("-inverse");
    defaults.push_back("n y n");
    infos.push_back("inverse triplet for F2 F1 F3 using y/n; F2 entry accepted but ignored");

    args.push_back("-delete-image");
    defaults.push_back("y y y");
    infos.push_back("delete-image triplet for F2 F1 F3 using y/n; y means delete imaginary after that axis step");

    args.push_back("-frq-poly-order");
    defaults.push_back("-1 -1 -1");
    infos.push_back("frequency-domain polynorminal order triplet for F2 F1 F3, each in {-1,0..4}; -1 means disabled");

    args.push_back("-final-poly-order");
    defaults.push_back("-1 -1 -1");
    infos.push_back("final frequency-domain polynorminal order triplet for F2 F1 F3 applied after full FFT; -1 means disabled");

    args.push_back("-segment-size");
    defaults.push_back("8");
    infos.push_back("segment size used for noise estimation and signal-free region detection; must be positive");

    args.push_back("-simulate-nus");
    defaults.push_back("no");
    infos.push_back("when enabled with -nus: read data as fully sampled, then zero out traces not in nuslist (yes/no)");

    cmdline.init(args, defaults, infos);
    cmdline.pharse(argc, argv);
    cmdline.print();

    auto to_lower = [](std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });
        return s;
    };

    auto is_requested = [](const std::string &name)
    {
        return !name.empty() && name != "none" && name != "null" && name != "no";
    };

    auto fail = [&](const std::string &message)
    {
        std::cerr << message << std::endl;
    };

    if (cmdline.query("-h") == "yes")
    {
        return 0;
    }

    bool use_bruker_input = false;
    bool use_ft3_input = false;
    std::string first_file;
    std::string last_file;
    std::string ft3_file;
    std::string acqus_file;
    std::string acqu2s_file;
    std::string acqu3s_file;
    std::string ser_file;
    std::string nuslist_file;
    bool nus_ser_inflated = false;
    bool simulate_nus = false;

    std::vector<std::string> bruker_tokens;
    const int n_bruker = fid_1d_helper::split(cmdline.query("-in"), bruker_tokens, ' ');
    if (n_bruker > 0)
    {
        const std::string token0 = to_lower(bruker_tokens[0]);
        if (!(n_bruker == 1 && (token0 == "none" || token0 == "no" || token0 == "null")))
        {
            use_bruker_input = true;
        }
    }

    if (use_bruker_input)
    {
        if (n_bruker != 4)
        {
            fail("Error: -in must provide 4 files: acqus acqu2s acqu3s ser.");
            return 0;
        }
        acqus_file = bruker_tokens[0];
        acqu2s_file = bruker_tokens[1];
        acqu3s_file = bruker_tokens[2];
        ser_file = bruker_tokens[3];
        nuslist_file = cmdline.query("-nus");
        const std::string nus_ser_inflated_raw = to_lower(cmdline.query("-nus-ser-inflated"));
        nus_ser_inflated = (nus_ser_inflated_raw == "yes" || nus_ser_inflated_raw == "y" || nus_ser_inflated_raw == "true" || nus_ser_inflated_raw == "1");
        const std::string simulate_nus_raw = to_lower(cmdline.query("-simulate-nus"));
        simulate_nus = (simulate_nus_raw == "yes" || simulate_nus_raw == "y" || simulate_nus_raw == "true" || simulate_nus_raw == "1");
    }
    else
    {
        ft3_file = cmdline.query("-ft3-in");
        use_ft3_input = is_requested(ft3_file);

        if (!use_ft3_input)
        {
        std::vector<std::string> fid_tokens;
        const int n_fid = fid_1d_helper::split(cmdline.query("-fid-in"), fid_tokens, ' ');
        if (n_fid < 1)
        {
            fail("Error: -fid-in must provide either one stacked file name or first/last plane file names.");
            return 0;
        }
        first_file = fid_tokens[0];
        last_file = (n_fid >= 2) ? fid_tokens[1] : fid_tokens[0];
        }
    }

    const std::string out_ft3 = cmdline.query("-out");
    const std::string out_ft2_first = cmdline.query("-out-ft2-first");
    const std::string out_direct = cmdline.query("-out-direct");

    if (use_ft3_input && !is_requested(out_ft3))
    {
        fail("Error: -ft3-in requires an output .ft3 file via -out.");
        return 0;
    }

    if (!is_requested(out_ft3) && !is_requested(out_ft2_first) && !is_requested(out_direct))
    {
        fail("Error: provide at least one output target via -out, -out-ft2-first, or -out-direct.");
        return 0;
    }

    if (is_requested(out_direct) && (is_requested(out_ft3) || is_requested(out_ft2_first)))
    {
        fail("Error: -out-direct is a direct-only intermediate mode and cannot be combined with -out or -out-ft2-first in the same run.");
        return 0;
    }

    const std::string indirect_only = to_lower(cmdline.query("-indirect-only"));
    const bool use_indirect_only = (indirect_only == "yes" || indirect_only == "y" || indirect_only == "true" || indirect_only == "1");
    if (use_indirect_only && is_requested(out_direct))
    {
        fail("Error: -indirect-only cannot be combined with -out-direct.");
        return 0;
    }

    fid_3d fid;
    const int zf = std::stoi(cmdline.query("-zf"));
    const int zf1 = std::stoi(cmdline.query("-zf-indirect1"));
    const int zf2 = std::stoi(cmdline.query("-zf-indirect2"));
    if (!fid.run_zf(zf, zf1, zf2))
    {
        return 0;
    }

    apodization apod_direct(cmdline.query("-apod"));
    apodization apod_indirect1(cmdline.query("-apod-indirect1"));
    apodization apod_indirect2(cmdline.query("-apod-indirect2"));
    fid.set_up_apodization(&apod_direct, &apod_indirect1, &apod_indirect2);

    const std::string phase_in = cmdline.query("-phase-in");
    if (phase_in != "none" && phase_in != "null" && phase_in != "no")
    {
        std::cout << "Read phase correction file: " << phase_in << std::endl;
        if (!fid.read_phase_correction(phase_in))
        {
            return 0;
        }
    }

    auto parse_order_triplet = [&](const std::string &raw,
                                   const std::string &arg_name,
                                   int &o_f2,
                                   int &o_f1,
                                   int &o_f3) -> bool
    {
        std::string normalized = raw;
        std::replace(normalized.begin(), normalized.end(), ',', ' ');

        std::istringstream iss(normalized);
        if (!(iss >> o_f2 >> o_f1 >> o_f3))
        {
            std::cerr << "Error: " << arg_name << " must provide three integers for F2 F1 F3." << std::endl;
            return false;
        }

        const auto valid_order = [](int v)
        { return v >= -1 && v <= 4; };
        if (!valid_order(o_f2) || !valid_order(o_f1) || !valid_order(o_f3))
        {
            std::cerr << "Error: " << arg_name << " values must be in {-1,0..4}." << std::endl;
            return false;
        }

        return true;
    };

    int td_order_f2 = -1;
    int td_order_f1 = -1;
    int td_order_f3 = -1;
    if (!parse_order_triplet(cmdline.query("-td-poly-order"), "-td-poly-order", td_order_f2, td_order_f1, td_order_f3))
    {
        return 0;
    }
    fid.set_time_domain_polynorminal_order(td_order_f2, td_order_f1, td_order_f3);
    std::cout << "Time-domain polynorminal order (F2,F1,F3): "
              << td_order_f2 << ", " << td_order_f1 << ", " << td_order_f3 << std::endl;

    auto parse_yn_triplet = [&](const std::string &raw,
                                const std::string &arg_name,
                                bool &v_f2,
                                bool &v_f1,
                                bool &v_f3) -> bool
    {
        std::string normalized = raw;
        std::replace(normalized.begin(), normalized.end(), ',', ' ');

        std::string t2, t1, t3;
        std::istringstream iss(normalized);
        if (!(iss >> t2 >> t1 >> t3))
        {
            std::cerr << "Error: " << arg_name << " must provide three y/n values for F2 F1 F3." << std::endl;
            return false;
        }

        auto parse_one = [&](const std::string &token, bool &out) -> bool
        {
            const std::string t = to_lower(token);
            if (t == "y" || t == "yes" || t == "true" || t == "1")
            {
                out = true;
                return true;
            }
            if (t == "n" || t == "no" || t == "false" || t == "0")
            {
                out = false;
                return true;
            }
            return false;
        };

        if (!parse_one(t2, v_f2) || !parse_one(t1, v_f1) || !parse_one(t3, v_f3))
        {
            std::cerr << "Error: " << arg_name << " values must be y/n (or yes/no, true/false, 1/0)." << std::endl;
            return false;
        }

        return true;
    };

    bool inverse_f2 = false;
    bool inverse_f1 = false;
    bool inverse_f3 = false;
    if (!parse_yn_triplet(cmdline.query("-inverse"), "-inverse", inverse_f2, inverse_f1, inverse_f3))
    {
        return 1;
    }
    if (inverse_f2)
    {
        std::cout << "Warning: direct dimension (F2) inversion is unsupported and will be ignored." << std::endl;
        inverse_f2 = false;
    }
    fid.set_inverse(inverse_f2, inverse_f1, inverse_f3);
    std::cout << "Inverse (F2,F1,F3): "
              << (inverse_f2 ? "y" : "n") << ", "
              << (inverse_f1 ? "y" : "n") << ", "
              << (inverse_f3 ? "y" : "n") << std::endl;

    bool delimg_f2 = false;
    bool delimg_f1 = false;
    bool delimg_f3 = false;
    if (!parse_yn_triplet(cmdline.query("-delete-image"), "-delete-image", delimg_f2, delimg_f1, delimg_f3))
    {
        return  1;
    }
    fid.set_delete_image(delimg_f2, delimg_f1, delimg_f3);
    std::cout << "Delete-image (F2,F1,F3): "
              << (delimg_f2 ? "y" : "n") << ", "
              << (delimg_f1 ? "y" : "n") << ", "
              << (delimg_f3 ? "y" : "n") << std::endl;

    int frq_order_f2 = -1;
    int frq_order_f1 = -1;
    int frq_order_f3 = -1;
    if (!parse_order_triplet(cmdline.query("-frq-poly-order"), "-frq-poly-order", frq_order_f2, frq_order_f1, frq_order_f3))
    {
        return 1;
    }
    fid.set_frq_domain_polynorminal_order(frq_order_f2, frq_order_f1, frq_order_f3);
    std::cout << "Frequency-domain polynorminal order (F2,F1,F3): "
              << frq_order_f2 << ", " << frq_order_f1 << ", " << frq_order_f3 << std::endl;

    int final_order_f2 = -1;
    int final_order_f1 = -1;
    int final_order_f3 = -1;
    if (!parse_order_triplet(cmdline.query("-final-poly-order"), "-final-poly-order", final_order_f2, final_order_f1, final_order_f3))
    {
        return 1;
    }
    fid.set_final_frq_polynorminal_order(final_order_f2, final_order_f1, final_order_f3);
    std::cout << "Final frequency-domain polynorminal order (F2,F1,F3): "
              << final_order_f2 << ", " << final_order_f1 << ", " << final_order_f3 << std::endl;

    const int segment_size = std::stoi(cmdline.query("-segment-size"));
    if (segment_size <= 0)
    {
        fail("Error: -segment-size must be a positive integer.");
        return 1;
    }
    fid.set_segment_size(segment_size);
    std::cout << "Segment size: " << segment_size << std::endl;

    std::string nuslist_contents;
    if (use_bruker_input && is_requested(nuslist_file))
    {
        std::ifstream nuslist_stream(nuslist_file);
        if (!nuslist_stream.good())
        {
            fail("Error: cannot open nuslist file: " + nuslist_file);
            return 1;
        }
        nuslist_contents.assign((std::istreambuf_iterator<char>(nuslist_stream)), std::istreambuf_iterator<char>());
    }

    if (use_ft3_input)
    {
        std::ifstream ft3_stream(ft3_file, std::ios::binary);
        if (!ft3_stream.good())
        {
            fail("Error: cannot open ft3 file: " + ft3_file);
            return 1;
        }

        std::vector<unsigned char> ft3_bytes((std::istreambuf_iterator<char>(ft3_stream)), std::istreambuf_iterator<char>());
        if (!fid.read_ft3_from_buffer(ft3_bytes))
        {
            fail("Error: failed to read FT3 input file.");
            return 1;
        }

        std::cout << "Post-processing FT3 input: direct phase correction, frequency-domain baseline, and final baseline." << std::endl;
        if (!fid.postprocess_loaded_ft3())
        {
            fail("Error: failed to post-process FT3 input.");
            return 1;
        }

        if (!fid.write(out_ft3, out_ft2_first))
        {
            fail("Error: failed to write updated FT3 output.");
            return 1;
        }

        return 0;
    }

    std::string ext_string = cmdline.query("-ext");
    std::vector<std::string> ext_string_split;
    const int n_ext = fid_1d_helper::split(ext_string, ext_string_split, ' ');
    if (n_ext != 2)
    {
        std::cout << "Error: -ext must provide two values. Skip user input." << std::endl;
    }
    else
    {
        auto parse_ext_token = [&](const std::string &token, double &value, bool &is_ppm)
        {
            std::string token_l = to_lower(token);
            is_ppm = false;
            if (token_l.size() >= 3 && token_l.substr(token_l.size() - 3) == "ppm")
            {
                is_ppm = true;
                token_l = token_l.substr(0, token_l.size() - 3);
            }
            try
            {
                value = std::stod(token_l);
            }
            catch (const std::exception &)
            {
                return false;
            }
            return true;
        };

        double ext_from = 0.0;
        double ext_to = 1.0;
        bool ext_from_is_ppm = false;
        bool ext_to_is_ppm = false;

        const bool ok_from = parse_ext_token(ext_string_split[0], ext_from, ext_from_is_ppm);
        const bool ok_to = parse_ext_token(ext_string_split[1], ext_to, ext_to_is_ppm);

        if (!ok_from || !ok_to)
        {
            std::cout << "Error: cannot parse -ext values. Use either fractions (0.0 1.0) or ppm values (8.8ppm 6.0ppm). Skip user input." << std::endl;
        }
        else if (ext_from_is_ppm != ext_to_is_ppm)
        {
            std::cout << "Error: -ext values must use the same unit (both fractions or both ppm). Skip user input." << std::endl;
        }
        else if (ext_from_is_ppm)
        {
            if (fid.extract_region_ppm(ext_from, ext_to))
            {
                std::cout << "Extract region requested (direct dimension): " << ext_from << "ppm to " << ext_to << "ppm" << std::endl;
            }
        }
        else
        {
            if (fid.extract_region(ext_from, ext_to))
            {
                std::cout << "Extract region requested (direct dimension): normalized [" << ext_from << ", " << ext_to << "]" << std::endl;
            }
        }
    }

    if (use_bruker_input)
    {
        std::cout << "Read Bruker input: acqus=" << acqus_file
                  << ", acqu2s=" << acqu2s_file
                  << ", acqu3s=" << acqu3s_file
                  << ", ser/fid=" << ser_file;
        if (is_requested(nuslist_file))
        {
            std::cout << ", nuslist=" << nuslist_file;
            std::cout << ", nus-ser-inflated=" << (nus_ser_inflated ? "yes" : "no");
        }
        if (simulate_nus && is_requested(nuslist_file))
        {
            std::cout << " [simulate-nus enabled: reading as fully sampled, then zeroing traces from nuslist]";
        }
        std::cout << std::endl;
        if (!fid.read_bruker_files("", acqu2s_file, acqu3s_file, acqus_file, ser_file, nuslist_contents, simulate_nus, nus_ser_inflated))
        {
            fail("Error: failed to read Bruker 3D input.");
            return 1;
        }
    }
    else
    {
        if (!fid.read(first_file, last_file))
        {
            fail("Error: failed to read input 3D data.");
            return 1;
        }
    }

    /**
     * Check the processing/writing mode based on output targets:
     * 1) If direct-only intermediate output is requested, run direct-only processing and write
     */
    if (is_requested(out_direct))
    {
        if (!fid.direct_only_process())
        {
            fail("Error: failed to run direct-only processing.");
            return 1;
        }

        if (!fid.write(out_direct, ""))
        {
            fail("Error: failed to write direct-only intermediate output.");
            return 1;
        }

        return 0;
    }

    /**
     * If indirect-only continuation mode is requested, skip direct processing and run only indirect steps, then write final outputs.
     * In this mode the program expects that the input files provided via -fid-in are already
     */
    else if (use_indirect_only)
    {
        if (!fid.indirect_only_process())
        {
            fail("Error: failed to run indirect-only continuation processing.");
            return 1;
        }

        if (!fid.write(out_ft3, out_ft2_first))
        {
            fail("Error: failed to write output 3D data after indirect-only continuation.");
            return 1;
        }

        return 0;
    }
    /**
     * Otherwise run full processing and write outputs.
     */
    else 
    {
        if (fid.full_process())
        {
            if(fid.write(out_ft3, out_ft2_first))
            {
                return 0;
            }
            else
            {
                fail("Error: failed to write output 3D data after full processing.");
                return 1;
            }    
        }
        else
        {
            fail("Error: failed to run full processing.");
            return 1;
        }
    }

    return 0;
}


