#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

/**
 * Below two lines are required to check whether folder or file exists
 */
#include <sys/types.h>
#include <sys/stat.h>

#ifdef WEBASSEMBLY
/**
* Bind to emscripten with exposed class methods.
*/
#include <emscripten/bind.h>
using namespace emscripten;
#endif


#include "kiss_fft.h"

#include "json/json.h"
#include "fid_1d.h"

fid_1d::fid_1d()
{
    zf = 1;              // no zero filling, default
    ndata_frq = 0;       // no data without FFT
    receiver_gain = 1.0; // default

    nmrpipe_header_data.resize(512, 0.0f); // nmrPipe header data 512*4

    user_scale = 5.5;
    user_scale2 = 3.0;
    noise_level = 0.01;
    ndata = 0;
    ndata_bruker = 0;
    ndata_original = 0;
    ndata_power_of_2 = 0;
    ndata_frq = 0;
    spectral_width = 0.0;
    observed_frequency = 0.0;
    origin = 0.0;
    begin1 = 0.0;
    step1 = 0.0;
    stop1 = 0.0;
}

/**
 * @brief fid_1d class destructor
 */
fid_1d::~fid_1d()
{
}

/**
 * @brief fid_1d::run_zf: set zero filling factor
 * zf will be read by run_fft_and_rm_bruker_filter
 */
bool fid_1d::run_zf(int _zf)
{
    zf = _zf;
    return true;
}

/**
 * @brief read_bruker_folder:  read fid and parameter files from Bruker folder
 * this function will change the value of the following variables:
 * fid_data_float (unused one will have size 0)
 * data_type, data_complexity
 * ndata_bruker, ndata
 * udict_acqus, udict_procs
 * spectral_width, observed_frequency, carrier_frequency
 */
bool fid_1d::read_bruker_folder(std::string folder_name)
{
    /**
     * @brief test if the folder exists, using system call stat
     * sb is a struct of type stat
     */
    struct stat sb;

    /**
     * Calls the function with path as argument. This is a C function.
     * If the file/directory exists at the path returns 0
     */

    int status = stat(folder_name.c_str(), &sb);

    if (status != 0)
    {
        std::cout << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }
    else if (!(sb.st_mode & S_IFDIR))
    {
        // exist but is not a directory
        std::cout << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }

    /**
     * now look for folder_name/fid or folder_name/ser
     * If none of them exist, return false
     */

    std::string fid_data_file_name = folder_name + "/fid";
    status = stat(fid_data_file_name.c_str(), &sb);
    if (status == 0)
    {
    }
    else
    {
        fid_data_file_name = folder_name + "/ser";
        status = stat(fid_data_file_name.c_str(), &sb);
        if (status == 0)
        {
        }
        else
        {
            fid_data_file_name = ""; // label it as empty
            return false;
        }
    }

    /**
     * Read acqus file and store the content in udict
     */
    std::string acqus_file_name = folder_name + "/acqus";
    status = stat(acqus_file_name.c_str(), &sb);
    if (status != 0)
    {
        std::cout << "Warning: cannot find acqus file in folder " << folder_name << std::endl;
        return false;
    }

    /**
     * if we are here, we have found the fid data file and acqus file
     */
    std::vector<std::string> fid_data_file_names;
    fid_data_file_names.push_back(fid_data_file_name);
    return read_bruker_acqus_and_fid(acqus_file_name, fid_data_file_names);
}


bool fid_1d::read_bruker_files_as_strings(const std::string &contents_acqus)
{
    return process_jcamp_as_string(contents_acqus, udict_acqus) && process_dictionary();      
}

bool fid_1d::set_fid_data(const std::vector<float> &fid_data_float_)
{
    fid_data_float = std::move(fid_data_float_);
    nspectra = fid_data_float.size()/ ndata_bruker; // number of spectra is the size of fid_data_float divided by ndata_bruker
    std::cout<<"Set fid_data_float with " << fid_data_float.size() << " elements, which is " << nspectra << " spectra." << std::endl;
    std::cout<<"ndata_bruker is " << ndata_bruker << std::endl;
    return true;
}

/**
 * Cut short of fid, in case later fid is mainly noise
 * @ndata_bruker_new: new size of fid data in Bruker format (including imaginary part if complex)
 * return: true.
 * Changes fid_data_float, ndata_bruker, ndata and ndata_original (if needed)
 * IMPORTANT: this function assumes fid_data_float has been set up already and input parameter ndata_bruker_new is in Bruker convention
 */
bool fid_1d::reduce_fid_size(int ndata_bruker_new)
{
    if(ndata_bruker_new <=0 || ndata_bruker_new > ndata_bruker)
    {
        std::cerr<<"Error: invalid ndata_bruker_new "<< ndata_bruker_new << std::endl;
        return false;
    }

    if(ndata_bruker_new == ndata_bruker)
    {
        // do nothing
        return true;
    }

    std::cout<<"Reduced fid size to from " << ndata_bruker << " to " << ndata_bruker_new << std::endl;

    std::vector<float> fid_data_float_new;
    fid_data_float_new.resize(ndata_bruker_new * nspectra);

    for(int i=0;i<nspectra;i++)
    {
        std::copy(fid_data_float.data()+i*ndata_bruker, fid_data_float.data()+i*ndata_bruker+ndata_bruker_new, fid_data_float_new.data()+i*ndata_bruker_new);
    }

    fid_data_float = std::move(fid_data_float_new);
    ndata_bruker = ndata_bruker_new;

    if(data_complexity == FID_DATA_COMPLEXITY_COMPLEX)
    {
        ndata = ndata_bruker / 2;
    }
    else
    {
        ndata = ndata_bruker;
    }
    /**
     * In case we cut shortter than ndata_original, we need to update ndata_original as well
     */
    if (ndata < ndata_original)
    {
        ndata_original = ndata;
    }

    return true;
}

int fid_1d::detect_signal_end_location() const
{
    // If multiple spectra, we analyze the first one
    int N = ndata_bruker; 
    
    // Access to the raw data (first spectrum)
    const float* data_ptr = fid_data_float.data();

    // 1. Parameters (matching example.js)
    if (N < 200) return N;
    
    const int skip = 150;
    // window_size = Math.max(1, Math.min(4096, Math.floor(N / 16)));
    int window_size = N / 16;
    if (window_size > 4096) window_size = 4096;
    if (window_size < 1) window_size = 1;

    // 2. Envelope Extraction
    std::vector<float> envelope_amp;
    std::vector<int> envelope_idx;
    // Reserve memory to avoid multiple reallocations
    size_t est_size = (N - skip)/window_size + 100;
    envelope_amp.reserve(est_size);
    envelope_idx.reserve(est_size);

    // example.js loop: for (let i = skip; i <= N - window_size; i++)
    for (int i = skip; i <= N - window_size; i++) {
        float local_max = -std::numeric_limits<float>::infinity();
        float local_min = std::numeric_limits<float>::infinity();
        
        for (int j = 0; j < window_size; j++) {
            float val = data_ptr[i + j];
            if (val > local_max) local_max = val;
            if (val < local_min) local_min = val;
        }
        float amp = (local_max - local_min) * 0.5f;
        envelope_amp.push_back(amp);
        envelope_idx.push_back(i);
    }

    if (envelope_amp.empty()) return N;

    // Estimate A0 (Max of envelope)
    // Envelope should be strictly decaying ideally, so max should be near start.
    float A0 = 0.0f;
    for (float val : envelope_amp) {
        if (val > A0) A0 = val;
    }

    if (A0 == 0.0f) return N;

    // 3. Rough End Definition (1%)
    const float thresh_1pct = 0.01f * A0;
    int rough_end_idx_in_envelope = (int)envelope_amp.size() - 1;

    for (size_t k = 0; k < envelope_amp.size(); k++) {
        if (envelope_amp[k] < thresh_1pct) {
            rough_end_idx_in_envelope = k;
            break;
        }
    }

    int noise_start_idx = envelope_idx[rough_end_idx_in_envelope];

    // 4. Noise Estimation (Median of RMSDs of 32-pt segments)
    const int segment_len = 32;
    std::vector<float> std_devs;
    
    // Iterate from noise_start_idx to N
    // We iterate on the raw data
    int noise_region_len = N - noise_start_idx;
    int n_segments = noise_region_len / segment_len;

    float noise_level = 0.0f;

    if (n_segments > 0) {
        std_devs.reserve(n_segments);
        for (int s = 0; s < n_segments; s++) {
            int seg_start = noise_start_idx + s * segment_len;
            
            // Calc Mean
            double sum = 0.0;
            for (int i = 0; i < segment_len; i++) {
                sum += data_ptr[seg_start + i];
            }
            double mean = sum / segment_len;

            // Calc Variance
            double sum_sq_diff = 0.0;
            for (int i = 0; i < segment_len; i++) {
                double diff = data_ptr[seg_start + i] - mean;
                sum_sq_diff += diff * diff;
            }
            // StdDev (Population or Sample? Usually Sample n-1)
            double variance = sum_sq_diff / (segment_len - 1);
            std_devs.push_back((float)std::sqrt(variance));
        }

        // Median
        std::sort(std_devs.begin(), std_devs.end());
        noise_level = std_devs[std_devs.size() / 2];
    } else {
        // Fallback if region is too small
        noise_level = 0.0f; 
    }

    // 5. Final Cutoff
    // "reach 2 times noise level or 0.1% of init" -> max(6 * noise, 0.001 * A0) 
    const float limit = std::max(6.0f * noise_level, 0.001f * A0);

    int final_cutoff_idx = N; // Default to end

    for (size_t k = 0; k < envelope_amp.size(); k++) {
        if (envelope_amp[k] < limit) {
            final_cutoff_idx = envelope_idx[k];
            break;
        }
    }

    return final_cutoff_idx;
}



bool fid_1d::read_bruker_acqus_and_fid(const std::string &acqus_file_name, const std::vector<std::string> &fid_data_file_names)
{

    if(!read_jcamp(acqus_file_name, udict_acqus))
    {
        std::cout << "Error: cannot read acqus file " << acqus_file_name << std::endl;
        return false;
    }

    if(!process_dictionary())
    {
        std::cerr << "Error: cannot process dictionary" << std::endl;
        return false;
    }

        /**
     * now we can actually read the fid data
     * For complex data, real and imaginary parts are stored interleaved by Bruker.
     * Here we leave them interleaved in fid_data_float or fid_data_int
     */

    fid_data_float.clear();

    for (int i = 0; i < fid_data_file_names.size(); i++)
    {
        std::vector<int32_t> fid_data_int;        // 32 bit
        std::vector<double> temp_fid_data_double; // 64 bit
        FILE *fp_fid_data = fopen(fid_data_file_names[i].c_str(), "rb");

        if (data_type == FID_DATA_TYPE_INT32)
        {
            int nread;
            fid_data_int.clear();
            fid_data_int.resize(ndata_bruker);
            nspectra = 0;
            while(fread(fid_data_int.data()+ndata_bruker*nspectra, sizeof(int32_t), ndata_bruker, fp_fid_data) == ndata_bruker)
            {
                nspectra++;
                /**
                 * Get new space for next round
                */
                fid_data_int.resize(ndata_bruker * (nspectra + 1));
            }
            /**
             * Set fid_data_int to correct size, remove the extra space applied in last round
            */
            fid_data_int.resize(ndata_bruker * nspectra);
            if (nspectra == 0)
            {
                std::cout << "Error: cannot read " << ndata_bruker << " int32 from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }
        else if (data_type == FID_DATA_TYPE_FLOAT64)
        {
            int nread;
            temp_fid_data_double.clear();
            temp_fid_data_double.resize(ndata_bruker);
            nspectra = 0;
            while(fread(temp_fid_data_double.data()+ndata_bruker*nspectra, sizeof(double), ndata_bruker, fp_fid_data) == ndata_bruker)
            {
                nspectra++;
                /**
                 * Get new space for next round
                */
                temp_fid_data_double.resize(ndata_bruker * (nspectra + 1));
            }
            /**
             * Set temp_fid_data_double to correct size, remove the extra space applied in last round
            */
            temp_fid_data_double.resize(ndata_bruker * nspectra);
            if (nspectra == 0)
            {
                std::cout << "Error: cannot read " << ndata_bruker << " double from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }

        /**
         * convert int32 or double64 to float32.
         */
        fid_data_float.resize(ndata_bruker * nspectra);    
        if (data_type == FID_DATA_TYPE_INT32)
        {
            for (int i = 0; i < ndata_bruker * nspectra; i++)
            {
                fid_data_float[i] += (float)fid_data_int[i];
            }
        }
        else
        {
            for (int i = 0; i < ndata_bruker * nspectra; i++)
            {
                fid_data_float[i] += temp_fid_data_double[i];
            }
        }
        fclose(fp_fid_data);
    }

    std::cout << "Read " << nspectra << " spectra from " << fid_data_file_names.size() << " files" << std::endl;

    

    return true;
}


bool fid_1d::read_magritek_folder(std::string folder_name)
{
    /**
     * @brief test if the folder exists, using system call stat
     * sb is a struct of type stat
     */
    struct stat sb;

    /**
     * Calls the function with path as argument. This is a C function.
     * If the file/directory exists at the path returns 0
     */
    int status = stat(folder_name.c_str(), &sb);

    if (status != 0)
    {
        std::cout << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }
    else if (!(sb.st_mode & S_IFDIR))
    {
        // exist but is not a directory
        std::cout << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }

    std::cout << "Placeholder: read_magritek_folder called for folder: " << folder_name << std::endl;
    std::string acqu_file_name = folder_name + "/acqu.par";
    std::string fid_file_name = folder_name + "/data.1d";

    std::vector<std::string> fid_data_file_names;
    fid_data_file_names.push_back(fid_file_name);

    return read_magritek_acqu_and_fid(acqu_file_name, fid_data_file_names);
}

bool fid_1d::read_magritek_acqu_and_fid(const std::string &acqu_file_name, const std::vector<std::string> &fid_data_file_names)
{
    std::cout << "Reading Magritek parameter file: " << acqu_file_name << std::endl;
    
    std::map<std::string, std::string> params;
    std::ifstream infile(acqu_file_name.c_str());
    if (!infile.is_open())
    {
        std::cout << "Error: cannot open Magritek parameter file " << acqu_file_name << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(infile, line))
    {
        if (line.size() > 0 && line[line.size() - 1] == '\r')
        {
            line = line.substr(0, line.size() - 1);
        }

        size_t pos = line.find('=');
        if (pos == std::string::npos)
        {
            continue;
        }

        std::string key = line.substr(0, pos);
        std::string val = line.substr(pos + 1);

        // Trim helper
        auto trim_func = [](std::string &s) {
            size_t start = s.find_first_not_of(" \t");
            if (start == std::string::npos) {
                s = "";
                return;
            }
            size_t end = s.find_last_not_of(" \t");
            s = s.substr(start, end - start + 1);
        };
        trim_func(key);
        trim_func(val);

        if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
        {
            val = val.substr(1, val.size() - 2);
        }

        if (!key.empty())
        {
            params[key] = val;
        }
    }
    infile.close();

    // Map parameters to class variables
    int nr_pnts = 0;
    if (params.find("nrPnts") != params.end())
    {
        nr_pnts = std::stoi(params["nrPnts"]);
    }
    else
    {
        std::cout << "Error: nrPnts not found in parameter file " << acqu_file_name << std::endl;
        return false;
    }

    b_read_from_ft1 = false;
    ndata = nr_pnts;
    ndata_original = nr_pnts;
    data_complexity = FID_DATA_COMPLEXITY_COMPLEX;
    ndata_bruker = nr_pnts * 2;
    nspectra = (int)fid_data_file_names.size();
    if (nspectra == 0)
    {
        nspectra = 1;
    }

    if (params.find("bandwidth") != params.end())
    {
        spectral_width = std::stod(params["bandwidth"]) * 1000.0;
    }
    else if (params.find("dwellTime") != params.end())
    {
        spectral_width = 1e6 / std::stod(params["dwellTime"]);
    }
    else if (params.find("totalAcquisitionTim") != params.end())
    {
        spectral_width = (nr_pnts * 1000.0) / std::stod(params["totalAcquisitionTim"]);
    }
    else
    {
        std::cout << "Error: cannot determine spectral width from parameter file" << std::endl;
        return false;
    }

    double freq = 0.0;
    if (params.find("b1Freq1H") != params.end())
    {
        freq = std::stod(params["b1Freq1H"]);
    }
    else if (params.find("b1Freq") != params.end())
    {
        freq = std::stod(params["b1Freq"]);
    }
    else
    {
        std::cout << "Warning: b1Freq1H or b1Freq not found. Defaulting to 80.0 MHz." << std::endl;
        freq = 80.0;
    }
    observed_frequency = freq;

    double sat_freq = 0.0;
    if (params.find("satFrequency1") != params.end())
    {
        sat_freq = std::stod(params["satFrequency1"]);
    }
    else
    {
        std::cout << "Warning: satFrequency1 not found. Defaulting to 4.7 ppm." << std::endl;
        sat_freq = 4.7;
    }
    carrier_frequency = sat_freq * observed_frequency;

    grpdly = 0.0;

    receiver_gain = 1.0;
    if (params.find("rxGain") != params.end())
    {
        receiver_gain = std::stod(params["rxGain"]);
    }

    // Populate udict_acqus for backward compatibility / metadata consistency
    udict_acqus["TD"] = std::to_string(ndata_bruker);
    udict_acqus["SW_h"] = std::to_string(spectral_width);
    udict_acqus["SFO1"] = std::to_string(observed_frequency);
    udict_acqus["O1"] = std::to_string(carrier_frequency);
    udict_acqus["GRPDLY"] = "0.0";
    udict_acqus["AQ_mod"] = "1"; // complex
    if (params.find("rxChannel") != params.end())
    {
        udict_acqus["NUC1"] = params["rxChannel"];
    }

    data_type = FID_DATA_TYPE_FLOAT32;

    fid_data_float.clear();
    fid_data_float.resize(ndata_bruker * nspectra, 0.0f);

    for (int i = 0; i < nspectra; i++)
    {
        std::string fid_file = fid_data_file_names[i];
        FILE *fp = fopen(fid_file.c_str(), "rb");
        if (fp == NULL)
        {
            std::cout << "Error: cannot open Magritek FID file " << fid_file << std::endl;
            return false;
        }

        int32_t header[8];
        if (fread(header, sizeof(int32_t), 8, fp) != 8)
        {
            std::cout << "Error: cannot read 32-byte header from file " << fid_file << std::endl;
            fclose(fp);
            return false;
        }

        if (header[3] != 504)
        {
            std::cout << "Error: validation failed for header[3] in " << fid_file << ". Expected 504, got " << header[3] << std::endl;
            fclose(fp);
            return false;
        }
        if (header[4] != nr_pnts)
        {
            std::cout << "Error: validation failed for header[4] in " << fid_file << ". Expected " << nr_pnts << ", got " << header[4] << std::endl;
            fclose(fp);
            return false;
        }
        if (header[5] != 1 || header[6] != 1 || header[7] != 1)
        {
            std::cout << "Error: validation failed for header[5-7] in " << fid_file << ". Expected 1, 1, 1, got " << header[5] << ", " << header[6] << ", " << header[7] << std::endl;
            fclose(fp);
            return false;
        }

        std::vector<float> temp_data(nr_pnts * 3);
        if (fread(temp_data.data(), sizeof(float), nr_pnts * 3, fp) != static_cast<size_t>(nr_pnts * 3))
        {
            std::cout << "Error: cannot read " << (nr_pnts * 3) << " floats from file " << fid_file << std::endl;
            fclose(fp);
            return false;
        }

        std::copy(temp_data.begin() + nr_pnts, temp_data.end(), fid_data_float.begin() + i * ndata_bruker);

        // Negate the imaginary part to conjugate the time-domain signal, un-flipping the frequency spectrum
        for (int k = 0; k < nr_pnts; k++)
        {
            fid_data_float[i * ndata_bruker + k * 2 + 1] = -fid_data_float[i * ndata_bruker + k * 2 + 1];
        }

        fclose(fp);
    }

    std::cout << "Successfully read Magritek FID data from " << nspectra << " file(s)." << std::endl;

    return true;
}



bool fid_1d::process_dictionary()
{
    b_read_from_ft1 = false; // default, we are not reading from ft1 file
    /**
     * Check udic_acqus["AQ_mod"] to determine the data type: real or complex
     */
    if (udict_acqus["AQ_mod"] == "3" || udict_acqus["AQ_mod"] == "1")
    {
        data_complexity = FID_DATA_COMPLEXITY_COMPLEX;
    }
    else
    {
        data_complexity = FID_DATA_COMPLEXITY_REAL;
    }

    /**
     * Check udic_acqus["BYTORDA"] to determine the data type: int32 or float32
     */
    if (udict_acqus["DTYPA"] == "2")
    {
        data_type = FID_DATA_TYPE_FLOAT64;
    }
    else
    {
        data_type = FID_DATA_TYPE_INT32;
    }

    /**
     * check udic_acqus["TD"] to determine size of fid data
     */
    if (udict_acqus.find("TD") == udict_acqus.end())
    {
        std::cout << "Error: cannot find TD in acqus file" << std::endl;
        return false;
    }

    int td0 = std::stoi(udict_acqus["TD"]);

    /**
     * According to Bruker manu,
     * when data_type = FID_DATA_TYPE_FLOAT64, fid is padded to 128 bytes
     * when data_type = FID_DATA_TYPE_INT32, fid is padded to 256 bytes
     */

    if (data_type == FID_DATA_TYPE_INT32)
    {
        int ntemp = int(std::ceil((double)td0 / 256.0));
        ndata_bruker = ntemp * 256;
    }
    else
    {
        int ntemp = int(std::ceil((double)td0 / 128.0));
        ndata_bruker = ntemp * 128;
    }

    /**
     * Bruker TD is the number of data points, not the number of complex points for complex data
     * But we define ndata as the number of complex points
     */
    if (data_complexity == FID_DATA_COMPLEXITY_COMPLEX)
    {
        ndata = ndata_bruker / 2;
        ndata_original = td0 / 2;
    }
    else
    {
        ndata = ndata_bruker;
        ndata_original = td0;
    }

    /**
     * Now setup the following variables:
     * spectral_width, observed_frequency, carrier_frequency
     */
    if (udict_acqus.find("SW_h") != udict_acqus.end())
    {
        spectral_width = std::stod(udict_acqus["SW_h"]);
    }
    else
    {
        std::cout << "Error: cannot find SW_h in acqus file" << std::endl;
        return false;
    }

    /**
     * get receiver_gain from acqus if it exists
     */
    if (udict_acqus.find("RG") != udict_acqus.end())
    {
        receiver_gain = std::stod(udict_acqus["RG"]);
    }
    else
    {
        std::cout << "Warning: cannot find RG in acqus file" << std::endl;
        receiver_gain = 1.0; // default
    }

    if (udict_acqus.find("SFO1") != udict_acqus.end())
    {
        observed_frequency = std::stod(udict_acqus["SFO1"]);
    }
    else
    {
        std::cout << "Error: cannot find SFO1 in acqus file" << std::endl;
        return false;
    }

    if (udict_acqus.find("O1") != udict_acqus.end())
    {
        carrier_frequency = std::stod(udict_acqus["O1"]);
    }
    else
    {
        std::cout << "Error: cannot find O1 in acqus file" << std::endl;
        return false;
    }

    /**
     * get parameters "GRPDLY" from acqus file
     */
    if (udict_acqus.find("GRPDLY") != udict_acqus.end())
    {
        grpdly = std::stod(udict_acqus["GRPDLY"]);
    }
    else
    {
        std::cout << "Error: cannot find GRPDLY in acqus file" << std::endl;
        return false;
    }

    /**
     * grpdly must > 0. Otherwiae it is from an early days Bruker spectrometer
     * We don't support this case at this time
     */
    if (grpdly <= 0.0)
    {
        std::cout << "Error: GRPDLY = " << grpdly << " is not supported" << std::endl;
        return false;
    }

    return true;
}




/**
 * write 1D fid to nmrpipe file
 */
bool fid_1d::write_nmrpipe_fid(const std::string outfname) const
{

    std::map<std::string, std::string> nmrpipe_dict_string;
    std::map<std::string, float> nmrpipe_dict_float;

    /**
     * create nmrpipe header.
     * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
     * from udict_acqus and derived values
     * false means we are saving frq data
     */
    create_nmrpipe_dictionary(false, nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * define header vector, because the function call nmrpipe_dictionary_to_header won't apply for memory
     */
    std::vector<float> nmrpipe_header_data(512, 0.0f);

    nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * write temp_data_float to file
     */
    FILE *fp = fopen(outfname.c_str(), "wb");
    if (fp == NULL)
    {
        std::cout << "Error: cannot open file " << outfname << std::endl;
        return false;
    }

    fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);

    /**
     * convert fid_data_float to temp_data_float, real part only: 0,2,4,6,...
     */
    std::vector<float> temp_data_float;
    temp_data_float.resize(ndata_bruker);
    for (int i = 0; i < ndata; i++)
    {
        temp_data_float[i] = (float)fid_data_float[i * 2];
    }
    fwrite(temp_data_float.data(), sizeof(float), ndata_bruker, fp);

    /**
     * convert fid_data_float to temp_data_float, imaginary part only: 1,3,5,7,...
     */
    for (int i = 0; i < ndata; i++)
    {
        temp_data_float[i] = (float)fid_data_float[i * 2 + 1];
    }
    fwrite(temp_data_float.data(), sizeof(float), ndata_bruker, fp);

    fclose(fp);

    return true;
}

/**
 * @brief fid_1d::run_fft_and_rm_bruker_filter: run fft on fid_data_float
 * @return true on success, false on failure
 * Will modify spectrum_real and spectrum_imag
 */
bool fid_1d::run_fft_and_rm_bruker_filter()
{
    /**
     * get ndata_power_of_2, which is the smallest power of 2
     * that is larger than or equal ndata
     */
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    ndata_frq = ndata_power_of_2 * zf;

    /**
     * Run apoization. We set ndata*2 in run_apodization because we are working on complex data
    */
    apod->set_sw(spectral_width);
    apod->set_n(ndata_original);
    

    /**
     * Clear spectrum_real and spectrum_imag. Do not resize because we will use insert to add data later
    */
    spectrum_real.clear();
    spectrum_imag.clear();


    for(int i=0;i<nspectra;i++)
    {   
        std::vector<float> temp_fid_data_float(0);

        /**
         * Copy fid_data_float to temp_fid_data_float
         */
        temp_fid_data_float.insert(temp_fid_data_float.end(),fid_data_float.begin()+i*ndata_bruker,fid_data_float.begin()+(i+1)*ndata_bruker);
       
        apod->run_apodization(temp_fid_data_float.data(), ndata*2, true /**complex data*/);
        
        kiss_fft_cfg cfg;
        kiss_fft_cpx *in, *out;

        in = new kiss_fft_cpx[ndata_frq];
        out = new kiss_fft_cpx[ndata_frq];
        for (int i = 0; i < ndata; i++)
        {
            in[i].r = temp_fid_data_float[i * 2];
            in[i].i = temp_fid_data_float[i * 2 + 1];
        }

        /**
         * fill remaining of in with 0
         */
        for (int i = ndata; i < ndata_frq; i++)
        {
            in[i].r = 0.0;
            in[i].i = 0.0;
        }

        if ((cfg = kiss_fft_alloc(ndata_frq, 0, NULL, NULL)) != NULL)
        {
            kiss_fft(cfg, in, out);
            free(cfg);
        }
        else
        {
            std::cerr << "Error: cannot allocate memory for fft" << std::endl;
            return false;
        }

        std::vector<float> temp_spectrum_real(ndata_frq, 0.0f);
        std::vector<float> temp_spectrum_imag(ndata_frq, 0.0f);

        for (int i = 0; i < ndata_frq; i++)
        {
            temp_spectrum_real[i] = out[i].r / sqrt(float(ndata_frq));
            temp_spectrum_imag[i] = out[i].i / sqrt(float(ndata_frq)); // scale by sqrt(ndata_frq) to be consistent with standard fft
        }

        /**
         * Important: this step need to be done before fft result swap and flip
         */
        remove_bruker_digitizer_filter(grpdly,temp_spectrum_real,temp_spectrum_imag);

        /**
         * To be consistent with nmrPipe, we also swap left and right halves of spectrum
         * then flip the spectrum
         * and apply receiver gain
         */
        std::vector<float> spectrum_real_reoganized(ndata_frq, 0.0f);
        std::vector<float> spectrum_imag_reoganized(ndata_frq, 0.0f);

        for (int i = 0; i < ndata_frq; i++)
        {
            spectrum_real_reoganized[ndata_frq - 1 - i] = (float)temp_spectrum_real[(i + ndata_frq / 2) % ndata_frq] * receiver_gain;
            spectrum_imag_reoganized[ndata_frq - 1 - i] = (float)temp_spectrum_imag[(i + ndata_frq / 2) % ndata_frq] * receiver_gain;
        }

        /**
         * free memory
        */
        delete [] in;
        delete [] out;

        /**
         * Copy to spectrum_real and spectrum_imag
        */

        spectrum_real.insert(spectrum_real.end(),spectrum_real_reoganized.begin(),spectrum_real_reoganized.end());
        spectrum_imag.insert(spectrum_imag.end(),spectrum_imag_reoganized.begin(),spectrum_imag_reoganized.end());
    }

    return true;
}

bool fid_1d::set_up_apodization(apodization *apod_)
{
    apod = apod_;
    return true;
};


bool fid_1d::set_up_apodization_from_string(const std::string &apod_string)
{
    apod = new apodization(apod_string);
    return true;
}



/**
 * @brief create_nmrpipe_dictionary: create nmrpipe dictionary from udict_acqus and derived values
 * @param b_frq: true if we are saving frq data, false if we are saving time data
 * @param nmrpipe_dict_string: output dictionary for string values
 * @param nmrpipe_dict_float: output dictionary for float values
 */
bool fid_1d::create_nmrpipe_dictionary(bool b_frq, std::map<std::string, std::string> &nmrpipe_dict_string, std::map<std::string, float> &nmrpipe_dict_float) const
{
    /**
     * @brief create_default_nmrpipe_dictionary: create empty nmrpipe header
     */
    nmrPipe::create_default_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * Fill in some parameters from what we have
     */
    nmrpipe_dict_float["FDDIMCOUNT"] = 1.0f; // 1D data
    nmrpipe_dict_float["FD2DPHASE"] = 0.0f;  // for 1D data, direct dimension phase is 0

    /**
     * Copied from nmrglue. no idea what they mean
     */
    if (nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF2QUADFLAG"] == nmrpipe_dict_float["FDF3QUADFLAG"] && nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF4QUADFLAG"] == 1.0f)
    {
        nmrpipe_dict_float["FDQUADFLAG"] = 1.0f;
    }

    /**
     * Some examples of parameters that we need to put in nmrpipe_dict_float
     * 'sw': 12019.2307692308
     *  'complex':True
     *   'obs': 600.06
     *  'car': 2820.28200001605
     *   'size': 32768
     *  'label': '1H'
     */

    nmrpipe_dict_float["FDF2SW"] = spectral_width;
    nmrpipe_dict_float["FDF2OBS"] = observed_frequency;
    nmrpipe_dict_float["FDF2CAR"] = carrier_frequency / observed_frequency; // normalized to observed frequency, per nmrPipe convention
    nmrpipe_dict_string["FDF2LABEL"] = "1H";                                // hardcoded for now

    if (data_complexity == FID_DATA_COMPLEXITY_COMPLEX)
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 0.0f;
    }
    else
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 1.0f;
    }

    if (b_frq)
    {
        /**
         * we are saving frq data, so set FDF2FTFLAG to 1 and set value for FDF2FTSIZE
         * set FDF2TDSIZE if we have time data
         */
        nmrpipe_dict_float["FDF2FTSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDF2FTFLAG"] = 1.0f; // frq data, instead of time

        nmrpipe_dict_float["FDSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDREALSIZE"] = ndata_frq;

        /**
         * ft2 file also keep original FID data size
         * But I am not sure nmrPipde does the same!!
        */
        nmrpipe_dict_float["FDF2TDSIZE"] = ndata;
    }
    else
    {
        /**
         * we are saving time data, so set FDF2FTFLAG to 0 and set value for FDF2TDSIZE
         * set FDF2FTSIZE if we have frq data
         */
        nmrpipe_dict_float["FDF2TDSIZE"] = ndata;
        nmrpipe_dict_float["FDF2FTFLAG"] = 0.0f; // time data, instead of frq

        nmrpipe_dict_float["FDSIZE"] = ndata;
        nmrpipe_dict_float["FDREALSIZE"] = ndata;
    }

    /**
     * set apodization information. meaningless for time domain data
     */
    nmrpipe_dict_float["FDF2APOD"] = ndata_frq;
    nmrpipe_dict_float["FDF2CENTER"] = int(ndata_frq / 2.0f) + 1.0f;

    /**
     * copied from nmrglue. no idea what they mean
     */
    nmrpipe_dict_float["FDF2ORIG"] = (nmrpipe_dict_float["FDF2CAR"] * nmrpipe_dict_float["FDF2OBS"] - nmrpipe_dict_float["FDF2SW"] * (ndata_frq - nmrpipe_dict_float["FDF2CENTER"]) / ndata_frq);

    return true;
}

/**
 * @brief fid_1d::write_nmrpipe_ft1: write 1D spectrum to nmrpipe file
 * Before writing, define nmrpipe header, set values from udict_acqus and derived values
 * @param outfname: output file name
 * @return true on success, false on failure
 */
bool fid_1d::write_nmrpipe_ft1(std::string outfname)
{
    std::map<std::string, std::string> nmrpipe_dict_string;
    std::map<std::string, float> nmrpipe_dict_float;

    /**
     * create nmrpipe header.
     * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
     * from udict_acqus and derived values
     * True means we are saving frq data
     */
    create_nmrpipe_dictionary(true, nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * define header vector, because the function call nmrpipe_dictionary_to_header won't apply for memory
     * We do not need to do this when reading from nmrPipe ft1 file (where header is read from file.)
     */
    if(b_read_from_ft1 == false)
    {
        nmrpipe_header_data.clear();
        nmrpipe_header_data.resize(512, 0.0f);
    }

    /**
     * Even when we read from .ft1 file, we may still change nmrpipe_header_data if warranted.
    */
    nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * now write nmrpipe_header_data and spectral to the file
     * if outfname is empty, we don't write to file. This function is called to generate header only
     */
    if (outfname != "")
    {
        /**
         * now write nmrpipe_header_data and spectral to the file
         */
        FILE *fp = fopen(outfname.c_str(), "wb");
        if (fp == NULL)
        {
            std::cerr << "Error: cannot open file " << outfname << " for writing" << std::endl;
            return false;
        }
        fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);
        fwrite(spectrum_real.data(), sizeof(float), ndata_frq, fp);
        fwrite(spectrum_imag.data(), sizeof(float), ndata_frq, fp);
        fclose(fp);

        if(nspectra>1)
        {
            /**
             * Find filename extension: after last . in outfname, such as .ft1.
             * In case there is no . in outfname, ext will be empty and basename will be outfname
             */
            std::string ext = outfname.substr(outfname.find_last_of(".") + 1);
            std::string basename = outfname.substr(0, outfname.find_last_of("."));

            for(int i=1;i<nspectra;i++)
            {
                std::string outfname2 = basename + "_" + std::to_string(i) + "." + ext;
                FILE *fp2 = fopen(outfname2.c_str(), "wb");
                if (fp2 == NULL)
                {
                    std::cerr << "Error: cannot open file " << outfname2 << " for writing" << std::endl;
                    return false;
                }
                fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp2);
                fwrite(spectrum_real.data()+ndata_frq*i, sizeof(float), ndata_frq, fp2);
                fwrite(spectrum_imag.data()+ndata_frq*i, sizeof(float), ndata_frq, fp2);
                fclose(fp2);
            }
        }
    }

    return true;
}



/**
 * @brief fid_1d::get_spectrum_header: return header as a vector of float
 */
std::vector<float> fid_1d::get_spectrum_header(void) const
{
    return nmrpipe_header_data;
};

/**
 * @brief fid_1d::get_spectrum_real: return real part of spectrum as a vector of float
 */
std::vector<float> fid_1d::get_spectrum_real(void) const
{
    return spectrum_real;
};

/**
 * @brief fid_1d::get_spectrum_imag: return imaginary part of spectrum as a vector of float
 */
std::vector<float> fid_1d::get_spectrum_imag(void) const
{
    return spectrum_imag;
};


bool fid_1d::init(double user_, double user2_, double noise_)
{
    user_scale = user_;
    user_scale2 = user_;
    noise_level = noise_;
    return true;
};


/**
 * @brief write spectrum in josn format as an array, with two or three columns: ppm and intensity (and image)
*/
bool fid_1d::write_spectrum_json(std::string outfname)
{
    std::ofstream outfile;
    outfile.open(outfname);
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << outfname << std::endl;
        return false;
    }

    Json::Value root, data;

    for (int j = 0; j < ndata_frq; j++)
    {
        data[j][0] = begin1 + j * step1;
        data[j][1] = spectrum_real[j];
        /**
         * @brief if spe_image has same size as spect, save it as well
        */
        if(spectrum_imag.size()==ndata_frq)
        {
            data[j][2] = spectrum_imag[j];
        }
    }
    root["spectrum"] = data;
    outfile << root;
    return true;
}


/**
 * @brief read frq domain spectrum from three vectors (buffers) in pipe format
 * Similar to read_spectrum_ft, but get the header and spectrum from three vectors instead of a file
*/
bool fid_1d::direct_set_spectrum_from_nmrpipe(const std::vector<float> &_header, const std::vector<float> &real, const std::vector<float> &imag)
{

    /**
     * copy _header (vector) to header (float *512)
    */
    for (int i = 0; i < 512; i++)
    {
        nmrpipe_header_data[i] = _header[i];
    }

    if (nmrpipe_header_data[10 - 1] != 1.0f)
    {
        std::cout << "Wrong file format, dimension (header[9]) is " << nmrpipe_header_data[9] << std::endl;
        return false;
    }

    ndata_frq = int(nmrpipe_header_data[220 - 1]) * int(nmrpipe_header_data[100 - 1]); // one of them is 1, the other one is true dimension

    spectral_width = double(nmrpipe_header_data[101 - 1]);
    observed_frequency = double(nmrpipe_header_data[120 - 1]);
    origin = double(nmrpipe_header_data[102 - 1]);
    carrier_frequency = double(nmrpipe_header_data[67 - 1]) * observed_frequency;

    /**
     * Sometimes, original acqusition size is needed. 
    */
    ndata = int(nmrpipe_header_data[386]);
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    stop1 = origin / observed_frequency;
    begin1 = stop1 + spectral_width / observed_frequency;
    step1 = (stop1 - begin1) / (ndata_frq); // direct dimension
    begin1 += step1;                    // here we have to + step because we want to be consistent with nmrpipe program
                                        // I have no idea why nmrpipe is different than topspin

    spectrum_real=real;
    spectrum_imag=imag;
    
    if(n_verbose>0)
    {
        std::cout << "Spectrum size is " << ndata_frq << std::endl;
        std::cout << "From " << begin1 << " to " << stop1 << " and step is " << step1 << std::endl;
    }

    if (noise_level < 1e-20)
    {
        est_noise_level();
    }

    return true;
}

/**
 * @brief read 1D spectrum
 * 
 * @param infname input file name
 * @param b_negative true: allow negative peaks, false: only positive peaks. Default is true
 * noise_level is estimated if it is zero
 * @return true 
 */

bool fid_1d::read_spectrum(std::string infname, bool b_negative)
{
    bool b_read = 0;

    input_spectrum_fname = infname; // save for later use

    std::string sldw(".ldw"); // my own format
    std::string stxt(".txt"); // topspin format. saved by totxt command
    std::string sft1(".ft1"); // nmrPipe format
    std::string sjson(".json"); // json format
    std::string scsv(".csv"); // csv format, used by Mnova software
    std::string sparky(".ucsf"); // sparky format, used by sparky software

    if (std::equal(stxt.rbegin(), stxt.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_txt(infname);
    }
    else if (std::equal(sldw.rbegin(), sldw.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_ldw(infname);
    }
    else if (std::equal(sft1.rbegin(), sft1.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_ft(infname);
    }
    else if (std::equal(sjson.rbegin(), sjson.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_json(infname);
    }
    else if (std::equal(scsv.rbegin(), scsv.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_csv(infname);
    }
    else if (std::equal(sparky.rbegin(), sparky.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_sparky(infname);
    }
    else
    {
        b_read = false;
    }

    if(n_verbose>0)
    {
        std::cout << "Spectrum size is " << ndata_frq << std::endl;
        std::cout << "From " << begin1 << " to " << stop1 << " and step is " << step1 << std::endl;
    }

    if (noise_level < 1e-20)
    {
        est_noise_level();
    }

    if(b_negative==false)
    {
        std::cout<<"Set negative data points to zero."<<std::endl;
        for(int i=0;i<spectrum_real.size();i++)
        {
            spectrum_real[i]=std::max(spectrum_real[i],0.0f);
        }
    }

    nspectra=1;
    return b_read;
};





/**
 * @brief read spectrum in csv format by Mnova software
*/
bool fid_1d::read_spectrum_csv(std::string fname)
{
    std::ifstream fin(fname);
    if (!fin)
        return false;

    std::string line;
    std::vector<std::string> line_split;
    std::stringstream ss;
    std::vector<float> ppm;
    std::vector<float> amplitude;

    while (std::getline(fin, line))
    {   
        /**
         * Remove leading or trailing spaces
         * Return false if the line is empty
        */
        if(ldw_math_spectrum_1d::trim(line)==false)
        {
            continue;
        }
        /**
         * Skip comment lines, start with #
        */
        if (line[0] == '#')
        {
            continue;
        }

        /**
         * Split line by space(s)
        */
        line_split.clear();
        ss.clear();
        ss.str(line);
        std::string temp;
        while (ss >> temp)
        {
            line_split.push_back(temp);
        }

        /**
         * The first column is ppm, the second column is amplitude. Ignore the rest
        */
        if(line_split.size()>=2)
        {
            ppm.push_back(std::stof(line_split[0]));
            amplitude.push_back(std::stof(line_split[1]));
        }
    }

    ndata_frq = amplitude.size();
    begin1 = ppm[0];
    stop1 = ppm[ndata_frq - 1];
    step1 = (stop1 - begin1) / (ndata_frq - 1);

    observed_frequency = 850.0; //default value, suppose it is 850
    carrier_frequency = 4.7; //default value, suppose it is 4.7 ppm

    spectrum_real = amplitude;

    ndata_frq = amplitude.size();

    /**
     * In this case, we suppose n_zf is 2
    */
    ndata = ndata_frq/2;
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    return true;

}
/**
 * @brief read spectrum in json format by Gissmo project
*/
bool fid_1d::read_spectrum_json(std::string infname)
{

    Json::Value root;
    std::ifstream fin(infname);
    if (!fin)
        return false;

    fin >> root;

    Json::Value data1, data2;
    data1 = root[0]; // ppm
    data2 = root[1]; // amplitude

    std::vector<float> ppm;

    for (int i = 0; i < data2.size(); i += 1)
    {
        if(data2[i].isDouble()==true)
        {
            spectrum_real.push_back(data2[i].asDouble());
        }
        else if(data2[i].isString()==true)
        {
            spectrum_real.push_back(std::stof(data2[i].asString()));
        }
        else
        {
            std::cout<<"Error: fid_1d::read_spectrum_json, data2[i] is not double or string."<<std::endl;
            return false;
        }

        if(data1[i].isDouble()==true)
        {
            ppm.push_back(data1[i].asDouble());
        }
        else if(data1[i].isString()==true)
        {
            ppm.push_back(std::stof(data1[i].asString()));
        }
        else
        {
            std::cout<<"Error: fid_1d::read_spectrum_json, data1[i] is not double or string."<<std::endl;
            return false;
        }
    }
    // std::reverse(spe.begin(),spe.end());

    ndata_frq = spectrum_real.size();
    begin1 = ppm[0];
    stop1 = ppm[ndata_frq - 1];
    step1 = (stop1 - begin1) / (ndata_frq - 1);

    observed_frequency = 850.0; //default value, suppose it is 850
    carrier_frequency = 4.7; //default value, suppose it is 4.7 ppm

    ndata_frq = spectrum_real.size();
    /**
     * In this case, we suppose n_zf is 2
    */
    ndata = ndata_frq/2;
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    return true;
};

/**
 * This function will replace fid_1d::read_spectrum(), but only support reading nmrPipe ft1 file and means for web assembly to direct set spectrum data from memory buffer
*/
bool fid_1d::read_first_spectrum_from_buffer(std::vector<float> &header, std::vector<float> &spectrum_real_, std::vector<float> &spectrum_imag)
{
    /**
     * copy first 512 float32 to nmrpipe_header_data
     */
    if(header.size()<512)
    {
        std::cout << "Error: buffer size is less than 512, cannot read nmrPipe header." << std::endl;
        return false;
    }
    /**
     * C++17 allows us to use std::move to avoid copying the header data. Simply transfer ownership of the header vector to nmrpipe_header_data.
    */
    nmrpipe_header_data = std::move(header); 

    process_nmrpipe_header();  

    spectrum_real = std::move(spectrum_real_); // move the spectrum_real vector to the class member

    if(spectrum_imag.size()>0)
    {
        spectrum_imag = std::move(spectrum_imag); // move the spectrum_imag vector to the class member
    }
    else
    {
        spectrum_imag.clear(); // clear the imaginary part if it is empty
    }
   
    if (noise_level < 1e-20)
    {
        est_noise_level();
    }
    return true;
}

/**
 * @brief read 1D spectrum from nmrPipe ft1 file. Will try to read imaginary part as well
*/
bool fid_1d::read_spectrum_ft(std::string infname)
{
    FILE *fp;
    fp = fopen(infname.c_str(), "rb");
    if (fp == NULL)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }
    unsigned int temp = fread(nmrpipe_header_data.data(), sizeof(float), 512, fp);
    if (temp != 512)
    {
        std::cout << "Wrong file format, can't read 2048 bytes of head information from " << infname << std::endl;
        fclose(fp);
        return false;
    }
    process_nmrpipe_header();    

    spectrum_real.clear();
    spectrum_real.resize(ndata_frq);
    temp = fread(spectrum_real.data(), sizeof(float), ndata_frq, fp);

    if (temp != ndata_frq)
    {
        std::cout << "Read nmrPipe 1D error, spectrum size is " << ndata_frq << " but I can only read in " << temp << " data points." << std::endl;
    }

    spectrum_imag.resize(ndata_frq);
    temp = fread(spectrum_imag.data(), sizeof(float), ndata_frq, fp);
    if (temp == ndata_frq)
    {
        if(n_verbose>0) std::cout << "Read nmrPipe 1D imaginary part successully." << std::endl;
    }
    else
    {
        if(n_verbose>0) std::cout << "Read nmrPipe 1D  imaginary part failed." << std::endl;
        spectrum_imag.clear();
    }

    fclose(fp);


    return true;
};

bool fid_1d::process_nmrpipe_header()
{
    if (nmrpipe_header_data[10 - 1] != 1.0f)
    {
        std::cout << "Wrong file format, dimension (header[9]) is " << nmrpipe_header_data[0] << std::endl;
        return false;
    }

    if (nmrpipe_header_data[222 - 1] == 0.0f) // transposed?
    {
        ndata_frq = int(nmrpipe_header_data[100 - 1]); //FDSIZE
    }
    else
    {
        ndata_frq = int(nmrpipe_header_data[220 - 1]); //FDSPECNUM
    }

    ndata_frq = int(nmrpipe_header_data[220 - 1]) * int(nmrpipe_header_data[100 - 1]); // one of them is 1, the other one is true dimension

    spectral_width = double(nmrpipe_header_data[101 - 1]); //FDF2SW
    observed_frequency = double(nmrpipe_header_data[120 - 1]); //FDF2OBS
    origin = double(nmrpipe_header_data[102 - 1]); //FDF2ORIG
    carrier_frequency = double(nmrpipe_header_data[67 - 1]) * observed_frequency; //FDF2CAR

    /**
     * Sometimes, original acqusition size is needed. 
    */
    ndata = int(nmrpipe_header_data[386]);
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    stop1 = origin / observed_frequency;
    begin1 = stop1 + spectral_width / observed_frequency; 
    step1 = (stop1 - begin1) / (ndata_frq); 
    /**
     * here we have to + step because we want to be consistent with nmrpipe program
     * I have no idea why nmrpipe is different than topspin
    */
    begin1 += step1;  

    b_read_from_ft1 = true; // set this flag to true, so we know we read from ft1 file

    return true;
}


/**
 * @brief write 1D spectrum to a file. Format is decided by file extension
 * .ft1 .json or .txt
*/
bool fid_1d::write_spectrum(std::string fname)
{
    bool b_write = false;
    std::string path_name, file_name, file_name_ext;
    /**
     * Get path name, file name and file extension
    */
    ldw_math_spectrum_1d::SplitFilename(fname, path_name, file_name, file_name_ext);

    if(file_name_ext=="ft1")
    {
        b_write=write_nmrpipe_ft1(fname);
    }
    else if(file_name_ext=="json")
    {
        b_write=write_spectrum_json(fname);
    }
    else if(file_name_ext=="txt")
    {
        b_write=write_spectrum_txt(fname);
    }
    else if(file_name_ext=="csv")
    {
        b_write=write_spectrum_csv(fname);
    }
    else
    {
        std::cout<<"Error: fid_1d::write_spectrum, unknown file extension "<<file_name_ext<<std::endl;
        return false;
    }
    return b_write;
}



//simple text file format, defined by myself
bool fid_1d::read_spectrum_ldw(std::string infname)
{
    std::ifstream fin(infname);

    float data;
    while (fin >> data)
    {
        spectrum_real.push_back(data);
        // fin>>data;
    }
    fin.close();

    ndata_frq = spectrum_real.size() - 2;

    stop1 = spectrum_real[ndata_frq + 1];
    begin1 = spectrum_real[ndata_frq];
    step1 = (stop1 - begin1) / ndata_frq;

    /**
     * In this case, we suppose n_zf is 2
    */
    ndata = ndata_frq/2;
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    spectrum_real.resize(ndata_frq);

    return true;
}

//ascii file saved by Topspin totxt command
bool fid_1d::read_spectrum_txt(std::string infname)
{
    std::string line;
    std::ifstream fin(infname);

    bool b_left = false;
    bool b_right = false;
    bool b_size = false;

    spectrum_real.clear();

    //read line by line
    while (std::getline(fin, line))
    {
        //if line is empty, skip
        if (line.empty())
        {
            continue;
        }

        /**
         * Example header:
         * # LEFT = 12.764570236206055 ppm. RIGHT = -3.217077331696382 ppm.
         * #
         * # SIZE = 65536 ( = number of points)
        */

        //if line starts with #, look for key words LEFT, RIGHT in one line and key words SIZE in another line
        if (line[0] == '#')
        {
            if (line.find("LEFT") != std::string::npos && line.find("RIGHT") != std::string::npos)
            {
                std::string temp;
                //get substring after LEFT
                std::string line_part1 = line.substr(line.find("LEFT") + 4 + 1); // + 1 to skip space
                std::istringstream iss1(line_part1);
                iss1 >> temp  >> begin1; //LEFT = 12.09875 ppm
                b_left = true;
            
                std::string line_part2 = line.substr(line.find("RIGHT") + 5 + 1); // + 1 to skip space
                std::istringstream iss2(line_part2);
                iss2 >> temp >> stop1; //RIGHT = -3.1234 ppm
                b_right = true;
            }

            if(line.find("SIZE") != std::string::npos)
            {
                std::string temp;
                std::string line_part = line.substr(line.find("SIZE") + 4 + 1); // + 1 to skip space
                std::istringstream iss(line_part);
                iss >> temp >> ndata_frq; // SIZE = 1024
                b_size = true;
            }
            continue;
        }

        /* *All other lines are data and they are all float
        * They should be after the two lines with key words LEFT, RIGHT and SIZE
        * One number per line
        * 
        * Example: (second part is for imaginary part, may not exist)
        * -3747.75-17118.0i
        * -3277.5+17781.5i
        * 2807.25+17781.5i
        */
        if (b_left == true && b_right == true && b_size == true)
        {
            /**
             * First, decide if this is a complex spectrum
            */
            if(line.find("i") != std::string::npos)
            {
                /**
                 * remove the last character, which is i
                */
                line=line.substr(0,line.size()-1);
                std::string line_part=line;

                /**
                 * remove first character if it is a + or -
                */
                if(line_part[0]=='+' || line_part[0]=='-')
                {
                    line_part=line_part.substr(1);
                }

                /**
                 * seperate real and imaginary part, using the last + or - as delimiter
                 * We use line, not line_part, because we need to keep the sign of the first number
                */
                std::string real_part=line.substr(0,line.find_last_of("+-"));
                std::string imag_part=line.substr(line.find_last_of("+-"));
                
                /**
                 * convert string to float, and push back to spect and spe_image
                */
                spectrum_real.push_back(std::stof(real_part));
                /**
                * Reasons unknown, but the sign of the imaginary part is reversed in Bruker's txt file
                */
                spectrum_imag.push_back(-std::stof(imag_part));
            }
            else
            {
                /**
                 * This is a real spectrum. Push back to spect only
                */
                float data=std::stof(line);
                spectrum_real.push_back(data);
            }
        }
    }

    if(spectrum_real.size()!=ndata_frq)
    {
        std::cout<<"Error: fid_1d::read_spectrum_txt, ndata_frq is not equal to the number of data points. Set ndata_frq=spect.size()"<<std::endl;
        ndata_frq = spectrum_real.size();
    }

    if(spectrum_imag.size()!=ndata_frq)
    {
        std::cout<<"Error: fid_1d::read_spectrum_txt, ndata_frq is not equal to the number of data points. Remove imaginary data."<<std::endl;
        spectrum_imag.clear(); //spe_image.size()==0 is used to indicate that there is no imaginary part
    }

    fin.close();

    step1 = (stop1 - begin1) / ndata_frq;

    observed_frequency = 850.0; //default value, suppose it is 850
    carrier_frequency = 4.7; //default value, suppose it is 4.7 ppm

    /**
     * In this case, we suppose n_zf is 2
    */
    ndata = ndata_frq/2;
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

/**
 * For debug
 */
// #ifdef DEBUG
//     std::ofstream fout("spect.txt");
//     for (int i = 0; i < spect.size(); i++)
//     {
//         fout << spect[i] << " " << spe_image[i] << std::endl;
//     }
//     fout.close();
// #endif

    return true;
}


bool fid_1d::read_spectrum_sparky(std::string infname)
{
    FILE *fp;

    char buffer[10];
    int temp;
    float center1;

    fp = fopen(infname.c_str(), "rb");
    if (fp == NULL)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }

    if (fread(buffer, 1, 10, fp) != 10)
    {
        std::cout << "Error reading sparky file header" << std::endl;
        fclose(fp);
        return false;
    }
    // std::cout<<buffer<<std::endl;

    if (fread(buffer, 1, 1, fp) != 1)
    {
        std::cout << "Error reading sparky file dimension" << std::endl;
        fclose(fp);
        return false;
    }
    temp = int(buffer[0]);
    if (temp != 1)
    {
        std::cout << "Error in sparky format file, dimension is not 1" << std::endl;
        fclose(fp);
        return false;
    }

    if (fread(buffer, 1, 1, fp) != 1)
    {
        std::cout << "Error reading sparky file quad type" << std::endl;
        fclose(fp);
        return false;
    }
    fseek(fp, 1, SEEK_CUR);
    temp = int(buffer[0]);
    if (temp != 1)
    {
        std::cout << "Error in sparky format file, it is not in real data" << std::endl;
        fclose(fp);
        return false;
    }

    if (fread(buffer, 1, 1, fp) != 1)
    {
        std::cout << "Error reading sparky file version" << std::endl;
        fclose(fp);
        return false;
    }
    // std::cout<<"Version is "<< int(buffer[0])<<std::endl;
    fseek(fp, 166, SEEK_CUR);  //at location 180

    if (fread(buffer, 1, 6, fp) != 6)
    {
        std::cout << "Error reading direct dimension nucleus" << std::endl;
        fclose(fp);
        return false;
    }
    buffer[6] = '\0';
    std::cout << "Direct dimension nuleus " << buffer << std::endl;
    fseek(fp, 2, SEEK_CUR); //at 188
    ndata_frq = read_int(fp); //at 192
    fseek(fp, 8, SEEK_CUR); //at 200
    observed_frequency = read_float(fp); //at 204
    spectral_width = read_float(fp); //at 208
    center1 = read_float(fp); //at 212
    fseek(fp, 96, SEEK_CUR);  // at location 308

    /**
     * now read the spectrum data. Remember that sparky is in big-endian format
     */
    spectrum_real.clear();
    spectrum_real.resize(ndata_frq);
    for(int i=0;i<ndata_frq;i++)
    {
        float temp_float=read_float(fp);
        /**
         * convert from big-endian to little-endian
        */
        spectrum_real[i] = temp_float;
    }

    float range1 = spectral_width / observed_frequency;
    step1 = -range1 / ndata_frq;
    begin1 = center1 + range1 / 2;
    stop1 = center1 - range1 / 2;

    origin = center1 * observed_frequency - spectral_width / 2;

    std::cout << "Spectrum width are " << spectral_width << " Hz" << std::endl;
    std::cout << "Fields are " << observed_frequency << " mHz" << std::endl;
    std::cout << "Direct dimension size is " << ndata_frq  << std::endl;
    std::cout << "Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;

    fclose(fp);


    return true;
}

/**
 * @brief set spectrum from a vector of data, with ppm information
 * This is the minimal requirement for a spectrum to be used for picking and fitting.
 * It gather similar set of information as read from text file or json file. 
 * read ft1 will get more information, such as SW1, frq1, ref1, etc.
*/
bool fid_1d::set_spectrum_from_data(const std::vector<float> &data, const double begin_, const double step_, const double stop_)
{
    /**
     * Set spect and ndata_frq
    */
    spectrum_real=data;
    ndata_frq=spectrum_real.size();

    spectrum_imag.clear(); //spe_image.size()==0 is used to indicate that there is no imaginary part
    
    /**
     * Set begin1, step1, stop1 for ppm information
    */
    begin1=begin_;
    step1=step_;
    stop1=stop_;

    return true;
}

/**
 * @brief write 1D spectrum to a csv file
*/
bool fid_1d::write_spectrum_csv(std::string outfname)
{
    std::ofstream outfile;
    outfile.open(outfname);
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << outfname << std::endl;
        return false;
    }

    for (int j = 0; j < ndata_frq; j++)
    {
        outfile << begin1 + j * step1<<" "<< spectrum_real[j] << std::endl;
    }
    outfile.close();
    return true;
}

/**
 * @brief write 1D spectrum to a text file
*/
bool fid_1d::write_spectrum_txt(std::string outfname)
{
    std::ofstream outfile;
    outfile.open(outfname);
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << outfname << std::endl;
        return false;
    }

    outfile << "# LEFT = " << begin1 << " ppm. RIGHT = " << stop1 << " ppm." << std::endl;
    outfile << "#" << std::endl;
    outfile << "# SIZE = " << ndata_frq << " ( = number of points)" << std::endl;
    outfile << "#" << std::endl;

    if(spectrum_imag.size()==0)
    {
        for (int j = 0; j < ndata_frq; j++)
        {
            outfile << spectrum_real[j] << std::endl;
        }
    }
    else
    {
        for (int j = 0; j < ndata_frq; j++)
        {
            /**
             * Reasons unknown, but the sign of the imaginary part is reversed in Bruker's txt file
            */
            float temp_data=-spectrum_imag[j];
            if(temp_data>=0.0)
            { 
                outfile << spectrum_real[j] << "+" << temp_data << "i" << std::endl;
            }
            else
            {
                outfile << spectrum_real[j] << temp_data << "i" << std::endl;
            }
        }
    }
    outfile.close();
    return true;
}

/**
 * @brief estimate noise level using a general purpose method: segment by segment variance
*/
bool fid_1d::est_noise_level()
{

    std::vector<double> variances;
    std::vector<double> sums;
    int n_segment = ndata_frq / 32;
    for (int i = 0; i < n_segment; i++)
    {
        double sum = 0;
        for (int j = 0; j < 32; j++)
        {
            sum += spectrum_real[i * 32 + j];
        }
        sum /= 32;
        // get variance of each segment
        double var = 0;
        for (int j = 0; j < 32; j++)
        {
            var += (spectrum_real[i * 32 + j] - sum) * (spectrum_real[i * 32 + j] - sum);
        }
        var /= 32;

        variances.push_back(var);
        sums.push_back(sum);
    }

    // output the sums and variance of each segment
    //  std::ofstream fout("variances.txt");
    //  for(int i=0;i<variances.size();i++)
    //  {
    //      fout<<sums[i]<<" "<<variances[i]<<std::endl;
    //  }
    //  fout.close();

    int n = variances.size() / 4;
    nth_element(variances.begin(), variances.begin() + n, variances.end());
    noise_level = sqrt(variances[n]);
    if(n_verbose>0) std::cout << "Noise level is estiamted to be " << noise_level << ", using a geneal purpose method." << std::endl;

    return true;
}

/**
 * @brief estimate noise level using MAD method of the whole spectrum.
 * Won't work if the spectrum is not phased and baseline corrected
*/
bool fid_1d::est_noise_level_mad()
{

    int ndim = spectrum_real.size();

    if (noise_level < 1e-20) // estimate noise level
    {
        std::vector<float> t = spectrum_real;
        for (unsigned int i = 0; i < t.size(); i++)
        {
            if (t[i] < 0)
                t[i] *= -1;
        }

        std::vector<float> scores = t;
        sort(scores.begin(), scores.end());
        noise_level = scores[scores.size() / 2] * 1.4826;
        if (noise_level <= 0.0)
            noise_level = 0.1; // artificail spectrum w/o noise
        std::cout << "First round, noise level is " << noise_level << std::endl;

        std::vector<int> flag(ndim, 0); // flag

        for (int i = 0; i < ndim; i++)
        {
            if (t[i] > 5.5 * noise_level)
            {
                int xstart = std::max(i - 5, 0);
                int xend = std::min(i + 6, ndim);

                for (int n = xstart; n < xend; n++)
                {
                    flag[n] = 1;
                }
            }
        }
        scores.clear();

        for (int i = 0; i < ndim; i++)
        {
            if (flag[i] == 0)
            {
                scores.push_back(t[i]);
            }
        }

        sort(scores.begin(), scores.end());
        noise_level = scores[scores.size() / 2] * 1.4826;
        if (noise_level <= 0.0)
            noise_level = 0.1; // artificail spectrum w/o noise
        std::cout << "Final noise level is estiamted to be " << noise_level << std::endl;
    }

    return true;
};


/**
 * @brief fid_1d::write_nmrpipe_ft1: write some information
 * This is mainly for web-server
*/
bool fid_1d::write_json(std::string fname)
{
    std::ofstream outfile(fname.c_str());
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    Json::Value root;
    root["ndata"] = ndata;
    root["ndata_frq"] = ndata_frq;
    root["ndata_original"] = ndata_original;
    root["ndata_power_of_2"]=ndata_power_of_2; //We suppose ZF=2 in processing
    /**
     * ref1 is the end of the spectrum
     * carrier_frequency is the middle of the spectrum
     * SW is the total width of the spectrum
     * All in Hz.
     * If read in from a file other than ft1, all will be set to 0
    */
    root["carrier_frequency"] = carrier_frequency;
    root["observed_frequency"]=observed_frequency;
    root["spectral_width"]=spectral_width; //in Hz

    outfile << root << std::endl;
    outfile.close();

    return true;
}

std::string fid_1d::write_json_as_string()
{
    Json::Value root;
    root["ndata"] = ndata;
    root["ndata_frq"] = ndata_frq;
    root["ndata_original"] = ndata_original;
    root["ndata_power_of_2"]=ndata_power_of_2; //We suppose ZF=2 in processing
    /**
     * ref1 is the end of the spectrum
     * carrier_frequency is the middle of the spectrum
     * SW is the total width of the spectrum
     * All in Hz.
     * If read in from a file other than ft1, all will be set to 0
    */
    root["carrier_frequency"] = carrier_frequency;
    root["observed_frequency"]=observed_frequency;
    root["spectral_width"]=spectral_width; //in Hz

    Json::StreamWriterBuilder writer;
    std::string output = Json::writeString(writer, root);
    
    return output;
}

/**
 * To save memory after peaks picking or fitting. User by 3D picker/fitter classes
*/
bool fid_1d::release_spectrum()
{
    spectrum_real.clear();
    return true;
}

/**
 * @brief get spectrum as a read-only vector
*/
const std::vector<float> & fid_1d::get_spectrum() const
{
    return spectrum_real;
}

#ifdef WEBASSEMBLY

/**
 * Function calls for web assembly to read header and data directly from variables without addtional copy
 */
uintptr_t fid_1d::get_data_of_real()
{
    return reinterpret_cast<uintptr_t>(spectrum_real.data());
}
uintptr_t fid_1d::get_data_of_imag()
{
    return reinterpret_cast<uintptr_t>(spectrum_imag.data());
}
uintptr_t fid_1d::get_data_of_header()
{
    return reinterpret_cast<uintptr_t>(nmrpipe_header_data.data());
}


/**
 * Exposed functions
*/
EMSCRIPTEN_BINDINGS(dp_1d_module_fid) {

    class_<fid_base>("fid_base")
        .constructor();

    class_<shared_data_1d>("shared_data_1d")
        .constructor()
        .class_property("n_verbose", &shared_data_1d::n_verbose);


    class_<fid_1d, base<fid_base> >("fid_1d")
        .constructor()
        .function("init", &fid_1d::init)
        .class_property("n_verbose", &shared_data_1d::n_verbose)
        .function("read_first_spectrum_from_buffer", &fid_1d::read_first_spectrum_from_buffer) //for dp_1d and vf_1d
        //below are for fid processing
        .function("set_up_apodization_from_string", &fid_1d::set_up_apodization_from_string)
        .function("read_bruker_files_as_strings", &fid_1d::read_bruker_files_as_strings)
        .function("get_fid_data_type", &fid_1d::get_fid_data_type) //to get data type of fid (float64 or int32)
        .function("set_fid_data", &fid_1d::set_fid_data) //to process fid
        .function("run_zf", &fid_1d::run_zf)
        .function("run_fft_and_rm_bruker_filter", &fid_1d::run_fft_and_rm_bruker_filter)
        .function("write_json_as_string", &fid_1d::write_json_as_string) 
        .function("get_nspectra", &fid_1d::get_nspectra)
        .function("get_ndata_frq", &fid_1d::get_ndata_frq)
        .function("write_nmrpipe_ft1", &fid_1d::write_nmrpipe_ft1) //write nmrPipe ft1 file
        .function("get_data_of_header", &fid_1d::get_data_of_header)
        .function("get_data_of_real", &fid_1d::get_data_of_real)
        .function("get_data_of_imag", &fid_1d::get_data_of_imag)
        .function("reduce_fid_size", &fid_1d::reduce_fid_size)
        ;

}
#endif // WEBASSEMBLY
