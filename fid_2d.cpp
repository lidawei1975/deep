#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

/**
 * Below two lines are required to check whether folder or file exists
 */
#include <sys/types.h>
#include <sys/stat.h>

/**
 * For FFT calculation
 */
#include "kiss_fft.h"

#include "json/json.h"

#include "fid_2d.h"

/**
 * Constructor and destructor
 */
fid_2d::fid_2d()
{
    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = false;
    grpdly = -1.0; //<0 means not set

    /**
     * These two are for phase correction along indirect dimension, read from pulse program
    */
    indirect_p0 = -400.0; // <-360 means (indirect_p0,indirect_p1) are both N.A
    indirect_p1 = 0.0;

    /**
     * These are user provided phase correction values
    */
    user_p0_direct = 0.0;
    user_p1_direct = 0.0;
    user_p0_indirect = 0.0;
    user_p1_indirect = 0.0;
}

fid_2d::~fid_2d()
{
    // dtor
}

/**
 * Load user provided phase correction values from a file
 * 4 values:
 * p0 (direct), p1 (direct), p0 (indirect), p1 (indirect)
*/
bool fid_2d::read_phase_correction(std::string fname)
{
    std::ifstream infile(fname);
    infile >> user_p0_direct >> user_p1_direct >> user_p0_indirect >> user_p1_indirect;
    std::cout<<"User provided phase correction values: "<<user_p0_direct<<" "<<user_p1_direct<<" "<<user_p0_indirect<<" "<<user_p1_indirect<<std::endl;
    return true;
}

/**
 * @brief read_bruker_folder:  read fid and parameter files from Bruker folder
 * this function will change the value of the following variables:
 * fid_data_float (unused one will have size 0)
 * data_type, data_complexity
 * ndata_bruker, ndata
 * udict_acqus
 * spectral_width, observed_frequency, carrier_frequency
 */
bool fid_2d::read_bruker_folder(std::string folder_name)
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
        std::cerr << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }
    else if (!(sb.st_mode & S_IFDIR))
    {
        // exist but is not a directory
        std::cerr << "Error: folder " << folder_name << " does not exist!" << std::endl;
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
            std::cerr<< "Error: cannot find fid or ser file in folder " << folder_name << std::endl;
            return false;
        }
    }

    /**
     * Read acqus file and store the content. Required for direct dimension
     */
    std::string acqus_file_name = folder_name + "/acqus";
    status = stat(acqus_file_name.c_str(), &sb);
    if (status != 0)
    {
        std::cerr << "Error: cannot find acqus file in folder " << folder_name << std::endl;
        return false;
    }

    /**
     * Read acqus file and store the content. Required for indirect dimension
     */
    std::string acqus_file2_name = folder_name + "/acqu2s";
    status = stat(acqus_file_name.c_str(), &sb);
    if (status != 0)
    {
        std::cerr << "Error: cannot find acqu2s file in folder " << folder_name << std::endl;
        return false;
    }

    /**
     * Read pulseprogram file and store the content.
     * This file is optional
     */
    std::string pulse_program_file_name = folder_name + "/pulseprogram";
    status = stat(pulse_program_file_name.c_str(), &sb);
    if (status != 0)
    {
        std::cout << "Warning: cannot find pulseprogram file in folder " << folder_name << std::endl;
        pulse_program_file_name = "";
    }

    /**
     * if we are here, we have found the fid data file and acqus file
     */
    std::vector<std::string> fid_data_file_names;
    fid_data_file_names.push_back(fid_data_file_name);
    return read_bruker_files(pulse_program_file_name,acqus_file2_name, acqus_file_name, fid_data_file_names);
}

/**
 * This is the main function to read Bruker data
 * It requires 1 pulse_program_name, 2 acqus files and a vector of fid data file names (1 is also ok)
 */
bool fid_2d::read_bruker_files(const std::string &pulse_program_name,const std::string &acqus_file2_name, const std::string &acqus_file_name, const std::vector<std::string> &fid_data_file_names)
{
    /**
     * At this time, we only try to read these two lines from pulseprogram file
     * ;PHC0(F1): 90
     * ;PHC1(F1): -180
     * It is possible we can only read p0 but not p1, in that case, p1 = 0.0.
     * If we cannot read both, we will set indirect_p0 = -360.0 (moonlighting as a flag) and indirect_p1 = 0.0
    */
    indirect_p0 = -400.0; 
    indirect_p1 = 0.0; 

    if(pulse_program_name != "")
    {
        std::ifstream infile(pulse_program_name);
        std::string line;
        while (std::getline(infile, line))
        {
            if (line.find(";PHC0(F1):") != std::string::npos)
            {
                indirect_p0 = std::stod(line.substr(10));
            }
            else if (line.find(";PHC1(F1):") != std::string::npos)
            {
                indirect_p1 = std::stod(line.substr(10));
            }
        }
    }
    
    read_jcamp(acqus_file_name, udict_acqus_direct);
    read_jcamp(acqus_file2_name, udict_acqus_indirect);

    /**
     * Check udic_acqus["AQ_mod"] to determine the data type: real or complex
     */
    if (udict_acqus_direct["AQ_mod"] == "3" || udict_acqus_direct["AQ_mod"] == "1")
    {
        data_complexity = FID_DATA_COMPLEXITY_COMPLEX;
    }
    else // 0 or 2
    {
        data_complexity = FID_DATA_COMPLEXITY_REAL;
    }

    /**
     * Check udic_acqus["BYTORDA"] to determine the data type: int32 or float32
     */
    if (udict_acqus_direct["DTYPA"] == "2")
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
    if (udict_acqus_direct.find("TD") == udict_acqus_direct.end())
    {
        std::cout << "Error: cannot find TD in acqus file" << std::endl;
        return false;
    }

    int td0 = std::stoi(udict_acqus_direct["TD"]);

    /**
     * Now we need to check the indirect dimension
     * check udic_acqus_indirect["TD"] to determine size of fid data
     */
    if (udict_acqus_indirect.find("TD") == udict_acqus_indirect.end())
    {
        std::cout << "Error: cannot find TD in acqu2s file" << std::endl;
        return false;
    }
    int td2 = std::stoi(udict_acqus_indirect["TD"]);

    /**
     * Need to know size of 1st fid_data_file_names
     */
    struct stat sb;
    int status = stat(fid_data_file_names[0].c_str(), &sb);
    if (status != 0)
    {
        std::cout << "Error: cannot find fid data file " << fid_data_file_names[0] << std::endl;
        return false;
    }
    /**
     * size of fid data file in bytes
     */
    int fid_data_file_size = sb.st_size;

    /**
     * Check for indirect dimension encoding
     */
    if (udict_acqus_indirect.find("FnMODE") == udict_acqus_indirect.end())
    {
        std::cout << "Error: cannot find FnMODE in acqu2s file" << std::endl;
        return false;
    }
    fnmode = std::stoi(udict_acqus_indirect["FnMODE"]);

    /**
     * According to Bruker manu,
     * when data_type = FID_DATA_TYPE_FLOAT64, fid is padded to 128 bytes
     * when data_type = FID_DATA_TYPE_INT32, fid is padded to 256 bytes
     * ndata_bruker is the direct dimension size
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

    ndata_bruker_indirect = td2;

    /**
     * Bruker TD is the number of data points, not the number of complex points for complex data
     * But we define ndata as the number of complex points
     */
    if (data_complexity == FID_DATA_COMPLEXITY_COMPLEX)
    {
        ndata = ndata_bruker / 2;
        ndata_indirect = ndata_bruker_indirect / 2;
    }
    else
    {
        ndata = ndata_bruker;
        ndata_indirect = ndata_bruker_indirect;
    }

    /**
     * get parameters "GRPDLY" from acqus file
     */

    if (udict_acqus_direct.find("GRPDLY") != udict_acqus_direct.end())
    {
        grpdly = std::stod(udict_acqus_direct["GRPDLY"]);
        /**
         * grpdly must > 0. Otherwiae it is from an early days Bruker spectrometer
         * We don't support this case at this time
         */
        if (grpdly <= 0.0)
        {
            std::cout << "Error: GRPDLY = " << grpdly << " is not supported" << std::endl;
        }
    }
    else
    {
        std::cout << "Error: cannot find GRPDLY in acqus file" << std::endl;
    }

    /**
     * now we can actually read the fid data
     * For complex data, real and imaginary parts are stored interleaved by Bruker.
     * Here we leave them interleaved in fid_data_float or fid_data_int
     *
     * Major order of fid_data_float:
     * indrect_dim, direct_dim, real/imaginary
     */

    fid_data_float.clear();
    fid_data_float.resize(ndata_bruker * ndata_bruker_indirect, 0.0);

    for (int i = 0; i < fid_data_file_names.size(); i++)
    {
        std::vector<int32_t> fid_data_int;       // 32 bit
        std::vector<double> temp_fid_data_float; // 64 bit
        FILE *fp_fid_data = fopen(fid_data_file_names[i].c_str(), "rb");

        if (data_type == FID_DATA_TYPE_INT32)
        {
            int nread;
            fid_data_int.clear();
            fid_data_int.resize(ndata_bruker * ndata_bruker_indirect);
            nread = fread(&fid_data_int[0], sizeof(int32_t), ndata_bruker * ndata_bruker_indirect, fp_fid_data);
            if (nread != ndata_bruker * ndata_bruker_indirect)
            {
                std::cout << "Error: cannot read " << ndata_bruker * ndata_bruker_indirect << " int32 from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }
        else if (data_type == FID_DATA_TYPE_FLOAT64)
        {
            int nread;
            temp_fid_data_float.clear();
            temp_fid_data_float.resize(ndata_bruker * ndata_bruker_indirect);
            nread = fread(&temp_fid_data_float[0], sizeof(double), ndata_bruker * ndata_bruker_indirect, fp_fid_data);
            if (nread != ndata_bruker * ndata_bruker_indirect)
            {
                std::cout << "Error: cannot read " << ndata_bruker * ndata_bruker_indirect << " float64 from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }

        /**
         * if data_type is int, copy them to double
         */
        if (data_type == FID_DATA_TYPE_INT32)
        {
            for (int i = 0; i < ndata_bruker * ndata_bruker_indirect; i++)
            {
                fid_data_float[i] += (float)fid_data_int[i];
            }
        }
        else
        {
            for (int i = 0; i < ndata_bruker * ndata_bruker_indirect; i++)
            {
                fid_data_float[i] += (float)temp_fid_data_float[i];
            }
        }
        fclose(fp_fid_data);
    }

    /**
     * get receiver_gain from acqus if it exists
     */
    if (udict_acqus_direct.find("RG") != udict_acqus_direct.end())
    {
        receiver_gain = std::stod(udict_acqus_direct["RG"]);
    }
    else
    {
        std::cout << "Warning: cannot find RG in acqus file" << std::endl;
        receiver_gain = 1.0; // default
    }

    /**
     * Now setup the following variables:
     * spectral_width, observed_frequency, carrier_frequency
     */
    if (udict_acqus_direct.find("SW_h") != udict_acqus_direct.end())
    {
        spectral_width = std::stod(udict_acqus_direct["SW_h"]);
    }
    else
    {
        std::cout << "Error: cannot find SW_h in acqus file" << std::endl;
        return false;
    }

    if (udict_acqus_direct.find("SFO1") != udict_acqus_direct.end())
    {
        observed_frequency = std::stod(udict_acqus_direct["SFO1"]);
    }
    else
    {
        std::cout << "Error: cannot find SFO1 in acqus file" << std::endl;
        return false;
    }

    if (udict_acqus_direct.find("O1") != udict_acqus_direct.end())
    {
        carrier_frequency = std::stod(udict_acqus_direct["O1"]);
    }
    else
    {
        std::cout << "Error: cannot find O1 in acqus file" << std::endl;
        return false;
    }

    /**
     * Do the same for indirect dimension
     */
    if (udict_acqus_indirect.find("SW_h") != udict_acqus_indirect.end())
    {
        spectral_width_indirect = std::stod(udict_acqus_indirect["SW_h"]);
    }
    else
    {
        std::cout << "Error: cannot find SW_h in acqu2s file" << std::endl;
        return false;
    }

    if (udict_acqus_indirect.find("SFO1") != udict_acqus_indirect.end())
    {
        observed_frequency_indirect = std::stod(udict_acqus_indirect["SFO1"]);
    }
    else
    {
        std::cout << "Error: cannot find SFO1 in acqu2s file" << std::endl;
        return false;
    }

    if (udict_acqus_indirect.find("O1") != udict_acqus_indirect.end())
    {
        carrier_frequency_indirect = std::stod(udict_acqus_indirect["O1"]);
    }
    else
    {
        std::cout << "Error: cannot find O1 in acqu2s file" << std::endl;
        return false;
    }

    b_read_bruker_acqus_and_fid = true;
    b_read_nmrpipe_fid = false;

    /**
     * Adjust indirect dimension data format according to FnMODE
     *

     * fid_data_float[ndata_bruker_indirect][ndata_bruker]. Data is interleaved along both dimensions (real,imaginary,real,imaginary,...)
    */

    if (fnmode == 6)
    {
        /**
         * Echo anti-echo pulse sequence.
         *
         * Step 1: along indirect dimension, (increment 0 - increment 1)/2.0 ==> increment 0
         * Step 2: along indirect dimension, (increment 0 + increment 1)/2.0 ==> increment 1, and so on
         * Step 3, for increment 1,3,5,..., apply 90 degree correction along direct dimension (i.e., multiply by i, real -> imaginary, imaginary -> -real)
         */
        for (int i = 0; i < ndata_bruker_indirect; i += 2)
        {
            std::vector<float> temp1(ndata_bruker, 0.0f), temp2(ndata_bruker, 0.0f), temp3(ndata_bruker, 0.0f);
            for (int j = 0; j < ndata_bruker; j++)
            {
                /**
                 * Step 1 here
                */
                temp1[j] = (fid_data_float[i * ndata_bruker + j] - fid_data_float[(i + 1) * ndata_bruker + j]) / 2.0f;
                /**
                 * Step 2 here
                */
                temp2[j] = (fid_data_float[i * ndata_bruker + j] + fid_data_float[(i + 1) * ndata_bruker + j]) / 2.0f;
            }
            /**
             * Step 3 here
             */
            for (int k = 0; k < ndata_bruker; k += 2)
            {
                temp3[k] = -temp2[k + 1];
                temp3[k + 1] = temp2[k];
            }
            /**
             * Copy temp1 and temp3 back to fid_data_float
             */
            for (int j = 0; j < ndata_bruker; j++)
            {
                fid_data_float[i * ndata_bruker + j] = temp1[j];
                fid_data_float[(i + 1) * ndata_bruker + j] = temp3[j];
            }
        }
    }

    return true;
}

/**
 * Read nmrPipe .fid file
 */
bool fid_2d::read_nmrpipe_fid(const std::string &fname)
{
    /**
     * Read 512 float32 from file as nmrPipe header
     */

    nmrpipe_header_data.clear();
    nmrpipe_header_data.resize(512, 0.0f);

    FILE *fp = fopen(fname.c_str(), "rb");
    if (fp == NULL)
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }
    int n_read = fread(nmrpipe_header_data.data(), sizeof(float), 512, fp);
    if (n_read != 512)
    {
        std::cout << "Error: cannot read 512 float from file " << fname << std::endl;
        return false;
    }

    nmrPipe::nmrpipe_header_to_dictionary(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);

    if (nmrpipe_dict_float["FDDIMCOUNT"] != 2.0)
    {
        std::cout << "Error: FDDIMCOUNT = " << nmrpipe_dict_float["FDDIMCOUNT"] << ", this is not 2D data" << std::endl;
        return false;
    }

    set_varibles_from_nmrpipe_dictionary();

    /**
     * Read data from fp as float32, and store them in fid_data_float
     * fid_data_float [0,2,4] is for real part
     * fid_data_float [1,3,5] is for imaginary part
     * But in nmrPipe, real and imaginary parts are stored separately
     */
    std::vector<float> temp_fid_data_float(ndata_bruker / 2), temp_fid_data_float_part2(ndata_bruker / 2);

    for (int i = 0; i < ndata_bruker_indirect; i++)
    {
        int n_read = fread(temp_fid_data_float.data(), sizeof(float), ndata_bruker / 2, fp);
        if (n_read != ndata_bruker / 2)
        {
            std::cout << "Error: cannot read " << ndata_bruker / 2 << " float from file " << fname << std::endl;
            return false;
        }
        n_read = fread(temp_fid_data_float_part2.data(), sizeof(float), ndata_bruker / 2, fp);
        if (n_read != ndata_bruker / 2)
        {
            std::cout << "Error: cannot read " << ndata_bruker / 2 << " float from file " << fname << std::endl;
            return false;
        }

        for (int j = 0; j < ndata_bruker / 2; j++)
        {
            fid_data_float.push_back(temp_fid_data_float[j]);
            fid_data_float.push_back(temp_fid_data_float_part2[j]);
        }
    }

    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = true;

    return true;
}

/**
 * This function will fillin most PART 1 varibles from PART 3 variables
 */
bool fid_2d::set_varibles_from_nmrpipe_dictionary()
{
    grpdly = -1.0;                                        //<0 means not set
    data_type = FID_DATA_TYPE::FID_DATA_TYPE_NOT_DEFINED; // not defined

    if (nmrpipe_dict_float["FDF2QUADFLAG"] == 1.0)
    {
        data_complexity = FID_DATA_COMPLEXITY_COMPLEX;
    }
    else
    {
        data_complexity = FID_DATA_COMPLEXITY_REAL;
    }

    /**
     * Read remaining data from the file
     */
    ndata_bruker = int(nmrpipe_dict_float["FDSIZE"]) * 2;
    ndata_bruker_indirect = int(nmrpipe_dict_float["FDSPECNUM"]);
    ndata = ndata_bruker / 2;
    ndata_indirect = ndata_bruker_indirect / 2;

    /**
     * Set receiver_gain to 1.0 (defail;t value)
     */
    receiver_gain = 1.0;

    /**
     * Set spectral_width, observed_frequency, carrier_frequency
     */
    spectral_width = nmrpipe_dict_float["FDF2SW"];
    observed_frequency = nmrpipe_dict_float["FDF2OBS"];
    carrier_frequency = nmrpipe_dict_float["FDF2ORIG"];
    spectral_width_indirect = nmrpipe_dict_float["FDF1SW"];
    observed_frequency_indirect = nmrpipe_dict_float["FDF1OBS"];
    carrier_frequency_indirect = nmrpipe_dict_float["FDF1ORIG"];

    return true;
}



/**
 * A low level function to run fft
 * @param n_dim1: size of first dimension.
 * @param n_dim2: size of second dimension. FFT is run along this dimension
 * @param n_dim2_frq: size of second dimension after FFT (because of zero filling)
 * @param in: input data [n_dim1][n_dim2*2] real/imaginary interleaved
 * @param out1: output data [n_dim1][n_dim2_frq] FFT result, real part
 * @param out2: output data [n_dim1][n_dim2_frq] FFT result, imaginary part
 * @param b_remove_filter: whether to remove Bruker digitizer filter
 * @param b_swap: whether to swap left and right halves of spectrum
 * 
 * Addtional information on b_swap
 * kiss_fft has different behavior than Bruker or nmrPipe FFT in a way that result needs to be swapped and flipped
 * So when b_swap == false, we swap left and right halves of spectrum.
 * When b_swap == true, we still do the swap, but we also apply alternate sign of input data (effectively a swap)
 */
bool fid_2d::fft_worker(int n_dim1, int n_dim2, int n_dim2_frq,
                        const std::vector<float> &in,
                        std::vector<float> &out1,
                        std::vector<float> &out2,
                        bool b_remove_filter,
                        bool b_swap,
                        int grpdly_) const
{
    /**
     * Assess whether n_dim1 and n_dim2 are valid
     */
    if (n_dim1 * n_dim2 * 2 != in.size())
    {
        std::cout << "Error: n_dim1*n_dim2*2!=in.size()" << std::endl;
        return false;
    }

    if (out1.size() != n_dim1 * n_dim2_frq || out2.size() != n_dim1 * n_dim2_frq)
    {
        std::cout << "Error: out1.size()!=n_dim1*n_dim2_frq || out2.size()!=n_dim1*n_dim2_frq" << std::endl;
        return false;
    }

    /**
     * Loop over first dimension
     */
    for (int j = 0; j < n_dim1; j++)
    {
        /**
         * Initialize cx_in and cx_out
         */
        kiss_fft_cpx *cx_in = new kiss_fft_cpx[n_dim2_frq];
        kiss_fft_cpx *cx_out = new kiss_fft_cpx[n_dim2_frq];

        /**
         * Copy data from fin to cx_in
         */
        for (int i = 0; i < n_dim2; i++)
        {
            if(b_swap == true && i%2==1)
            {
                cx_in[i].r = -in[i * 2 + j * n_dim2 * 2];
                cx_in[i].i = -in[i * 2 + 1 + j * n_dim2 * 2];
            }
            else
            {
                cx_in[i].r = in[i * 2 + j * n_dim2 * 2];
                cx_in[i].i = in[i * 2 + 1 + j * n_dim2 * 2];
            }
        }

        /**
         * Zero filling
         */
        for (int i = n_dim2; i < n_dim2_frq; i++)
        {
            cx_in[i].r = 0.0;
            cx_in[i].i = 0.0;
        }

        /**
         * Run fft
         */
        if (kiss_fft_cfg cfg = kiss_fft_alloc(n_dim2_frq, 0, NULL, NULL))
        {
            kiss_fft(cfg, cx_in, cx_out);
            free(cfg);
        }
        else
        {
            std::cout << "Error: cannot allocate memory for kiss_fft_cfg" << std::endl;
            return false;
        }

        /**
         * Copy data from cx_out to spectrum_real and spectrum_imag
         */
        std::vector<float> spectrum_real(n_dim2_frq), spectrum_imag(n_dim2_frq);
        for (int i = 0; i < n_dim2_frq; i++)
        {
            spectrum_real[i] = cx_out[i].r / sqrt(float(n_dim2_frq));
            spectrum_imag[i] = cx_out[i].i / sqrt(float(n_dim2_frq)); // scale by sqrt(ndata_frq) to be consistent with standard fft
        }

        /**
         * Important: this step need to be done before fft result swap and flip
         */
        if (b_remove_filter)
        {
            remove_bruker_digitizer_filter(grpdly_, spectrum_real, spectrum_imag);
        }

        std::vector<float> spectrum_real_reoganized(n_dim2_frq, 0.0f);
        std::vector<float> spectrum_imag_reoganized(n_dim2_frq, 0.0f);

        /**
         * Swap left and right halves of spectrum. Aways do this because kiss_fft has different behavior than Bruker or nmrPipe FFT
         */
        
        for (int i = 1; i <= n_dim2_frq; i++)
        {
            spectrum_real_reoganized[i - 1] = (float)spectrum_real[(i + n_dim2_frq / 2) % n_dim2_frq];
            spectrum_imag_reoganized[i - 1] = (float)spectrum_imag[(i + n_dim2_frq / 2) % n_dim2_frq];
        }
       

        /**
         * Flip the spectrum then copy data to out1 and out2, because kiss_fft has different behavior than Bruker or nmrPipe FFT
         */
       
        for (int i = 0; i < n_dim2_frq; i++)
        {
            out1[i + j * n_dim2_frq] = spectrum_real_reoganized[n_dim2_frq - i - 1];
            out2[i + j * n_dim2_frq] = spectrum_imag_reoganized[n_dim2_frq - i - 1];
        }
        
    }
    return true;
}

/**
 * @brief fid_2d::phase_correction_worker: apply phase correction along dim2 for all rows in spectrum
 * @param n_dim1: size of first dimension.
 * @param n_dim2: size of second dimension. FFT is run along this dimension
 * @param spectrum_real: input/output data [n_dim1][n_dim2] real part
 * @param spectrum_imag: input/output data [n_dim1][n_dim2] imaginary part
 * @param p0: phase correction degree 0-th order
 * @param p1: phase correction degree 1-st order
*/
bool fid_2d::phase_correction_worker(int n_dim1,int n_dim2, std::vector<float> &spectrum_real, std::vector<float> &spectrum_imag, double p0, double p1) const
{
    /**
     * Apply phase correction along dim2.
     * Loop over dim1
    */
    for (int i = 0; i < n_dim1; i++)
    {
        for (int j = 0; j < n_dim2; j++)
        {
            float phase = p0 + p1 * j / n_dim2;
            float cos_phase = cos(phase*M_PI/180.0);
            float sin_phase = sin(phase*M_PI/180.0);

            float real = spectrum_real[j + i * n_dim2];
            float imag = spectrum_imag[j + i * n_dim2];

            spectrum_real[j + i * n_dim2] = real * cos_phase - imag * sin_phase;
            spectrum_imag[j + i * n_dim2] = real * sin_phase + imag * cos_phase;
        }
    }
    return true;
}

bool fid_2d::set_up_apodization(apodization *a1, apodization *a2)
{
    apodization_direct = a1;
    apodization_indirect = a2;
    return true;
};



/**
 * @brief fid_1d::run_fft_and_rm_bruker_filter: run fft on fid_data_float
 * @return true on success, false on failure
 */
bool fid_2d::run_fft_and_rm_bruker_filter()
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
     * Do the same for indirect dimension
     */
    ndata_power_of_2_indirect = 1;
    while (ndata_power_of_2_indirect < ndata_indirect)
    {
        ndata_power_of_2_indirect *= 2;
    }
    ndata_frq_indirect = ndata_power_of_2_indirect * zf_indirect;

    /**
     * Define spectrum_intermediate_real and spectrum_intermediate_imag
     */
    std::vector<float> intermediate_spectrum(ndata_frq * ndata_bruker_indirect, 0.0f);
    std::vector<float> intermediate_spectrum_imag(ndata_frq * ndata_bruker_indirect, 0.0f);

    /**
     * Apodization along direct dimension
     */
    apodization_direct->set_n(ndata);
    for(int i=0;i<ndata_bruker_indirect;i++)
    {
        apodization_direct->run_apodization(fid_data_float.data()+i*ndata_bruker,ndata*2);
    }


    /**
     * FFT along direct dimension
    */
    fft_worker(ndata_bruker_indirect, ndata, ndata_frq, fid_data_float, intermediate_spectrum, intermediate_spectrum_imag, true /** digital fileter*/, false /** swap*/, grpdly);


    /**
     * Apply phase correction along direct dimension. Debug code
     * Mathmatically equivalent to direct dimension phase correction code below
    */
    phase_correction_worker(ndata_bruker_indirect, ndata_frq, intermediate_spectrum,intermediate_spectrum_imag, user_p0_direct /** P0 */,user_p1_direct /** P1*/);


    /**
     * Debug code. write intermediate_spectrum to file
     */
    // std::ofstream fout("intermediate_spectrum_real.txt");
    // for (int i = 0; i < ndata_bruker_indirect; i++)
    // {
    //     for (int j = 0; j < ndata_frq; j++)
    //     {
    //         fout << intermediate_spectrum[j + i * ndata_frq] << " ";
    //     }
    //     fout << std::endl;
    // }
    // fout.close();
    // fout.open("intermediate_spectrum_imag.txt");
    // for (int i = 0; i < ndata_bruker_indirect; i++)
    // {
    //     for (int j = 0; j < ndata_frq; j++)
    //     {
    //         fout << intermediate_spectrum_imag[j + i * ndata_frq] << " ";
    //     }
    //     fout << std::endl;
    // }
    // fout.close();

    transpose_2d(intermediate_spectrum, ndata_bruker_indirect, ndata_frq);
    transpose_2d(intermediate_spectrum_imag, ndata_bruker_indirect, ndata_frq);

   
    /**
     * Set correct size for spectrum_real_real and spectrum_real_imag spectrum_imag_real and spectrum_imag_imag
     */
    spectrum_real_real.clear();
    spectrum_real_imag.clear();
    spectrum_imag_real.clear();
    spectrum_imag_imag.clear();

    spectrum_real_real.resize(ndata_frq_indirect * ndata_frq);
    spectrum_real_imag.resize(ndata_frq_indirect * ndata_frq);
    spectrum_imag_real.resize(ndata_frq_indirect * ndata_frq);
    spectrum_imag_imag.resize(ndata_frq_indirect * ndata_frq);

    /**
     * size of intermediate_spectrum is [ndata_frq][ndata_bruker_indirect]
    */

    apodization_indirect->set_n(ndata_indirect);
    for(int i=0;i<ndata_frq;i++)
    {
        apodization_indirect->run_apodization(intermediate_spectrum.data()+i*ndata_bruker_indirect,ndata_indirect*2);
        apodization_indirect->run_apodization(intermediate_spectrum_imag.data()+i*ndata_bruker_indirect,ndata_indirect*2);
    }


    /**
     * FFT along indirect dimension.
     * fnmode == 6 means echo anti-echo pulse sequence
    */
    bool b_swap = false;
    if(fnmode==6)
    {
        b_swap = false;
    }
    else if(fnmode==5) //state-tppi. Need to swap
    {
        b_swap = true;
    }
    /**
     * TODO: other cases
    */

    fft_worker(ndata_frq, ndata_indirect, ndata_frq_indirect, intermediate_spectrum, spectrum_real_real, spectrum_real_imag, false /**no grpdly correction*/, b_swap, -1 /** not used.*/); 

    fft_worker(ndata_frq, ndata_indirect, ndata_frq_indirect, intermediate_spectrum_imag, spectrum_imag_real, spectrum_imag_imag, false /**no grpdly correction*/, b_swap, -1 /** not used.*/); 

    /**
     * Apply phase correction along indirect dimension. This is debug code.
    */
    phase_correction_worker(ndata_frq, ndata_frq_indirect, spectrum_real_real,spectrum_real_imag, user_p0_indirect /** P0 */,user_p1_indirect /** P1*/);
    phase_correction_worker(ndata_frq, ndata_frq_indirect, spectrum_imag_real,spectrum_imag_imag, user_p0_indirect /** P0 */,user_p1_indirect /** P1*/);

    transpose_2d(spectrum_real_real, ndata_frq, ndata_frq_indirect);
    transpose_2d(spectrum_real_imag, ndata_frq, ndata_frq_indirect);
    transpose_2d(spectrum_imag_real, ndata_frq, ndata_frq_indirect);
    transpose_2d(spectrum_imag_imag, ndata_frq, ndata_frq_indirect);

    /**
     * Apply phase correction along direct dimension. This is debug code.
    */
    // phase_correction_worker(ndata_frq_indirect, ndata_frq, spectrum_real_real,spectrum_imag_real, user_p0_direct /** P0 */,user_p1_direct /** P1*/);
    // phase_correction_worker(ndata_frq_indirect, ndata_frq, spectrum_real_imag,spectrum_imag_imag, user_p0_direct /** P0 */,user_p1_direct /** P1*/);

    /**
     * Apply receiver_gain if > 1.0
    */
    if (receiver_gain > 1.0)
    {
        for (int i = 0; i < spectrum_real_real.size(); i++)
        {
            spectrum_real_real[i] *= receiver_gain;
            spectrum_real_imag[i] *= receiver_gain;
            spectrum_imag_real[i] *= receiver_gain;
            spectrum_imag_imag[i] *= receiver_gain;
        }
    }

    return true;
}

/**
 * Transpose a 2D array 
 * @param in: input array. size is n_dim1*n_dim2
 * @param n_dim1: size of first dimension (before transpose)
 * @param n_dim2: size of second dimension (before transpose)
*/
bool fid_2d::transpose_2d(std::vector<float> &in, int n_dim1, int n_dim2)
{
    std::vector<float> temp(in.size());
    for (int i = 0; i < n_dim1; i++)
    {
        for (int j = 0; j < n_dim2; j++)
        {
            temp[j * n_dim1 + i] = in[i*n_dim2+j];
        }
    }
    in = temp;
    return true;
}

/**
 * @brief create_nmrpipe_dictionary: create nmrpipe dictionary from udict_acqus and derived values
 * @param b_frq: true if we are saving frq data, false if we are saving time data
 * @param nmrpipe_dict_string: output dictionary for string values
 * @param nmrpipe_dict_float: output dictionary for float values
 */
bool fid_2d::create_nmrpipe_dictionary(bool b_frq, std::map<std::string, std::string> &nmrpipe_dict_string, std::map<std::string, float> &nmrpipe_dict_float) const
{
    /**
     * @brief create_empty_nmrpipe_header: create empty nmrpipe header
     */
    // nmrPipe::create_empty_nmrpipe_header(nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * Fill in some parameters from what we have
     */
    nmrpipe_dict_float["FDDIMCOUNT"] = 2.0f; // 2D data
    nmrpipe_dict_float["FDDIMORDER"] = 2.0f; // direct dimension is dimension 2
    nmrpipe_dict_float["FDDIMORDER2"] = 1.0f; // indirect dimension is dimension 1

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

    /**
     * indirect dimension
     */
    nmrpipe_dict_float["FDF1SW"] = spectral_width_indirect;
    nmrpipe_dict_float["FDF1OBS"] = observed_frequency_indirect;
    nmrpipe_dict_float["FDF1CAR"] = carrier_frequency_indirect / observed_frequency_indirect; // normalized to observed frequency, per nmrPipe convention
    nmrpipe_dict_string["FDF1LABEL"] = "13C";                                                 // hardcoded for now!!

    int n_scale_factor_indirect = 1;

    if(data_complexity==FID_DATA_COMPLEXITY_COMPLEX)
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 0.0f;
        nmrpipe_dict_float["FDF1QUADFLAG"] = 0.0f;
        n_scale_factor_indirect = 2;
    }
    else
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 1.0f;
        nmrpipe_dict_float["FDF1QUADFLAG"] = 1.0f;
        n_scale_factor_indirect = 1;
    }

    /**
     * Copied from nmrglue. no idea what they mean
     */
    if (nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF2QUADFLAG"] == nmrpipe_dict_float["FDF3QUADFLAG"] && nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF4QUADFLAG"] == 1.0f)
    {
        nmrpipe_dict_float["FDQUADFLAG"] = 1.0f;
    }

    nmrpipe_dict_float["FDTRANSPOSED"] = 0.0f; // not transposed

    if (b_frq)
    {
        /**
         * we are saving frq data, so set FDF2FTFLAG to 1 and set value for FDF2FTSIZE
         * set FDF2TDSIZE if we have time data
         */
        nmrpipe_dict_float["FDF2FTSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDF1FTSIZE"] = ndata_frq_indirect;
        nmrpipe_dict_float["FDSPECNUM"] = ndata_frq_indirect * n_scale_factor_indirect; // *2 to be consistent with nmrPipe for complex data
        nmrpipe_dict_float["FDF2FTFLAG"] = 1.0f;                                        // frq data, instead of time

        nmrpipe_dict_float["FDSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDREALSIZE"] = ndata_frq * ndata_frq_indirect * n_scale_factor_indirect;
    }
    else
    {
        /**
         * Not sure this is correct at this time
         */
        nmrpipe_dict_float["FDF2TDSIZE"] = ndata;
        nmrpipe_dict_float["FDF2FTFLAG"] = 0.0f; // time data, instead of frq

        nmrpipe_dict_float["FDSIZE"] = ndata;
        nmrpipe_dict_float["FDREALSIZE"] = ndata;
    }

    /**
     * set apodization information. meaningless for time domain data
     * copied from nmrglue. Basically we need to adjust APOD,CENTER and ORIG according to zero filling
     * FDF2APOD and FDF2APOD seem to be the original size before apoization, ZF and FT
     */
    nmrpipe_dict_float["FDF2APOD"] = ndata;
    nmrpipe_dict_float["FDF2CENTER"] = int(ndata_frq / 2.0f) + 1.0f;
    nmrpipe_dict_float["FDF2ORIG"] = (nmrpipe_dict_float["FDF2CAR"] * nmrpipe_dict_float["FDF2OBS"] - nmrpipe_dict_float["FDF2SW"] * (ndata_frq - nmrpipe_dict_float["FDF2CENTER"]) / ndata_frq);

    nmrpipe_dict_float["FDF1APOD"] = ndata_indirect;
    nmrpipe_dict_float["FDF1CENTER"] = int(ndata_frq_indirect / 2.0f) + 1.0f;   
    nmrpipe_dict_float["FDF1ORIG"] = (nmrpipe_dict_float["FDF1CAR"] * nmrpipe_dict_float["FDF1OBS"] - nmrpipe_dict_float["FDF1SW"] * (ndata_frq_indirect - nmrpipe_dict_float["FDF1CENTER"]) / ndata_frq_indirect);

    /**
     * Save indirect dimension phase correction information, which are read from pulse program
     * indirect_p0 < -360 means not set. 
     */
    nmrpipe_dict_float["FDUSER1"] = indirect_p0;
    nmrpipe_dict_float["FDUSER2"] = indirect_p1;
    

    return true;
}

/**
 * @brief fid_1d::write_nmrpipe_ft1: write 1D spectrum to nmrpipe file
 * Before writing, define nmrpipe header, set values from udict_acqus and derived values
 * @param outfname: output file name
 * @return true on success, false on failure
 */
bool fid_2d::write_nmrpipe_ft2(std::string outfname, bool b_real_only) 
{

    if (outfname != "")
    {   
        if(b_real_only==true)
        { 
            /**
             * We are saving real data only. This will cause some changes in nmrpipe header (such as ydim)
             */
            data_complexity = FID_DATA_COMPLEXITY::FID_DATA_COMPLEXITY_REAL; 

            /**
             * Below are for cases when we read nmrpipe .fid file
            */
            nmrpipe_dict_float["FDF2QUADFLAG"] = 1.0f;
            nmrpipe_dict_float["FDF1QUADFLAG"] = 1.0f;
            nmrpipe_dict_float["FDSPECNUM"] = ndata_frq_indirect; // *2 to be consistent with nmrPipe for complex data
            nmrpipe_dict_float["FDREALSIZE"] = ndata_frq * ndata_frq_indirect;
        }
       

        if (b_read_bruker_acqus_and_fid == true)
        {
            /**
             * create nmrpipe header.
             * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
             * from udict_acqus and derived values
             * True means we are saving frq data
             */
            create_nmrpipe_dictionary(true, nmrpipe_dict_string, nmrpipe_dict_float);

            /**
             * define header vector, because the function call nmrpipe_dictionary_to_header won't apply for memory
             */
            nmrpipe_header_data.clear();
            nmrpipe_header_data.resize(512, 0.0f);

            nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);
        }
        else
        {
            /**
             * We read nmrpipe file, so we already have nmrpipe_header_data
             * But we updated some values in nmrpipe_dict_float, so we need to update nmrpipe_header_data
             */
            nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);
        }

        /**
         * now write nmrpipe_header_data and spectral to the file
         */
        FILE *fp = fopen(outfname.c_str(), "wb");
        if (fp == NULL)
        {
            std::cerr << "Error: cannot open file " << outfname << " for writing" << std::endl;
            return false;
        }
        /**
         * Write header first
         */
        fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);

        /**
         * This is how data are organized in nmrpipe file
         */
        for (int i = 0; i < ndata_frq_indirect; i++)
        {
            fwrite(spectrum_real_real.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
            /**
             * If we are saving real data only, we don't need to save imaginary part for either direct or indirect dimension
            */
            if(b_real_only==false)
            {
                fwrite(spectrum_real_imag.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                fwrite(spectrum_imag_real.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                fwrite(spectrum_imag_imag.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
            }
        }
        fclose(fp);
    }

    return true;
}