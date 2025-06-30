#include <vector>
#include <valarray>
#include <set>
#include <array>
#include <fstream>
#include <iostream>
#include <cstring>

/**
 * Below two lines are required to check whether folder or file exists
 */
#include <sys/types.h>
#include <sys/stat.h>

#include <Eigen/Dense>
#include <Eigen/Cholesky>

/**
 * For FFT calculation
 */
#include "kiss_fft.h"

#include "json/json.h"

#include "lmminimizer.h"
#include "fid_2d.h"

namespace fid_2d_math {
    bool movemean(float *data, float * new_data, int ndata, int nwindow )
    {
        if (nwindow <= 0)
        {
            return false;
        }
        if (nwindow > ndata)
        {
            return false;
        }
        for (int i = 0; i < ndata; i++)
        {
            int n1 = i - nwindow / 2;
            int n2 = i + nwindow / 2;
            if (n1 < 0)
            {
                n1 = 0;
            }
            if (n2 >= ndata)
            {
                n2 = ndata - 1;
            }
            float sum = 0.0;
            for (int j = n1; j <= n2; j++)
            {
                sum += data[j];
            }
            new_data[i] = sum / (n2 - n1 + 1);
        }

        return true;
    };

    bool spectrum_minus(float *data1, float *data2, int  n)
    {
        for (int i = 0; i < n; i++)
        {
            data1[i] -= data2[i];
        }
        return true;
    };

    bool polynomial_baseline(
        const std::vector<float> &x,
        const std::vector<float> &y,
        const std::vector<float> &z,
        int xdim,
        int ydim,
        int order,
        std::vector<float> &baseline_parameters
        )
    {
        /**
         * Using SVD to solve the linear equation
         * d * p = b, where
         * p is the baseline parameters p[0] + p[1]*x + p[2]*y + p[3]*x^2 + p[4]*y*y (no cross term in NMR)
         * b is the data
        */
        int n = xdim * ydim;
        int m = order*2 + 1;

        Eigen::MatrixXd d(n,m);
        Eigen::VectorXd b(n);
        Eigen::VectorXd solution(m);

        /**
         * Copy z to b
         */
        for(int i=0;i<n;i++)
        {
            b(i) = z[i];
        }

        /**
         * Fill d
         */
        for(int i=0;i<n;i++)
        {
            d(i,0) = 1.0;
            if(order>=1)
            {
                d(i,1) = x[i];
                d(i,2) = y[i];
            }
            if(order>=2)
            {
                d(i,3) = x[i]*x[i];
                d(i,4) = y[i]*y[i];
            }
            if(order>=3)
            {
                d(i,5) = x[i]*x[i]*x[i];
                d(i,6) = y[i]*y[i]*y[i];
            }
        }

        /**
         * Solve the linear equation
         */
        solution = d.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

        /**
         * Copy solution to baseline_parameters
         */
        baseline_parameters.clear();
        for(int i=0;i<m;i++)
        {
            baseline_parameters.push_back(solution(i));
        }

        return true;
    }
};

namespace ldw_math_spectrum_2d
{

    bool SplitFilename(const std::string &str, std::string &path_name, std::string &file_name, std::string &file_name_ext)
    {
        bool b = false;
        std::string file_name_full;
        std::size_t found = str.find_last_of("/\\");
        if (found != std::string::npos)
        {
            path_name = str.substr(0, found);
            file_name_full = str.substr(found + 1);
            b = true;
        }
        else
        {
            path_name = ".";
            file_name_full = str;
        }

        // std::cout<<"file_name_full is "<<file_name_full<<std::endl;

        found = file_name_full.find_last_of(".");
        if (found != std::string::npos)
        {
            file_name = file_name_full.substr(0, found);
            file_name_ext = file_name_full.substr(found + 1);
            b = true;
        }
        else
        {
            file_name_ext = "";
            file_name = file_name_full;
        }

        // std::cout<<"path is "<<path_name<<std::endl;
        // std::cout<<"file_name is "<<file_name<<std::endl;
        // std::cout<<"file_name_ext is "<<file_name_ext<<std::endl;

        return b;
    }

    void get_ppm_from_header(const double ref, const double sw, const double frq, double &stop, double &begin)
    {
        if (fabs(frq) > 1e-10)
        {
            stop = ref / frq;
            begin = stop + sw / frq;
        }
        else
        {
            stop = 2.0;
            begin = 1.0; // arbitary
        }
        return;
    }

    double calcualte_median(std::vector<double> scores)
    {
        size_t size = scores.size();

        if (size == 0)
        {
            return 0; // Undefined, really.
        }
        else
        {
            sort(scores.begin(), scores.end());
            if (size % 2 == 0)
            {
                return (scores[size / 2 - 1] + scores[size / 2]) / 2;
            }
            else
            {
                return scores[size / 2];
            }
        }
    };

    template <typename T>
    void sortArr(std::vector<T> &arr, std::vector<int> &ndx)
    {
        std::vector<std::pair<T, int>> vp;

        for (int i = 0; i < arr.size(); ++i)
        {
            vp.push_back(std::make_pair(arr[i], i));
        }

        std::sort(vp.begin(), vp.end());

        for (int i = 0; i < vp.size(); i++)
        {
            ndx.push_back(vp[i].second);
        }
    };

    template void sortArr<int>(std::vector<int> &arr, std::vector<int> &ndx);
    template void sortArr<float>(std::vector<float> &arr, std::vector<int> &ndx);
    template void sortArr<double>(std::vector<double> &arr, std::vector<int> &ndx);
    


    int next_power_of_2(int n)
    {
        double logbase2 = log(double(n)) / log(2);

        int count = int(ceil(logbase2));
  
        return int(pow(2, count));
    };

};

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

    aqseq ="321"; //default

    b_negative = false; //default. data is normal

    b_first_only = false; //default. process all spectra

    b_nmrPipe_transposed = false; //default. data is not transposed

    b_frq = false; //default. keep time domain data before FFT

    b_frq_indirect = false; //default. keep time domain data before FFT

    b_imaginary = true; //default. keep imaginary data
    
    b_imaginary_indirect = true; //default. keep imaginary data

    nuslists.clear(); //clear nuslists, not really needed ??

    apodization_code = 0;
    apodization_code_indirect = 0;

    noise_level = -0.1; // negative nosie means we don't have it yet

    // spectrum range
    begin1 = 100;
    stop1 = -100;
    begin2 = 1000;
    stop2 = -1000; // will be updated once read in spe

    indirect_p0 = -400.0; // <-360 means N.A.
    indirect_p1 = 0.0; //N.A when indirect_p0 is N.A.
}

fid_2d::~fid_2d()
{
    // dtor
}

/**
 * Read nus list from a file. One line per sampling point or one element per sampling point in one line
*/
bool fid_2d::read_nus_list(std::string fname)
{
    std::ifstream infile(fname);
    std::string line;
    nuslists.clear();
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        int temp;
        while (iss >> temp)
        {
            nuslists.push_back(temp);
        }
    }
    infile.close();

    std::cout<<"Read "<<nuslists.size()<<" nus points from file "<<fname<<std::endl;
    for(int i=0;i<nuslists.size();i++)
    {
        std::cout<<nuslists[i]<<" ";
    }
    std::cout<<std::endl;

    return true;
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
        std::cerr << "Error: FID data is real, not complex" << std::endl;
        return false;
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
        ndata_original = td0 / 2;
    }
    else
    {
        ndata = ndata_bruker;
        ndata_indirect = ndata_bruker_indirect;
        ndata_original = td0;
    }

    nucleus = udict_acqus_direct["NUC1"];
    nucleus_indirect = udict_acqus_indirect["NUC1"];
    
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
     * If nuslists is not empty, we need to adjustt ndata_bruker_indirect
     * We save the original value in ndata_bruker_indirect_original, so that we can expand the data to its original size (with zeros in non-sampled traces)
     */
    int ndata_bruker_indirect_original = ndata_bruker_indirect;
    if(nuslists.size()>0)
    {
        ndata_bruker_indirect = nuslists.size()*2; //complex data
        ndata_bruker_indirect_original = (nuslists[nuslists.size()-1]+1)*2;
    }
     
    /**
     * now we can actually read the fid data
     * For complex data, real and imaginary parts are stored interleaved by Bruker.
     * Here we leave them interleaved in fid_data_float or fid_data_int
     *
     * Major order of fid_data_float:
     * indrect_dim, direct_dim, real/imaginary interleaved
     */

    fid_data_float.clear();

    for (int i = 0; i < fid_data_file_names.size(); i++)
    {
        std::vector<int32_t> fid_data_int;       // 32 bit
        std::vector<double> temp_fid_data_float; // 64 bit
        FILE *fp_fid_data = fopen(fid_data_file_names[i].c_str(), "rb");

        if (data_type == FID_DATA_TYPE_INT32)
        {
            nspectra = 0;
            fid_data_int.clear();
            fid_data_int.resize(ndata_bruker * ndata_bruker_indirect);
            while(fread(fid_data_int.data()+nspectra*ndata_bruker * ndata_bruker_indirect, sizeof(int32_t), ndata_bruker * ndata_bruker_indirect, fp_fid_data)==ndata_bruker * ndata_bruker_indirect)
            {
                nspectra++;
                /**
                 * Get space for the next spectrum (in case of pseudo 3D NMR)
                 */
                fid_data_int.resize(ndata_bruker * ndata_bruker_indirect * (nspectra + 1));
            }
            /**
             * Set correct size for fid_data_int
             */
            fid_data_int.resize(ndata_bruker * ndata_bruker_indirect * nspectra);
            if(nspectra==0)
            {
                std::cout << "Error: cannot read int32 from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }
        else if (data_type == FID_DATA_TYPE_FLOAT64)
        {
            nspectra = 0;
            temp_fid_data_float.clear();
            temp_fid_data_float.resize(ndata_bruker * ndata_bruker_indirect);
            nspectra = 0;
            while(fread(temp_fid_data_float.data()+nspectra*ndata_bruker * ndata_bruker_indirect, sizeof(double), ndata_bruker * ndata_bruker_indirect, fp_fid_data) == ndata_bruker * ndata_bruker_indirect)
            {
                nspectra++;
                /**
                 * Get space for the next spectrum (in case of pseudo 3D NMR)
                 */
                temp_fid_data_float.resize(ndata_bruker * ndata_bruker_indirect * (nspectra + 1));
            }
            /**
             * Set correct size for temp_fid_data_float
             */
            temp_fid_data_float.resize(ndata_bruker * ndata_bruker_indirect * nspectra);
            if(nspectra==0)
            {
                std::cout << "Error: cannot read float64 from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }

        /**
         * Add data to fid_data_float
         */
        fid_data_float.resize(ndata_bruker * ndata_bruker_indirect * nspectra, 0.0);
        if (data_type == FID_DATA_TYPE_INT32)
        {
            for (int i = 0; i < ndata_bruker * ndata_bruker_indirect * nspectra; i++)
            {
                fid_data_float[i] += (float)fid_data_int[i];
            }
        }
        else
        {
            for (int i = 0; i < ndata_bruker * ndata_bruker_indirect * nspectra; i++)
            {
                fid_data_float[i] += (float)temp_fid_data_float[i];
            }
        }
        fclose(fp_fid_data);
    }

    std::cout << "Read " << nspectra << " spectra from " << fid_data_file_names.size() << " files" << std::endl;
    std::cout <<" ndata_bruker = "<<ndata_bruker<<" ndata_bruker_indirect = "<<ndata_bruker_indirect<<std::endl;


    /**
     * Now we need to reorganize fid_data_float when order is 321 (from inner to outer), which means order is
     * - indirect dimension (acqu3s ==> 1)
     * - - pseudo dimension (acqu2s ==> 2)
     * - - - direct dimension (acqus ==> 3)
     * We need to reorganize it to 312, which means order is: pseudo dimension, indirect dimension, direct dimension
     * then we can process all 2D spectra at one pseudo dimension data point one by one
    */
    if(nspectra>1)
    {
        if(aqseq=="321")
        {
            /**
             * Rroganize fid_data_float, in case 2nd dimension is pseudo 3D dimension while 3rd dimension is the indirect dimension
            */
            std::vector<float> fid_data_float_temp=fid_data_float;
            fid_data_float.clear();
            fid_data_float.resize(nspectra * ndata_bruker * ndata_bruker_indirect, 0.0f);
            for(int m=0;m<nspectra;m++)
            {
                for(int i=0;i<ndata_bruker_indirect;i++)
                {
                    int k_from = m + nspectra * i;
                    int k_to = m * ndata_bruker_indirect + i;
                    /**
                     * copy from fid_data_float_temp[k_from*ndata_bruker:(k_from+1)*ndata_bruker] to 
                     * fid_data_float[k_to*ndata_bruker:(k_to+1)*ndata_bruker]
                     */
                    std::memcpy(fid_data_float.data()+k_to*ndata_bruker,fid_data_float_temp.data()+k_from*ndata_bruker,sizeof(float)*ndata_bruker);
                }
            }
            /**
             * Release memory
             */
            fid_data_float_temp.clear();
        }
        else if(aqseq=="312")
        {
            /**
             * Order is already 312 (from iner to outer), do nothing
             * - pseudo dimension (acqu2s ==> 2)
             * - - indirect dimension (acqu3s ==> 1)
             * - - - direct dimension (acqus ==> 3)
            */
        }
        else
        {
            /**
             * Won't happen because we have checked aqseq in set_aqseq
            */
        }
    }

    /**
     * If b_first_only is true, we only process the first spectrum
     */
    if(b_first_only)
    {
        nspectra=1;
        /**
         * Resize fid_data_float to contain only the first spectrum, discard the rest
         */
        fid_data_float.resize(ndata_bruker * ndata_bruker_indirect);
    }

    /**
     * Expand fid_data_float to its original size (with zeros in non-sampled traces) if nuslists is not empty
     */
    if(ndata_bruker_indirect_original > ndata_bruker_indirect)
    {
        fid_data_float.resize(nspectra * ndata_bruker * ndata_bruker_indirect_original, 0.0f);
        /**
         * We do in-place expansion, so we work from the last spectrum to the first
         * and from the last trace to the first within each spectrum
         */
        for(int m=nspectra-1;m>=0;m--)
        {
            int spectrum_start_target = m * ndata_bruker * ndata_bruker_indirect_original;
            int spectrum_start_source = m * ndata_bruker * ndata_bruker_indirect;
            for(int n=nuslists.size()-1;n>=0;n--)
            {
                /**
                 * At index*2: imaginary part of indirect dimension
                */
                int nn=n*2+1;
                int mm=nuslists[n]*2+1;
                int trace_start_source = spectrum_start_source + nn * ndata_bruker;
                int trace_start_target = spectrum_start_target + mm * ndata_bruker;
                if(trace_start_target > trace_start_source)
                {
                    std::memcpy(fid_data_float.data()+trace_start_target,fid_data_float.data()+trace_start_source,sizeof(float)*ndata_bruker);
                    /**
                     * Set data at trace_start_source to 0.0f
                    */
                    std::fill(fid_data_float.begin()+trace_start_source,fid_data_float.begin()+trace_start_source+ndata_bruker,0.0f);
                }

                /**
                 * At index*2-1: real part of indirect dimension
                */
                nn=n*2;
                mm=nuslists[n]*2;
                trace_start_source = spectrum_start_source + nn * ndata_bruker;
                trace_start_target = spectrum_start_target + mm * ndata_bruker;
                if(trace_start_target > trace_start_source)
                {
                    std::memcpy(fid_data_float.data()+trace_start_target,fid_data_float.data()+trace_start_source,sizeof(float)*ndata_bruker);
                    /**
                     * Set data at trace_start_source to 0.0f
                    */
                    std::fill(fid_data_float.begin()+trace_start_source,fid_data_float.begin()+trace_start_source+ndata_bruker,0.0f);
                }


                /**
                 * else, do nothing, because trace_start_target == trace_start_source
                */
            }
        }
        /**
         * Restore ndata_bruker_indirect to its original size
         */
        ndata_bruker_indirect = ndata_bruker_indirect_original;
        ndata_indirect = ndata_bruker_indirect / 2;
    }
    
    /**
     * Define nusflags, for convenience. nusflags.size() == ndata_indirect
     * and nusflags[i] == 1 means the i-th trace is sampled, 0 means the i-th trace is not sampled
     */
    if(nuslists.size()>0)
    {
        nusflags.clear();
        nusflags.resize(ndata_indirect, 0);
        for(int i=0;i<nuslists.size();i++)
        {
            nusflags[nuslists[i]]=1;
        }
    }
    else
    { 
        /**
         * nuslists is empty. This is a fully sampled experiments, so all traces are sampled.
        */
        nusflags.clear();
        nusflags.resize(ndata_indirect, 1);
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

    n_center = int(ndata / 2.0f)+1;
    n_center_indirect = int(ndata_indirect / 2.0f)+1;
    origin_frequency = (carrier_frequency - spectral_width * (ndata - n_center) / ndata);
    origin_frequency_indirect = (carrier_frequency_indirect - spectral_width_indirect * (ndata_indirect - n_center_indirect) / ndata_indirect);

    b_read_bruker_acqus_and_fid = true;
    b_read_nmrpipe_fid = false;



    int ndata_bruker_indirect_by2 = ndata_bruker_indirect;
    /**
     * In case ndata_bruker_indirect is odd, the last trace is not used. We need to adjust ndata_bruker_indirect_by2
     */
    if (ndata_bruker_indirect % 2 == 1)
    {
        ndata_bruker_indirect_by2 = ndata_bruker_indirect - 1;
        /**
         * ndata_indirect is always ndata_bruker_indirect_by2/2, because we write
         * ndata_indirect = ndata_bruker_indirect/2. When ndata_bruker_indirect is odd, ndata_indirect (type is int) will discard the 0.5
         */
    }

    /**
     * Adjust indirect dimension data format according to FnMODE
     * fid_data_float[ndata_bruker_indirect][ndata_bruker]. Data is interleaved along both dimensions (real,imaginary,real,imaginary,...)
     */
    if (fnmode == 6)
    {
        for(int index_spectrum=0;index_spectrum<nspectra;index_spectrum++)
        {
            int data_start = index_spectrum * ndata_bruker * ndata_bruker_indirect;
            /**
             * Echo anti-echo pulse sequence.
             *
             * Step 1: along indirect dimension, (increment 0 - increment 1)/2.0 ==> increment 0
             * Step 2: along indirect dimension, (increment 0 + increment 1)/2.0 ==> increment 1, and so on
             * Step 3, for increment 1,3,5,..., apply 90 degree correction along direct dimension (i.e., multiply by i, real -> imaginary, imaginary -> -real)
             */


            for (int i = 0; i < ndata_bruker_indirect_by2; i += 2)
            {
                std::vector<float> temp1(ndata_bruker, 0.0f), temp2(ndata_bruker, 0.0f), temp3(ndata_bruker, 0.0f);
                for (int j = 0; j < ndata_bruker; j++)
                {
                    /**
                     * Step 1 here
                    */
                    temp1[j] = (fid_data_float[data_start + i * ndata_bruker + j] - fid_data_float[data_start + (i + 1) * ndata_bruker + j]) / 2.0f;
                    /**
                     * Step 2 here
                    */
                    temp2[j] = (fid_data_float[data_start + i * ndata_bruker + j] + fid_data_float[data_start + (i + 1) * ndata_bruker + j]) / 2.0f;
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
                    fid_data_float[data_start + i * ndata_bruker + j] = temp1[j];
                    fid_data_float[data_start + (i + 1) * ndata_bruker + j] = temp3[j];
                }
            } // for (int i = 0; i < ndata_bruker_indirect; i += 2)
        } // for(int index_spectrum=0;index_spectrum<nspectra;index_spectrum++)
    }

    /**
     * Convert fid_data_float to:
     * fid_data_real_real, fid_data_real_imag, fid_data_imag_real, fid_data_imag_imag
     * (This is lazy. It is better to read data into fid_data_real_real,... directly, but memory is not a big issue here.)
     * fid_data_float is interleaved (rr, ri of 1st trace, ir, ii of 1st trace, rr, ri of 2nd trace, ir, ii of 2nd trace, ...)
     * ndata = ndata_bruker / 2; ndata_indirect = ndata_bruker_indirect / 2;
     */
    fid_data_real_real.resize(nspectra * ndata * ndata_indirect, 0.0f);
    fid_data_real_imag.resize(nspectra * ndata * ndata_indirect, 0.0f);
    fid_data_imag_real.resize(nspectra * ndata * ndata_indirect, 0.0f);
    fid_data_imag_imag.resize(nspectra * ndata * ndata_indirect, 0.0f);
    for(int k=0;k<nspectra;k++)
    {
        int data_start = k * ndata_bruker * ndata_bruker_indirect;
        int data_start2 = k * ndata * ndata_indirect; //same for all 4 vector. 
        for (int i = 0; i < ndata_bruker_indirect_by2; i+=2)
        {
            int trace_start = i * ndata_bruker;
            int trace_start2 = i/2 * ndata;
            for (int j = 0; j < ndata_bruker; j += 2)
            {
                fid_data_real_real[data_start2 + trace_start2 + j / 2] = fid_data_float[data_start + i * ndata_bruker + j];
                fid_data_real_imag[data_start2 + trace_start2 + j / 2] = fid_data_float[data_start + i * ndata_bruker + j + 1];
                fid_data_imag_real[data_start2 + trace_start2 + j / 2] = fid_data_float[data_start + (i + 1) * ndata_bruker + j];
                fid_data_imag_imag[data_start2 + trace_start2 + j / 2] = fid_data_float[data_start + (i + 1) * ndata_bruker + j + 1];
            }
        }
    }
    
    /**
     * To save memory, we can release fid_data_float
     */
    fid_data_float.clear();

    n_outer_dim = ndata_indirect;
    n_inner_dim = ndata;

    return true;
}

/**
 * Read nmrPipe .fid file
 */
bool fid_2d::read_nmrpipe_file(const std::string &fname)
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
        std::cout << "FDDIMCOUNT = " << nmrpipe_dict_float["FDDIMCOUNT"] << ", this is not true 2D data" << std::endl;
    }

    set_varibles_from_nmrpipe_dictionary();

    /**
     * Read time domain data if FDF2FTFLAG = 0 and FDF1FTFLAG = 0
     * We suppose all data is complex at this time
     */
    if (b_frq == false && b_frq_indirect == false)
    {

        fid_data_real_real.resize(ndata * ndata_indirect, 0.0f);
        fid_data_real_imag.resize(ndata * ndata_indirect, 0.0f);
        fid_data_imag_real.resize(ndata * ndata_indirect, 0.0f);
        fid_data_imag_imag.resize(ndata * ndata_indirect, 0.0f);


        for (int i = 0; i < ndata_indirect; i++)
        {
            /**
             * At this time, we suppose the data is complex for time domain data
            */
            fread(fid_data_real_real.data(), sizeof(float), ndata, fp);
            fread(fid_data_real_imag.data(), sizeof(float), ndata, fp);
            fread(fid_data_imag_real.data(), sizeof(float), ndata, fp);
            fread(fid_data_imag_imag.data(), sizeof(float), ndata, fp);
        }
    }
    /**
     * Read frq data if FDF2FTFLAG = 1 and FDF1FTFLAG = 1
    */
    else if(b_frq == true && b_frq_indirect == true)
    {
        ndata_frq = n_inner_dim;
        ndata_frq_indirect = n_outer_dim;

        spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_real_imag.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_imag_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_imag_imag.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        

        for (int i = 0; i < n_outer_dim; i++)
        {
            fread(spectrum_real_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp);
            if(b_imaginary==true)
            {
                fread(spectrum_real_imag.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp);
            }
            if(b_imaginary_indirect==true)
            {
                fread(spectrum_imag_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp);
                
            }
            if(b_imaginary==true && b_imaginary_indirect==true)
            {
                fread(spectrum_imag_imag.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp);
            }
        }
    }
        /**
     * Read intermediate data if FDF2FTFLAG = 1 and FDF1FTFLAG = 0
     * We supose b_imaginary_indirect = true
     */
    else if (b_frq == true && b_frq_indirect == false)
    {
        intermediate_data_real_real.resize(ndata_frq * ndata_indirect, 0.0f);
        intermediate_data_real_imag.resize(ndata_frq * ndata_indirect, 0.0f);
        intermediate_data_imag_real.resize(ndata_frq * ndata_indirect, 0.0f);
        intermediate_data_imag_imag.resize(ndata_frq * ndata_indirect, 0.0f);

        std::cout<<"n_out_dim = "<<n_outer_dim<<" n_in_dim = "<<n_inner_dim<<std::endl;

        for (int i = 0; i < n_outer_dim; i++)
        {
            fread(intermediate_data_real_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp);
            if(b_imaginary==true)
            {
                fread(intermediate_data_real_imag.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp);
                fread(intermediate_data_imag_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp);
                fread(intermediate_data_imag_imag.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp);
            }
            else
            {
                fread(intermediate_data_imag_real.data() + i * n_inner_dim, sizeof(float), n_inner_dim, fp);    
            }
        }
    }
    else
    {
        std::cout << "Error: FDF2FTFLAG = " << nmrpipe_dict_float["FDF2FTFLAG"] << ", FDF1FTFLAG = " << nmrpipe_dict_float["FDF1FTFLAG"] << " is not supported" << std::endl;
        return false;
    }

    b_read_bruker_acqus_and_fid = false;
    b_read_nmrpipe_fid = true;

    nspectra = 1; // only one spectrum, not a pseudo 3D NMR

    spect = spectrum_real_real.data(); //alias to spectrum_real_real

    return true;
}

/**
 * This function will fillin most PART 1 varibles from PART 3 variables
 */
bool fid_2d::set_varibles_from_nmrpipe_dictionary()
{
    /**
     * These are not used in nmrPipe dictionary ?
    */
    grpdly = -1.0;                                        //<0 means not set
    data_type = FID_DATA_TYPE::FID_DATA_TYPE_NOT_DEFINED; // not defined
    receiver_gain = 1.0;

    /**
     * We read nmrPipe dictionary, we suppose it is complex data (originally)
     * and use flag b_imaginary and b_imaginary_indirect to determine whether we have 
     * Remember in nmrPipe, 0.0 means complex, 1.0 means real
    */
    data_complexity = FID_DATA_COMPLEXITY_COMPLEX;
    b_imaginary = nmrpipe_dict_float["FDF2QUADFLAG"] == 0.0;
    b_imaginary_indirect = nmrpipe_dict_float["FDF1QUADFLAG"] == 0.0;

      /**
     * Set spectral_width, observed_frequency, carrier_frequency
     */
    spectral_width = nmrpipe_dict_float["FDF2SW"];
    observed_frequency = nmrpipe_dict_float["FDF2OBS"];
    carrier_frequency = nmrpipe_dict_float["FDF2CAR"]*observed_frequency;
    origin_frequency = nmrpipe_dict_float["FDF2ORIG"];
    spectral_width_indirect = nmrpipe_dict_float["FDF1SW"];
    observed_frequency_indirect = nmrpipe_dict_float["FDF1OBS"];
    carrier_frequency_indirect = nmrpipe_dict_float["FDF1CAR"]*observed_frequency_indirect;
    origin_frequency_indirect = nmrpipe_dict_float["FDF1ORIG"];

    /**
     * Set begin1, step1, stop1, begin2, step2, stop2
    */
    int direct_ndx = nmrpipe_dict_float["FDDIMORDER1"]; //must be 2 for no transposed data
    int indirect_ndx = nmrpipe_dict_float["FDDIMORDER2"]; //can be 1 or 3 for no transposed data

    if (indirect_ndx == 3)
    {
        spectral_width_indirect = nmrpipe_dict_float["FDF3SW"];
        observed_frequency_indirect = nmrpipe_dict_float["FDF3OBS"];
        carrier_frequency_indirect = nmrpipe_dict_float["FDF3CAR"] * observed_frequency_indirect;
        origin_frequency_indirect = nmrpipe_dict_float["FDF3ORIG"];
    }

    /**
     * Check for FT flag. 
    */
    b_frq = false;
    if(nmrpipe_dict_float["FDF2FTFLAG"] == 1.0)
    {
        b_frq = true;
    }
    b_frq_indirect = false;
    if(nmrpipe_dict_float["FDF1FTFLAG"] == 1.0 && indirect_ndx==1)
    {
        b_frq_indirect = true;
    }
    if(nmrpipe_dict_float["FDF3FTFLAG"] == 1.0 && indirect_ndx==3)
    {
        b_frq_indirect = true;
    }

    /**
     * This is true for nmrPipe data, both frq or time domain data
    */
    n_outer_dim = int(nmrpipe_dict_float["FDSPECNUM"]);
    n_inner_dim = int(nmrpipe_dict_float["FDSIZE"]);

         
    if(b_imaginary && b_imaginary_indirect)
    {
        n_outer_dim /= 2; //nmrPipe convention, when both complex, FDSPECNUM = 2 * n_outer_dim; 
    }

    int n_direct_dimension = n_inner_dim;
    int n_indirect_dimension = n_outer_dim;

    if(nmrpipe_dict_float["FDTRANSPOSED"] == 1.0f) 
    {
        b_nmrPipe_transposed = true;
        std::swap(n_direct_dimension,n_indirect_dimension);
    }
    else
    {
        b_nmrPipe_transposed = false;
    }

    /**
     * Read remaining data from the file
     */
    if(b_frq==true){
        ndata_frq = n_direct_dimension;
        ndata = int(nmrpipe_dict_float["FDF2TDSIZE"]);
    }
    else{
        ndata = n_direct_dimension;
        ndata_frq = 0; //not used
    }
    n_center = int(nmrpipe_dict_float["FDF2CENTER"]);


    if(b_frq_indirect==true){
        ndata_frq_indirect = n_indirect_dimension;
        ndata_indirect = int(nmrpipe_dict_float["FDF1TDSIZE"]);
    }
    else{
        ndata_indirect = n_indirect_dimension;
        ndata_frq_indirect = 0; //not used
    }
    n_center_indirect = int(nmrpipe_dict_float["FDF1CENTER"]);


    
    if(b_imaginary){
        ndata_bruker = ndata * 2;
    }

    ndata_original = nmrpipe_dict_float["FDF2TDSIZE"];

    if(b_imaginary_indirect){
        ndata_bruker_indirect = ndata_indirect * 2;
    }

   
  

    /**
     * Need to update begin1,step1,stop1,begin2,step2,stop2 from nmrPipe dictionary
     */
    stop1 = origin_frequency / observed_frequency;
    begin1 = stop1 + spectral_width / observed_frequency;
    step1 = (stop1 - begin1) / (ndata_frq); // direct dimension
    begin1 += step1;                        // here we have to + step because we want to be consistent with nmrpipe program,I have no idea why nmrpipe is different than topspin

    stop2 = origin_frequency_indirect / observed_frequency_indirect;
    begin2 = stop2 + spectral_width_indirect / observed_frequency_indirect;
    step2 = (stop2 - begin2) / (ndata_frq_indirect); // indirect dimension
    begin2 += step2;                                 // here we have to + step because we want to be consistent with nmrpipe program,I have no idea why nmrpipe is different than topspin

    return true;
}



/**
 * A low level function to run fft
 * @param n_dim1: size of first dimension.
 * @param dim1_flag: whether to run FFT for this trace. Size is n_dim1
 * @param n_dim2: size of second dimension. FFT is run along this dimension
 * @param n_dim2_frq: size of second dimension after FFT (because of zero filling)
 * @param in1: input data, real part [n_dim1][n_dim2] 
 * @param in2: input data, imaginary part [n_dim1][n_dim2]
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
bool fid_2d::fft_worker(int n_dim1, std::vector<int> dim1_flag, int n_dim2, int n_dim2_frq,
                        const float  *in1, const float  *in2,
                        float * out1, float * out2,
                        bool b_remove_filter,
                        bool b_swap,
                        double grpdly_) const
{
    /**
     * Loop over first dimension
     */
    for (int j = 0; j < n_dim1; j++)
    {
        if(dim1_flag[j]==0)
        {
            /**
             * Set output to 0.0
            */
           for (int i = 0; i < n_dim2_frq; i++)
            {
                out1[i + j * n_dim2_frq] = 0.0f;
                out2[i + j * n_dim2_frq] = 0.0f;
            }
            continue;
        }
        /**
         * Initialize cx_in and cx_out
         */
        kiss_fft_cpx *cx_in = new kiss_fft_cpx[n_dim2_frq];
        kiss_fft_cpx *cx_out = new kiss_fft_cpx[n_dim2_frq];

        /**
         * Copy data from in to cx_in
         */
        for (int i = 0; i < n_dim2; i++)
        {
            /**
             * If b_swap is true, we apply alternate sign to input data
            */
            if(b_swap == true && i%2==1) 
            {
                cx_in[i].r = -in1[i  + j * n_dim2 ];
                cx_in[i].i = -in2[i  + j * n_dim2];
            }
            else
            {
                cx_in[i].r = in1[i + j * n_dim2];
                cx_in[i].i = in2[i + j * n_dim2 ];
            }
        }

        /**
         * Zero filling. Input data has already been apodized
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
bool fid_2d::phase_correction_worker(int n_dim1,int n_dim2,float *spectrum_real, float *spectrum_imag, double p0, double p1) const
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
 * @brief fid_1d::write_nmrpipe_ft1: write some information
 * This is mainly for web-server
 */
bool fid_2d::write_json(std::string fname)
{
    std::ofstream outfile(fname.c_str());
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    Json::Value root;
    root["ndata"] = ndata;
    root["ndata_original"] = ndata_original;
    root["ndata_power_of_2"] = ndata_power_of_2;
    root["carrier_frequency"] = carrier_frequency;
    root["observed_frequency"] = observed_frequency;
    root["spectral_width"] = spectral_width;

    /**
     * ft2 part. 0 if not exist
     */
    root["n_direct"] = ndata_frq;
    root["n_indirect"] = ndata_frq_indirect;
    root["noise_level"] = noise_level;
    root["begin_direct"] = begin1; 
    root["begin_indirect"]=begin2;
    root["step_direct"]=step1;
    root["step_indirect"]=step2;
    

    outfile << root << std::endl;
    outfile.close();

    return true;
}


bool fid_2d::write_pseudo3d_json(std::string fname)
{
    std::ofstream outfile(fname.c_str());
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    Json::Value root;
    root["spectra"]=nspectra;

    outfile << root << std::endl;
    outfile.close();

    return true;
}

/**
 * @brief process 2D FID data to get 2D frequency domain spectra
 * @return true on success, false on failure
 */
bool fid_2d::full_process(bool b_di_direct,bool b_indirect_direct)
{
    run_direct_dimension(b_di_direct);
    transpose_intermediate_data();
    run_indirect_dimension(b_indirect_direct);
    untranspose_frquency_domain_data();
    return true;
}

/**
 * @brief direct dimension process only
*/
bool fid_2d::direct_only_process(bool b_di_direct)
{
    run_direct_dimension(b_di_direct);
    transpose_intermediate_data();
    return true;
}

/**
 * @brief indirect dimension process only, need to read in a half processed data
*/
bool fid_2d::indirect_only_process(bool b_indirect_direct)
{
    /**
     * Assess that b_frq is true and b_nmrPipe_transposed is true
    */
    if(b_frq == false || b_nmrPipe_transposed == false)
    {
        std::cout<<" Require b_frq = true and b_nmrPipe_transposed = true"<<std::endl;
        std::cout << "Error: b_frq = " << b_frq << " and/or b_nmrPipe_transposed = " << b_nmrPipe_transposed << std::endl;
        return false;
    }
    run_indirect_dimension(b_indirect_direct);
    untranspose_frquency_domain_data();
    return true;
}

/**
 * Other process functions
 * Can run phase correction and remove imaginary part
*/
bool fid_2d::other_process(bool b_di_direct,  bool b_di_indirect)
{
    /**
     * Apply phase correction along direct dimension only if b_imaginary is true
    */
    if(b_imaginary==true)
    {
        phase_correction_worker(ndata_frq_indirect, ndata_frq, spectrum_real_real.data(), spectrum_real_imag.data(), user_p0_direct /** P0 */,user_p1_direct /** P1*/);
        /**
         * If b_imaginary_indirct is also true, we need to apply phase correction for the imag part of indirect dimension too
         */
        if(b_imaginary_indirect==true)
        {
            phase_correction_worker(ndata_frq_indirect, ndata_frq, spectrum_imag_real.data(), spectrum_imag_imag.data(), user_p0_direct /** P0 */,user_p1_direct /** P1*/);
        }
    }

    
    /**
     * Apply phase correction along indirect dimension when b_imaginary_indirect is true
    */
    if(b_imaginary_indirect==true)
    {
        transpose_2d(spectrum_real_real.data(),ndata_frq,ndata_frq_indirect);
        transpose_2d(spectrum_imag_real.data(),ndata_frq,ndata_frq_indirect);
        phase_correction_worker(ndata_frq, ndata_frq_indirect, spectrum_real_real.data(), spectrum_imag_real.data(), user_p0_indirect /** P0 */,user_p1_indirect /** P1*/);
        transpose_2d(spectrum_real_real.data(),ndata_frq_indirect,ndata_frq);
        transpose_2d(spectrum_imag_real.data(),ndata_frq_indirect,ndata_frq);
    }

    if(b_di_direct == true)
    {
        /**
         * If b_di_direct is true, we set b_imaginary to false, and we do not need real_imag and imag_imag
        */
        b_imaginary = false;
        spectrum_real_imag.clear();
        spectrum_imag_imag.clear();
    }
    else
    {
        b_imaginary = true;
        if(b_imaginary_indirect==true)
        {  
            transpose_2d(spectrum_real_imag.data(),ndata_frq,ndata_frq_indirect);
            transpose_2d(spectrum_imag_imag.data(),ndata_frq,ndata_frq_indirect);
            phase_correction_worker(ndata_frq, ndata_frq_indirect, spectrum_real_imag.data(), spectrum_imag_imag.data(), user_p0_indirect /** P0 */,user_p1_indirect /** P1*/);
            transpose_2d(spectrum_real_imag.data(),ndata_frq_indirect,ndata_frq);
            transpose_2d(spectrum_imag_imag.data(),ndata_frq_indirect,ndata_frq);
        }
    }

    if(b_di_indirect == true)
    {
        /**
         * If b_imaginary_indirect is false, we do not need real_imag and imag_imag
        */
        b_imaginary_indirect = false;
        spectrum_real_imag.clear(); //might be already cleared, but no harm to clear again
        spectrum_imag_imag.clear();
    }
    
    
    return true;
}

/**
 * Remove huge water signal from the first trace (of the indirect dimension)
 * This is done after FFT along direct dimension
 * because I don't want to deal with Bruker digitazer filter 
 * (first 68 or 76 points are artifical and the following several points are crushed by the filter)
*/
bool fid_2d::water_suppression()
{

    if(fid_data_real_real.size()==0 || fid_data_real_imag.size()==0 || fid_data_imag_real.size()==0 || fid_data_imag_imag.size()==0)
    {
        std::cout<<"Error: fid_data is empty"<<std::endl;
        return false;
    }

    int grpdly_int = int(grpdly+0.5);
    float * ave_signal = new float[ndata-grpdly_int];

    for(int i=0;i<ndata_indirect;i++)
    {
        fid_2d_math::movemean(fid_data_real_real.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int,32);
        fid_2d_math::spectrum_minus(fid_data_real_real.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int);

        fid_2d_math::movemean(fid_data_real_imag.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int,32);
        fid_2d_math::spectrum_minus(fid_data_real_imag.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int);

        fid_2d_math::movemean(fid_data_imag_real.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int,32);
        fid_2d_math::spectrum_minus(fid_data_imag_real.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int);

        fid_2d_math::movemean(fid_data_imag_imag.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int,32);
        fid_2d_math::spectrum_minus(fid_data_imag_imag.data()+i*ndata+grpdly_int,ave_signal,ndata-grpdly_int);
    }


    return true;
};

/**
 * Run a polynomial baseline correction on the final spectrum (real real part only)
 * @param order: order of the polynomial (0,1 or 2)
*/
bool fid_2d::polynorminal_baseline(int order)
{
    if(spectrum_real_real.size()==0)
    {
        std::cout<<"Error: spectrum_real_real is empty"<<std::endl;
        return false;
    }

    int n_dim1 = ndata_frq_indirect;
    int n_dim2 = ndata_frq;

    int step1 = 1;
    int step2 = 1;

    /**
     * To make SVD manageable, we need to reduce the size of the spectrum
     * to be <=524288 (512k)
    */
    while(n_dim1*n_dim2>524288)
    {
        n_dim1 /= 2;
        step1 *= 2;
        n_dim2 /= 2;
        step2 *= 2;
    }

    std::vector<float> x(n_dim1*n_dim2,0.0f);
    std::vector<float> y(n_dim1*n_dim2,0.0f);
    std::vector<float> z(n_dim1*n_dim2,0.0f);

    for(int i=0;i<n_dim1;i++)
    {
        for(int j=0;j<n_dim2;j++)
        {
            x[i*n_dim2+j] = j;
            y[i*n_dim2+j] = i;
            z[i*n_dim2+j] = spectrum_real_real[i*step1*n_dim2+j*step2];
        }
    }



    std::vector<float> baseline_params;
    fid_2d_math::polynomial_baseline(x,y,z,n_dim1,n_dim2,order,baseline_params);

    /**
     * Substract baseline from the spectrum
    */
    for(int i=0;i<ndata_frq_indirect;i++)
    {
        for(int j=0;j<ndata_frq;j++)
        {
            float baseline = baseline_params[0]; //constant term
            if(order>=1) 
            {
                baseline += baseline_params[1]*j; //linear term along dim2(inner dimension)
                baseline += baseline_params[2]*i; //linear term along dim1(outer dimension)
            }
            if(order>=2)
            {
                baseline += baseline_params[3]*j*j; //quadratic term along dim2(inner dimension)
                baseline += baseline_params[4]*i*i; //quadratic term along dim1(outer dimension)
                /** No cross term in NMR */
            }
            if(order>=3)
            {
                baseline += baseline_params[5]*j*j*j; //cubic term along dim2(inner dimension)
                baseline += baseline_params[6]*i*i*i; //cubic term along dim1(outer dimension)
                /** No cross term in NMR */
            }
            spectrum_real_real[i*ndata_frq+j] -= baseline;
        }
    }
    std::cout<<"Baseline correction done using polynorminal at order "<<order<<std::endl;

    return true;
}


bool fid_2d::run_direct_dimension(bool b_di_direct)
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

    n_center = int(ndata_frq / 2.0f) + 1;
    origin_frequency = (carrier_frequency - spectral_width * (ndata_frq - n_center) / ndata_frq);


    /**
     * Clear intermediate_data then allocate memory
    */
    intermediate_data_real_real.clear();
    intermediate_data_real_imag.clear();
    intermediate_data_imag_real.clear();
    intermediate_data_imag_imag.clear();
    intermediate_data_real_real.resize(nspectra * ndata_frq * ndata_indirect, 0.0f);
    intermediate_data_real_imag.resize(nspectra * ndata_frq * ndata_indirect, 0.0f);
    intermediate_data_imag_real.resize(nspectra * ndata_frq * ndata_indirect, 0.0f);
    intermediate_data_imag_imag.resize(nspectra * ndata_frq * ndata_indirect, 0.0f);

   
    for(int k=0;k<nspectra;k++)
    {
        int fid_start = k * ndata * ndata_indirect;
        int intermediate_start = k * ndata_frq * ndata_indirect;
        int spectrum_start = k * ndata_frq * ndata_frq_indirect; 

        /**
         * Apodization along direct dimension
         */
        apodization_direct->set_sw(spectral_width);
        apodization_direct->set_n(ndata_original);
        apodization_direct->set_first_point(0.5);
        for(int j=0;j<ndata_indirect;j++)
        {
            /**
             * Skip non-sampled data
            */
            if(nusflags[j]==0) continue;

            apodization_direct->run_apodization(fid_data_real_real.data()+fid_start+j*ndata,ndata,false /**not intevend data*/);
            apodization_direct->run_apodization(fid_data_real_imag.data()+fid_start+j*ndata,ndata,false /**not intevend data*/);
            apodization_direct->run_apodization(fid_data_imag_real.data()+fid_start+j*ndata,ndata,false /**not intevend data*/);
            apodization_direct->run_apodization(fid_data_imag_imag.data()+fid_start+j*ndata,ndata,false /**not intevend data*/);
        }
        apodization_code = 1;

        /**
         * FFT along direct dimension. For real part of indirect dimension
        */
        fft_worker(
            ndata_indirect, nusflags, ndata, ndata_frq,
            fid_data_real_real.data()+fid_start,fid_data_real_imag.data()+fid_start,
            intermediate_data_real_real.data()+intermediate_start, intermediate_data_real_imag.data()+intermediate_start,
            true /** digital fileter*/, false /** swap*/, grpdly
        );
        /**
         * FFT along direct dimension. For imaginary part of indirect dimension
        */
        fft_worker(
            ndata_indirect,nusflags, ndata, ndata_frq,
            fid_data_imag_real.data()+fid_start,fid_data_imag_imag.data()+fid_start,
            intermediate_data_imag_real.data()+intermediate_start, intermediate_data_imag_imag.data()+intermediate_start,
            true /** digital fileter*/, false /** swap*/, grpdly
        );
        /**
         * Apply phase correction along direct dimension.
         * Mathmatically equivalent to direct dimension phase correction code below
        */
        phase_correction_worker(ndata_indirect, ndata_frq, intermediate_data_real_real.data()+intermediate_start, intermediate_data_real_imag.data()+intermediate_start, user_p0_direct /** P0 */,user_p1_direct /** P1*/);
        phase_correction_worker(ndata_indirect, ndata_frq, intermediate_data_imag_real.data()+intermediate_start, intermediate_data_imag_imag.data()+intermediate_start, user_p0_direct /** P0 */,user_p1_direct /** P1*/);
    }
    b_nmrPipe_transposed = false;
    b_frq = true;

    if(b_di_direct == true)
    {
        /**
         * If b_imaginary is false, we do not need real_imag and imag_imag
        */
        b_imaginary = false;
        intermediate_data_real_imag.clear();
        intermediate_data_imag_imag.clear();
    }

    n_outer_dim = ndata_indirect;
    n_inner_dim = ndata_frq;

    /**
     * Run direct dimension extraction here
     * [extraction_from, extraction_to] is the range of data to be extracted.
     *  [0,1] is the full range
    */
    int extraction_from_pos = int(ndata_frq * extraction_from+0.5);
    int extraction_to_pos = int(ndata_frq * extraction_to+0.5);

    if(extraction_from_pos<0) extraction_from_pos = 0;
    if(extraction_to_pos>ndata_frq) extraction_to_pos = ndata_frq;

    if(extraction_from_pos>=extraction_to_pos)
    {
        std::cout<<"Error: extraction_from_pos >= extraction_to_pos. SKIP"<<std::endl;
    }
    else
    {
        std::cout<<"Extraction from "<<extraction_from_pos<<" to "<<extraction_to_pos<<std::endl;
        int n_inner_dim_new = extraction_to_pos - extraction_from_pos;
        
        /**
         * Note: we can copy in place, because new data is always smaller than original data
        */
        for(int k=0;k<nspectra;k++)
        {
            int spectrum_start = k * n_inner_dim * n_outer_dim; 
            int spectrum_start_new = k * n_inner_dim_new * n_outer_dim;
            for(int i=0;i<n_outer_dim;i++)
            {
                int line_start = i * n_inner_dim + spectrum_start + extraction_from_pos;
                int line_start_new = i * n_inner_dim_new + spectrum_start_new;
                /**
                 * Copy data from line_start to line_start_new, for a total of n_inner_dim_new
                */
                for(int j=0;j<n_inner_dim_new;j++)
                {
                    intermediate_data_real_real[line_start_new+j] = intermediate_data_real_real[line_start+extraction_from_pos+j];
                    intermediate_data_imag_real[line_start_new+j] = intermediate_data_imag_real[line_start+extraction_from_pos+j];
                    if(b_imaginary == true)
                    {
                        intermediate_data_real_imag[line_start_new+j] = intermediate_data_real_imag[line_start+extraction_from_pos+j];
                        intermediate_data_imag_imag[line_start_new+j] = intermediate_data_imag_imag[line_start+extraction_from_pos+j];
                    }
                }
            }
        }
        /**
         * Resize spectrum_real_real, spectrum_imag_real, spectrum_real_imag, spectrum_imag_imag
        */
        intermediate_data_real_real.resize(nspectra * n_inner_dim_new * n_outer_dim);
        intermediate_data_imag_real.resize(nspectra * n_inner_dim_new * n_outer_dim);
        if(b_imaginary == true)
        {
            intermediate_data_real_imag.resize(nspectra * n_inner_dim_new * n_outer_dim);
            intermediate_data_imag_imag.resize(nspectra * n_inner_dim_new * n_outer_dim);
        }

        origin_frequency = carrier_frequency - spectral_width  / n_inner_dim * (extraction_to_pos - n_center);
        n_center -= extraction_from_pos;
        spectral_width = spectral_width * ((double)n_inner_dim_new / n_inner_dim);
        n_inner_dim = n_inner_dim_new;
    }

    return true;
}

bool fid_2d::transpose_intermediate_data()
{
    for(int k=0;k<nspectra;k++)
    {
        int intermediate_start = k * n_inner_dim * n_outer_dim;
        transpose_2d(intermediate_data_real_real.data()+intermediate_start, n_outer_dim, n_inner_dim);
        transpose_2d(intermediate_data_imag_real.data()+intermediate_start, n_outer_dim, n_inner_dim);
        
        if(b_imaginary ==true)
        {
            transpose_2d(intermediate_data_real_imag.data()+intermediate_start, n_outer_dim, n_inner_dim);
            transpose_2d(intermediate_data_imag_imag.data()+intermediate_start, n_outer_dim, n_inner_dim);
        }
    }

    b_nmrPipe_transposed = true;

    std::swap(n_outer_dim,n_inner_dim);
    return true;
}

bool fid_2d::run_indirect_dimension(bool b_di_indirect)
{
     /**
     * Do the same for indirect dimension
     */
    ndata_power_of_2_indirect = 1;
    while (ndata_power_of_2_indirect < ndata_indirect)
    {
        ndata_power_of_2_indirect *= 2;
    }
    ndata_frq_indirect = ndata_power_of_2_indirect * zf_indirect;
    n_center_indirect = int(ndata_frq_indirect / 2.0f) + 1;
    origin_frequency_indirect = (carrier_frequency_indirect - spectral_width_indirect * (ndata_frq_indirect - n_center_indirect) / ndata_frq_indirect);


     /**
     * clear frq domain spectrum then allocate memory
     */
    spectrum_real_real.clear();
    spectrum_real_imag.clear();
    spectrum_imag_real.clear();
    spectrum_imag_imag.clear();

    spectrum_real_real.resize(nspectra * ndata_frq * ndata_frq_indirect, 0.0f);
    spectrum_imag_real.resize(nspectra * ndata_frq * ndata_frq_indirect, 0.0f);
    
    if(b_imaginary ==true)
    {
        spectrum_real_imag.resize(nspectra * ndata_frq * ndata_frq_indirect, 0.0f);
        spectrum_imag_imag.resize(nspectra * ndata_frq * ndata_frq_indirect, 0.0f);
    }



    for(int k=0;k<nspectra;k++)
    {
        int intermediate_start = k * ndata_frq * ndata_indirect;
        int spectrum_start = k * ndata_frq * ndata_frq_indirect; 
   
        /**
         * size of intermediate_spectrum is [ndata_frq][ndata_bruker_indirect]
        */
        apodization_indirect->set_sw(spectral_width_indirect);
        apodization_indirect->set_n(ndata_indirect);
        /**
         * For indirect dimension, if user_p1_indirect is around 0 (unit is degree), we set first point to 0.5
         * otherwise, we set first point to 1.0
         * IMPORTANT: This means we can finalize phased spectrum using phased spectrum from program phase_2d, if user_p1_indirect is around +-180 degree
         * we have to rerun this program (fid_2d) with phase information.
        */
        if(fabs(user_p1_indirect)<2.0)
        {
            apodization_indirect->set_first_point(0.5);
        }
        else
        {
            apodization_indirect->set_first_point(1.0);
        }
        for(int i=0;i<ndata_frq;i++)
        {
            apodization_indirect->run_apodization(intermediate_data_real_real.data()+intermediate_start+i*ndata_indirect,ndata_indirect,false /**not intevent data*/);
            apodization_indirect->run_apodization(intermediate_data_imag_real.data()+intermediate_start+i*ndata_indirect,ndata_indirect,false /**not intevent data*/);
            if(b_imaginary ==true)
            {
                apodization_indirect->run_apodization(intermediate_data_real_imag.data()+intermediate_start+i*ndata_indirect,ndata_indirect,false /**not intevent data*/);
                apodization_indirect->run_apodization(intermediate_data_imag_imag.data()+intermediate_start+i*ndata_indirect,ndata_indirect,false /**not intevent data*/);
            }
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

        /**
         * If b_negative == true. Change sign of imaginary part of intermediate_spectrum and intermediate_spectrum_imag
         * For both intermediate_spectrum and intermediate_spectrum_imag, real part is at even index (0,2,4), imaginary part is at odd index (1,3, 5)
         */
        if (b_negative)
        {

            for (int i = 0; i < ndata_frq * ndata_indirect; i++)
            {
                intermediate_data_imag_real[intermediate_start + i] = -intermediate_data_imag_real[intermediate_start + i];
                if(b_imaginary ==true)
                {
                    intermediate_data_imag_imag[intermediate_start + i] = -intermediate_data_imag_imag[intermediate_start + i];
                }
            }
        }
        /**
         * FFT along indirect dimension. there is no NUS flag for indirect dimension
        */
        std::vector<int> dummy_flag(ndata_frq,1);

        fft_worker(ndata_frq, dummy_flag, ndata_indirect, ndata_frq_indirect,
                    intermediate_data_real_real.data()+intermediate_start, 
                    intermediate_data_imag_real.data()+intermediate_start,
                    spectrum_real_real.data()+spectrum_start, spectrum_imag_real.data()+spectrum_start,
                    false /** remove filter*/, b_swap /** swap*/, -1 /**grpdly is not applicable*/);
        phase_correction_worker(ndata_frq, ndata_frq_indirect, spectrum_real_real.data()+spectrum_start, spectrum_imag_real.data()+spectrum_start, user_p0_indirect /** P0 */,user_p1_indirect /** P1*/);


        if(b_imaginary ==true)
        {
            fft_worker(ndata_frq, dummy_flag, ndata_indirect, ndata_frq_indirect,
                        intermediate_data_real_imag.data()+intermediate_start, 
                        intermediate_data_imag_imag.data()+intermediate_start,
                        spectrum_real_imag.data()+spectrum_start, spectrum_imag_imag.data()+spectrum_start,
                        false /** remove filter*/, b_swap /** swap*/, -1 /**grpdly is not applicable*/);
            phase_correction_worker(ndata_frq, ndata_frq_indirect, spectrum_real_imag.data()+spectrum_start, spectrum_imag_imag.data()+spectrum_start, user_p0_indirect /** P0 */,user_p1_indirect /** P1*/);
        }

    }
    b_frq_indirect = true;

    apodization_code_indirect = 1;

    if(b_di_indirect == true)
    {
        /**
         * If b_imaginary_indirect is false, we do not need imag_real and imag_imag
        */
        b_imaginary_indirect = false;
        spectrum_imag_real.clear();
        spectrum_imag_imag.clear();
    }
    
    
    /**
     * Apply receiver_gain if > 1.0
    */
    if (receiver_gain > 1.0)
    {
        for (int i = 0; i < spectrum_real_real.size(); i++)
        {
            spectrum_real_real[i] *= receiver_gain;
            if(b_imaginary_indirect ==true)
            {
                spectrum_imag_real[i] *= receiver_gain;
            }
            if(b_imaginary ==true)
            {
                spectrum_real_imag[i] *= receiver_gain;
                if(b_imaginary_indirect ==true)
                {
                    spectrum_imag_imag[i] *= receiver_gain;
                }
            }
        }
    }
    b_nmrPipe_transposed = true;
    n_inner_dim = ndata_frq_indirect;

    return true;
}

bool fid_2d::untranspose_frquency_domain_data()
{
    for(int k=0;k<nspectra;k++)
    {
        int spectrum_start = k * n_outer_dim * n_inner_dim; 
        transpose_2d(spectrum_real_real.data()+spectrum_start, n_outer_dim, n_inner_dim);
        if(b_imaginary_indirect ==true)
        {
            transpose_2d(spectrum_imag_real.data()+spectrum_start, n_outer_dim, n_inner_dim);
        }

        if(b_imaginary==true)
        {
            transpose_2d(spectrum_real_imag.data()+spectrum_start, n_outer_dim, n_inner_dim);
            if(b_imaginary_indirect ==true)
            {
                transpose_2d(spectrum_imag_imag.data()+spectrum_start, n_outer_dim, n_inner_dim);
            }
        }
    }
    b_nmrPipe_transposed = false;

    std::swap(n_outer_dim,n_inner_dim);

    return true;
}

/**
 * Transpose a 2D array 
 * @param in: input array. size is n_dim1*n_dim2
 * @param n_dim1: size of first dimension (before transpose)
 * @param n_dim2: size of second dimension (before transpose)
*/
bool fid_2d::transpose_2d(float *in, int n_dim1, int n_dim2)
{
    std::vector<float> temp(n_dim1 * n_dim2, 0.0f);
    for (int i = 0; i < n_dim1; i++)
    {
        for (int j = 0; j < n_dim2; j++)
        {
            temp[j * n_dim1 + i] = in[i*n_dim2+j];
        }
    }
    /**
     * Copy data back to in
    */
    for (int i = 0; i < n_dim1 * n_dim2; i++)
    {
        in[i] = temp[i];
    }
    return true;
}

/**
 * @brief create_nmrpipe_dictionary: create nmrpipe dictionary from udict_acqus and derived values
 * @param b_init: whether to initialize nmrpipe_dict_string and nmrpipe_dict_float
 * @param nmrpipe_dict_string: output dictionary for string values
 * @param nmrpipe_dict_float: output dictionary for float values
 */
bool fid_2d::create_nmrpipe_dictionary(std::map<std::string, std::string> &nmrpipe_dict_string, std::map<std::string, float> &nmrpipe_dict_float, bool b_init) const
{
    /**
     * @brief create_default_nmrpipe_dictionary. 
     * This is useful we are NOT reading nmrPipe file.
     */
    if(b_init)
    {
        nmrPipe::create_default_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float);
    }

    /**
     * Fill in some parameters from what we have
     */
    nmrpipe_dict_float["FDDIMCOUNT"] = 2.0f; // 2D data

    if(b_nmrPipe_transposed)
    {
        nmrpipe_dict_float["FDTRANSPOSED"] = 1.0f; // transposed
        nmrpipe_dict_float["FDDIMORDER1"] = 1.0f; // inner loop is FDF1
        nmrpipe_dict_float["FDDIMORDER2"] = 2.0f; // outer loop is FDF2
    }
    else
    {
        nmrpipe_dict_float["FDTRANSPOSED"] = 0.0f; // not transposed
        nmrpipe_dict_float["FDDIMORDER1"] = 2.0f; // inner loop is FDF2
        nmrpipe_dict_float["FDDIMORDER2"] = 1.0f; // outer loop is FDF1
    }

    nmrpipe_dict_string["FDF2LABEL"] = nucleus;
    nmrpipe_dict_string["FDF1LABEL"] = nucleus_indirect;
    

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

    

    if(b_imaginary==true)
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 0.0f;
    }
    else
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 1.0f;
    }

    if(b_imaginary_indirect==true)
    {
        nmrpipe_dict_float["FDF1QUADFLAG"] = 0.0f;
    }
    else
    {
        nmrpipe_dict_float["FDF1QUADFLAG"] = 1.0f;
    }

   
    /**
     * Copied from nmrglue. no idea what they mean
     */
    if (nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF2QUADFLAG"] == nmrpipe_dict_float["FDF3QUADFLAG"] && nmrpipe_dict_float["FDF4QUADFLAG"] == 1.0f)
    {
        nmrpipe_dict_float["FDQUADFLAG"] = 1.0f;
    }

   
    /**
     * Outer loop size need to be adjusted when both are complex, per nmrPipe convention
    */
    int n_scale_factor = 1;
    if(b_imaginary==true && b_imaginary_indirect==true)
    {
        n_scale_factor = 2; 
    }

    /**
     * For both frq and time data, need save these
     * ndata_original is the original size before 0-pad, apoization, ZF and FT
     * ndata is the original size after 0-pad (by Bruker in its fid file), but before apoization, ZF and FT
     * Note: there is no 0-padding along indirect dimension(s)
     */
    nmrpipe_dict_float["FDF2TDSIZE"] = ndata_original;
    nmrpipe_dict_float["FDF1TDSIZE"] = ndata_indirect;

    /**
     * These are the size before apod.
     * nmrPipe ext will change these values, but I am 100% sure how.
     * It seems FDF2APOD = FDF2TDSIZE * percentage of ext operation
     */
    nmrpipe_dict_float["FDF2APOD"] = ndata_original; // or ndata_original, I am not sure
    nmrpipe_dict_float["FDF1APOD"] = ndata_indirect;

    /**
     * Basically we need to adjust CENTER and ORIG according to zero filling and extract (extract has not been implemented yet)
     * In nmrPipe, "-fn EXT -x1 10.8ppm -xn 5.8ppm -sw" will only change ORIG and CENTER, but not SW, OBS, CAR
     */
    nmrpipe_dict_float["FDF2CENTER"] = n_center;
    nmrpipe_dict_float["FDF2ORIG"] = origin_frequency;

    nmrpipe_dict_float["FDF1CENTER"] = n_center_indirect;
    nmrpipe_dict_float["FDF1ORIG"] = origin_frequency_indirect;

    nmrpipe_dict_float["FDF2APODCODE"] = apodization_code;
    nmrpipe_dict_float["FDF1APODCODE"] = apodization_code_indirect;

    /**
     * Both dimension are frq data
    */
    if (b_frq && b_frq_indirect)
    {
        /**
         * we are saving frq data, so set FDF2FTFLAG to 1 and set value for FDF2FTSIZE
         * FDSIZE is the size of inner loop
         * FDSPECNUM is the size of outer loop (when both are complex, we need to double FDSPECNUM)
         * When spectrum is not transposed, FDF2FTSIZE == FDSIZE and FDF1FTSIZE == FDSPECNUM
         * When spectrum is transposed, FDF2FTSIZE == FDSPECNUM and FDF1FTSIZE == FDSIZE
         * In nmrPipe, ext operation will change FDSIZE and/or FDSPECNUM, but not FDF2FTSIZE and FDF1FTSIZE
         */
        nmrpipe_dict_float["FDF2FTSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDF1FTSIZE"] = ndata_frq_indirect;
        nmrpipe_dict_float["FDREALSIZE"] = ndata_frq * ndata_frq_indirect;

        nmrpipe_dict_float["FDF2ZF"] = -ndata_frq; //negative of size after zf
        nmrpipe_dict_float["FDF1ZF"] = -ndata_frq_indirect; //negative of size after zf
       
        nmrpipe_dict_float["FDF2FTFLAG"] = 1.0f;
        nmrpipe_dict_float["FDF1FTFLAG"] = 1.0f;

        nmrpipe_dict_float["FDSIZE"] = n_inner_dim;
        nmrpipe_dict_float["FDSPECNUM"] = n_outer_dim * n_scale_factor;

        nmrpipe_dict_float["FDF2P0"] = user_p0_direct;
        nmrpipe_dict_float["FDF2P1"] = user_p1_direct;
    }
    /**
     * Case 2, only direct dimension is frq data. This is required to run Smile NUS reconstruction
    */
    else if(b_frq == true && b_frq_indirect == false)
    {
        nmrpipe_dict_float["FDF2FTSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDF1FTSIZE"] = 0;
        nmrpipe_dict_float["FDREALSIZE"] = ndata_frq * ndata_indirect;

        nmrpipe_dict_float["FDF2ZF"] = -ndata_frq; //negative of size after zf
        nmrpipe_dict_float["FDF1ZF"] = 0; //negative of size after zf
       
        
        nmrpipe_dict_float["FDF2FTFLAG"] = 1.0f;
        nmrpipe_dict_float["FDF1FTFLAG"] = 0.0f;


        nmrpipe_dict_float["FDSIZE"] = n_inner_dim;
        nmrpipe_dict_float["FDSPECNUM"] = n_outer_dim * n_scale_factor;

        nmrpipe_dict_float["FDF2P0"] = user_p0_direct;
        nmrpipe_dict_float["FDF2P1"] = user_p1_direct;
    }

    /**
     * Todo: other cases  only indirect dimension is frq data.
     * Not sure if this is needed
    */
    else if(b_frq == false && b_frq_indirect == false)
    {
        nmrpipe_dict_float["FDF2FTSIZE"] = 0;
        nmrpipe_dict_float["FDF1FTSIZE"] = 0;
        nmrpipe_dict_float["FDREALSIZE"] = 0;

        nmrpipe_dict_float["FDF2ZF"] = 0; //negative of size after zf
        nmrpipe_dict_float["FDF1ZF"] = 0; //negative of size after zf
       
        
        nmrpipe_dict_float["FDF2FTFLAG"] = 0.0f;
        nmrpipe_dict_float["FDF1FTFLAG"] = 0.0f;


        nmrpipe_dict_float["FDSIZE"] = n_inner_dim;
        nmrpipe_dict_float["FDSPECNUM"] = n_outer_dim * n_scale_factor;     
    }

   
    /**
     * Save indirect dimension phase correction information, which are read from pulse program
     * indirect_p0 < -360 means not set. 
     */
    nmrpipe_dict_float["FDF1P0"] = indirect_p0;
    nmrpipe_dict_float["FDF1P1"] = indirect_p1;
    

    return true;
}

bool fid_2d::write_nmrpipe_ft2_virtual(std::array<float, 512> &nmrpipe_header_data, std::vector<float> &data)
{
    /**
     * create nmrpipe header.
     * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
     * from udict_acqus and derived values
     * True means we are saving frq data
     */
    create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float,true);
    nmrpipe_header_data.fill(0.0f);
    nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);

    data.clear();
    data.resize(ndata_frq_indirect*ndata_frq*4,0.0f);

    for (int j = 0; j < ndata_frq_indirect; j++)
    {
        /**
         * copy spectrum_real_real from j * ndata_frq to (j + 1) * ndata_frq to data at j * ndata_frq * 4
         * then spectrum_real_imag from j * ndata_frq to (j + 1) * ndata_frq to data at j * ndata_frq * 4 + ndata_frq
         * then spectrum_imag_real from j * ndata_frq to (j + 1) * ndata_frq to data at j * ndata_frq * 4 + ndata_frq*2
         * then spectrum_imag_imag from j * ndata_frq to (j + 1) * ndata_frq to data at j * ndata_frq * 4 + ndata_frq*3
         */
        std::copy(spectrum_real_real.begin() + j * ndata_frq, spectrum_real_real.begin() + (j + 1) * ndata_frq, data.begin() + j * ndata_frq * 4);
        std::copy(spectrum_real_imag.begin() + j * ndata_frq, spectrum_real_imag.begin() + (j + 1) * ndata_frq, data.begin() + j * ndata_frq * 4 + ndata_frq);
        std::copy(spectrum_imag_real.begin() + j * ndata_frq, spectrum_imag_real.begin() + (j + 1) * ndata_frq, data.begin() + j * ndata_frq * 4 + ndata_frq*2);
        std::copy(spectrum_imag_imag.begin() + j * ndata_frq, spectrum_imag_imag.begin() + (j + 1) * ndata_frq, data.begin() + j * ndata_frq * 4 + ndata_frq*3);
    }
    return true;
}


bool fid_2d::write_nmrpipe_intermediate(std::string outfname)
{
    if (b_read_bruker_acqus_and_fid == true)
    {
        /**
         * create nmrpipe header.
         * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
         * from udict_acqus and derived values
         * True means we are saving frq data
         */
        create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float,true);

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
        create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float,false);
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
    for (int j = 0; j < n_outer_dim; j++)
    {
        fwrite(intermediate_data_real_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        /**
         * If we are saving real data only, we don't need to save imaginary part for either direct or indirect dimension
         */
        if (b_imaginary == true && b_imaginary_indirect == true)
        {
            fwrite(intermediate_data_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(intermediate_data_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(intermediate_data_imag_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary_indirect == true)
        {
            fwrite(intermediate_data_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary == true)
        {
            fwrite(intermediate_data_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
    }
    fclose(fp);

    return true;
}

/**
 * @brief fid_1d::write_nmrpipe_ft1: write 1D spectrum to nmrpipe file
 * Before writing, define nmrpipe header, set values from udict_acqus and derived values
 * @param outfname: output file name
 * @return true on success, false on failure
 */
bool fid_2d::write_nmrpipe_ft2(std::string outfname)
{

    if (b_read_bruker_acqus_and_fid == true)
    {
        /**
         * create nmrpipe header.
         * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
         * from udict_acqus and derived values
         */
        create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float, true);

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
        create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float, false);
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
    for (int j = 0; j < n_outer_dim; j++)
    {
        fwrite(spectrum_real_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        /**
         * If we are saving real data only, we don't need to save imaginary part for either direct or indirect dimension
         */
        if (b_imaginary == true && b_imaginary_indirect == true)
        {
            fwrite(spectrum_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(spectrum_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(spectrum_imag_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary_indirect == true)
        {
            fwrite(spectrum_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary == true)
        {
            fwrite(spectrum_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
    }
    fclose(fp);

    /**
     * If nspcetra>1, we need to save the rest of the spectra to separate files
     */
    if (nspectra > 1)
    {
        /**
         * Find filename extension: after last . in outfname, such as .ft2.
         * In case there is no . in outfname, ext will be empty and basename will be outfname
         */
        std::string ext = outfname.substr(outfname.find_last_of(".") + 1);
        std::string basename = outfname.substr(0, outfname.find_last_of("."));

        for (int i = 1; i < nspectra; i++) // i starts from 1 because we already saved the first spectrum
        {
            std::string outfname2 = basename + "_" + std::to_string(i) + "." + ext;
            FILE *fp2 = fopen(outfname2.c_str(), "wb");
            if (fp2 == NULL)
            {
                std::cerr << "Error: cannot open file " << outfname2 << " for writing" << std::endl;
                return false;
            }
            /**
             * Write header first
             */
            fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp2);

            /**
             * Write the rest of the spectra
             */
            for (int j = 0; j < n_outer_dim; j++)
            {
                fwrite(spectrum_real_real.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                /**
                 * If we are saving real data only, we don't need to save imaginary part for either direct or indirect dimension
                 */
                if (b_imaginary == true && b_imaginary_indirect == true)
                {
                    fwrite(spectrum_real_imag.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                    fwrite(spectrum_imag_real.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                    fwrite(spectrum_imag_imag.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                }
                else if (b_imaginary_indirect == true)
                {
                    fwrite(spectrum_imag_real.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                }
                else if (b_imaginary == true)
                {
                    fwrite(spectrum_real_imag.data() + i * n_outer_dim * n_inner_dim + j * n_inner_dim, sizeof(float), n_inner_dim, fp2);
                }
            }
            fclose(fp2);
        }
    }

    return true;
}

bool fid_2d::write_nmrpipe_fid(std::string outfname)
{
    if (b_read_bruker_acqus_and_fid == true)
    {
        /**
         * create nmrpipe header.
         * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
         * from udict_acqus and derived values
         */
        create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float, true);

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
        create_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float, false);
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
    for (int j = 0; j < n_outer_dim; j++)
    {
        fwrite(fid_data_real_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        /**
         * If we are saving real data only, we don't need to save imaginary part for either direct or indirect dimension
         */
        if (b_imaginary == true && b_imaginary_indirect == true)
        {
            fwrite(fid_data_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(fid_data_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
            fwrite(fid_data_imag_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary_indirect == true)
        {
            fwrite(fid_data_imag_real.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
        else if (b_imaginary == true)
        {
            fwrite(fid_data_real_imag.data() + j * n_inner_dim, sizeof(float), n_inner_dim, fp);
        }
    }
    fclose(fp);

    return true;
}


/**
 * Functions copied from fid_2d.cpp when combine them
*/

bool fid_2d::read_spectrum(std::string infname)
{
    bool b_read = 0;

    std::string stxt(".txt");
    std::string sft2(".ft2");
    std::string sft3(".ft3");
    std::string sucsf(".ucsf");
    std::string scsv(".csv");

    if (std::equal(stxt.rbegin(), stxt.rend(), infname.rbegin()))
    {
        b_read = read_topspin_txt(infname);
    }
    else if (std::equal(sft2.rbegin(), sft2.rend(), infname.rbegin()))
    {
        b_read = read_nmrpipe_file(infname);
    }
    else if (std::equal(sft3.rbegin(), sft3.rend(), infname.rbegin()))
    {
        b_read = read_nmrpipe_file(infname);
    }
    else if (std::equal(sucsf.rbegin(), sucsf.rend(), infname.rbegin()))
    {
        b_read = read_sparky(infname);
    }
    else if (std::equal(scsv.rbegin(), scsv.rend(), infname.rbegin()))
    {
        b_read = read_mnova(infname);
    }
    else
    {
        b_read = false;
    }

    return b_read;
};

bool fid_2d::init(std::string infname, int noise_flag)
{

    // std::cout<<"peak_diag is "<<peak_diag<<std::endl;
    // std::cout<<"flag_shoulder is "<<flag_shoulder<<std::endl;
    bool b_read;

    b_read = read_spectrum(infname);

    if (b_read)
    {
        std::cout << "Done reading" << std::endl;

        if (noise_flag == 1)
        {
            // estimate_noise_level_mad();    //estimate noise level using MAD
            estimate_noise_level(); // estimate noise level using region by region standard deviation
        }
    }
    return b_read;
}

bool fid_2d::save_mnova(std::string outfname)
{
    std::ofstream fout;

    std::vector<double> ppm1, ppm2;

    for (int i = 0; i < ndata_frq; i++)
    {
        ppm1.push_back(begin1 + i * step1);
    }

    for (int i = 0; i < ndata_frq_indirect; i++)
    {
        ppm2.push_back(begin2 + i * step2);
    }

    // first line, ppm along direct dimension
    for (int i = 0; i < ndata_frq; i++)
    {
        fout << ppm1[i] << " ";
    }
    fout << std::endl;

    for (int i = 0; i < ndata_frq_indirect; i++)
    {
        fout << ppm2[i]; // first element is ppm along indirect dimension
        for (int j = 0; j < ndata_frq; j++)
        {
            fout << " " << spect[i * ndata_frq + j];
        }
        fout << std::endl;
    }
    fout.close();

    return true;
}

bool fid_2d::read_mnova(std::string infname)
{
    int tp;
    std::string line, p;
    std::vector<std::string> ps, lines;
    std::stringstream iss;
    std::ifstream fin(infname.c_str());

    getline(fin, line); // first line
    iss.clear();
    iss.str(line);
    ps.clear();
    while (iss >> p)
    {
        ps.push_back(p);
    }

    if (ps.size() == 2) // new format in mnova 14.0
    {
        std::vector<double> x, y, z;

        lines.push_back(line);
        while (getline(fin, line))
        {
            lines.push_back(line);
        }

        std::string p1, p2;
        for (int i = 0; i < lines.size(); i++)
        {
            iss.clear();
            iss.str(lines[i]);
            iss >> p1;
            iss >> p2;
            z.push_back(atof(p2.c_str()));

            p1.erase(0, 1);
            p1.pop_back();
            iss.clear();
            iss.str(p1);
            ps.clear();
            while (std::getline(iss, p, ':'))
            {
                ps.push_back(p);
            }
            x.push_back(atof(ps[0].c_str()));
            y.push_back(atof(ps[1].c_str()));
        }

        std::set<double> xx(x.begin(), x.end());
        std::set<double> yy(y.begin(), y.end());

        ndata_frq = xx.size();
        ndata_frq_indirect = yy.size();

        begin1 = *xx.rbegin();
        stop1 = *xx.begin();
        step1 = -(begin1 - stop1) / (ndata_frq - 1);
        begin2 = *yy.rbegin();
        stop2 = *yy.begin();
        step2 = -(begin2 - stop2) / (ndata_frq_indirect - 1);

        std::cout << "Direct dimension size is " << ndata_frq << " indirect dimension is " << ndata_frq_indirect << std::endl;
        std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm, stop at " << stop1 << std::endl;
        std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm, stop at " << stop2 << std::endl;

        spectrum_real_real.clear();
        spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spect = spectrum_real_real.data();

        for (int i = 0; i < z.size(); i++)
        {
            int k1 = round(-(begin1 - x[i]) / step1);
            int k2 = round(-(begin2 - y[i]) / step2);
            if (k1 < 0 || k1 >= ndata_frq || k2 < 0 || k2 >= ndata_frq_indirect)
            {
                std::cout << "ERROR in read mnova format, k1=" << k1 << " k2=" << k2 << std::endl;
            }
            spect[k2 * ndata_frq + k1] = float(z[i]);
            // std::cout<<x[i]<<" "<<k1<<" "<<y[i]<<" "<<k2<<" "<<z[i]<<std::endl;
        }
    }

    else // format in mnova 10.0
    {

        if (fabs(atof(ps[ps.size() - 1].c_str())) < 0.0000001)
        {
            ps.erase(ps.end() - 1); // last element is meaningless for first line if it is zero!!
        }

        tp = ps.size();
        ndata_frq = tp;

        float nouse;
        double tswap;
        bool bswap1 = 0, bswap2 = 0;

        begin1 = atof(ps[0].c_str());
        stop1 = atof(ps[tp - 1].c_str());

        if (begin1 < stop1)
        {
            tswap = begin1;
            begin1 = stop1;
            stop1 = tswap;
            bswap1 = 1;
        }

        step1 = (stop1 - begin1) / (ndata_frq - 1);

        while (getline(fin, line))
        {
            if (line.length() > ndata_frq * 2 - 1)
            {
                lines.push_back(line);
            }
        }

        ndata_frq_indirect = lines.size();

        float *spect0;
        spect0 = new float[ndata_frq * ndata_frq_indirect];
        spectrum_real_real.clear();
        spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
        spect = spectrum_real_real.data();
        for (int i = 0; i < lines.size(); i++)
        {
            iss.clear();
            iss.str(lines[i]);
            if (i == 0)
                iss >> begin2;
            else if (i == ndata_frq_indirect - 1)
                iss >> stop2;
            else
                iss >> nouse;

            for (int j = 0; j < ndata_frq; j++)
            {
                iss >> nouse;
                spect0[i * ndata_frq + j] = nouse;
            }
        }
        if (begin2 < stop2)
        {
            tswap = begin2;
            begin2 = stop2;
            stop2 = tswap;
            bswap2 = 1;
        }
        step2 = (stop2 - begin2) / (ndata_frq_indirect - 1);

        if (bswap2 == 0)
        {
            for (int i = 0; i < ndata_frq_indirect; i++)
            {
                for (int j = 0; j < ndata_frq; j++)
                {
                    if (bswap1 == 1)
                        spect[i * ndata_frq + ndata_frq - 1 - j] = spect0[i * ndata_frq + j];
                    else
                        spect[i * ndata_frq + j] = spect0[i * ndata_frq + j];
                }
            }
        }
        else
        {
            for (int i = 0; i < ndata_frq_indirect; i++)
            {
                for (int j = 0; j < ndata_frq; j++)
                {
                    if (bswap1 == 1)
                        spect[(ndata_frq_indirect - 1 - i) * ndata_frq + ndata_frq - 1 - j] = spect0[i * ndata_frq + j];
                    else
                        spect[(ndata_frq_indirect - 1 - i) * ndata_frq + j] = spect0[i * ndata_frq + j];
                }
            }
        }
        delete[] spect0;
    }

    // aribitary frq becuase we don't have that infor
    observed_frequency = observed_frequency_indirect = 850.0;
    spectral_width = observed_frequency * (begin1 - stop1);
    spectral_width_indirect = observed_frequency_indirect * (begin2 - stop2);
    origin_frequency = stop1 * observed_frequency;
    origin_frequency_indirect = stop2 * observed_frequency_indirect;

    std::cout << "Direct dimension size is " << ndata_frq << " indirect dimension is " << ndata_frq_indirect << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm" << std::endl;

    return true;
}

bool fid_2d::get_ppm_from_point()
{
    // get ppm
    p1_ppm.clear();
    p2_ppm.clear();

    for (unsigned int i = 0; i < p1.size(); i++)
    {
        double f1 = begin1 + step1 * (p1[i]); // direct dimension
        double f2 = begin2 + step2 * (p2[i]); // indirect dimension
        p1_ppm.push_back(f1);
        p2_ppm.push_back(f2);
    }

    return true;
};

// read topspin file in ASCIC format, genearted using command totxt
bool fid_2d::read_topspin_txt(std::string infname)
{

    int tp;
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;
    std::ifstream fin(infname.c_str());

    if (!fin)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }

    double f1left, f1right, f2left, f2right;

    f1left = f1right = f2left = f2right = 0.0;
    ndata_frq = ndata_frq_indirect = 0;

    // read in head information
    while (getline(fin, line) && (ndata_frq == 0 || ndata_frq_indirect == 0))
    {
        if (line.find("F1LEFT") != std::string::npos && line.find("F1RIGHT") != std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp = ps.size();
            if (tp >= 9)
            {
                f1left = atof(ps[3].c_str());
                f1right = atof(ps[7].c_str());
            }
        }

        if (line.find("F2LEFT") != std::string::npos && line.find("F2RIGHT") != std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp = ps.size();
            if (tp >= 9)
            {
                f2left = atof(ps[3].c_str());
                f2right = atof(ps[7].c_str());
            }
        }

        if (line.find("NROWS") != std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp = ps.size();
            if (tp >= 4)
            {
                ndata_frq_indirect = atoi(ps[3].c_str());
            }
        }

        if (line.find("NCOLS") != std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp = ps.size();
            if (tp >= 4)
            {
                ndata_frq = atoi(ps[3].c_str());
            }
        }
    }

    spectrum_real_real.clear();
    spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
    spect = spectrum_real_real.data();

    int row_index = -1;
    int flag = 0;
    int col_index;

    while (getline(fin, line))
    {
        if (line.find("# row = ") != std::string::npos)
        {
            row_index = atoi(line.substr(7).c_str());
            col_index = 0;
            continue;
        }
        else if (line.find("#") != std::string::npos)
        {
            continue;
        }

        spect[col_index + row_index * ndata_frq] = atof(line.c_str());
        col_index++;
    }

    begin1 = f2left;
    stop1 = f2right;
    step1 = (stop1 - begin1) / (ndata_frq);
    stop1 = f2right + step1; // stop is the ppm of the last col.

    begin2 = f1left;
    stop2 = f1right;
    step2 = (stop2 - begin2) / (ndata_frq_indirect);
    stop2 = f1right + step2; // stop is the ppm of the last col.

    // fill in required variable to save pipe format
    //  set frq= 850. This is arbitary
    observed_frequency = observed_frequency_indirect = 850.0;
    spectral_width = observed_frequency * (begin1 - stop1);
    spectral_width_indirect = observed_frequency_indirect * (begin2 - stop2);
    origin_frequency = stop1 * observed_frequency;
    origin_frequency_indirect = stop2 * observed_frequency_indirect;

    std::cout << "Direct dimension size is " << ndata_frq << " indirect dimension is " << ndata_frq_indirect << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm" << std::endl;

    return true;
}

bool fid_2d::read_txt(std::string infname)
{
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;
    std::vector<float> temp;

    std::ifstream fin(infname.c_str());

    if (!fin)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }

    ndata_frq = ndata_frq_indirect = 0;
    while (getline(fin, line))
    {
        ndata_frq_indirect++;
        iss.clear();
        iss.str(line);
        ps.clear();
        while (iss >> p)
        {
            temp.push_back(atof(p.c_str()));
        }
        if (ndata_frq == 0)
        {
            ndata_frq = temp.size();
        }
    }

    spect = new float[ndata_frq * ndata_frq_indirect];
    for (unsigned int i = 0; i < ndata_frq * ndata_frq_indirect; i++)
    {
        spect[i] = temp[i];
    }

    begin1 = 0.6;
    stop1 = 0;
    step1 = (stop1 - begin1) / (ndata_frq);
    stop1 = stop1 - step1; // stop is the ppm of the last col.

    begin2 = 0.6;
    stop2 = 0;
    step2 = (stop2 - begin2) / (ndata_frq_indirect);
    stop2 = stop2 - step2; // stop is the ppm of the last col.

    return true;
}

bool fid_2d::read_nmr_ft2_virtual(std::array<float,512> header_, std::vector<float> data)
{
    /**
     * Copy from header_ to header
    */
    std::copy(header_.begin(), header_.end(), nmrpipe_header_data.data());

    process_pipe_header(nmrpipe_header_data);

    if (b_imaginary == true && b_imaginary_indirect == true && b_nmrPipe_transposed == 0) // both are complex
    {
        /**
         * Clear first to be safe
        */
        spectrum_real_real.clear();
        spectrum_real_imag.clear();
        spectrum_imag_real.clear();
        spectrum_imag_imag.clear();

        spectrum_real_real.resize(ndata_frq * ndata_frq_indirect);
        spectrum_real_imag.resize(ndata_frq * ndata_frq_indirect);
        spectrum_imag_real.resize(ndata_frq * ndata_frq_indirect);
        spectrum_imag_imag.resize(ndata_frq * ndata_frq_indirect);
        /**
         * Create a alias for spect to spect_real_real
         */
        spect = spectrum_real_real.data();
        

        for (unsigned int i = 0; i < ndata_frq_indirect; i++)
        {
            /**
             * Copy from data to spectrum_real_real[i*ndata_frq], length is ndata_frq
             * Then copy from data to spectrum_real_imag[i*ndata_frq], length is ndata_frq
             * Then copy from data to spectrum_imag_real[i*ndata_frq], length is ndata_frq
             * Then copy from data to spectrum_imag_imag[i*ndata_frq], length is ndata_frq
             */
            std::copy(data.data() + i * 4 * ndata_frq,               data.data() + i * 4 * ndata_frq + ndata_frq,   spectrum_real_real.data() + i * ndata_frq);
            std::copy(data.data() + i * 4 * ndata_frq + ndata_frq,   data.data() + i * 4 * ndata_frq + ndata_frq*2, spectrum_real_imag.data() + i * ndata_frq);
            std::copy(data.data() + i * 4 * ndata_frq + ndata_frq*2, data.data() + i * 4 * ndata_frq + ndata_frq*3, spectrum_imag_real.data() + i * ndata_frq);
            std::copy(data.data() + i * 4 * ndata_frq + ndata_frq*3, data.data() + i * 4 * ndata_frq + ndata_frq*4, spectrum_imag_imag.data() + i * ndata_frq);
        }


        return true;
    }
    else
    {
        std::cout << "ERROR: complex/real and transpose combination is not supported." << std::endl;
        std::cout << "b_imaginary=" << b_imaginary << " b_imaginary_indirect=" << b_imaginary_indirect << std::endl;
        return false;
    }
}


bool fid_2d::process_pipe_header(std::vector<float> &header)
{
    b_nmrPipe_transposed = int(header[221]) == 1; // 0 normal, 1: transpose

    if (b_nmrPipe_transposed == 0 && int(header[24]) != 2)
    {
        std::cout << " First dimension should always be dimension 2 in non-transposed spectrum, it is " << int(header[24]) << std::endl;
        return false;
    }

    if (b_nmrPipe_transposed == 1 && int(header[24]) != 1)
    {
        std::cout << " First dimension should always be dimension 1 in transposed spectrum, it is " << int(header[24]) << std::endl;
        return false;
    }

    int dims[4];
    double refs[4], sws[4], frqs[4];
    double begins[4], stops[4];

    /**
     * These are from nmrPipe header
     * 0: FDF1
     * 1: FDF2
     * 2: FDF3
     * 3: FDF4
     * Their meaning will be decided by 
     * header[24], header[25], header[26], header[27]
     * (FDDIMORDER1,FDDIMORDER2,FDDIMORDER3,FDDIMORDER4)
    */
    sws[0] = double(header[230 - 1]);
    frqs[0] = double(header[219 - 1]);
    refs[0] = double(header[250 - 1]);

    sws[1] = double(header[101 - 1]);
    frqs[1] = double(header[120 - 1]);
    refs[1] = double(header[102 - 1]);

    sws[2] = double(header[12 - 1]);
    frqs[2] = double(header[11 - 1]);
    refs[2] = double(header[13 - 1]);

    sws[3] = double(header[30 - 1]);
    frqs[3] = double(header[29 - 1]);
    refs[3] = double(header[31 - 1]);

    indirect_p0=float(header[70]); //FDUSER1 in nmrPipe
    indirect_p1=float(header[71]); //FDUSER2 in nmrPipe. Used by this projects only 

    /**
     * 0: complex float
     * 1: real float
     * 
     * FDF1QUADFLAG and FDF2QUADFLAG, respectively
     */
    if(int(header[56 - 1])==0)
    {
        b_imaginary = true;
    }
    else
    {
        b_imaginary = false;
    }
    if(int(header[57 - 1]) == 0)
    {
        b_imaginary_indirect = true;
    }
    else
    {
        b_imaginary_indirect = false;
    }

    /**
     * FDSPECNUM at 219, means FDF1SIZE
     * FDSIZE at 99, means FDF2SIZE (usually direct dimension)
    */
    ndata_frq_indirect = int(header[220 - 1]);
    ndata_frq = int(header[100 - 1]);

    /**
     * NDAPOD, Current valid time-domain size. 
     * FDF2APOD and FDF1APOD, respectively
    */
    ndata_original=int(header[96 - 1]); 
    ndata_indirect=int(header[429 - 1]); 

    
    /**
     * @brief add xdim_ori0 and ydim_ori0 to the next power of 2
     */
    ndata_original=ldw_math_spectrum_2d::next_power_of_2(ndata_original);
    ndata_indirect=ldw_math_spectrum_2d::next_power_of_2(ndata_indirect);


    int ndim = int(header[10 - 1]);
    for (int i = 0; i < 4; i++)
    {
        ldw_math_spectrum_2d::get_ppm_from_header(refs[i], sws[i], frqs[i], stops[i], begins[i]);
    }

    // X is always the 2rd dimension. direct_ndx must ==1 
    int direct_ndx = int(header[24]) - 1;

    // Z can be 4 or 3; indirect_ndxz must ==3 or 2
    int indirect_ndxz = int(header[26]) - 1; // C start from 0,not 1

    // Y can be 3rd or 1st dimension, indirect_ndx must ==0 or 2
    int indirect_ndx = int(header[25]) - 1; // C start from 0,not 1 (For true 2D exp, this is always 0, can be 2 or 3 for slice of high dimension experiment)

    if (b_nmrPipe_transposed == 1)
    {
        std::swap(ndata_frq, ndata_frq_indirect);
        std::swap(direct_ndx, indirect_ndx);
        std::swap(ndata_original, ndata_indirect);
        std::swap(b_imaginary, b_imaginary_indirect);
    }

    /**
     * Get begin and stop for each dimension
     * step will be calculated later after read in spectrum because ydim might be changed
    */
    begin1 = begins[direct_ndx];
    stop1 = stops[direct_ndx];

    begin2 = begins[indirect_ndx];
    stop2 = stops[indirect_ndx];

    /**
     * We don't have zdim or step3
    */
    begin3 = begins[indirect_ndxz];
    stop3 = stops[indirect_ndxz];

    if (b_imaginary == true && b_imaginary_indirect == true && b_nmrPipe_transposed == 0) // both are complex
    {
        /**
         * In nmrPipe, complex data is stored as real/imaginary pair
         * ydim/=2 to be consistent with nmrPipe
         */
        ndata_frq_indirect = ndata_frq_indirect / 2;
    }
    

    double sws_direct = sws[direct_ndx];
    double sws_indirect = sws[indirect_ndx];
    double frqs_direct = frqs[direct_ndx];
    double frqs_indirect = frqs[indirect_ndx];

    /**
     * Calculate step and update begin (to be consistent with nmrPipe)
     * Step is negative.
     * Step calculation is moved after read in spectrum, because ydim might be changed
    */
    step1 = (stop1 - begin1) / ndata_frq;
    begin1 += step1;
    
    step2 = (stop2 - begin2) / ndata_frq_indirect;
    begin2 += step2;

    spectral_width = sws_direct;
    spectral_width_indirect = sws_indirect; // copy to member variables

    observed_frequency = frqs_direct;
    observed_frequency_indirect = frqs_indirect; // copy to member variables

    
    std::cout << "Spectrum width are " << spectral_width << " Hz and " << spectral_width_indirect << " Hz" << std::endl;
    std::cout << "Fields are " << observed_frequency << " mHz and " << observed_frequency_indirect << " mHz" << std::endl;
    std::cout << "Direct dimension size is " << ndata_frq << " indirect dimension is " << ndata_frq_indirect << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " and last is " << stop1 << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per step is " << step2 << " and last is " << stop2 << std::endl;

    if (b_imaginary == true && b_imaginary_indirect == true)
    {
        std::cout << "Data type is complex" << std::endl;
    }
    else
    {
        std::cout << "Data type is real" << std::endl;
    }

    return true;
}

/**
 * This is a protected function, not to be called by user
 * This fucntion only write a ft2 file, for which deep pickerfitter and nmrdraw can read.
 * it is not a 100% compatible ft2 file.
 */

bool fid_2d::write_pipe(std::vector<std::vector<float>> spect, std::string fname)
{

    if(b_read_nmrpipe_fid == false)
    {
        /**
         * Initialize the header with all zeros
        */
        nmrpipe_header_data.clear();
        nmrpipe_header_data.resize(512, 0.0f);
    }

    if (1) // otherwise we just use the input header
    {
        nmrpipe_header_data[0] = 0.0f;
        nmrpipe_header_data[219] = float(ndata_frq_indirect);
        nmrpipe_header_data[99] = float(ndata_frq);
        nmrpipe_header_data[106] = float(1);
        nmrpipe_header_data[256] = float(2);

        nmrpipe_header_data[9] = 2;      // dimension is 2
        nmrpipe_header_data[57] = 0.0f;  // 2d spectrum
        nmrpipe_header_data[221] = 0.0f; // not transposed
        nmrpipe_header_data[24] = 2.0f;  // first dimension is 2
        nmrpipe_header_data[25] = 1.0f;  // second dimension is 1
        nmrpipe_header_data[26] = 3.0f;  // z dimension is 3
        nmrpipe_header_data[27] = 4.0f;  // A dimension is 4

        nmrpipe_header_data[101 - 1] = (float)spectral_width; // second dimension
        nmrpipe_header_data[120 - 1] = (float)observed_frequency;
        nmrpipe_header_data[102 - 1] = (float)origin_frequency;

        nmrpipe_header_data[230 - 1] = (float)spectral_width_indirect; // first dimension
        nmrpipe_header_data[219 - 1] = (float)observed_frequency_indirect;
        nmrpipe_header_data[250 - 1] = (float)origin_frequency_indirect;
    }

    nmrpipe_header_data[56 - 1] = 1.0f; // real data only along indirect dimension
    nmrpipe_header_data[57 - 1] = 1.0f; // real data only along direct dimension

    FILE *fp = fopen(fname.c_str(), "w");
    if (fp == NULL)
    {
        std::cout << "cannot open file " << fname.c_str() << "to write" << std::endl;
    }
    else
    {
        fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);
        for (unsigned int i = 0; i < ndata_frq_indirect; i++)
        {
            fwrite(spect[i].data(), sizeof(float), ndata_frq, fp);
        }
        fclose(fp);
    }

    return true;
};

/**
 * Write a ft2 file, for which deep pickerfitter and nmrdraw can read.
 */
bool fid_2d::write_pipe(std::string fname,bool b_real_only)
{
    FILE *fp = fopen(fname.c_str(), "w");
    if (fp == NULL)
    {
        std::cout << "cannot open file " << fname.c_str() << "to write" << std::endl;
    }
    else
    {
        if(b_real_only==true)
        {
            /**
             * We read in complex data, but only write real data
            */
            if(b_imaginary==true && b_imaginary_indirect==true)
            {
                nmrpipe_header_data[56 - 1] = 1.0f; // real data only along indirect dimension
                nmrpipe_header_data[57 - 1] = 1.0f; // real data only along direct dimension
                nmrpipe_header_data[220 - 1] = ndata_frq_indirect; // ydim is half of the original ydim
            }
        }
        fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);
        if (b_imaginary_indirect== false && b_imaginary == false)
        {
            fwrite(spect, sizeof(float), ndata_frq*ndata_frq_indirect, fp);
        }
        else if (b_imaginary == true && b_imaginary_indirect == true)
        {
            for (unsigned int i = 0; i < ndata_frq_indirect; i++)
            {
                fwrite(spectrum_real_real.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                /**
                 * We have complex data available, and we want to write complex data (not real only)
                */
                if(b_real_only == false && b_imaginary == true && b_imaginary_indirect == true)
                {
                    fwrite(spectrum_real_imag.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                    fwrite(spectrum_imag_real.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                    fwrite(spectrum_imag_imag.data() + i * ndata_frq, sizeof(float), ndata_frq, fp);
                }
            }
        }
        fclose(fp);
    }

    return true;
};



bool fid_2d::read_sparky(std::string infname)
{
    FILE *fp;

    char buffer[10];
    int temp;

    fp = fopen(infname.c_str(), "rb");

    if (fp == NULL)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }

    fread(buffer, 1, 10, fp);
    // std::cout<<buffer<<std::endl;

    fread(buffer, 1, 1, fp);
    temp = int(buffer[0]);
    if (temp != 2)
    {
        std::cout << "Error in sparky format file, dimension is not 2" << std::endl;
        return false;
    }

    fread(buffer, 1, 1, fp);
    fseek(fp, 1, SEEK_CUR);
    temp = int(buffer[0]);
    if (temp != 1)
    {
        std::cout << "Error in sparky format file, it is not in real data" << std::endl;
        return false;
    }

    fread(buffer, 1, 1, fp);
    // std::cout<<"Version is "<< int(buffer[0])<<std::endl;
    fseek(fp, 166, SEEK_CUR);

    // read 2d header

    float center1, center2;
    int tile1, tile2;

    fread(buffer, 1, 6, fp); // nuleus name
    std::cout << "Indirect dimension nuleus " << buffer << std::endl;
    fseek(fp, 2, SEEK_CUR);
    ndata_frq_indirect = read_int(fp);
    fseek(fp, 4, SEEK_CUR);
    tile2 = read_int(fp);
    observed_frequency_indirect = read_float(fp);
    spectral_width_indirect = read_float(fp);
    center2 = read_float(fp);
    fseek(fp, 96, SEEK_CUR);

    fread(buffer, 1, 6, fp); // nuleus name
    std::cout << "Direct dimension nuleus " << buffer << std::endl;
    fseek(fp, 2, SEEK_CUR);
    ndata_frq = read_int(fp);
    fseek(fp, 4, SEEK_CUR);
    tile1 = read_int(fp);
    observed_frequency = read_float(fp);
    spectral_width = read_float(fp);
    center1 = read_float(fp);
    fseek(fp, 96, SEEK_CUR);

    // read in data here
    spectrum_real_real.clear();
    spectrum_real_real.resize(ndata_frq * ndata_frq_indirect, 0.0f);
    spect = spectrum_real_real.data();

    int ntile1 = int(ceil((double(ndata_frq)) / tile1));
    int ntile2 = int(ceil((double(ndata_frq_indirect)) / tile2));

    int last_tile1 = ndata_frq % tile1;
    if (last_tile1 == 0)
        last_tile1 = tile1;

    float *float_buff;
    float_buff = new float[tile1];

    for (int i = 0; i < ntile2; i++)
    {
        for (int j = 0; j < ntile1; j++)
        {
            for (int m = 0; m < tile2; m++)
            {
                read_float(fp, tile1, float_buff);
                if (i * tile2 + m < ndata_frq_indirect)
                {
                    if (j == ntile1 - 1)
                        memcpy(spect + (i * tile2 + m) * ndata_frq + (j * tile1), float_buff, last_tile1 * 4);
                    else
                        memcpy(spect + (i * tile2 + m) * ndata_frq + (j * tile1), float_buff, tile1 * 4);
                }
            }
        }
    }

    delete[] float_buff;

    float range1 = spectral_width / observed_frequency;
    step1 = -range1 / ndata_frq;
    begin1 = center1 + range1 / 2;
    stop1 = center1 - range1 / 2;

    float range2 = spectral_width_indirect / observed_frequency_indirect;
    step2 = -range2 / ndata_frq_indirect;
    begin2 = center2 + range2 / 2;
    stop2 = center2 - range2 / 2;

    // file in ref from center, so that we can save in pipe format if required.
    origin_frequency = center1 * observed_frequency - spectral_width / 2;
    origin_frequency_indirect = center2 * observed_frequency_indirect - spectral_width_indirect / 2;

    std::cout << "Spectrum width are " << spectral_width << " Hz and " << spectral_width_indirect << " Hz" << std::endl;
    std::cout << "Fields are " << observed_frequency << " mHz and " << observed_frequency_indirect << " mHz" << std::endl;
    std::cout << "Direct dimension size is " << ndata_frq << " indirect dimension is " << ndata_frq_indirect << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm" << std::endl;

    fclose(fp);

    return true;
};

void fid_2d::estimate_noise_level_mad()
{
    std::cout << "In noise estimation, ndata_frq*ydim is " << ndata_frq * ndata_frq_indirect << std::endl;

    std::vector<float> t(spect, spect + ndata_frq * ndata_frq_indirect);

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

    // noise_level=87353.0;
    // std::cout<<"ERROR: set noise level to "<<noise_level<<std::endl;

    std::cout << "First round, noise level is " << noise_level << std::endl;

    std::vector<int> flag(ndata_frq * ndata_frq_indirect, 0); // flag

    for (int j = 0; j < ndata_frq_indirect; j++)
    {
        for (int i = 0; i < ndata_frq; i++)
        {
            if (t[j * ndata_frq + i] > 5.5 * noise_level)
            {
                int ystart = std::max(j - 5, 0);
                int yend = std::min(j + 6, ndata_frq_indirect);
                int xstart = std::max(i - 5, 0);
                int xend = std::min(i + 6, ndata_frq);
                for (int m = ystart; m < yend; m++)
                {
                    for (int n = xstart; n < xend; n++)
                    {
                        flag[m * ndata_frq + n] = 1;
                    }
                }
            }
        }
    }
    // after this, all datapoint > 5.5*noise and their surrounding datapoint are labeled as 1.
    scores.clear();
    for (int j = 0; j < ndata_frq_indirect; j++)
    {
        for (int i = 0; i < ndata_frq; i++)
        {
            if (flag[j * ndata_frq + i] == 0)
            {
                scores.push_back(t[j * ndata_frq + i]);
            }
        }
    }
    sort(scores.begin(), scores.end());
    noise_level = scores[scores.size() / 2] * 1.4826;
    if (noise_level <= 0.0)
        noise_level = 0.1; // artificail spectrum w/o noise
    std::cout << "Final noise level is estiamted to be " << noise_level << std::endl;

    // estimate noise level column by column for TOCSY t1 noise belt identification!!
    for (int i = 0; i < ndata_frq; i++)
    {
        std::vector<float> scores;
        scores.clear();
        for (int j = 0; j < ndata_frq_indirect; j++)
        {
            scores.push_back(fabs(spect[j * ndata_frq + i]));
        }
        sort(scores.begin(), scores.end());
        noise_level_columns.push_back(scores[ndata_frq_indirect / 3] * 1.4826);
    }

    // estimate noise level row by row
    for (int j = 0; j < ndata_frq_indirect; j++)
    {
        std::vector<float> scores;
        scores.clear();
        for (int i = 0; i < ndata_frq; i++)
        {
            scores.push_back(fabs(spect[j * ndata_frq + i]));
        }
        sort(scores.begin(), scores.end());
        noise_level_rows.push_back(scores[ndata_frq / 3] * 1.4826);
    }
};

void fid_2d::estimate_noise_level()
{
    std::cout << "In noise estimation, ndata_frq*ydim is " << ndata_frq * ndata_frq_indirect << std::endl;

    int n_segment_x = ndata_frq / 32;
    int n_segment_y = ndata_frq_indirect / 32;

    std::vector<float> variances;      // variance of each segment
    std::vector<float> maximal_values; // maximal value of each segment

    /**
     * loop through each segment, and calculate variance
     */
    for (int i = 0; i < n_segment_x; i++)
    {
        for (int j = 0; j < n_segment_y; j++)
        {
            std::vector<float> t;
            for (int m = 0; m < 32; m++)
            {
                for (int n = 0; n < 32; n++)
                {
                    t.push_back(spect[(j * 32 + m) * ndata_frq + i * 32 + n]);
                }
            }

            /**
             * calculate variance of this segment. Substract the mean value of this segment first
             * also calculate the max value of this segment
             */
            float max_of_t = 0.0f;
            float mean_of_t = 0.0f;
            for (int k = 0; k < t.size(); k++)
            {
                mean_of_t += t[k];
                if (fabs(t[k]) > max_of_t)
                {
                    max_of_t = fabs(t[k]);
                }
            }
            mean_of_t /= t.size();

            float variance_of_t = 0.0f;
            for (int k = 0; k < t.size(); k++)
            {
                variance_of_t += (t[k] - mean_of_t) * (t[k] - mean_of_t);
            }
            variance_of_t /= t.size();
            variances.push_back(variance_of_t);
            maximal_values.push_back(max_of_t);
        }
    }

    /**
     * sort the variance, and get the median value
     */
    std::vector<float> variances_sorted = variances;
    sort(variances_sorted.begin(), variances_sorted.end());
    noise_level = sqrt(variances_sorted[variances_sorted.size() / 2]);
    std::cout << "Noise level is " << noise_level << " using variance estimation." << std::endl;

    /**
     * loop through maximal_values, remove the ones that are larger than 10.0*noise_level
     * remove the corresponding variance as well
     */
    for (int i = maximal_values.size() - 1; i >= 0; i--)
    {
        if (maximal_values[i] > 10.0 * noise_level)
        {
            maximal_values.erase(maximal_values.begin() + i);
            variances.erase(variances.begin() + i);
        }
    }

    /**
     * sort the variance, and get the median value
     */
    variances_sorted = variances;
    sort(variances_sorted.begin(), variances_sorted.end());
    noise_level = sqrt(variances_sorted[variances_sorted.size() / 2]);

    std::cout << "Final noise level is estiamted to be " << noise_level << std::endl;

    // estimate noise level column by column for TOCSY t1 noise belt identification!!
    for (int i = 0; i < ndata_frq; i++)
    {
        std::vector<float> scores;
        scores.clear();
        for (int j = 0; j < ndata_frq_indirect; j++)
        {
            scores.push_back(fabs(spect[j * ndata_frq + i]));
        }
        sort(scores.begin(), scores.end());
        noise_level_columns.push_back(scores[ndata_frq_indirect / 3] * 1.4826);
    }

    // estimate noise level row by row
    for (int j = 0; j < ndata_frq_indirect; j++)
    {
        std::vector<float> scores;
        scores.clear();
        for (int i = 0; i < ndata_frq; i++)
        {
            scores.push_back(fabs(spect[j * ndata_frq + i]));
        }
        sort(scores.begin(), scores.end());
        noise_level_rows.push_back(scores[ndata_frq / 3] * 1.4826);
    }

    return;
}