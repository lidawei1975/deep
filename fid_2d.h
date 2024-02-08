
#ifndef FID_2D_H
#define FID_2D_H

#include <vector>
#include <map>
#include <string>

#include "fid_1d.h"

class fid_2d : public fid_base
{
protected:

    /**
     * PART 0. flags
    */
    bool b_read_bruker_acqus_and_fid; //true if read_bruker_files is called. Only one can be true
    bool b_read_nmrpipe_fid; //true if read_nmrpipe_fid is called. Only one can be true


    /**
     * PART1. These variables are read from acqus and acqu2s files
    */
    int fnmode; //indirection dimension encoding (negative image, fftshift, etc)
    double grpdly; //group delay. Required to remove digitizer filter
    FID_DATA_TYPE data_type;  //Note: Bruker may save FID as int32 or float32
    FID_DATA_COMPLEXITY data_complexity; //real or complex

    std::map<std::string, std::string> udict_acqus_direct; //udict for acqus file along direct dimension
    std::map<std::string, std::string> udict_acqus_indirect; //udict for acqus file along indirect dimension

    /**
     * The following variables are read from pulse program
    */
    double indirect_p0; //linear phase correction along indirect dimension, provided by standard pulse program
    double indirect_p1; //linear phase correction along indirect dimension, provided by standard pulse program

    /**
     * The following variables are read from acqus and acqus2 file
    */
    double spectral_width; //spectral width
    double observed_frequency; //observed frequency
    double carrier_frequency; //carrier frequency

    double spectral_width_indirect; //spectral width along indirect dimension
    double observed_frequency_indirect; //observed frequency along indirect dimension
    double carrier_frequency_indirect; //carrier frequency along indirect dimension

    double receiver_gain; //receiver gain. pipe doesn't use this value

    

    /**
     * ndata_bruker: number of data points in the fid file, using Bruker convention. 
     * ndata: number of data points in the fid file, using our convention.
     * For complex data, ndata_bruker=ndata*2
     * For real data, ndata_bruker=ndata
    */
    int ndata_bruker; 
    int ndata; 
    int ndata_bruker_indirect;
    int ndata_indirect;

    /**
     * ndata must be a power of 2. We define ndata_power_of_2 as the smallest power of 2 that is larger than or equal to ndata.
    */
    int ndata_power_of_2;
    int ndata_power_of_2_indirect;

    int zf; //zero filling factor: 2,4,8, etc. 
    int zf_indirect; //zero filling factor: 2,4,8, etc for indirect dimension
    int ndata_frq; //ndata_frq=ndata_power_of_2*zf=spectrum_real.size()
    int ndata_frq_indirect;


    /**
     * PART 2. Spectral data. always convert to float in our calculations
    */
    std::vector<float> fid_data_float; 

    /**
     * These 4 vectors are used to save the frequency domain data (fft result)
    */
    std::vector<float> spectrum_real_real;
    std::vector<float> spectrum_real_imag;
    std::vector<float> spectrum_imag_real;
    std::vector<float> spectrum_imag_imag;

    /**
     * Optional phase correction, provided by user
    */
    double user_p0_direct,user_p1_direct,user_p0_indirect,user_p1_indirect;

    /**
     * PART 3. nmrPipe header information
    */
    std::map<std::string, std::string> nmrpipe_dict_string;
    std::map<std::string, float> nmrpipe_dict_float;
    std::vector<float> nmrpipe_header_data;  //nmrPipe header data 512*4

    /**
     * Apodization 
    */
    class apodization *apodization_direct,*apodization_indirect;
    

    /**
     * PART 4. Functions
    */


    /**
     * Helper function
     * Will fillin most PART 1 varibles from PART 3 variables
     * 
    */
    bool set_varibles_from_nmrpipe_dictionary();

    /**
     * Helper function for write ft2
    */
    bool create_nmrpipe_dictionary(bool b_frq,std::map<std::string, std::string> &nmrpipe_dict_string,std::map<std::string, float> &nmrpipe_dict_float) const;

    /**
     * Lower level FFT runner
    */
    bool fft_worker(int n_dim1, int n_dim2, int n_dim2_frq,const std::vector<float> &in, 
                    std::vector<float> &out1, std::vector<float> &out2,
                    bool b_remove_filter,bool b_swap,int grpdly_) const;

    /**
     * Apply linear phase correction along dim2
     * for all dim1 rows
    */
    bool phase_correction_worker(int n_dim1,int n_dim2, std::vector<float> &spectrum_real, std::vector<float> &spectrum_imag, double p0, double p1) const;

    /**
     * Transpose a 2D array 
     * @param in: input array. size is n_dim1*n_dim2
     * @param n_dim1: size of first dimension (before transpose)
     * @param n_dim2: size of second dimension (before transpose)
    */
    bool transpose_2d(std::vector<float> &in, int n_dim1, int n_dim2);

public:
    fid_2d();
    ~fid_2d();

    /**
     * This function will get reference to two apodization objects
    */
    bool set_up_apodization(apodization *apodization_direct_in,apodization *apodization_indirect_in);


    /**
     * @brief read_bruker_folder:  read fid and parameter files from Bruker folder
     * This function will find the 3 files: acqus, acqu2s and fid, and call read_bruker_files to read them
     * this function will change the value of the following variables:
     * fid_data_float or fid_data_int
     * udict
    */
    bool read_bruker_folder(std::string folder_name); 

    /**
     * @brief run_zf (set up zf and zf_indirect only)
    */
    inline bool run_zf(int zf_in, int zf_indirect_in)
    {
        zf=zf_in;
        zf_indirect=zf_indirect_in;
        return true;
    };

    /**
     * Load user provided phase correction values from a file
    */
    bool read_phase_correction(std::string fname);
    

    /**
     * @brief read_bruker_files: read 3 files: acqu2s acqus and fid. 
     * fid files can be vector of files. In that case, we will read all of them and added them together.
     * We provide both read_bruker_folder and read_bruker_files for convenience.
     * this function will change the value of the following variables:
     * fid_data_float or fid_data_int
     * udict_acqus
    */
    bool read_bruker_files(const std::string &pulse_program_name,const std::string &acqus_file2_name,const std::string &acqus_file_name, const std::vector<std::string> &fid_data_file_names);

    /**
     * We can also read a nmrPipe format file
    */
    bool read_nmrpipe_fid(const std::string &nmrpipe_fid_file_name);


    bool run_fft_and_rm_bruker_filter();

    bool write_nmrpipe_ft2(std::string outfname,bool b_real_only=false);


    bool run_apodization(FID_APODIZATION_TYPE apodization_type, double p1,double p2=0.0,double p3=0.0,double p4=0.0,double p5=0.0,double p6=0.0);
};

#endif