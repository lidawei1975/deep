
#ifndef FID_2D_H
#define FID_2D_H

#include <vector>
#include <map>
#include <string>

#include "fid_1d.h"

namespace ldw_math_spectrum_2d
{
    bool SplitFilename (const std::string& str, std::string &path_name, std::string &file_name, std::string &file_name_ext);
    
    double calcualte_median(std::vector<double> scores);
    
    template<typename T>
    void sortArr(std::vector<T> &arr, std::vector<int> &ndx);
    
    bool spline_expand(int ndata_frq, int ydim, float *spect,std::vector<double> &final_data);
};

class fid_2d : public fid_base
{
public:
    double begin3,stop3,step3;  //3rd (indirect) dimension, in case of 3D NMR

protected:

    /**
     * PART 0. flags
    */
    bool b_read_bruker_acqus_and_fid; //true if read_bruker_files is called. Only one can be true
    bool b_read_nmrpipe_fid; //true if read_nmrpipe_file is called. Only one can be true

    std::string aqseq; //"321" or "312"

    bool b_negative; //true if imaginary data along indirect dimension is negative (needs to be flipped)

    bool b_first_only; //if true, only process the first spectrum in a pseudo 3D NMR

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

    std::string nucleus; //nucleus
    std::string nucleus_indirect; //nucleus along indirect dimension

    /**
     * The following variables are read from acqus and acqus2 file.
     * They will be written to nmrPipe header
     * When tranasposed flag in nmrPipe is on, spectral_width and spectral_width_indirect will be swapped
     * as well as
     * observed_frequency and observed_frequency_indirect
     * carrier_frequency and carrier_frequency_indirect
    */
    double spectral_width; //spectral width
    double observed_frequency; //observed frequency
    double carrier_frequency; //carrier frequency
    double origin_frequency; //reference frequency

    double spectral_width_indirect; //spectral width along indirect dimension
    double observed_frequency_indirect; //observed frequency along indirect dimension
    double carrier_frequency_indirect; //carrier frequency along indirect dimension
    double reference_frequency_indirect; //reference frequency along indirect dimension
    double origin_frequency_indirect; //reference frequency along indirect dimension

    bool b_frq; //true if we run FFT along direct dimension, false if we still keep the time domain data
    bool b_frq_indirect; //true if we run FFT along indirect dimension, false if we still keep the time domain data

    bool b_imaginary; //true if imaginary data is kept, false if imaginary data is removed
    bool b_imaginary_indirect; //true if imaginary data is kept for indirect dimension, false if imaginary data is removed

    double receiver_gain; //receiver gain. pipe doesn't use this value

    /**
     * ture if nmrPipe header is transposed
     * When transposed, we still save direct dimension to FDF2????, indirect dimension to FDF1????
     * but we will ONLY change the following keys in nmrPipe header
     * FROM:
     *  dict_float["FDDIMORDER1"] = 2.0f;
     *  dict_float["FDDIMORDER2"] = 1.0f;
     * TO:
     *  dict_float["FDDIMORDER1"] = 1.0f;
     *  dict_float["FDDIMORDER2"] = 2.0f;
     *  And we also change how we save the spectral data. 
    */
    bool  b_nmrPipe_transposed; 

    

    /**
     * @ndata_bruker: number of data points in the fid file, using Bruker convention. 
     * @ndata: number of data points in the fid file, using our convention.
     * @ndata_original: number of data points in the fid file, using our convention, before padding zeros
     * Padding is NOT necessary for indirect dimension
     * For complex data, ndata_bruker=ndata*2
     * For real data, ndata_bruker=ndata
    */
    int ndata_bruker; 
    int ndata; 
    int ndata_bruker_indirect;
    int ndata_indirect;
    int ndata_original; 

    /**
     * How many spectra are there in the ser (or fid) file. Useful for pseudo 3D NMR
     * which contains multiple 2D NMR spectra
    */
    int nspectra; //number of spectra (in pseudo 3D NMR, nspectra>1, in 3D NMR, nspectra=1)

    /**
     * ndata must be a power of 2. We define ndata_power_of_2 as the smallest power of 2 that is larger than or equal to ndata.
    */
    int ndata_power_of_2;
    int ndata_power_of_2_indirect;

    int zf; //zero filling factor: 2,4,8, etc. 
    int zf_indirect; //zero filling factor: 2,4,8, etc for indirect dimension
    int ndata_frq; //ndata_frq=ndata_power_of_2*zf=spectrum_real.size()
    int ndata_frq_indirect;

    int apodization_code;
    int apodization_code_indirect;

    /**
     * nmrPipe use center to track EXTraction operation.
    */
    int n_center; //center of the spectrum before any EXTraction operation
    int n_center_indirect; //center of the spectrum before any EXTraction operation

    /**
     * Extraction operation along direct dimension [0,1)
    */
    double extraction_from;
    double extraction_to;

    /**
     * Current dimension of the data
     * If b_frq, n_inner_dim=ndata_frq, otherwise n_inner_dim=ndata
     * if b_frq_indirect, n_outer_dim=ndata_frq_indirect, otherwise n_outer_dim=ndata_indirect
     * When transposed, n_outer_dim and n_inner_dim will be swapped
     * EXTraction will change current n_inner_dim
    */
    int n_outer_dim; //outer dimension
    int n_inner_dim; //inner dimension

    /**
     * NUS list. empty means no NUS (fully sampled). Starts from 0
     * nusflags.size()=ndata_indirect, nusflags[nuslists]=1, others are 0
    */
    std::vector<int> nuslists;
    std::vector<int> nusflags; 


    /**
     * PART 2. Spectral data. always convert to float in our calculations
    */
    std::vector<float> fid_data_float; 

    /**
     * Follow nmrPipe convention, we convert the data into the following 4 vectors from fid_data_float
     * fid_data_real_real, fid_data_real_imag, fid_data_imag_real, fid_data_imag_imag
    */
    std::vector<float> fid_data_real_real;
    std::vector<float> fid_data_real_imag;
    std::vector<float> fid_data_imag_real;
    std::vector<float> fid_data_imag_imag;

    /**
     * There are intermediate data (after fft along direct dimension, before fft along indirect dimension)
    */
    std::vector<float> intermediate_data_real_real;
    std::vector<float> intermediate_data_real_imag;
    std::vector<float> intermediate_data_imag_real;
    std::vector<float> intermediate_data_imag_imag;

    /**
     * These 4 vectors are used to save the frequency domain data (fft result)
    */
    std::vector<float> spectrum_real_real;
    std::vector<float> spectrum_real_imag;
    std::vector<float> spectrum_imag_real;
    std::vector<float> spectrum_imag_imag;
    float *spect;  //alias of spectrum_real_real

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
     * Part 4, spectrum_io for DP and VF
    */
    double begin1,step1,stop1;  //direct dimension
    double begin2,step2,stop2;  //indirect dimension
    double spectral_width_3,observed_frequency_3,origin_frequency_3; //ppm information in case of 3D
    float noise_level;  //spectrum noise level, 1.4826*medium(abs(spect))
    std::vector<double> noise_level_columns; //noise level of each column
    std::vector<double> noise_level_rows; //noise level of each row
    

    /**
     * PART 5. Functions
    */
protected:
    void estimate_noise_level_mad();   //estimate noise level, using MAD method 1.4826*medium(abs(spect))
    void estimate_noise_level(); //estimate noise level, using region by region standard deviation
    //These three are for endian conversion, required by sparky format!
    float read_float(FILE *);
    bool read_float(FILE *,int, float *);
    int read_int(FILE *);
    bool process_pipe_header(std::vector<float> &header);

    bool read_topspin_txt(std::string fname); //topspin totxt format
    bool read_txt(std::string fname);  //from matlab save -ASCII
    bool read_sparky(std::string fname);  //ucsf sparky
    bool read_mnova(std::string fname); //mnova csv file format
    bool get_ppm_from_point();
    /**
     * For internal use only. Write spectrum to a file
    */
    bool write_pipe(std::vector<std::vector<float> > spect, std::string fname); 

    
    std::string infname; //file name
    double user_scale,user_scale2; //minimal peak intesntiy in picking and fitting
    double median_width_x,median_width_y; //peak width median from either picking or fitting

    //peaks, used by both picking and fitting classes
    std::vector<int> peak_index; //used when read in peak list to keep track of all peaks.

/**
 * Peaks defined for DP and VF
*/

public:
    std::vector<double> p1,p2,p_intensity;  //grid peak position, peak amp 
    std::vector<double> p1_ppm,p2_ppm; //ppm value of peak pos, obtained from p1 and p2
    std::vector<double> p_confidencex,p_confidencey; //confidence level of peaks
    std::vector<std::string> user_comments; //user comment of about a peak!
    std::vector<double> sigmax,sigmay;  //for Gaussian and Voigt fitting, est in picking too
    std::vector<double> gammax,gammay; //for voigt fittting and ann picking, set to 0 in picking!


    /**
     * Helper function
     * Will fillin most PART 1 varibles from PART 3 variables
     * 
    */
    bool set_varibles_from_nmrpipe_dictionary();

    /**
     * Helper function for write ft2
    */
    bool create_nmrpipe_dictionary(std::map<std::string, std::string> &nmrpipe_dict_string,std::map<std::string, float> &nmrpipe_dict_float, bool) const;

    /**
     * Lower level FFT runner
    */
    bool fft_worker(int n_dim1, const std::vector<int> dim1_flag, int n_dim2, int n_dim2_frq,
                    const float  *in1, const float  *in2,
                    float *out1, float *out2,
                    bool b_remove_filter,bool b_swap,double grpdly_) const;

    /**
     * Apply linear phase correction along dim2
     * for all dim1 rows
    */
    bool phase_correction_worker(int n_dim1,int n_dim2, float *spectrum_real, float *spectrum_imag, double p0, double p1) const;

    /**
     * Transpose a 2D array 
     * @param data: input array. size is n_dim1*n_dim2
     * @param n_dim1: size of first dimension (before transpose)
     * @param n_dim2: size of second dimension (before transpose)
    */
    bool transpose_2d(float *data, int n_dim1, int n_dim2);

    /**
     * Processing functions
    */
    bool run_direct_dimension(bool);
    bool transpose_intermediate_data();
    bool run_indirect_dimension(bool);
    bool untranspose_frquency_domain_data();

public:
    fid_2d();
    ~fid_2d();

    /**
     * Read nus list from a file. One line per sampling point or one element per sampling point in one line
    */
    bool read_nus_list(std::string fname);

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

    inline bool extract_region(double from, double to)
    {
        if(from<0 || from>=1 || to<=0 || to>1 || from>=to)
        {
            std::cerr << "Error: from and to must be in [0,1) and from<to." << std::endl;
            return false;
        }
        extraction_from=from;
        extraction_to=to;
        return true;
    }   

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
    bool read_nmrpipe_file(const std::string &nmrpipe_fid_file_name);

    bool water_suppression();

    bool polynorminal_baseline(int order);

    bool full_process(bool b_di_direct=false, bool b_di_indirect=false);

    bool direct_only_process(bool b_di_direct=false);

    bool indirect_only_process(bool b_di_indirect=false);

    bool other_process(bool b_di_direct=false, bool b_di_indirect=false);

    bool write_nmrpipe_ft2(std::string outfname);

    bool write_nmrpipe_intermediate(std::string outfname);

    bool write_nmrpipe_fid(std::string outfname);

    bool write_nmrpipe_ft2_virtual(std::array<float,512> &header, std::vector<float> &data);

    bool run_apodization(FID_APODIZATION_TYPE apodization_type, double p1,double p2=0.0,double p3=0.0,double p4=0.0,double p5=0.0,double p6=0.0);

    bool write_json(std::string fname);

    bool write_pseudo3d_json(std::string fname);

    inline bool set_aqseq(std::string aqseq_in)
    {
        if(aqseq_in != "321" && aqseq_in != "312")
        {
            std::cerr << "Error: aqseq must be 321 or 312." << std::endl;
            return false;
        }
        aqseq=aqseq_in;
        return true;
    };

    inline void set_negative(bool b)
    {
        b_negative=b;
        return;
    }

    inline void set_first_only(bool b)
    {
        b_first_only=b;
        return;
    }

    /**
     * Read only access of nspectra
    */
    inline int get_nspectra() const
    {
        return nspectra;
    };


public:
    inline void set_scale(double x,double y)
    {
        user_scale=x;
        user_scale2=y;
    };

    inline void set_noise_level(double t) {noise_level=t; std::cout<<"Direct set noise level to "<<t<<std::endl;}

    bool init(std::string, int noise_flag=1);  //read spectrum and est noise

    bool write_pipe(std::string fname, bool b_real_only = false);
    bool save_mnova(std::string fname);




    
    bool read_spectrum(std::string); //read spectrum only
    bool read_nmr_ft2_virtual(std::array<float,512> header, std::vector<float> data);
    
    /**
     * Public function to write spectrum to a file.
     * b_real_only means we will remove all imaginary part of the spectrum
    */


    
    inline float * get_spect_data()
    {
        return spect;
    };
    
    inline float get_noise_level() {return noise_level;};
    
    inline void get_ppm_infor(double *begins,double *steps)
    {
        begins[0]=begin1;
        begins[1]=begin2;
        steps[0]=step1;
        steps[1]=step2;
        
    };

    inline void get_dim(int *n1, int *n2){*n1=ndata_frq;*n2=ndata_frq_indirect;};


    inline void get_median_width(double *x, double *y)
    {
        *x=median_width_x;
        *y=median_width_y;
    };

};

#endif