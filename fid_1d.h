
#ifndef FID_1D_H
#define FID_1D_H

#include <vector>
#include <map>
#include <iostream>

enum FID_DATA_COMPLEXITY
{
    FID_DATA_COMPLEXITY_REAL = 0,
    FID_DATA_COMPLEXITY_COMPLEX = 1
};


enum FID_DATA_TYPE
{
    FID_DATA_TYPE_FLOAT64 = 0,
    FID_DATA_TYPE_INT32 = 1,
    FID_DATA_TYPE_FLOAT32 = 2,
    FID_DATA_TYPE_NOT_DEFINED = 3
};

enum FID_APODIZATION_TYPE
{
    FID_APODIZATION_NONE = 0,
    FID_APODIZATION_SP = 1,
    FID_APODIZATION_EM = 2,
    FID_APODIZATION_GM = 3,
};


namespace nmrPipe
{
    bool nmrpipe_header_to_dictionary(float * nmrpipe_header_data,std::map<std::string,std::string> & dict_string,std::map<std::string, float> & dict_float);
    bool nmrpipe_dictionary_to_header(float * nmrpipe_header_data,const std::map<std::string,std::string> &dict_string,const std::map<std::string, float> & dict_float);
    bool create_default_nmrpipe_dictionary(std::map<std::string,std::string> & dict_string,std::map<std::string, float> &dict_float);
};

namespace fid_1d_helper
{
    size_t split(const std::string &txt, std::vector<std::string> &strs, char ch);
};

struct spectrum_1d_peaks
{
    std::vector<double> a;          //peak intensity 
    std::vector<double> x;          //peak coordinates
    std::vector<double> ppm;        //peak coordinates in ppm
    std::vector<double> sigmax;     //Gaussian peak shape parameter. IMPORTANT: in Gaussian fit, this is actually 2*sigma*sigma
    std::vector<double> gammax;     //Lorentzian peak shape parameter
    std::vector<double> volume;    //peak volume
    std::vector<double> confidence; //confidence level of peak
};

struct shared_data_1d
{
  static int n_verbose; //0: minimal output, 1: normal output
  static bool b_dosy;  // true: doesy fitting, false: normal fitting
    /**
   * Z_gradient is used in pseudo 2D DOSY fitting only
  */
  static std::vector<double> z_gradients;
};


namespace ldw_math_spectrum_1d
{
    bool SplitFilename(const std::string &str, std::string &path_name, std::string &file_name, std::string &file_name_ext);
};

class apodization{

private:
    FID_APODIZATION_TYPE apodization_type;
    float p1;
    float p2;
    float p3;
    float p4;
    float p5;
    float p6;

    double spectral_width; //spectral width in Hz


    std::vector<float> apodization_values;

public:
    apodization();
    apodization(std::string);
    apodization(FID_APODIZATION_TYPE apodization_type, double p1,double p2,double p3,double p4,double p5,double p6);
    ~apodization();

    /**
     * Requied for EM and GM, and non-zero elb in SP
    */
    inline bool set_sw(double sw){
        spectral_width=sw;
        return true;
    }; 

    bool set_n(int); //set ndata
    bool run_apodization(float *,int,bool b_complex=true) const;

    inline bool set_first_point(float v){
        /**
         * @brief set_first_point: set the first point of the window function
         * return false if the window function is not defined yet
         * (this function should be called after set_n)
        */
        if(apodization_values.size()==0){
            std::cerr<<"Error: window function is not defined yet. Please call set_n first."<<std::endl;
            return true;
        }
        /**
         * @brief set_first_point: set the first point of the window function
         * then return true
        */
        apodization_values[0]=v;
        return true;
    };
    /**
     * return a read only copy of apodization_values
    */
    inline std::vector<float> get_apodization_values() const{
        return apodization_values;
    };

};


class fid_base
{
protected:
    /**
     * @brief read_jcamp: read jcamp file and save the data to a udict
    */
    bool read_jcamp(std::string file_name, std::map<std::string, std::string> &udict) const;
    bool read_jcmap_line(std::ifstream &,std::string line, std::string &key, std::string &value) const; 

    /**
     * Helper function for run_fft_and_rm_bruker_filter
     * @param grpdly: group delay
     * @param spectrum_real: real part of spectrum
     * @param spectrum_imag: imaginary part of spectrum
     * @return true if successful
     * This function will change the value of the following variables:
     * spectrum_real
     * spectrum_imag
    */
    bool remove_bruker_digitizer_filter(double grpdly, std::vector<float> &s_real, std::vector<float> &s_imag) const;


public :
    fid_base();
    ~fid_base();
};


class fid_1d : public fid_base, public shared_data_1d
{
protected:

    /**
     * Note: Bruker may save FID as int32 or float32
    */
    FID_DATA_TYPE data_type; 
    FID_DATA_COMPLEXITY data_complexity;

    std::vector<float> fid_data_float; //64 bit
    

    std::map<std::string, std::string> udict_acqus;
    std::map<std::string, std::string> udict_procs;

    /**
     * These two vectors are used to save the frequency domain data (fft result)
    */
    std::vector<float> spectrum_real;
    std::vector<float> spectrum_imag;
    std::vector<float> nmrpipe_header_data;  //nmrPipe header data 512*4

    /**
     * The following variables are read from acqus file
    */
    double spectral_width; //spectral width
    double observed_frequency; //observed frequency
    double carrier_frequency; //carrier frequency
    double origin;
    double receiver_gain; //receiver gain. pipe doesn't use this value

    apodization *apod;

    /**
     * @ndata_bruker: number of data points in the fid file, using Bruker convention. 
     * @ndata: number of data points in the fid file, using our convention.
     * @ndata_original: number of data points in the fid file, using our convention, before padding zeros
     * ndata(and ndata_bruker) is used to read the fid file
     * while ndata_original is used to define the size of apo window function
     * For complex data, ndata_bruker=ndata*2
     * For real data, ndata_bruker=ndata
    */
    int ndata_bruker; 
    int ndata; 
    int ndata_original;
    int nspectra; //number of spectra (in pseudo 2D NMR, nspectra>1, in 1D NMR, nspectra=1)

    double grpdly; //group delay for Bruker digitizer filter

    /**
     * ndata must be a power of 2. We define ndata_power_of_2 as the smallest power of 2 that is larger than or equal to ndata.
    */
    int ndata_power_of_2;

    int zf; //zero filling factor: 2,4,8, etc. 
    int ndata_frq; //ndata_frq=ndata_power_of_2*zf=spectrum_real.size()

    /**
     * @brief read_acqus: save data to a dictionary for nmrPipe
    */
    bool create_nmrpipe_dictionary(bool b_frq,std::map<std::string, std::string> &nmrpipe_dict_string,std::map<std::string, float> &nmrpipe_dict_float) const;

    double user_scale,user_scale2;
    double noise_level;
    int mod_selection;

    std::vector<float> baseline;

    std::vector<int> signa_boudaries;
    std::vector<int> noise_boudaries; // peak partition,used by both picker and fitter

    double begin1, step1, stop1; // ppm information
    std::string input_spectrum_fname;

    bool read_spectrum_ldw(std::string);
    bool read_spectrum_txt(std::string);
    bool read_spectrum_ft(std::string);
    bool read_spectrum_json(std::string);
    bool read_spectrum_csv(std::string);

    bool write_spectrum_json(std::string); // save spectrum
    bool write_spectrum_txt(std::string);  // write spectrum in txt format, header is copied from reading but spectrum might be changed

public:
    fid_1d();
    ~fid_1d();

    bool set_up_apodization(apodization *apod_);


    /**
     * @brief read_bruker_folder:  read fid and parameter files from Bruker folder
     * This function will find the two files: acqus and fid, and call read_bruker_acqus_and_fid to read them
     * this function will change the value of the following variables:
     * fid_data_float or fid_data_int
     * udict
    */
    bool read_bruker_folder(std::string folder_name); 

    /**
     * @brief read_bruker_acqus_and_fid: read two files: acqus and fid. 
     * We provide both read_bruker_folder and read_bruker_acqus_and_fid for convenience.
     * this function will change the value of the following variables:
     * fid_data_float or fid_data_int
     * udict_acqus
    */
    bool read_bruker_acqus_and_fid(const std::string &acqus_file_name, const std::vector<std::string> &fid_data_file_names);

    /**
     * @brief run_fft_and_rm_bruker_filter: run fft on the fid data.
    */
    bool run_fft_and_rm_bruker_filter();

    /**
     * @brief run_zf. Set a new zero filling factor only. 
    */
    bool run_zf(int new_zf);



    /**
     * @brief write_nmrpipe_fid: write the fid data to a file (in nmrPipe format)
    */
    bool write_nmrpipe_fid(const std::string outfname) const;

    /**
     * @brief write_nmrpipe_ft1: create nmrpipe file header then write the frequency domain data to a file (in nmrPipe format)
    */
    bool write_nmrpipe_ft1(std::string file_name);

    /**
     * @brief write_json: write some useful information to a json file
    */
    bool write_json(std::string file_name);

    /**
     * get header and the frequency domain data, for fid_1d to use
    */
    std::vector<float> get_spectrum_header(void) const;
    std::vector<float> get_spectrum_real(void) const;
    std::vector<float> get_spectrum_imag(void) const;

        bool est_noise_level_mad(); //for phased, baseline corrected spectrum only

    bool est_noise_level(); //general purpose noise estimation
 
    bool init(double,double,double);

    /**
     * read frq domain spectrum from a file
    */
    bool read_spectrum(std::string,bool b_negative=true);

    /**
     * read frq domain spectrum from three vectors (buffers) in pipe format
    */
    bool direct_set_spectrum_from_nmrpipe(const std::vector<float> &header, const std::vector<float> &real, const std::vector<float> &imag);
    /**
     * Like a simple reading from text file with only real data and ppm information
    */
    bool set_spectrum_from_data(const std::vector<float> &data, const double begin, const double step, const double stop);
    bool release_spectrum(); //to save memory after peaks picking.
    bool write_spectrum(std::string); //write spectrum. Format is determined by file extension.
    const std::vector<float> & get_spectrum() const;
};

#endif