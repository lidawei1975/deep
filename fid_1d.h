
#ifndef FID_1D_H
#define FID_1D_H

#include <vector>
#include <map>

enum FID_DATA_COMPLEXITY
{
    FID_DATA_COMPLEXITY_REAL = 0,
    FID_DATA_COMPLEXITY_COMPLEX = 1
};


enum FID_DATA_TYPE
{
    FID_DATA_TYPE_FLOAT64 = 0,
    FID_DATA_TYPE_INT32 = 1
};

enum FID_APODIZATION_TYPE
{
    FID_APODIZATION_KAISER = 0    
};


namespace nmrPipe
{
    bool nmrpipe_header_to_dictionary(float * nmrpipe_header_data,std::map<std::string,std::string> & dict_string,std::map<std::string, float> & dict_float);
    bool nmrpipe_dictionary_to_header(float * nmrpipe_header_data,const std::map<std::string,std::string> &dict_string,const std::map<std::string, float> & dict_float);
    bool create_empty_nmrpipe_header(std::map<std::string,std::string> & dict_string,std::map<std::string, float> &dict_float);
};

namespace fid_1d_helper
{
    size_t split(const std::string &txt, std::vector<std::string> &strs, char ch);
};


class fid_1d
{
protected:

    /**
     * Note: Bruker may save FID as int32 or float32
    */
    FID_DATA_TYPE data_type; 
    FID_DATA_COMPLEXITY data_complexity;

    std::vector<double> fid_data_double; //64 bit
    

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
    double receiver_gain; //receiver gain. pipe doesn't use this value

    

    /**
     * ndata_bruker: number of data points in the fid file, using Bruker convention. 
     * ndata: number of data points in the fid file, using our convention.
     * For complex data, ndata_bruker=ndata*2
     * For real data, ndata_bruker=ndata
    */
    int ndata_bruker; 
    int ndata; 

    /**
     * ndata must be a power of 2. We define ndata_power_of_2 as the smallest power of 2 that is larger than or equal to ndata.
    */
    int ndata_power_of_2;

    int zf; //zero filling factor: 2,4,8, etc. 
    int ndata_frq; //ndata_frq=ndata_power_of_2*zf=spectrum_real.size()

    /**
     * @brief read_jcamp: read jcamp file and save the data to a udict
    */
    bool read_jcamp(std::string file_name, std::map<std::string, std::string> &udict) const;
    bool read_jcmap_line(std::ifstream &,std::string line, std::string &key, std::string &value) const;

    /**
     * @brief read_acqus: save data to a dictionary for nmrPipe
    */
    bool create_nmrpipe_dictionary(bool b_frq,std::map<std::string, std::string> &nmrpipe_dict_string,std::map<std::string, float> &nmrpipe_dict_float) const;

    /**
     * @brief remove_bruker_digitizer_filter: remove the Bruker digitizer filter
     * Currently, this function only works for frequency domain data for Bruker Avance spectrometers.
    */
    bool remove_bruker_digitizer_filter();

public:
    fid_1d();
    ~fid_1d();


    /**
     * @brief read_bruker_folder:  read fid and parameter files from Bruker folder
     * This function will find the two files: acqus and fid, and call read_bruker_acqus_and_fid to read them
     * this function will change the value of the following variables:
     * fid_data_double or fid_data_int
     * udict
    */
    bool read_bruker_folder(std::string folder_name); 

    /**
     * @brief read_bruker_acqus_and_fid: read two files: acqus and fid. 
     * We provide both read_bruker_folder and read_bruker_acqus_and_fid for convenience.
     * this function will change the value of the following variables:
     * fid_data_double or fid_data_int
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
     * @brief run_apodization: run apodization on the fid data.
     * @param apodization_type: type of apodization
     * @param p1-p6: parameters for apodization.
     * Not all apodization types need 6 parameters.
    */
    bool run_apodization(FID_APODIZATION_TYPE apodization_type, double p1,double p2=0.0,double p3=0.0,double p4=0.0,double p5=0.0,double p6=0.0);


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
     * get header and the frequency domain data, for spectrum_io_1d to use
    */
    std::vector<float> get_spectrum_header(void) const;
    std::vector<float> get_spectrum_real(void) const;
    std::vector<float> get_spectrum_imag(void) const;
};

#endif