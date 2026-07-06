#ifndef FID_1D_H
#define FID_1D_H

#include "fid_base.h"

class fid_1d : public fid_base, public shared_data_1d
{
protected:

    bool b_read_from_ft1; // true if the spectrum data is read from ft1 file, false if processed from fid data

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
    bool write_spectrum_csv(std::string); // write spectrum in csv format, header is copied from reading but spectrum might be changed
    bool read_spectrum_sparky(std::string infname);

    bool process_nmrpipe_header();

    bool process_dictionary();
    
public:
    fid_1d();
    ~fid_1d();

    bool set_up_apodization(apodization *apod_);
    bool set_up_apodization_from_string(const std::string &apod_string);

    /**
     * Reduce fid size to new_size, in case n_acqisition_size is larger than needed (collecting noise)
     * new_size must be smaller than current ndata_bruker
     */
    bool reduce_fid_size(int new_size); 

    /**
     * @brief detect_signal_end_location: Estimate the cutoff location for the FID signal (where signal ends and noise dominates).
     * Follows logic similar to example.js.
     * @return int: The index where the signal is estimated to end. Returns ndata if not detected or signal covers entire range.
     */
    int detect_signal_end_location() const; 


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
     * @brief read_magritek_folder: read acqu.par and data.1d files from Magritek folder
    */
    bool read_magritek_folder(std::string folder_name);

    /**
     * @brief read_magritek_acqu_and_fid: read acqu.par and data.1d files
    */
    bool read_magritek_acqu_and_fid(const std::string &acqu_file_name, const std::vector<std::string> &fid_data_file_names);

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
    std::string write_json_as_string();

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


    /**
     * Below functions are mainly used for web assembly version to read froma and write to buffer
     * that is accessible from JavaScript.
     */
    bool read_first_spectrum_from_buffer(std::vector<float> &header,
                                          std::vector<float> &spectrum_real_,
                                          std::vector<float> &spectrum_imag_);

    /**
     * Read only access of nspectra
    */
    inline int get_nspectra() const
    {
        return nspectra;
    };

    /**
     * read only access of ndata_frq and ndata_frq_indirect
     */
    inline int get_ndata_frq() const
    {
        return ndata_frq;
    };

    /**
     * These two functions are for web assembly binding.
     * They work together to read acqus file and fid data, equivalent to read_bruker_acqus_and_fid()
     */
    bool read_bruker_files_as_strings(const std::string &contents_acqus);
    bool set_fid_data(const std::vector<float> &fid_data_float_);

#ifdef WEBASSEMBLY
    uintptr_t get_data_of_header();
    uintptr_t get_data_of_real();
    uintptr_t get_data_of_imag();
    int get_fid_data_type() const{
        std::cout<<"udict_acqus.at('DTYPA') = "<<udict_acqus.at("DTYPA")<<std::endl;
        return std::stoi(udict_acqus.at("DTYPA")); //2: double (float64) or 0: int32
    }
#endif
};


#endif
