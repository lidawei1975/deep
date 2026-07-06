
#ifndef FID_BASE_H
#define FID_BASE_H

#include <vector>
#include <map>
#include <string>
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
    static int n_verbose; // 0: minimal output, 1: normal output
    static bool b_dosy;   // true: doesy fitting, false: normal fitting
                          /**
                           * Z_gradient is used in pseudo 2D DOSY fitting only
                           */
    static std::vector<double> z_gradients;
    /**
     * Peak combine cutoff
     */
    static double peak_combine_cutoff;
    /**
     * Remove failed peaks in peak fitting from output, default is false
     * If not removed, failed peaks will be set to have 0 intensity in the output spectrum
     */
    static bool b_remove_failed_peaks;
};

namespace ldw_math_spectrum_1d
{
    bool SplitFilename(const std::string &str, std::string &path_name, std::string &file_name, std::string &file_name_ext);
    bool trim(std::string &str);
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
    bool read_jcmap_line(std::istream &,std::string line, std::string &key, std::string &value) const; 
    bool process_jcamp_as_string(const std::string &contents, std::map<std::string, std::string> &udict) const;

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
    //These three are for endian conversion, required by sparky format!
    float read_float(FILE *);
    bool read_float(FILE *,int, float *);
    int read_int(FILE *);


public :
    fid_base();
    ~fid_base();
};



#endif
