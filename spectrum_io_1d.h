
#ifndef SPEC_IO_1D_H
#define SPEC_IO_1D_H

namespace ldw_math_spectrum_1d
{
    bool SplitFilename(const std::string &str, std::string &path_name, std::string &file_name, std::string &file_name_ext);
};


class spectrum_io_1d
{
protected:
    bool b_header;
    float header[512];  //nmrpipe format 
    double user_scale,user_scale2;
    double noise_level;
    int mod_selection;
    std::vector<float> spect;  //1d spectrum
    std::vector<float> spe_image;  //1d spectrum, imaginary part
    
    std::vector<float> baseline; 
   
    std::vector<int> signa_boudaries;
    std::vector<int> noise_boudaries; //peak partition,used by both picker and fitter

    int ndata; //size of the spectrum
    int& xdim=ndata;
    double SW1,frq1,ref1; //spectrum information
    double begin1,step1,stop1; //ppm information

    std::string input_spectrum_fname;

    bool read_spectrum_ldw(std::string);
    bool read_spectrum_txt(std::string);
    bool read_spectrum_ft(std::string);
    bool read_spectrum_json(std::string);
    bool read_spectrum_json_format1(Json::Value &root);
    
    

public:
    spectrum_io_1d();
    ~spectrum_io_1d();
    bool est_noise_level_mad(); //for phased, baseline corrected spectrum only

    bool est_noise_level(); //general purpose noise estimation
 
    bool init(double,double,double);
    bool read_spectrum(std::string,bool b_negative=true);
    bool direct_set_spectrum(std::vector<float> &);
    bool stride_spectrum(int);
    bool release_spectrum(); //to save memory after peaks picking.
    bool save_experimental_spectrum(std::string);  //save spectrum in json format
    bool write_spectrum(std::string); //write spectrum in nmrpipe format, header is copied from reading!!

    const std::vector<float> & get_spectrum() const;
};

#endif