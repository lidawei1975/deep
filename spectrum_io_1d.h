
#ifndef SPEC_IO_1D_H
#define SPEC_IO_1D_H

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

namespace ldw_math_spectrum_1d
{
    bool SplitFilename(const std::string &str, std::string &path_name, std::string &file_name, std::string &file_name_ext);
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


class spectrum_io_1d : public shared_data_1d
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
    // int& xdim=ndata;

    /**
     * frq1: unit is Hz. so called observed frequency
    */

    double SW1,frq1,ref1; //spectrum information
    double begin1,step1,stop1; //ppm information

    std::string input_spectrum_fname;

    bool read_spectrum_ldw(std::string);
    bool read_spectrum_txt(std::string);
    bool read_spectrum_ft(std::string);
    bool read_spectrum_json(std::string);
    bool read_spectrum_csv(std::string);

    bool write_spectrum_json(std::string);  //save spectrum 
    bool write_spectrum_ft1(std::string); //write spectrum in nmrpipe format, header is copied from reading but spectrum might be changed
    bool write_spectrum_txt(std::string); //write spectrum in txt format, header is copied from reading but spectrum might be changed

public:

    spectrum_io_1d();
    ~spectrum_io_1d();
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
    bool write_json(std::string fname);
    const std::vector<float> & get_spectrum() const;
};

#endif