#include <vector>
#include <set>
#include <deque>
#include <string>

#ifndef SPECT_IO_H
#define SPECT_IO_H

namespace ldw_math_spectrum_2d
{
    bool SplitFilename (const std::string& str, std::string &path_name, std::string &file_name, std::string &file_name_ext);
    double calcualte_median(std::vector<double> scores);
    void sortArr(std::vector<double> &arr, std::vector<int> &ndx);
    bool spline_expand(int xdim, int ydim, float *spect,std::vector<double> &final_data);
};


class spectrum_io
{
public:
    double begin3,stop3,step3;  //3rd (indirect) dimension, in case of 3D

protected:
    //spectrum dimension
    int xdim,ydim;
    //sepctrum range
    double begin1,step1,stop1;  //direct dimension
    double begin2,step2,stop2;  //indirect dimension
    /**
     * @brief spectrum data, after zero filling and fft
     * Row major, xdim*ydim size, access as spect[j*xdim+i]
    */
    float * spect;  

    /**
     * When "-di" is NOT included in nmrpipe processing, we have imaginary data in the spectrum too
     * All are row major and have xdim*ydim size
     * Suppose we need to run phase correction. 
     * Firstly we use spect and spect_imagine_direct to do phase correction along direct dimension ==> new spect
     * We also apply the same phase correction to spect_imagine_indirect and spect_imagine_imagine ==> new spect_imagine_indirect
     * We then do indirect dimension phase correction using new spect and new spect_imagine_indirect ==> final spect
    */
    float * spectrum_real_real; //direct dimension, real part === spect
    float * spectrum_real_imag; //direct dimension, imaginary part
    float * spectrum_imag_real; //indirect dimension, real part
    float * spectrum_imag_imag; //indirect dimension, imaginary part

    float indirect_p0,indirect_p1; //phase correction for indirect dimension, provided by standard pulse program

    int data_type_indirect;  //0 means complex, 1 means real
    int data_type_direct; //0 means complex, 1 means real

    float noise_level;  //spectrum noise level, 1.4826*medium(abs(spect))
    std::vector<double> noise_level_columns; //noise level of each column
    std::vector<double> noise_level_rows; //noise level of each row

    double SW1,SW2,frq1,frq2,ref1,ref2; //ppm information
    double SW3,frq3,ref3; //ppm information in case of 3D
    
    void estimate_noise_level_mad();   //estimate noise level, using MAD method 1.4826*medium(abs(spect))
    void estimate_noise_level(); //estimate noise level, using region by region standard deviation

private:
    bool b_pipe;
    float header[512];  //nmrpipe format 

    

    //save original spectrum after we do invers , zero filling and fft
    float * spect_ori;
    

protected:

    int xdim_ori,ydim_ori; //original spectrum dimension (before zero filling and fft)

    //These three are for endian conversion, required by sparky format!
    float read_float(FILE *);
    bool read_float(FILE *,int, float *);
    int read_int(FILE *);

    bool read_pipe(std::string fname); //nmrpipe
    bool read_topspin_txt(std::string fname); //topspin totxt format
    bool read_txt(std::string fname);  //from matlab save -ASCII
    bool read_sparky(std::string fname);  //ucsf sparky
    bool read_mnova(std::string fname); //mnova csv file format

    
    std::string infname; //file name
    double user_scale,user_scale2; //minimal peak intesntiy in picking and fitting
    double median_width_x,median_width_y; //peak width median from either picking or fitting

    //peaks, used by both picking and fitting classes
    std::vector<int> peak_index; //used when read in peak list to keep track of all peaks.
public:
    std::vector<double> p1,p2,p_intensity;  //grid peak position, peak amp 
    std::vector<double> p1_ppm,p2_ppm; //ppm value of peak pos, obtained from p1 and p2
    std::vector<double> p_confidencex,p_confidencey; //confidence level of peaks
    std::vector<std::string> user_comments; //user comment of about a peak!
    std::vector<double> sigmax,sigmay;  //for Gaussian and Voigt fitting, est in picking too
    std::vector<double> gammax,gammay; //for voigt fittting and ann picking, set to 0 in picking!
 protected:   
    bool get_ppm_from_point();
    /**
     * For internal use only. Write spectrum to a file
    */
    bool write_pipe(std::vector<std::vector<float> > spect, std::string fname); 

public:
    spectrum_io();
    ~spectrum_io();

    bool init(std::string, int noise_flag=1);  //read spectrum and est noise
    bool read_spectrum(std::string); //read spectrum only
    
    /**
     * Public function to write spectrum to a file.
     * b_real_only means we will remove all imaginary part of the spectrum
    */
    bool write_pipe(std::string fname, bool b_real_only = false);
    bool save_mnova(std::string fname);

    
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

    inline void get_dim(int *n1, int *n2){*n1=xdim;*n2=ydim;};

    inline void set_noise_level(double t) {noise_level=t; std::cout<<"Direct set noise level to "<<t<<std::endl;}

    inline void set_scale(double x,double y)
    {
        user_scale=x;
        user_scale2=y;
    };

    inline void get_median_width(double *x, double *y)
    {
        *x=median_width_x;
        *y=median_width_y;
    };
    
};

#endif
