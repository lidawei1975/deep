#include <vector>
#include <set>
#include <deque>
#include <string>




#ifndef SPECTRUM_TYPE
#define SPECTRUM_TYPE
enum spectrum_type 
{
    null_spectrum,
    hsqc_spectrum,
    tocsy_spectrum
};
#endif

class spectrum_io
{

protected:
    enum spectrum_type spectrum_type; //0: unknonw, 1: hsqc, 2:tocsy
    //spectrum dimension
    int xdim,ydim;
    //sepctrum range
    double begin1,step1,stop1;
    double begin2,step2,stop2;
    float * spect;  //freq data
    float noise_level;  //spectrum noise level, 1.4826*medium(abs(spect))
    std::vector<double> noise_level_columns; //noise level of each column
    std::vector<double> noise_level_rows; //noise level of each row
    
    void noise();   //estimate noise level

private:
    int indirect_ndx; //which dimension is the indirect dimension, 1 or 3
    float header[512];  //nmrpipe format 
    double SW1,SW2,frq1,frq2,ref1,ref2; //ppm information

    //save original spectrum after we do invers , zero filling and fft
    float * spect_ori;
    int xdim_ori,ydim_ori;

protected:
    //These three are for endian conversion, required by sparky format!
    float read_float(FILE *);
    bool read_float(FILE *,int, float *);
    int read_int(FILE *);

    bool read_pipe(std::string fname); //nmrpipe
    bool read_topspin_txt(std::string fname); //topspin totxt format
    bool read_txt(std::string fname);  //from matlab save -ASCII
    bool read_sparky(std::string fname);  //ucsf sparky
    bool read_mnova(std::string fname); //mnova csv file format
    
public:
    spectrum_io();
    ~spectrum_io();

    bool init(std::string fname);  ////read spectrum and zero filling, noise est
    bool read(std::string infname); //read spectrum
    bool write_pipe(std::vector<std::vector<float> > spect, std::string fname);
    bool save_mnova(std::string fname);
    
    inline float * get_spect_data() {return spect; };
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
    
};

