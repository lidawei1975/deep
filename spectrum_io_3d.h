#include <vector>
#include <array>
#include <deque>
#include <string>

#include "spectrum_io.h"
#include "spectrum_io_1d.h"

#ifndef SPECTRUM_TYPE
#define SPECTRUM_TYPE
enum spectrum_type 
{
    null_spectrum,
    hsqc_spectrum,
    tocsy_spectrum
};
#endif

#ifndef SPECTRUM_IO_3D_H
#define SPECTRUM_IO_3D_H

namespace ldw_math_3d
{
    double calcualte_median(std::vector<double> scores);
 
    bool voigt_convolution(double a, double x, double y, double z, double sigmax, double sigmay, double sigmaz, double gammax, double gammay, double gammaz, int xdim, int ydim,int zdim, std::vector<float> &kernel, int &i0, int &i1, int &j0, int &j1, int &k0, int &k1,std::array<int,6> region,double scale);
    bool gaussian_convolution(double a, double x, double y, double z, double sigmax, double sigmay, double sigmaz, int xdim, int ydim,int zdim, std::vector<float> &kernel, int &i0, int &i1, int &j0, int &j1, int &k0, int &k1,std::array<int,6> region, double scale);
    bool voigt_convolution_region(double x, double y, double z, double sigmax, double sigmay, double sigmaz, double gammax, double gammay, double gammaz, int xdim, int ydim,int zdim, int &i0, int &i1, int &j0, int &j1, int &k0, int &k1);
    bool gaussian_convolution_region(double x, double y, double z, double sigmax, double sigmay, double sigmaz, int xdim, int ydim,int zdim, int &i0, int &i1, int &j0, int &j1, int &k0, int &k1);



    bool calcualte_principal_axis(std::vector<float> data, int xdim, int ydim, int zdim);
    double interp2_point(int min_x, int max_x, int min_y,int max_y, std::vector<double> data, double x,double y);
    bool interp2(int min_x, int max_x, int min_y,int max_y, std::vector<double> data, std::vector<double> x,std::vector<double> y,std::vector<double> &line_v);
    std::vector<int> find_neighboring_peaks(std::vector< std::array<double,3> > peaks_3d_pos,std::vector<double> inten,int p); 
    std::vector<std::deque<int> > bread_first(std::vector<int> &neighbor, int n);
    int find_best_from_peaks(std::vector< std::array<double,3> >, std::vector< std::array<double,3> > x2, std::vector<int> &ndxs);
    bool get_perpendicular_line(std::array<double,3> x,std::array<double,3> x0, 
            std::vector<double> &line_x1, std::vector<double> &line_y1, std::vector<double> &line_z1,
            std::vector<double> &line_x2, std::vector<double> &line_y2, std::vector<double> &line_z2,double step);
    
};


class spectrum_io_3d
{
protected:
    int class_flag; //io: 0, derived: 1 or 2

    bool b_negative; //flag to pick negative peaks too
    int mod_selection;
    int zf;
    double user_scale1,user_scale2;
    enum spectrum_type spectrum_type; //0: unknonw, 1: hsqc, 2:tocsy
    int xdim,ydim,zdim; 
    double begin[3],step[3],stop[3];
    float noise_level;
    std::vector<class spectrum_io> spectra_2d;

    //3d peaks, fitted also need to read them
    std::vector< std::array<double,3> > peaks_pos;
    std::vector< std::array<double,3> > sigma,gamma;
    std::vector<double> intensity;
    std::vector<int> peak_cannot_move_flag;
    

    bool read_pipe(std::string fname); //read pipe format files
   
public:
    spectrum_io_3d();
    ~spectrum_io_3d();
    bool init_parameters(double,double,double,int m=1,int z=0, bool b=false); //user_scale1,user_scale2,noise_level,mod_selection,zf,b_negative
    
    bool peak_reading(std::string infname);
};

#endif