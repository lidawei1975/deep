#include <vector>
#include <array>
#include <deque>
#include <string>

#include "spectrum_io_3d.h"
#include "spectrum_pick.h"
#include "spectrum_pick_1d.h"

#ifndef SPECTRUM_PICK_3D_H
#define SPECTRUM_PICK_3D_H

class spectrum_pick_3d: public spectrum_io_3d
{
private:
    std::vector<class spectrum_pick> spectra_2d_for_picking;
    std::vector<class spectrum_1d_peaks> spectra_1d;

    //these 5 have the same size and 1-1 correspondance
    std::vector< std::vector<int> > z_triangles11,z_triangles12,z_triangles21,z_triangles22;
    std::vector< std::vector<int> > z_triangles_sign; //1 for positive, -1 for negative

    //These 3 have the same size and 1-1 correspondance
    std::vector< std::vector<int> > z_line1,z_line2;
    std::vector<std::vector<int>> z_line_sign; //1 for positive, -1 for negative

   
    //for debug
    std::vector< std::vector<double> > for_debug;

    //for special case, 
    std::vector<int> peak_tilt;
    bool cut_one_peak(std::vector<double> target_line_x,std::vector<double> target_line_y,std::vector<double> target_line_z,int current_pos,std::vector<int> ndx_neighbors, int anchor_pos,int &pos_start,int &pos_end);
    bool find_nearest_normal_peak(std::array<double,3> x,std::vector<int> ndxs, int p);

    //functions
    bool get_triangles(int,int);
    bool get_lines(int);
    bool interp3(std::vector<double> line_x, std::vector<double> line_y,std::vector<double> line_z, std::vector<double> &line_v);
    double interp2(int min_x, int max_x, int min_y,int max_y, std::vector<double> data,double x,double y);
    // bool interp2(int min_x, int max_x, int min_y,int max_y, std::vector<double> data,std::vector<double> x,std::vector<double> y, std::vector<double> &line_v);
    bool get_line_polygon_intersection(int,int,int,int,int,int,int,int,double IntersectionPoint[3]);
    bool RayIntersectsTriangle(double rayOrigin[3],double rayVector[3],double vertex0[3],double vertex1[3],double vertex2[3],double outIntersectionPoint[3]);


public:
    spectrum_pick_3d();
    ~spectrum_pick_3d();
    bool read_for_picking(std::string fname1, std::string fname2);
    bool peak_picking();
    bool simple_peak_picking();
    bool special_case_peaks();
    bool print_peaks(std::string outfname);
};

#endif