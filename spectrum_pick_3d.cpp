//#include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <valarray>
#include <string>
#include <cstring>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "spline.h"
#include "json/json.h"
#include "commandline.h"
#include "dnn_picker.h"
#include "spectrum_pick_3d.h"


extern "C"  
{
    double voigt(double x, double sigma, double gamma);
};
    template <class myType>
    void sortArr(std::vector<myType> &arr, std::vector<int> &ndx) 
    { 
        std::vector<std::pair<myType, int> > vp; 
    
        for (int i = 0; i < arr.size(); ++i) { 
            vp.push_back(std::make_pair(arr[i], i)); 
        } 
    
        std::sort(vp.begin(), vp.end()); 
    
        for (int i = 0; i < vp.size(); i++)
        { 
            ndx.push_back(vp[i].second);
        } 
    };


//
spectrum_pick_3d::spectrum_pick_3d()
{
    mod_selection=1;   
    class_flag=1; //pick_3d
};

spectrum_pick_3d::~spectrum_pick_3d()
{
   
};

bool spectrum_pick_3d::read_for_picking(std::string fname1, std::string fname2)
{
    std::string sft2(".ft3");

    //make sure both filenames are ended with .ft3
    if(!std::equal(sft2.rbegin(), sft2.rend(), fname1.rbegin()) || !std::equal(sft2.rbegin(), sft2.rend(), fname2.rbegin()) )
    {
        std::cout<<"both filenames must end with .ft3"<<std::endl;
        return false;
    }

    //remove .ft3 from both file names
    fname1=fname1.substr(0,fname1.length()-4);
    fname2=fname2.substr(0,fname2.length()-4);
    
    std::size_t found1 = fname1.find_first_of("0123456789");
    std::size_t found2 = fname2.find_first_of("0123456789");

    std::string basename1=fname1.substr(0,found1);
    std::string basename2=fname2.substr(0,found2);

    int n_zero=fname1.length()-basename1.length();
    
    if(basename1!=basename2)
    {
        std::cout<<"The two filenames have different base."<<std::endl;
        return false;
    }

    std::string num1=fname1.substr(found1);
    std::string num2=fname2.substr(found2);

    int n=num1.length();
    int n1=atoi(num1.c_str());
    int n2=atoi(num2.c_str());

    for(int i=n1;i<=n2;i++)
    {
        std::string new_string;
        std::string old_string=std::to_string(i);
        int nt=n_zero - old_string.length();
        if(nt>0)
        {
            new_string = std::string(nt, '0') + old_string;
        }
        else
        {
            new_string = old_string;
        }


        std::string current_fname=basename1+new_string+".ft3";
        std::cout<<"Read in "<<current_fname<<std::endl;

        spectrum_pick f;
        // if(b_negative==true)   f.set_negative_mode();
        if((i-n1)%100==0) f.init(current_fname,1);
        else f.init(current_fname,0);
        spectra_2d_for_picking.emplace_back(f);

        if(i==n1)
        {
            f.get_ppm_infor(begin,step);
            f.get_dim(&xdim,&ydim);
            begin[2]=f.begin3;
            stop[2]=f.stop3;
            std::cout<<"3rd dimension is from "<<begin[2]<<" to "<<stop[2]<<std::endl;
        }

        std::cout<<std::endl<<std::endl;
    }

    zdim=n2-n1+1;
    step[2]=(stop[2]-begin[2])/zdim;
    begin[2]+=step[2]; //to be consistent with pipe.

   
    std::cout << "Direct dimension size is " << xdim << " indirect dimension is " << ydim << " and " << zdim << std::endl;
    std::cout << "  Direct dimension   offset is " << begin[0] << ", ppm per step is " << step[0] << " ppm and end is " <<begin[0]+step[0]*xdim<< std::endl;
    std::cout << "Indirect dimension 1 offset is " << begin[1] << ", ppm per step is " << step[1] << " ppm and end is " <<begin[1]+step[1]*ydim<< std::endl;
    std::cout << "Indirect dimension 2 offset is " << begin[2] << ", ppm per step is " << step[2] << " ppm and end is "<<begin[2]+step[2]*zdim << std::endl;

    //define median noise of all planes as 3D spectral noise level 
    if(noise_level<0.00001)
    {
        std::vector<float> scores;
        for(int i=0;i<spectra_2d_for_picking.size();i++)
        {   
            if(i%100==0)
            {
                scores.push_back(spectra_2d_for_picking[i].get_noise_level());
            }
        }
        sort(scores.begin(), scores.end());
        noise_level = scores[scores.size() / 2];
        std::cout<<"Estimated noise level is "<<noise_level<<std::endl;
    }
    else
    {
        std::cout<<"Set noise level directly to "<<noise_level<<std::endl;
    }

    return true;
};


bool spectrum_pick_3d::special_case_peaks()
{
    int npeak=intensity.size();
    peak_tilt.clear();

    std::vector<float> sum_of_all_peaks;
    sum_of_all_peaks.resize(xdim * ydim, 0.0f);

    std::vector<std::vector<float> > spectrum_of_peaks;
    spectrum_of_peaks.resize(intensity.size());
    int i0, i1, j0, j1, k0, k1;

    // for (int i = 0; i < npeak; i++)
    // {
    //     ldw_math_3d::voigt_convolution(intensity[i],peaks_3d_pos[i][0],peaks_3d_pos[i][1],peaks_3d_pos[i][2],
    //                                     sigma[i][0],sigma[i][0],sigma[i][0],
    //                                     gamma[i][0],gamma[i][0],gamma[i][0],
    //                                     xdim, ydim,zdim,
    //                                     spectrum_of_peaks[i], i0, i1, j0, j1, k0, k1);
    //     for (int kk = k0; kk < k1; kk++)
    //         for (int ii = i0; ii < i1; ii++)
    //             for (int jj = j0; jj < j1; jj++)
    //             {
    //                 sum_of_all_peaks[kk * xdim * ydim + ii * ydim + jj] += spectrum_of_peaks[i][(kk - k0) * (i1 - i0) * (j1 - j0) + (ii - i0) * (j1 - j0) + jj - j0];
    //             }
    // }

    // for (int i = 0; i < npeak ; i++)
    // {
    //     ldw_math_3d::voigt_convolution_region(peaks_3d_pos[i][0],peaks_3d_pos[i][1],peaks_3d_pos[i][2],
    //                                     sigma[i][0],sigma[i][0],sigma[i][0],
    //                                     gamma[i][0],gamma[i][0],gamma[i][0],
    //                                     xdim, ydim,zdim,
    //                                     i0, i1, j0, j1,k0,k1);
    //     spectrum_of_peaks[i].resize((k1 - k0) * (i1 - i0) * (j1 - j0));
    //     for (int kk = k0; kk < k1; kk++)
    //         for (int ii = i0; ii < i1; ii++)
    //             for (int jj = j0; jj < j1; jj++)
    //             {
    //                 double z1 = spectrum_of_peaks[i][(kk - k0) * (i1 - i0) * (j1 - j0) + (ii - i0) * (j1 - j0) + jj - j0];
    //                 double z2 = sum_of_all_peaks[ii * ydim + jj];
    //                 spectrum_of_peaks[i][(kk - k0) * (i1 - i0) * (j1 - j0)+(ii - i0) * (j1 - j0) + jj - j0] = z1 / z2 * spectra_2d_for_picking.at(kk).get_spect_data()[ii * ydim + jj];
    //             }
    //     std::cout << peaks_3d_pos[i][0] + 1 << " " << peaks_3d_pos[i][1] + 1  << " " << peaks_3d_pos[i][2] + 1 << std::endl;
    //     ldw_math_3d::calcualte_principal_axis(spectrum_of_peaks[i], i1 - i0, j1 - j0, k1-k0,  orientation_x[i], orientation_x2[i]);
    // }

    std::vector<int> to_remove; 
    to_remove.resize(npeak,0); //once two add peaks are added, the old peak will be labeled (=1)

    for (int i = 0; i < npeak ; i++)
    {   
        //This function will not generate spectral data but only update i0,i1,
        ldw_math_3d::voigt_convolution_region(peaks_pos[i][0],peaks_pos[i][1],peaks_pos[i][2],
                                        sigma[i][0],sigma[i][0],sigma[i][0],
                                        gamma[i][0],gamma[i][0],gamma[i][0],
                                        xdim, ydim,zdim,
                                        i0, i1, j0, j1, k0, k1);

        spectrum_of_peaks[i].resize((k1 - k0) * (i1 - i0) * (j1 - j0));

        //if the peak is a negative peak, we need to flip the sign of the spectrum
        double sign_of_peak = 1.0;
        if (intensity[i] < 0.0)
        {
            sign_of_peak = -1.0;
        }

        for (int kk = k0; kk < k1; kk++)
            for (int ii = i0; ii < i1; ii++)
                for (int jj = j0; jj < j1; jj++)
                {
                    // spectrum_of_peaks[i][(kk - k0) * (i1 - i0) * (j1 - j0)+(ii - i0) * (j1 - j0) + jj - j0] = std::max(spectra_2d_for_picking.at(kk).get_spect_data()[ii * ydim + jj],float(noise_level*user_scale2));
                    spectrum_of_peaks[i][(kk - k0) * (i1 - i0) * (j1 - j0)+(ii - i0) * (j1 - j0) + jj - j0] = sign_of_peak*spectra_2d_for_picking.at(kk).get_spect_data()[ii  + jj* xdim];
                }
        std::cout<<"Check peak "<<i<<" at "<<peaks_pos[i][0]<<" "<<peaks_pos[i][1]<<" "<<peaks_pos[i][2]<<" "<<intensity[i]<<" ";
        peak_tilt.push_back(ldw_math_3d::calcualte_principal_axis(spectrum_of_peaks[i], i1 - i0, j1 - j0, k1-k0));
    }

    //build array of all orientations that we need to check
    //first, uniform distribution in 3D space (2 DOF)
    std::vector<double> title_angle_x, title_angle_y, title_angle_z;
    title_angle_x.clear();
    title_angle_y.clear();
    title_angle_z.clear();
    for (double theta = 10.0 ; theta < 90; theta += 10.0)
    {
        double cos_theta = cos(theta / 180.0 * M_PI);
        double sin_theta = sin(theta / 180.0 * M_PI);
        double phi_step = 10.0 / sin_theta;

        for (double phi = 0; phi < 360; phi += phi_step)
        {
            title_angle_x.push_back(sin_theta * cos(phi / 180.0 * M_PI));
            title_angle_y.push_back(sin_theta * sin(phi / 180.0 * M_PI));
            title_angle_z.push_back(cos_theta);
        }
    }

    //next, remove some orientation 
    for (int j = title_angle_z.size() - 1; j >= 0; j--)
    {
        bool b1 = fabs(title_angle_z[j]) > 0.99 || fabs(title_angle_x[j]) > 0.99 || fabs(title_angle_y[j]) > 0.99; //along x,y or z
        bool b2 = title_angle_x[j] + title_angle_y[j] + title_angle_z[j] > 0.99 * sqrt(3.0);                       //along [1 1 1]
        bool b3 = title_angle_x[j] - title_angle_y[j] + title_angle_z[j] > 0.99 * sqrt(3.0);                       //along [1 -1 1]
        bool b4 = -title_angle_x[j] + title_angle_y[j] + title_angle_z[j] > 0.99 * sqrt(3.0);                      //along [-1 1 1]
        bool b5 = -title_angle_x[j] - title_angle_y[j] + title_angle_z[j] > 0.99 * sqrt(3.0);                      //along [-1 -1 1]
        bool b6 = title_angle_x[j] + title_angle_y[j] > 0.99 * sqrt(2.0);                                          //along [1 1 0]
        bool b7 = title_angle_x[j] - title_angle_y[j] > 0.99 * sqrt(2.0);                                          //along [1 -1 0]
        bool b8 = title_angle_x[j] + title_angle_z[j] > 0.99 * sqrt(2.0);                                          //along [1 0 1]
        bool b9 = title_angle_x[j] - title_angle_z[j] > 0.99 * sqrt(2.0);                                          //along [1 0 -1 ]
        bool ba = title_angle_y[j] + title_angle_z[j] > 0.99 * sqrt(2.0);                                          //along [0 1 1]
        bool bb = title_angle_y[j] - title_angle_z[j] > 0.99 * sqrt(2.0);                                          //along [0 1 -1 ]

        if (b1 || b2 || b3 || b4 || b5 || b6 || b7 || b8 || b9 || ba || bb)
        {
            title_angle_x.erase(title_angle_x.begin() + j);
            title_angle_y.erase(title_angle_y.begin() + j);
            title_angle_z.erase(title_angle_z.begin() + j);
        }
    }

    double zf_step;
    if (zf == 1)
        zf_step = 0.5;
    else
        zf_step = 1.0;

    std::ofstream fnewpeak("new_peak.txt"); 
    for (int peak_index = 0; peak_index < npeak ; peak_index++)
    {
        //get sign of peak
        int sign_of_peak = 1;
        if (intensity[peak_index] < 0)
        {
            sign_of_peak = -1;
        }

        std::cout << "Begin to check peak " << peak_index << " at coor: " << peaks_pos[peak_index][0]  << " " << peaks_pos[peak_index][1]  << " "<<peaks_pos[peak_index][2];
        std::cout << " " << intensity[peak_index];
        std::cout <<std::endl;
        if( fabs(intensity[peak_index]) < noise_level * user_scale1 ) continue; //weak peak
        if(peak_tilt[peak_index]==0) continue;   //perfect aligned peak
        // std::cout<<"done check tilt"<<std::endl;

        std::vector<int> ndx_neighbors=ldw_math_3d::find_neighboring_peaks(peaks_pos,intensity,peak_index);
        std::vector< std::array<double,3> > new_peak1_pos, new_peak2_pos;
        // std::cout<<"done find neighboring peaks"<<std::endl;

        // if (peak_index == 195)
        // {
        //     std::cout << "neighboring peaks are:" << std::endl;
        //     for (int nn = 0; nn < ndx_neighbors.size(); nn++)
        //     {
        //         std::cout << ndx_neighbors[nn] << " ==>";
        //         std::cout<<peaks_pos[ndx_neighbors[nn]][0]<<" "<<peaks_pos[ndx_neighbors[nn]][1]<<" "<<peaks_pos[ndx_neighbors[nn]][2]<<std::endl;
        //     }
        // }

        for (int j = 0; j < title_angle_x.size(); j++)
        {
            std::vector<double> target_line, target_line_x, target_line_y, target_line_z;
            target_line.clear();
            target_line_x.clear();
            target_line_y.clear();
            target_line_z.clear();

            for (int m = -22; m <= 22; m++)
            {
                double target_x = peaks_pos[peak_index][0] + m * zf_step * title_angle_x[j];
                double target_y = peaks_pos[peak_index][1] + m * zf_step * title_angle_y[j];
                double target_z = peaks_pos[peak_index][2] + m * zf_step * title_angle_z[j];
                
                if (target_x >= 0 && target_x < xdim - 1.0 && target_y >= 0 && target_y < ydim - 1.0 && target_z >= 0 && target_z < zdim - 1.0)
                {
                    target_line_x.push_back(target_x);
                    target_line_y.push_back(target_y);
                    target_line_z.push_back(target_z);
                }
            }
            interp3(target_line_x, target_line_y,target_line_z,target_line);


            //cut target_line and get center.
            int anchor_pos = 22;
            int pos_start = 0;
            int pos_end = target_line_x.size();
            cut_one_peak(target_line_x, target_line_y, target_line_z, peak_index, ndx_neighbors, anchor_pos, pos_start, pos_end); //remove segment belongs to other peaks
            // std::cout<<"done cut one peak"<<std::endl;
            // if (peak_index == 195)
            // {
            //     std::cout<<"anchor_pos="<<anchor_pos<<" pos_start="<<pos_start<<" pos_end="<<pos_end<<std::endl;
            // }

            anchor_pos -= pos_start;
            pos_end -= pos_start;

 

            target_line.erase(target_line.begin(), target_line.begin() + pos_start);
            target_line.erase(target_line.begin() + pos_end, target_line.end());

            if(target_line.size()<=3)
            {
                std::cout<<"Peak "<<peak_index<<" is overshadowed by neighboring peaks. skip checking special cases at orientation index j="<<j<<std::endl;
                continue;
            }

            //flip it if it is a negative peak
            std::vector<float> target_line_float;
            for (int ii = 0; ii < target_line.size(); ii++)
            {
                target_line_float.push_back(sign_of_peak*target_line.at(ii));
            }


            // class peak1d p1;
            // if(mod_selection==1) p1.load();
            // else if(mod_selection==2) p1.load_m2();
            // p1.predict(target_line_float);
            // p1.predict_step2();


            class spectrum_pick_1d pp1;
            struct spectrum_1d_peaks p1;
            pp1.init(user_scale1,user_scale2,noise_level);
            pp1.init_mod(mod_selection);
            pp1.direct_set_spectrum(target_line_float);
            pp1.work2(false); //run ann picker, positve peak only
            pp1.get_peaks(p1);
            for(int m=0;m<p1.x.size();m++) p1.x.at(m)*=zf_step;
            anchor_pos*=zf_step;
            // std::cout<<"done 1d prediction"<<std::endl;

            double s1 = 0.5346 * gamma[peak_index][0] * 2 + std::sqrt(0.2166 * 4 * gamma[peak_index][0] * gamma[peak_index][0] + sigma[peak_index][0] * sigma[peak_index][0] * 8 * 0.6931); //need revision here!!!!

            std::vector<int> kk;
            for (int k = 0; k < p1.x.size(); k++)
            {
                if (p1.x.at(k) >= anchor_pos - s1 * 1.2 && p1.x.at(k) <= anchor_pos + s1 * 1.2 && p1.intens.at(k) > noise_level * user_scale1)
                {
                    kk.push_back(k);
                }
            }

            if (kk.size() == 2) //
            {
                int pl = std::min(p1.x[kk[0]], p1.x[kk[1]]);
                int pr = std::max(p1.x[kk[0]], p1.x[kk[1]]);
                // if(pl>anchor_pos+1 || pr<anchor_pos-1) continue; //both peak are both on the left (or right) of the old peak.
                if (pl > anchor_pos || pr < anchor_pos)
                    continue; //both peak are both on the left (or right) of the old peak.

                if (p1.x[kk[0]] < p1.x[kk[1]])
                {
                    std::swap(kk[0], kk[1]);
                }

                std::array<double,3> tpos;
                tpos[0]=(p1.x[kk[0]] - anchor_pos) * title_angle_x[j] + peaks_pos[peak_index][0];
                tpos[1]=(p1.x[kk[0]] - anchor_pos) * title_angle_y[j] + peaks_pos[peak_index][1];
                tpos[2]=(p1.x[kk[0]] - anchor_pos) * title_angle_z[j] + peaks_pos[peak_index][2];
                new_peak1_pos.push_back(tpos);


                tpos[0]=(p1.x[kk[1]] - anchor_pos) * title_angle_x[j] + peaks_pos[peak_index][0];
                tpos[1]=(p1.x[kk[1]] - anchor_pos) * title_angle_y[j] + peaks_pos[peak_index][1];
                tpos[2]=(p1.x[kk[1]] - anchor_pos) * title_angle_z[j] + peaks_pos[peak_index][2];
                new_peak2_pos.push_back(tpos);    
            }
            // std::cout<<"j="<<j<<" and size of add peak is "<<kk.size()<<std::endl;
        }

        if(new_peak1_pos.size()>0)  std::cout << "Potential new peak groups:" << std::endl;
        for (int j = 0; j < new_peak1_pos.size(); j++)
        {
            for(int jj=0;jj<3;jj++)
            {
                std::cout << new_peak1_pos[j][jj] << " ";
            }
            for(int jj=0;jj<3;jj++)
            {
                std::cout << new_peak2_pos[j][jj] << " ";
            }
            std::cout<<std::endl;
        }

        // std::cout << "Finish to check all orientations for peak " << peak_index << " at coor: " << peaks_pos[peak_index][0] + 1 << " " << peaks_pos[peak_index][1] + 1 + 1 << " "<<peaks_pos[peak_index][2] + 1<<std::endl;

        if (new_peak1_pos.size() >= 2) //2 or more peaks.
        {
            std::vector<int> ndxs;
            ldw_math_3d::find_best_from_peaks(new_peak1_pos, new_peak2_pos, ndxs);

            for (int m = 0; m < ndxs.size(); m++)
            {
                bool b_add = false;
                int pos = ndxs.at(m);
                if (find_nearest_normal_peak(new_peak1_pos[pos],ndx_neighbors,peak_index) && find_nearest_normal_peak(new_peak2_pos[pos],ndx_neighbors,peak_index))
                {
                    //check again along perpendicular direction
                    std::vector<double> perpendicular_line_x1, perpendicular_line_y1, perpendicular_line_z1, perpendicular_line_v1;
                    std::vector<double> perpendicular_line_x2, perpendicular_line_y2, perpendicular_line_z2, perpendicular_line_v2;
                    

                    double td1 = (new_peak1_pos[pos][0] - peaks_pos[peak_index][0]) * (new_peak1_pos[pos][0] - peaks_pos[peak_index][0]);
                    td1+= (new_peak1_pos[pos][1] - peaks_pos[peak_index][1]) * (new_peak1_pos[pos][1] - peaks_pos[peak_index][1]);
                    td1+= (new_peak1_pos[pos][2] - peaks_pos[peak_index][2]) * (new_peak1_pos[pos][2] - peaks_pos[peak_index][2]);
                    
                    double td2 = (new_peak2_pos[pos][0] - peaks_pos[peak_index][0]) * (new_peak2_pos[pos][0] - peaks_pos[peak_index][0]);
                    td2+= (new_peak2_pos[pos][1] - peaks_pos[peak_index][1]) * (new_peak2_pos[pos][1] - peaks_pos[peak_index][1]);
                    td2+= (new_peak2_pos[pos][2] - peaks_pos[peak_index][2]) * (new_peak2_pos[pos][2] - peaks_pos[peak_index][2]);

                    if (td1 > td2)
                        ldw_math_3d::get_perpendicular_line(new_peak1_pos[pos], peaks_pos[peak_index], perpendicular_line_x1, perpendicular_line_y1, perpendicular_line_z1, perpendicular_line_x2, perpendicular_line_y2, perpendicular_line_z2,zf_step);
                    else
                        ldw_math_3d::get_perpendicular_line(new_peak2_pos[pos], peaks_pos[peak_index], perpendicular_line_x1, perpendicular_line_y1, perpendicular_line_z1, perpendicular_line_x2, perpendicular_line_y2, perpendicular_line_z2,zf_step);


                    interp3(perpendicular_line_x1, perpendicular_line_y1, perpendicular_line_z1, perpendicular_line_v1);
                    interp3(perpendicular_line_x2, perpendicular_line_y2, perpendicular_line_z2, perpendicular_line_v2);

                    //flip perpendicular_line_v1 and perpendicular_line_v1 if it is a negative peak
                    if(sign_of_peak==-1)
                    {
                        for(int k=0;k<perpendicular_line_v1.size();k++)
                        {
                            perpendicular_line_v1[k]=-perpendicular_line_v1[k];
                            perpendicular_line_v2[k]=-perpendicular_line_v2[k];
                        }
                    }
                    
                    double max_ele;
                    int max_ele_pos1,max_ele_pos2;

                    max_ele = 0.0;
                    max_ele_pos1 = -1;
                    for (int k = 19; k < 26; k++)
                    {
                        if (perpendicular_line_v1.at(k) > max_ele)
                        {
                            max_ele = perpendicular_line_v1.at(k);
                            max_ele_pos1 = k;
                        }
                    }

                    max_ele = 0.0;
                    max_ele_pos2 = -1;
                    for (int k = 19; k < 26; k++)
                    {
                        if (perpendicular_line_v2.at(k) > max_ele)
                        {
                            max_ele = perpendicular_line_v2.at(k);
                            max_ele_pos2 = k;
                        }
                    }

                    if( (abs(max_ele_pos1 - 22) <= 2) && (abs(max_ele_pos1 - 22) <= 2) )
                    {
                        b_add = true;
                        std::cout << "ADD new peak at: " << new_peak1_pos[pos][0] << " " << new_peak1_pos[pos][1]  << " " << new_peak1_pos[pos][2] ;
                        std::cout << " and " << new_peak2_pos[pos][0] << " " << new_peak2_pos[pos][1] << " " << new_peak2_pos[pos][2] << std::endl;
                        fnewpeak << peaks_pos[peak_index][0] << " " << peaks_pos[peak_index][1] << " " << peaks_pos[peak_index][2];
                        fnewpeak << " " << new_peak1_pos[pos][0] << " " << new_peak1_pos[pos][1] << " " << new_peak1_pos[pos][2];
                        fnewpeak << " " << new_peak2_pos[pos][0] << " " << new_peak2_pos[pos][1] << " " << new_peak2_pos[pos][2];
                        fnewpeak << std::endl;

                        to_remove[peak_index]=1;
                        std::array<double,3> t1,t2;

                        //add new peaks pair, position
                        t1=new_peak1_pos[pos];
                        t2=new_peak2_pos[pos];
                        peaks_pos.push_back(t1);
                        peaks_pos.push_back(t2);

                        //sigma
                        t1=sigma[peak_index];
                        sigma.push_back(t1);
                        sigma.push_back(t1);

                        //gamma
                        t1=gamma[peak_index];
                        gamma.push_back(t1);
                        gamma.push_back(t1);

                        //intensity
                        intensity.push_back(intensity[peak_index]/2.0);
                        intensity.push_back(intensity[peak_index]/2.0);
                        
                    }
                }
                if (b_add == false)
                {
                    std::cout << "Potential new peak at: " << new_peak1_pos[pos][0] << " " << new_peak1_pos[pos][1]  << " " << new_peak1_pos[pos][2] ;
                    std::cout << " and " << new_peak2_pos[pos][0] << " " << new_peak2_pos[pos][1] << " " << new_peak2_pos[pos][2] << std::endl;
                }
            }
        }
        std::cout << "Finish peak " << peak_index << " at coor: " << peaks_pos[peak_index][0] << " " << peaks_pos[peak_index][1] << " "<<peaks_pos[peak_index][2]<<std::endl;
    }
    fnewpeak.close();

    //remove peaks that have been replace by peak pairs in special case processing.
    for(int i=npeak;i>=0;i--)
    {
        if(to_remove[i]==1)
        {
            peaks_pos.erase(peaks_pos.begin()+i);
            sigma.erase(sigma.begin()+i);
            gamma.erase(gamma.begin()+i);
            intensity.erase(intensity.begin()+i);
        }
    }

    return true;
};



bool spectrum_pick_3d::find_nearest_normal_peak(std::array<double,3> x,std::vector<int> ndxs, int p)
{

    double max_amp=0.0;
    double max_effect=0.0;
    for(int ii=0;ii<ndxs.size();ii++)
    {
        int i=ndxs.at(ii);
        double eff=voigt(peaks_pos[i][0]-x[0],sigma[i][0],gamma[i][0])*voigt(peaks_pos[i][1]-x[1],sigma[i][1],gamma[i][1])*voigt(peaks_pos[i][2]-x[2],sigma[i][2],gamma[i][2]);
        double amp=intensity[i]*eff;
       
        if(amp>max_amp)
        {
            max_amp=amp;
        }
      
    }
    double amp_at_p=intensity[p]*voigt(peaks_pos[p][0]-x[0],sigma[p][0],gamma[p][0])*voigt(peaks_pos[p][1]-x[1],sigma[p][1],gamma[p][1])*voigt(peaks_pos[p][2]-x[2],sigma[p][2],gamma[p][2]);

    if(max_amp>amp_at_p/3.0)
        return false;
    else
        return true;
};

/**
 * @brief Get the triangles object
 * From 4 1D spectra, defined by ii,ii+1,jj,jj+1
 * @param ii 
 * @param jj 
 * @return true 
 * @return false 
 */

bool spectrum_pick_3d::get_triangles(int ii, int jj)
{
    std::vector<int> t11,t12,t21,t22,tsign;

    double ncut=2.0;

    int k11=ii*ydim+jj;
    int k12=(ii+1)*ydim+jj;
    int k21=ii*ydim+jj+1;
    int k22=(ii+1)*ydim+jj+1;

    std::vector<int> pos12,pos21,pos22;

    for(int i11=0;i11<spectra_1d.at(k11).x.size();i11++)
    {

        for(int i12=0;i12<spectra_1d.at(k12).x.size();i12++)
        {
            if(abs(spectra_1d.at(k11).x.at(i11)-spectra_1d.at(k12).x.at(i12))<=ncut)
            {
                pos12.push_back(i12);
            }
        }
        for(int i21=0;i21<spectra_1d.at(k21).x.size();i21++)
        {
            if(abs(spectra_1d.at(k11).x.at(i11)-spectra_1d.at(k21).x.at(i21))<=ncut)
            {
                pos21.push_back(i21);
            }
        }

        for(int m=0;m<pos12.size();m++)
        {
            for(int n=0;n<pos21.size();n++)
            {
                int i12=pos12.at(m);
                int i21=pos21.at(n);
                if(abs(spectra_1d.at(k12).x.at(i12)-spectra_1d.at(k21).x.at(i21))<=ncut)   
                {
                    for(int i22=0;i22<spectra_1d.at(k22).x.size();i22++)
                    {
                        if(abs(spectra_1d.at(k11).x.at(i11)-spectra_1d.at(k22).x.at(i22))<=ncut && abs(spectra_1d.at(k21).x.at(i21)-spectra_1d.at(k22).x.at(i22))<=ncut && abs(spectra_1d.at(k12).x.at(i12)-spectra_1d.at(k22).x.at(i22))<=ncut)
                        {
                            //all 4 peaks are within ncut along z direction!!

                            //only if all 4 peaks have same sign.
                            if(spectra_1d.at(k11).a.at(i11)*spectra_1d.at(k12).a.at(i12)>0 &&
                                spectra_1d.at(k11).a.at(i11)*spectra_1d.at(k21).a.at(i21)>0 &&
                                spectra_1d.at(k11).a.at(i11)*spectra_1d.at(k22).a.at(i22)>0)
                            {
                                t11.push_back(i11);
                                t12.push_back(i12);
                                t21.push_back(i21);
                                t22.push_back(i22);
                                if(spectra_1d.at(k11).a.at(i11)>0)
                                    tsign.push_back(1);
                                else
                                    tsign.push_back(-1);
                            }
                        }
                    }
                }
            }
        }
    }
    if(t11.size()>0)
    {
        // std::cout<<"I find "<<t11.size()<<" "<<t12.size()<<" "<<t21.size()<<" "<<t22.size()<<" "<<"triangles at "<<ii<<" "<<jj<<std::endl;
        for(int n=0;n<t11.size();n++)
        {
            double z1=spectra_1d.at(k11).x.at(t11.at(n));
            double z2=spectra_1d.at(k12).x.at(t12.at(n));
            double z3=spectra_1d.at(k21).x.at(t21.at(n));
            double z4=spectra_1d.at(k22).x.at(t22.at(n));
            
            std::cout<<"triangle is "<<z1<<" "<<z2<<" "<<z3<<" "<<z4<<" at "<<ii<<" "<<jj<<std::endl;
        }
    }
    z_triangles11.push_back(t11);
    z_triangles21.push_back(t21);
    z_triangles12.push_back(t12);
    z_triangles22.push_back(t22);
    z_triangles_sign.push_back(tsign);
    
    return true;
};

/**
 * @brief get lines along Z direction that connect plane ii and ii+1
 * Modify z_lines1 and z_lines2. Both have the size of # of planes -1
 * @param ii plane index
 * @return true 
 * @return false 
 */
bool spectrum_pick_3d::get_lines(int ii)
{
    double ncut=2.0;
    std::vector<int> t1,t2,t3;

    for(int k1=0;k1<spectra_2d_for_picking.at(ii).p1.size();k1++)
    {
        for(int k2=0;k2<spectra_2d_for_picking.at(ii+1).p1.size();k2++)
        {
            //only if peaks have the same sign, they can be connected
            if(spectra_2d_for_picking.at(ii).p_intensity.at(k1)*spectra_2d_for_picking.at(ii+1).p_intensity.at(k2)<0) continue;

            double d1=fabs(spectra_2d_for_picking.at(ii).p1.at(k1)-spectra_2d_for_picking.at(ii+1).p1.at(k2));
            double d2=fabs(spectra_2d_for_picking.at(ii).p2.at(k1)-spectra_2d_for_picking.at(ii+1).p2.at(k2));
            if(d1+d2<=ncut)
            {
                t1.push_back(k1);
                t2.push_back(k2);
                if(spectra_2d_for_picking.at(ii).p_intensity.at(k1)>0)
                {
                    t3.push_back(1);
                }
                else
                {
                    t3.push_back(-1);
                }
            }
        }    
    }
    // std::cout<<"Total "<<t1.size()<<" lines between plane "<<ii<<" and "<<ii+1<<std::endl;
    for(int n=0;n<t1.size();n++)
    {
        std::cout<<"between plane "<<ii<<" and "<<ii+1;
        std::cout<<" from "<<spectra_2d_for_picking.at(ii).p1.at(t1.at(n))<<" "<<spectra_2d_for_picking.at(ii).p2.at(t1.at(n));
        std::cout<<" to "<<spectra_2d_for_picking.at(ii+1).p1.at(t2.at(n))<<" "<<spectra_2d_for_picking.at(ii+1).p2.at(t2.at(n));
        std::cout<<std::endl;
    }
    z_line1.push_back(t1);
    z_line2.push_back(t2);
    z_line_sign.push_back(t3);
    return true;
};

bool spectrum_pick_3d::interp3(std::vector<double> line_x, std::vector<double> line_y,std::vector<double> line_z, std::vector<double> &line_v)
{
    int ndata=line_x.size();
    int min_x=std::max(int(floor(std::min(line_x.at(0),line_x.back())))-2,0);
    int max_x=std::min(int(ceil(std::max(line_x.at(0),line_x.back())))+2,xdim-1);
    int min_y=std::max(int(floor(std::min(line_y.at(0),line_y.back())))-2,0);
    int max_y=std::min(int(ceil(std::max(line_y.at(0),line_y.back())))+2,ydim-1);
    int min_z=std::max(int(floor(std::min(line_z.at(0),line_z.back())))-2,0);
    int max_z=std::min(int(ceil(std::max(line_z.at(0),line_z.back())))+2,zdim-1);
    
    std::vector<  std::vector<double>  > spe_at_z; //spe_at_z[i*ydim+j][z_coor]
    std::vector< std::vector<double> > row_spe_at_x;
    std::vector< std::vector<double> > column_spe_at_x;


    std::vector<double> z_input;
    for(int k=min_z;k<=max_z;k++) z_input.push_back(k);

    for(int i=min_x;i<=max_x;i++)
    {
        for(int j=min_y;j<=max_y;j++)
        {
            std::vector<double> v_input,t;
            t.clear();
            v_input.clear();
            for(int k=min_z;k<=max_z;k++) v_input.push_back(spectra_2d_for_picking.at(k).get_spect_data()[i+j*xdim]);
            tk::spline st(z_input,v_input);   
            for(int m=0;m<ndata;m++)
            {
                t.push_back(st(line_z.at(m)));  
            }
            spe_at_z.push_back(t);
        }
    }

    line_v.clear();
    for(int k=0;k<ndata;k++)
    {
        std::vector<double> tdata;
        for(int i=0;i<spe_at_z.size();i++) tdata.push_back(spe_at_z.at(i).at(k));
        line_v.push_back(ldw_math_3d::interp2_point(min_x,max_x,min_y,max_y,tdata,line_x.at(k),line_y.at(k)));
    }

    return true;
};

//find intersection of polygon with a line
//polygon is defind as (0,0,p11),(0,1,p12),(1,0,p21) and (1,1, p22)
//line is defined as (x1,y1,0) and (x2,y2,1)
bool spectrum_pick_3d::get_line_polygon_intersection(int p11,int p12,int p21,int p22,int x1,int x2,int y1,int y2, double IntersectionPoint[3])
{
    bool b=false;
    double rayOrigin[3],rayVector[3],vertex0[3],vertex1[3],vertex2[3],vertex3[3];

    rayOrigin[0]=x1;
    rayOrigin[1]=y1;
    rayOrigin[2]=0.0;

    rayVector[0]=x2-x1;
    rayVector[1]=y2-y1;
    rayVector[2]=1.0;

    
    vertex0[0]=0;
    vertex0[1]=1;
    vertex0[2]=p12;
    
    vertex1[0]=1;
    vertex1[1]=0;
    vertex1[2]=p21;

    vertex2[0]=0;
    vertex2[1]=0;
    vertex2[2]=p11;

    vertex3[0]=1;
    vertex3[1]=1;
    vertex3[2]=p22;

    
    b=RayIntersectsTriangle(rayOrigin,rayVector,vertex0,vertex1,vertex2,IntersectionPoint) || RayIntersectsTriangle(rayOrigin,rayVector,vertex0,vertex1,vertex3,IntersectionPoint);

    
    return b;
};


bool spectrum_pick_3d::RayIntersectsTriangle(double rayOrigin[3], 
                           double rayVector[3], 
                           double vertex0[3],
                           double vertex1[3],
                           double vertex2[3],
                           double outIntersectionPoint[3])
{
    const float EPSILON = 0.0000001;
    double edge1[3], edge2[3], h[3], s[3], q[3];
    float a,f,u,v;

    for(int i=0;i<3;i++)
    {
        edge1[i]=vertex1[i]-vertex0[i];
        edge2[i]=vertex2[i]-vertex0[i];        
    }

    h[0]=rayVector[1]*edge2[2]-rayVector[2]*edge2[1];
    h[1]=rayVector[2]*edge2[0]-rayVector[0]*edge2[2];
    h[2]=rayVector[0]*edge2[1]-rayVector[1]*edge2[0];
    
    a = edge1[0]*h[0]+edge1[1]*h[1]+edge1[2]*h[2];

    if (a > -EPSILON && a < EPSILON)
        return false;    // This ray is parallel to this triangle.

    f = 1.0/a;
    for(int i=0;i<3;i++)
    {
        s[i]=rayOrigin[i]-vertex0[i];
    }
    u=(s[0]*h[0]+s[1]*h[1]+s[2]*h[2])*f;
    
    if (u < 0.0 || u > 1.0)
        return false;

    q[0]=s[1]*edge1[2]-s[2]*edge1[1];
    q[1]=s[2]*edge1[0]-s[0]*edge1[2];
    q[2]=s[0]*edge1[1]-s[1]*edge1[0];

    v=(rayVector[0]*q[0]+rayVector[1]*q[1]+rayVector[2]*q[2])*f;
    if (v < 0.0 || u + v > 1.0)
        return false;


    // At this stage we can compute t to find out where the intersection point is on the line.
    float t=(edge2[0]*q[0]+edge2[1]*q[1]+edge2[2]*q[2])*f;

    if (t >= 0 && t<=1.0) // line segment intersection
    {
        for(int i=0;i<3;i++)
        {
            outIntersectionPoint[i] = rayOrigin[i] + rayVector[i] * t;
        }
        return true;
    }
    else // This means that there is a line intersection (t<0) or ray intersection (t>1) but not a line segment intersection.
        return false;
};


bool spectrum_pick_3d::cut_one_peak(std::vector<double> target_line_x,std::vector<double> target_line_y,std::vector<double> target_line_z,int current_pos,std::vector<int> ndx_neighbors, int anchor_pos,int &pos_start,int &pos_end)
{
    pos_start=0;
    pos_end=target_line_x.size();

    for(int i=anchor_pos;i>=0;i--)
    {
        double x=target_line_x[i];
        double y=target_line_y[i];
        double z=target_line_z[i];
        
        double z_current=intensity[current_pos]*voigt(peaks_pos[current_pos][0]-x,sigma[current_pos][0],gamma[current_pos][0])
                            *voigt(peaks_pos[current_pos][1]-y,sigma[current_pos][1],gamma[current_pos][1])
                            *voigt(peaks_pos[current_pos][2]-z,sigma[current_pos][2],gamma[current_pos][2]);

        bool b_found=false;
        for(int j=0;j<ndx_neighbors.size();j++)
        {
            int test_pos=ndx_neighbors[j];
            double z_test=intensity[test_pos]*voigt(peaks_pos[test_pos][0]-x,sigma[test_pos][0],gamma[test_pos][0])
                                            *voigt(peaks_pos[test_pos][1]-y,sigma[test_pos][1],gamma[test_pos][1])
                                            *voigt(peaks_pos[test_pos][2]-z,sigma[test_pos][2],gamma[test_pos][2]);
            if(z_test>z_current)
            {
                b_found=true;
                break;
            }
        }
        if(b_found)
        {
            pos_start=i;
            break;
        }
    }

    for(int i=anchor_pos;i<target_line_x.size();i++)
    {
        double x=target_line_x[i];
        double y=target_line_y[i];

        double z_current=intensity[current_pos]*voigt(peaks_pos[current_pos][0]-x,sigma[current_pos][0],gamma[current_pos][0])
                            *voigt(peaks_pos[current_pos][1]-x,sigma[current_pos][1],gamma[current_pos][1])
                            *voigt(peaks_pos[current_pos][2]-x,sigma[current_pos][2],gamma[current_pos][2]);

        bool b_found=false;
        for(int j=0;j<ndx_neighbors.size();j++)
        {
            int test_pos=ndx_neighbors[j];
            double z_test=intensity[test_pos]*voigt(peaks_pos[test_pos][0]-x,sigma[test_pos][0],gamma[test_pos][0])
                                            *voigt(peaks_pos[test_pos][1]-x,sigma[test_pos][1],gamma[test_pos][1])
                                            *voigt(peaks_pos[test_pos][2]-x,sigma[test_pos][2],gamma[test_pos][2]);
            if(z_test>z_current)
            {
                b_found=true;
                break;
            }
        }
        if(b_found)
        {
            pos_end=i;
            break;
        }
    }
    return true;
}


bool spectrum_pick_3d::simple_peak_picking()
{
    for(int k=1;k<spectra_2d_for_picking.size()-1;k++)
    {
        spectra_2d_for_picking.at(k).set_scale(user_scale1,user_scale2);
        spectra_2d_for_picking.at(k).set_model_selection(mod_selection);
        spectra_2d_for_picking.at(k).set_noise_level(noise_level);
        spectra_2d_for_picking.at(k).simple_peak_picking(b_negative);

        for(int m=0;m<spectra_2d_for_picking.at(k).p1.size();m++)
        {
            int i=spectra_2d_for_picking.at(k).p1.at(m);
            int j=spectra_2d_for_picking.at(k).p2.at(m);
            
            float f=spectra_2d_for_picking.at(k).get_spect_data()[i+j*xdim];
            if(f>0 && f>spectra_2d_for_picking[k-1].get_spect_data()[i+j*xdim] && f>spectra_2d_for_picking[k+1].get_spect_data()[i+j*xdim])
            {
                intensity.push_back(spectra_2d_for_picking.at(k).p_intensity.at(m));
                std::array<double,3> temp={double(i),double(j),double(k)};
                peaks_pos.push_back(temp);
            }
            else if(b_negative && f<0 && -f>-spectra_2d_for_picking[k-1].get_spect_data()[i+j*xdim] && -f>-spectra_2d_for_picking[k+1].get_spect_data()[i+j*xdim])
            {
                intensity.push_back(spectra_2d_for_picking.at(k).p_intensity.at(m));
                std::array<double,3> temp={double(i),double(j),double(k)};
                peaks_pos.push_back(temp);    
            }
        }
    }

    for(int m=0;m<intensity.size();m++)
    {
        std::array<double,3> temp={0.,0.,0.};
        sigma.push_back(temp);
        gamma.push_back(temp);   
    }
    return true;
};

bool spectrum_pick_3d::peak_picking()
{
    

    //plane by plane picking (x,y)
    for(int i=0;i<spectra_2d_for_picking.size();i++)
    {
        spectra_2d_for_picking.at(i).set_scale(user_scale1,user_scale2);
        spectra_2d_for_picking.at(i).set_model_selection(mod_selection);
        spectra_2d_for_picking.at(i).set_noise_level(noise_level);
        spectra_2d_for_picking.at(i).ann_peak_picking(0,zf,0,b_negative); 
        std::cout<<std::endl<<"Finish 2D peaks picking of plane "<<i<<" out of "<<spectra_2d_for_picking.size()<<std::endl;
    }

    //intermediate peaks for debug purpose
#ifdef LDW_DEBUG
    std::ofstream f_intermediate("intermediate.txt");
    f_intermediate<<"plane by plane peaks:"<<std::endl;
    for(int i=0;i<spectra_2d_for_picking.size();i++)
    {
        for(int j=0;j<spectra_2d_for_picking.at(i).p1.size();j++)
        {
            f_intermediate<<i<<" "<<spectra_2d_for_picking.at(i).p1.at(j)<<" "<<spectra_2d_for_picking.at(i).p2.at(j);
            f_intermediate<<" "<<spectra_2d_for_picking.at(i).p_intensity.at(j)<<std::endl;
            f_intermediate<<std::endl;
        }
    }
    f_intermediate<<std::endl;
#endif

    //connect dots to lines. Lines are along z direction
    for(int i=0;i<spectra_2d_for_picking.size()-1;i++)
    {
        get_lines(i); //get lines connected from plane i to plane i+1
    }
    std::cout<<std::endl<<"Finish connection of dots into lines from 2D peaks picking."<<std::endl;

    //1D picking of all traces along z
    int count=0;
    #ifdef LDW_DEBUG
    f_intermediate<<"Z direction 1D peaks:"<<std::endl;
    #endif
    for(int i=0;i<xdim;i++)
    {
        for(int j=0;j<ydim;j++)
        {
           
            class spectrum_pick_1d x;
            struct spectrum_1d_peaks y;
            std::vector<int> temp;
            x.init(user_scale1,user_scale2,noise_level);
            x.init_mod(mod_selection);
            if(zf==0)
            {
                std::vector<float> trace;
                for(int k=0;k<zdim;k++)
                {
                    trace.push_back(spectra_2d_for_picking.at(k).get_spect_data()[i+j*xdim]);
                }
                x.direct_set_spectrum(trace);
                x.work2(b_negative); //run ann picker
                x.get_peaks(y);
            }
            else
            {
                std::vector<double> trace;
                std::vector<float> trace2;
                for(int k=0;k<zdim;k++)
                {
                    trace.push_back(spectra_2d_for_picking.at(k).get_spect_data()[i+j*xdim]);
                }
                std::vector<double> z_input,z_output;
                for(int m = 0; m < zdim-1; m++)
                {
                    z_input.push_back(m);
                    z_output.push_back(m+0.5);
                }
                z_input.push_back(zdim-1);
                tk::spline st(z_input,trace);
                
                for(int m=0;m<zdim-1;m++)
                {
                    trace2.push_back(trace.at(m));
                    trace2.push_back(st(z_output.at(m)));
                }
                trace2.push_back(trace[zdim-1]);
                x.direct_set_spectrum(trace2);
                x.work2(); //run ann picker
                x.get_peaks(y);
                for(int m=0;m<y.x.size();m++) y.x.at(m)*=0.5;
            }
            spectra_1d.emplace_back(y); //keep only peaks without spectrum, to save memory
            count+=temp.size();
            #ifdef LDW_DEBUG
            for(int k=0;k<y.x.size();k++)
            {
                f_intermediate<<i<<" "<<j<<" "<<y.x[k]<<std::endl;
            }
            #endif
        }
        std::cout<<"Finish "<<i<<" out of "<<xdim-1<<" in 1D picking along Z"<<std::endl;
    }
    #ifdef LDW_DEBUG
    f_intermediate<<std::endl;
    f_intermediate.close();
    #endif
    std::cout<<std::endl<<"Finish 1D peaks picking of all traces along Z direction, total "<<count<<" peaks."<<std::endl;

    //get triangles 
    for(int i=0;i<xdim-1;i++)
    {
        for(int j=0;j<ydim-1;j++)
        {
            get_triangles(i,j);
            // std::cout<<j<<" ";
        }
        // std::cout<<"Finish "<<i<<" out of "<<xdim-1<<std::endl;
    }
    std::cout<<std::endl<<"Finish connect Z peaks to triangles."<<std::endl;


    //put z_line into x-y grid to speed up (like cell method in MD non-bonded force calculation)

    std::vector< std::vector< std::array<int,3> > > cell_of_zline;
    cell_of_zline.resize((xdim-1)*(ydim-1));
    for (int kk = 0; kk < z_line1.size(); kk++)
    {
        for (int kk2 = 0; kk2 < z_line1.at(kk).size(); kk2++)
        {
            int k1 = z_line1.at(kk).at(kk2); //peak index in plane kk
            int k2 = z_line2.at(kk).at(kk2); //peak index in plane kk+1

            //a line segment along Z direction
            double x1 = spectra_2d_for_picking.at(kk).p1.at(k1);     //x coor of first point
            double x2 = spectra_2d_for_picking.at(kk + 1).p1.at(k2); //x coor of second point
            double y1 = spectra_2d_for_picking.at(kk).p2.at(k1);     //y coor of first point
            double y2 = spectra_2d_for_picking.at(kk + 1).p2.at(k2); //y coor of second point

            int x=int(std::min(x1,x2));
            int y=int(std::min(y1,y2));

            if(x>=xdim-1 || y>=ydim-1) 
            {
                continue;
            }

            std::array<int,3> temp={kk,k1,k2};
            cell_of_zline[x*(ydim-1)+y].push_back(temp);
        }
    }

    //find intersection of lines (along Z dirction) and all the triangles (largely on x-y plane), which are 3D peaks
    for(int ii=0;ii<xdim-1;ii++)
    {
        for(int jj=0;jj<ydim-1;jj++)
        {
            int kk11=ii*(ydim-1)+jj;
            int k11=ii*ydim+jj;
            int k12=(ii+1)*ydim+jj;
            int k21=ii*ydim+jj+1;
            int k22=(ii+1)*ydim+jj+1;
            
            //z_triangles.at(kk11) have all 3D polygons that are within ii, ii+1 (x) and jj,jj+1 (y)
            //we will then check x and y coordinates of all z lines and select those are possible to intersect

            // for (int kk = 0; kk < z_line1.size(); kk++)
            // {
            //     for (int kk2 = 0; kk2 < z_line1.at(kk).size(); kk2++)
            //     {
            //         int k1 = z_line1.at(kk).at(kk2); //peak index in plane kk
            //         int k2 = z_line2.at(kk).at(kk2); //peak index in plane kk+1
            //     }
            // }

            for (int k0 = 0; k0 < cell_of_zline[ii * (ydim - 1) + jj].size(); k0++)
            {
                int kk = cell_of_zline[ii * (ydim - 1) + jj][k0][0];
                int k1 = cell_of_zline[ii * (ydim - 1) + jj][k0][1];
                int k2 = cell_of_zline[ii * (ydim - 1) + jj][k0][2];

                //a line segment along Z direction
                double x1 = spectra_2d_for_picking.at(kk).p1.at(k1);     //x coor of first point
                double x2 = spectra_2d_for_picking.at(kk + 1).p1.at(k2); //x coor of second point
                double y1 = spectra_2d_for_picking.at(kk).p2.at(k1);     //y coor of first point
                double y2 = spectra_2d_for_picking.at(kk + 1).p2.at(k2); //y coor of second point

                //their are lots of ploygons that within ii,ii+1 and jj,jj+1 (they have various z coor)
                //check whether these polygons intersection with our line, whose Z is from kk to kk+1
                for (int m = 0; m < z_triangles11.at(kk11).size(); m++)
                {
                    //if different sign, skip because it is a saddle point, not a peak
                    if(spectra_2d_for_picking.at(kk).p_intensity.at(k1)*z_triangles_sign.at(kk11).at(m)<0)
                    {
                        continue;
                    }

                    double z1 = spectra_1d.at(k11).x.at(z_triangles11.at(kk11).at(m));
                    double z2 = spectra_1d.at(k12).x.at(z_triangles12.at(kk11).at(m));
                    double z3 = spectra_1d.at(k21).x.at(z_triangles21.at(kk11).at(m));
                    double z4 = spectra_1d.at(k22).x.at(z_triangles22.at(kk11).at(m));

                    double z_low = std::min(z1, std::min(z2, std::min(z3, z4)));
                    double z_high = std::max(z1, std::max(z2, std::max(z3, z4)));

                    bool bx1 = z_low < kk && z_high < kk;
                    bool bx2 = z_low > kk + 1 && z_high > kk + 1;

                    if (!(bx1 || bx2))
                    {
                        //All are in relative coords of ii,jj and kk
                        std::array<double, 3> IntersectionPoint = {0.0, 0.0, 0.0};

                        if (get_line_polygon_intersection(z1 - kk, z2 - kk, z3 - kk, z4 - kk, x1 - ii, x2 - ii, y1 - jj, y2 - jj, IntersectionPoint.data()))
                        {
                            //for debug

                            IntersectionPoint[0] += ii;
                            IntersectionPoint[1] += jj;
                            IntersectionPoint[2] += kk; //temp fix. still a pazzle for me. kk+1

                            double a;
                            std::array<double, 3> s = {0., 0., 0.};
                            std::array<double, 3> g = {0., 0., 0.};

                            s[2] = spectra_1d.at(ii * ydim + jj).sigmax.at(z_triangles11.at(kk11).at(m)); // 1d peak para
                            s[0] = spectra_2d_for_picking.at(kk).sigmax.at(k1);                           //2d peak para
                            s[1] = spectra_2d_for_picking.at(kk).sigmay.at(k1);                           //2d peak para
                            g[2] = spectra_1d.at(ii * ydim + jj).gammax.at(z_triangles11.at(kk11).at(m)); // 1d peak para
                            g[0] = spectra_2d_for_picking.at(kk).gammax.at(k1);                           //2d peak para
                            g[1] = spectra_2d_for_picking.at(kk).gammay.at(k1);                           //2d peak para

  
                            //3D peak intensity is the minimum of 1D and 2D peak intensity. It has the same sign as 2D peak intensity (or 1D peak intensity)
                            a = std::min(fabs(spectra_1d.at(ii * ydim + jj).a.at(z_triangles11.at(kk11).at(m))), fabs(spectra_2d_for_picking.at(kk).p_intensity.at(k1)));
                            if(spectra_2d_for_picking.at(kk).p_intensity.at(k1)<0)
                            {
                                a=-a;
                            }

                            intensity.push_back(a);
                            peaks_pos.push_back(IntersectionPoint);
                            sigma.push_back(s);
                            gamma.push_back(g);

                            std::vector<double> t;
                            t.push_back(IntersectionPoint[0]);
                            t.push_back(IntersectionPoint[1]);
                            t.push_back(IntersectionPoint[2]);
                            t.push_back(ii);
                            t.push_back(jj);
                            t.push_back(z1);
                            t.push_back(z2);
                            t.push_back(z3);
                            t.push_back(z4);
                            t.push_back(kk);
                            t.push_back(x1);
                            t.push_back(y1);
                            t.push_back(x2);
                            t.push_back(y2);
                            for_debug.push_back(t);
                        }
                    }
                } //for(int m=0;m<z_triangles11.at(k11).size();m++)

            } //(int kk=0;kk<cell_of_zline[ii*(ydim-1)+jj].size();kk++)
        }     //for(int jj=0;jj<ydim-1;jj++)
    }         //for(int ii=0;ii<xdim-1;ii++)

    std::cout<<std::endl<<"Finish identify all 3D peaks. Total "<<intensity.size()<<" peaks. "<<std::endl;

    int npeak=intensity.size();

    std::cout<<"Sizes intensity: "<<intensity.size()<<" peak_pos: "<<peaks_pos.size()<<" sigmas "<<sigma.size()<<" gammas "<<gamma.size()<<std::endl;

    //print peaks before filtering for debug purpose
#ifdef LDW_DEBUG
    FILE *fp_peaks=fopen("for_debug_peaks.txt","w");
    if(fp_peaks==NULL)
    {
        std::cout<<"can't open for_debug_peaks.txt to write."<<std::endl;
    }
    else
    {
        for(int i=0;i<npeak;i++)
        {
            fprintf(fp_peaks,"%7.2f %7.2f %7.2f %7.2f ",peaks_pos.at(i)[0],peaks_pos.at(i)[1],peaks_pos.at(i)[2],intensity.at(i));
            fprintf(fp_peaks,"%7.2f %7.2f %7.2f ",sigma.at(i)[0],sigma.at(i)[1],sigma.at(i)[2]);
            fprintf(fp_peaks,"%7.2f %7.2f %7.2f ",gamma.at(i)[0],gamma.at(i)[1],gamma.at(i)[2]);
            fprintf(fp_peaks,"\n");
        }
        fclose(fp_peaks);
    }

    std::cout<<std::endl<<"Finish for_debug_peaks.txt "<<std::endl;

    FILE *fdebug=fopen("for_debug_base.txt","w");
    for(int i=0;i<for_debug.size();i++)
    {
        for(int j=0;j<for_debug.at(i).size();j++)
        {
            fprintf(fdebug,"%8.3f ",for_debug.at(i).at(j));
        }
        fprintf(fdebug,"\n");
    }
    fclose(fdebug);
    

    FILE *fd_list=fopen("for_debug_list.txt","w");
    for(int i=0;i<npeak;i++)
    {
        for(int j=i+1;j<npeak;j++)
        {
            double d1=peaks_pos.at(i)[0]-peaks_pos.at(j)[0];
            double d2=peaks_pos.at(i)[1]-peaks_pos.at(j)[1];
            double d3=peaks_pos.at(i)[2]-peaks_pos.at(j)[2];
            double dd=fabs(d1)+fabs(d2)+fabs(d3);
            if(dd<0.1)
            {
                fprintf(fd_list,"%d %d %f\n",i,j,dd);
            }
        }
    }
    fclose(fd_list);
#endif


    //filter deplicated peaks.
    std::vector<int> remove_flags;
    remove_flags.resize(npeak,0);
    for(int i=0;i<npeak;i++)
    {
        if(remove_flags.at(i)==1) continue;
        for(int j=0;j<npeak;j++)
        {
            if(j==i) continue;
            if(remove_flags.at(j)==1) continue;
            double d1=peaks_pos.at(i)[0]-peaks_pos.at(j)[0];
            double d2=peaks_pos.at(i)[1]-peaks_pos.at(j)[1];
            double d3=peaks_pos.at(i)[2]-peaks_pos.at(j)[2];
            if(fabs(d1)+fabs(d2)+fabs(d3)<3.1)
            {
                if(fabs(intensity.at(i))>fabs(intensity.at(j)))
                {
                    remove_flags.at(j)=1;
                }    
                else
                {
                    remove_flags.at(i)=1;
                }
            }
        }
    }

    for(int i=npeak-1;i>=0;i--)
    {
        if(remove_flags.at(i)==1)
        {
            intensity.erase(intensity.begin()+i);   
            peaks_pos.erase(peaks_pos.begin()+i);
            sigma.erase(sigma.begin()+i);
            gamma.erase(gamma.begin()+i);
        }
    }

    return true;
};



bool spectrum_pick_3d::print_peaks(std::string fname)
{
    FILE *fp=fopen(fname.c_str(),"w");

    fprintf(fp,"Name F1ppm F2ppm F3ppm intensity F1 F2 F3 sigma1 sigma2 sigma3 gamma1 gamma2 gamma3\n");


    std::vector<int> ndx;
    sortArr(intensity,ndx);

    for(int j=ndx.size()-1;j>=0;j--)
    {
        int i=ndx.at(j);

        double ppm[3];
        for(int m=0;m<3;m++)
        {
            ppm[m]=begin[m] + step[m] * peaks_pos.at(i)[m];
        }

        fprintf(fp,"(%d)N-C-H %7.2f %7.2f %7.2f %7.2f ",j+1,ppm[0],ppm[1],ppm[2],intensity.at(i));
        fprintf(fp,"%7.2f %7.2f %7.2f ",peaks_pos.at(i)[0],peaks_pos.at(i)[1],peaks_pos.at(i)[2]);
        fprintf(fp,"%7.2f %7.2f %7.2f ",sigma.at(i)[0],sigma.at(i)[1],sigma.at(i)[2]);
        fprintf(fp,"%7.2f %7.2f %7.2f ",gamma.at(i)[0],gamma.at(i)[1],gamma.at(i)[2]);
        fprintf(fp,"\n");
    }
    fclose(fp);
    return true;
};







