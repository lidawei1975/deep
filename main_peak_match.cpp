
#include <fstream>
#include <iostream>
#include <iomanip>      // std::setw
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>
#include <limits>

#include "commandline.h"
#include "peak_manipulation.h"
#include "DeepConfig.h"







int main(int argc, char **argv)
{ 
    std::cout<<"DEEP Picker package Version "<<deep_picker_VERSION_MAJOR<<"."<<deep_picker_VERSION_MINOR<<std::endl;
    
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-f");
    args2.push_back("arguments_peak_match.txt");
    args3.push_back("command line arguments file");
    
    args.push_back("-in1");
    args2.push_back("test.tab");
    args3.push_back("Peak file 1 with assignment (.tab or .list)");   

    args.push_back("-in2");
    args2.push_back("peaks.tab");
    args3.push_back("Peak file 2 without assignment (.tab only)");

    args.push_back("-cutoff");
    args2.push_back("0.06");
    args3.push_back("peak matching cutoff sqrt(1H^2+13C^2*0.01) in unit of ppm");

    args.push_back("-out");
    args2.push_back("match.tab");
    args3.push_back("output file name with assignment transfer from input 1 to input 2");

    args.push_back("-out-ass");
    args2.push_back("assignment.txt");
    args3.push_back("out file contain assignments, one line per peak in in2 peak file");
   
    std::vector<std::string> header,header1;
    std::vector<std::string> lines,lines1; 
    std::vector<std::array<double,2>> peak_pos1, peak_pos2;
    std::vector<std::string> peak_infor1;
    std::array<double,4> region1,region;

    std::string fin_name1,fin_name2,fout_name,fout_ass_name;

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    if (cmdline.query("-h") != "yes")
    {
        double d_max=std::stod(cmdline.query("-cutoff"));
        fin_name1 = cmdline.query("-in1");
        fin_name2 = cmdline.query("-in2");
        fout_name = cmdline.query("-out");
        fout_ass_name = cmdline.query("-out-ass");

        peak_tools::peak_reading_sparky(fin_name1,peak_pos1,peak_infor1);
        int npeak1=peak_pos1.size(); //with assignment
        
        peak_manipulation peaks; 
        peaks.read_file(fin_name2);

        std::vector<std::string> data_lines;
        /**
         * Spectral region [x_ppm_start, x_ppm_end, y_ppm_start, y_ppm_end]
        */
        std::array<double,4> region;

        region[2] = region[0] =  std::numeric_limits<double>::max()/3.0;
        region[3] = region[1] = -std::numeric_limits<double>::max()/3.0;
        
        peaks.get_data_lines(data_lines);


        for(int i=0;i<data_lines.size();i++)
        {
            std::string p;
            std::vector<std::string> ps;
            std::stringstream ss;
            ss.str(data_lines[i]);

            while(ss>>p)
            {
                ps.push_back(p);
            }

            if(ps[1]=="X_AXIS")
            {
                region[0]=std::stod(ps[5]);
                region[1]=std::stod(ps[6]);
            }  
            else if(ps[1]=="Y_AXIS")
            {
                region[2]=std::stod(ps[5]);
                region[3]=std::stod(ps[6]);   
            }
        }

        double region_width=region[0]-region[1];
        double region_height=region[2]-region[3];
        
        int n_fold=0;
        for(int i=0;i<npeak1;i++)
        {
            bool b=false;
            while(peak_pos1[i][0]<=region[1])
            {
                peak_pos1[i][0]+=region_width;    
                b=true;
            }  
            while(peak_pos1[i][0]>region[0])
            {
                peak_pos1[i][0]-=region_width;   
                b=true;
            } 
            while(peak_pos1[i][1]<=region[3])
            {
                peak_pos1[i][1]+=region_height;    
                b=true;
            }  
            while(peak_pos1[i][1]>region[2])
            {
                peak_pos1[i][1]-=region_height;   
                b=true;
            } 
            if(b==true) n_fold++;
        }
        std::cout<<"Folded "<<n_fold<<" peaks from set 1 into regions defined by set 2."<<std::endl;

        std::vector<std::string> s_x,s_y;
        peaks.get_column(peaks.get_column_index("X_PPM"),s_x);
        peaks.get_column(peaks.get_column_index("Y_PPM"),s_y);

        for(int i=0;i<s_x.size();i++)
        {
            std::array<double,2> t;
            t[0]=std::stod(s_x[i]);
            t[1]=std::stod(s_y[i]);
            peak_pos2.push_back(t);
        }

        int npeak2=peak_pos2.size();
        std::cout<<"npeak1="<<npeak1<<" and npeak2="<<npeak2<<std::endl;
        
        std::vector<std::vector<int>> cost_int;
        std::vector<int> vd, vi;
        std::vector<int> temp;

        double scale= 1000000;

        for(int i=0;i<npeak1;i++)
        {
            temp.clear();
            for(int j=0;j<peak_pos2.size();j++)
            {
                double d1=peak_pos1[i][0]-peak_pos2[j][0];  //proton
                double d2=peak_pos1[i][1]-peak_pos2[j][1];  //carbon or n
                double d=sqrt(d1*d1+d2*d2*0.01);

                if(peak_tools::is_assignment(peak_infor1[i])==false)
                {
                    d=d+d_max/2.0;
                }

                if(d>d_max) d=d_max;
                temp.push_back(int(d*scale));
            }

            for(int j=npeak2;j<npeak1;j++)
            {
                temp.push_back(int(d_max*scale));
            }

            cost_int.push_back(temp);
        }

        for(int i=npeak1;i<npeak2;i++)
        {
            temp.clear();
            temp.resize(npeak2,int(d_max*scale));
            cost_int.push_back(temp);
        }

        peak_tools::MinCostMatching(cost_int, vd, vi);


        std::cout<<std::endl;
        for (int i = 0; i < npeak1; i++)
        {
            if(vd[i]>=npeak2 || cost_int[i][vd[i]]>=int(d_max*scale)) 
            {
                printf("Peak %3d %10s cannot be transfered, coors are %6.2f %6.2f\n", i, peak_infor1[i].c_str(), peak_pos1[i][0], peak_pos1[i][1]);
            }
        }
        std::cout<<std::endl;

        std::vector<std::string> new_assignment, new_assignment_with_old;

        /**
         * Initially empty, will be filled with the assignments from peaks1, if they are matched to peaks2
        */
        new_assignment.resize(npeak2,"");
        /**
         * Initially from peaks2, will be filled with the assignments from peaks1, if they are matched to peaks2.
         * Keep the old assignments if no matching is found.
         * ASS is the column name for assignment in nmrPipe .tab format.
        */
        peaks.get_column(peaks.get_column_index("ASS"),new_assignment_with_old);

        for (int i = 0; i < npeak1; i++)
        {
            if(vd[i]<npeak2 && cost_int[i][vd[i]]<int(d_max*scale)) 
            {
                /**
                 * Formatted output
                */
                printf("Peak %3d %10s match %3d, coors are %6.2f %6.2f and %6.2f %6.2f\n", i, peak_infor1[i].c_str(), vd[i], peak_pos1[i][0], peak_pos1[i][1], peak_pos2[vd[i]][0], peak_pos2[vd[i]][1]);

                //trasfer assignment here from i of peaks1 to vd[i] of peaks2
                new_assignment[vd[i]] = peak_infor1[i];
                new_assignment_with_old[vd[i]] = peak_infor1[i];
            }
        }
        std::cout<<std::endl;

        for(int i=npeak1;i<npeak2;i++)
        {
            printf("No matching found for peak %3d, coors are %6.2f %6.2f\n", vd[i], peak_pos2[vd[i]][0], peak_pos2[vd[i]][1]);
        }
        std::cout << std::endl;

        peaks.operate_on_column(peaks.get_column_index("ASS"),column_operation::REPLACE,"","",new_assignment_with_old);

        peaks.write_file(fout_name);

        /**
         * Print a file with only the assignment column.
         * empty line if not matched.
        */
        std::ofstream fout(fout_ass_name);
        for(int i=0;i<new_assignment.size();i++)
        {
            fout<<new_assignment[i]<<std::endl;
        }
        fout.close();

    }

    return 0;
}