//#include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>

#include "commandline.h"

bool is_assignment(std::string ass)
{
    if (ass.find("?") == 0 || ass.find("Peak") == 0 || ass.find("peak") == 0 || ass.find("None") == 0 || ass.find("none") == 0 || ass.find("X") || ass.find("x") == 0)
    {
        return false;
    }
    else
    {
        return true;
    }
};

typedef std::vector<int> VD;
typedef std::vector<VD> VVD;
typedef std::vector<int> VI;

void MinCostMatching(const VVD &cost, VI &Lmate, VI &Rmate)
{
    int n = int(cost.size());

    // construct dual feasible solution
    VD u(n);
    VD v(n);
    for (int i = 0; i < n; i++)
    {
        u[i] = cost[i][0];
        for (int j = 1; j < n; j++)
            u[i] = std::min(u[i], cost[i][j]);
    }
    for (int j = 0; j < n; j++)
    {
        v[j] = cost[0][j] - u[0];
        for (int i = 1; i < n; i++)
            v[j] = std::min(v[j], cost[i][j] - u[i]);
    }

    // construct primal solution satisfying complementary slackness
    Lmate = VI(n, -1);
    Rmate = VI(n, -1);
    int mated = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (Rmate[j] != -1)
                continue;
            if (std::fabs(cost[i][j] - u[i] - v[j]) < 1e-10)
            {
                Lmate[i] = j;
                Rmate[j] = i;
                mated++;
                break;
            }
        }
    }

    VD dist(n);
    VI dad(n);
    VI seen(n);

    // repeat until primal solution is feasible
    while (mated < n)
    {

        // find an unmatched left node
        int s = 0;
        while (Lmate[s] != -1)
            s++;

        // initialize Dijkstra
        fill(dad.begin(), dad.end(), -1);
        fill(seen.begin(), seen.end(), 0);
        for (int k = 0; k < n; k++)
            dist[k] = cost[s][k] - u[s] - v[k];

        int j = 0;
        while (true)
        {

            // find closest
            j = -1;
            for (int k = 0; k < n; k++)
            {
                if (seen[k])
                    continue;
                if (j == -1 || dist[k] < dist[j])
                    j = k;
            }
            seen[j] = 1;

            // termination condition
            if (Rmate[j] == -1)
                break;

            // relax neighbors
            const int i = Rmate[j];
            for (int k = 0; k < n; k++)
            {
                if (seen[k])
                    continue;
                const double new_dist = dist[j] + cost[i][k] - u[i] - v[k];
                if (dist[k] > new_dist)
                {
                    dist[k] = new_dist;
                    dad[k] = j;
                }
            }
        }

        // update dual variables
        for (int k = 0; k < n; k++)
        {
            if (k == j || !seen[k])
                continue;
            const int i = Rmate[k];
            v[k] += dist[k] - dist[j];
            u[i] -= dist[k] - dist[j];
        }
        u[s] += dist[j];

        // augment along path
        while (dad[j] >= 0)
        {
            const int d = dad[j];
            Rmate[j] = Rmate[d];
            Lmate[Rmate[j]] = j;
            j = d;
        }
        Rmate[j] = s;
        Lmate[s] = j;

        mated++;
    }
}

/*
int test()
{

    VVD cost_int;
    VD vd, vi;
    VD temp;

    temp.clear();
    temp.push_back(10);
    temp.push_back(0);
    temp.push_back(10);
    cost_int.push_back(temp);

    temp.clear();
    temp.push_back(10);
    temp.push_back(10);
    temp.push_back(0);
    cost_int.push_back(temp);

    temp.clear();
    temp.push_back(0);
    temp.push_back(10);
    temp.push_back(10);
    cost_int.push_back(temp);

    // temp.clear();
    // temp.push_back(1);
    // temp.push_back(10);
    // temp.push_back(10);
    // cost_int.push_back(temp);


    MinCostMatching(cost_int, vd, vi);


    for (int i = 0; i < cost_int.size(); i++)
    {
        for (int j = 0; j < cost_int[i].size(); j++)
        {
            std::cout << cost_int[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout<<"VD is "<<std::endl;
    for (int i = 0; i < vd.size(); i++)
    {
        std::cout << vd[i] << " ";
    }
    std::cout << std::endl;

    std::cout<<"VI is "<<std::endl;
    for (int i = 0; i < vi.size(); i++)
    {
        std::cout << vi[i] << " ";
    }
    std::cout << std::endl;

    return 0;
};
*/

void peak_reading
(
    std::string fname,
    std::vector<std::string> &header,
    std::vector<std::string> &lines, 
    int peak_pos_ndx, 
    int peak_pos_ndx2, 
    std::vector<std::array<double,2>> &peak_pos, 
    int peak_infor_ndx, 
    std::vector<std::string> &peak_info,
    std::array<double,4> &region
)
{
    bool b_data=false;
    std::string line,p;
    std::vector< std::string> ps;
    std::stringstream iss;
    std::ifstream fin(fname);


    region={1e10,-1e10,1e10,-1e10}; //default if DATA line is not present
   
    while(getline(fin,line))
    {
        iss.clear();
        iss.str(line);
        ps.clear();
        while(iss>>p)
        {
            ps.push_back(p);
        }

        if(ps.size()<3) continue; //empty line??
        else if(ps[0]=="REMARK") continue; //remark lines
        else if(ps[0]=="VARS") //header lines
        {
            header.push_back(line);
            //scan for x_ppm and ass
            for(int i=1;i<ps.size();i++)
            {
                if(ps[i]=="X_PPM")
                {
                    peak_pos_ndx=i-1;   
                }
                else if(ps[i]=="Y_PPM")
                {
                    peak_pos_ndx2=i-1;   
                }
                else if(ps[i]=="ASS")
                {
                    peak_infor_ndx=i-1;   
                }
            }
            b_data=true;
        }
        else if(ps[0]=="FORMAT") //format line
        {
            header.push_back(line);   
        }
        else if(ps[0]=="DATA")
        {
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
        else if(b_data==true)//data line
        {
            std::array<double,2> t;
            t[0]=atof(ps[peak_pos_ndx].c_str());
            t[1]=atof(ps[peak_pos_ndx2].c_str());
            peak_pos.push_back(t);
            peak_info.push_back(ps[peak_infor_ndx]);
            lines.push_back(line);
        }
    }
    return;
};

bool peak_reading_sparky(std::string fname,    std::vector<std::array<double,2>> &peak_pos, std::vector<std::string> &peak_info)
{
    std::string line,p;
    std::vector< std::string> ps;
    std::stringstream iss;

    int xpos=-1;
    int ypos=-1;
    int ass=-1;

    std::ifstream fin(fname);

    if(!fin) return false;


    bool b_data=false;
    while(getline(fin,line))
    {
        iss.clear();
        iss.str(line);
        ps.clear();
        while(iss>>p)
        {
            ps.push_back(p);
        }
        if(ps.size()<3) continue; //empty line??
        
        if(ps[0]=="Assignment" || ps[0]=="w2" || ps[0]=="w1" )
        {
            for(int i=0;i<ps.size();i++)
            {
                if(ps[i]=="w2") {xpos=i;}  //in sparky, w2 is direct dimension
                else if(ps[i]=="w1") {ypos=i;}
                else if(ps[i]=="Assignment") {ass=i;}   
            }
            b_data=true;
            continue;
        }

        if(b_data==true)
        {
            std::array<double,2> t;
            t[0]=stod(ps[xpos]);
            t[1]=stod(ps[ypos]);
            peak_pos.push_back(t);
            peak_info.push_back(ps[ass]);
        } 
    }
    return true;
}



int main(int argc, char **argv)
{
    
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");
    
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

    args.push_back("-n");
    args2.push_back("1");
    args3.push_back("Every peak in input 1 can match either 1 or 2 peak(s) in input 2");
   
    std::vector<std::string> header,header1;
    std::vector<std::string> lines,lines1; 
    std::vector<std::array<double,2>> peak_pos1, peak_pos2;
    int peak_pos_ndx1,peak_pos_ndx12,peak_infor_ndx1,peak_pos_ndx2,peak_pos_ndx22,peak_infor_ndx2;
    std::vector<std::string> peak_infor1,peak_infor2;
    std::array<double,4> region1,region2;

    std::string fin_name1,fin_name2,fout_name;

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    if (cmdline.query("-h") != "yes")
    {
        double d_max=std::stod(cmdline.query("-cutoff"));
        fin_name1 = cmdline.query("-in1");
        fin_name2 = cmdline.query("-in2");
        fout_name = cmdline.query("-out");

        std::string stab(".tab");
        std::string slist(".list");
        
        if(std::equal(stab.rbegin(), stab.rend(), fin_name1.rbegin()))
        {
            peak_reading(fin_name1,header1,lines1,peak_pos_ndx1,peak_pos_ndx12,peak_pos1,peak_infor_ndx1,peak_infor1,region1);
        }
        else if(std::equal(slist.rbegin(), slist.rend(), fin_name1.rbegin()))
        {
            peak_reading_sparky(fin_name1,peak_pos1,peak_infor1);
        }
        else
        {
            return 1;
        }
        
        peak_reading(fin_name2,header,lines,peak_pos_ndx2,peak_pos_ndx22,peak_pos2,peak_infor_ndx2,peak_infor2,region2);
        


        int npeak1=peak_pos1.size(); //with assignment
        int npeak2=peak_pos2.size();
        
        std::cout<<"npeak1="<<npeak1<<" and npeak2="<<npeak2<<std::endl;


        
        double region_width=region2[0]-region2[1];
        double region_height=region2[2]-region2[3];
        
        int n_fold=0;
        for(int i=0;i<npeak1;i++)
        {
            bool b=false;
            while(peak_pos1[i][0]<=region2[1])
            {
                peak_pos1[i][0]+=region_width;    
                b=true;
            }  
            while(peak_pos1[i][0]>region2[0])
            {
                peak_pos1[i][0]-=region_width;   
                b=true;
            } 
            while(peak_pos1[i][1]<=region2[3])
            {
                peak_pos1[i][1]+=region_height;    
                b=true;
            }  
            while(peak_pos1[i][1]>region2[2])
            {
                peak_pos1[i][1]-=region_height;   
                b=true;
            } 
            if(b==true) n_fold++;
        }
        std::cout<<"Folded "<<n_fold<<" peaks from set 1 into regions defined by set 2."<<std::endl;

        // std::cout<<"peaks1:"<<std::endl;
        // for(int i=0;i<npeak1;i++)
        // {
        //     std::cout<<peak_pos1[i][0]<<" "<<peak_pos1[i][1]<<std::endl;
        // }
        // std::cout<<std::endl;
        // std::cout<<"peaks1:"<<std::endl;
        // for(int i=0;i<npeak2;i++)
        // {
        //     std::cout<<peak_pos2[i][0]<<" "<<peak_pos2[i][1]<<std::endl;
        // }
        // std::cout<<std::endl;
        

        int n=atoi( cmdline.query("-n").c_str());
        if(n==2)
        {
            std::cout<<"Each peak in peak file 1 can match 2 peaks in peak file 2."<<std::endl;
            std::vector<std::array<double,2>> t1=peak_pos1;
            peak_pos1.insert( peak_pos1.end(), t1.begin(), t1.end() );
            std::vector<std::string> temp=peak_infor1;
            peak_infor1.insert( peak_infor1.end(), temp.begin(), temp.end() );
            npeak1=peak_pos1.size();
            std::cout<<"npeak1="<<npeak1<<" and npeak2="<<npeak2<<std::endl;
        }

        
        VVD cost_int;
        VD vd, vi;
        VD temp;

        double scale= 1000000;


        for(int i=0;i<npeak1;i++)
        {
            temp.clear();
            for(int j=0;j<peak_pos2.size();j++)
            {
                double d1=peak_pos1[i][0]-peak_pos2[j][0];  //proton
                double d2=peak_pos1[i][1]-peak_pos2[j][1];  //carbon or n
                double d=sqrt(d1*d1+d2*d2*0.01);

                if(is_assignment(peak_infor1[i])==false)
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

        MinCostMatching(cost_int, vd, vi);

        std::vector< std::vector<int> > used(npeak1/n);


        for (int i = 0; i < npeak1; i++)
        {
            if(vd[i]>=npeak2 || cost_int[i][vd[i]]>=int(d_max*scale)) 
            {
                std::cout<<"Peak "<<i<<" "<<peak_infor1[i]<<" coors are "<<peak_pos1[i][0]<<" "<<peak_pos1[i][1]<<" cannot be transfered."<<std::endl;
            }
            
            else
            {
                std::cout << "Peak " << i <<" "<<peak_infor1[i] << " match " << vd[i] << ", coors are " << peak_pos1[i][0] << " " << peak_pos1[i][1];
                std::cout << " and " << peak_pos2[vd[i]][0] << " " << peak_pos2[vd[i]][1];
                std::cout << " cost is " << cost_int[i][vd[i]] << std::endl;

                //trasfer assignment here from i of peaks1 to vd[i] of peaks2

                std::size_t found = lines[vd[i]].find(peak_infor2[vd[i]]);
                int infor_length = peak_infor2[vd[i]].length();

                std::string part1 = lines[vd[i]].substr(0, found);
                std::string part2 = lines[vd[i]].substr(found + infor_length);

                lines[vd[i]] = part1 + peak_infor1[i] + part2;

                int ii = i;
                if (n == 2)
                {
                    if (ii >= npeak1 / 2)
                    {
                        ii -= npeak1 / 2;
                    }
                }
                used[ii].push_back(vd[i]);
            }
        }

        for(int i=npeak1;i<npeak2;i++)
        {
            std::cout<<"Peak "<<-1<<" match "<<vd[i]<<", coors are "<<-1<<" "<<-1;
            std::cout<<" and "<<peak_pos2[vd[i]][0]<<" "<<peak_pos2[vd[i]][1];
            std::cout<<" cost is "<<cost_int[i][vd[i]]<<std::endl;  
        }

        std::cout << std::endl;

        std::vector<int> removal(npeak2,0);
        //two peak match 1
        if(n==2)
        {
            for(int i=0;i<used.size();i++)
            {
                if(used[i].size()>1)
                {
                    removal[used[i][1]]=1;
                }
            }
        }


        //output
        std::ofstream fout(fout_name);
        fout<<header[0]<<std::endl;
        fout<<header[1]<<std::endl;
        
        for(int i=0;i<npeak2;i++)
        {
            if(removal[i]==0)
            {
                fout<<lines[i]<<std::endl;
            }
        }
        fout.close();

    }

    return 0;
}