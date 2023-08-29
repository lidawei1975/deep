//#include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <valarray>
#include <string>
#include <cstring>
#include <vector>

#include "commandline.h"
#include "spectrum_io.h"


namespace ldw_math_spectrum_2d
{

    bool SplitFilename (const std::string& str, std::string &path_name, std::string &file_name, std::string &file_name_ext)
    {
        bool b=false;
        std::string file_name_full;
        std::size_t found = str.find_last_of("/\\");
        if(found!=std::string::npos)
        {
            path_name=str.substr(0,found);
            file_name_full=str.substr(found+1);
            b=true;
        }
        else
        {
            path_name=".";   
            file_name_full=str;
        }

        // std::cout<<"file_name_full is "<<file_name_full<<std::endl;

        found = file_name_full.find_last_of(".");
        if(found!=std::string::npos)
        {
            file_name=file_name_full.substr(0,found);
            file_name_ext=file_name_full.substr(found+1);
            b=true;
        }
        else
        {
            file_name_ext="";   
            file_name=file_name_full;
        }

        // std::cout<<"path is "<<path_name<<std::endl;
        // std::cout<<"file_name is "<<file_name<<std::endl;
        // std::cout<<"file_name_ext is "<<file_name_ext<<std::endl;
        

        return b;
    }

    void get_ppm_from_header(const double ref,const double sw,const double frq, double &stop, double &begin)
    {
        if(fabs(frq)>1e-10)
        {
            stop = ref / frq;
            begin = stop + sw / frq;
        }
        else
        {
            stop=2.0;
            begin=1.0; //arbitary
        }
        return;
    }

    double calcualte_median(std::vector<double> scores)
    {
        size_t size = scores.size();

        if (size == 0)
        {
            return 0; // Undefined, really.
        }
        else
        {
            sort(scores.begin(), scores.end());
            if (size % 2 == 0)
            {
                return (scores[size / 2 - 1] + scores[size / 2]) / 2;
            }
            else
            {
                return scores[size / 2];
            }
        }
    };

    void sortArr(std::vector<double> &arr, std::vector<int> &ndx) 
    { 
        std::vector<std::pair<double, int> > vp; 
    
        for (int i = 0; i < arr.size(); ++i) { 
            vp.push_back(std::make_pair(arr[i], i)); 
        } 
    
        std::sort(vp.begin(), vp.end()); 
    
        for (int i = 0; i < vp.size(); i++)
        { 
            ndx.push_back(vp[i].second);
        } 
    };


    bool spline_expand(int xdim, int ydim, float *spect,std::vector<double> &final_data)
    {
        std::vector<double> x_input,y_input,x_output,y_output;
        
        for(int j = 0; j < ydim-1; j++)
        {
            y_input.push_back(j);
            y_output.push_back(j+0.5);
        }
        y_input.push_back(ydim-1);

        for(int i = 0; i < xdim-1; i++)
        {
            x_input.push_back(i);
            x_output.push_back(i+0.5);
        }
        x_input.push_back(xdim-1);

        std::vector<double> intermediate_data;
        for(int j=0;j<ydim;j++)
        {
            std::vector<double> tdata(xdim),t;
            for(int i=0;i<xdim;i++) tdata[i]=spect[j*xdim+i]; 
            tk::spline st(x_input,tdata);
            for(int m=0;m<xdim-1;m++)
            {
                t.push_back(tdata[m]);
                t.push_back(st(x_output[m]));
            }
            t.push_back(tdata[xdim-1]);
            intermediate_data.insert(intermediate_data.end(),t.begin(),t.end()); 
        }
        //At this time, intermediate_data is (2*xdim-1)*ydim, row by row format

        for(int i=0;i<xdim*2-1;i++)
        {
            std::vector<double> tdata(ydim),t;
            for(int j=0;j<ydim;j++) tdata[j]=intermediate_data[j*(2*xdim-1)+i];
            tk::spline st(y_input,tdata);
            for(int m=0;m<ydim-1;m++)
            {
                t.push_back(tdata[m]);
                t.push_back(st(y_output[m]));
            }
            t.push_back(tdata[ydim-1]);
            final_data.insert(final_data.end(),t.begin(),t.end()); 
        }

        return true;
    };
};

spectrum_io::spectrum_io()
{
    noise_level=-0.1; //negative nosie means we don't have it yet
    
    //spectrum range 
    begin1=100;
    stop1=-100;
    begin2=1000;
    stop2=-1000; //will be updated once read in spe

    b_pipe=false;
};

spectrum_io::~spectrum_io()
{
    // if(spect!=NULL) //to be safe
    // {
    //     delete [] spect;
    // }
};



bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}


bool spectrum_io::read_spectrum(std::string infname)
{
    bool b_read=0;
    
    std::string stxt(".txt");
    std::string sft2(".ft2");
    std::string sft3(".ft3");
    std::string sucsf(".ucsf");
    std::string scsv(".csv");

    if(std::equal(stxt.rbegin(), stxt.rend(), infname.rbegin()))
    {
        b_read=read_topspin_txt(infname);
    }
    else if(std::equal(sft2.rbegin(), sft2.rend(), infname.rbegin())){
        b_read=read_pipe(infname);
    }
    else if(std::equal(sft3.rbegin(), sft3.rend(), infname.rbegin())){
        b_read=read_pipe(infname);
    }
    else if(std::equal(sucsf.rbegin(), sucsf.rend(), infname.rbegin())){
        b_read=read_sparky(infname);
    }
    else if(std::equal(scsv.rbegin(), scsv.rend(), infname.rbegin())){
        b_read=read_mnova(infname);
    }
    else{
        b_read=false;
    }

    return b_read;
};



bool spectrum_io::init(std::string infname, int noise_flag)
{

    //std::cout<<"peak_diag is "<<peak_diag<<std::endl;
    //std::cout<<"flag_shoulder is "<<flag_shoulder<<std::endl;
    bool b_read;

    b_read=read_spectrum(infname);

    if(b_read)
    {
        if(fabs(begin2-stop2)>5*fabs(begin1-stop1) || begin2>20.0)
        {
            spectrum_type=hsqc_spectrum;//hsqc
        }
        else
        {
            spectrum_type=tocsy_spectrum;//tocsy
        }
        std::cout<<"Done reading"<<std::endl;

        if(noise_flag==1) 
        {
            // estimate_noise_level_mad();    //estimate noise level using MAD
            estimate_noise_level(); //estimate noise level using region by region standard deviation
        }
    }
    return b_read;
}


bool spectrum_io::save_mnova(std::string outfname)
{
    std::ofstream fout;

    std::vector<double> ppm1,ppm2;

    for(int i=0;i<xdim;i++)
    {
        ppm1.push_back(begin1+i*step1);
    }

    for(int i=0;i<ydim;i++)
    {
        ppm2.push_back(begin2+i*step2);
    }
    
    //first line, ppm along direct dimension
    for(int i=0;i<xdim;i++)
    {
        fout<<ppm1[i]<<" ";
    }
    fout<<std::endl;

    for(int i=0;i<ydim;i++)
    {
        fout<<ppm2[i]; //first element is ppm along indirect dimension
        for(int j=0;j<xdim;j++)
        {
            fout<<" "<<spect[i*xdim+j];
        }
        fout<<std::endl;
    }
    fout.close();

    return true;
}

bool spectrum_io::read_mnova(std::string infname)
{
    int tp;
    std::string line, p;
    std::vector<std::string> ps,lines;
    std::stringstream iss;
    std::ifstream fin(infname.c_str());

    getline(fin,line); //first line
    iss.clear();
    iss.str(line);
    ps.clear();
    while(iss>>p)
    {
        ps.push_back(p);
    }

    if(ps.size()==2) //new format in mnova 14.0
    {
        std::vector<double> x,y,z;

        lines.push_back(line);
        while(getline(fin,line))
        {
            lines.push_back(line);
        }

        std::string p1,p2;
        for(int i=0;i<lines.size();i++)
        {
            iss.clear();
            iss.str(lines[i]);
            iss>>p1;
            iss>>p2;
            z.push_back(atof(p2.c_str()));

            p1.erase(0,1);
            p1.pop_back();
            iss.clear();
            iss.str(p1);
            ps.clear();
            while(std::getline(iss, p, ':'))
            {
                ps.push_back(p);
            } 
            x.push_back(atof(ps[0].c_str()));      
            y.push_back(atof(ps[1].c_str()));      
        }

        std::set<double> xx(x.begin(),x.end());
        std::set<double> yy(y.begin(),y.end());
        
        xdim=xx.size();
        ydim=yy.size();

        begin1=*xx.rbegin();
        stop1=*xx.begin();
        step1=-(begin1-stop1)/(xdim-1);
        begin2=*yy.rbegin();
        stop2=*yy.begin();
        step2=-(begin2-stop2)/(ydim-1);

        std::cout << "Direct dimension size is " << xdim << " indirect dimension is " << ydim << std::endl;
        std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm, stop at " <<stop1<< std::endl;
        std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm, stop at " <<stop2<< std::endl;


        spect=new float[xdim*ydim];

        for(int i=0;i<z.size();i++)
        {
            int k1=round(-(begin1-x[i])/step1);
            int k2=round(-(begin2-y[i])/step2);
            if(k1<0 || k1>=xdim || k2<0 || k2>=ydim)
            {
                std::cout<<"ERROR in read mnova format, k1="<<k1<<" k2="<<k2<<std::endl;
            }
            spect[k2*xdim+k1]=float(z[i]);
            //std::cout<<x[i]<<" "<<k1<<" "<<y[i]<<" "<<k2<<" "<<z[i]<<std::endl;
        }
    }

    else //format in mnova 10.0
    {

        if( fabs(atof(ps[ps.size()-1].c_str()))<0.0000001)
        {
            ps.erase(ps.end()-1); //last element is meaningless for first line if it is zero!!
        }

        tp=ps.size();
        xdim=tp;
        
        float nouse;
        double tswap;
        bool bswap1=0,bswap2=0;

        begin1=atof(ps[0].c_str());
        stop1=atof(ps[tp-1].c_str());

        if(begin1<stop1) {tswap=begin1;begin1=stop1;stop1=tswap;bswap1=1;}

        step1=(stop1-begin1)/(xdim-1);

        while(getline(fin,line))
        {
            if(line.length()>xdim*2-1)
            {
                lines.push_back(line);
            }
        }


        ydim=lines.size();

        float *spect0;
        spect0=new float[xdim*ydim];
        spect=new float[xdim*ydim];
        for(int i=0;i<lines.size();i++)
        {
            iss.clear();
            iss.str(lines[i]);
            if(i==0) iss>>begin2;
            else if(i==ydim-1) iss>>stop2;
            else iss>> nouse;

            for(int j=0;j<xdim;j++)
            {
                iss>>nouse;
                spect0[i*xdim+j]=nouse;
            }
        }
        if(begin2<stop2) {tswap=begin2;begin2=stop2;stop2=tswap;bswap2=1;}
        step2=(stop2-begin2)/(ydim-1);

        if(bswap2==0)
        {
            for(int i=0;i<ydim;i++)
            {
                for(int j=0;j<xdim;j++)
                {
                    if(bswap1==1)
                        spect[i*xdim+xdim-1-j]=spect0[i*xdim+j];
                    else
                        spect[i*xdim+j]=spect0[i*xdim+j];
                }
            }   
        }
        else
        {
            for(int i=0;i<ydim;i++)
            {
                for(int j=0;j<xdim;j++)
                {
                    if(bswap1==1)
                        spect[(ydim-1-i)*xdim+xdim-1-j]=spect0[i*xdim+j];
                    else
                        spect[(ydim-1-i)*xdim+j]=spect0[i*xdim+j];
                }
            }   
        }
        delete [] spect0;
    }

    

    //aribitary frq becuase we don't have that infor
    frq1=frq2=850.0;
    SW1=frq1*(begin1-stop1);
    SW2=frq2*(begin2-stop2);
    ref1=stop1*frq1;
    ref2=stop2*frq2;

    std::cout << "Direct dimension size is " << xdim << " indirect dimension is " << ydim << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm" << std::endl;


    return true;
}

bool spectrum_io::get_ppm_from_point()
{
    //get ppm
    p1_ppm.clear();
    p2_ppm.clear();

    for (unsigned int i = 0; i < p1.size(); i++)
    {
        double f1 = begin1 + step1 * (p1[i]);  //direct dimension
        double f2 = begin2 + step2 * (p2[i]);  //indirect dimension
        p1_ppm.push_back(f1);
        p2_ppm.push_back(f2);
    }

    return true;
};


//read topspin file in ASCIC format, genearted using command totxt
bool spectrum_io::read_topspin_txt(std::string infname)
{

    int tp;
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;
    std::ifstream fin(infname.c_str());

    if(!fin)
    {
        std::cout<<"Can't open "<<infname<<" to read."<<std::endl;
        return false;
    }

    double f1left,f1right,f2left,f2right;

    f1left=f1right=f2left=f2right=0.0;
    xdim=ydim=0;
    

    //read in head information
    while( getline(fin, line) && (xdim==0 || ydim==0) )
    {
        if (line.find("F1LEFT")!=std::string::npos && line.find("F1RIGHT")!=std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp=ps.size();
            if(tp>=9)
            {
                f1left=atof(ps[3].c_str());
                f1right=atof(ps[7].c_str());
            }
        }

        if (line.find("F2LEFT")!=std::string::npos && line.find("F2RIGHT")!=std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp=ps.size();
            if(tp>=9)
            {
                f2left=atof(ps[3].c_str());
                f2right=atof(ps[7].c_str());
            }
        }

        if (line.find("NROWS")!=std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp=ps.size();
            if(tp>=4)
            {
                ydim=atoi(ps[3].c_str());
            }
        }

        if (line.find("NCOLS")!=std::string::npos)
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            tp=ps.size();
            if(tp>=4)
            {
                xdim=atoi(ps[3].c_str());
            }
        }

    }



    spect=new float[xdim*ydim];


    int row_index=-1;
    int flag=0;
    int col_index;

    while( getline(fin, line))
    {
        if(line.find("# row = ")!=std::string::npos)
        {
            row_index=atoi(line.substr(7).c_str());
            col_index=0;
            continue;
        }
        else if(line.find("#")!=std::string::npos)
        {
            continue;
        }

        spect[col_index+row_index*xdim]=atof(line.c_str());
        col_index++;

    }

    begin1=f2left;
    stop1=f2right;
    step1=(stop1-begin1)/(xdim);
    stop1=f2right+step1; //stop is the ppm of the last col. 


    begin2=f1left;
    stop2=f1right;
    step2=(stop2-begin2)/(ydim);
    stop2=f1right+step2; //stop is the ppm of the last col. 

    //fill in required variable to save pipe format
    // set frq= 850. This is arbitary
    frq1=frq2=850.0;
    SW1=frq1*(begin1-stop1);
    SW2=frq2*(begin2-stop2);
    ref1=stop1*frq1;
    ref2=stop2*frq2;
    


    std::cout << "Direct dimension size is " << xdim << " indirect dimension is " << ydim << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm" << std::endl;

    return true;
}


bool spectrum_io::read_txt(std::string infname)
{
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;
    std::vector<float> temp;

    std::ifstream fin(infname.c_str());

    if(!fin)
    {
        std::cout<<"Can't open "<<infname<<" to read."<<std::endl;
        return false;
    }

    xdim=ydim=0;
    while( getline(fin, line))
    {
        ydim++;
        iss.clear();
        iss.str(line);
        ps.clear();
        while (iss >> p)
        {
            temp.push_back(atof(p.c_str()));
        }
        if(xdim==0)
        {
            xdim=temp.size();
        }
    }

    spect=new float[xdim*ydim];
    for(unsigned int i=0;i<xdim*ydim;i++)
    {
        spect[i]=temp[i];
    }

    begin1=0.6;
    stop1=0;
    step1=(stop1-begin1)/(xdim);
    stop1=stop1-step1; //stop is the ppm of the last col. 


    begin2=0.6;
    stop2=0;
    step2=(stop2-begin2)/(ydim);
    stop2=stop2-step2; //stop is the ppm of the last col. 

    return true;

}


bool spectrum_io::read_pipe(std::string infname)
{
    b_pipe=1;

    FILE *fp;
    fp = fopen(infname.c_str(), "rb");
    if (fp == NULL)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }
    unsigned int temp = fread(header, sizeof(float), 512, fp);
    if (temp != 512)
    {
        std::cout << "Wrong file format, can't read 2048 bytes of head information from " << infname << std::endl;
        return false;
    }

    bool b_transpose = int(header[221]) == 1;  //0 normal, 1: transpose

    if(b_transpose==0 && int(header[24])!=2)
    {
        std::cout<<" First dimension should always be dimension 2 in non-transposed spectrum, it is "<<int(header[24])<<std::endl;
        return false;
    }
    
    if(b_transpose==1 && int(header[24])!=1)
    {
        std::cout<<" First dimension should always be dimension 1 in transposed spectrum, it is "<<int(header[24])<<std::endl;
        return false;
    }

    int dims[4];
    double refs[4], sws[4], frqs[4];
    double begins[4],stops[4];

    sws[1] = double(header[101 - 1]);
    frqs[1] = double(header[120 - 1]);
    refs[1] = double(header[102 - 1]);

    sws[0] = double(header[230 - 1]);
    frqs[0] = double(header[219 - 1]);
    refs[0] = double(header[250 - 1]);

    sws[2] = double(header[12 - 1]);
    frqs[2] = double(header[11 - 1]);
    refs[2] = double(header[13 - 1]);

    sws[3] = double(header[30 - 1]);
    frqs[3] = double(header[29 - 1]);
    refs[3] = double(header[31 - 1]);


    ydim = int(header[220 - 1]);
    xdim = int(header[100 - 1]);

    int ndim=int(header[10-1]);
    for(int i=0;i<4;i++)
    {
        ldw_math_spectrum_2d::get_ppm_from_header( refs[i], sws[i], frqs[i],  stops[i],  begins[i]);
    }

    //X is always the 2rd dimension
    int direct_ndx=int(header[24])-1;

    //Z can be 4 or 3
    int indirect_ndxz=int(header[26])-1;  //C start from 0,not 1
    
    //Y can be 3rd or 1st dimension
    int indirect_ndx=int(header[25])-1;  //C start from 0,not 1

    if(b_transpose==1)
    {
        std::swap(xdim,ydim);
        std::swap(direct_ndx,indirect_ndx);
    }

    begin1=begins[direct_ndx];
    stop1=stops[direct_ndx];
    step1=(stop1-begin1)/xdim;
    begin1+=step1;
    
    begin3=begins[indirect_ndxz];
    stop3=stops[indirect_ndxz];
    //we don't have zdim or step3

    begin2=begins[indirect_ndx];
    stop2=stops[indirect_ndx];
    step2=(stop2-begin2)/ydim;
    begin2+=step2;

    double sws_direct=sws[direct_ndx];
    double sws_indirect=sws[indirect_ndx];
    double frqs_direct=frqs[direct_ndx];
    double frqs_indirect=frqs[indirect_ndx];

    //read in spectrum
    //saved row by row in nmrpipe
    spect = new float[xdim * ydim];

    if(b_transpose==0)
    {
        for (unsigned int i = 0; i < ydim; i++)
        {
            unsigned int temp;
            temp = fread(spect + i * xdim, sizeof(float), xdim, fp);
            if (temp != xdim)
                return false;
        }
    }
    else
    {
        for (unsigned int i = 0; i < xdim; i++)
        {
            for (unsigned int j = 0; j < ydim; j++)
            {
                unsigned int temp;
                temp = fread(spect + j * xdim + i, sizeof(float), 1, fp);
                if (temp != 1)
                    return false;
            }
        }
    }

    SW1=sws_direct;
    SW2=sws_indirect; //copy to member variables

    frq1=frqs_direct;
    frq2=frqs_indirect; //copy to member variables


    std::cout << "Spectrum width are " << SW1 << " Hz and " << SW2 << " Hz" << std::endl;
    std::cout << "Fields are " << frq1 << " mHz and " << frq2 << " mHz" << std::endl;
    std::cout << "Direct dimension size is " << xdim << " indirect dimension is " << ydim << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " and last is " << stop1 << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per step is " << step2 << " and last is " << stop2 << std::endl;

    fclose(fp);

    return true;
};

//This fucntion only write a ft2 file, for which deep pickerfitter and nmrdraw can read.
//It is not a 100% compatible ft2 file. 

bool spectrum_io::write_pipe(std::vector<std::vector<float> > spect, std::string fname)
{

    if(b_pipe==false) //otherwise we just use the input header 
    {
        header[0]=0.0f;
        header[219]=float(ydim); 
        header[99]=float(xdim); 
        header[106]=float(1); 
        header[256]=float(2); 

        
        header[9] = 2; //dimension is 2
        header[57] = 0.0f;  //2d spectrum
        header[221] = 0.0f; //not transposed
        header[24] = 2.0f;  //first dimension is 2
        header[25] = 1.0f;  //second dimension is 1
        header[26] = 3.0f;  //z dimension is 3
        header[27] = 4.0f;  //A dimension is 4
        


        header[101 - 1] = (float)SW1;  //second dimension
        header[120 - 1] = (float)frq1;
        header[102 - 1] = (float)ref1;

        header[230 - 1] = (float)SW2; //first dimension
        header[219 - 1] = (float)frq2;
        header[250 - 1] = (float)ref2;
    }
    
   
    FILE *fp = fopen(fname.c_str(), "w");
    if(fp==NULL)
    {
        std::cout<<"cannot open file "<<fname.c_str()<<"to write"<<std::endl;
    }
    else
    {
        fwrite(header, sizeof(float), 512, fp);
        for (unsigned int i = 0; i < ydim; i++)
        {
            fwrite(spect[i].data(), sizeof(float), xdim, fp);
        }
        fclose(fp);
    }

    return true;
};



float spectrum_io::read_float(FILE *fp)
{
    char buff[4];
    fread(buff,4,1,fp); // dimension
    std::swap(buff[0],buff[3]);
    std::swap(buff[1],buff[2]);
    return *((float*)buff);
};

bool spectrum_io::read_float(FILE *fp, int n, float *pf)
{
    fread(pf,4,n,fp);
    char *buff = (char *) pf;
    for(int i=0;i<n;i++)
    {
        std::swap(buff[0+i*4],buff[3+i*4]);
        std::swap(buff[1+i*4],buff[2+i*4]);   
    }
    return true;
}

int spectrum_io::read_int(FILE *fp)
{
    char buff[4];
    fread(buff,4,1,fp); // dimension
    std::swap(buff[0],buff[3]);
    std::swap(buff[1],buff[2]);
    return *((int*)buff);
};


bool spectrum_io::read_sparky(std::string infname)
{
    FILE *fp;

    char buffer[10];
    int temp;


    fp = fopen(infname.c_str(), "rb");
    
    if (fp == NULL)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }

    fread(buffer, 1, 10, fp);
    //std::cout<<buffer<<std::endl;
    
    fread(buffer, 1, 1, fp);
    temp=int(buffer[0]);
    if(temp!=2) {std::cout<<"Error in sparky format file, dimension is not 2"<<std::endl; return false;}

    fread(buffer, 1, 1, fp);
    fseek(fp,1,SEEK_CUR);
    temp=int(buffer[0]);
    if(temp!=1) {std::cout<<"Error in sparky format file, it is not in real data"<<std::endl; return false;}

    fread(buffer, 1, 1, fp);
    //std::cout<<"Version is "<< int(buffer[0])<<std::endl;
    fseek(fp,166,SEEK_CUR);

    //read 2d header

    float center1, center2;
    int tile1,tile2;


    fread(buffer, 1, 6, fp); //nuleus name
    std::cout<<"Indirect dimension nuleus "<<buffer<<std::endl;
    fseek(fp,2,SEEK_CUR);    
    ydim=read_int(fp);
    fseek(fp,4,SEEK_CUR); 
    tile2=read_int(fp);
    frq2=read_float(fp);
    SW2=read_float(fp);
    center2=read_float(fp);
    fseek(fp,96,SEEK_CUR); 
    
    fread(buffer, 1, 6, fp); //nuleus name
    std::cout<<"Direct dimension nuleus "<<buffer<<std::endl;
    fseek(fp,2,SEEK_CUR);    
    xdim=read_int(fp);
    fseek(fp,4,SEEK_CUR); 
    tile1=read_int(fp);
    frq1=read_float(fp);
    SW1=read_float(fp);
    center1=read_float(fp);
    fseek(fp,96,SEEK_CUR); 


    //read in data here
    spect = new float[xdim * ydim];

    int ntile1=int(ceil((double(xdim))/tile1));
    int ntile2=int(ceil((double(ydim))/tile2));

    int last_tile1=xdim%tile1;
    if(last_tile1==0) last_tile1=tile1;

    

    float * float_buff;
    float_buff=new float[tile1];
    
    for(int i=0;i<ntile2;i++)
    {
        for(int j=0;j<ntile1;j++)
        {
            for(int m=0;m<tile2;m++)
            {
                read_float(fp,tile1,float_buff);
                if(i*tile2+m<ydim)
                {
                    if (j==ntile1-1)
                        memcpy(spect+(i*tile2+m)*xdim+(j*tile1),float_buff,last_tile1*4);
                    else
                        memcpy(spect+(i*tile2+m)*xdim+(j*tile1),float_buff,tile1*4);
                }
            }
        }
    }

    delete [] float_buff;


    float range1=SW1/frq1;
    step1=-range1/xdim;
    begin1=center1+range1/2;
    stop1=center1-range1/2;

    float range2=SW2/frq2;
    step2=-range2/ydim;
    begin2=center2+range2/2;
    stop2=center2-range2/2;

    //file in ref from center, so that we can save in pipe format if required.
    ref1=center1*frq1-SW1/2;
    ref2=center2*frq2-SW2/2;
    

    std::cout << "Spectrum width are " << SW1 << " Hz and " << SW2 << " Hz" << std::endl;
    std::cout << "Fields are " << frq1 << " mHz and " << frq2 << " mHz" << std::endl;
    std::cout << "Direct dimension size is " << xdim << " indirect dimension is " << ydim << std::endl;
    std::cout << "  Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;
    std::cout << "Indirect dimension offset is " << begin2 << ", ppm per steo is " << step2 << " ppm" << std::endl;

    fclose(fp);

    return true;

};


void spectrum_io::estimate_noise_level_mad()
{
    std::cout<<"In noise estimation, xdim*ydim is "<<xdim*ydim<<std::endl;

    std::vector<float> t(spect, spect+xdim*ydim);
    

    for(unsigned int i = 0; i < t.size(); i++)
    {
        if(t[i] < 0)
            t[i] *= -1;
    }

    std::vector<float> scores=t;

    sort(scores.begin(), scores.end());
    noise_level = scores[scores.size() / 2]*1.4826;
    if(noise_level<=0.0) noise_level=0.1; //artificail spectrum w/o noise


    // noise_level=87353.0;  
    // std::cout<<"ERROR: set noise level to "<<noise_level<<std::endl;

    std::cout<<"First round, noise level is "<<noise_level<<std::endl;

    std::vector<int> flag(xdim*ydim,0); //flag 

    for(int j=0;j<ydim;j++)
    {
        for(int i=0;i<xdim;i++)
        {
            if(t[j*xdim+i]>5.5*noise_level)
            {
                int ystart=std::max(j-5,0);
                int yend=std::min(j+6,ydim);
                int xstart=std::max(i-5,0);
                int xend=std::min(i+6,xdim);
                for(int m=ystart;m<yend;m++)
                {
                    for(int n=xstart;n<xend;n++)
                    {
                        flag[m*xdim+n]=1;
                    }
                }
            }
        }
    }
    //after this, all datapoint > 5.5*noise and their surrounding datapoint are labeled as 1.
    scores.clear();
    for(int j=0;j<ydim;j++)
    {
        for(int i=0;i<xdim;i++)
        {
            if(flag[j*xdim+i]==0)
            {
                scores.push_back(t[j*xdim+i]);
            }
        }
    }
    sort(scores.begin(), scores.end());
    noise_level = scores[scores.size() / 2]*1.4826;
    if(noise_level<=0.0) noise_level=0.1; //artificail spectrum w/o noise
    std::cout<<"Final noise level is estiamted to be "<<noise_level<<std::endl;



    //estimate noise level column by column for TOCSY t1 noise belt identification!!
    for(int i=0;i<xdim;i++)
    {
        std::vector<float> scores;
        scores.clear();
        for(int j=0;j<ydim;j++)
        {
            scores.push_back(fabs(spect[j*xdim+i]));   
        }
        sort(scores.begin(), scores.end());
        noise_level_columns.push_back(scores[ydim/3]*1.4826);
    }

    //estimate noise level row by row 
    for(int j=0;j<ydim;j++)
    {
        std::vector<float> scores;
        scores.clear();
        for(int i=0;i<xdim;i++)
        {
            scores.push_back(fabs(spect[j*xdim+i]));   
        }
        sort(scores.begin(), scores.end());
        noise_level_rows.push_back(scores[xdim/3]*1.4826);
    }

};

void spectrum_io::estimate_noise_level()
{
    std::cout<<"In noise estimation, xdim*ydim is "<<xdim*ydim<<std::endl;

    int n_segment_x = xdim / 32;
    int n_segment_y = ydim / 32;

    std::vector<float> variances; //variance of each segment
    std::vector<float> maximal_values; //maximal value of each segment

    /**
     * loop through each segment, and calculate variance
    */
    for(int i=0;i<n_segment_x;i++)
    {
        for(int j=0;j<n_segment_y;j++)
        {
            std::vector<float> t;
            for(int m=0;m<32;m++)
            {
                for(int n=0;n<32;n++)
                {
                    t.push_back(spect[(j*32+m)*xdim+i*32+n]);
                }
            }
        
            /**
             * calculate variance of this segment. Substract the mean value of this segment first
             * also calculate the max value of this segment
             */
            float max_of_t = 0.0f;
            float mean_of_t = 0.0f;
            for (int k = 0; k < t.size(); k++)
            {
                mean_of_t += t[k];
                if (fabs(t[k]) > max_of_t)
                {
                    max_of_t = fabs(t[k]);
                }
            }
            mean_of_t /= t.size();

            float variance_of_t = 0.0f;
            for (int k = 0; k < t.size(); k++)
            {
                variance_of_t += (t[k] - mean_of_t) * (t[k] - mean_of_t);
            }
            variance_of_t /= t.size();
            variances.push_back(variance_of_t);
            maximal_values.push_back(max_of_t);
        }
    }
    

    /**
     * sort the variance, and get the median value
     */
    std::vector<float> variances_sorted = variances;
    sort(variances_sorted.begin(), variances_sorted.end());
    noise_level = sqrt(variances_sorted[variances_sorted.size() / 2]);
    std::cout<<"Noise level is "<<noise_level<<" using variance estimation."<<std::endl;

    /**
     * loop through maximal_values, remove the ones that are larger than 10.0*noise_level
     * remove the corresponding variance as well
     */
    for(int i=maximal_values.size()-1;i>=0;i--)
    {
        if(maximal_values[i]>10.0*noise_level)
        {
            maximal_values.erase(maximal_values.begin()+i);
            variances.erase(variances.begin()+i);
        }
    }

    /**
     * sort the variance, and get the median value
     */
    variances_sorted = variances;
    sort(variances_sorted.begin(), variances_sorted.end());
    noise_level = sqrt(variances_sorted[variances_sorted.size() / 2]);

    std::cout<<"Final noise level is estiamted to be "<<noise_level<<std::endl;
    
    return;
}