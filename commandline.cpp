#include <string>
#include <cstring>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "commandline.h"

bool CCommandline::pharse_core(int argc, char ** argv)
{
    int i, j;
    for (i = 1; i < argc; i++)
    {
        bool b=false;
        for (j = 0; j < narg; j++)
        {
            if (arguments.at(j).compare(argv[i]) == 0) 
            {
                parameters.at(j) = ""; 
                if(i+1>=argc || argv[i + 1][0] == '-')
                {
                    parameters.at(j) = "yes";
                }
                while (i + 1 < argc && argv[i + 1][0] != '-') 
                {
                    parameters.at(j).append(argv[i + 1]);
                    parameters.at(j).append(" ");
                    i++;
                }    
                b=true;
                continue;                
            }
        }
        
        if(b==false)
        {
            std::cout<<"ERROR, unrecognized commandline argument "<<argv[i]<<std::endl;
            return false;
        }
    }
    return true;
}


bool CCommandline::pharse(int argc, char **argv)
{
    
    if (pharse_core(argc, argv) == false)
    {
        return false;
    }

    //check to see whether -f is set or not
    std::string arguments_file=query("-f");
    if(arguments_file != "none")
    {
        std::ifstream fin(arguments_file);
        if(!fin)
        {
            return true;
        }
        std::vector<std::string> ps;
        std::string s;

        //simulte commandline arguments, first argument is the program name
        ps.push_back(argv[0]);

        while(fin>>s)
        {
            ps.push_back(s);
        }
        fin.close();

        //convert to char **
        char ** argv2=new char*[ps.size()];
        for(int i=0;i<ps.size();i++)
        {
            argv2[i]=new char[ps.at(i).size()+1];
            strcpy(argv2[i],ps.at(i).c_str());
        }

        //pharse again
        if (pharse_core(ps.size(), argv2) == false)
        {
            return false;
        }
        return true;
    }



    return true;
}

void CCommandline::init(std::vector<std::string> in, std::vector<std::string> in2)
{
    int i;
    std::string t;

    narg = in.size();
    for (i = 0; i < narg; i++)
    {
        if (in.at(i).at(0) == '-')
        {
            arguments.push_back(in.at(i));
            parameters.push_back(in2.at(i));
        }
        else
        {
            t = "-";
            t += in.at(i);
            arguments.push_back(t);
            parameters.push_back(in2.at(i));
        }
    }
    return;
};

void CCommandline::init(std::vector<std::string> in, std::vector<std::string> in2, std::vector<std::string> in3)
{
    int i;
    std::string t;

    narg = in.size();
    for (i = 0; i < narg; i++)
    {
        if (in.at(i).at(0) == '-')
        {
            arguments.push_back(in.at(i));
            parameters.push_back(in2.at(i));
            informations.push_back(in3.at(i));
        }
        else
        {
            t = "-";
            t += in.at(i);
            arguments.push_back(t);
            parameters.push_back(in2.at(i));
            informations.push_back(in3.at(i));
        }
    }
    return;
};

std::string CCommandline::query(std::string in)
{
    std::string out;
    int i;

    error_flag=1;

    out = "no";
    for (i = 0; i < narg; i++)
    {
        if (arguments.at(i).compare(in) == 0)
        {
            out = parameters.at(i);
            error_flag=0;
        }
    }
    if(error_flag==1)
    {
        std::cout<<"Unrecognized command line argument: "<<in<<std::endl;
    }

    const std::size_t strBegin = out.find_first_not_of(" ");
    const std::size_t strEnd = out.find_last_not_of(" ");
    const std::size_t strRange = strEnd - strBegin + 1;

    return out.substr(strBegin, strRange);

}

void CCommandline::print(void)
{
    int i;
    std::printf("Command line arguments:\n");
    for (i = 0; i < (int)arguments.size(); i++)
    {
        std::printf("%-15s %15s", arguments.at(i).c_str(), parameters.at(i).c_str());
        if (i < (int)informations.size())
            std::printf("     %-50s", informations.at(i).c_str());
        std::printf("\n");
    }
    std::printf("\n");

    if(error_flag==1)
    {
        std::printf("**********************************WARNING**********************************\n");
        std::printf("At least one unrecognized command line arguments!!\n");
    }

    return;
}

CCommandline::CCommandline()
{
    error_flag=0;
};
CCommandline::~CCommandline(){};
