#include <string>
#include <vector>
#include <cstdio>
#include <iostream>

#include "commandline.h"


void CCommandline::pharse(int argc, char **argv)
{
    int i, j;
    for (i = 1; i < argc; i++)
    {
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
            }
        }
    }
    return;
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

    const auto strBegin = out.find_first_not_of(" ");
    const auto strEnd = out.find_last_not_of(" ");
    const auto strRange = strEnd - strBegin + 1;

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
