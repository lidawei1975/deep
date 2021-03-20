#include <string>
#include <vector>
#include <cstdio>

using namespace std;


#include "commandline.h"


void CCommandline::pharse(int argc, char **argv)
{
    int i, j;
    for (i = 1; i < argc; i++)
    {
        for (j = 0; j < narg; j++)
        {
            if (arguments.at(j).compare(argv[i]) == 0 && argv[i][0] == '-' && argv[i][1] == '-') //started with --, multi followings
            {
                parameters.at(j) = " ";                       // remove default one.
                while (i + 1 < argc && argv[i + 1][0] != '-') //store multi entries in to this value
                {
                    parameters.at(j).append(argv[i + 1]);
                    parameters.at(j).append(" ");
                    i++;
                }
            }
            else if (arguments.at(j).compare(argv[i]) == 0) // //started with -, normal one
            {
                if (i + 1 < argc && argv[i + 1][0] != '-')
                {
                    parameters.at(j) = argv[i + 1];
                    i++;
                }
                else
                    parameters.at(j) = "yes";
            }
        }
    }
    return;
}

void CCommandline::init(vector<string> in, vector<string> in2)
{
    int i;
    string t;

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

void CCommandline::init(vector<string> in, vector<string> in2, vector<string> in3)
{
    int i;
    string t;

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

string CCommandline::query(string in)
{
    string out;
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

    return out;
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
