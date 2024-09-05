#include <string>
#include <cstring>
#include <vector>
#include <cstdio>
#include <iostream>
#include <fstream>

#include "commandline.h"

/**
 * @brief pharse_file: pharse the commandline arguments (in argc and argv)
 * Once we match with arguments ("-f" only)
 * we will store the parameters in the parameters vector.
*/
bool CCommandline::pharse_file(int argc, char ** argv)
{
    int j_of_file=-1;
    for (int j = 0; j < narg; j++)
    {
        if (arguments.at(j).compare("-f") == 0) 
        {
            j_of_file=j;
            break;   
        }
    }
    if(j_of_file==-1)
    {
        /**
         * there is no "-f"
        */
        return false;
    }

    for (int i = 1; i < argc; i++)
    {
        if (arguments.at(j_of_file).compare(argv[i]) == 0) 
        {
            parameters.at(j_of_file) = ""; 
            if(i+1>=argc || is_key(argv[i + 1])==true)
            {
                parameters.at(j_of_file) = "yes";
            }
            while (i + 1 < argc && is_key(argv[i + 1])==false)
            {
                parameters.at(j_of_file).append(argv[i + 1]);
                parameters.at(j_of_file).append(" ");
                i++;
            }    
            continue;                
        }  
    }


    return true;
}

/**
 * @brief pharse_core: pharse the commandline arguments (in argc and argv) 
 * Once we match with arguments,
 * we will store the parameters in the parameters vector.
*/
bool CCommandline::pharse_core(int argc, char ** argv)
{
    int i, j;
    for (i = 1; i < argc; i++)
    {   
        /**
         * @brief -f is a special case, we already process earlier
         * so we will skip it from commandline arguments list, including its parameters
        */
        if(strcmp(argv[i],"-f")==0)
        {
            while (i + 1 < argc && is_key(argv[i + 1]) == false)
            {
                i++;
            }
            continue;
        }

        bool b=false;
        for (j = 0; j < narg; j++)
        {
            /**
             * @brief arguments is the list of arguments, e.g. -in, -out, -v, etc.
             * Each argument always starts with a dash, e.g. -in
             */
            if (arguments.at(j).compare(argv[i]) == 0) 
            {
                parameters.at(j) = ""; 
                if(i+1>=argc || is_key(argv[i + 1])==true)
                {
                    parameters.at(j) = "yes";
                }
                while (i + 1 < argc && is_key(argv[i + 1])==false)
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

/**
 * @brief check if the input is a key such as "-in","-h", etc.
 * "-0.3" is not a key, it is a parameter.
 * "--" is not a key, it is a parameter.
 * @param in 
 * @return true 
 * @return false 
 */
bool CCommandline::is_key(std::string in)
{
    /**
     * @brief minimal legnth for a key is 2, e.g. -in, -out, -v, etc.
     * 
     */
    if(in.length()<2)
    {
        return false;
    }
    /**
     * @brief starts with a "-" AND the next character is from a to z or A to Z, then it is a key
     */
    if(in.at(0)=='-' && (in.at(1)>='a' && in.at(1)<='z' || in.at(1)>='A' && in.at(1)<='Z'))
    {
        return true;
    }
    return false;
}


bool CCommandline::pharse(int argc, char **argv)
{

    /**
     * Update -f arguments if it is set in the commandline arguments
     * otherwise, we will use the default arguments.
    */
    pharse_file(argc, argv);

    //check to see whether -f is set or not
    std::string arguments_file=query("-f");
    std::ifstream fin(arguments_file);
    if(arguments_file != "none" && fin)
    {

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
    
        /**
         * pharse everything in the file to replace the default arguments
        */
        pharse_core(ps.size(), argv2);
    }

    /**
     * pharse the commandline arguments.
     * argument in commandline arguments will overwrite the arguments in the file
    */
    pharse_core(argc, argv);


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
