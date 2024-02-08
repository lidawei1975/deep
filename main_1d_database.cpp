// #include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>
#include <set>

#include "commandline.h"
#include "db_1d.h"

enum operation_mode
{
    list,
    search_by_id,
    search_by_name,
    search_by_formula,
    search_by_inchi
};


int main(int argc, char **argv)
{

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-f");
    args2.push_back("arguments_db_manipulate.txt");
    args3.push_back("read arguments from file (arguments.txt)");

    args.push_back("-mol_folder");
    args2.push_back("mol_files");
    args3.push_back("folder name for mol files");

    args.push_back("-in");
    args2.push_back("All.list");
    args3.push_back("input database file name");

    args.push_back("-out");
    args2.push_back("db_result.json");
    args3.push_back("output file names (db_result.json)");

    args.push_back("-method");
    args2.push_back("list");
    args3.push_back("list, (search by) id, inchi, name or formula");

    args.push_back("-item");
    args2.push_back("string for search");
    args3.push_back("Alanine");

    args.push_back("-name_cutoff");
    args2.push_back("0.4");
    args3.push_back("cutoff for fuzzy search");



    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    std::string outfname = cmdline.query("-out");

    //if outfname==stdout, then result will be print to stdout as a json string. we skip all other output.
    bool b_print_to_stdout = (outfname == "stdout");

    if(!b_print_to_stdout)
    {
        cmdline.print();
    }

    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    
    std::string db_files = cmdline.query("-in");

    std::vector<CDatabase_item> db_items;

    std::string line;
    std::ifstream fin(db_files);
    while (getline(fin, line))
    {
        if (line[0] == '#')
        {
            continue;
        }
        class CDatabase_item db_item;

        if(db_item.process_input(line))
        {
            db_item.mol_folder=cmdline.query("-mol_folder");
            db_items.emplace_back(db_item);
        }
    }

    std::string mode=cmdline.query("-method");
    operation_mode mode_enum;

    if(mode=="list")
    {
        mode_enum=operation_mode::list;
    }
    else if(mode=="id")
    {
        mode_enum=operation_mode::search_by_id;
    }
    else if(mode=="inchi")
    {
        mode_enum=operation_mode::search_by_inchi;
    }
    else if(mode=="name")
    {
        mode_enum=operation_mode::search_by_name;
    }
    else if(mode=="formula")
    {
        mode_enum=operation_mode::search_by_formula;
    }
    else
    {
        std::cout<<"Error: mode not recognized"<<std::endl;
        exit(1);
    }

    Json::Value root;
    std::string item=cmdline.query("-item");
    switch(mode_enum)
    {
        case operation_mode::list:
        {
            for (auto &db_item : db_items)
            {
                db_item.print_to_json(root);
            }
            break;
        }
        case operation_mode::search_by_id:
        {
            
            for (auto &db_item : db_items)
            {
                if(db_item.search_by_id(item))
                {
                    db_item.print_to_json(root);
                }
            }
            break;
        }
        case operation_mode::search_by_inchi:
        {
            for (auto &db_item : db_items)
            {
                if(db_item.search_by_inchi(item))
                {
                    db_item.print_to_json(root);
                }
            }
            break;
        }
        case operation_mode::search_by_name:
        {
            for (auto &db_item : db_items)
            {
                double distance=db_item.search_by_name(item);
                if(distance<std::stod(cmdline.query("-name_cutoff")))
                {
                    if(!b_print_to_stdout)
                    {
                        std::cout<<"name: "<<item<<" "<<db_item.get_name()<<" distance:"<<distance<<std::endl;
                    }
                    db_item.print_to_json(root);
                }
            }
            break;
        }
        case operation_mode::search_by_formula:
        {
            for (auto &db_item : db_items)
            {
                if(db_item.search_by_formular(item))
                {
                    db_item.print_to_json(root);
                }
            }
            break;
        }
        default:
        {
            std::cout<<"mode not supported"<<std::endl;
            root["error"]="mode not supported";
            break;
        }
    }

    //write to file or stdout if outfname=="stdout"
    if(b_print_to_stdout)
    {
        std::cout<<root<<std::endl;
    }
    else
    {
        std::ofstream fout(outfname);
        fout<<root;
        fout.close();
    }

    return 0;
}