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
#include "DeepConfig.h"
#include "json/json.h"



int main(int argc, char **argv)
{
    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-v");
    args2.push_back("0");
    args3.push_back("verbose level (0: minimal, 1:normal)");

    args.push_back("-db-all");
    args2.push_back("/srv/www/orapps/exec/gissmo-db/database/Database.list");
    args3.push_back("List of all database items");

    args.push_back("-db-in");
    args2.push_back("db_1d.list");
    args3.push_back("database item ID list (subset of All.list)");

    args.push_back("-db-out");
    args2.push_back("db_1d_verified.list");
    args3.push_back("sanctuary checked of database item ID list (subset of All.list)");

    args.push_back("-out-json");
    args2.push_back("db_1d_verified.json");
    args3.push_back("sanctuary checked of database item ID list (subset of All.list) in JSON format");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);

    int n_verbose_1d_check = std::stoi(cmdline.query("-v"));
    std::cout << "DEEP Picker package Version " << deep_picker_VERSION_MAJOR << "." << deep_picker_VERSION_MINOR << std::endl;
    cmdline.print();

    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    /**
     * Read line by line from db_all_file:
     * example:
     * colmar0019_1.tab colmar1d/database?id=colmar0019_1 InChI=1S/C6H8N/c1-7-5-3-2-4-6-7/h2-6H,1H3/q+1 N-Methylpyridinium
     * save first field (remove .tab if exist) to all_db_ids
     */
    std::string db_all_fname = cmdline.query("-db-all");
    std::ifstream db_all_file(db_all_fname);
    std::vector<std::string> all_db_ids;
    std::string line;
    while (getline(db_all_file, line))
    {
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        while (ss >> field)
        {
            fields.push_back(field);
        }
        if (fields.size() > 0)
        {
            std::string db_id = fields[0];
            /**
             * If db_id end with .tab, remove it
            */
            if(db_id.substr(db_id.size()-4,4)==".tab")
            {
                db_id = db_id.substr(0,db_id.size()-4);
            }
            all_db_ids.push_back(db_id);
        }
    }
    db_all_file.close();
    

    /**
     * Read from -db, which is a file with predifined list of db_ids (subset of db_all)
     */
    std::string db_fname = cmdline.query("-db-in");
    std::ifstream db_file(db_fname);
    std::ofstream db_out_file(cmdline.query("-db-out"));
    std::vector<std::string> db_ids;
    while (getline(db_file, line))
    {
        /**
         * Seperate db_id into fields, seperated by space(s)
         */
        std::cout << line << std::endl;
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> fields;
        while (ss >> field)
        {
            fields.push_back(field);
        }
        /**
         * Skip empty lines
         */
        if (fields.size() == 0)
        {
            continue;
        }

        /**
         * First field is db_id
        */
        std::string db_id = fields[0];
        /**
         * Remove .tab if db_id ends with .tab
         */
        if(db_id.length() > 4 && (db_id.substr(db_id.size()-4,4)==".tab"))
        {
            db_id = db_id.substr(0,db_id.size()-4);
        }

        /**
         * All other fields are saved to remain_part
        */
        std::string remain_part = "";
        for (int i = 1; i < fields.size(); i++)
        {
            remain_part += " ";
            remain_part += fields[i];
        }

        std::cout << "db_id:" << db_id << " remain: " << remain_part << std::endl;

        /**
         * If db_id is in all_db_ids, save the line to db_out_file.
         * If not, print out a warning message.
         */
        bool b_found = false;
        for (int i = 0; i < all_db_ids.size(); i++)
        {
            if (all_db_ids[i] == db_id)
            {
                b_found = true;
                break;
            }
        }
        if (b_found == true)
        {
            db_out_file << fields[0] << remain_part << std::endl;
            db_ids.push_back(db_id);
        }
        else
        {
            std::cout << "Warning: " << db_id << " is not found in " << db_all_fname << std::endl;
        } 
    }
    db_file.close();
    db_out_file.close();

    /**
     * Write db_ids to JSON file
     */
    Json::Value root;
    for(int i=0;i<db_ids.size();i++)
    {
        root["db_ids"][i] = db_ids[i];
    }
    std::ofstream json_out_file(cmdline.query("-out-json"));
    json_out_file << root;
    json_out_file.close();

    return 0;
}