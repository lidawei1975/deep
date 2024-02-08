/**
 * @file db_1d.cpp
 * @brief used for search and retrieve database items for the users
*/


#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#include "db_1d.h"


CDatabase_item::CDatabase_item(){};

CDatabase_item::~CDatabase_item(){};

const std::string CDatabase_item::get_id() const
{
    return id;
}

const std::string CDatabase_item::get_name() const
{
    return name;
}

/**
 * @param line
 * example input line
 * bmse000005_1.tab https://bmrb.io/metabolomics/mol_summary/show_data.php?id=bmse000005 InChI=1S/C10H14N5O7P/c11-8-5-9(13-2-12-8)15(3-14-5)10-7(17)6(16)4(22-10)1-21-23(18,19)20/h2-4,6-7,10,16-17H,1H2,(H2,11,12,13)(H2,18,19,20)/t4-,6-,7-,10-/m1/s1 AMP
 * 
 * @return true on success
 * @return false on failure
 */

bool CDatabase_item::process_input(std::string line)
{
    // seperate line by space
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token)
    {
        tokens.push_back(token);
    }
    if (tokens.size() == 4)
    {
        id = tokens[0]; //example id: bmse000005_1.tab
        link = tokens[1];
        inchi = tokens[2];
        name = tokens[3];

        // get base name. might rewrite to be more robust.
        base_name = id.substr(0, id.length() - 4);

        // get structure name. might rewrite to be more robust.
        structure_name = base_name.substr(0, base_name.length() - 2);

        return true;
    }
    else
    {
        return false;
    }
};

/**
 * @brief search the database item by id. return true if match
 * Example id: bmse000005_1.tab
 * Example id_: bmse000005
 * @param id_ input id
 * @return true if id starts with query id_
 * @return false if not match
 */

bool CDatabase_item::search_by_id(std::string id_)
{
    if(id.find(id_) == 0){
        return true;
    }
    else{
        return false;
    }
};

/**
 * @brief search the database item by id. return true if match
 * Example id: bmse000005_1.tab
 * Then base_name is bmse000005_1
 * @param id_ input id
 * @return true if match
 * @return false if not match
 */

bool CDatabase_item::search_by_id_exact(std::string id_)
{
    if(base_name == id_){
        return true;
    }
    else{
        return false;
    }
};

/**
 * @brief search the database item by structure name. return a distance between 0 and 1.
 * 
 * @param input_name 
 * @return double 
 */
double CDatabase_item::search_by_name(std::string input_name)
{
    double distance=0.0;
    int n = input_name.length();
    int m = name.length(); 

    // if input name is longer than database name, get the levenstein distance
    if (n > m)
    {
        distance = levenshtein_distance(name,input_name);
    }
    else
    {
        // If input name is shorter than database name. get the levenstein distance between input and all substrs of database name with same length as input name
        // Then get the minimum levenstein distance
        double min_distance = 1e100;
        for(int i=0;i<m-n+1;i++)
        {
            std::string substr = name.substr(i,n);
            double distance = levenshtein_distance(substr,input_name);
            if(distance < min_distance)
            {
                min_distance = distance;
            }
        }
        distance = min_distance;
    }
    return distance;
} 

/**
 * @brief //Levenshtein Distance between name1 and name2.
 * Case insensitive.
 */
double CDatabase_item::levenshtein_distance(const std::string &name1,const std::string &name2)
{
    int n = name1.length();
    int m = name2.length();
    std::vector<std::vector<int>> d(n + 1, std::vector<int>(m + 1));
    
    for (int i = 0; i <= n; i++)
    {
        d[i][0] = i;
    }
    
    for (int j = 0; j <= m; j++)
    {
        d[0][j] = j;
    }

    for (int j = 1; j <= m; j++)
    {
        for (int i = 1; i <= n; i++)
        {
            int cost;
            if (tolower(name1[i - 1]) == tolower(name2[j - 1])) // case insensitive comparison
            {
                cost = 0;
            }
            else
            {
                cost = 1;
            }
            d[i][j] = std::min(std::min(d[i - 1][j] + 1, d[i][j - 1] + 1), d[i - 1][j - 1] + cost);
        }
    }

    return (double)d[n][m] / (double)std::max(n, m);
};


bool CDatabase_item::search_by_formular(std::string formula_)
{
    if (formula == formula_)
    {
        return true;
    }
    else
    {
        return false;
    }
};

bool CDatabase_item::search_by_inchi(std::string inchi_)
{
    if (inchi == inchi_)
    {
        return true;
    }
    else
    {
        return false;
    }
};

bool CDatabase_item::print_to_json(Json::Value &root)
{
    Json::Value item;

    // bmse000005_1.tab -> bmse000005_1
    std::string  id_short = id.substr(0, id.length() - 4); // remove .tab. might rewrite to be more robust.

    item["id"] = id_short;
    item["name"] = name;
    item["link"] = link;
    item["inchi"] = inchi;
    item["formula"] = formula;
    root.append(item);
    return true;
};