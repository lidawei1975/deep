#include <string>
#include "json/json.h"

#ifndef DATABASE_ITEM_H
#define DATABASE_ITEM_H

class CDatabase_item
{

public:
    /**
     * @brief base folder name for .tab files e.g. "./database-500".
     *
     */
    std::string folder;

    /**
     * @brief pka folder name for .txt files e.g. "/d/orapps/exec/gissmo-db/database/pka_shift"
    */
    std::string pka_folder;

    /**
     * @brief base folder name for .mol and .inchi files e.g. "./mol".
     * 
     */
    std::string mol_folder;

protected:
    /**
     * @brief These 6 variables are read from the input file.
     * @brief id. e.g. "bmse000042_1.tab". this is also the file name where the .tab file is stored.
     * Then basename will be "bmse000042_1".
     * Then structure_name will be "bmse000042".
     */

    std::string id;
    std::string base_name;
    std::string structure_name;
    std::string name;
    std::string link;
    std::string inchi;
    std::string formula;
    double mass;


    double levenshtein_distance(const std::string &s1, const std::string &s2);


public:
    CDatabase_item();
    ~CDatabase_item();

    const std::string get_id() const;
    const std::string get_name() const;

    bool process_input(std::string); // one line from the input file, which contains all the information of a database item. Space is used as the delimiter.

    bool search_by_id(std::string);       // search the database item by id. return true if found.
    bool search_by_id_exact(std::string);       // search the database item by id. return true if found.
    double search_by_name(std::string);   // search the database item by name. return Levenshtein Distance over length of name.
    bool search_by_formular(std::string); // search the database item by formular. return true if found.
    bool search_by_inchi(std::string);    // search the database item by inchi. return true if found.

    bool print_to_json(Json::Value &root); // print the database item to json format. return true if successful.
};

#endif