
#include "peak_manipulation.h"


peak_manipulation::peak_manipulation() {};
peak_manipulation::~peak_manipulation() {};

/**
 * Read a peak file and save the values to col_names, col_formats, and col_values
*/
bool peak_manipulation::read_file(const std::string fname)
{
    std::ifstream fin(fname.c_str());
    if (!fin)
    {
        std::cerr << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    std::string line;
    while(getline(fin, line))
    {
        if (line.size() == 0)
        {
            //do nothing
        }

        else if (line[0] == '#')
        {
            header.push_back(line);
        }

        /**
         * If line start with DATA, then it is a header line. Copy it to the output file.
        */
        else if (line.substr(0, 4) == "DATA")
        {
            header.push_back(line);
        }

        /**
         * If line start with VARS, it is a lists of column names. save them to col_names and copy line to the output file.
        */
        else if (line.substr(0, 4) == "VARS")
        {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp; // skip VARS
            while(iss >> tmp)
            {
                col_names.push_back(tmp);
            }
        }

        /**
         * if line start with FORMAT, it is a lists of column formats. save them to col_formats and copy line to the output file.
        */
        else if (line.substr(0, 6) == "FORMAT")
        {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp; // skip FORMAT
            while(iss >> tmp)
            {
                col_formats.push_back(tmp);
            }
        }
        /**
         * All other lines are data lines. Split them by space(or tab) and save them to col_values.
         * The good way is to use FORMAT information to split the line, but we always add a space between columns, so we can use space to split the line.
        */
        else
        {
            std::vector<std::string> fields;
            std::istringstream iss(line);
            std::string tmp;
            int i = 0;
            while(iss >> tmp)
            {
                fields.push_back(tmp);
            }
            col_values.push_back(fields);
        }
    }

    fin.close();

    return true;
}

/**
 * Write the values to a new peak file
*/
bool peak_manipulation::write_file(const std::string fname)
{
    FILE *fout = fopen(fname.c_str(), "w");
    if (!fout)
    {
        std::cerr << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    for (int i = 0; i < header.size(); i++)
    {
        fprintf(fout, "%s\n", header[i].c_str());
    }

    fprintf(fout, "VARS ");
    for (int i = 0; i < col_names.size(); i++)
    {
        fprintf(fout, "%s ", col_names[i].c_str());
    }
    fprintf(fout, "\n");

    fprintf(fout, "FORMAT ");
    for (int i = 0; i < col_formats.size(); i++)
    {
        fprintf(fout, "%s ", col_formats[i].c_str());
    }
    fprintf(fout, "\n");

    for (int i = 0; i < col_values.size(); i++)
    {
        for (int j = 0; j < col_values[i].size(); j++)
        {
            /**
             * If col_formats[j] contains %s, then it is a string,
             * else if col_formats[j] contains %d, it is a integer,
             * else it is a float.
            */
            if(col_formats[j].find("s") != std::string::npos)
            {
                fprintf(fout, col_formats[j].c_str(), col_values[i][j].c_str());
            }
            else if(col_formats[j].find("d") != std::string::npos)
            {
                fprintf(fout, col_formats[j].c_str(), atoi(col_values[i][j].c_str()));
            }
            else //%f or %e
            {
                fprintf(fout, col_formats[j].c_str(), atof(col_values[i][j].c_str()));
            }
            /**
             * Add a space between columns
            */
            if(j<col_values[i].size()-1)
            {
                fprintf(fout, " ");
            }
        }
        fprintf(fout, "\n");
    }

    fclose(fout);
    return true;
}

/**
 * get mutiple column indexes by column name match
 * return first one if there are multiple matches
 * return -1 if no match
*/
int peak_manipulation::get_column_index(const std::string col_name)
{
    int ndx = -1;
    for (int i = 0; i < col_names.size(); i++)
    {
        if (col_names[i] == col_name)
        {
            ndx = i;
            break;
        }
    }
    return ndx;
}

/**
 * Get mutiple column indexes by column names starts with a prefix
 * empty vector if no match
 */
std::vector<int> peak_manipulation::get_column_indexes_by_prefix(const std::string prefix)
{
    std::vector<int> ndx;
    for (int i = 0; i < col_names.size(); i++)
    {
        if (col_names[i].substr(0, prefix.size()) == prefix)
        {
            ndx.push_back(i);
        }
    }
    return ndx;
};

/**
 * Get a column by column index
 */
bool peak_manipulation::get_column(int index, std::vector<std::string> &v)
{
    if (index < 0 || index >= col_names.size())
    {
        return false;
    }

    for (int i = 0; i < col_values.size(); i++)
    {
        v.push_back(col_values[i][index]);
    }
    return true;
};

/**
 * Operate on the column
 * @param index: column index
 * @param opt: operation type
 * @param v: value
 */
bool peak_manipulation::operate_on_column(int index, enum column_operation opt, double v)
{
    if (index < 0 || index >= col_names.size() || (opt != column_operation::ADD && opt != column_operation::MULTIPLY))
    {
        return false;
    }

    for (int i = 0; i < col_values.size(); i++)
    {
        double tmp = atof(col_values[i][index].c_str());
        switch (opt)
        {
        case column_operation::ADD:
            tmp += v;
            break;
        case column_operation::MULTIPLY:
            tmp *= v;
            break;
        default:
            break;
        }
        char buffer[100];
        snprintf(buffer, 100, col_formats[index].c_str(), tmp);
        col_values[i][index] = buffer;
    }
    return true;
};

/**
 * Operate on the column, delete or insert a column
 * @param index: column index
 * @param opt: operation type: DELETE or INSERT only
 */
bool peak_manipulation::operate_on_column(int index, enum column_operation opt, const std::string col_name, std::string col_format, const std::vector<std::string> &v)
{
    if(index<0 || index>col_names.size() || (opt!=column_operation::DELETE && opt!=column_operation::INSERT))
    {
        return false;
    }

    if(v.size() != col_values.size())
    {
        return false;
    }
    
    if(opt==column_operation::DELETE)
    {
        col_names.erase(col_names.begin()+index);
        col_formats.erase(col_formats.begin()+index);
        for(int i=0;i<col_values.size();i++)
        {
            col_values[i].erase(col_values[i].begin()+index);
        }
    }
    else if(opt==column_operation::INSERT)
    {
        col_names.insert(col_names.begin()+index, col_name);
        col_formats.insert(col_formats.begin()+index, col_format);
        for(int i=0;i<col_values.size();i++)
        {
            col_values[i].insert(col_values[i].begin()+index, v[i]);
        }
    }
    return true;
}