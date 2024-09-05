#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#ifndef PEAK_MANIP
#define PEAK_MANIP

enum column_operation
{
    ADD,  // add a value to the column (can be negative)
    MULTIPLY, // multiply a value to the column

    DELETE, // delete the column

    INSERT, // insert a string just after the column
    REPLACE, // replace the column with a string at the same position
};


class peak_manipulation
{
    /**
     * These are the values in the input file.
     * @brief col_names: column names
     * @brief col_formats: column formats
     * @brief col_values: column values
     * col_names and col_formats have the same size
     * col_values.size() == # of peaks
     * col_values[i].size() == col_names.size()
    */
    std::vector<std::string> col_names;
    std::vector<std::string> col_formats;
    std::vector<std::vector<std::string>> col_values;

    /**
     * @brief header: header lines. They will be copied to the output file without any change.
    */
    std::vector<std::string> header;


public:
    peak_manipulation();
    ~peak_manipulation();

    /**
     * Read a peak file and save the values to col_names, col_formats, and col_values
    */
    bool read_file(const std::string fname);

    /**
     * Write the values to a new peak file
    */
    bool write_file(const std::string fname);

    /**
     * Get column index by column name
    */
    int get_column_index(const std::string);

    /**
     * Get mutiple column indexes by column names starts with a prefix
    */
    std::vector<int> get_column_indexes_by_prefix(const std::string);

    /**
     * Get a column by column index
    */
    bool get_column(int,std::vector<std::string>&); 

    /**
     * Operate on the column
     * @param index: column index
     * @param opt: operation type
     * @param v: value
    */
    bool operate_on_column(int index, enum column_operation opt, double v); 

    /**
     * Operate on the column, delete or insert a column
     * @param index: column index
     * @param opt: operation type: DELETE or INSERT only
    */
    bool operate_on_column(int index, enum column_operation opt, const std::string col_name, std::string col_format, const std::vector<std::string> &v);

};

#endif