
#include "peak_manipulation.h"

namespace peak_tools
{
    bool peak_reading_sparky(std::string fname, std::vector<std::array<double, 2>> &peak_pos, std::vector<std::string> &peak_info)
    {
        std::string line, p;
        std::vector<std::string> ps;
        std::stringstream iss;

        int xpos = -1;
        int ypos = -1;
        int ass = -1;

        std::ifstream fin(fname);

        if (!fin)
            return false;

        bool b_data = false;
        while (getline(fin, line))
        {
            iss.clear();
            iss.str(line);
            ps.clear();
            while (iss >> p)
            {
                ps.push_back(p);
            }
            if (ps.size() < 3)
                continue; // empty line??

            if (ps[0] == "Assignment" || ps[0] == "w2" || ps[0] == "w1")
            {
                for (int i = 0; i < ps.size(); i++)
                {
                    if (ps[i] == "w2")
                    {
                        xpos = i;
                    } // in sparky, w2 is direct dimension
                    else if (ps[i] == "w1")
                    {
                        ypos = i;
                    }
                    else if (ps[i] == "Assignment")
                    {
                        ass = i;
                    }
                }
                b_data = true;
                continue;
            }

            if (b_data == true)
            {
                std::array<double, 2> t;
                t[0] = stod(ps[xpos]);
                t[1] = stod(ps[ypos]);
                peak_pos.push_back(t);
                peak_info.push_back(ps[ass]);
            }
        }
        return true;
    };

    bool is_assignment(std::string ass)
    {
        if (ass.find("?") == 0 || ass.find("Peak") == 0 || ass.find("peak") == 0 || ass.find("None") == 0 || ass.find("none") == 0 || ass.find("X") || ass.find("x") == 0)
        {
            return false;
        }
        else
        {
            return true;
        }
    };

    void MinCostMatching(const std::vector<std::vector<int>> &cost, std::vector<int> &Lmate, std::vector<int> &Rmate)
    {
        int n = int(cost.size());

        // construct dual feasible solution
        std::vector<int> u(n);
        std::vector<int> v(n);
        for (int i = 0; i < n; i++)
        {
            u[i] = cost[i][0];
            for (int j = 1; j < n; j++)
                u[i] = std::min(u[i], cost[i][j]);
        }
        for (int j = 0; j < n; j++)
        {
            v[j] = cost[0][j] - u[0];
            for (int i = 1; i < n; i++)
                v[j] = std::min(v[j], cost[i][j] - u[i]);
        }

        // construct primal solution satisfying complementary slackness
        Lmate = std::vector<int>(n, -1);
        Rmate = std::vector<int>(n, -1);
        int mated = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (Rmate[j] != -1)
                    continue;
                if (std::fabs(cost[i][j] - u[i] - v[j]) < 1e-10)
                {
                    Lmate[i] = j;
                    Rmate[j] = i;
                    mated++;
                    break;
                }
            }
        }

        std::vector<int> dist(n);
        std::vector<int> dad(n);
        std::vector<int> seen(n);

        // repeat until primal solution is feasible
        while (mated < n)
        {

            // find an unmatched left node
            int s = 0;
            while (Lmate[s] != -1)
                s++;

            // initialize Dijkstra
            fill(dad.begin(), dad.end(), -1);
            fill(seen.begin(), seen.end(), 0);
            for (int k = 0; k < n; k++)
                dist[k] = cost[s][k] - u[s] - v[k];

            int j = 0;
            while (true)
            {

                // find closest
                j = -1;
                for (int k = 0; k < n; k++)
                {
                    if (seen[k])
                        continue;
                    if (j == -1 || dist[k] < dist[j])
                        j = k;
                }
                seen[j] = 1;

                // termination condition
                if (Rmate[j] == -1)
                    break;

                // relax neighbors
                const int i = Rmate[j];
                for (int k = 0; k < n; k++)
                {
                    if (seen[k])
                        continue;
                    const double new_dist = dist[j] + cost[i][k] - u[i] - v[k];
                    if (dist[k] > new_dist)
                    {
                        dist[k] = new_dist;
                        dad[k] = j;
                    }
                }
            }

            // update dual variables
            for (int k = 0; k < n; k++)
            {
                if (k == j || !seen[k])
                    continue;
                const int i = Rmate[k];
                v[k] += dist[k] - dist[j];
                u[i] -= dist[k] - dist[j];
            }
            u[s] += dist[j];

            // augment along path
            while (dad[j] >= 0)
            {
                const int d = dad[j];
                Rmate[j] = Rmate[d];
                Lmate[Rmate[j]] = j;
                j = d;
            }
            Rmate[j] = s;
            Lmate[s] = j;

            mated++;
        }
    };

    std::vector<std::deque<int>> breadth_first(std::vector<int> &neighbor, int n)
    {
        std::vector<std::deque<int>> clusters;
        std::deque<int> work, work2;
        std::vector<int> used;

        used.resize(n, 0);

        for (int i = 0; i < n; i++)
        {
            if (used.at(i) != 0)
            {
                continue;
            }

            used.at(i) = 1;
            work.clear();
            work2.clear();
            work.push_back(i);
            work2.push_back(i);

            while (!work.empty())
            {
                int c = work.at(0);
                work.pop_front();

                for (int j = 0; j < n; j++)
                {
                    if (j == c || used.at(j) != 0)
                    {
                        continue;
                    }
                    if (neighbor[j * n + c] == 1)
                    {
                        #pragma omp critical
                        {
                            work.push_back(j);
                            work2.push_back(j);
                        }
                        used.at(j) = 1;
                    }
                }
            }
            if (work2.size() >= 1)
            {
                clusters.push_back(work2);
            }
        }
        return clusters;
    };
}

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

    /**
     * Change D to d in col_formats (%4D -> %4d)
     * Capital D means something else, it is a bug in our DP and VF printing code.
    */
    for (int i = 0; i < col_formats.size(); i++)
    {
        for (int j = 0; j < col_formats[i].size(); j++)
        {
            if (col_formats[i][j] == 'D')
            {
                col_formats[i][j] = 'd';
            }
        }
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
 * Get lines in header that start with DATA
*/
bool peak_manipulation::get_data_lines(std::vector<std::string> &data_lines)
{
    for (int i = 0; i < header.size(); i++)
    {
        if (header[i].substr(0, 4) == "DATA")
        {
            data_lines.push_back(header[i]);
        }
    }
    return true;
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
    if(index<0 || index>col_names.size() || (opt!=column_operation::DELETE && opt!=column_operation::INSERT && opt!=column_operation::REPLACE))
    {
        return false;
    }

    if(v.size() != col_values.size())
    {
        return false;
    }
    
    if(opt==column_operation::DELETE)
    {
        /**
         * New col_name and col_format are ignored
        */
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
    else if(opt==column_operation::REPLACE)
    {
        /**
         * For replace, we do not change the column name and format
         * col_name and col_format are ignored
        */
        for(int i=0;i<col_values.size();i++)
        {
            col_values[i][index]=v[i];
        }
    }
    return true;
}