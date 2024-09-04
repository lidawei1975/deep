#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

/**
 * Below two lines are required to check whether folder or file exists
 */
#include <sys/types.h>
#include <sys/stat.h>

#include "kiss_fft.h"

#include "json/json.h"
#include "fid_1d.h"

namespace fid_1d_helper
{
    size_t split(const std::string &txt, std::vector<std::string> &strs, char ch)
    {

        /**
         * find consecutive ch and replace them with a single ch in txt
         */
        std::string txt2 = txt;
        for (int i = 0; i < txt2.size(); i++)
        {
            if (txt2[i] == ch)
            {
                int j = i + 1;
                while (j < txt2.size() && txt2[j] == ch)
                {
                    j++;
                }
                if (j > i + 1)
                {
                    txt2.erase(i + 1, j - i - 1);
                }
            }
        }

        size_t pos = txt2.find(ch);
        size_t initialPos = 0;
        strs.clear();

        // Decompose statement
        while (pos != std::string::npos)
        {
            strs.push_back(txt2.substr(initialPos, pos - initialPos));
            initialPos = pos + 1;

            pos = txt2.find(ch, initialPos);
        }

        // Add the last one
        strs.push_back(txt.substr(initialPos, std::min(pos, txt.size()) - initialPos + 1));

        return strs.size();
    }
};

namespace nmrPipe
{
    /**
     * nmrPipe file header information dictionary
     * Key: parameter name
     * Value: parameter index in the nmrPipe header (nmrpipe_header_data). Need to convert to int (0 to 511)
     */
    std::map<std::string, std::string> nmrpipe_dictionary = {
        {"FDF4CENTER", "82"}, {"FDF2P0", "109"}, {"FDF2P1", "110"}, {"FDF1P1", "246"}, {"FDF2X1", "257"}, {"FDF1P0", "245"}, {"FDF3AQSIGN", "476"}, {"FDDISPMAX", "251"}, {"FDF4FTFLAG", "31"}, {"FDF3X1", "261"}, {"FDRANK", "180"}, {"FDF2C1", "418"}, {"FDF2QUADFLAG", "56"}, {"FDSLICECOUNT", "443"}, {"FDFILECOUNT", "442"}, {"FDMIN", "248"}, {"FDF3OBS", "10"}, {"FDF4APODQ2", "407"}, {"FDF4APODQ1", "406"}, {"FDF3FTSIZE", "200"}, {"FDF1LB", "243"}, {"FDF4C1", "409"}, {"FDF4QUADFLAG", "54"}, {"FDF1SW", "229"}, {"FDTRANSPOSED", "221"}, {"FDSECS", "285"}, {"FDF1APOD", "428"}, {"FDF2APODCODE", "413"}, {"FDPIPECOUNT", "75"}, {"FDOPERNAME", "464"}, {"FDF3LABEL", "20"}, {"FDPEAKBLOCK", "362"}, {"FDREALSIZE", "97"}, {"FDF4SIZE", "32"}, {"FDF4SW", "29"}, {"FDF4ORIG", "30"}, {"FDF3XN", "262"}, {"FDF1OBS", "218"}, {"FDDISPMIN", "252"}, {"FDF2XN", "258"}, {"FDF3P1", "61"}, {"FDF3P0", "60"}, {"FDF1ORIG", "249"}, {"FDF2FTFLAG", "220"}, {"FDF1TDSIZE", "387"}, {"FDLASTPLANE", "78"}, {"FDF1ZF", "437"}, {"FDF4FTSIZE", "201"}, {"FDF3C1", "404"}, {"FDFLTFORMAT", "1"}, {"FDF4CAR", "69"}, {"FDF1FTFLAG", "222"}, {"FDF2OFFPPM", "480"}, {"FDF1LABEL", "18"}, {"FDSIZE", "99"}, {"FDYEAR", "296"}, {"FDF1C1", "423"}, {"FDUSER3", "72"}, {"FDF1FTSIZE", "98"}, {"FDMINS", "284"}, {"FDSCALEFLAG", "250"}, {"FDF3TDSIZE", "388"}, {"FDTITLE", "297"}, {"FDPARTITION", "65"}, {"FDF3FTFLAG", "13"}, {"FDF2APODQ1", "415"}, {"FD2DVIRGIN", "399"}, {"FDF2APODQ3", "417"}, {"FDF2APODQ2", "416"}, {"FD2DPHASE", "256"}, {"FDMAX", "247"}, {"FDF3SW", "11"}, {"FDF4TDSIZE", "389"}, {"FDPIPEFLAG", "57"}, {"FDDAY", "295"}, {"FDF2UNITS", "152"}, {"FDF4APODQ3", "408"}, {"FDFIRSTPLANE", "77"}, {"FDF3SIZE", "15"}, {"FDF3ZF", "438"}, {"FDDIMORDER", "24"}, {"FDF3ORIG", "12"}, {"FD1DBLOCK", "365"}, {"FDF1AQSIGN", "475"}, {"FDF2OBS", "119"}, {"FDF1XN", "260"}, {"FDF4UNITS", "59"}, {"FDDIMCOUNT", "9"}, {"FDF4XN", "264"}, {"FDUSER2", "71"}, {"FDF4APODCODE", "405"}, {"FDUSER1", "70"}, {"FDMCFLAG", "135"}, {"FDFLTORDER", "2"}, {"FDUSER5", "74"}, {"FDCOMMENT", "312"}, {"FDF3QUADFLAG", "51"}, {"FDUSER4", "73"}, {"FDTEMPERATURE", "157"}, {"FDF2APOD", "95"}, {"FDMONTH", "294"}, {"FDF4OFFPPM", "483"}, {"FDF3OFFPPM", "482"}, {"FDF3CAR", "68"}, {"FDF4P0", "62"}, {"FDF4P1", "63"}, {"FDF1OFFPPM", "481"}, {"FDF4APOD", "53"}, {"FDF4X1", "263"}, {"FDLASTBLOCK", "359"}, {"FDPLANELOC", "14"}, {"FDF2FTSIZE", "96"}, {"FDUSERNAME", "290"}, {"FDF1X1", "259"}, {"FDF3CENTER", "81"}, {"FDF1CAR", "67"}, {"FDMAGIC", "0"}, {"FDF2ORIG", "101"}, {"FDSPECNUM", "219"}, {"FDF2LABEL", "16"}, {"FDF2AQSIGN", "64"}, {"FDF1UNITS", "234"}, {"FDF2LB", "111"}, {"FDF4AQSIGN", "477"}, {"FDF4ZF", "439"}, {"FDTAU", "199"}, {"FDF4LABEL", "22"}, {"FDNOISE", "153"}, {"FDF3APOD", "50"}, {"FDF1APODCODE", "414"}, {"FDF2SW", "100"}, {"FDF4OBS", "28"}, {"FDQUADFLAG", "106"}, {"FDF2TDSIZE", "386"}, {"FDHISTBLOCK", "364"}, {"FDSRCNAME", "286"}, {"FDBASEBLOCK", "361"}, {"FDF1APODQ2", "421"}, {"FDF1APODQ3", "422"}, {"FDF1APODQ1", "420"}, {"FDF1QUADFLAG", "55"}, {"FDF3UNITS", "58"}, {"FDF2ZF", "108"}, {"FDCONTBLOCK", "360"}, {"FDDIMORDER4", "27"}, {"FDDIMORDER3", "26"}, {"FDDIMORDER2", "25"}, {"FDDIMORDER1", "24"}, {"FDF2CAR", "66"}, {"FDF3APODCODE", "400"}, {"FDHOURS", "283"}, {"FDF1CENTER", "80"}, {"FDF3APODQ1", "401"}, {"FDF3APODQ2", "402"}, {"FDF3APODQ3", "403"}, {"FDBMAPBLOCK", "363"}, {"FDF2CENTER", "79"},
        {"FDUSER1","70"},{"FDUSER2","71"},{"FDUSER3","72"},{"FDUSER4","73"},{"FDUSER5","74"},{"FDUSER6","76"}   
        };

    /**
     * @brief nmrpipe_header_to_dictionary: convert the nmrPipe header to two dictionaries:
     * dict_string: values are strings
     * dict_float: values are floats
     *
     * nmrpipe_header_data is the nmrPipe header data, 512 float numbers (4 bytes each)
     */
    bool nmrpipe_header_to_dictionary(float *nmrpipe_header_data,
                                      std::map<std::string, std::string> &dict_string,
                                      std::map<std::string, float> &dict_float)
    {
        /**
         * loop through nmrpipe_dictionary where
         * keys are the parameter names
         * values are the parameter indices in the nmrPipe header (nmrpipe_header_data)
         */
        for (auto it = nmrpipe_dictionary.begin(); it != nmrpipe_dictionary.end(); ++it)
        {
            std::string key = it->first;
            std::string value = it->second;
            int index = std::stoi(value);
            {
                dict_float[key] = nmrpipe_header_data[index];
            }
        }

        char *nmrpipe_header_data_as_char = (char *)nmrpipe_header_data;

        /**
         * hardcode string parameters
         * e.g., nmrpipe_header_data[16:18] is a string with size of 2, which should be converted to a string
         * and assigned to dict_string["FDF2LABEL"]
         */
        dict_string["FDF2LABEL"] = std::string(nmrpipe_header_data_as_char + 16 * 4, 2 * 4);
        dict_string["FDF1LABEL"] = std::string(nmrpipe_header_data_as_char + 18 * 4, 2 * 4);
        dict_string["FDF3LABEL"] = std::string(nmrpipe_header_data_as_char + 20 * 4, 2 * 4);
        dict_string["FDF4LABEL"] = std::string(nmrpipe_header_data_as_char + 22 * 4, 2 * 4);
        dict_string["FDSRCNAME"] = std::string(nmrpipe_header_data_as_char + 286 * 4, 4 * 4);
        dict_string["FDUSERNAME"] = std::string(nmrpipe_header_data_as_char + 290 * 4, 4 * 4);
        dict_string["FDTITLE"] = std::string(nmrpipe_header_data_as_char + 297 * 4, 15 * 4);
        dict_string["FDCOMMENT"] = std::string(nmrpipe_header_data_as_char + 312 * 4, 40 * 4);
        dict_string["FDOPERNAME"] = std::string(nmrpipe_header_data_as_char + 464 * 4, 8 * 4);
        return true;
    }

    bool create_empty_nmrpipe_header(std::map<std::string, std::string> &dict_string, std::map<std::string, float> &dict_float)
    {
        std::vector<float> nmrpipe_header_data(512, 0.0f);
        nmrpipe_header_to_dictionary(nmrpipe_header_data.data(), dict_string, dict_float);

        /**
         * set some parameters to default values
         */
        dict_float["FDF1CENTER"] = 1.0f;
        dict_float["FDF2CENTER"] = 1.0f;
        dict_float["FDF3CENTER"] = 1.0f;
        dict_float["FDF4CENTER"] = 1.0f;

        dict_float["FDF3SIZE"] = 1.0f;
        dict_float["FDF4SIZE"] = 1.0f;

        dict_float["FDF1QUADFLAG"] = 1.0f;
        dict_float["FDF2QUADFLAG"] = 1.0f;
        dict_float["FDF3QUADFLAG"] = 1.0f;
        dict_float["FDF4QUADFLAG"] = 1.0f;

        dict_float["FDSPECNUM"] = 1.0f;
        dict_float["FDFILECOUNT"] = 1.0f;
        dict_float["FD2DVIRGIN"] = 1.0f;

        dict_float["FDDIMORDER1"] = 2.0f;
        dict_float["FDDIMORDER2"] = 1.0f;
        dict_float["FDDIMORDER3"] = 3.0f;
        dict_float["FDDIMORDER4"] = 4.0f;

        dict_string["FDF1LABEL"] = "Y";
        dict_string["FDF2LABEL"] = "X";
        dict_string["FDF3LABEL"] = "Z";
        dict_string["FDF4LABEL"] = "A";

        dict_float["FDFLTFORMAT"] = 4008636160.0f;      // ??
        dict_float["FDFLTORDER"] = 2.3450000286102295f; // ??

        return true;
    }

    /**
     * @brief nmrpipe_dictionary_to_header: convert the two dictionaries to nmrPipe header
     * inverse of nmrpipe_header_to_dictionary
     * nmrpipe_header_data must be a float array of size 512 (4 bytes each) with memory allocated already
     */
    bool nmrpipe_dictionary_to_header(float *nmrpipe_header_data, const std::map<std::string, std::string> &dict_string, const std::map<std::string, float> &dict_float)
    {
        for (auto it = nmrpipe_dictionary.begin(); it != nmrpipe_dictionary.end(); ++it)
        {
            std::string key = it->first;
            std::string value = it->second;
            int index = std::stoi(value);
            {
                if (dict_float.find(key) != dict_float.end())
                {
                    nmrpipe_header_data[index] = dict_float.at(key);
                }
            }
        }

        char *nmrpipe_header_data_as_char = (char *)nmrpipe_header_data;

        /**
         * hardcode string parameters
         * e.g., nmrpipe_header_data[16:18] is a string with size of 2, which should be converted to a string
         * and assigned to dict_string["FDF2LABEL"]
         */
        std::string FDF2LABEL = dict_string.at("FDF2LABEL");
        std::string FDF1LABEL = dict_string.at("FDF1LABEL");
        // std::string FDF3LABEL = dict_string.at("FDF3LABEL");
        // std::string FDF4LABEL = dict_string.at("FDF4LABEL");
        // std::string FDSRCNAME = dict_string.at("FDSRCNAME");
        // std::string FDUSERNAME = dict_string.at("FDUSERNAME");
        // std::string FDTITLE = dict_string.at("FDTITLE");
        // std::string FDCOMMENT = dict_string.at("FDCOMMENT");
        // std::string FDOPERNAME = dict_string.at("FDOPERNAME");

        std::copy(FDF2LABEL.begin(), FDF2LABEL.end(), nmrpipe_header_data_as_char + 16 * 4);
        std::copy(FDF1LABEL.begin(), FDF1LABEL.end(), nmrpipe_header_data_as_char + 18 * 4);
        // std::copy(FDF3LABEL.begin(), FDF3LABEL.end(), nmrpipe_header_data_as_char+20*4);
        // std::copy(FDF4LABEL.begin(), FDF4LABEL.end(), nmrpipe_header_data_as_char+22*4);
        // std::copy(FDSRCNAME.begin(), FDSRCNAME.end(), nmrpipe_header_data_as_char+286*4);
        // std::copy(FDUSERNAME.begin(), FDUSERNAME.end(), nmrpipe_header_data_as_char+290*4);
        // std::copy(FDTITLE.begin(), FDTITLE.end(), nmrpipe_header_data_as_char+297*4);
        // std::copy(FDCOMMENT.begin(), FDCOMMENT.end(), nmrpipe_header_data_as_char+312*4);
        // std::copy(FDOPERNAME.begin(), FDOPERNAME.end(), nmrpipe_header_data_as_char+464*4);

        return true;
    }
};

/**
 * Class apodiization
 */
apodization::apodization()
{
    apodization_type = FID_APODIZATION_NONE;
    p1 = 0.0;
    p2 = 0.0;
    p3 = 0.0;
    p4 = 0.0;
    p5 = 0.0;
    p6 = 0.0;
    sps.clear();
}

apodization::apodization(std::string apodization_string)
{
    sps.clear();
    /**
     * Convert to all lower case
     */
    std::transform(apodization_string.begin(), apodization_string.end(), apodization_string.begin(), ::tolower);

    std::vector<std::string> apodization_string_split;
    int n_fileds = fid_1d_helper::split(apodization_string, apodization_string_split, ' ');

    /**
     * At this time, first field is apodization function name, which must be "kaiser" or "none"
     * Skip apodization if not.
     */
    if (n_fileds > 0)
    {
        if (apodization_string_split[0] == "kaiser")
        {
            if (n_fileds != 4)
            {
                std::cerr << "Error: apodization function kaiser requires 3 parameters." << std::endl;
                return;
            }
            else
            {
                apodization_type = FID_APODIZATION_KAISER;
                p1 = std::stod(apodization_string_split[1]);
                p2 = std::stod(apodization_string_split[2]);
                p3 = std::stod(apodization_string_split[3]);
                p4 = 0.0;
                p5 = 0.0;
                p6 = 0.0;
            }
        }
        else if (apodization_string_split[0] == "none" || apodization_string_split[0] == "null" || apodization_string_split[0] == "n" || apodization_string_split[0] == "no")
        {
            // do nothing. Skip apodization
            apodization_type = FID_APODIZATION_NONE;
            p1 = 0.0;
            p2 = 0.0;
            p3 = 0.0;
            p4 = 0.0;
            p5 = 0.0;
            p6 = 0.0;
        }
        else
        {
            std::cerr << "Error: apodization function name must be kaiser or none." << std::endl;
            return;
        }
    }
}

apodization::apodization(FID_APODIZATION_TYPE apodization_type_, double p1_, double p2_, double p3_, double p4_, double p5_, double p6_)
{
    sps.clear();
    apodization_type = apodization_type_;
    p1 = p1_;
    p2 = p2_;
    p3 = p3_;
    p4 = p4_;
    p5 = p5_;
    p6 = p6_;
}

apodization::~apodization()
{
    sps.clear();
}

bool apodization::set_n(int ndata)
{
    /**
     * Set up sps
     */
    sps.clear(); // clear sps first in case it has been set up before
    sps.push_back(0.5); // special case for first point
    for (int j = 1; j < ndata; j++)
    {
        sps.push_back(pow(sin(M_PI * p1 + M_PI * p2 / 2.0 / ndata * j), p3));
    }

    return true;
}

bool apodization::run_apodization(float *data, int ndata) const
{
    if(ndata==sps.size()*2)
    {
        for (int i = 0; i < ndata/2; i++)
        {
            data[i * 2] *= sps[i];
            data[i * 2 + 1] *= sps[i];
        }
    }
    else if(ndata==sps.size())
    {
        for (int i = 0; i < ndata; i++)
        {
            data[i] *= sps[i];
        }
    }
    else
    {
        std::cerr << "Error: data size is not correct." << std::endl;
        return false;
    }
    return true;
}

/**
 * Constructor and destructor
 */
fid_base::fid_base()
{
}
fid_base::~fid_base()
{
}

/**
 * @brief read_jcamp: read jcamp file and save the data to a udict
 * @param file_name: input file name
 * @param udict: output dictionary
 */
bool fid_base::read_jcamp(std::string file_name, std::map<std::string, std::string> &udict) const
{
    std::ifstream infile(file_name.c_str());
    if (!infile.is_open())
    {
        std::cout << "Error: cannot open file " << file_name << std::endl;
        return false;
    }

    std::string line;
    std::string key, value;
    while (std::getline(infile, line))
    {
        /**
         * On linux, the end of line is \n, so the value of line is correct here
         * On Windows, the end of line is \r\n, so the value of line has an extra \r, we need to remove it
         */
        if (line.size() > 0 && line[line.size() - 1] == '\r')
        {
            line = line.substr(0, line.size() - 1);
        }

        /**
         * check end of file indicator "##END="
         */
        if (line.find("##END=") != std::string::npos)
        {
            break;
        }
        /**
         * comment line starts with "##"
         */
        else if (line.find("$$") == 0)
        {
            udict["comment"] += line + "\n"; // append to comment. Keep the header
        }
        /**
         * header line starts with "##" and the 3rd character is not "$"
         */
        else if (line.find("##") == 0 && line[2] != '$')
        {
            udict["header"] += line + "\n"; // append to header. Keep the header
        }
        /**
         * key value pair line starts with "##" and the 3rd character is "$"
         * @brief read_jcmap_line: read one line of jcamp file and store the content in key and value
         */
        else if (line.find("##") == 0 && line[2] == '$')
        {
            if (read_jcmap_line(infile, line, key, value))
            {
                udict[key] = value;
            }
            else
            {
                std::cerr << "Warning: cannot parse line " << line << " from file " << file_name << std::endl;
            }
        }
        /**
         * other lines are undefined
         */
        else
        {
            std::cerr << "Warning: unknown line " << line << " from file " << file_name << std::endl;
        }
    }

    infile.close();
    return true;
}

/**
 * @brief read_jcmap_line: read one line of jcamp file and store the content in key and value
 * In rare cases, the value may have "<" but without ">", which means a multi-line value
 * In this case, we need to read the next line and append to the value
 */
bool fid_base::read_jcmap_line(std::ifstream &infile, std::string line, std::string &key, std::string &value) const
{
    /**
     * Step 1, find "=", which separates key and value. Skip the first two characters "##" and the third character "$"
     */
    size_t pos = line.find("=");
    if (pos == std::string::npos)
    {
        return false;
    }

    key = line.substr(3, pos - 3);
    value = line.substr(pos + 1);

    /**
     * remove leading space in value, if any
     */
    pos = value.find_first_not_of(" ");
    if (pos != std::string::npos)
    {
        value = value.substr(pos);
    }

    /**
     * if value starts with "<" but not ">" it is a multi-line value
     */
    if (value.find("<") != std::string::npos)
    {
        while (value.find(">") == std::string::npos)
        {
            std::string line2;
            std::getline(infile, line2);
            size_t pos = line2.find_first_not_of(" ");
            if (pos != std::string::npos)
            {
                line2 = line2.substr(pos);
            }
            value += "\n" + line2;
        }
        /**
         * remove "<" and ">" from value
         */
        value = value.substr(1, value.size() - 2);
    }
    /**
     * if "(" is found in value, it is a list of values
     * example:
     * (0..7)
     */
    else if (value.find("(") != std::string::npos)
    {
        /**
         * Get number of values ( between .. and ) )
         */
        size_t pos1 = value.find("..");
        size_t pos2 = value.find(")");
        if (pos1 == std::string::npos || pos2 == std::string::npos)
        {
            return false;
        }
        std::string str_num = value.substr(pos1 + 2, pos2 - pos1 - 2);
        int num = std::stoi(str_num);

        /**
         * get remaining of the line (after ")" )
         */
        std::string str_list = value.substr(pos2 + 1);

        /**
         * get list of values from str_list, separated by space
         */
        std::vector<std::string> list;
        fid_1d_helper::split(str_list, list, ' ');

        /**
         * grap addtional lines until all values are read
         */
        while (list.size() < num)
        {
            std::string line2;
            std::vector<std::string> list2;
            std::getline(infile, line2);
            size_t pos = line2.find_first_not_of(" ");
            if (pos != std::string::npos)
            {
                line2 = line2.substr(pos);
            }
            fid_1d_helper::split(line2, list2, ' ');
            list.insert(list.end(), list2.begin(), list2.end());
        }

        /**
         * convert list back to string, separated by space
         */
        value = "";
        for (size_t i = 0; i < list.size(); i++)
        {
            value += list[i] + " ";
        }
    }
    /**
     * simple value, do nothing
     */
    return true;
};


/**
 * @brief fid_base::remove_bruker_digitizer_filter: remove Bruker digitizer filter
 * We are working on frequency domain data only to make it simple
 * require grpdly > 0, i.e., spectrum from an old Bruker spectrometer is not supported
 * This function will modify spectrum_real and spectrum_imag
 */
bool fid_base::remove_bruker_digitizer_filter(double grpdly_, std::vector<float> &s_real, std::vector<float> &s_imag) const
{
    if (grpdly_ < 0.0 || s_real.size() != s_imag.size())
    {
        return false;
    }

    /**
     * apply grpdly degree 0 order phase correction to frquency domain data
     * s_real and s_imag
     */
    for (int i = 0; i < s_real.size(); i++)
    {
        float phase = 2 * M_PI * grpdly_ * i / s_real.size();
        float cos_phase = cos(phase);
        float sin_phase = sin(phase);

        float real = s_real[i];
        float imag = s_imag[i];

        s_real[i] = real * cos_phase - imag * sin_phase;
        s_imag[i] = real * sin_phase + imag * cos_phase;
    }

    return true;
};

/**
 * @brief fid_1d class constructor
 */
fid_1d::fid_1d()
{
    zf = 1;              // no zero filling, default
    ndata_frq = 0;       // no data without FFT
    receiver_gain = 1.0; // default

    nmrpipe_header_data.resize(512, 0.0f); // nmrPipe header data 512*4
}

/**
 * @brief fid_1d class destructor
 */
fid_1d::~fid_1d()
{
}

/**
 * @brief fid_1d::run_zf: set zero filling factor
 * zf will be read by run_fft_and_rm_bruker_filter
 */
bool fid_1d::run_zf(int _zf)
{
    zf = _zf;
    return true;
}

/**
 * @brief read_bruker_folder:  read fid and parameter files from Bruker folder
 * this function will change the value of the following variables:
 * fid_data_float (unused one will have size 0)
 * data_type, data_complexity
 * ndata_bruker, ndata
 * udict_acqus, udict_procs
 * spectral_width, observed_frequency, carrier_frequency
 */
bool fid_1d::read_bruker_folder(std::string folder_name)
{
    /**
     * @brief test if the folder exists, using system call stat
     * sb is a struct of type stat
     */
    struct stat sb;

    /**
     * Calls the function with path as argument. This is a C function.
     * If the file/directory exists at the path returns 0
     */

    int status = stat(folder_name.c_str(), &sb);

    if (status != 0)
    {
        std::cout << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }
    else if (!(sb.st_mode & S_IFDIR))
    {
        // exist but is not a directory
        std::cout << "Error: folder " << folder_name << " does not exist!" << std::endl;
        return false;
    }

    /**
     * now look for folder_name/fid or folder_name/ser
     * If none of them exist, return false
     */

    std::string fid_data_file_name = folder_name + "/fid";
    status = stat(fid_data_file_name.c_str(), &sb);
    if (status == 0)
    {
    }
    else
    {
        fid_data_file_name = folder_name + "/ser";
        status = stat(fid_data_file_name.c_str(), &sb);
        if (status == 0)
        {
        }
        else
        {
            fid_data_file_name = ""; // label it as empty
            return false;
        }
    }

    /**
     * Read acqus file and store the content in udict
     */
    std::string acqus_file_name = folder_name + "/acqus";
    status = stat(acqus_file_name.c_str(), &sb);
    if (status != 0)
    {
        std::cout << "Warning: cannot find acqus file in folder " << folder_name << std::endl;
        return false;
    }

    /**
     * if we are here, we have found the fid data file and acqus file
     */
    std::vector<std::string> fid_data_file_names;
    fid_data_file_names.push_back(fid_data_file_name);
    return read_bruker_acqus_and_fid(acqus_file_name, fid_data_file_names);
}

bool fid_1d::read_bruker_acqus_and_fid(const std::string &acqus_file_name, const std::vector<std::string> &fid_data_file_names)
{

    read_jcamp(acqus_file_name, udict_acqus);

    /**
     * Check udic_acqus["AQ_mod"] to determine the data type: real or complex
     */
    if (udict_acqus["AQ_mod"] == "3" || udict_acqus["AQ_mod"] == "1")
    {
        data_complexity = FID_DATA_COMPLEXITY_COMPLEX;
    }
    else
    {
        data_complexity = FID_DATA_COMPLEXITY_REAL;
    }

    /**
     * Check udic_acqus["BYTORDA"] to determine the data type: int32 or float32
     */
    if (udict_acqus["DTYPA"] == "2")
    {
        data_type = FID_DATA_TYPE_FLOAT64;
    }
    else
    {
        data_type = FID_DATA_TYPE_INT32;
    }

    /**
     * check udic_acqus["TD"] to determine size of fid data
     */
    if (udict_acqus.find("TD") == udict_acqus.end())
    {
        std::cout << "Error: cannot find TD in acqus file" << std::endl;
        return false;
    }

    int td0 = std::stoi(udict_acqus["TD"]);

    /**
     * According to Bruker manu,
     * when data_type = FID_DATA_TYPE_FLOAT64, fid is padded to 128 bytes
     * when data_type = FID_DATA_TYPE_INT32, fid is padded to 256 bytes
     */

    if (data_type == FID_DATA_TYPE_INT32)
    {
        int ntemp = int(std::ceil((double)td0 / 256.0));
        ndata_bruker = ntemp * 256;
    }
    else
    {
        int ntemp = int(std::ceil((double)td0 / 128.0));
        ndata_bruker = ntemp * 128;
    }

    /**
     * Bruker TD is the number of data points, not the number of complex points for complex data
     * But we define ndata as the number of complex points
     */
    if (data_complexity == FID_DATA_COMPLEXITY_COMPLEX)
    {
        ndata = ndata_bruker / 2;
    }
    else
    {
        ndata = ndata_bruker;
    }

    /**
     * now we can actually read the fid data
     * For complex data, real and imaginary parts are stored interleaved by Bruker.
     * Here we leave them interleaved in fid_data_float or fid_data_int
     */

    fid_data_float.clear();

    for (int i = 0; i < fid_data_file_names.size(); i++)
    {
        std::vector<int32_t> fid_data_int;        // 32 bit
        std::vector<double> temp_fid_data_double; // 64 bit
        FILE *fp_fid_data = fopen(fid_data_file_names[i].c_str(), "rb");

        if (data_type == FID_DATA_TYPE_INT32)
        {
            int nread;
            fid_data_int.clear();
            fid_data_int.resize(ndata_bruker);
            nspectra = 0;
            while(fread(fid_data_int.data()+ndata_bruker*nspectra, sizeof(int32_t), ndata_bruker, fp_fid_data) == ndata_bruker)
            {
                nspectra++;
                /**
                 * Get new space for next round
                */
                fid_data_int.resize(ndata_bruker * (nspectra + 1));
            }
            /**
             * Set fid_data_int to correct size, remove the extra space applied in last round
            */
            fid_data_int.resize(ndata_bruker * nspectra);
            if (nspectra == 0)
            {
                std::cout << "Error: cannot read " << ndata_bruker << " int32 from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }
        else if (data_type == FID_DATA_TYPE_FLOAT64)
        {
            int nread;
            temp_fid_data_double.clear();
            temp_fid_data_double.resize(ndata_bruker);
            nspectra = 0;
            while(fread(temp_fid_data_double.data()+ndata_bruker*nspectra, sizeof(double), ndata_bruker, fp_fid_data) == ndata_bruker)
            {
                nspectra++;
                /**
                 * Get new space for next round
                */
                temp_fid_data_double.resize(ndata_bruker * (nspectra + 1));
            }
            /**
             * Set temp_fid_data_double to correct size, remove the extra space applied in last round
            */
            temp_fid_data_double.resize(ndata_bruker * nspectra);
            if (nspectra == 0)
            {
                std::cout << "Error: cannot read " << ndata_bruker << " double from file " << fid_data_file_names[i] << std::endl;
                return false;
            }
        }

        /**
         * convert int32 or double64 to float32.
         */
        fid_data_float.resize(ndata_bruker * nspectra);    
        if (data_type == FID_DATA_TYPE_INT32)
        {
            for (int i = 0; i < ndata_bruker * nspectra; i++)
            {
                fid_data_float[i] += (float)fid_data_int[i];
            }
        }
        else
        {
            for (int i = 0; i < ndata_bruker * nspectra; i++)
            {
                fid_data_float[i] += temp_fid_data_double[i];
            }
        }
        fclose(fp_fid_data);
    }

    std::cout << "Read " << nspectra << " spectra from " << fid_data_file_names.size() << " files" << std::endl;

    /**
     * Now setup the following variables:
     * spectral_width, observed_frequency, carrier_frequency
     */
    if (udict_acqus.find("SW_h") != udict_acqus.end())
    {
        spectral_width = std::stod(udict_acqus["SW_h"]);
    }
    else
    {
        std::cout << "Error: cannot find SW_h in acqus file" << std::endl;
        return false;
    }

    /**
     * get receiver_gain from acqus if it exists
     */
    if (udict_acqus.find("RG") != udict_acqus.end())
    {
        receiver_gain = std::stod(udict_acqus["RG"]);
    }
    else
    {
        std::cout << "Warning: cannot find RG in acqus file" << std::endl;
        receiver_gain = 1.0; // default
    }

    if (udict_acqus.find("SFO1") != udict_acqus.end())
    {
        observed_frequency = std::stod(udict_acqus["SFO1"]);
    }
    else
    {
        std::cout << "Error: cannot find SFO1 in acqus file" << std::endl;
        return false;
    }

    if (udict_acqus.find("O1") != udict_acqus.end())
    {
        carrier_frequency = std::stod(udict_acqus["O1"]);
    }
    else
    {
        std::cout << "Error: cannot find O1 in acqus file" << std::endl;
        return false;
    }

    /**
     * get parameters "GRPDLY" from acqus file
     */
    if (udict_acqus.find("GRPDLY") != udict_acqus.end())
    {
        grpdly = std::stod(udict_acqus["GRPDLY"]);
    }
    else
    {
        std::cout << "Error: cannot find GRPDLY in acqus file" << std::endl;
        return false;
    }

    /**
     * grpdly must > 0. Otherwiae it is from an early days Bruker spectrometer
     * We don't support this case at this time
     */
    if (grpdly <= 0.0)
    {
        std::cout << "Error: GRPDLY = " << grpdly << " is not supported" << std::endl;
        return false;
    }

    return true;
}

/**
 * write 1D fid to nmrpipe file
 */
bool fid_1d::write_nmrpipe_fid(const std::string outfname) const
{

    std::map<std::string, std::string> nmrpipe_dict_string;
    std::map<std::string, float> nmrpipe_dict_float;

    /**
     * create nmrpipe header.
     * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
     * from udict_acqus and derived values
     * false means we are saving frq data
     */
    create_nmrpipe_dictionary(false, nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * define header vector, because the function call nmrpipe_dictionary_to_header won't apply for memory
     */
    std::vector<float> nmrpipe_header_data(512, 0.0f);

    nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * write temp_data_float to file
     */
    FILE *fp = fopen(outfname.c_str(), "wb");
    if (fp == NULL)
    {
        std::cout << "Error: cannot open file " << outfname << std::endl;
        return false;
    }

    fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);

    /**
     * convert fid_data_float to temp_data_float, real part only: 0,2,4,6,...
     */
    std::vector<float> temp_data_float;
    temp_data_float.resize(ndata_bruker);
    for (int i = 0; i < ndata; i++)
    {
        temp_data_float[i] = (float)fid_data_float[i * 2];
    }
    fwrite(temp_data_float.data(), sizeof(float), ndata_bruker, fp);

    /**
     * convert fid_data_float to temp_data_float, imaginary part only: 1,3,5,7,...
     */
    for (int i = 0; i < ndata; i++)
    {
        temp_data_float[i] = (float)fid_data_float[i * 2 + 1];
    }
    fwrite(temp_data_float.data(), sizeof(float), ndata_bruker, fp);

    fclose(fp);

    return true;
}

/**
 * @brief fid_1d::run_fft_and_rm_bruker_filter: run fft on fid_data_float
 * @return true on success, false on failure
 * Will modify spectrum_real and spectrum_imag
 */
bool fid_1d::run_fft_and_rm_bruker_filter()
{
    /**
     * get ndata_power_of_2, which is the smallest power of 2
     * that is larger than or equal ndata
     */
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    ndata_frq = ndata_power_of_2 * zf;

    /**
     * Run apoization. We set ndata*2 in run_apodization because we are working on complex data
    */
    apod->set_n(ndata);
    

    /**
     * Clear spectrum_real and spectrum_imag. Do not resize because we will use insert to add data later
    */
    spectrum_real.clear();
    spectrum_imag.clear();


    for(int i=0;i<nspectra;i++)
    {   
        std::vector<float> temp_fid_data_float(0);

        /**
         * Copy fid_data_float to temp_fid_data_float
         */
        temp_fid_data_float.insert(temp_fid_data_float.end(),fid_data_float.begin()+i*ndata_bruker,fid_data_float.begin()+(i+1)*ndata_bruker);
       
        apod->run_apodization(temp_fid_data_float.data(), ndata*2);
        
        kiss_fft_cfg cfg;
        kiss_fft_cpx *in, *out;

        in = new kiss_fft_cpx[ndata_frq];
        out = new kiss_fft_cpx[ndata_frq];
        for (int i = 0; i < ndata; i++)
        {
            in[i].r = temp_fid_data_float[i * 2];
            in[i].i = temp_fid_data_float[i * 2 + 1];
        }

        /**
         * fill remaining of in with 0
         */
        for (int i = ndata; i < ndata_frq; i++)
        {
            in[i].r = 0.0;
            in[i].i = 0.0;
        }

        if ((cfg = kiss_fft_alloc(ndata_frq, 0, NULL, NULL)) != NULL)
        {
            kiss_fft(cfg, in, out);
            free(cfg);
        }
        else
        {
            std::cerr << "Error: cannot allocate memory for fft" << std::endl;
            return false;
        }

        std::vector<float> temp_spectrum_real(ndata_frq, 0.0f);
        std::vector<float> temp_spectrum_imag(ndata_frq, 0.0f);

        for (int i = 0; i < ndata_frq; i++)
        {
            temp_spectrum_real[i] = out[i].r / sqrt(float(ndata_frq));
            temp_spectrum_imag[i] = out[i].i / sqrt(float(ndata_frq)); // scale by sqrt(ndata_frq) to be consistent with standard fft
        }

        /**
         * Important: this step need to be done before fft result swap and flip
         */
        remove_bruker_digitizer_filter(grpdly,temp_spectrum_real,temp_spectrum_imag);

        /**
         * To be consistent with nmrPipe, we also swap left and right halves of spectrum
         * then flip the spectrum
         * and apply receiver gain
         */
        std::vector<float> spectrum_real_reoganized(ndata_frq, 0.0f);
        std::vector<float> spectrum_imag_reoganized(ndata_frq, 0.0f);

        for (int i = 0; i < ndata_frq; i++)
        {
            spectrum_real_reoganized[ndata_frq - i] = (float)temp_spectrum_real[(i + ndata_frq / 2) % ndata_frq] * receiver_gain;
            spectrum_imag_reoganized[ndata_frq - i] = (float)temp_spectrum_imag[(i + ndata_frq / 2) % ndata_frq] * receiver_gain;
        }

        /**
         * free memory
        */
        delete [] in;
        delete [] out;

        /**
         * Copy to spectrum_real and spectrum_imag
        */

        spectrum_real.insert(spectrum_real.end(),spectrum_real_reoganized.begin(),spectrum_real_reoganized.end());
        spectrum_imag.insert(spectrum_imag.end(),spectrum_imag_reoganized.begin(),spectrum_imag_reoganized.end());
    }

    return true;
}

bool fid_1d::set_up_apodization(apodization *apod_)
{
    apod = apod_;
    return true;
};



/**
 * @brief create_nmrpipe_dictionary: create nmrpipe dictionary from udict_acqus and derived values
 * @param b_frq: true if we are saving frq data, false if we are saving time data
 * @param nmrpipe_dict_string: output dictionary for string values
 * @param nmrpipe_dict_float: output dictionary for float values
 */
bool fid_1d::create_nmrpipe_dictionary(bool b_frq, std::map<std::string, std::string> &nmrpipe_dict_string, std::map<std::string, float> &nmrpipe_dict_float) const
{
    /**
     * @brief create_empty_nmrpipe_header: create empty nmrpipe header
     */
    nmrPipe::create_empty_nmrpipe_header(nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * Fill in some parameters from what we have
     */
    nmrpipe_dict_float["FDDIMCOUNT"] = 1.0f; // 1D data
    nmrpipe_dict_float["FD2DPHASE"] = 0.0f;  // for 1D data, direct dimension phase is 0

    /**
     * Copied from nmrglue. no idea what they mean
     */
    if (nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF2QUADFLAG"] == nmrpipe_dict_float["FDF3QUADFLAG"] && nmrpipe_dict_float["FDF1QUADFLAG"] == nmrpipe_dict_float["FDF4QUADFLAG"] == 1.0f)
    {
        nmrpipe_dict_float["FDQUADFLAG"] = 1.0f;
    }

    /**
     * Some examples of parameters that we need to put in nmrpipe_dict_float
     * 'sw': 12019.2307692308
     *  'complex':True
     *   'obs': 600.06
     *  'car': 2820.28200001605
     *   'size': 32768
     *  'label': '1H'
     */

    nmrpipe_dict_float["FDF2SW"] = spectral_width;
    nmrpipe_dict_float["FDF2OBS"] = observed_frequency;
    nmrpipe_dict_float["FDF2CAR"] = carrier_frequency / observed_frequency; // normalized to observed frequency, per nmrPipe convention
    nmrpipe_dict_string["FDF2LABEL"] = "1H";                                // hardcoded for now

    if (data_complexity == FID_DATA_COMPLEXITY_COMPLEX)
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 0.0f;
    }
    else
    {
        nmrpipe_dict_float["FDF2QUADFLAG"] = 1.0f;
    }

    if (b_frq)
    {
        /**
         * we are saving frq data, so set FDF2FTFLAG to 1 and set value for FDF2FTSIZE
         * set FDF2TDSIZE if we have time data
         */
        nmrpipe_dict_float["FDF2FTSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDF2FTFLAG"] = 1.0f; // frq data, instead of time

        nmrpipe_dict_float["FDSIZE"] = ndata_frq;
        nmrpipe_dict_float["FDREALSIZE"] = ndata_frq;
    }
    else
    {
        /**
         * we are saving time data, so set FDF2FTFLAG to 0 and set value for FDF2TDSIZE
         * set FDF2FTSIZE if we have frq data
         */
        nmrpipe_dict_float["FDF2TDSIZE"] = ndata;
        nmrpipe_dict_float["FDF2FTFLAG"] = 0.0f; // time data, instead of frq

        nmrpipe_dict_float["FDSIZE"] = ndata;
        nmrpipe_dict_float["FDREALSIZE"] = ndata;
    }

    /**
     * set apodization information. meaningless for time domain data
     */
    nmrpipe_dict_float["FDF2APOD"] = ndata_frq;
    nmrpipe_dict_float["FDF2CENTER"] = int(ndata_frq / 2.0f) + 1.0f;

    /**
     * copied from nmrglue. no idea what they mean
     */
    nmrpipe_dict_float["FDF2ORIG"] = (nmrpipe_dict_float["FDF2CAR"] * nmrpipe_dict_float["FDF2OBS"] - nmrpipe_dict_float["FDF2SW"] * (ndata_frq - nmrpipe_dict_float["FDF2CENTER"]) / ndata_frq);

    return true;
}

/**
 * @brief fid_1d::write_nmrpipe_ft1: write 1D spectrum to nmrpipe file
 * Before writing, define nmrpipe header, set values from udict_acqus and derived values
 * @param outfname: output file name
 * @return true on success, false on failure
 */
bool fid_1d::write_nmrpipe_ft1(std::string outfname)
{
    std::map<std::string, std::string> nmrpipe_dict_string;
    std::map<std::string, float> nmrpipe_dict_float;

    /**
     * create nmrpipe header.
     * This will set values for nmrpipe_dict_string and nmrpipe_dict_float
     * from udict_acqus and derived values
     * True means we are saving frq data
     */
    create_nmrpipe_dictionary(true, nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * define header vector, because the function call nmrpipe_dictionary_to_header won't apply for memory
     */
    nmrpipe_header_data.resize(512, 0.0f);

    nmrPipe::nmrpipe_dictionary_to_header(nmrpipe_header_data.data(), nmrpipe_dict_string, nmrpipe_dict_float);

    /**
     * now write nmrpipe_header_data and spectral to the file
     * if outfname is empty, we don't write to file. This function is called to generate header only
     */
    if (outfname != "")
    {
        /**
         * now write nmrpipe_header_data and spectral to the file
         */
        FILE *fp = fopen(outfname.c_str(), "wb");
        if (fp == NULL)
        {
            std::cerr << "Error: cannot open file " << outfname << " for writing" << std::endl;
            return false;
        }
        fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp);
        fwrite(spectrum_real.data(), sizeof(float), ndata_frq, fp);
        fwrite(spectrum_imag.data(), sizeof(float), ndata_frq, fp);
        fclose(fp);

        if(nspectra>1)
        {
            /**
             * Find filename extension: after last . in outfname, such as .ft1.
             * In case there is no . in outfname, ext will be empty and basename will be outfname
             */
            std::string ext = outfname.substr(outfname.find_last_of(".") + 1);
            std::string basename = outfname.substr(0, outfname.find_last_of("."));

            for(int i=1;i<nspectra;i++)
            {
                std::string outfname2 = basename + "_" + std::to_string(i) + "." + ext;
                FILE *fp2 = fopen(outfname2.c_str(), "wb");
                if (fp2 == NULL)
                {
                    std::cerr << "Error: cannot open file " << outfname2 << " for writing" << std::endl;
                    return false;
                }
                fwrite(nmrpipe_header_data.data(), sizeof(float), 512, fp2);
                fwrite(spectrum_real.data()+ndata_frq*i, sizeof(float), ndata_frq, fp2);
                fwrite(spectrum_imag.data()+ndata_frq*i, sizeof(float), ndata_frq, fp2);
                fclose(fp2);
            }
        }
    }

    return true;
}

/**
 * @brief fid_1d::write_nmrpipe_ft1: write some information
 * This is mainly for web-server
 */
bool fid_1d::write_json(std::string fname)
{
    std::ofstream outfile(fname.c_str());
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    Json::Value root;
    root["ndata"] = ndata;
    root["ndata_power_of_2"] = ndata_power_of_2;
    root["carrier_frequency"] = carrier_frequency;
    root["observed_frequency"] = observed_frequency;
    root["spectral_width"] = spectral_width;

    outfile << root << std::endl;
    outfile.close();

    return true;
}

/**
 * @brief fid_1d::get_spectrum_header: return header as a vector of float
 */
std::vector<float> fid_1d::get_spectrum_header(void) const
{
    return nmrpipe_header_data;
};

/**
 * @brief fid_1d::get_spectrum_real: return real part of spectrum as a vector of float
 */
std::vector<float> fid_1d::get_spectrum_real(void) const
{
    return spectrum_real;
};

/**
 * @brief fid_1d::get_spectrum_imag: return imaginary part of spectrum as a vector of float
 */
std::vector<float> fid_1d::get_spectrum_imag(void) const
{
    return spectrum_imag;
};