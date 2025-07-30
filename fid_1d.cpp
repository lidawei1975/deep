#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>
#include <limits>

/**
 * Below two lines are required to check whether folder or file exists
 */
#include <sys/types.h>
#include <sys/stat.h>

#ifdef WEBASSEMBLY
/**
* Bind to emscripten with exposed class methods.
*/
#include <emscripten/bind.h>
using namespace emscripten;
#endif


#include "kiss_fft.h"

#include "json/json.h"
#include "fid_1d.h"

/**
 * These are shared varibles between db_match_1d and pattern_match_1d and spectrum_pick_1d
*/
int shared_data_1d::n_verbose=1; 
bool shared_data_1d::b_dosy=false;
std::vector<double> shared_data_1d::z_gradients;
double shared_data_1d::peak_combine_cutoff=0.01;
bool shared_data_1d::b_remove_failed_peaks=false;

namespace ldw_math_spectrum_1d
{
    bool SplitFilename(const std::string &str, std::string &path_name, std::string &file_name, std::string &file_name_ext)
    {
        bool b = false;
        std::string file_name_full;
        std::size_t found = str.find_last_of("/\\");
        if (found != std::string::npos)
        {
            path_name = str.substr(0, found);
            file_name_full = str.substr(found + 1);
            b = true;
        }
        else
        {
            path_name = ".";
            file_name_full = str;
        }

        // std::cout<<"file_name_full is "<<file_name_full<<std::endl;

        found = file_name_full.find_last_of(".");
        if (found != std::string::npos)
        {
            file_name = file_name_full.substr(0, found);
            file_name_ext = file_name_full.substr(found + 1);
            b = true;
        }
        else
        {
            file_name_ext = "";
            file_name = file_name_full;
        }

        // std::cout<<"path is "<<path_name<<std::endl;
        // std::cout<<"file_name is "<<file_name<<std::endl;
        // std::cout<<"file_name_ext is "<<file_name_ext<<std::endl;

        return b;
    };

    bool trim(std::string &str)
    {
        if (str.empty())
        {
            return false;
        }

        str.erase(0, str.find_first_not_of(" "));
        str.erase(str.find_last_not_of(" ") + 1);
        return true;
    }
};

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
        {"FDBASEBLOCK", "361"},
        {"FDBMAPBLOCK", "363"},
        {"FDCOMMENT", "312"},
        {"FDCONTBLOCK", "360"},
        {"FDDAY", "295"},
        {"FDDIMCOUNT", "9"},
        {"FDDIMORDER1", "24"},
        {"FDDIMORDER2", "25"},
        {"FDDIMORDER3", "26"},
        {"FDDIMORDER4", "27"},
        {"FDF2AQSIGN", "64"},
        {"FDF2APOD", "95"},
        {"FDF2APODCODE", "413"},
        {"FDF2APODQ1", "415"},
        {"FDF2APODQ2", "416"},
        {"FDF2APODQ3", "417"},
        {"FDF2CAR", "66"},
        {"FDF2CENTER", "79"},
        {"FDF2C1", "418"},
        {"FDF2FTFLAG", "220"},
        {"FDF2FTSIZE", "96"},
        {"FDF2LABEL", "16"},
        {"FDF2LB", "111"},
        {"FDF2OBS", "119"},
        {"FDF2OFFPPM", "480"},
        {"FDF2ORIG", "101"},
        {"FDF2P0", "109"},
        {"FDF2P1", "110"},
        {"FDF2QUADFLAG", "56"},
        {"FDF2SW", "100"},
        {"FDF2TDSIZE", "386"},
        {"FDF2UNITS", "152"},
        {"FDF2X1", "257"},
        {"FDF2XN", "258"},
        {"FDF2ZF", "108"},
        {"FDF3APOD", "50"},
        {"FDF3APODCODE", "400"},
        {"FDF3APODQ1", "401"},
        {"FDF3APODQ2", "402"},
        {"FDF3APODQ3", "403"},
        {"FDF3AQSIGN", "476"},
        {"FDF3CAR", "68"},
        {"FDF3CENTER", "81"},
        {"FDF3C1", "404"},
        {"FDF3FTFLAG", "13"},
        {"FDF3FTSIZE", "200"},
        {"FDF3LABEL", "20"},
        {"FDF3OBS", "10"},
        {"FDF3OFFPPM", "482"},
        {"FDF3ORIG", "12"},
        {"FDF3P0", "60"},
        {"FDF3P1", "61"},
        {"FDF3QUADFLAG", "51"},
        {"FDF3SIZE", "15"},
        {"FDF3SW", "11"},
        {"FDF3TDSIZE", "388"},
        {"FDF3UNITS", "58"},
        {"FDF3X1", "261"},
        {"FDF3XN", "262"},
        {"FDF3ZF", "438"},
        {"FDF4APOD", "53"},
        {"FDF4APODCODE", "405"},
        {"FDF4APODQ1", "406"},
        {"FDF4APODQ2", "407"},
        {"FDF4APODQ3", "408"},
        {"FDF4AQSIGN", "477"},
        {"FDF4CAR", "69"},
        {"FDF4CENTER", "82"},
        {"FDF4C1", "409"},
        {"FDF4FTFLAG", "31"},
        {"FDF4FTSIZE", "201"},
        {"FDF4LABEL", "22"},
        {"FDF4OBS", "28"},
        {"FDF4OFFPPM", "483"},
        {"FDF4ORIG", "30"},
        {"FDF4P0", "62"},
        {"FDF4P1", "63"},
        {"FDF4QUADFLAG", "54"},
        {"FDF4SIZE", "32"},
        {"FDF4SW", "29"},
        {"FDF4TDSIZE", "389"},
        {"FDF4UNITS", "59"},
        {"FDF4X1", "263"},
        {"FDF4XN", "264"},
        {"FDF4ZF", "439"},
        {"FDFILECOUNT", "442"},
        {"FDFIRSTPLANE", "77"},
        {"FDFLTFORMAT", "1"},
        {"FDFLTORDER", "2"},
        {"FDFMIN", "248"},
        {"FDHISTBLOCK", "364"},
        {"FDHOURS", "283"},
        {"FDLASTBLOCK", "359"},
        {"FDLASTPLANE", "78"},
        {"FDMAGIC", "0"},
        {"FDMCFLAG", "135"},
        {"FDMAX", "247"},
        {"FDMIN", "248"},
        {"FDMINS", "284"},
        {"FDMONTH", "294"},
        {"FDNOISE", "153"},
        {"FDOPERNAME", "464"},
        {"FDPARTITION", "65"},
        {"FDPEAKBLOCK", "362"},
        {"FDPIPECOUNT", "75"},
        {"FDPIPEFLAG", "57"},
        {"FDPLANELOC", "14"},
        {"FDQUADFLAG", "106"},
        {"FDRANK", "180"},
        {"FDREALSIZE", "97"},
        {"FDSCALEFLAG", "250"},
        {"FDSECS", "285"},
        {"FDSIZE", "99"},
        {"FDSLICECOUNT", "443"},
        {"FDSPECNUM", "219"},
        {"FDSRCNAME", "286"},
        {"FDTEMPERATURE", "157"},
        {"FDTITLE", "297"},
        {"FDTRANSPOSED", "221"},
        {"FDUSER1", "70"},
        {"FDUSER2", "71"},
        {"FDUSER3", "72"},
        {"FDUSER4", "73"},
        {"FDUSER5", "74"},
        {"FDUSER6", "76"},
        {"FDUSERNAME", "290"},
        {"FDYEAR", "296"},
        {"FDF1APOD", "428"},
        {"FDF1APODCODE", "414"},
        {"FDF1APODQ1", "420"},
        {"FDF1APODQ2", "421"},
        {"FDF1APODQ3", "422"},
        {"FDF1AQSIGN", "475"},
        {"FDF1CAR", "67"},
        {"FDF1CENTER", "80"},
        {"FDF1C1", "423"},
        {"FDF1FTFLAG", "222"},
        {"FDF1FTSIZE", "98"},
        {"FDF1LABEL", "18"},
        {"FDF1LB", "243"},
        {"FDF1OBS", "218"},
        {"FDF1OFFPPM", "481"},
        {"FDF1ORIG", "249"},
        {"FDF1P0", "245"},
        {"FDF1P1", "246"},
        {"FDF1QUADFLAG", "55"},
        {"FDF1SW", "229"},
        {"FDF1TDSIZE", "387"},
        {"FDF1UNITS", "234"},
        {"FDF1X1", "259"},
        {"FDF1XN", "260"},
        {"FDF1ZF", "437"}
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

    bool create_default_nmrpipe_dictionary(std::map<std::string, std::string> &dict_string, std::map<std::string, float> &dict_float)
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
        std::string FDF3LABEL = dict_string.at("FDF3LABEL");
        std::string FDF4LABEL = dict_string.at("FDF4LABEL");
        std::string FDSRCNAME = dict_string.at("FDSRCNAME");
        std::string FDUSERNAME = dict_string.at("FDUSERNAME");
        std::string FDTITLE = dict_string.at("FDTITLE");
        std::string FDCOMMENT = dict_string.at("FDCOMMENT");
        std::string FDOPERNAME = dict_string.at("FDOPERNAME");

        std::copy(FDF2LABEL.begin(), FDF2LABEL.end(), nmrpipe_header_data_as_char + 16 * 4);
        std::copy(FDF1LABEL.begin(), FDF1LABEL.end(), nmrpipe_header_data_as_char + 18 * 4);
        std::copy(FDF3LABEL.begin(), FDF3LABEL.end(), nmrpipe_header_data_as_char+20*4);
        std::copy(FDF4LABEL.begin(), FDF4LABEL.end(), nmrpipe_header_data_as_char+22*4);
        std::copy(FDSRCNAME.begin(), FDSRCNAME.end(), nmrpipe_header_data_as_char+286*4);
        std::copy(FDUSERNAME.begin(), FDUSERNAME.end(), nmrpipe_header_data_as_char+290*4);
        std::copy(FDTITLE.begin(), FDTITLE.end(), nmrpipe_header_data_as_char+297*4);
        std::copy(FDCOMMENT.begin(), FDCOMMENT.end(), nmrpipe_header_data_as_char+312*4);
        std::copy(FDOPERNAME.begin(), FDOPERNAME.end(), nmrpipe_header_data_as_char+464*4);

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
    /**
     * Set default spectral width to maximum, so that apodizaiton will be 1.0 const (no apodization)
     * when spectral width is not set
    */
    spectral_width = std::numeric_limits<double>::max();
    apodization_values.clear();
}

apodization::apodization(std::string apodization_string)
{
    apodization_values.clear();
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
        /**
         * SP: Adjustable Sine Window. Follow nmrPipe nomenclature
        */
        if (apodization_string_split[0] == "sp") 
        {   
            /**
             * Example string "sp off 0.5 end 0.896 pow 3.684 elb 0.0 c 0.5"
             * sp: apodization function name, must be sp for adjustable sine window
             * off: offset, default is 0.5 p1
             * end: end, default is 0.95 p2
             * pow: power, default is 2.0 p3
             * elb: exponential widen, default is 0.0 p4
             * c: first point rescale factor, default is 0.5 p5
             */
            {
                apodization_type = FID_APODIZATION_SP;
                /**
                 * Set up default values. p6 is not used
                 */
                p1 = 0.5;
                p2 = 0.95;
                p3 = 2.0;
                p4 = 0.0;
                p5 = 0.5;
                /**
                 * Check the input string and set up the values. 
                 */
                for(int i=1;i<apodization_string_split.size();i+=2)
                {
                    if(apodization_string_split[i]=="off")
                    {
                        p1 = std::stod(apodization_string_split[i+1]);
                    }
                    else if(apodization_string_split[i]=="end")
                    {
                        p2 = std::stod(apodization_string_split[i+1]);
                    }
                    else if(apodization_string_split[i]=="pow")
                    {
                        p3 = std::stod(apodization_string_split[i+1]);
                    }
                    else if(apodization_string_split[i]=="elb")
                    {
                        p4 = std::stod(apodization_string_split[i+1]);
                    }
                    else if(apodization_string_split[i]=="c")
                    {
                        p5 = std::stod(apodization_string_split[i+1]);
                    }
                }
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
            std::cerr << "Error: apodization function name must be sp or none." << std::endl;
            return;
        }
    }
}

apodization::apodization(FID_APODIZATION_TYPE apodization_type_, double p1_, double p2_, double p3_, double p4_, double p5_, double p6_=0.0)
{
    apodization_values.clear();
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
    apodization_values.clear();
}

/**
 * Generate adjustable sine window function
 * @param ndata_: number of valid time domain data points (padding should not been included)
*/
bool apodization::set_n(int ndata_)
{
    /**
     * Set up apodization_values. For SP, formular is
     * SP[i] = sin( (PI*off + PI*(end-off)*i/(tSize-1) )^pow
     * the EM part is
     * EM[i] = exp( -PI*i*lb/sw )
     */
    apodization_values.clear(); // clear apodization_values first in case it has been set up before
    /**
     * special case for first point: 
     * Can be overwritten by calling set_first_point() after set_n()
    */
    apodization_values.push_back(p5);
    for (int j = 1; j < ndata_; j++)
    {
        double em = exp(-M_PI * j * p4 / spectral_width);
        double shifted_sine = pow(sin(M_PI * p1 + M_PI * (p2-p1) / (ndata_-1) * j), p3);
        apodization_values.push_back(em*shifted_sine);
    }

    return true;
}

bool apodization::run_apodization(float *data, int ndata_, bool b_complex) const
{
    if(b_complex)
    {
        for (int i = 0; i < ndata_/2; i++)
        {
            data[i * 2] *= apodization_values[i];
            data[i * 2 + 1] *= apodization_values[i];
        }
    }
    else
    {
        for (int i = 0; i < ndata_; i++)
        {
            data[i] *= apodization_values[i];
        }
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


bool fid_base::process_jcamp_as_string(const std::string &contents, std::map<std::string, std::string> &udict) const
{

    /**
     * This function will read the contents of a JCAMP file and store the key-value pairs in udict
     * The JCAMP file is expected to have the format:
     * ##key1=value1
     * ##key2=value2
     * ...
     */
    std::istringstream infile(contents);
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
                std::cerr << "Warning: cannot parse line " << line << std::endl;
            }
        }
        /**
         * other lines are undefined
         */
        else
        {
            std::cerr << "Warning: unknown line " << line << std::endl;
        }
    }

    return true;
}


/**
 * @brief read_jcmap_line: read one line of jcamp file and store the content in key and value
 * In rare cases, the value may have "<" but without ">", which means a multi-line value
 * In this case, we need to read the next line and append to the value
 */
bool fid_base::read_jcmap_line(std::istream &infile, std::string line, std::string &key, std::string &value) const
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

float fid_base::read_float(FILE *fp)
{
    char buff[4];
    fread(buff, 4, 1, fp); // dimension
    std::swap(buff[0], buff[3]);
    std::swap(buff[1], buff[2]);
    return *((float *)buff);
};

bool fid_base::read_float(FILE *fp, int n, float *pf)
{
    fread(pf, 4, n, fp);
    char *buff = (char *)pf;
    for (int i = 0; i < n; i++)
    {
        std::swap(buff[0 + i * 4], buff[3 + i * 4]);
        std::swap(buff[1 + i * 4], buff[2 + i * 4]);
    }
    return true;
}

int fid_base::read_int(FILE *fp)
{
    char buff[4];
    fread(buff, 4, 1, fp); // dimension
    std::swap(buff[0], buff[3]);
    std::swap(buff[1], buff[2]);
    return *((int *)buff);
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

    user_scale = 5.5;
    user_scale2 = 3.0;
    noise_level = 0.01;
    ndata = 0;
    ndata_bruker = 0;
    ndata_original = 0;
    ndata_power_of_2 = 0;
    ndata_frq = 0;
    spectral_width = 0.0;
    observed_frequency = 0.0;
    origin = 0.0;
    begin1 = 0.0;
    step1 = 0.0;
    stop1 = 0.0;
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


bool fid_1d::read_bruker_files_as_strings(const std::string &contents_acqus)
{
    return process_jcamp_as_string(contents_acqus, udict_acqus) && process_dictionary();      
}

bool fid_1d::set_fid_data(const std::vector<float> &fid_data_float_)
{
    fid_data_float = std::move(fid_data_float_);
    nspectra = fid_data_float.size()/ ndata_bruker; // number of spectra is the size of fid_data_float divided by ndata_bruker
    std::cout<<"Set fid_data_float with " << fid_data_float.size() << " elements, which is " << nspectra << " spectra." << std::endl;
    std::cout<<"ndata_bruker is " << ndata_bruker << std::endl;
    return true;
}

bool fid_1d::read_bruker_acqus_and_fid(const std::string &acqus_file_name, const std::vector<std::string> &fid_data_file_names)
{

    read_jcamp(acqus_file_name, udict_acqus);

    process_dictionary();

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

    

    return true;
}


bool fid_1d::process_dictionary()
{
    b_read_from_ft1 = false; // default, we are not reading from ft1 file
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
        ndata_original = td0 / 2;
    }
    else
    {
        ndata = ndata_bruker;
        ndata_original = td0;
    }

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
    apod->set_sw(spectral_width);
    apod->set_n(ndata_original);
    

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
       
        apod->run_apodization(temp_fid_data_float.data(), ndata*2, true /**complex data*/);
        
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


bool fid_1d::set_up_apodization_from_string(const std::string &apod_string)
{
    apod = new apodization(apod_string);
    return true;
}



/**
 * @brief create_nmrpipe_dictionary: create nmrpipe dictionary from udict_acqus and derived values
 * @param b_frq: true if we are saving frq data, false if we are saving time data
 * @param nmrpipe_dict_string: output dictionary for string values
 * @param nmrpipe_dict_float: output dictionary for float values
 */
bool fid_1d::create_nmrpipe_dictionary(bool b_frq, std::map<std::string, std::string> &nmrpipe_dict_string, std::map<std::string, float> &nmrpipe_dict_float) const
{
    /**
     * @brief create_default_nmrpipe_dictionary: create empty nmrpipe header
     */
    nmrPipe::create_default_nmrpipe_dictionary(nmrpipe_dict_string, nmrpipe_dict_float);

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

        /**
         * ft2 file also keep original FID data size
         * But I am not sure nmrPipde does the same!!
        */
        nmrpipe_dict_float["FDF2TDSIZE"] = ndata;
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
     * We do not need to do this when reading from nmrPipe ft1 file (where header is read from file.)
     */
    if(b_read_from_ft1 == false)
    {
        nmrpipe_header_data.clear();
        nmrpipe_header_data.resize(512, 0.0f);
    }

    /**
     * Even when we read from .ft1 file, we may still change nmrpipe_header_data if warranted.
    */
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


bool fid_1d::init(double user_, double user2_, double noise_)
{
    user_scale = user_;
    user_scale2 = user_;
    noise_level = noise_;
    return true;
};


/**
 * @brief write spectrum in josn format as an array, with two or three columns: ppm and intensity (and image)
*/
bool fid_1d::write_spectrum_json(std::string outfname)
{
    std::ofstream outfile;
    outfile.open(outfname);
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << outfname << std::endl;
        return false;
    }

    Json::Value root, data;

    for (int j = 0; j < ndata_frq; j++)
    {
        data[j][0] = begin1 + j * step1;
        data[j][1] = spectrum_real[j];
        /**
         * @brief if spe_image has same size as spect, save it as well
        */
        if(spectrum_imag.size()==ndata_frq)
        {
            data[j][2] = spectrum_imag[j];
        }
    }
    root["spectrum"] = data;
    outfile << root;
    return true;
}


/**
 * @brief read frq domain spectrum from three vectors (buffers) in pipe format
 * Similar to read_spectrum_ft, but get the header and spectrum from three vectors instead of a file
*/
bool fid_1d::direct_set_spectrum_from_nmrpipe(const std::vector<float> &_header, const std::vector<float> &real, const std::vector<float> &imag)
{

    /**
     * copy _header (vector) to header (float *512)
    */
    for (int i = 0; i < 512; i++)
    {
        nmrpipe_header_data[i] = _header[i];
    }

    if (nmrpipe_header_data[10 - 1] != 1.0f)
    {
        std::cout << "Wrong file format, dimension (header[9]) is " << nmrpipe_header_data[9] << std::endl;
        return false;
    }

    ndata_frq = int(nmrpipe_header_data[220 - 1]) * int(nmrpipe_header_data[100 - 1]); // one of them is 1, the other one is true dimension

    spectral_width = double(nmrpipe_header_data[101 - 1]);
    observed_frequency = double(nmrpipe_header_data[120 - 1]);
    origin = double(nmrpipe_header_data[102 - 1]);
    carrier_frequency = double(nmrpipe_header_data[67 - 1]) * observed_frequency;

    /**
     * Sometimes, original acqusition size is needed. 
    */
    ndata = int(nmrpipe_header_data[386]);
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    stop1 = origin / observed_frequency;
    begin1 = stop1 + spectral_width / observed_frequency;
    step1 = (stop1 - begin1) / (ndata_frq); // direct dimension
    begin1 += step1;                    // here we have to + step because we want to be consistent with nmrpipe program
                                        // I have no idea why nmrpipe is different than topspin

    spectrum_real=real;
    spectrum_imag=imag;
    
    if(n_verbose>0)
    {
        std::cout << "Spectrum size is " << ndata_frq << std::endl;
        std::cout << "From " << begin1 << " to " << stop1 << " and step is " << step1 << std::endl;
    }

    if (noise_level < 1e-20)
    {
        est_noise_level();
    }

    return true;
}

/**
 * @brief read 1D spectrum
 * 
 * @param infname input file name
 * @param b_negative true: allow negative peaks, false: only positive peaks. Default is true
 * noise_level is estimated if it is zero
 * @return true 
 */

bool fid_1d::read_spectrum(std::string infname, bool b_negative)
{
    bool b_read = 0;

    input_spectrum_fname = infname; // save for later use

    std::string sldw(".ldw"); // my own format
    std::string stxt(".txt"); // topspin format. saved by totxt command
    std::string sft1(".ft1"); // nmrPipe format
    std::string sjson(".json"); // json format
    std::string scsv(".csv"); // csv format, used by Mnova software
    std::string sparky(".ucsf"); // sparky format, used by sparky software

    if (std::equal(stxt.rbegin(), stxt.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_txt(infname);
    }
    else if (std::equal(sldw.rbegin(), sldw.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_ldw(infname);
    }
    else if (std::equal(sft1.rbegin(), sft1.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_ft(infname);
    }
    else if (std::equal(sjson.rbegin(), sjson.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_json(infname);
    }
    else if (std::equal(scsv.rbegin(), scsv.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_csv(infname);
    }
    else if (std::equal(sparky.rbegin(), sparky.rend(), infname.rbegin()))
    {
        b_read = read_spectrum_sparky(infname);
    }
    else
    {
        b_read = false;
    }

    if(n_verbose>0)
    {
        std::cout << "Spectrum size is " << ndata_frq << std::endl;
        std::cout << "From " << begin1 << " to " << stop1 << " and step is " << step1 << std::endl;
    }

    if (noise_level < 1e-20)
    {
        est_noise_level();
    }

    if(b_negative==false)
    {
        std::cout<<"Set negative data points to zero."<<std::endl;
        for(int i=0;i<spectrum_real.size();i++)
        {
            spectrum_real[i]=std::max(spectrum_real[i],0.0f);
        }
    }

    return b_read;
};





/**
 * @brief read spectrum in csv format by Mnova software
*/
bool fid_1d::read_spectrum_csv(std::string fname)
{
    std::ifstream fin(fname);
    if (!fin)
        return false;

    std::string line;
    std::vector<std::string> line_split;
    std::stringstream ss;
    std::vector<float> ppm;
    std::vector<float> amplitude;

    while (std::getline(fin, line))
    {   
        /**
         * Remove leading or trailing spaces
         * Return false if the line is empty
        */
        if(ldw_math_spectrum_1d::trim(line)==false)
        {
            continue;
        }
        /**
         * Skip comment lines, start with #
        */
        if (line[0] == '#')
        {
            continue;
        }

        /**
         * Split line by space(s)
        */
        line_split.clear();
        ss.clear();
        ss.str(line);
        std::string temp;
        while (ss >> temp)
        {
            line_split.push_back(temp);
        }

        /**
         * The first column is ppm, the second column is amplitude. Ignore the rest
        */
        if(line_split.size()>=2)
        {
            ppm.push_back(std::stof(line_split[0]));
            amplitude.push_back(std::stof(line_split[1]));
        }
    }

    ndata_frq = amplitude.size();
    begin1 = ppm[0];
    stop1 = ppm[ndata_frq - 1];
    step1 = (stop1 - begin1) / (ndata_frq - 1);

    observed_frequency = 850.0; //default value, suppose it is 850
    carrier_frequency = 4.7; //default value, suppose it is 4.7 ppm

    spectrum_real = amplitude;

    ndata_frq = amplitude.size();

    /**
     * In this case, we suppose n_zf is 2
    */
    ndata = ndata_frq/2;
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    return true;

}
/**
 * @brief read spectrum in json format by Gissmo project
*/
bool fid_1d::read_spectrum_json(std::string infname)
{

    Json::Value root;
    std::ifstream fin(infname);
    if (!fin)
        return false;

    fin >> root;

    Json::Value data1, data2;
    data1 = root[0]; // ppm
    data2 = root[1]; // amplitude

    std::vector<float> ppm;

    for (int i = 0; i < data2.size(); i += 1)
    {
        if(data2[i].isDouble()==true)
        {
            spectrum_real.push_back(data2[i].asDouble());
        }
        else if(data2[i].isString()==true)
        {
            spectrum_real.push_back(std::stof(data2[i].asString()));
        }
        else
        {
            std::cout<<"Error: fid_1d::read_spectrum_json, data2[i] is not double or string."<<std::endl;
            return false;
        }

        if(data1[i].isDouble()==true)
        {
            ppm.push_back(data1[i].asDouble());
        }
        else if(data1[i].isString()==true)
        {
            ppm.push_back(std::stof(data1[i].asString()));
        }
        else
        {
            std::cout<<"Error: fid_1d::read_spectrum_json, data1[i] is not double or string."<<std::endl;
            return false;
        }
    }
    // std::reverse(spe.begin(),spe.end());

    ndata_frq = spectrum_real.size();
    begin1 = ppm[0];
    stop1 = ppm[ndata_frq - 1];
    step1 = (stop1 - begin1) / (ndata_frq - 1);

    observed_frequency = 850.0; //default value, suppose it is 850
    carrier_frequency = 4.7; //default value, suppose it is 4.7 ppm

    ndata_frq = spectrum_real.size();
    /**
     * In this case, we suppose n_zf is 2
    */
    ndata = ndata_frq/2;
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    return true;
};

/**
 * This function will replace fid_1d::read_spectrum(), but only support reading nmrPipe ft1 file and means for web assembly to direct set spectrum data from memory buffer
*/
bool fid_1d::read_first_spectrum_from_buffer(std::vector<float> &header, std::vector<float> &spectrum_real_, std::vector<float> &spectrum_imag)
{
    /**
     * copy first 512 float32 to nmrpipe_header_data
     */
    if(header.size()<512)
    {
        std::cout << "Error: buffer size is less than 512, cannot read nmrPipe header." << std::endl;
        return false;
    }
    /**
     * C++17 allows us to use std::move to avoid copying the header data. Simply transfer ownership of the header vector to nmrpipe_header_data.
    */
    nmrpipe_header_data = std::move(header); 

    process_nmrpipe_header();  

    spectrum_real = std::move(spectrum_real_); // move the spectrum_real vector to the class member

    if(spectrum_imag.size()>0)
    {
        spectrum_imag = std::move(spectrum_imag); // move the spectrum_imag vector to the class member
    }
    else
    {
        spectrum_imag.clear(); // clear the imaginary part if it is empty
    }
   
    if (noise_level < 1e-20)
    {
        est_noise_level();
    }
    return true;
}

/**
 * @brief read 1D spectrum from nmrPipe ft1 file. Will try to read imaginary part as well
*/
bool fid_1d::read_spectrum_ft(std::string infname)
{
    FILE *fp;
    fp = fopen(infname.c_str(), "rb");
    if (fp == NULL)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }
    unsigned int temp = fread(nmrpipe_header_data.data(), sizeof(float), 512, fp);
    if (temp != 512)
    {
        std::cout << "Wrong file format, can't read 2048 bytes of head information from " << infname << std::endl;
        return false;
    }
    process_nmrpipe_header();    

    spectrum_real.clear();
    spectrum_real.resize(ndata_frq);
    temp = fread(spectrum_real.data(), sizeof(float), ndata_frq, fp);

    if (temp != ndata_frq)
    {
        std::cout << "Read nmrPipe 1D error, spectrum size is " << ndata_frq << " but I can only read in " << temp << " data points." << std::endl;
    }

    spectrum_imag.resize(ndata_frq);
    temp = fread(spectrum_imag.data(), sizeof(float), ndata_frq, fp);
    if (temp == ndata_frq)
    {
        if(n_verbose>0) std::cout << "Read nmrPipe 1D imaginary part successully." << std::endl;
    }
    else
    {
        if(n_verbose>0) std::cout << "Read nmrPipe 1D  imaginary part failed." << std::endl;
        spectrum_imag.clear();
    }

    fclose(fp);


    return true;
};

bool fid_1d::process_nmrpipe_header()
{
    if (nmrpipe_header_data[10 - 1] != 1.0f)
    {
        std::cout << "Wrong file format, dimension (header[9]) is " << nmrpipe_header_data[0] << std::endl;
        return false;
    }

    if (nmrpipe_header_data[222 - 1] == 0.0f) // transposed?
    {
        ndata_frq = int(nmrpipe_header_data[100 - 1]); //FDSIZE
    }
    else
    {
        ndata_frq = int(nmrpipe_header_data[220 - 1]); //FDSPECNUM
    }

    ndata_frq = int(nmrpipe_header_data[220 - 1]) * int(nmrpipe_header_data[100 - 1]); // one of them is 1, the other one is true dimension

    spectral_width = double(nmrpipe_header_data[101 - 1]); //FDF2SW
    observed_frequency = double(nmrpipe_header_data[120 - 1]); //FDF2OBS
    origin = double(nmrpipe_header_data[102 - 1]); //FDF2ORIG
    carrier_frequency = double(nmrpipe_header_data[67 - 1]) * observed_frequency; //FDF2CAR

    /**
     * Sometimes, original acqusition size is needed. 
    */
    ndata = int(nmrpipe_header_data[386]);
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    stop1 = origin / observed_frequency;
    begin1 = stop1 + spectral_width / observed_frequency; 
    step1 = (stop1 - begin1) / (ndata_frq); 
    /**
     * here we have to + step because we want to be consistent with nmrpipe program
     * I have no idea why nmrpipe is different than topspin
    */
    begin1 += step1;  

    b_read_from_ft1 = true; // set this flag to true, so we know we read from ft1 file

    return true;
}


/**
 * @brief write 1D spectrum to a file. Format is decided by file extension
 * .ft1 .json or .txt
*/
bool fid_1d::write_spectrum(std::string fname)
{
    bool b_write = false;
    std::string path_name, file_name, file_name_ext;
    /**
     * Get path name, file name and file extension
    */
    ldw_math_spectrum_1d::SplitFilename(fname, path_name, file_name, file_name_ext);

    if(file_name_ext=="ft1")
    {
        b_write=write_nmrpipe_ft1(fname);
    }
    else if(file_name_ext=="json")
    {
        b_write=write_spectrum_json(fname);
    }
    else if(file_name_ext=="txt")
    {
        b_write=write_spectrum_txt(fname);
    }
    else if(file_name_ext=="csv")
    {
        b_write=write_spectrum_csv(fname);
    }
    else
    {
        std::cout<<"Error: fid_1d::write_spectrum, unknown file extension "<<file_name_ext<<std::endl;
        return false;
    }
    return b_write;
}



//simple text file format, defined by myself
bool fid_1d::read_spectrum_ldw(std::string infname)
{
    std::ifstream fin(infname);

    float data;
    while (fin >> data)
    {
        spectrum_real.push_back(data);
        // fin>>data;
    }
    fin.close();

    ndata_frq = spectrum_real.size() - 2;

    stop1 = spectrum_real[ndata_frq + 1];
    begin1 = spectrum_real[ndata_frq];
    step1 = (stop1 - begin1) / ndata_frq;

    /**
     * In this case, we suppose n_zf is 2
    */
    ndata = ndata_frq/2;
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    spectrum_real.resize(ndata_frq);

    return true;
}

//ascii file saved by Topspin totxt command
bool fid_1d::read_spectrum_txt(std::string infname)
{
    std::string line;
    std::ifstream fin(infname);

    bool b_left = false;
    bool b_right = false;
    bool b_size = false;

    spectrum_real.clear();

    //read line by line
    while (std::getline(fin, line))
    {
        //if line is empty, skip
        if (line.empty())
        {
            continue;
        }

        /**
         * Example header:
         * # LEFT = 12.764570236206055 ppm. RIGHT = -3.217077331696382 ppm.
         * #
         * # SIZE = 65536 ( = number of points)
        */

        //if line starts with #, look for key words LEFT, RIGHT in one line and key words SIZE in another line
        if (line[0] == '#')
        {
            if (line.find("LEFT") != std::string::npos && line.find("RIGHT") != std::string::npos)
            {
                std::string temp;
                //get substring after LEFT
                std::string line_part1 = line.substr(line.find("LEFT") + 4 + 1); // + 1 to skip space
                std::istringstream iss1(line_part1);
                iss1 >> temp  >> begin1; //LEFT = 12.09875 ppm
                b_left = true;
            
                std::string line_part2 = line.substr(line.find("RIGHT") + 5 + 1); // + 1 to skip space
                std::istringstream iss2(line_part2);
                iss2 >> temp >> stop1; //RIGHT = -3.1234 ppm
                b_right = true;
            }

            if(line.find("SIZE") != std::string::npos)
            {
                std::string temp;
                std::string line_part = line.substr(line.find("SIZE") + 4 + 1); // + 1 to skip space
                std::istringstream iss(line_part);
                iss >> temp >> ndata_frq; // SIZE = 1024
                b_size = true;
            }
            continue;
        }

        /* *All other lines are data and they are all float
        * They should be after the two lines with key words LEFT, RIGHT and SIZE
        * One number per line
        * 
        * Example: (second part is for imaginary part, may not exist)
        * -3747.75-17118.0i
        * -3277.5+17781.5i
        * 2807.25+17781.5i
        */
        if (b_left == true && b_right == true && b_size == true)
        {
            /**
             * First, decide if this is a complex spectrum
            */
            if(line.find("i") != std::string::npos)
            {
                /**
                 * remove the last character, which is i
                */
                line=line.substr(0,line.size()-1);
                std::string line_part=line;

                /**
                 * remove first character if it is a + or -
                */
                if(line_part[0]=='+' || line_part[0]=='-')
                {
                    line_part=line_part.substr(1);
                }

                /**
                 * seperate real and imaginary part, using the last + or - as delimiter
                 * We use line, not line_part, because we need to keep the sign of the first number
                */
                std::string real_part=line.substr(0,line.find_last_of("+-"));
                std::string imag_part=line.substr(line.find_last_of("+-"));
                
                /**
                 * convert string to float, and push back to spect and spe_image
                */
                spectrum_real.push_back(std::stof(real_part));
                /**
                * Reasons unknown, but the sign of the imaginary part is reversed in Bruker's txt file
                */
                spectrum_imag.push_back(-std::stof(imag_part));
            }
            else
            {
                /**
                 * This is a real spectrum. Push back to spect only
                */
                float data=std::stof(line);
                spectrum_real.push_back(data);
            }
        }
    }

    if(spectrum_real.size()!=ndata_frq)
    {
        std::cout<<"Error: fid_1d::read_spectrum_txt, ndata_frq is not equal to the number of data points. Set ndata_frq=spect.size()"<<std::endl;
        ndata_frq = spectrum_real.size();
    }

    if(spectrum_imag.size()!=ndata_frq)
    {
        std::cout<<"Error: fid_1d::read_spectrum_txt, ndata_frq is not equal to the number of data points. Remove imaginary data."<<std::endl;
        spectrum_imag.clear(); //spe_image.size()==0 is used to indicate that there is no imaginary part
    }

    fin.close();

    step1 = (stop1 - begin1) / ndata_frq;

    observed_frequency = 850.0; //default value, suppose it is 850
    carrier_frequency = 4.7; //default value, suppose it is 4.7 ppm

    /**
     * In this case, we suppose n_zf is 2
    */
    ndata = ndata_frq/2;
    ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

/**
 * For debug
 */
// #ifdef DEBUG
//     std::ofstream fout("spect.txt");
//     for (int i = 0; i < spect.size(); i++)
//     {
//         fout << spect[i] << " " << spe_image[i] << std::endl;
//     }
//     fout.close();
// #endif

    return true;
}


bool fid_1d::read_spectrum_sparky(std::string infname)
{
    FILE *fp;

    char buffer[10];
    int temp;
    float center1;

    fp = fopen(infname.c_str(), "rb");
    if (fp == NULL)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }

    fread(buffer, 1, 10, fp);
    // std::cout<<buffer<<std::endl;

    fread(buffer, 1, 1, fp);
    temp = int(buffer[0]);
    if (temp != 1)
    {
        std::cout << "Error in sparky format file, dimension is not 1" << std::endl;
        return false;
    }

    fread(buffer, 1, 1, fp);
    fseek(fp, 1, SEEK_CUR);
    temp = int(buffer[0]);
    if (temp != 1)
    {
        std::cout << "Error in sparky format file, it is not in real data" << std::endl;
        return false;
    }

    fread(buffer, 1, 1, fp);
    // std::cout<<"Version is "<< int(buffer[0])<<std::endl;
    fseek(fp, 166, SEEK_CUR);  //at location 180

    fread(buffer, 1, 6, fp); // nuleus name, at location 186
    std::cout << "Direct dimension nuleus " << buffer << std::endl;
    fseek(fp, 2, SEEK_CUR); //at 188
    ndata_frq = read_int(fp); //at 192
    fseek(fp, 8, SEEK_CUR); //at 200
    observed_frequency = read_float(fp); //at 204
    spectral_width = read_float(fp); //at 208
    center1 = read_float(fp); //at 212
    fseek(fp, 96, SEEK_CUR);  // at location 308

    /**
     * now read the spectrum data. Remember that sparky is in big-endian format
     */
    spectrum_real.clear();
    spectrum_real.resize(ndata_frq);
    for(int i=0;i<ndata_frq;i++)
    {
        float temp_float=read_float(fp);
        /**
         * convert from big-endian to little-endian
        */
        spectrum_real[i] = temp_float;
    }

    float range1 = spectral_width / observed_frequency;
    step1 = -range1 / ndata_frq;
    begin1 = center1 + range1 / 2;
    stop1 = center1 - range1 / 2;

    origin = center1 * observed_frequency - spectral_width / 2;

    std::cout << "Spectrum width are " << spectral_width << " Hz" << std::endl;
    std::cout << "Fields are " << observed_frequency << " mHz" << std::endl;
    std::cout << "Direct dimension size is " << ndata_frq  << std::endl;
    std::cout << "Direct dimension offset is " << begin1 << ", ppm per step is " << step1 << " ppm" << std::endl;

    fclose(fp);


    return true;
}

/**
 * @brief set spectrum from a vector of data, with ppm information
 * This is the minimal requirement for a spectrum to be used for picking and fitting.
 * It gather similar set of information as read from text file or json file. 
 * read ft1 will get more information, such as SW1, frq1, ref1, etc.
*/
bool fid_1d::set_spectrum_from_data(const std::vector<float> &data, const double begin_, const double step_, const double stop_)
{
    /**
     * Set spect and ndata_frq
    */
    spectrum_real=data;
    ndata_frq=spectrum_real.size();

    spectrum_imag.clear(); //spe_image.size()==0 is used to indicate that there is no imaginary part
    
    /**
     * Set begin1, step1, stop1 for ppm information
    */
    begin1=begin_;
    step1=step_;
    stop1=stop_;

    return true;
}

/**
 * @brief write 1D spectrum to a csv file
*/
bool fid_1d::write_spectrum_csv(std::string outfname)
{
    std::ofstream outfile;
    outfile.open(outfname);
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << outfname << std::endl;
        return false;
    }

    for (int j = 0; j < ndata_frq; j++)
    {
        outfile << begin1 + j * step1<<" "<< spectrum_real[j] << std::endl;
    }
    outfile.close();
    return true;
}

/**
 * @brief write 1D spectrum to a text file
*/
bool fid_1d::write_spectrum_txt(std::string outfname)
{
    std::ofstream outfile;
    outfile.open(outfname);
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << outfname << std::endl;
        return false;
    }

    outfile << "# LEFT = " << begin1 << " ppm. RIGHT = " << stop1 << " ppm." << std::endl;
    outfile << "#" << std::endl;
    outfile << "# SIZE = " << ndata_frq << " ( = number of points)" << std::endl;
    outfile << "#" << std::endl;

    if(spectrum_imag.size()==0)
    {
        for (int j = 0; j < ndata_frq; j++)
        {
            outfile << spectrum_real[j] << std::endl;
        }
    }
    else
    {
        for (int j = 0; j < ndata_frq; j++)
        {
            /**
             * Reasons unknown, but the sign of the imaginary part is reversed in Bruker's txt file
            */
            float temp_data=-spectrum_imag[j];
            if(temp_data>=0.0)
            { 
                outfile << spectrum_real[j] << "+" << temp_data << "i" << std::endl;
            }
            else
            {
                outfile << spectrum_real[j] << temp_data << "i" << std::endl;
            }
        }
    }
    outfile.close();
    return true;
}

/**
 * @brief estimate noise level using a general purpose method: segment by segment variance
*/
bool fid_1d::est_noise_level()
{

    std::vector<double> variances;
    std::vector<double> sums;
    int n_segment = ndata_frq / 32;
    for (int i = 0; i < n_segment; i++)
    {
        double sum = 0;
        for (int j = 0; j < 32; j++)
        {
            sum += spectrum_real[i * 32 + j];
        }
        sum /= 32;
        // get variance of each segment
        double var = 0;
        for (int j = 0; j < 32; j++)
        {
            var += (spectrum_real[i * 32 + j] - sum) * (spectrum_real[i * 32 + j] - sum);
        }
        var /= 32;

        variances.push_back(var);
        sums.push_back(sum);
    }

    // output the sums and variance of each segment
    //  std::ofstream fout("variances.txt");
    //  for(int i=0;i<variances.size();i++)
    //  {
    //      fout<<sums[i]<<" "<<variances[i]<<std::endl;
    //  }
    //  fout.close();

    int n = variances.size() / 4;
    nth_element(variances.begin(), variances.begin() + n, variances.end());
    noise_level = sqrt(variances[n]);
    if(n_verbose>0) std::cout << "Noise level is estiamted to be " << noise_level << ", using a geneal purpose method." << std::endl;

    return true;
}

/**
 * @brief estimate noise level using MAD method of the whole spectrum.
 * Won't work if the spectrum is not phased and baseline corrected
*/
bool fid_1d::est_noise_level_mad()
{

    int ndim = spectrum_real.size();

    if (noise_level < 1e-20) // estimate noise level
    {
        std::vector<float> t = spectrum_real;
        for (unsigned int i = 0; i < t.size(); i++)
        {
            if (t[i] < 0)
                t[i] *= -1;
        }

        std::vector<float> scores = t;
        sort(scores.begin(), scores.end());
        noise_level = scores[scores.size() / 2] * 1.4826;
        if (noise_level <= 0.0)
            noise_level = 0.1; // artificail spectrum w/o noise
        std::cout << "First round, noise level is " << noise_level << std::endl;

        std::vector<int> flag(ndim, 0); // flag

        for (int i = 0; i < ndim; i++)
        {
            if (t[i] > 5.5 * noise_level)
            {
                int xstart = std::max(i - 5, 0);
                int xend = std::min(i + 6, ndim);

                for (int n = xstart; n < xend; n++)
                {
                    flag[n] = 1;
                }
            }
        }
        scores.clear();

        for (int i = 0; i < ndim; i++)
        {
            if (flag[i] == 0)
            {
                scores.push_back(t[i]);
            }
        }

        sort(scores.begin(), scores.end());
        noise_level = scores[scores.size() / 2] * 1.4826;
        if (noise_level <= 0.0)
            noise_level = 0.1; // artificail spectrum w/o noise
        std::cout << "Final noise level is estiamted to be " << noise_level << std::endl;
    }

    return true;
};


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
    root["ndata_frq"] = ndata_frq;
    root["ndata_original"] = ndata_original;
    root["ndata_power_of_2"]=ndata_power_of_2; //We suppose ZF=2 in processing
    /**
     * ref1 is the end of the spectrum
     * carrier_frequency is the middle of the spectrum
     * SW is the total width of the spectrum
     * All in Hz.
     * If read in from a file other than ft1, all will be set to 0
    */
    root["carrier_frequency"] = carrier_frequency;
    root["observed_frequency"]=observed_frequency;
    root["spectral_width"]=spectral_width; //in Hz

    outfile << root << std::endl;
    outfile.close();

    return true;
}

std::string fid_1d::write_json_as_string()
{
    Json::Value root;
    root["ndata"] = ndata;
    root["ndata_frq"] = ndata_frq;
    root["ndata_original"] = ndata_original;
    root["ndata_power_of_2"]=ndata_power_of_2; //We suppose ZF=2 in processing
    /**
     * ref1 is the end of the spectrum
     * carrier_frequency is the middle of the spectrum
     * SW is the total width of the spectrum
     * All in Hz.
     * If read in from a file other than ft1, all will be set to 0
    */
    root["carrier_frequency"] = carrier_frequency;
    root["observed_frequency"]=observed_frequency;
    root["spectral_width"]=spectral_width; //in Hz

    Json::StreamWriterBuilder writer;
    std::string output = Json::writeString(writer, root);
    
    return output;
}

/**
 * To save memory after peaks picking or fitting. User by 3D picker/fitter classes
*/
bool fid_1d::release_spectrum()
{
    spectrum_real.clear();
    return true;
}

/**
 * @brief get spectrum as a read-only vector
*/
const std::vector<float> & fid_1d::get_spectrum() const
{
    return spectrum_real;
}

#ifdef WEBASSEMBLY

/**
 * Function calls for web assembly to read header and data directly from variables without addtional copy
 */
uintptr_t fid_1d::get_data_of_real()
{
    return reinterpret_cast<uintptr_t>(spectrum_real.data());
}
uintptr_t fid_1d::get_data_of_imag()
{
    return reinterpret_cast<uintptr_t>(spectrum_imag.data());
}
uintptr_t fid_1d::get_data_of_header()
{
    return reinterpret_cast<uintptr_t>(nmrpipe_header_data.data());
}


/**
 * Exposed functions
*/
EMSCRIPTEN_BINDINGS(dp_1d_module_fid) {

    class_<fid_base>("fid_base")
        .constructor();

    class_<shared_data_1d>("shared_data_1d")
        .constructor()
        .class_property("n_verbose", &shared_data_1d::n_verbose);


    class_<fid_1d, base<fid_base> >("fid_1d")
        .constructor()
        .function("init", &fid_1d::init)
        .class_property("n_verbose", &shared_data_1d::n_verbose)
        .function("read_first_spectrum_from_buffer", &fid_1d::read_first_spectrum_from_buffer) //for dp_1d and vf_1d
        //below are for fid processing
        .function("set_up_apodization_from_string", &fid_1d::set_up_apodization_from_string)
        .function("read_bruker_files_as_strings", &fid_1d::read_bruker_files_as_strings)
        .function("get_fid_data_type", &fid_1d::get_fid_data_type) //to get data type of fid (float64 or int32)
        .function("set_fid_data", &fid_1d::set_fid_data) //to process fid
        .function("run_zf", &fid_1d::run_zf)
        .function("run_fft_and_rm_bruker_filter", &fid_1d::run_fft_and_rm_bruker_filter)
        .function("write_json_as_string", &fid_1d::write_json_as_string) 
        .function("get_nspectra", &fid_1d::get_nspectra)
        .function("get_ndata_frq", &fid_1d::get_ndata_frq)
        .function("write_nmrpipe_ft1", &fid_1d::write_nmrpipe_ft1) //write nmrPipe ft1 file
        .function("get_data_of_header", &fid_1d::get_data_of_header)
        .function("get_data_of_real", &fid_1d::get_data_of_real)
        .function("get_data_of_imag", &fid_1d::get_data_of_imag)
        ;

}
#endif // WEBASSEMBLY
