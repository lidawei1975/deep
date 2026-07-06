#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>
#include <iostream>
#include <limits>
#include <algorithm>
#include <cmath>

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
#include "fid_base.h"

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
        {"FDF1ZF", "437"},

        /**
         * Define by Dawei by my best guess. Not in official nmrPipe documentation.
         * In fid2pipe command line, "-AMX" means delayed processing of bruker digitizer filter, 
         * and the grpdly value from bruker files is saved here.create_default_nmrpipe_dictionary
         * "-DMX" means appriximated method to fix in the conversion process, 
         * there is no need to do additonal processing in nmrpipe command.
         * Therefore, this value is not stored in the nmrPipe header.
         */
        {"GRPDLY", "40"}
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
    char buff[4] = {0};
    if (fread(buff, 4, 1, fp) != 1)
    {
        std::cerr << "Error: fread failed in read_float" << std::endl;
    }
    std::swap(buff[0], buff[3]);
    std::swap(buff[1], buff[2]);
    return *((float *)buff);
};

bool fid_base::read_float(FILE *fp, int n, float *pf)
{
    if (fread(pf, 4, n, fp) != static_cast<size_t>(n))
    {
        std::cerr << "Error: fread failed in read_float batch" << std::endl;
        return false;
    }
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
    char buff[4] = {0};
    if (fread(buff, 4, 1, fp) != 1)
    {
        std::cerr << "Error: fread failed in read_int" << std::endl;
    }
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
