#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "json/json.h"
#include "commandline.h"

#include "spectrum_io_1d.h"


/**
 * These are shared varibles between db_match_1d and pattern_match_1d and spectrum_pick_1d
*/
int shared_data_1d::n_verbose=0; 
bool shared_data_1d::b_doesy=false;

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
}

spectrum_io_1d::spectrum_io_1d()
{
    b_header = false;
    user_scale = 5.5;
    user_scale2 = 3.0;
    noise_level = 0.01;
    ndata = 0;
    SW1 = 0.0;
    frq1 = 0.0;
    ref1 = 0.0;
    begin1 = 0.0;
    step1 = 0.0;
    stop1 = 0.0;
};
spectrum_io_1d::~spectrum_io_1d(){};

bool spectrum_io_1d::init(double user_, double user2_, double noise_)
{
    user_scale = user_;
    user_scale2 = user_;
    noise_level = noise_;
    return true;
};


/**
 * @brief write spectrum in josn format as an array, with two or three columns: ppm and intensity (and image)
*/
bool spectrum_io_1d::write_spectrum_json(std::string outfname)
{
    std::ofstream outfile;
    outfile.open(outfname);
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << outfname << std::endl;
        return false;
    }

    Json::Value root, data;

    for (int j = 0; j < ndata; j++)
    {
        data[j][0] = begin1 + j * step1;
        data[j][1] = spect[j];
        /**
         * @brief if spe_image has same size as spect, save it as well
        */
        if(spe_image.size()==ndata)
        {
            data[j][2] = spe_image[j];
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
bool spectrum_io_1d::direct_set_spectrum_from_nmrpipe(const std::vector<float> &_header, const std::vector<float> &real, const std::vector<float> &imag)
{

    /**
     * copy _header (vector) to header (float *512)
    */
    for (int i = 0; i < 512; i++)
    {
        header[i] = _header[i];
    }
    b_header = true;

    if (header[10 - 1] != 1.0f)
    {
        std::cout << "Wrong file format, dimension (header[9]) is " << header[9] << std::endl;
        return false;
    }

    ndata = int(header[220 - 1]) * int(header[100 - 1]); // one of them is 1, the other one is true dimension

    SW1 = double(header[101 - 1]);
    frq1 = double(header[120 - 1]);
    ref1 = double(header[102 - 1]);

    stop1 = ref1 / frq1;
    begin1 = stop1 + SW1 / frq1;
    step1 = (stop1 - begin1) / (ndata); // direct dimension
    begin1 += step1;                    // here we have to + step because we want to be consistent with nmrpipe program
                                        // I have no idea why nmrpipe is different than topspin

    spect=real;
    spe_image=imag;
    
    if(n_verbose>0)
    {
        std::cout << "Spectrum size is " << ndata << std::endl;
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

bool spectrum_io_1d::read_spectrum(std::string infname, bool b_negative)
{
    bool b_read = 0;

    input_spectrum_fname = infname; // save for later use

    std::string sldw(".ldw"); // my own format
    std::string stxt(".txt"); // topspin format. saved by totxt command
    std::string sft1(".ft1"); // nmrPipe format
    std::string sjson(".json"); // json format
    std::string scsv(".csv"); // csv format, used by Mnova software

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
    else
    {
        b_read = false;
    }

    if(n_verbose>0)
    {
        std::cout << "Spectrum size is " << ndata << std::endl;
        std::cout << "From " << begin1 << " to " << stop1 << " and step is " << step1 << std::endl;
    }

    if (noise_level < 1e-20)
    {
        est_noise_level();
    }

    if(b_negative==false)
    {
        std::cout<<"Set negative data points to zero."<<std::endl;
        for(int i=0;i<spect.size();i++)
        {
            spect[i]=std::max(spect[i],0.0f);
        }
    }

    return b_read;
}
/**
 * @brief read spectrum in csv format by Mnova software
*/
bool spectrum_io_1d::read_spectrum_csv(std::string fname)
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

    ndata = amplitude.size();
    begin1 = ppm[0];
    stop1 = ppm[ndata - 1];
    step1 = (stop1 - begin1) / (ndata - 1);

    spect = amplitude;

    return true;

}
/**
 * @brief read spectrum in json format by Gissmo project
*/
bool spectrum_io_1d::read_spectrum_json(std::string infname)
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
            spect.push_back(data2[i].asDouble());
        }
        else if(data2[i].isString()==true)
        {
            spect.push_back(std::stof(data2[i].asString()));
        }
        else
        {
            std::cout<<"Error: spectrum_io_1d::read_spectrum_json, data2[i] is not double or string."<<std::endl;
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
            std::cout<<"Error: spectrum_io_1d::read_spectrum_json, data1[i] is not double or string."<<std::endl;
            return false;
        }
    }
    // std::reverse(spe.begin(),spe.end());

    ndata = spect.size();
    begin1 = ppm[0];
    stop1 = ppm[ndata - 1];
    step1 = (stop1 - begin1) / (ndata - 1);
    return true;
};

/**
 * @brief read 1D spectrum from nmrPipe ft1 file. Will try to read imaginary part as well
*/
bool spectrum_io_1d::read_spectrum_ft(std::string infname)
{
    FILE *fp;
    fp = fopen(infname.c_str(), "rb");
    if (fp == NULL)
    {
        std::cout << "Can't open " << infname << " to read." << std::endl;
        return false;
    }
    unsigned int temp = fread(header, sizeof(float), 512, fp);
    if (temp != 512)
    {
        std::cout << "Wrong file format, can't read 2048 bytes of head information from " << infname << std::endl;
        return false;
    }

    if (header[10 - 1] != 1.0f)
    {
        std::cout << "Wrong file format, dimension (header[9]) is " << header[0] << std::endl;
        return false;
    }

    if (header[222 - 1] == 0.0f) // transposed?
    {
        ndata = int(header[100 - 1]);
    }
    else
    {
        ndata = int(header[220 - 1]);
    }

    ndata = int(header[220 - 1]) * int(header[100 - 1]); // one of them is 1, the other one is true dimension

    SW1 = double(header[101 - 1]);
    frq1 = double(header[120 - 1]);
    ref1 = double(header[102 - 1]);

    stop1 = ref1 / frq1;
    begin1 = stop1 + SW1 / frq1;
    step1 = (stop1 - begin1) / (ndata); // direct dimension
    begin1 += step1;                    // here we have to + step because we want to be consistent with nmrpipe program
                                        // I have no idea why nmrpipe is different than topspin

    spect.clear();
    spect.resize(ndata);
    temp = fread(spect.data(), sizeof(float), ndata, fp);

    if (temp != ndata)
    {
        std::cout << "Read nmrPipe 1D error, spectrum size is " << ndata << " but I can only read in " << temp << " data points." << std::endl;
    }

    spe_image.resize(ndata);
    temp = fread(spe_image.data(), sizeof(float), ndata, fp);
    if (temp == ndata)
    {
        if(n_verbose>0) std::cout << "Read nmrPipe 1D imaginary part successully." << std::endl;
    }
    else
    {
        if(n_verbose>0) std::cout << "Read nmrPipe 1D  imaginary part failed." << std::endl;
        spe_image.clear();
    }

    fclose(fp);

    b_header = true;

    return true;
};

/**
 * @brief write 1D spectrum to a file. Format is decided by file extension
 * .ft1 .json or .txt
*/
bool spectrum_io_1d::write_spectrum(std::string fname)
{
    bool b_write = false;
    std::string path_name, file_name, file_name_ext;
    /**
     * Get path name, file name and file extension
    */
    ldw_math_spectrum_1d::SplitFilename(fname, path_name, file_name, file_name_ext);

    if(file_name_ext=="ft1")
    {
        b_write=write_spectrum_ft1(fname);
    }
    else if(file_name_ext=="json")
    {
        b_write=write_spectrum_json(fname);
    }
    else if(file_name_ext=="txt")
    {
        b_write=write_spectrum_txt(fname);
    }
    else
    {
        std::cout<<"Error: spectrum_io_1d::write_spectrum, unknown file extension "<<file_name_ext<<std::endl;
        return false;
    }
    return b_write;
}

/**
 * @brief write 1D spectrum from nmrPipe ft1 file. Write image part too if it exists
 * This function will not work if the spectrum is not read from a ft1 file
 * ( Please use fid_1d class to write a spectrum to ft1 file in that case )
*/
bool spectrum_io_1d::write_spectrum_ft1(std::string infname)
{
    if (b_header == false)
    {
        std::cout << "Warning: PIpe format header has not been read." << std::endl;
    }

    FILE *fp;
    fp = fopen(infname.c_str(), "wb");
    fwrite(header, sizeof(float), 512, fp);
    fwrite(spect.data(), sizeof(float), ndata, fp);
    if (spe_image.size() > 0)
    {
        fwrite(spe_image.data(), sizeof(float), ndata, fp);
    }
    fclose(fp);

    return true;
}



//simple text file format, defined by myself
bool spectrum_io_1d::read_spectrum_ldw(std::string infname)
{
    std::ifstream fin(infname);

    float data;
    while (fin >> data)
    {
        spect.push_back(data);
        // fin>>data;
    }
    fin.close();

    ndata = spect.size() - 2;

    stop1 = spect[ndata + 1];
    begin1 = spect[ndata];
    step1 = (stop1 - begin1) / ndata;

    spect.resize(ndata);

    return true;
}

//ascii file saved by Topspin totxt command
bool spectrum_io_1d::read_spectrum_txt(std::string infname)
{
    std::string line;
    std::ifstream fin(infname);

    bool b_left = false;
    bool b_right = false;
    bool b_size = false;

    spect.clear();

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
                iss >> temp >> ndata; // SIZE = 1024
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
                spect.push_back(std::stof(real_part));
                /**
                * Reasons unknown, but the sign of the imaginary part is reversed in Bruker's txt file
                */
                spe_image.push_back(-std::stof(imag_part));
            }
            else
            {
                /**
                 * This is a real spectrum. Push back to spect only
                */
                float data=std::stof(line);
                spect.push_back(data);
            }
        }
    }

    if(spect.size()!=ndata)
    {
        std::cout<<"Error: spectrum_io_1d::read_spectrum_txt, ndata is not equal to the number of data points. Set ndata=spect.size()"<<std::endl;
        ndata = spect.size();
    }

    if(spe_image.size()!=ndata)
    {
        std::cout<<"Error: spectrum_io_1d::read_spectrum_txt, ndata is not equal to the number of data points. Remove imaginary data."<<std::endl;
        spe_image.clear(); //spe_image.size()==0 is used to indicate that there is no imaginary part
    }

    fin.close();

    step1 = (stop1 - begin1) / ndata;

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

/**
 * @brief set spectrum from a vector of data, with ppm information
 * This is the minimal requirement for a spectrum to be used for picking and fitting.
 * It gather similar set of information as read from text file or json file. 
 * read ft1 will get more information, such as SW1, frq1, ref1, etc.
*/
bool spectrum_io_1d::set_spectrum_from_data(const std::vector<float> &data, const double begin_, const double step_, const double stop_)
{
    /**
     * Set spect and ndata
    */
    spect=data;
    ndata=spect.size();

    spe_image.clear(); //spe_image.size()==0 is used to indicate that there is no imaginary part
    
    /**
     * Set begin1, step1, stop1 for ppm information
    */
    begin1=begin_;
    step1=step_;
    stop1=stop_;

    return true;
}

/**
 * @brief write 1D spectrum to a text file
*/
bool spectrum_io_1d::write_spectrum_txt(std::string outfname)
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
    outfile << "# SIZE = " << ndata << " ( = number of points)" << std::endl;
    outfile << "#" << std::endl;

    if(spe_image.size()==0)
    {
        for (int j = 0; j < ndata; j++)
        {
            outfile << spect[j] << std::endl;
        }
    }
    else
    {
        for (int j = 0; j < ndata; j++)
        {
            /**
             * Reasons unknown, but the sign of the imaginary part is reversed in Bruker's txt file
            */
            float temp_data=-spe_image[j];
            if(temp_data>=0.0)
            { 
                outfile << spect[j] << "+" << temp_data << "i" << std::endl;
            }
            else
            {
                outfile << spect[j] << temp_data << "i" << std::endl;
            }
        }
    }
    outfile.close();
    return true;
}

/**
 * @brief estimate noise level using a general purpose method: segment by segment variance
*/
bool spectrum_io_1d::est_noise_level()
{

    std::vector<double> variances;
    std::vector<double> sums;
    int n_segment = ndata / 32;
    for (int i = 0; i < n_segment; i++)
    {
        double sum = 0;
        for (int j = 0; j < 32; j++)
        {
            sum += spect[i * 32 + j];
        }
        sum /= 32;
        // get variance of each segment
        double var = 0;
        for (int j = 0; j < 32; j++)
        {
            var += (spect[i * 32 + j] - sum) * (spect[i * 32 + j] - sum);
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
bool spectrum_io_1d::est_noise_level_mad()
{

    int ndim = spect.size();

    if (noise_level < 1e-20) // estimate noise level
    {
        std::vector<float> t = spect;
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
bool spectrum_io_1d::write_json(std::string fname)
{
    std::ofstream outfile(fname.c_str());
    if (!outfile.is_open())
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    Json::Value root;
    root["ndata"] = ndata/2;
    root["ndata_power_of_2"]=ndata/2; //We suppose ZF=2 in processing
    /**
     * ref1 is the end of the spectrum
     * carrier_frequency is the middle of the spectrum
     * SW is the total width of the spectrum
     * All in Hz.
     * If read in from a file other than ft1, all will be set to 0
    */
    root["carrier_frequency"]=ref1+SW1/2; 
    root["observed_frequency"]=frq1;
    root["spectral_width"]=SW1; //in Hz

    outfile << root << std::endl;
    outfile.close();

    return true;
}

/**
 * To save memory after peaks picking or fitting. User by 3D picker/fitter classes
*/
bool spectrum_io_1d::release_spectrum()
{
    spect.clear();
    return true;
}

/**
 * @brief get spectrum as a read-only vector
*/
const std::vector<float> & spectrum_io_1d::get_spectrum() const
{
    return spect;
}