#include <vector>
#include <valarray>
#include <fstream>
#include <iostream>

#include "json/json.h"
#include "commandline.h"
#include "dnn_picker.h"

#include "spectrum_io_1d.h"

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

bool spectrum_io_1d::direct_set_spectrum(std::vector<float> &spe_)
{
    spect = spe_;

    ndata = spect.size();
    stop1 = ndata;
    begin1 = 1.0;
    step1 = 1.0; // arbitary

    est_noise_level();
    return true;
}

bool spectrum_io_1d::save_experimental_spectrum(std::string outfname)
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
    }
    root["spectrum"] = data;
    outfile << root;
    return true;
}

bool spectrum_io_1d::read_spectrum(std::string infname)
{
    bool b_read = 0;

    input_spectrum_fname = infname; // save for later use

    std::string sldw(".ldw");
    std::string stxt(".txt");
    std::string sft1(".ft1");
    std::string sjson(".json");

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
    else
    {
        b_read = false;
    }

    std::cout << "Spectrum size is " << ndata << std::endl;
    std::cout << "From " << begin1 << " to " << stop1 << " and step is " << step1 << std::endl;

    est_noise_level();

    // std::cout<<"Set negative data points to zero."<<std::endl;
    // for(int i=0;i<spe.size();i++)
    // {
    //     spe[i]=std::max(spe[i],0.0f);
    // }

    return b_read;
}

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
        std::cout << "Read nmrPipe 1D imaginary part successully." << std::endl;
    }
    else
    {
        std::cout << "Read nmrPipe 1D  imaginary part failed." << std::endl;
        spe_image.clear();
    }

    fclose(fp);

    b_header = true;

    return true;
};

bool spectrum_io_1d::write_spectrum(std::string infname)
{
    if (b_header == false)
    {
        std::cout << "Warning: PIpe format header has not been read." << std::endl;
        return false;
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

bool spectrum_io_1d::stride_spectrum(int n)
{
    if (n < 1)
    {
        std::cout << "Error: stride number must be larger than 1." << std::endl;
        return false;
    }

    int nstride =  ndata / n;

    std::vector<float> temp;
    temp.resize(nstride);
    for (int i = 0; i < nstride; i++)
    {
        temp[i] = spect[i * n];
    }
    spect = temp;
    ndata = spect.size();
    step1 = step1 * n;
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
        */
        if (b_left == true && b_right == true && b_size == true)
        {
            float data=std::stof(line);
            spect.push_back(data);
        }
    }

    if(spect.size()!=ndata)
    {
        std::cout<<"Error: spectrum_io_1d::read_spectrum_txt, ndata is not equal to the number of data points. Set ndata=spect.size()"<<std::endl;
        ndata = spect.size();
    }

    step1 = (stop1 - begin1) / ndata;
    return true;
}

bool spectrum_io_1d::read_spectrum_json(std::string infname)
{
    Json::Value root;
    std::ifstream fin(infname);
    if (!fin)
        return false;

    fin >> root;
    read_spectrum_json_format1(root);
    return true;
};

bool spectrum_io_1d::read_spectrum_json_format1(Json::Value &root)
{
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
            std::cout<<"Error: spectrum_io_1d::read_spectrum_json_format1, data2[i] is not double or string."<<std::endl;
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
            std::cout<<"Error: spectrum_io_1d::read_spectrum_json_format1, data1[i] is not double or string."<<std::endl;
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


bool spectrum_io_1d::est_noise_level_general()
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
    std::cout << "Noise level is estiamted to be " << noise_level << ", using a geneal purpose method." << std::endl;
    return true;
}

bool spectrum_io_1d::est_noise_level()
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

bool spectrum_io_1d::release_spectrum()
{
    spect.clear();
    return true;
}

const std::vector<float> & spectrum_io_1d::get_spectrum() const
{
    return spect;
}