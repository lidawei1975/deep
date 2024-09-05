#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <set>

#include "json/json.h"
#include "commandline.h"
#include "contour.h"
#include "spectrum_prediction.h"
#include "DeepConfig.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::CostFunction;
using ceres::NumericDiffCostFunction;
using ceres::DynamicNumericDiffCostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;


/**
 * Define a functor for fitting a exp1 model
 */
struct exp_cost_functor
{
    exp_cost_functor(std::vector<double> peak_heights_, std::vector<double> gradients_);
    bool operator()(double const* const* parameters, double* residuals) const;

private:
    std::vector<double> peak_heights, gradients;
};

exp_cost_functor::exp_cost_functor(std::vector<double> peak_heights_, std::vector<double> gradients_)
{
    peak_heights = peak_heights_;
    gradients = gradients_;  
};


bool exp_cost_functor::operator()(double const* const* parameters, double* residuals) const
{
    for (int i = 0; i < peak_heights.size(); i++)
    {
        residuals[i] = peak_heights[i] - parameters[0][0] * exp(-parameters[0][1] * gradients[i] * gradients[i]);
    }

    return true;
};

/**
 * Define a exp functor with baseline shift
*/
struct exp_cost_functor_baseline
{
    exp_cost_functor_baseline(std::vector<double> peak_heights_, std::vector<double> gradients_);
    bool operator()(double const* const* parameters, double* residuals) const;

private:
    std::vector<double> peak_heights, gradients;
};

exp_cost_functor_baseline::exp_cost_functor_baseline(std::vector<double> peak_heights_, std::vector<double> gradients_)
{
    peak_heights = peak_heights_;
    gradients = gradients_;  
};

bool exp_cost_functor_baseline::operator()(double const* const* parameters, double* residuals) const
{
    for (int i = 0; i < peak_heights.size(); i++)
    {
        residuals[i] = peak_heights[i] - parameters[0][0] * exp(-parameters[0][1] * gradients[i] * gradients[i]) - parameters[0][2];
    }

    return true;
};

/**
 * Class for fitting
*/
class CDosyFit
{
private:
    /**
     * @brief Fitted peaks from 1D DOSY
     * fitted_peak_heights[peak_index] = [1st trace height, 2nd trace height, ...], normalized to 1st trace height
     */
    std::vector<std::vector<double>> fitted_peak_heights;

    /**
     * Z filed strength gradient of all traces. Must be the same size as fitted_peak_heights[peak_index]
     */
    std::vector<double> z_gradients;

    /**
     * Corresponding trace indices for each z gradient
     * If all are used, trace_indices = [0, 1, 2, ..z_gradients.size()-1]
    */
    std::vector<int> trace_indices;

    /**
     * Diffusion constant D* for each peak
     */
    std::vector<double> diffusion_constants;

    /**
     * Baseline shift for each peak
     */
    std::vector<double> baseline_shifts;

    /**
     * @brief Fitting parameters[0] is 0 gradient peak height
    */
    std::vector<double> zero_gradient_peak_heights;

    /**
     * Include a constant shift in fitting
     */
    bool include_baseline;

public:
    CDosyFit(bool);
    CDosyFit();
    ~CDosyFit();

    bool read_fitted_peaks(const std::string &filename);
    bool read_z_gradients(const std::string &filename);
    bool fit_diffusion_constant();
    bool write_result(const std::string &filename,double rescale_factor=1.0);
};

CDosyFit::CDosyFit(bool b_include_baseline)
{
    include_baseline = b_include_baseline;
}

CDosyFit::CDosyFit()
{
    include_baseline = false;
}

CDosyFit::~CDosyFit()
{
}

/**
 * @brief Read z gradients from file
 */
bool CDosyFit::read_z_gradients(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Cannot open file " << filename << std::endl;
        return false;
    }

    double z_gradient;
    std::string line, token;
    std::vector<std::string> tokens;
    bool b_first_line = true;
    int n_fields = 1;
    while (getline(file, line))
    {
        tokens.clear();
        std::istringstream iss(line);
        while(iss>>token)
        {
            tokens.push_back(token);
        }
        if(b_first_line==true)
        {
            b_first_line = false;
            /**
             * If one column, it is z gradients
            */
            if(tokens.size() == 1)
            {
                n_fields = 1;
                z_gradients.push_back(std::stod(tokens[0]));
            }
            else if(tokens.size() == 2)
            {   
                /**
                 * If two columns, it is trace index and z gradients
                */
                n_fields = 2;
                trace_indices.push_back(std::stoi(tokens[0]));
                z_gradients.push_back(std::stod(tokens[1]));
            }
            /**
             * If more than two columns, it is an error., Stop
            */
            else
            {
                std::cerr << "Z gradients file must have 1 or 2 columns." << std::endl;
                file.close();
                return false;
            }
        }
        else //not first line
        {
            if(tokens.size() == 1 && n_fields == 1)
            {
                z_gradients.push_back(std::stod(tokens[0]));
            }
            else if(tokens.size() == 2 && n_fields == 2)
            {
                trace_indices.push_back(std::stoi(tokens[0]));
                z_gradients.push_back(std::stod(tokens[1]));
            }
            else
            {
                std::cerr << "Z gradients file must have same columns in all rows" << std::endl;
                file.close();
                return false;
            }
        }
    }
    file.close();

    /**
     * If n_fields ==1, fill trace_indices with 0, 1, 2, ...
    */
    if(n_fields == 1)
    {
        trace_indices.clear();
        for(int i = 0; i < z_gradients.size(); i++)
        {
            trace_indices.push_back(i);
        }
    }

    return true;
}

/**
 * @brief read fitted peaks from file
 */
bool CDosyFit::read_fitted_peaks(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        std::cerr << "Cannot open file " << filename << std::endl;
        return false;
    }

    if (z_gradients.size() == 0)
    {
        std::cerr << "Z gradients must be read before fitted peaks." << std::endl;
        return false;
    }

    std::string line;
    getline(file, line);

    /**
     * First line is the header.
     * Example:
     * VARS INDEX X_AXIS X_PPM XW HEIGHT DHEIGHT ASS INTEGRAL VOL SIMGAX GAMMAX CONFIDENCE NROUND BACKGROUND
     * FORMAT %5d %9.4f %8.4f %7.3f %+e %+e %s %+e %+e %f %f %f %4d %1D
     * 0 17993.8069   9.0417  33.808 +3.068540e+05 +1.407235e+00 peaks2 +7.307113e+06 +1.104369e+07 14.356642 0.001638 0.979000    2 0  1.0000  1.0020  0.9634  0.8929  0.8084  0.6435  0.5451  0.4022  0.5131  0.3532  0.0824  0.1752  0.1231  0.0614  0.0793  0.0000  0.0528
     * 1 18023.6752   9.0390  36.754 +3.607473e+05 +1.474697e+00 peaks1 +1.101843e+07 +1.411457e+07 15.608198 0.000984 0.988000    2 0  1.0000  0.9712  0.8735  0.9138  0.7746  0.6461  0.6620  0.5510  0.3185  0.3581  0.3315  0.1206  0.1772  0.1257  0.0000  0.0000  0.0686
     */

    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> tokens;

    while (iss >> token)
    {
        tokens.push_back(token);
    }

    int n_other_fields = tokens.size() - 1; // Line 1 start with VARS, which doesn't correspond to any data

    /**
     * Skip second line, which is FORMAT
     */
    getline(file, line);

    /**
     * Read from the third line, til the end of file
     */
    while (getline(file, line))
    {
        std::istringstream iss(line);
        std::string token;

        /**
         * Skip the first n_other_fields fields
         */
        for (int i = 0; i < n_other_fields; i++)
        {
            iss >> token;
        }

        /**
         * Now read the peak heights
         */
        std::vector<double> peak_heights;
        double height;
        bool b_nan = false;
        while (iss >> token)
        {
            /**
             * If token is nan or -nan, set height to 0
            */
            if (token == "nan" || token == "-nan")
            {
                height = 0.0;
                b_nan = true;
            }
            else
            {
                height = std::stod(token);
            }
            peak_heights.push_back(height);
        }

        /**
         * If b_nan is true, Set peak_heights to [0,0,0,0,...]
         * So that the fitting will just fail.
        */
        if (b_nan == true)
        {
            for(int i = 0; i < peak_heights.size(); i++)
            {
                peak_heights[i] = 0.0;
            }
        }

        /**
         *  peak_heights.size() must >= last trace_indices
        */
        if (peak_heights.size() <= trace_indices[trace_indices.size() - 1])
        {
            std::cerr << "Not enough traces, peak_heights.size() = " << peak_heights.size() << ", last trace index = " << trace_indices[trace_indices.size() - 1] << std::endl;
            std::cerr << std::endl;
            return false;
        }

        fitted_peak_heights.push_back(peak_heights);
    }

    file.close();

    /**
     * fitted_peak_heights must have same number of elements for all peaks
    */
    for (int i = 1; i < fitted_peak_heights.size(); i++)
    {
        if (fitted_peak_heights[i].size() != fitted_peak_heights[0].size())
        {
            std::cerr << "All peaks must have the same number of traces." << std::endl;
            return false;
        }
    }

    /**
     * Largest trace index in trace_indices must be less than the number of traces
    */
    int max_trace_index = *std::max_element(trace_indices.begin(), trace_indices.end());
    if (max_trace_index >= fitted_peak_heights[0].size())
    {
        std::cerr << "Trace index must be less than the number of traces." << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief Fit diffusion constant D* for each peak
 */
bool CDosyFit::fit_diffusion_constant()
{
    if (fitted_peak_heights.size() == 0)
    {
        std::cerr << "No fitted peaks read." << std::endl;
        return false;
    }

    if (z_gradients.size() == 0)
    {
        std::cerr << "No z gradients read." << std::endl;
        return false;
    }

    
    for (int i = 0; i < fitted_peak_heights.size(); i++)
    {
        /**
         * Make subsets of peak_heights and z_gradients_copy, using trace_indices
        */
        std::vector<double> peak_heights;
        std::vector<double> z_gradients_copy;

        for (int j = 0; j < trace_indices.size(); j++)
        {
            peak_heights.push_back(fitted_peak_heights[i][trace_indices[j]]);
            z_gradients_copy.push_back(z_gradients[j]);
        }

        /**
         * Search from the end of the peak heights, find the first non-zero peak height
        */
        int last_non_zero_index = peak_heights.size() - 1;
        for (int j = peak_heights.size() - 1; j >= 0; j--)
        {
            if (peak_heights[j] > 0.0001)
            {
                last_non_zero_index = j;
                break;
            }
        }

        /**
         * If last_non_zero_index <=1 (unless we have only two traces),skip this peak and set D* to -1.0 as a flag (normal D is > 0)
        */
        if (last_non_zero_index <= 1 && trace_indices.size()<2)
        {
            zero_gradient_peak_heights.push_back(peak_heights[0]);
            diffusion_constants.push_back(-1.0);
            baseline_shifts.push_back(0.0);
            continue;
        }

        /**
         * If peak_heights[0] == 0.0, set D* to -1.0 as a flag (normal D is > 0)
        */
        if (peak_heights[0] < std::numeric_limits<double>::epsilon())
        {
            zero_gradient_peak_heights.push_back(0.0);
            diffusion_constants.push_back(-1.0);
            baseline_shifts.push_back(0.0);
            continue;
        }

        /**
         * Remove zero peak heights at the end, remove the corresponding z z_gradients_copy too
        */
        peak_heights.resize(last_non_zero_index + 1);
        z_gradients_copy.resize(last_non_zero_index + 1);


        /**
         * A special case: if peak_heights.size()==z_gradients_copy.size()==2,
         * we can directly calculate D = -log(peak_heights[1]/peak_heights[0])/(z_gradients_copy[1]^2 - z_gradients_copy[0]^2)
         * No need to run non-linear optimization
         */
        if (peak_heights.size() == 2)
        {
            double D = -log(peak_heights[1] / peak_heights[0]) / (z_gradients_copy[1] * z_gradients_copy[1] - z_gradients_copy[0] * z_gradients_copy[0]);
            /**
             * Update 0 gradient peak height, diffusion constant, and baseline shift
            */
            zero_gradient_peak_heights.push_back(peak_heights[0]*exp(D*z_gradients_copy[0]*z_gradients_copy[0]));
            diffusion_constants.push_back(D);
            baseline_shifts.push_back(0.0);
            continue;
        }


        /**
         * Fit a model as
         * peak_heights = A * exp(-D * z_gradients^2)
         */

        /**
         * Define the optimization problem.
         * Two fitting parameters: A, D, and (optional) baseline shift
         * We still need A because first data point is not at z=0
         */
        ceres::Problem problem;
        double fitting_parameters[3] = {1.0 , 0.0025, 0.0};

        if(include_baseline == true)
        {
            ceres::DynamicNumericDiffCostFunction<exp_cost_functor_baseline> *cost_function 
                = new ceres::DynamicNumericDiffCostFunction<exp_cost_functor_baseline>
                (
                    new exp_cost_functor_baseline(peak_heights, z_gradients_copy)
                );

            cost_function->AddParameterBlock(3);
            cost_function->SetNumResiduals(z_gradients_copy.size());
            
            /**
             * @param fitting_parameters: [A, D, baseline_shift]
             * A is similar to first peak height
             * D is the diffusion constant, usually around 0.0025
             */
            problem.AddResidualBlock(cost_function, nullptr, fitting_parameters);

        }
        else
        {
            ceres::DynamicNumericDiffCostFunction<exp_cost_functor> *cost_function 
                = new ceres::DynamicNumericDiffCostFunction<exp_cost_functor>
                (
                    new exp_cost_functor(peak_heights, z_gradients_copy)
                );

            cost_function->AddParameterBlock(2);
            cost_function->SetNumResiduals(z_gradients_copy.size());
            problem.AddResidualBlock(cost_function, nullptr, fitting_parameters); //fitting_parameters[2] will be ignored
        }

        /**
         * Set lower bound of 1.0 for peak height
        */
        problem.SetParameterLowerBound(fitting_parameters, 0, 1.0);
        /**
         * Set lower bound of 0.0 for diffusion constant
        */
        problem.SetParameterLowerBound(fitting_parameters, 1, 0.0);

        ceres::Solver::Options options;
        options.max_num_iterations = 250;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        /**
         * Update diffusion constant, 
         */
        zero_gradient_peak_heights.push_back(fitting_parameters[0]);
        diffusion_constants.push_back(fitting_parameters[1]);
        /**
         * If include_baseline is true, update baseline shift, otherwise it is 0
        */
        baseline_shifts.push_back(fitting_parameters[2]);

    }

    return true;
}

/**
 * @brief Write fitted D* to file.
 * If file name ends with .json, write in json format, otherwise write in text format
*/
bool CDosyFit::write_result(const std::string &filename,double rescale_factor)
{
    std::ofstream file(filename);
    if (!file)
    {
        std::cerr << "Cannot open file " << filename << std::endl;
        return false;
    }

    /**
     * Check filename extension
     */
    std::string extension = filename.substr(filename.find_last_of(".") + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == "json")
    {
        Json::Value root;
        for (int i = 0; i < diffusion_constants.size(); i++)
        {
            root["zero_gradient_peak_height"][i] = zero_gradient_peak_heights[i];
            root["diffusion_constant"][i] = diffusion_constants[i]/rescale_factor;
            root["baseline_shift"][i] = baseline_shifts[i];
        }
        file << root;
    }
    /**
     * Write in text format for other extensions
    */
    else
    {
        for (int i = 0; i < diffusion_constants.size(); i++)
        {
            file << zero_gradient_peak_heights[i] << " " << diffusion_constants[i]/rescale_factor << " " << baseline_shifts[i] << std::endl;
        }
    }

    file.close();

    return true;
}

int main(int argc, char **argv)
{
    std::cout << "DEEP Picker package Version " << deep_picker_VERSION_MAJOR << "." << deep_picker_VERSION_MINOR << std::endl;
    std::cout << "This program will fit a diffussion constant D* from 1D DOSY (pseudo-2D) fitted peaks by VF_1D " << std::endl;

    CCommandline cmdline;
    std::vector<std::string> args, args2, args3;

    args.push_back("-h");
    args2.push_back("no");
    args3.push_back("print help message then quit (no)");

    args.push_back("-v");
    args2.push_back("0");
    args3.push_back("verbose level (0: minimal, 1:normal)");

    args.push_back("-f");
    args2.push_back("arguments_dosy_fit.txt");
    args3.push_back("read arguments from file");

    args.push_back("-in");
    args2.push_back("fitted.tab");
    args3.push_back("input fitted peak list with pseudo-2D DOSY peaks");

    args.push_back("-z");
    args2.push_back("z_gradients.txt");
    args3.push_back("input z gradients for each trace");

    args.push_back("-out");
    args2.push_back("result.txt");
    args3.push_back("output fitted D* result in text format");

    args.push_back("-baseline");
    args2.push_back("no");
    args3.push_back("include a constant shift in fitting (no)");

    /**
     * Dosy paramters: Diffusion time, length of the gradient and delay time for gradient recovery
    */
    args.push_back("-t_diffusion");
    args2.push_back("80");
    args3.push_back("DOSY diffusion time in ms");

    args.push_back("-g_length");
    args2.push_back("1.6");
    args3.push_back("DOSY gradient length in ms");

    args.push_back("-delay");
    args2.push_back("0.2");
    args3.push_back("DOSY gradient delay for recovery in ms");

    /**
     * Alternatively, use can provide a custome rescale factor for D* fitting
    */
    args.push_back("-rescale");
    args2.push_back("1454078.85082151");
    args3.push_back("rescale factor for Diffusion constant fitting. Negative value means calculating from DOSY parameters");

    cmdline.init(args, args2, args3);
    cmdline.pharse(argc, argv);
    cmdline.print();

    if (cmdline.query("-h") == "yes")
    {
        exit(0);
    }

    bool b_include_baseline = cmdline.query("-baseline")[0] == 'y' || cmdline.query("-baseline")[0] == 'Y';

    CDosyFit dosy_fit(b_include_baseline);

    if (dosy_fit.read_z_gradients(cmdline.query("-z")) 
        && dosy_fit.read_fitted_peaks(cmdline.query("-in")) 
        && dosy_fit.fit_diffusion_constant() )
    {
        /**
         * Scale D* if rescale factor is provided
        */
        double rescale_factor = std::stod(cmdline.query("-rescale"));

        /**
         * If rescale factor is negative, calculate from DOSY parameters
        */
        if(rescale_factor <= 0)
        {
            double t_diffusion = std::stod(cmdline.query("-t_diffusion"));
            double g_length = std::stod(cmdline.query("-g_length"));
            double delay = std::stod(cmdline.query("-delay"));
            /**
             * Error if any of the DOSY parameters <=0 or t_diffusion <= g_length/3 + delay/2
            */
            if(t_diffusion <= 0 || g_length <= 0 || delay <= 0)
            {
                std::cerr << "DOSY parameters must be positive, set rescale_factor to 1.0 " << std::endl;
                rescale_factor = 1.0;
            }
            else if( t_diffusion <= g_length/3 + delay/2)
            {
                std::cerr << "DOSY diffusion time must be larger than g_length/3 + delay/2, set rescale_factor to 1.0 " << std::endl;
                rescale_factor = 1.0;
            }
            else
            {
                rescale_factor = (2 * M_PI * 4257.7 * g_length) * (2 * M_PI * 4257.7 * g_length) * (t_diffusion - g_length/3 - delay/2) * 1e-5;
                std::cout<<" Rescale factor is calculated from DOSY parameters: "<<rescale_factor<<std::endl;
            }
        }


        /**
         * Get output filenames. Multiple output files are separated by space
        */
        std::string output_files = cmdline.query("-out");

        std::istringstream iss(output_files);
        std::string output_file;
        while (iss >> output_file)
        {
            dosy_fit.write_result(output_file,rescale_factor);
        }
        
        std::cout << "Fitting D* from 1D DOSY peaks finished." << std::endl;
    }
    else
    {
        std::cerr << "Fitting D* from 1D DOSY peaks failed." << std::endl;
    }

    return 0;
}
