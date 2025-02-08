#include <fstream>
#include "json/json.h"

#include "peak_manipulation.h"
#include "dosy.h"


/**
 * Cost function for fitting a dosy model
*/
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
 * Cost function for fitting a dosy model with baseline shift
*/
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
 * Main class for DOSY fitting
*/
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

    pm.read_file(filename);

    /**
     * z_indices are the column indexes for all relative z values, 
     * They are called Z_A0, Z_A1, Z_A2, ... in nmrPipe .tab file
    */
    std::vector<int> z_indices = pm.get_column_indexes_by_prefix("Z_");

    if(z_gradients.size() == 0)
    {
        std::cerr << "Z gradients must be read before fitted peaks." << std::endl;
        return false;
    }

    /**
     * Read peak heights into fitted_peak_heights[peak_index] = [1st trace height, 2nd trace height, ...], normalized to 1st trace height
     * As saved in nmrPipe .tab file Z_A0, Z_A1, Z_A2, ...
    */
    fitted_peak_heights.clear();
    fitted_peak_heights.resize(pm.get_n_peaks(), std::vector<double>(z_indices.size(), 0.0));

    for(int i=0;i<z_indices.size();i++)
    {
        std::vector<std::string> peak_heights;
        pm.get_column(z_indices[i], peak_heights);
        for(int j=0;j<peak_heights.size();j++)
        {
            fitted_peak_heights[j][i] = std::stod(peak_heights[j]);
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

    /**
     * Also get X_PPM, Y_PPM, HEIGHT and VOL to define multiple peaks
    */
    std::vector<int> indices = pm.get_column_indexes_by_prefix("X_PPM");
    for(int i=0;i<indices.size();i++)
    {
        std::vector<std::string> x_ppm_string;
        pm.get_column(indices[i], x_ppm_string);
        for(int j=0;j<x_ppm_string.size();j++)
        {
            x_ppm.push_back(std::stod(x_ppm_string[j]));
        }
    }

    indices = pm.get_column_indexes_by_prefix("Y_PPM");
    for(int i=0;i<indices.size();i++)
    {
        std::vector<std::string> y_ppm_string;
        pm.get_column(indices[i], y_ppm_string);
        for(int j=0;j<y_ppm_string.size();j++)
        {
            y_ppm.push_back(std::stod(y_ppm_string[j]));
        }
    }

    indices = pm.get_column_indexes_by_prefix("HEIGHT");
    for(int i=0;i<indices.size();i++)
    {
        std::vector<std::string> height_string;
        pm.get_column(indices[i], height_string);
        for(int j=0;j<height_string.size();j++)
        {
            height.push_back(std::stod(height_string[j]));
        }
    }

    indices = pm.get_column_indexes_by_prefix("VOL");
    for(int i=0;i<indices.size();i++)
    {
        std::vector<std::string> vol_string;
        pm.get_column(indices[i], vol_string);
        for(int j=0;j<vol_string.size();j++)
        {
            volume.push_back(std::stod(vol_string[j]));
        }
    }

    /**
     * Assess they have same size: x_ppm, y_ppm, height and fitted_peak_heights
    */
    if(x_ppm.size() != height.size() || x_ppm.size() != fitted_peak_heights.size() || x_ppm.size() != volume.size())
    {
        std::cerr << "X_PPM, Y_PPM, HEIGHT, and fitted peaks must have the same size." << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief Fit diffusion constant D* for each peak
 */
bool CDosyFit::fit_diffusion_constant(bool b_combine)
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

    zero_gradient_peak_heights.clear();
    diffusion_constants.clear();
    baseline_shifts.clear();

    zero_gradient_peak_heights.resize(fitted_peak_heights.size(), 0.0);
    diffusion_constants.resize(fitted_peak_heights.size(), -1.0); // -1.0 is a flag for D* not calculated
    baseline_shifts.resize(fitted_peak_heights.size(), 0.0);

    /**
     * Get mutiplets using breadth first search
     * Two peaks are neighbors if they are within 0.02 ppm in both x and y dimensions and height ratio is less than 5.0
    */
    if(b_combine == true)
    {
        int n_peaks = fitted_peak_heights.size();
        std::vector<int> neighbor(n_peaks * n_peaks, 0);

        for(int i=0;i<n_peaks;i++)
        {
            for(int j=i+1;j<n_peaks;j++)
            {
                double peak_ratio = height[i]/height[j];
                if(peak_ratio>1.0)
                {
                    peak_ratio = 1.0/peak_ratio;
                }
                if(fabs(x_ppm[i]-x_ppm[j])<0.03 && fabs(y_ppm[i]-y_ppm[j])<0.03 && peak_ratio > 0.2)
                {
                    neighbor[i*n_peaks+j] = 1;
                    neighbor[j*n_peaks+i] = 1;
                }
            }
        }
        peak_multiplets = peak_tools::breadth_first(neighbor, n_peaks);
    }
    /**
     * If not combining peaks, each peak is a singlet
    */
    else
    {
        for(int i=0;i<fitted_peak_heights.size();i++)
        {
            std::deque<int> peak_multiplet;
            peak_multiplet.push_back(i);
            peak_multiplets.push_back(peak_multiplet);
        }
    }

    
    for(int ii=0;ii<peak_multiplets.size(); ii++)
    {
        /**
         * Make subsets of peak_heights and z_gradients_copy, using trace_indices
        */
        std::vector<double> peak_heights;
        std::vector<double> z_gradients_copy;

        for (int j = 0; j < trace_indices.size(); j++)
        {
            double true_peak_height = 0.0;
            for(int m=0;m<peak_multiplets[ii].size();m++)
            {
                 true_peak_height += fitted_peak_heights[peak_multiplets[ii][m]][trace_indices[j]]*volume[peak_multiplets[ii][m]];
            }
            peak_heights.push_back(true_peak_height);
            z_gradients_copy.push_back(z_gradients[j]);
        }

        /**
         * peak_heights[0] == 0 means failed peak in fitting. Do not fit D* for this peak
        */
        if (peak_heights[0] < std::numeric_limits<double>::epsilon())
        {
            for(int m=0;m<peak_multiplets[ii].size();m++)
            {
                zero_gradient_peak_heights[peak_multiplets[ii][m]] = peak_heights[0];
                diffusion_constants[peak_multiplets[ii][m]] = -1.0;
                baseline_shifts[peak_multiplets[ii][m]] = 0.0;
            }
            continue;
        }

        /**
         * Rescale peak_heights so that 0th element is 1.0
        */
        double scale_factor = peak_heights[0];
        for (int j = 0; j < peak_heights.size(); j++)
        {
            peak_heights[j] /= scale_factor;
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
         * If peak_heights[0] == 0.0, set D* to -1.0 as a flag (normal D is > 0)
         * 
        */
        if ( (last_non_zero_index <= 1 && trace_indices.size()<2) || peak_heights[0] < std::numeric_limits<double>::epsilon())
        {
            for(int m=0;m<peak_multiplets[ii].size();m++)
            {
                zero_gradient_peak_heights[peak_multiplets[ii][m]] = peak_heights[0];
                diffusion_constants[peak_multiplets[ii][m]] = -1.0;
                baseline_shifts[peak_multiplets[ii][m]] = 0.0;
            }
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
            for(int m=0;m<peak_multiplets[ii].size();m++)
            {
                zero_gradient_peak_heights[peak_multiplets[ii][m]] = peak_heights[0]*exp(D*z_gradients_copy[0]*z_gradients_copy[0]);
                diffusion_constants[peak_multiplets[ii][m]] = D;
                baseline_shifts[peak_multiplets[ii][m]] = 0.0;
            }
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
        for(int m=0;m<peak_multiplets[ii].size();m++)
        {
            zero_gradient_peak_heights[peak_multiplets[ii][m]] = fitting_parameters[0];
            diffusion_constants[peak_multiplets[ii][m]] = fitting_parameters[1];
            baseline_shifts[peak_multiplets[ii][m]] = fitting_parameters[2];
        }
        std::cout<<"Done fitting D* for peak multiplet "<<ii<<std::endl;
    }

    return true;
}

bool CDosyFit::rescale(double rescale_factor)
{
    if (rescale_factor <= 0)
    {
        std::cerr << "Rescale factor must be positive." << std::endl;
        return false;
    }

    for (int i = 0; i < diffusion_constants.size(); i++)
    {
        diffusion_constants[i] /= rescale_factor;
    }

    return true;
}
/**
 * @brief Write fitted D* to file.
 * If file name ends with .json, write in json format, otherwise write in text format
*/
bool CDosyFit::write_result(const std::string &filename)
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
            root["diffusion_constant"][i] = diffusion_constants[i];
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
            file << zero_gradient_peak_heights[i] << " " << diffusion_constants[i] << " " << baseline_shifts[i] << std::endl;
        }
    }

    file.close();

    return true;
};

/**
 * @brief Insert D to the peak file and write it
*/
bool CDosyFit::write_peak_file(const std::string &filename)
{
    if (diffusion_constants.size() == 0)
    {
        std::cerr << "No diffusion constants calculated." << std::endl;
        return false;
    }

    std::vector<std::string> col_values;

    for (int i = 0; i < diffusion_constants.size(); i++)
    {
        char buffer[100];
        snprintf(buffer, 100, "%6.4e", diffusion_constants[i]);
        col_values.push_back(buffer);
    }

    /**
     * Insert a new column "DIFFUSION" with diffusion constants, just before "Z_A0"
    */
    int index = pm.get_column_index("Z_A0");
    pm.operate_on_column(index,column_operation::INSERT, "DIFFUSION", "%6.4e", col_values);

    pm.write_file(filename);

    return true;
};

