
// #include <omp.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <limits>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <array>
#include <vector>
#include <set>

#include "json/json.h"
#include "commandline.h"
#include "hungary.h"

#include "kiss_fft.h"
#include "spectrum_pick_1d.h"
#include "spectrum_fit_1d.h"

#include "db_match_1d.h"

namespace ldw_math_db_match_1d
{
    // linear interpolation function, input is vector<double> x and vector<double> y
    // return y_interp from x_interp
    double linear_interpolation(std::vector<double> x, std::vector<double> y, double x_interp)
    {
        int n = x.size();
        double y_interp = 0.0;
        if (x_interp < x[0])
        {
            y_interp = y[0];
        }
        else if (x_interp > x[n - 1])
        {
            y_interp = y[n - 1];
        }
        else
        {
            int i = 0;
            while (x_interp > x[i])
            {
                i++;
            }
            if (fabs(x_interp - x[i - 1]) < 0.0001)
            {
                y_interp = y[i - 1];
            }
            else
            {
                double d = (x[i] - x[i - 1]) / (x_interp - x[i - 1]);
                if (fabs(d - 1) < 0.00001)
                {
                    y_interp = y[i];
                }
                else
                {
                    y_interp = y[i - 1] + (y[i] - y[i - 1]) / d;
                }
            }
        }
        return y_interp;
    };

    void sortArr(const std::vector<double> &arr, std::vector<int> &ndx, bool b_Descending = true)
    {
        std::vector<std::pair<double, int>> vp;

        for (int i = 0; i < arr.size(); ++i)
        {
            vp.push_back(std::make_pair(arr[i], i));
        }

        std::sort(vp.begin(), vp.end()); // ascending order

        if (b_Descending == true)
        {
            std::reverse(vp.begin(), vp.end()); // descending order
        }

        for (int i = 0; i < vp.size(); i++)
        {
            ndx.push_back(vp[i].second);
        }
    };

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
};

// class pattern_recognition_1d
// constructor
pattern_match_1d::pattern_match_1d()
{
    cutoff = 0.02;
    cutoff_pattern = 0.003;

    best_assignment_index = -1;
    log_detection_limit = 9.9;

    b_fake = false;
};

// destructor
pattern_match_1d::~pattern_match_1d(){

};

// return some information about the class
int pattern_match_1d::get_number_of_peaks()
{
    return database_peak_ppm_original.size();
};

int pattern_match_1d::get_number_normal()
{
    return number_normal;
};

double pattern_match_1d::get_mean_confident_level()
{
    return mean_confident_level;
};

/**
 * @brief initialize the class variables
 *
 * @param i  peak group index
 * @param x_
 * @param y_
 * @param log_dl_ log of detection limit
 * @param pattern_height_  db peak height
 * @param height_  exp peak height. Pointer to a vector
 * @param pattern_ db peak position in ppm
 * @param v_  exp peak position in ppm. Pointer to a vector
 * @param spe_ spectrum. Pointer to a vector
 * @param stop_ ppm of first point of spectrum
 * @param step_ ppm step between two neighboring points of spectrum
 * @param begin_ ppm of last+1 point of spectrum
 * @return true not used
 */

bool pattern_match_1d::init(int i, double x_, double y_,
                            std::vector<double> pattern_height_,
                            std::vector<double> pattern_,
                            double lb_, double ub_,
                            double max_value_of_amplitude2_)
{
    peak_group_index = i;
    cutoff = x_;
    cutoff_pattern = y_;
    database_peak_height = pattern_height_;
    database_peak_ppm = pattern_;

    lb = lb_;
    ub = ub_;
    max_value_of_amplitude2 = max_value_of_amplitude2_;

    database_peak_ppm_original = database_peak_ppm;
    database_peak_height_original = database_peak_height;

    // init mapping_2_original as 0,1,2,....,database_peak_ppm.size()-1
    mapping_2_original.resize(database_peak_ppm.size());
    for (int i = 0; i < database_peak_ppm.size(); i++)
    {
        mapping_2_original[i] = i;
    }

    if (database_peak_ppm.size() > n_max_peak_number)
    {

        // reduce # of peaks in database if they are too close
        database_peak_removed.resize(database_peak_ppm.size(), 0); // 0: not removed, 1: removed

        std::vector<int> height_sortted_index;
        ldw_math_db_match_1d::sortArr(database_peak_height, height_sortted_index); // descending order. height is not changed

        int number_removed = 0;
        double cutoff_scale = 0.0;

        while (database_peak_ppm.size() - number_removed > n_max_peak_number)
        {
            cutoff_scale = cutoff_scale + 1.0;
            bool b_reduce = true;
            while (b_reduce)
            {
                b_reduce = false;
                for (int kk = 0; kk < height_sortted_index.size(); kk++)
                {
                    int k = height_sortted_index[kk];
                    if (database_peak_removed[k] == 1)
                    {
                        continue;
                    }

                    // label peaks that are too close to the this peak as removed
                    for (int i = 0; i < database_peak_height.size(); i++)
                    {
                        if (i != k)
                        {
                            if (fabs(database_peak_ppm[i] - database_peak_ppm[k]) < cutoff_pattern * cutoff_scale && database_peak_removed[i] == 0)
                            {
                                database_peak_removed[i] = 1;
                                number_removed++;
                                b_reduce = true;
                            }
                        }
                    }
                }
            }
        }

        /**
         * We will not remove peaks. but for all removed peaks, we will only match them as covered peaks in the next step
         */

        if (n_verbose_match_1d > 0)
        {
            std::cout << "For group " << peak_group_index << ", # of peaks in database: " << database_peak_ppm_original.size() << " ==> " << database_peak_ppm.size() << std::endl;
        }
    }
    else
    {
        database_peak_removed.resize(database_peak_ppm.size(), 0); // 0: not removed, 1: removed
        if (n_verbose_match_1d > 0)
        {
            std::cout << "For group " << peak_group_index << ", # of peaks in database: " << database_peak_ppm_original.size() << std::endl;
            for(int k=0;k<database_peak_ppm.size();k++){
                std::cout << "  " << k << " " << database_peak_ppm[k] << " " << database_peak_height[k] << std::endl;
            }
        }
    }

    // sort database_peak_height, keep the original index.
    ldw_math_db_match_1d::sortArr(database_peak_height, sort_ndx);

    // sort_ndx_inversed is the inverse of sort_ndx
    sort_ndx_inversed.resize(sort_ndx.size());
    for (int i = 0; i < sort_ndx.size(); i++)
    {
        sort_ndx_inversed[sort_ndx[i]] = i;
    }

    // Apply the same sorting to database_peak_ppm and database_peak_height
    std::vector<double> database_peak_ppm_sorted;
    std::vector<double> database_peak_height_sorted;
    std::vector<int> database_peak_removed_sorted;
    for (int i = 0; i < database_peak_ppm.size(); i++)
    {
        database_peak_ppm_sorted.push_back(database_peak_ppm[sort_ndx[i]]);
        database_peak_height_sorted.push_back(database_peak_height[sort_ndx[i]]);
        database_peak_removed_sorted.push_back(database_peak_removed[sort_ndx[i]]);
    }
    database_peak_ppm = database_peak_ppm_sorted;
    database_peak_height = database_peak_height_sorted;
    database_peak_removed = database_peak_removed_sorted;

    return true;
};

/** recursive function to find the matched peaks
 *  @param i the index of database peak
 *  @param accum the vector of matched exp peak index up to current database peak
 *  When i==database_peak_ppm.size(), the function will find all the possible matched exp peak index, stopping more recursion
 * It is not possible that the function will stop recursion when the accum.size() is still less than database_peak_ppm.size()
 * because BDL is always added to the condidate list
 * Current matching algorithm doesn't consider peak width (sigma and gamma) but only use peak height and location (ppm)
 */

void pattern_match_1d::combinations(const int current_peak_ndx, std::vector<int> accum, long unsigned int bdl_position)
{

    // get current max and min of heights for type normal peaks
    // heights are all normalized with respect to the database (expected) peak height. In other words, all matched exp peaks should have same heights (ideally)
    double min_height_type_normal = 1.0e100;
    double max_height_type_normal = 0.0;
    for (int k = 0; k < std::min(accum.size(), bdl_position); k++) // k is DB (part of) peak index. accum is current assignment
    {
        if (matching_candidates_type[k][accum[k]] == MatchType_normal)
        {
            double temp_height = matching_candidates_height_ratio[k][accum[k]];
            if (temp_height < min_height_type_normal)
            {
                min_height_type_normal = temp_height;
            }
            if (temp_height > max_height_type_normal)
            {
                max_height_type_normal = temp_height;
            }
        }
    }

    for (int j = 0; j < matching_candidates_ppm_diff[current_peak_ndx].size(); ++j)
    {
        // test ppm compatibility between tmp and ppm_diff[i][j]
        // matching_candidates_type[current_peak_ndx][j] is the type of the current exp peak, to be matched to the ith database peak.
        // Matching of [0,current_peak_ndx-1] database peaks are saved
        // matching_candidates_height_ratio[current_peak_ndx][j] is the height of the current exp peak, normalized with respect to the database (expected) peak height

        bool b = true;
        if (matching_candidates_type[current_peak_ndx][j] == MatchType_normal)
        {
            // Step 1, check heights consistency among all normal peaks
            // New matched normal peak can't be too high. It must be less than 2 times of the min height of normal peaks
            if (matching_candidates_height_ratio[current_peak_ndx][j] > min_height_type_normal * 4.0)
            {
                b = false;
                continue; // go to next j
            }
            // new matched normal peak can't be too low. It must be more than 0.5 times of the max height of normal peaks
            if (matching_candidates_height_ratio[current_peak_ndx][j] < max_height_type_normal / 4.0)
            {
                b = false;
                continue; // go to next j
            }

            for (int k = 0; k < accum.size(); ++k)
            {
                // skip all  covered peak from already matched peak list
                if (matching_candidates_type[k][accum[k]] == MatchType_covered)
                {
                    continue; // go to next k
                }

                // check ppm difference
                if (fabs(matching_candidates_ppm_diff[k][accum[k]] - matching_candidates_ppm_diff[current_peak_ndx][j]) > cutoff_pattern)
                {
                    b = false;
                    break; // go to next j
                }

                // order of peak must be preserved in matching
                if ((matching_candidates_ppm[current_peak_ndx][j] < matching_candidates_ppm[k][accum[k]]) && (database_peak_ppm[k] < database_peak_ppm[current_peak_ndx]))
                {
                    b = false;
                    break;
                }
                if ((matching_candidates_ppm[current_peak_ndx][j] > matching_candidates_ppm[k][accum[k]]) && (database_peak_ppm[k] > database_peak_ppm[current_peak_ndx]))
                {
                    b = false;
                    break;
                }

                // two database peak cannot match same experimental peak
                if (matching_candidates[k][accum[k]] == matching_candidates[current_peak_ndx][j])
                {
                    b = false;
                    break;
                }
            }
        }

        if (b == false)
        {
            continue; // skip this j
        }

        // j will be added to accum
        if (matching_candidates_type[current_peak_ndx][j] == MatchType_normal)
        {
            // get its height ratio
            double current_height_ratio = matching_candidates_height_ratio[current_peak_ndx][j];
            // get expected height of last matched peak, except for BDL, suppose this peak and last peak have same height ratio
            double expected_height = current_height_ratio * database_peak_height[bdl_position - 1];
            // if expected height is too low (below BDL) skip this j. Because this particular possible match will be covered when bdl_position is smaller
            if (log(expected_height) < log_detection_limit)
            {
                continue; // skip this j
            }
        }

        std::vector<int> new_accm(accum); // copy accum to new_accm
        new_accm.push_back(j);            // add current assignment to new_accm
        b = true;

        std::vector<double> current_covered_height_ratio;
        std::vector<double> current_covered_ppm;

        // Get current min of heights for type normal peaks, including the current matched peak
        // Heights are all normalized with respect to the database (expected) peak height.
        // In other words, all matched exp peaks should have same heights (ideally)
        double min_height_type_normal2 = 1.0e100;
        bool b_all_covered = true;

        // find allowed peak shift range from all MatchType_normal peaks.
        // For each MatchType_normal peak, the allowed region is from matching_candidates_ppm_diff-cutoff_pattern to matching_candidates_ppm_diff+cutoff_pattern
        double allowed_shift_left = -cutoff; // allowed shift range if there is no MatchType_normal peak
        double allowed_shift_right = cutoff;

        for (int k = 0; k < new_accm.size(); k++) // k is DB (part of) peak index. new_accum is current assignment
        {
            if (matching_candidates_type[k][new_accm[k]] == MatchType_normal)
            {
                double temp_height = matching_candidates_height_ratio[k][new_accm[k]];
                if (temp_height < min_height_type_normal2)
                {
                    min_height_type_normal2 = temp_height;
                    b_all_covered = false;
                }

                double temp_left = matching_candidates_ppm_diff[k][new_accm[k]] - cutoff_pattern;
                double temp_right = matching_candidates_ppm_diff[k][new_accm[k]] + cutoff_pattern;
                if (temp_left > allowed_shift_left)
                {
                    allowed_shift_left = temp_left;
                }
                if (temp_right < allowed_shift_right)
                {
                    allowed_shift_right = temp_right;
                }
            }
        }

        // If there is no MatchType_normal peak, we set min_height_type_normal2 to 0.0, so that all MatchType_covered peaks have no restriction on height
        if (b_all_covered == true)
        {
            min_height_type_normal2 = 0.0;
        }

        // for each MatchType_covered peak, we find maximal spectral data from matching_candidates_ppm+allowed_shift_left to matching_candidates_ppm+allowed_shift_right
        // For covered peaks, matching_candidates_ppm is the database peak ppm, not the matched exp peak ppm
        for (int k = 0; k < new_accm.size(); ++k)
        {
            if (matching_candidates_type[k][new_accm[k]] == MatchType_covered)
            {
                int pos;
                double max_height = get_max_spectral_data(matching_candidates_ppm[k][new_accm[k]] + allowed_shift_left, matching_candidates_ppm[k][new_accm[k]] + allowed_shift_right, pos);

                // if max_height is below BDL, skip this assignment because this particular possible match will be covered when bdl_position is smaller
                if (log(max_height) < log_detection_limit)
                {
                    b = false;
                    break;
                }

                // normalize the height with respect to the database peak height
                // matching_candidates_height[k][new_accm[k]] is the database peak height for this covered peak
                max_height /= matching_candidates_height[k][new_accm[k]];

                if (max_height * 1.20 < min_height_type_normal2)
                {
                    b = false; // this assignment is not allowed because the db height of this covered peak is too high to be covered by the spectral data.
                    break;
                }
                current_covered_height_ratio.push_back(max_height); // save the height ratio for later use
                // convert pos to ppm using eq: (ppm - begin) / step = pos
                current_covered_ppm.push_back(pos * step + begin);
            }
            else
            {
                current_covered_height_ratio.push_back(0.0); // placeholder for MatchType_normal peak
                current_covered_ppm.push_back(0.0);          // placeholder for MatchType_normal peak
            }
        }

        if (b == false)
        {
            continue; // skip this j
        }

        if (current_peak_ndx + 1 >= bdl_position)
        {
            // we have matched all peaks before bdl_position, we will fill all remaining peaks with bdl

            // Last check. If all matched peak beong to MatchType_covered, we need to apply strick peak shift restriction because we don't have any MatchType_normal peak to help us
            if (b_all_covered == true)
            {
                current_covered_height_ratio.clear();
                current_covered_ppm.clear();

                double window_step = cutoff_pattern / 4.0;

                double max_of_min = -std::numeric_limits<double>::max();

                for (double fake_ppm_diff = -cutoff; fake_ppm_diff <= cutoff; fake_ppm_diff += window_step)
                {
                    std::vector<double> temp_current_covered_height_ratio;
                    std::vector<double> temp_current_covered_ppm;
                    bool b_bdl_possible = false;
                    double min_of_max_height_of_all_covered_peaks = 1.0e100;
                    // loop through all covered peaks
                    for (int k = 0; k < new_accm.size(); ++k)
                    {
                        int pos;
                        double max_height = get_max_spectral_data(matching_candidates_ppm[k][new_accm[k]] + fake_ppm_diff - cutoff_pattern, matching_candidates_ppm[k][new_accm[k]] + fake_ppm_diff + cutoff_pattern, pos);

                        // this peak should be treated as MatchType_bdl, so we don't need to include this situtation here.
                        if (log(max_height) < log_detection_limit)
                        {
                            b_bdl_possible = true;
                            break;
                        }
                        // normalize the height with respect to the database peak height
                        // matching_candidates_height[k][new_accm[k]] is the database peak height for this covered peak
                        max_height /= matching_candidates_height[k][new_accm[k]];

                        /** In the case of all covered peaks, we prefer max of spectral data
                         * and we also want good match (ppm diff) for all covered peaks
                         * so we reduce max_height by the ppm diff (when ppm diff is = cutoff, reduce by exp(2)*detec_limit)
                         */

                        double max_height2 = max_height - exp(2.0 * fabs(fake_ppm_diff / cutoff)) * exp(log_detection_limit);

                        if (max_height2 < min_of_max_height_of_all_covered_peaks)
                        {
                            min_of_max_height_of_all_covered_peaks = max_height2;
                        }
                        temp_current_covered_height_ratio.push_back(max_height);
                        temp_current_covered_ppm.push_back(pos * step + begin);
                    }
                    if (b_bdl_possible == true)
                    {
                        continue;
                    }
                    // save the best case (at this fake ppm diff) to current_covered_height_ratio
                    if (min_of_max_height_of_all_covered_peaks > max_of_min)
                    {
                        max_of_min = min_of_max_height_of_all_covered_peaks;
                        current_covered_height_ratio = temp_current_covered_height_ratio;
                        current_covered_ppm = temp_current_covered_ppm;
                    }
                }
            }

            // only add to assignment_wrt_candidate if the current_covered_height_ratio is not empty or b_all_covered is false
            if (b_all_covered == false || current_covered_height_ratio.size() > 0)
            {
                // add BDL to accum from bdl_position to the end. The size of accum is the same as the size of matching_candidates
                for (int k = bdl_position; k < matching_candidates.size(); k++)
                {
                    new_accm.push_back(-1);
                    current_covered_height_ratio.push_back(0.0); // placeholder for MatchType_bdl peak
                    current_covered_ppm.push_back(0.0);          // placeholder for MatchType_bdl peak
                }
                // add accum to assignment_wrt_candidate then return. We are done.
                assignment_wrt_candidate.push_back(new_accm);
                assignment_covered_height_ratio.push_back(current_covered_height_ratio);
                assignment_covered_ppm.push_back(current_covered_ppm);
            }
        }
        else // not all peaks are matched, we need to continue the recursion
        {
            combinations(current_peak_ndx + 1, new_accm, bdl_position); // tmp will be the "accum" for the next round of recursion
        }
    } // end of for (int j = 0; j < matching_candidates[current_peak_ndx].size(); ++j)
    return;
};

double pattern_match_1d::get_max_spectral_data(double left, double right, int &max_position)
{

    // convert ppm to data index, using stop,step and begin. By tradition, step is negative because stop<begin. step = (stop - begin) / (ndata). ndata is the number of data points
    int left_ndx = (int)round((left - begin) / step);
    int right_ndx = (int)round((right - begin) / step);

    if (right_ndx < 0)
    {
        right_ndx = 0;
    }
    if (left_ndx > spectrum.size())
    {
        left_ndx = spectrum.size();
    }

    double max_height = 0.0;
    for (int i = right_ndx; i < left_ndx; i++)
    {
        if (spectrum.operator[](i) > max_height)
        {
            max_height = spectrum.operator[](i);
            max_position = i;
        }
    }
    return max_height;
}

int pattern_match_1d::get_number_of_assignment()
{
    return assignment.size();
}

/**
 * @brief create one assignment with all peaks matched to water
 * @return true always the case
 */
bool pattern_match_1d::run_fake()
{
    std::vector<int> one_assignment(database_peak_ppm.size(), -1);
    std::vector<MatchType> one_assignment_type(database_peak_ppm.size(), MatchType_water);

    assignment.clear();
    assignment_type.clear();

    assignment.push_back(one_assignment);
    assignment_type.push_back(one_assignment_type);

    b_fake = true;

    /**
     * @brief possible_assigned_peak is used to store the index of peaks that are assigned to this pms
     * For fake_run, all are un-assigned: 0
     */
    possible_assigned_peak.clear();
    possible_assigned_peak.resize(ppm.size(), 0);

    return true;
}

/**
 * @brief create all possible assignments
 * Change assignment and assignment_type for downstream use
 * @return true always the case
 */
bool pattern_match_1d::run()
{
    b_fake = false; // default of b_fake is false but we want to make sure

    int n_database_peak = database_peak_ppm.size(); // number of peaks in database. Defined to have a readable code

    // std::vector<std::vector<int>> assignment; [all possible assignments][assignment for each db peak]
    // std::vector<std::vector<MatchType>> assignment_type;
    assignment.clear();
    assignment_type.clear();

    // A slide window to find all peaks in experimental data that is within cutoff_pattern of each database peak
    // the window can move both ways as fas as cutoff

    matching_candidates.clear();
    matching_candidates_type.clear();
    matching_candidates_ppm.clear();
    matching_candidates_ppm_diff.clear();
    matching_candidates_height_ratio.clear();
    matching_candidates_height.clear();

    for (int j = 0; j < n_database_peak; j++)
    {
        std::vector<double> temp_peak_ppm, temp_peak_ppm_diff, temp_peak_height_ratio, temp_peak_height;
        std::vector<int> temp_exp_peak_index;
        std::vector<MatchType> temp_match_type;
        int ndx = -1;

        /**
         * Apply lb and ub about peak height of matched peak. lb and ub are relative to max_value_of_exp_spectrum
         */
        double ub_absoluate = max_value_of_exp_spectrum * ub;
        double lb_absoluate = max_value_of_exp_spectrum * lb;

        /**
         * lb and ub are also relative to max_value_of_amplitude2.
         * lb and ub are good for the maximal database peak only
         * If a database peak has a height of 0.65 of the maximal database peak, we need to reduce lb and ub by 0.65
         */
        ub_absoluate *= database_peak_height[j] / max_value_of_amplitude2;
        lb_absoluate *= database_peak_height[j] / max_value_of_amplitude2;

        /**
         * Because of intensities are not 100% correct and we allow match to a very high experiment peak, we need to relax the upper bound by 4  time.
         */
        ub_absoluate *= 4.0;

        /**
         * if database_peak_removed[j] == 1, this peak will not be matched agains experimental peaks, but will be added to assignment as covered peak
         */

        if (database_peak_removed[j] == 0)
        {
            double max_height = 0.0;

            for (int i = 0; i < ppm.size(); i++)
            {
                /**
                 * Skip if backgroud_flag of this peak == 1
                 */
                if (backgroud_flag[i] == 1)
                {
                    continue;
                }

                if (std::abs(ppm.operator[](i) - database_peak_ppm[j]) <= cutoff && intensity[i] > lb_absoluate // peak height is above lb_absoluate
                    && intensity[i] < ub_absoluate)                                                             // peak height is below ub_absoluate
                {
                    temp_peak_ppm_diff.push_back(ppm.operator[](i) - database_peak_ppm[j]);
                    temp_peak_ppm.push_back(ppm.operator[](i));
                    temp_exp_peak_index.push_back(i);
                    temp_peak_height_ratio.push_back(intensity[i] / database_peak_height[j]);
                    temp_peak_height.push_back(intensity[i]);
                    temp_match_type.push_back(MatchType_normal);

                    if (intensity[i] > max_height)
                    {
                        max_height = intensity[i];
                        ndx = i;
                    }
                }
            }
        }

        // add 1 covered peak matching if we are not in an emtpy region
        int ndx2 = -1;
        double max_spectral_height = get_max_spectral_data(database_peak_ppm[j] - cutoff, database_peak_ppm[j] + cutoff, ndx);

        /**
         * For covered peaks, max_spectral_height must be above detection limit and above lower bound.
         * There is no need to check upper bound because covered peaks can be as lower as needed
         */

        if (log(max_spectral_height) > log_detection_limit && max_spectral_height > lb_absoluate)
        {
            // these 3 lines generate a place holder for a covered peak so that all these temp_* vectors have the same size
            temp_peak_ppm_diff.push_back(0.0);
            temp_exp_peak_index.push_back(-1);
            temp_peak_height_ratio.push_back(database_peak_height[j]);

            // we need these 3 lines to assess posisble covered peaks
            temp_peak_ppm.push_back(database_peak_ppm[j]);
            temp_peak_height.push_back(database_peak_height[j]);
            temp_match_type.push_back(MatchType_covered);
        }

        matching_candidates.push_back(temp_exp_peak_index);
        matching_candidates_type.push_back(temp_match_type);
        matching_candidates_ppm.push_back(temp_peak_ppm);
        matching_candidates_ppm_diff.push_back(temp_peak_ppm_diff);
        matching_candidates_height_ratio.push_back(temp_peak_height_ratio);
        matching_candidates_height.push_back(temp_peak_height);
    }

    // Because peaks are sorted by database height, we define a BDL position in all database peaks.
    // 0: BDL position is the first peak, hence all peaks are BDL
    // 1: BDL position is the second peak, hence the first peak is not BDL but the rest are BDL
    // 2,3,4 etc.
    // n-1: BDL position is the last peak, hence the first n-1 peaks are not BDL but the last peak is BDL
    // n: BDL position is after the last peak, hence all peaks are not BDL

    assignment_wrt_candidate.clear();        // will be updated in function combinations
    assignment_covered_height_ratio.clear(); // will be updated in function combinations
    assignment_covered_ppm.clear();          // will be updated in function combinations

    /**
     * Get lb of each peak, from high to low. Find the first one that is below detection limit
     * This is the BDL position
     * If all peaks are above detection limit, BDL position is n_database_peak
     * means we have no BDL peak
     */
    int start_bdl_position = database_peak_height.size();
    for (int j = 0; j < database_peak_height.size(); j++)
    {
        double lb_of_peak = lb * max_value_of_exp_spectrum * database_peak_height[j] / max_value_of_amplitude2;
        if (log(lb_of_peak) < log_detection_limit)
        {
            start_bdl_position = j;
            break;
        }
    }

    // for bdl_position = 0, recursion will not be called (because it won't work for bdl_position = 0). We will the two varibles manually
    if (start_bdl_position == 0)
    {
        assignment_wrt_candidate.push_back(std::vector<int>(n_database_peak, -1));
        assignment_covered_height_ratio.push_back(std::vector<double>(n_database_peak, 0.0));
        assignment_covered_ppm.push_back(std::vector<double>(n_database_peak, 0.0));
        start_bdl_position = 1;
    }

    /**
     * At this time, peaks are sorted by database height in an descending order
     * bdl_position==0 means all peaks are BDL. no need to check, we just add it
     * bdl_position==1 means 1st peak is non-BDL peak, the rest are BDL.
     * bdl_position==n_database_peak means no BDL.
     */
    for (long unsigned int bdl_position = start_bdl_position; bdl_position <= n_database_peak; bdl_position++)
    {
        std::vector<int> accum;
        accum.clear();                        // accum is the accumulated number of possible matches for each peak. clear it before each iteration.
        combinations(0, accum, bdl_position); // assignment_wrt_candidate is class member variable. Function combinations will update it.
    }

    // std::vector<std::vector<int>> assignment_wrt_candidate: [all possible assignments][assignment for each db peak, WRT matching_candidates]
    for (int i = 0; i < assignment_wrt_candidate.size(); i++)
    {
        std::vector<int> tmp, tmp2;
        std::vector<MatchType> tmp_type;
        for (int j = 0; j < assignment_wrt_candidate[i].size(); j++)
        {
            int n = assignment_wrt_candidate[i][j];
            if (n >= 0)
            {
                tmp.push_back(matching_candidates[j][n]);
                tmp_type.push_back(matching_candidates_type[j][n]);
            }
            else //-1 means no matched peak (BDL)
            {
                tmp.push_back(-1);
                tmp_type.push_back(MatchType_bdl);
            }
        }
        assignment.push_back(tmp);
        assignment_type.push_back(tmp_type);
    }

    /**
     * Set possible_assigned_peak from any assignment that has at least one normal peak
     */
    possible_assigned_peak.clear();
    possible_assigned_peak.resize(ppm.size(), 0);
    for (int i = 0; i < assignment.size(); i++)
    {
        for (int j = 0; j < assignment[i].size(); j++)
        {
            if (assignment_type[i][j] == MatchType_normal)
            {
                possible_assigned_peak[assignment[i][j]] = 1;
            }
        }
    }

    return true;
};

/**
 * @brief pattern_match_1d::get_assignment, return read only assignment
 */
const std::vector<int> &pattern_match_1d::get_assignment(int index) const
{
    return assignment[index];
}

/**
 * Get a readonly reference to possible_assigned_peak
 */
const std::vector<int> &pattern_match_1d::get_possible_assigned_peak() const
{
    return possible_assigned_peak;
}

/**
 * @brief pattern_match_1d::calculate_cost
 * This function will only get cost1 from chemical shift matching and cost2 from intensity pattern matching within assignment group
 * Total cost will also include the penalty for BDL peaks, covered peaks, and mismatch of peak intensity across all assignment groups.
 * See function calculate_cost_at_intensity
 * @param fdebug file handle for debug output
 */
void pattern_match_1d::calculate_cost(std::ofstream &fdebug)
{
    /**
     * for each assignment, calculate cost1 based on chemical shift difference
     * for each assignment, calculate cost2 based on intensity variance
     * WE start from 1 because assignment[0] is all bdl peaks. We known the cost and min_intensity for all bdl peaks
     */
    assignment_cost.push_back({0.0, 0.0});                   // addtional cost will be added to bdl peaks.
    assignment_min_intensity.push_back(log_detection_limit); // min intensity of bdl peaks is detection limit

    allowed_intensity_upper_bound = 0.0;

    for (int j = 1; j < assignment.size(); j++)
    {
        std::vector<int> assignment_tmp_j = assignment[j];
        double cost1 = 0.0;
        double cost2 = 0.0;
        double shift1 = 0.0;

        // shift pattern_ppm globally to best match ppm when assignment_type[j][k]==MatchType_normal
        int c_normal = 0;
        for (int k = 0; k < assignment_tmp_j.size(); k++)
        {
            if (assignment_type[j][k] == MatchType_normal)
            {
                shift1 += database_peak_ppm[k] - ppm.operator[](assignment_tmp_j[k]);
                ++c_normal;
            }
        }
        if (c_normal > 0)
        {
            shift1 /= c_normal;
        }

        for (int k = 0; k < assignment_tmp_j.size(); k++)
        {
            if (assignment_type[j][k] == MatchType_normal)
            {
                cost1 += std::fabs(database_peak_ppm[k] - ppm.operator[](assignment_tmp_j[k]) - shift1);
                cost1 += clashing_additional_cost[assignment_tmp_j[k]]; // add clashing penalty
            }
        }
        if (c_normal > 0)
        {
            /**
             * Get the mean absolute shift of all normal peaks
             */
            cost1 /= c_normal;

            /**
             * rescale cost1 by predefined const cost_peak_pattern_locations
             */
            cost1 *= cost_peak_pattern_locations;

            /**
             * Add overall shift penalty.
             * Rescaled by 0.02/cutoff. That is, if cutoff is larger, the penalty is smaller. 0.02 is the default cutoff
             */
            cost1 += std::fabs(shift1) * cost_overall_shift / cutoff * 0.02;
        }

        // calculate cost2 based on intensity variance when assignment_type[j][k]==MatchType_normal
        //  get array of log of normalized experimental intensity
        std::vector<double> inten_normal_array;
        for (int k = 0; k < assignment_tmp_j.size(); k++)
        {
            if (assignment_type[j][k] == MatchType_normal)
            {
                inten_normal_array.push_back(std::log(intensity.at(assignment_tmp_j[k]) / database_peak_height[k]));
            }
            else if (assignment_type[j][k] == MatchType_covered)
            {
                inten_normal_array.push_back(std::log(assignment_covered_height_ratio[j][k]));
            }
        }

        double min_intensity = 1e100;
        double mean_intensity = log_detection_limit;
        cost2 = 0.0;

        if (inten_normal_array.size() > 0)
        {
            // calculate min and mean intensity
            for (int k = 0; k < inten_normal_array.size(); k++)
            {
                if (inten_normal_array[k] < min_intensity)
                {
                    min_intensity = inten_normal_array[k];
                }
                mean_intensity += inten_normal_array[k];
            }
            mean_intensity /= inten_normal_array.size();

            // calculate variance of normalized intensity
            double var_intensity = 0.0;
            for (int k = 0; k < inten_normal_array.size(); k++)
            {
                var_intensity += (inten_normal_array[k] - mean_intensity) * (inten_normal_array[k] - mean_intensity);
            }
            var_intensity /= inten_normal_array.size();
            cost2 = std::sqrt(var_intensity);
            cost2 *= cost_peak_pattern_height; // scale factor, less weights on intensity variance
        }
        else /** these is no normal or covered peak*/
        {
            cost2 = 0.0;
            min_intensity = log_detection_limit;
        }

        assignment_cost.push_back({cost1, cost2});
        assignment_min_intensity.push_back(min_intensity); // each assignment has a min_intensity

        if (min_intensity > allowed_intensity_upper_bound)
        {
            allowed_intensity_upper_bound = min_intensity;
        }

        // std::cout<< "assignment " << j << " cost1=" << cost1 << " cost2=" << cost2 << " min_intensity=" << min_intensity;
        // std::cout << "out of total " << assignment.size() << " assignments" << std::endl;
    }

    allowed_intensity_upper_bound += 1.0; // add 1.0 to allowed_intensity_upper_bound to make sure we have some room for possible errors

// some output for debug
#ifdef DEBUG
    for (int j = 0; j < assignment.size(); j++)
    {
        fdebug << "assignment " << j << " cost1=" << assignment_cost[j][0] << " cost2=" << assignment_cost[j][1] << " min_intensity=" << assignment_min_intensity[j] << std::endl;
        for (int ii = 0; ii < assignment[j].size(); ii++)
        {
            int i = sort_ndx_inversed[ii];
            fdebug << assignment[j][i];

            if (assignment_type[j][i] == MatchType_normal)
            {
                fdebug << " " << ppm.at(assignment[j][i]);
                fdebug << " " << std::scientific << intensity.at(assignment[j][i]);
            }
            else if (assignment_type[j][i] == MatchType_covered)
            {
                fdebug << " " << assignment_covered_ppm[j][i];
                fdebug << " " << std::scientific << assignment_covered_height_ratio[j][i] * database_peak_height[i];
            }
            else // bdl or skipped
            {
                fdebug << " bdl bdl";
            }
            fdebug << " " << database_peak_ppm[i];
            fdebug << " " << database_peak_height[i];
            fdebug << " " << assignment_type[j][i];
            fdebug << std::endl;
        }
        fdebug << std::endl;
    }
#endif
};

/**
 * @brief For each possible assignment, get the corresponding min_intensity.
 * It is not the intensity, but log of intensity
 * Expand it to 5 numbers, +-1, +-2.
 * @param check_intensities a vector of min_intensity for each possible assignment,
 */
bool pattern_match_1d::get_key_intensity(std::vector<double> &check_intensities, std::vector<double> &allowed_intensities_upper_bound)
{
    // log of experimental intensity - log of database intensity for each possible assignment.
    if (b_fake == false)
    {
        for (int i = 0; i < assignment_min_intensity.size(); i++)
        {
            std::array<double, 5> grid = {-2.0, -1.0, 0.0, 1.0, 2.0};
            for (int k = 0; k < 5; k++)
            {
                check_intensities.push_back(assignment_min_intensity[i] + grid[k]);
            }
        }
        allowed_intensities_upper_bound.push_back(allowed_intensity_upper_bound);
    }
    return true;
};

/**
 * @brief This function will calculate the cost for each assignment group based on the compound concentration (match between cencentration and experimetnal peak intensity
 * then cost1 and cost2 from chemical shift and intensity pattern matching (see function calculate_cost) will be added to the cost.
 * @param intensity_check_ expected compound intensity
 * @param fdebug file handle for debug output
 * @param pms_index index of pattern_match_1d object, used for debug output only
 * @return double total cost at this intensity
 */

double pattern_match_1d::calculate_cost_at_intensity(double intensity_check_, std::ofstream &fdebug, int pms_index)
{
    // for fake run. always return 0.0
    if (b_fake == true)
    {
        return 0.0;
    }

    // intensity_check is based on log of experimental intensity - log of database intensity
    intensity_check = intensity_check_;

    // to speed up the calculation, we first calculate concentration-intensity matching cost for all peaks in matching_candidates
    // then for each assignment, we just add up cost from all peaks assigned in this assignment

    std::vector<std::vector<double>> cost_of_matching_candidate_normal;

    // for BLD, cost_bdl only depends on databse peak height, so we can calculate it once for each database peak
    std::vector<double> cost_of_matching_candidate_bdl;

    std::vector<double> x = {-100.0, -1.0, -0.5, 0.0, 0.50, 1.00, 101.00};
    std::vector<double> y_normal = {100000.0, 1000.02, 0.02, 0.0, 0.02, 0.52, 200.52};

    std::vector<double> x_type2 = {-100.0, -1.0, -0.5, 0.0, 101.00};
    std::vector<double> y_covered = {100000.0, 1000.02, 0.02, 0.0, 0.00};
    std::vector<double> y_bdl = {100000.0, 1000.02, 0.02, 0.0, 0.00};

    // calculate cost for each peak in matching_candidates
    for (int i = 0; i < matching_candidates.size(); i++) // i is index of database peak
    {
        std::vector<double> cost_of_matching_candidate_normal_tmp, cost_of_matching_candidate_bdl_tmp, cost_of_matching_candidate_covered_tmp;
        for (int j = 0; j < matching_candidates[i].size(); j++)
        {
            if (matching_candidates[i][j] < 0) // not a normal peak. we will calculate cost in other ways.
            {
                cost_of_matching_candidate_normal_tmp.push_back(0.0); // place holder. we don't need this value
                continue;
            }
            double cost_normal;
            double log_exp = log(intensity.at(matching_candidates[i][j])) - log(database_peak_height[i]);
            cost_normal = ldw_math_db_match_1d::linear_interpolation(x, y_normal, log_exp - intensity_check);
            cost_of_matching_candidate_normal_tmp.push_back(cost_normal);
        }
        cost_of_matching_candidate_normal.push_back(cost_of_matching_candidate_normal_tmp);

        // for BLD, cost_bdl only depends on databse peak height, so we can calculate it once
        double intensity_true = intensity_check + std::log(database_peak_height[i]);
        cost_of_matching_candidate_bdl.push_back(ldw_math_db_match_1d::linear_interpolation(x_type2, y_bdl, log_detection_limit - intensity_true));
    }

    double min_cost = 1e100;
    for (int i = 0; i < assignment.size(); i++) // i is assigment index
    {
        double t_cost = 0.0;

        for (int j = 0; j < assignment[i].size(); j++) // j is peak index
        {
            if (assignment_type[i][j] == MatchType_normal)
            {
                /**
                 * assignment_cost is the cost from chemical shift matching and intensity pattern matching. We rescale them by predefined constants peak_match_scale
                 */
                t_cost += cost_of_matching_candidate_normal[j][assignment_wrt_candidate[i][j]] + (assignment_cost[i][0] + assignment_cost[i][1]) * peak_match_scale;
            }
            else if (assignment_type[i][j] == MatchType_covered)
            {
                // calcualte cost of covered peak, which is assignment dependent
                double temp_cost = assignment_covered_height_ratio[i][j] * 1.2;                        // ratio of covered peak height to database peak height. Scaled by 1.2 to accormodate some error
                temp_cost = log(temp_cost) - intensity_check;                                          // log of ratio (intensity_check is log of checking intensity)
                temp_cost = ldw_math_db_match_1d::linear_interpolation(x_type2, y_covered, temp_cost); // cost of covered peak
                t_cost += temp_cost + additional_cost_covered * peak_match_scale;                      // cost of covered. Add addtional_cost_for_covered to make sure it is not the best match
            }
            else if (assignment_type[i][j] == MatchType_bdl)
            {
                t_cost += cost_of_matching_candidate_bdl[j] + additional_cost_bdl * peak_match_scale; // cost of bdl. Add addtional_cost_for_bdl to make sure it is not the best match
            }
        }
        t_cost /= assignment[i].size();
#ifdef DEBUG
        fdebug << "At " << intensity_check << ", for PMS " << pms_index << " assignment " << i << " cost=" << t_cost << std::endl;
#endif
        cost_at_intensity.push_back(t_cost);
        // find the minimum cost
        if (t_cost < min_cost)
        {
            min_cost = t_cost;
            best_assignment_index = i;
        }
    }
#ifdef DEBUG
    fdebug << "At " << intensity_check << " for PMS " << pms_index << " Best assignment is " << best_assignment_index << " cost=" << min_cost << std::endl
           << std::endl;
#endif

    min_cost -= intensity_check * 0.2; // we prefer to match at higher intensity. So we reduce the cost by 0.02 of intensity_check

    return min_cost;
};

/**
 * @brief This function will calculate the cost for each assignment group based on the compound concentration (match between cencentration and experimetnal peak intensity
 * then cost1 and cost2 from chemical shift and intensity pattern matching (see function calculate_cost) will be added to the cost.
 * @param fdebug file handle for debug output
 * @param intensity_check_ expected compound intensity
 * @param costs_sorted cost for each possible assignment of this group. The size of costs is the same as the number of possible assignments. Sorted in ascending order
 * @param ndx index of the best assignment in costs_sorted ndx[0] is the best assignment, ndx[1] is the second best assignment, etc.
 * @return true always the case
 */
bool pattern_match_1d::calculate_cost_at_intensity_with_details(std::ofstream &fdebug, double intensity_check_, std::vector<double> &costs_sorted, std::vector<int> &ndx)
{

    // for fake run. always return a single cost of 0.0. nothingelse is needed.
    if (b_fake == true)
    {
        costs_sorted.clear();
        costs_sorted.push_back(0.0);
    }

    // intensity_check is based on log of experimental intensity - log of database intensity
    intensity_check = intensity_check_;

    // to speed up the calculation, we first calculate concentration-intensity matching cost for all peaks in matching_candidates
    // then for each assignment, we just add up cost from all peaks assigned in this assignment

    std::vector<std::vector<double>> cost_of_matching_candidate_normal;

    // for BLD, cost_bdl only depends on databse peak height, so we can calculate it once for each database peak
    std::vector<double> cost_of_matching_candidate_bdl;

    std::vector<double> x = {-100.0, -1.0, -0.5, 0.0, 0.50, 1.00, 101.00};
    std::vector<double> y_normal = {100000.0, 1000.02, 0.02, 0.0, 0.02, 0.52, 200.52};

    std::vector<double> x_type2 = {-100.0, -1.0, -0.5, 0.0, 101.00};
    std::vector<double> y_covered = {300000.0, 3000.02, 0.02, 0.0, 0.00};
    std::vector<double> y_bdl = {300000.0, 3000.02, 0.02, 0.0, 0.00};

    // calculate cost for each peak in matching_candidates
    for (int i = 0; i < matching_candidates.size(); i++) // i is index of database peak
    {
        std::vector<double> cost_of_matching_candidate_normal_tmp, cost_of_matching_candidate_bdl_tmp, cost_of_matching_candidate_covered_tmp;
        for (int j = 0; j < matching_candidates[i].size(); j++)
        {
            if (matching_candidates[i][j] < 0) // not a normal peak. we will calculate cost in other ways.
            {
                cost_of_matching_candidate_normal_tmp.push_back(0.0); // place holder. we don't need this value
                continue;
            }
            double cost_normal;
            double log_exp = log(intensity.at(matching_candidates[i][j])) - log(database_peak_height[i]);
            double to_check = log_exp - intensity_check;

            /**
             * if x is from -0.25 to 0.0, we tolerate it. That is, predicted peak can be 1.28 times higher than experiment peak.
             */
            if (to_check < 0.0)
            {
                to_check += std::min(0.25, -to_check);
            }
            cost_normal = ldw_math_db_match_1d::linear_interpolation(x, y_normal, to_check);
            cost_of_matching_candidate_normal_tmp.push_back(cost_normal);
        }
        cost_of_matching_candidate_normal.push_back(cost_of_matching_candidate_normal_tmp);

        // for BLD, cost_bdl only depends on databse peak height, so we can calculate it once
        double intensity_true = intensity_check + std::log(database_peak_height[i]);
        cost_of_matching_candidate_bdl.push_back(ldw_math_db_match_1d::linear_interpolation(x_type2, y_bdl, log_detection_limit - intensity_true));
    }

    std::vector<double> costs;
    for (int i = 0; i < assignment.size(); i++) // i is assigment index
    {
        double t_cost = 0.0;

        for (int j = 0; j < assignment[i].size(); j++) // j is peak index
        {
            if (assignment_type[i][j] == MatchType_normal)
            {
                t_cost += cost_of_matching_candidate_normal[j][assignment_wrt_candidate[i][j]] + (assignment_cost[i][0] + assignment_cost[i][1]) * peak_match_scale;
            }
            else if (assignment_type[i][j] == MatchType_covered)
            {
                // calcualte cost of covered peak, which is assignment dependent
                double temp_cost = assignment_covered_height_ratio[i][j] * 1.2;                        // ratio of covered peak height to database peak height. Scaled by 1.2 to accormodate some error
                temp_cost = log(temp_cost) - intensity_check;                                          // log of ratio (intensity_check is log of checking intensity)
                temp_cost = ldw_math_db_match_1d::linear_interpolation(x_type2, y_covered, temp_cost); // cost of covered peak
                t_cost += temp_cost + additional_cost_covered * peak_match_scale;                      // cost of covered. Add addtional_cost_for_covered to make sure it is not the best match
            }
            else if (assignment_type[i][j] == MatchType_bdl)
            {
                t_cost += cost_of_matching_candidate_bdl[j] + additional_cost_bdl * peak_match_scale; // cost of bdl. Add addtional_cost_for_bdl to make sure it is not the best match
            }
        }
        t_cost /= assignment[i].size();

        costs.push_back(t_cost);
    }

    ldw_math_db_match_1d::sortArr(costs, ndx, false /** b_descending*/); // sort in ascending order

    /**
     * get the sorted cost
     */
    for (int i = 0; i < ndx.size(); i++)
    {
        costs_sorted.push_back(costs[ndx[i]]);
    }

#ifdef DEBUG
    for (int i = 0; i < ndx.size(); i++)
    {
        fdebug << "At " << intensity_check << ", for assignment " << ndx[i] << " cost=" << costs[ndx[i]] << std::endl;
    }
#endif

    return true;
};

/**
 * @brief pattern_match_1d::get_best_assignment_v2
 * Fill in best_assignment_, best_assignment_type_,best_assignment_height_ratio_, best_assignment_ppm_ with the best assignment
 * according to n, which is the index of the best assignment in costs_sorted
 */
bool pattern_match_1d::get_best_assignment_v2(std::vector<int> &best_assignment_, std::vector<MatchType> &best_assignment_type_, std::vector<double> &best_assignment_height_ratio_, std::vector<double> &best_assignment_ppm_, int n)
{
    best_assignment_index = n;
    return get_best_assignment(best_assignment_, best_assignment_type_, best_assignment_height_ratio_, best_assignment_ppm_);
};

/**
 * @brief pattern_match_1d::get_best_assignment
 * Fill in best_assignment_, best_assignment_type_,best_assignment_height_ratio_, best_assignment_ppm_ with the best assignment
 * according to member varible best_assignment_index, which is saved in function calculate_cost_at_intensity
 * this function need to reindex the assignment to the original order of peaks in the database
 */

bool pattern_match_1d::get_best_assignment(std::vector<int> &best_assignment_, std::vector<MatchType> &best_assignment_type_, std::vector<double> &best_assignment_height_ratio_, std::vector<double> &best_assignment_ppm_)
{
    if (b_fake == true)
    {
        best_assignment_.insert(best_assignment_.end(), assignment[0].begin(), assignment[0].end());
        best_assignment_type_.insert(best_assignment_type_.end(), assignment_type[0].begin(), assignment_type[0].end());
        std::vector<double> temp_assignment_height_ratio(database_peak_ppm_original.size(), 0.0);
        std::vector<double> temp_assignment_ppm(database_peak_ppm_original.size(), 0.0);
        best_assignment_height_ratio_.insert(best_assignment_height_ratio_.end(), temp_assignment_height_ratio.begin(), temp_assignment_height_ratio.end());
        best_assignment_ppm_.insert(best_assignment_ppm_.end(), temp_assignment_ppm.begin(), temp_assignment_ppm.end());
        return true;
    }

    mean_confident_level = 0;

    number_normal = 0;
    const int num_peak = database_peak_ppm_original.size();
    std::vector<int> temp_assignment(num_peak, -2);
    std::vector<MatchType> temp_assignment_type(num_peak, MatchType_skipped);
    std::vector<double> temp_assignment_height_ratio(num_peak, 0.0);
    std::vector<double> temp_assignment_ppm(num_peak, 0.0);

    peak_confident_level_of_assignment.clear(); // we need this because we may call this function multiple times
    peak_confident_level_of_assignment.resize(num_peak, 0);
    for (int i = 0; i < sort_ndx_inversed.size(); i++)
    {
        temp_assignment[mapping_2_original[i]] = assignment[best_assignment_index][sort_ndx_inversed[i]];
        MatchType temp_type = assignment_type[best_assignment_index][sort_ndx_inversed[i]];
        temp_assignment_type[mapping_2_original[i]] = temp_type;

        if (temp_type == MatchType_normal)
        {
            number_normal++;
            // we actually don't use peak_confident_level_of_assignment, but we keep it for future extension of the algorithm
            peak_confident_level_of_assignment[mapping_2_original[i]] = confident_level.at(assignment[best_assignment_index][sort_ndx_inversed[i]]);
            mean_confident_level += confident_level.at(assignment[best_assignment_index][sort_ndx_inversed[i]]);
        }
        // for all other type, effectively we run total_confident_level +=0

        // temp_assignment_height_ratio is the ratio of exp peak height to database peak height for normal assignment
        // for covered assignment, it is the ratio of maximal exp height in the allowed region to database peak height
        // for bdl assignment, it is 0 (initial value)
        if (temp_type == MatchType_covered)
        {
            temp_assignment_height_ratio[mapping_2_original[i]] = assignment_covered_height_ratio[best_assignment_index][sort_ndx_inversed[i]];
            temp_assignment_ppm[mapping_2_original[i]] = assignment_covered_ppm[best_assignment_index][sort_ndx_inversed[i]];
        }
        else if (temp_type == MatchType_normal)
        {
            int temp_ndx = assignment[best_assignment_index][sort_ndx_inversed[i]];
            temp_assignment_height_ratio[mapping_2_original[i]] = intensity.at(temp_ndx) / database_peak_height[sort_ndx_inversed[i]];
            temp_assignment_ppm[mapping_2_original[i]] = ppm.at(temp_ndx);
        }
    }

    mean_confident_level /= database_peak_ppm.size(); // skipped peaks are not counted. So use database_peak_ppm.size() instead of database_peak_ppm_original.size()

    if (database_peak_ppm_original.size() == 2)
    {
        mean_confident_level *= 0.6; // scale down the confident level for doublet
    }
    else if (database_peak_ppm_original.size() == 3)
    {
        mean_confident_level *= 0.8; // scale down the confident level for triplet
    }
    else if (database_peak_ppm_original.size() == 1)
    {
        mean_confident_level = 0.0; // single peak has very low confident level
    }

    if (number_normal < database_peak_ppm.size() * 0.6)
    {
        mean_confident_level = 0.0; // reduce to 0 if the number of normal assignment is less than half of the total number of tested peaks
    }

    best_assignment_.insert(best_assignment_.end(), temp_assignment.begin(), temp_assignment.end());
    best_assignment_type_.insert(best_assignment_type_.end(), temp_assignment_type.begin(), temp_assignment_type.end());
    best_assignment_height_ratio_.insert(best_assignment_height_ratio_.end(), temp_assignment_height_ratio.begin(), temp_assignment_height_ratio.end());
    best_assignment_ppm_.insert(best_assignment_ppm_.end(), temp_assignment_ppm.begin(), temp_assignment_ppm.end());

    return true;
};

// class db_match_1d, constructor
db_match_1d::db_match_1d(double c1, double c2, double c3, double c4)
{
    cutoff = c1;
    cutoff_pattern = c2;
    water_width = c3;
    step2 = c4;

    /**
     * for simulated database spectrum, we use it to adjust ppp
     * for loaded database peak directly, we do not use it. So we set it to 1.0
     */
    nstride_database_spectrum = 1;

    reference_correction = 0.0;
};
// destructor
db_match_1d::~db_match_1d(){
    //
};

bool db_match_1d::load_pka_shift()
{
    /**
     * Setup pka_shift to 0.0
     * In case there is no pka file, we will use 0.0 as pka_shift.
     */
    pka_shift.clear();
    pka_shift.resize(proton_index_of_each_group.size(), 0.0);

    std::string fname = pka_folder + "/" + base_name + ".pka";
    std::ifstream fin(fname);
    if (!fin)
    {
        /**
         *
         * This is not an error. We just don't have pka information
         */
        if (n_verbose_match_1d > 0)
        {
            std::cout << "Warning: cannot open pka file " << fname << std::endl;
        }
        return false;
    }

    std::string line;
    while (std::getline(fin, line))
    {
        std::istringstream iss(line);
        int n;
        double pka;
        if ((iss >> n >> pka))
        {
            /**
             * proton n has an overwitten ppm cutoff because pda is near 7.4
             * Loop proton_index_of_each_group to find the proton index == n
             */
            for (int i = 0; i < proton_index_of_each_group.size(); i++)
            {
                if (proton_index_of_each_group[i] == n)
                {
                    pka_shift[i] = pka;
                    break;
                }
            }
        }
    }

    if (n_verbose_match_1d > 0)
    {
        std::cout << "Read pka_shift from file " << fname << std::endl;
    }

    return true;
};

/**
 * load in database from file
 *  We will only need ppm2,amplitude2 to be ready for pattern matching
 *  We also need to set peak_stop and npeak_group
 */
bool db_match_1d::load_db(double cutoff_group, double db_cutoff)
{
    std::string fname = folder + "/" + id;

    if (peak_reading_database(fname) == false)
    {
        std::cout << "Error: cannot read database peak file " << fname << std::endl;
        return false;
    }

    // remove peaks with height less than db_height_cutoff
    for (int i = ppm2.size() - 1; i >= 0; i--)
    {
        if (amplitude2[i] < db_cutoff)
        {
            ppm2.erase(ppm2.begin() + i);
            amplitude2.erase(amplitude2.begin() + i);
            vol2.erase(vol2.begin() + i);
            sigmax2.erase(sigmax2.begin() + i);
            gammax2.erase(gammax2.begin() + i);
            group_id.erase(group_id.begin() + i);
        }
    }

    if (ppm2.size() == 0)
    {
        std::cout << "Error: no peak in database file " << fname << std::endl;
        return false;
    }

    /**
     * Get max_value_of_amplitude2 from amplitude2
     */
    max_value_of_amplitude2 = 0.0;
    for (int i = 0; i < amplitude2.size(); i++)
    {
        /**
         * Skip water region peaks!!
         */
        if (fabs(ppm2[i] - (4.7 + reference_correction)) < water_width)
        {
            continue;
        }
        if (amplitude2[i] > max_value_of_amplitude2)
        {
            max_value_of_amplitude2 = amplitude2[i];
        }
    }

    /** Find the peak grou
     * Group number -1 means the not defined.
     * Group number 0,1,2,3 .... means true group number.
     * Because ppm is sorted, we can use the following method to find the group number.
     * Suppose we have group_id like this: -1, -1, 1, 2, -1, -1, 5, -1 ....
     * After following block, it will become  100, 101, 1, 2, 102, 102, 5, 103
     */

    double current_ppm = 10000.0; // a large number, far away from any ppm value
    int current_group_id = 99;    // a large number, far away from any preassigned group_id value: 1,2,3

    for (int i = 0; i < ppm2.size(); i++) // ppm2 has same size as group_id
    {
        // peak i+1 alreay has a group_id, we skip it
        if (group_id[i] != -1)
        {
            current_ppm = 10000.0; // reset current_ppm so that next peak will be assigned a new group_id
            continue;
        }

        if (fabs(ppm2[i] - current_ppm) > cutoff_group) // assign a new group_id
        {
            current_group_id++;
            group_id[i] = current_group_id;
            current_ppm = ppm2[i];
        }
        else // assign the same group_id
        {
            group_id[i] = current_group_id;
            current_ppm = ppm2[i];
        }
    }

    /**
     * @brief peak_stop is a vector of the index of the last peak in each group.
     * and npeak_group is the number of groups (size of peak_stop)
     * This data structure is more convenient for later use.
     */
    int current_group_number = group_id[0];
    for (int i = 1; i < group_id.size(); i++)
    {
        if (group_id[i] != current_group_number)
        {
            peak_stop.push_back(i);
            current_group_number = group_id[i];
        }
    }
    peak_stop.push_back(group_id.size());
    npeak_group = peak_stop.size();

    return true;
};

/**
 * 1. Load individual simulated peak pos,height and assignment from a file, such as bmse000047_1.txt
 * 2. Simulate a spectrum, using b0,ndata and nzf and window function
 * 3. Run Deep Picker 1D to get peak list (with assignment to each protons). Peaks belong to same proton form a group.
 * @param b0 magnetic field strength
 * @param spectral_width spectral width in ppm. To match experimental peak width, this should also match experiments. But exacly center is not important
 * @param ndata number of data points in FID, should be the same as the experimental spectrum
 * @param nzf number of zero filling in FT1, should be the same as the experimental spectrum
 * @param apodization_method apodization method, such as "exponential", "kaiser", should be the same as the experimental spectrum
 * @param db_height_cutoff cutoff for database peak height. Peaks with height less than this value will be removed from database, relative to the tallest peak in the database
 * @param R2 R2 = 1/T2 in 1/s: relxation rate of the simulated spectrum. 3-5 is good value to match experiment for most small molecules
 */
bool db_match_1d::simulate_database(double b0, double spectral_width, int ndata, int nzf, std::string apodization_method, double db_height_cutoff, double R2)
{
    /**
     * load in simulated peak pos,height and assignment from a file, such as bmse000047_1.txt
     *
     */
    std::string fname = folder + "/" + id;

    /**
     * But id is bmse000047_1.tab, so we need to replace .tab with .txt to get correct file name
     */
    fname = fname.substr(0, fname.size() - 4) + ".txt";

    /**
     * Example of .txt file:
     *     2.3437    0.0079   12.0000
     *     2.0642    0.0049   11.0000
     *     2.0207    0.0137    9.0000
     *     1.9862    0.0050   10.0000
     * First column is ppm, second column is height, third column is assignment (proton index in the .mol file)
     * We will read them into varibles spin_simulation_ppm, spin_simulation_height, spin_simulation_assignment
     */
    std::vector<double> spin_simulation_ppm, spin_simulation_height;
    std::vector<int> spin_simulation_assignment;

    std::ifstream fin(fname);
    if (!fin)
    {
        std::cout << "Error: cannot open file " << fname << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(fin, line))
    {
        std::istringstream iss(line);
        double ppm, height;
        double assignment;                  // actually an interger, but saved as double in the file
        iss >> ppm >> height >> assignment; // only read the first three columns. Skip the rest if any (reserved for future use)
        spin_simulation_ppm.push_back(ppm);
        spin_simulation_height.push_back(height);
        spin_simulation_assignment.push_back((int)assignment);
    }
    fin.close();

    /**
     * Get a unique list of spin_simulation_assignment.
     * using std::set
     */
    std::set<int> spin_simulation_assignment_unique_set(spin_simulation_assignment.begin(), spin_simulation_assignment.end());

    /**
     * Simulate FID and run FFT to get simulated spectrum
     */
    double SW = spectral_width * b0;
    double dt2 = 1 / SW;

    /**
     * get ndata_power_of_2, which is the smallest power of 2
     * that is larger than or equal ndata
     */
    int ndata_power_of_2 = 1;
    while (ndata_power_of_2 < ndata)
    {
        ndata_power_of_2 *= 2;
    }

    std::vector<double> times_of_fid;

    for (int i = 0; i < ndata; i++)
    {
        times_of_fid.push_back(i * dt2);
    }

    std::vector<double> ppm, ppm0; // ppm of simulated spectrum

    for (int i = 0; i < ndata_power_of_2 * nzf; i++)
    {
        ppm0.push_back(i * spectral_width / (ndata_power_of_2 * nzf)); // [0 12] when combined with (un)fftshift below
    }

    /**
     * move the last 200 data points to the beginning of the vector for ppm
     * ppm will be [-small number 12-small number]
     * Because we don't want 0 ppm peak (for DSS) to be at edge of the spectrum
     */
    for (int i = 0; i < 200; i++)
    {
        ppm.push_back(ppm0[ndata_power_of_2 * nzf - 200 + i] - spectral_width);
    }
    for (int i = 200; i < ndata_power_of_2 * nzf; i++)
    {
        ppm.push_back(ppm0[i - 200]);
    }

    peak_stop.clear();
    npeak_group = 0;
    proton_index_of_each_group.clear();

    /**
     * loop through each element of spin_simulation_assignment_unique_set
     */
    for (std::set<int>::iterator it = spin_simulation_assignment_unique_set.begin(); it != spin_simulation_assignment_unique_set.end(); ++it)
    {
        int protin_index = int(*it);
        /**
         * Get entries in spin_simulation_ppm, spin_simulation_height, spin_simulation_assignment that match protin_index
         */
        std::vector<double> spin_simulation_ppm_temp, spin_simulation_height_temp;
        for (int i = 0; i < spin_simulation_assignment.size(); i++)
        {
            if (spin_simulation_assignment[i] == protin_index)
            {
                spin_simulation_ppm_temp.push_back(spin_simulation_ppm[i]);
                spin_simulation_height_temp.push_back(spin_simulation_height[i]);
            }
        }

        /**
         * Simulate FID from spin_simulation_ppm_temp, spin_simulation_height_temp
         * remember ndata is the same as experimental spectrum (# of complex data points acquired)
         */
        std::vector<float> fid_real(ndata, 0.0f), fid_imag(ndata, 0.0);

        for (int i = 0; i < spin_simulation_ppm_temp.size(); i++)
        {
            for (int j = 0; j < ndata; j++)
            {
                fid_real[j] += spin_simulation_height_temp[i] * std::cos(2.0 * M_PI * b0 * spin_simulation_ppm_temp[i] * times_of_fid[j]) * std::exp(-times_of_fid[j] * R2);
                /**
                 * fid_imag is the imaginary part of FID.
                 * plus sign here to be consistent with kiss_fft,so that I will get same result as my Matlab code.
                 */
                fid_imag[j] += spin_simulation_height_temp[i] * std::sin(2.0 * M_PI * b0 * spin_simulation_ppm_temp[i] * times_of_fid[j]) * std::exp(-times_of_fid[j] * R2);
            }
        }

        /**
         * generate apodiization function
         */
        std::vector<std::string> apodization_string_split;
        int n_fileds = ldw_math_db_match_1d::split(apodization_method, apodization_string_split, ' ');

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
                    std::cerr << "Error: apodization function kaiser requires 3 parameters.  Skip apodization." << std::endl;
                }
                else
                {
                    double beta = std::stod(apodization_string_split[1]);
                    double alpha = std::stod(apodization_string_split[2]);
                    double gamma = std::stod(apodization_string_split[3]);

                    /**
                     * Apply apodization function to fid_real and fid_imag
                     */
                    fid_real[0] *= 0.5;
                    fid_imag[0] *= 0.5;
                    for (int i = 1; i < ndata; i++)
                    {
                        double sp = pow(sin(M_PI * beta + M_PI * alpha / 2.0 / ndata * i), gamma);
                        fid_real[i] *= sp;
                        fid_imag[i] *= sp;
                    }
                }
            }
            else if (apodization_string_split[0] == "none" || apodization_string_split[0] == "null" || apodization_string_split[0] == "n" || apodization_string_split[0] == "no")
            {
                // do nothing. Skip apodization
            }
            else
            {
                std::cerr << "Error: apodization function name must be kaiser or none. Skip apodization." << std::endl;
            }
        }
        else
        {
            std::cerr << "Error: apodization function name must be kaiser or none. Skip apodization." << std::endl;
        }

        kiss_fft_cfg cfg;
        kiss_fft_cpx *in, *out;

        int ndata_frq = ndata_power_of_2 * nzf;

        in = new kiss_fft_cpx[ndata_frq];
        out = new kiss_fft_cpx[ndata_frq];
        for (int i = 0; i < ndata; i++)
        {
            in[i].r = fid_real[i];
            in[i].i = fid_imag[i];
        }

        /**
         * fill remaining of in with 0, i.e., zero filling according to nzf
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

        /**
         * @var spectrum_real, spectrum_imag: real and imaginary part of the simulated spectrum in frequency domain
         * We don't need swap (fftshift) because how we generate ppm above
         */
        std::vector<float> spectrum_real, spectrum_imag, spectrum_real_temp, spectrum_imag_temp;
        spectrum_real_temp.resize(ndata_frq);
        spectrum_imag_temp.resize(ndata_frq);

        for (int i = 0; i < ndata_frq; i++)
        {
            spectrum_real_temp[i] = out[i].r / sqrt(float(ndata_frq));
            spectrum_imag_temp[i] = out[i].i / sqrt(float(ndata_frq)); // scale by sqrt(ndata_frq) to be consistent with standard fft
        }

        /**
         * move the last 200 data points to the beginning of the vector for spectrum_real and spectrum_imag to be consistent with ppm
         */
        for (int i = 0; i < 200; i++)
        {
            spectrum_real.push_back(spectrum_real_temp[ndata_frq - 200 + i]);
            spectrum_imag.push_back(spectrum_imag_temp[ndata_frq - 200 + i]);
        }
        for (int i = 200; i < ndata_frq; i++)
        {
            spectrum_real.push_back(spectrum_real_temp[i - 200]);
            spectrum_imag.push_back(spectrum_imag_temp[i - 200]);
        }

        // #ifdef DEBUG
        // std::ofstream fdebug("debug-spectrum-real.txt");
        // for (int i = 0; i < ndata_frq; i++)
        // {
        //     fdebug << ppm[i] << " " << spectrum_real[i] << std::endl;
        // }
        // fdebug.close();
        // #endif

        /**
         * Run Deep Picker 1D to get peak list (with assignment to each protons). Peaks belong to same proton form a group.
         * There is no need to run spectrum_pick_1d, because simulated spectrum is very clean with less dynamic range.
         * At r2=6.0, FWHH is about 45 data points, when ndata_frq=262144. So we can estiamte fwhh analytically.
         */
        double fwhh = 45.0 / 262144.0 * ndata_frq;

        /**
         * Optimal DP model 2 fwhh is 6.0.
         */
        nstride_database_spectrum = int(fwhh / 6.0 + 0.5); // stride is 1/6 of fwhh

        /**
         * nstride_database_spectrum must >=1
         */
        if (nstride_database_spectrum < 1)
        {
            nstride_database_spectrum = 1;
        }

        /**
         * Get strided real spectrum and corresponding ppm
         */
        std::vector<float> spectrum_real_strided, ppm_strided;
        for (int i = 0; i < ndata_frq; i += nstride_database_spectrum)
        {
            spectrum_real_strided.push_back(spectrum_real[i]);
            ppm_strided.push_back(ppm[i]);
        }

        /**
         * begin, step and stop of simulated spectrum, not to be confused with global variables begin,step,step
         * which are for the experimental spectrum
         */
        double simualted_spectrum_begin, simualted_spectrum_step, simualted_spectrum_stop; //

        simualted_spectrum_begin = ppm_strided[0];
        simualted_spectrum_step = (ppm_strided[spectrum_real_strided.size() - 1] - ppm_strided[0]) / double(spectrum_real_strided.size() - 1);
        simualted_spectrum_stop = ppm_strided[spectrum_real_strided.size() - 1] + simualted_spectrum_step;

        /**
         * step2 is used to convert from points to ppm when write sigma and gamma of database peaks to json file
         * step2 is set by user when loading database peak directly.
         * step2 need to be set to simualted_spectrum_step when simulating database peaks
         */
        step2 = simualted_spectrum_step;

        /**
         * peak1d is a class defined in dnn_picker.h
         * We will use it to get peak list, using model 2
         */
        class spectrum_pick_1d peak_picker;

        peak_picker.n_verbose = -1; // suppress output

        /**
         * set parameters for peak_picker.
         * In our simulated spectrum, a single peak from one proton has a peak height of 1.0
         */
        peak_picker.init(10 /*peak height cutoff*/, 3.0 /* noise floor*/, 0.001 /*noise level*/);
        peak_picker.init_mod(2); // DNN model 2, optimal fwhh is 6.0.

        /**
         * set spectrum for peak_picker
         */
        peak_picker.set_spectrum_from_data(spectrum_real_strided, simualted_spectrum_begin, simualted_spectrum_step, simualted_spectrum_stop);

        /**
         * run the peak picker, without negative peaks
         */
        peak_picker.spectrum_pick_1d_work(false /*b_negative*/);

        /**
         * Get the peak list form picker
         */
        spectrum_1d_peaks peak_list;
        peak_picker.get_peaks(peak_list);

        class spectrum_fit_1d peak_fitter;
        /**
         * save initial values as picker
         */
        peak_fitter.init(10 /*peak height cutoff*/, 3.0 /* noise floor*/, 0.001 /*noise level*/);
        peak_fitter.init_fit(2 /** voigt*/, 50 /*maxround*/, 0.0000001 /* to near cutoff*/);
        peak_fitter.set_spectrum_from_data(spectrum_real_strided, simualted_spectrum_begin, simualted_spectrum_step, simualted_spectrum_stop);
        peak_fitter.set_for_one_spectrum();
        peak_fitter.set_peaks(peak_list);

        /**
         * run peak fitting to optimize peak position, height, width, etc.
         * Set n_verbose to 0 to suppress output
         */
        peak_fitter.peak_fitting();

        /**
         * clear peak_list before we get fitted peaks
         */
        peak_list.a.clear();
        peak_list.x.clear();
        peak_list.ppm.clear();
        peak_list.volume.clear();
        peak_list.sigmax.clear();
        peak_list.gammax.clear();
        peak_list.confidence.clear();

        /**
         * Get peak list after fitting
         */
        peak_fitter.get_fitted_peaks(peak_list);

        /**
         * Find the maximum peak height for this peak group in the simulated spectrum
         */
        double max_peak_height = 0.0;
        for (int i = 0; i < peak_list.a.size(); i++)
        {
            if (peak_list.a[i] > max_peak_height)
            {
                max_peak_height = peak_list.a[i];
            }
        }

        /**
         * Append peak with height >= db_height_cutoff*max_peak_height to ppm2, amplitude2
         * For varification purpose, get total volume of each peak
         */
        double temp_volume = 0.0;
        for (int i = 0; i < peak_list.a.size(); i++)
        {
            if (peak_list.a[i] >= db_height_cutoff * max_peak_height)
            {
                ppm2.push_back(peak_list.ppm[i]);
                amplitude2.push_back(peak_list.a[i]);
                vol2.push_back(peak_list.volume[i]);
                temp_volume += peak_list.volume[i];
                sigmax2.push_back(peak_list.sigmax[i]);
                gammax2.push_back(peak_list.gammax[i]);
            }
        }
        total_volume.push_back(temp_volume);

        peak_stop.push_back(ppm2.size());
        proton_index_of_each_group.push_back(protin_index);

    } // end of loop through each element of spin_simulation_assignment_unique_set

    npeak_group = peak_stop.size();

    /**
     * Get max_value_of_amplitude2 from amplitude2
     */
    max_value_of_amplitude2 = 0.0;
    for (int i = 0; i < amplitude2.size(); i++)
    {
        /**
         * Skip water region peaks!!
         */
        if (fabs(ppm2[i] - (4.7 + reference_correction)) < water_width)
        {
            continue;
        }
        if (amplitude2[i] > max_value_of_amplitude2)
        {
            max_value_of_amplitude2 = amplitude2[i];
        }
    }

    if (n_verbose_match_1d > 0)
    {
        for (int i = 0; i < total_volume.size(); i++)
        {
            std::cout << "Total volume of peak group " << i << " is " << total_volume[i] << std::endl;
        }
    }

    return true;
}

bool db_match_1d::try_alternative_match(bool b_keep_bdl, bool b_keep_covered)
{
    /**
     * @brief At this time, only do this if npeak_group==1 && ppm2.size()==1
     */
    if (npeak_group != 1 || ppm2.size() != 1)
    {
        return false;
    }

    // if the best match is not normal, then no need to try alternative match
    if (final_best_assignment_type[0] != MatchType_normal)
    {
        return false;
    }

    int ndx = final_best_assignment[0];

    // this db item doesn't share any experimental peak with other db items. no need to try alternative match
    if (matching_compound_index[ndx].size() < 2)
    {
        return false;
    }

    /**
     * @brief run_match() again after we define a addtional cost for each experiment peaks
     * to penalize the match if the experimental peak is aleady used by others.
     */

    // reset the additional cost
    clashing_additional_cost.clear();
    clashing_additional_cost.resize(ppm.size(), 0.0);

    for (int i = 0; i < matching_compound_index.size(); i++)
    {
        if (matching_compound_index[i].size() > 0)
        {
            /**
             * @brief //as a reference, 0.02 ppm shift is equivalent to 0.3 (cost1). bdl is 1.0 and covered is 0.6 by default
             * As a result, most likely we will match another nearby free peak within cutoff
             * if not possible, then we will match the same peak again as a covered match.
             */
            clashing_additional_cost[i] = 1.0;
        }
    }

    /**
     * @brief We need to reset clashing_additional_cost to 0.0 if that peak is ONLY used by the current db item
     */
    for (int i = 0; i < matching_compound_index.size(); i++)
    {
        if (matching_compound_index[i].size() == 1 && matching_compound_index[i][0] == db_index)
        {
            clashing_additional_cost[i] = 0.0;
        }
    }

    std::cout << "Try alternative match for " << db_index << " " << name << std::endl;
    run_match(b_keep_bdl, b_keep_covered);
    std::cout << "Done alternative match for " << db_index << " " << name << std::endl;

    return true;
}

bool db_match_1d::run_match(double lb, double up, bool b_keep_bdl, bool b_keep_covered)
{

    pms.clear(); // clear the previous match result if any

#ifdef DEBUG
    std::ofstream fdebug(base_name + "_debug1.txt");
#else
    std::ofstream fdebug;
#endif

    std::vector<int> need_to_run(npeak_group, 1); // 1 means need to run, 0 means no need to run
    for (int i = 0; i < npeak_group; i++)
    {
        int b = 0;
        if (i > 0)
        {
            b = peak_stop[i - 1];
        }
        std::vector<double> ppm2_temp(ppm2.begin() + b, ppm2.begin() + peak_stop[i]);

        // get the center of the peak group (ppm2_temp)
        double center = 0;
        for (int j = 0; j < ppm2_temp.size(); j++)
        {
            center += ppm2_temp[j];
        }
        center /= ppm2_temp.size();

        // if center is within water_width of 4.7 ppm, we will skip this group
        if (fabs(center - (4.7 + reference_correction)) < water_width)
        {
            need_to_run[i] = 0;
        }
    }

    /**
     * If there is no peak group to run (all are within water region)
     * We will still run them.
     */
    bool b_all_within_water = true;
    for (int i = 0; i < need_to_run.size(); i++)
    {
        if (need_to_run[i] == 1)
        {
            b_all_within_water = false;
            break;
        }
    }

    if (b_all_within_water)
    {
        for (int i = 0; i < need_to_run.size(); i++)
        {
            need_to_run[i] = 1;
        }
    }

    for (int i = 0; i < npeak_group; i++)
    {
        int b = 0;
        if (i > 0)
        {
            b = peak_stop[i - 1];
        }
        std::vector<double> ppm2_temp(ppm2.begin() + b, ppm2.begin() + peak_stop[i]);
        std::vector<double> intensity2_temp(amplitude2.begin() + b, amplitude2.begin() + peak_stop[i]);

        if (need_to_run[i] == 0)
        {
            if (n_verbose_match_1d > 0)
                std::cout << "Skip group " << i << " because it is within water region , which has " << ppm2_temp.size() << " peaks" << std::endl;
            pattern_match_1d pm;
            pm.db_index = db_index;
            pm.init(i, cutoff, cutoff_pattern, intensity2_temp, ppm2_temp, lb, up, max_value_of_amplitude2);
            pm.run_fake(); // fake run, will assign all peaks to MatchType_water
            pms.emplace_back(pm);
        }
        else
        {
            std::vector<std::vector<int>> assignment_tmp;
            std::vector<std::vector<MatchType>> assignment_type_tmp;

            /**
             * pm is one peak group. It is a class defined in pattern_match_1d.h
             */
            pattern_match_1d pm;
            pm.db_index = db_index;

            /**
             * cutoff will be reset to pka_shift[i] if pka_shift[i] is larger than cutoff
             */
            double cutoff_temp = cutoff;
            if (pka_shift[i] > cutoff)
            {
                cutoff_temp = pka_shift[i];
            }

            /**
             * initialize pattern matching object pm for peak group i
             */
            pm.init(i, cutoff_temp, cutoff_pattern, intensity2_temp, ppm2_temp, lb, up, max_value_of_amplitude2);

            /**
             * Update assignment_tmp and assignment_type_tmp
             */
            pm.run();

            /**
             * update assignment_cost and assignment_min_intensity
             */
            pm.calculate_cost(fdebug);
            pms.emplace_back(pm);
            if (n_verbose_match_1d > 0)
            {
                std::cout << "Total " << pm.get_number_of_assignment() << " assignments for group " << i << ", which has " << ppm2_temp.size() << " peaks" << std::endl;
                std::cout << std::endl;
            }
        }
    }
#ifdef DEBUG
    fdebug.close();
#endif
    if (n_verbose_match_1d > 0)
    {
        std::cout << "Finished matching of peak clusters" << std::endl;
    }

    /**
     * @brief find the best assignment
     * scores of each pms are segmental linear interpolation of the cost at each key_intensity
     * So that we only need to find the minimum of the scores at all the key_intensities, from all the pms
     * Both are log of the intensity
     */
    std::vector<double> check_intensities, check_intensities_unique, allowed_intensities_upper_bound;
    // run get_key_intensity of each pm in pms and concatenate the result
    for (int i = 0; i < pms.size(); i++)
    {
        /**
         * Update check_intensities, from assignment_min_intensity of all pms
         * and update allowed_intensities_upper_bound, from allowed_intensity_upper_bound of all pms
         */
        pms[i].get_key_intensity(check_intensities, allowed_intensities_upper_bound);
    }

    /**
     * Get the lowest of allowed_intensities_upper_bound (one value is from one PMS)
     */
    double allowed_intensity_upper_bound = allowed_intensities_upper_bound[0];
    for (int i = 1; i < allowed_intensities_upper_bound.size(); i++)
    {
        if (allowed_intensities_upper_bound[i] < allowed_intensity_upper_bound)
        {
            allowed_intensity_upper_bound = allowed_intensities_upper_bound[i];
        }
    }

    // this is a special case, when pms.size() == 0
    if (check_intensities.size() == 0)
    {
        if (n_verbose_match_1d > 0)
        {
            std::cout << "No peak cluster is matched. No assignment is found." << std::endl;
        }
        v_fitted = 0;
        v_fitted2 = 0;
        return false;
    }

    std::sort(check_intensities.begin(), check_intensities.end()); // ascending order

    /**
     * Remove duplicates from check_intensities. The result is stored in check_intensities_unique
     */
    check_intensities_unique.push_back(check_intensities[0]);
    for (int i = 1; i < check_intensities.size(); i++)
    {
        /**
         * Too big, no need to check further
         */
        if (check_intensities[i] > allowed_intensity_upper_bound)
            break;
        /**
         * skip if check_intensities[i] is too close to the last element of check_intensities_unique
         */
        if (fabs(check_intensities[i] - check_intensities_unique[check_intensities_unique.size() - 1]) > 0.2)
        {
            check_intensities_unique.push_back(check_intensities[i]);
        }
    }

    if (n_verbose_match_1d > 0)
    {
        std::cout << "Need to check total number of key intensities: " << check_intensities_unique.size() << " for " << pms.size() << " peak clusters" << std::endl;
    }

    /**
     * For any pair of pms, check for possible clash (same peak is assigned to both pms)
     * @param clash_matrix: 2D vector of int. 1 means clash, 0 means no clash, saved as 1D vector
     */
    std::vector<int> clash_matrix(pms.size() * pms.size(), 0);

    std::vector<std::set<int>> experimental_peak_assignment_pms(ppm.size());

    for (int i = 0; i < pms.size(); i++)
    {
        /**
         * get_possible_assigned_peak() will return a vector of int, which has same size as # of experimental peaks
         * 0 means not assigned, 1 means assigned as Matchtype_normal (from any assignment of this pm)
        */
        std::vector<int> ass1 = pms[i].get_possible_assigned_peak();
        for (int j = i + 1; j < pms.size(); j++)
        {
            std::vector<int> ass2 = pms[j].get_possible_assigned_peak();
            /**
             * if ass1 and ass2 are both 1 at the same index, then there is a clash
             */
            bool b_clash = false;
            for (int k = 0; k < ass1.size(); k++)
            {
                if (ass1[k] == 1 && ass2[k] == 1)
                {
                    b_clash = true;
                    experimental_peak_assignment_pms[k].insert(i);
                    experimental_peak_assignment_pms[k].insert(j);
                }
            }
            if (b_clash == true)
            {
                clash_matrix[i * pms.size() + j] = 1;
                clash_matrix[j * pms.size() + i] = 1;
            }
        }
    }

    /**
     * Breadth first search to find all possible pms groups that are connected together (clash with each other)
     */
    std::vector<std::deque<int>> pms_groups = ldw_math_db_match_1d::breadth_first(clash_matrix, pms.size());

    /**
     * Sort pms_gropus by size in acending order
    */
    std::sort(pms_groups.begin(), pms_groups.end(), [](const std::deque<int> &a, const std::deque<int> &b) { return a.size() < b.size(); });

// for each intensity in check_intensities_unique, find the best match
// find minimal total_cost of all pms for each intensity
#ifdef DEBUG
    std::ofstream fdebug2(base_name + "_debug2.txt");
#else
    std::ofstream fdebug2; // null as a placeholder
#endif
    int ndx_min_cost = -1;
    double min_cost = 100.0; // any assignment >=100 is not considered. All BDL assignments are less than 100


    /**
     * Loop through all the possible intensities and find the best assignment (lowest cost)
     */
    for (int i_inten = check_intensities_unique.size() - 1; i_inten >= 0; i_inten--)
    {
#ifdef DEBUG
        fdebug2 << "check_intensity=" << check_intensities_unique[i_inten] << std::endl;
#endif

        if (n_verbose_match_1d > 0)
        {
            std::cout << "check_intensity=" << check_intensities_unique[i_inten] << " at " << i_inten << std::endl;
        }

        /**
         * assignment_index_of_each_pms_at_this_intensity is a vector of int, which is the index of the assignment of each pms at this intensity
         * pms[3].assignment[assignment_index_of_each_pms_at_this_intensity[3]] is the best assignment of pms[3] at this intensity
         */
        std::vector<int> best_assignment_index_of_each_pms_at_this_intensity(pms.size(), -1);
        /**
         * Corresponding total_cost, which is the sum of cost of all pms at this intensity with their best assignment
         *  We prefer to match big peaks, so we reduce cost for big peaks
         */
        double total_cost = -check_intensities_unique[i_inten] * cost_reduce_high_peak;

        /**
         * Each pms group will be handled separately
         * pms_group is a vector of vectors. [group_index][pms_index]
         * eg. {{0,1,2},{3,4},{5}} means pms[0],pms[1],pms[2] are in group 0, pms[3],pms[4] are in group 1, pms[5] is in group 2
         */
        for (int i_pms = 0; i_pms < pms_groups.size(); i_pms++)
        {
            /**
             * Case 0: pms group has only one pms
             */
            if (pms_groups[i_pms].size() == 1)
            {
                /**
                 * costs is a vector of double, which is the cost at each assignment in pms[j]. Cost is sorted in ascending order by the function
                 * ndx is a vector of int, which is the index of the assignment in pms[j]. ndx is sorted in ascending order by the function
                 * cost[0] is the lowest cost, ndx[0] is the index of the assignment with the lowest cost
                 */
                std::vector<double> costs;
                std::vector<int> ndx;
                pms[pms_groups[i_pms][0]].calculate_cost_at_intensity_with_details(fdebug2, check_intensities_unique[i_inten], costs, ndx);
                total_cost += costs[0];
                best_assignment_index_of_each_pms_at_this_intensity[pms_groups[i_pms][0]] = ndx[0];
            }
            /**
             * For pms groups with more than one pms, we need to find the best assignment without clash. These pms need to be assigned together.
             */
            else
            {
                /**
                 * Now we have a pms group with more than one pms
                 * For convience, define a vector of "pms pointer" for this group
                 */
                std::vector<pattern_match_1d *> p_pms;
                for (int i = 0; i < pms_groups[i_pms].size(); i++)
                {
                    p_pms.push_back(&pms[pms_groups[i_pms][i]]);
                }

                /**
                 * these two varibles have the same size as p_pms (# of pms in this gorup)
                 * see below explanation of costs and ndx for details of each element of these two varibles
                 */
                std::vector<std::vector<int>> indices_of_all_pms;
                std::vector<std::vector<double>> costs_of_all_pms;

                for (int j = 0; j < p_pms.size(); j++)
                {
                    /**
                     * costs is a vector of double, which is the cost at each assignment in p_pms[j]. Cost is sorted in ascending order by the function
                     * ndx is a vector of int, which is the index of the assignment in pms[j]. ndx is sorted in ascending order by the function
                     * cost[0] is the lowest cost, ndx[0] is the index of the assignment with the lowest cost
                     * eg. cost={ 0.1, 0.2, 0.3, 0.4, 0.5}, ndx={3, 1, 0, 2, 4}. In p_pms object. assignment[3] has the lowest cost, assignment[1] has the second lowest cost, etc.
                     */
                    std::vector<double> costs;
                    std::vector<int> ndx;
                    p_pms[j]->calculate_cost_at_intensity_with_details(fdebug2, check_intensities_unique[i_inten], costs, ndx);                    

                    /**
                     * trucate costs and ndx to only keep assignments with cost < min_cost-total_cost to save time
                     * (min_cost is the lowest cost of all pms at this intensity, total_cost is the sum of cost of all pms at this intensity with their best assignment already found)
                     * but frist assignments are always kept to make sure we have at least one assignment, otherwise code may crash
                     */
                    int n_keep = 1;
                    for (int k = 2; k < costs.size(); k++)
                    {
                        if (costs[k] < min_cost - total_cost)
                        {
                            n_keep++;
                        }
                        else
                        {
                            break;
                        }
                    }
                    if (n_keep < costs.size())
                    {
                        costs.erase(costs.begin() + n_keep + 1, costs.end());
                        ndx.erase(ndx.begin() + n_keep + 1, ndx.end());
                    }

                    indices_of_all_pms.push_back(ndx);
                    costs_of_all_pms.push_back(costs);
                }



                std::vector<int> best_assigment;
                double best_cost;

                combinatory_optimization_solver(
                                                p_pms,
                                                indices_of_all_pms,
                                                costs_of_all_pms,
                                                best_assigment,
                                                best_cost
                                                );

                
                /**
                 * At this time, we have found the best assignment without clash
                 * which is array_of_current_assignments_of_each_pms[current_min_cost_assignment]
                 * and its cost is array_of_current_total_cost[current_min_cost_assignment]
                 */
                total_cost += best_cost;

                for (int j = 0; j < p_pms.size(); j++)
                {
                    /**
                     * @param jj: index of the pms in pms. For convience and code readability. Compiler will optimize it.
                     */
                    int jj = pms_groups[i_pms][j];
                    /**
                     * @param n: index of the assignment in pms[j], which is the best assignment of pms[j], conresponding to costs_of_all_pms[j],
                     * which is sorted in ascending order.
                     */
                    int n = best_assigment[j];
                    /**
                     * @param nn: index of the assignment in pms[j], which is the best assignment of pms[j], conresponding to indices_of_all_pms[j]
                     * which is unsorted index in pms[j]
                     */
                    int nn = indices_of_all_pms[j][n];

                    /**
                     * update best_assignment_index_of_each_pms_at_this_intensity for pms[jj]
                     */
                    best_assignment_index_of_each_pms_at_this_intensity[jj] = nn;
                }
            } // end of else (pms_groups[i_pms].size() > 1)

            /**
             * early end of loop through all the pms groups if total_cost is already larger than min_cost
             */
            if (total_cost > min_cost)
            {
                break;
            }

        } // end of loop through all the pms groups

        if (n_verbose_match_1d > 0)
        {
            std::cout << "Total cost at " << check_intensities_unique[i_inten] << " is " << total_cost << std::endl;
        }

#ifdef DEBUG
        fdebug2 << "total_cost=" << total_cost << " at " << check_intensities_unique[i_inten] << std::endl;
        fdebug2 << " best assignment of each PMS are ";
        for (int i = 0; i < best_assignment_index_of_each_pms_at_this_intensity.size(); i++)
        {
            fdebug2 << best_assignment_index_of_each_pms_at_this_intensity[i] << " ";
        }
        fdebug2 << std::endl
                << std::endl;
#endif

        /**
         * Remember we need to find best assignment from all the possible intensities.
         * total_cost is the best cost at this intensity.
         */

        if (total_cost < min_cost)
        {
            min_cost = total_cost;
            ndx_min_cost = i_inten;
            final_best_assignment.clear();
            final_best_assignment_type.clear();
            final_best_assignment_height_ratio.clear();
            final_best_assignment_ppm.clear();
            for (int j = 0; j < pms.size(); j++)
            {
                int nn = best_assignment_index_of_each_pms_at_this_intensity[j];
                pms[j].get_best_assignment_v2(final_best_assignment, final_best_assignment_type, final_best_assignment_height_ratio, final_best_assignment_ppm, nn);
            }
        }

    } // end of loop through all the possible intensities

    bool b_fit_result = fit_volume(b_keep_bdl, b_keep_covered); // fit volume. this function will update v_fitted and v_fitted2

    calculate_matched_ppm(); // calculate matched ppm. this function will update matched_ppm_with_group_adjust

    // output the best assignment to std::cout
    print_result(check_intensities_unique[ndx_min_cost], min_cost);

    // update matching_compound_index (static member variable)
    for (int i = 0; i < final_best_assignment.size(); i++)
    {
        if (final_best_assignment_type[i] == MatchType_normal)
        {
            matching_compound_index.at(final_best_assignment[i]).push_back(db_index);
        }
    }

    return b_fit_result;
};

/**
 * @brief This function is used to detect clash in the current assignment
 * @param current_assignment_of_each_pms_unsorted: current assignment of each pms, unsorted.
 * @param p_pms: vector of pms pointer. Same size as current_assignment_of_each_pms_unsorted and one-to-one correspondence
 * @return true if there is clash, false if there is no clash
 */
bool db_match_1d::clash_detection(const std::vector<int> current_assignment_of_each_pms_unsorted, const std::vector<pattern_match_1d *> &p_pms) const
{
    /**
     * @brief We need to check if there is any clash in the current assignment
     * peak_matched is a vector of int, whose size is the same as ppm.size(), total number of peaks in the experiment
     * -1 means no match, otherwise, it is the index of the matched peak in ppm2
     *
     */
    std::vector<int> peaks_matched(ppm.size(), -1);
    for (int i = 0; i < p_pms.size(); i++)
    {
        /**
         * @param ndx is the index of the assignment of p_pms
         */
        int ndx = current_assignment_of_each_pms_unsorted[i];
        /**
         * get the assignment of pms[i] from ndx. eg, [2,-1,4,13] means db peak 1,2,3,4 assigned to experimental peak 2,"bdl or covered", 4,13
         */
        std::vector<int> assignment = p_pms[i]->get_assignment(ndx);
        /**
         * update peaks_matched
         */
        for (int j = 0; j < assignment.size(); j++)
        {
            if (assignment[j] >= 0) // only update peaks_matched if assignment[j] is not -1 (covered, skipped or bdl)
            {
                if (peaks_matched[assignment[j]] != -1)
                {
                    /**
                     * already matched by other pms, so there is clash
                     */
                    return true;
                }
                else
                {
                    peaks_matched[assignment[j]] = i;
                }
            }
        }
    }
    /**
     * no clash if we reach here
     */
    return false;
}

bool db_match_1d::update_matching_compound_index()
{
    // update matching_compound_index (static member variable)
    for (int i = 0; i < final_best_assignment.size(); i++)
    {
        if (final_best_assignment_type[i] == MatchType_normal)
        {
            matching_compound_index.at(final_best_assignment[i]).push_back(db_index);
        }
    }
    return true;
}

/**
 * @brief calculate matched ppm.
 * For each peak group: get a global shift between database and matched experiment peak (for type normal)
 * If there is no normal peak, then global shift is 0
 * Copy the database ppm to matched_ppm_with_group_adjust and add the global shift to it
 *
 * matched_ppm_with_group_adjust will keep original ppm diff within group and the whole group is shifted
 * matched_ppm will NOT keep original ppm diff within group and the whole group is shifted
 *
 * IMPORTANT: This changed the meaning in the output json file. Required by the new web interface (colmar1d.js)
 * @return true
 */
bool db_match_1d::calculate_matched_ppm()
{
    /**
     * @brief clear matched_ppm_with_group_adjust and matched_ppm, then resize them to the size of ppm2
     */
    matched_ppm_with_group_adjust.clear();
    matched_ppm_with_group_adjust.resize(ppm2.size(), -100.0);

    matched_ppm.clear();
    matched_ppm.resize(ppm2.size(), -100.0);

    int b = 0;
    int e;
    for (int i = 0; i < pms.size(); i++)
    {
        e = b + pms[i].get_number_of_peaks();
        // get global shift from b to e
        double global_shift = 0.0;
        int count = 0;
        for (int j = b; j < e; j++)
        {
            if (final_best_assignment_type[j] == MatchType_normal)
            {
                global_shift += final_best_assignment_ppm[j] - ppm2[j];
                count++;
            }
        }
        // no need to set global_shift to 0.0 if count==0. It is already 0.0
        if (count > 0)
        {
            global_shift /= count;
        }
        /**
         * Copy from database ppm to matched_ppm_with_group_adjust and add global_shift
         */

        for (int j = b; j < e; j++)
        {
            matched_ppm_with_group_adjust[j] = ppm2[j] + global_shift;
        }

        /**
         * For type normal and covered, we will update matched_ppm to final_best_assignment_ppm
         * For type other than normal and covered, matched ppm is database ppm + global_shift
         */
        for (int j = b; j < e; j++)
        {
            if (final_best_assignment_type[j] == MatchType_normal)
            {
                matched_ppm[j] = final_best_assignment_ppm[j];
            }
            else
            {
                matched_ppm[j] = ppm2[j] + global_shift;
            }
        }

        b = e;
    }
    return true;
}

bool db_match_1d::fit_volume(bool b_keep_bdl, bool b_keep_covered)
{
    double v1 = 0.0;
    double v2 = 0.0;

    // std::vector<double> v1s, v2s, v3s; //These are for debug purpose only

    double v_upper_limit_bdl = 0.0;
    double v_upper_limit_covered = std::numeric_limits<double>::max();
    bool b_covered_set = false;

    for (int j = 0; j < final_best_assignment.size(); j++)
    {
        // get upper limit for BDL peak
        if (final_best_assignment_type[j] == MatchType_bdl)
        {
            if (amplitude2[j] > v_upper_limit_bdl)
            {
                v_upper_limit_bdl = amplitude2[j];
            }
        }
        else if (final_best_assignment_type[j] == MatchType_covered)
        {
            b_covered_set = true;
            if (final_best_assignment_height_ratio[j] < v_upper_limit_covered)
            {
                v_upper_limit_covered = final_best_assignment_height_ratio[j];
            }
        }
        else if (final_best_assignment_type[j] == MatchType_skipped)
        {
            // do nothing
        }
        else if (final_best_assignment_type[j] == MatchType_water)
        {
            // do nothing
        }
        else
        {
            // v1s.push_back(intensity[final_best_assignment[j]]);
            // v2s.push_back(amplitude2[j]);
            // v3s.push_back(intensity[final_best_assignment[j]] / amplitude2[j]);
            v1 += intensity[final_best_assignment[j]];
            v2 += amplitude2[j];
        }
    }

    // There is no normal peak, only covered or bdl
    if (v1 < 1e-100)
    {
        v_fitted = 0.0;
        v_fitted2 = 0.0;
        if (b_keep_bdl == true) // keep compounds whose all peaks are BDL. We must keep all covered if keep_bdl is true.
        {
            // set v_fitted to lower of  covered or bdl
            v_fitted = std::exp(log_detection_limit) / v_upper_limit_bdl;
            if (b_covered_set == true && v_fitted > v_upper_limit_covered)
            {
                v_fitted = v_upper_limit_covered;
            }
            v_fitted2 = v_fitted;
            return true;
        }
        else if (b_keep_covered == true && b_covered_set == true) // only keep compounds whose all peaks are covered, not BDL
        {
            v_fitted = v_upper_limit_covered;
            v_fitted2 = v_fitted;
            return true;
        }
        else
        {
            return false;
        }
    }
    // We have normal peak.
    // v_fitted is weighted average of intensity[final_best_assignment[j]] / amplitude2[j].
    // Weights are amplitude2 (of MatchType_normal peaks)
    else
    {
        v_fitted = v1 / v2;
        if (v_fitted * v_upper_limit_bdl > std::exp(log_detection_limit))
        {
            v_fitted = std::exp(log_detection_limit) / v_upper_limit_bdl;
            if (n_verbose_match_1d > 0)
            {
                std::cout << "Warning: v * v_upper_limit_bdl > std::exp(log_detection_limit), v * v_upper_limit_bdl=" << v_fitted * v_upper_limit_bdl << ", detection limit is " << std::exp(log_detection_limit) << std::endl;
                std::cout << "Set v_fitted from " << v_fitted << " to ";
                std::cout << v_fitted << std::endl;
            }
        }
        if (b_covered_set == true && v_fitted > v_upper_limit_covered)
        {
            v_fitted = v_upper_limit_covered;
            if (n_verbose_match_1d > 0)
            {
                std::cout << "Warning: v>v_upper_limit_covered. V=" << v_fitted << ", v_upper_limit_covered=" << v_upper_limit_covered << std::endl;
                std::cout << "Set v_fitted from " << v_fitted << " to ";
                std::cout << v_fitted << std::endl;
            }
        }

        // for each MatchType_normal peak, calculate v2, which is  v_fitted * effective_width.
        //  v_fitted2 is the weighted averaging of v2s, with weights being amplitude2[j]

        v_fitted2 = 0.0;
        double total_weight = 0.0;
        for (int j = 0; j < final_best_assignment.size(); j++)
        {
            if (final_best_assignment_type[j] == MatchType_normal)
            {
                int idx = final_best_assignment[j];
                v_fitted2 += effective_width[idx] * amplitude2[j];
                total_weight += amplitude2[j];
            }
        }
        v_fitted2 /= total_weight;
        v_fitted2 *= v_fitted / 10.0; // divide by 10 so that v2 is in the same order of magnitude as v1. Both are relative anyway.
        return true;
    }
    return true; // should never reach here
};

bool db_match_1d::print_result(double inten_check, double cost)
{
    // print match_idx to stdout, one number per line
    if (n_verbose_match_1d > 0)
    {
        std::cout << "Final result, inten_check is " << inten_check << " " << exp(inten_check) << ", final cost is " << cost << ", V is " << v_fitted << ", V2 is " << v_fitted2 << std::endl;
        for (int i = 0; i < final_best_assignment.size(); i++)
        {
            std::cout << final_best_assignment[i];
            if (final_best_assignment[i] >= 0)
            {
                std::cout << " " << ppm[final_best_assignment[i]];       // ppm: experimental peak
                std::cout << " " << intensity[final_best_assignment[i]]; // intensity: experimental peak
            }
            else
            {
                std::cout << " -1 -1";
            }
            std::cout << " " << ppm2[i];       // ppm2 is the ppm of the peak in the database
            std::cout << " " << amplitude2[i]; // amplitude2 is the intensity of the peak in the database
            switch (final_best_assignment_type[i])
            {
            case MatchType_normal:
                std::cout << " normal";
                break;
            case MatchType_covered:
                std::cout << " covered";
                break;
            case MatchType_bdl:
                std::cout << " bdl";
                break;
            case MatchType_skipped:
                std::cout << " skipped";
                break;
            case MatchType_water:
                std::cout << " suppressed";
                break;
            default:
                std::cout << " other";
                break;
            }
            std::cout << std::endl;
        }

        std::cout << std::endl;
    }
    return true;
};

bool db_match_1d::save_summary(std::ofstream &fout)
{
    double total_mean_confident_level = 0.0;
    for (int i = 0; i < pms.size(); i++)
    {
        total_mean_confident_level += pms[i].get_mean_confident_level();
    }

    fout << 0 << " "; // fake index
    fout << base_name << " ";
    fout << total_mean_confident_level << " ";
    fout << "0 0 "; // two place holders
    fout << v_fitted << " " << v_fitted2 << " ";
    return true;
};

bool db_match_1d::save_summary_json(Json::Value &root, Json::Value &root_name, Json::Value &root_link)
{
    root_name = name;
    root_link = link;

    root["base_name"] = base_name; // database item name, such as "bmse000015_1"
    root["v_fitted"] = v_fitted;   // peak height based compound concentration
    for (int i = 0; i < final_best_assignment.size(); i++)
    {
        root["match"][i] = final_best_assignment[i];           // index of the matched peak in the experimental peak list
        root["match_type"][i] = final_best_assignment_type[i]; // normal, covered, bdl, skipped, water
        /**
         * match_ppm is the database ppm + global shift of each pms group (peaks from one proton)
         */
        root["match_ppm_with_adjust"][i] = matched_ppm_with_group_adjust[i]; // matched ppm, keep database ppm diff within group
        root["match_ppm"][i] = matched_ppm[i];                               // matched ppm, don't keep database ppm diff within group
        root["original_ppm"][i] = matched_ppm[i];                            // initially matched ppm, we need it in global opt steps.
        // if(final_best_assignment[i]>=0)
        // normal peak, save ppm, intensity, confident_level
        if (final_best_assignment_type[i] == MatchType_normal)
        {
            root["match_confident_level"][i] = confident_level[final_best_assignment[i]]; // confident_level is the confident level of the peak in the experimental peak list
        }
        else if (final_best_assignment_type[i] == MatchType_covered)
        {
            root["match_confident_level"][i] = -1.0;
        }
        else
        {
            root["match_confident_level"][i] = -1.0;
        }
    }

    // also save the database peak list. ppm2 has same size as final_best_assignment
    for (int i = 0; i < ppm2.size(); i++)
    {
        root["database_ppm"][i] = ppm2[i];
        root["database_amplitude"][i] = amplitude2[i];
        /**
         * In optimization step, we will change amplitude of each database peak but keep intensity unchanged.
         * Before optimization, amplitude2[i] is the intensity of the peak in the database.
         */
        root["match_amplitude"][i] = amplitude2[i];
        root["database_sigma"][i] = sigmax2[i] * step2; // convert from point to ppm.
        root["database_gamma"][i] = gammax2[i] * step2; // convert from point to ppm.
        root["match_sigma"][i] = sigmax2[i] * step2;    // convert from point to ppm.
        root["match_gamma"][i] = gammax2[i] * step2;    // convert from point to ppm.

        /**
         * Calcuate peak area (intensity) of each database peak.
         * In optimization step, we will change sigma, gamma and amplitude of each database peak but keep intensity unchanged.
         */
        root["database_intensity"][i] = amplitude2[i] / voigt(0.0, sigmax2[i] * step2, gammax2[i] * step2);
    }

    // save number of peaks in each pms group and the number of peaks in each pms group that are normal assignment
    for (int i = 0; i < pms.size(); i++)
    {
        root["pms_group"][i]["n_peaks"] = pms[i].get_number_of_peaks();
        root["pms_group"][i]["n_normal"] = pms[i].get_number_normal();
    }

    double total_mean_confident_level = 0.0;
    for (int i = 0; i < pms.size(); i++)
    {
        total_mean_confident_level += pms[i].get_mean_confident_level();
    }
    root["confident_level"] = total_mean_confident_level; // more pms group, more confident. So we don;t need to divide by pms.size()

    return true;
};

// read in experimental peak list
bool db_match_1d::peak_reading(std::string infname, double ref_correction, double dl)
{
    bool b_read;
    std::string stab(".tab");
    std::string slist(".list");
    std::string sjson(".json");

    if (n_verbose_match_1d > 0)
    {
        std::cout << "read peaks from file " << infname << std::endl;
    }

    if (std::equal(stab.rbegin(), stab.rend(), infname.rbegin()))
    {
        b_read = peak_reading_pipe(infname);
    }
    else
    {
        b_read = false;
        std::cout << "ERROR: unknown peak list file format. Skip peaks reading." << std::endl;
    }

    // set gamma if it is not readed in.
    gammax.resize(ppm.size(), 1e-20);

    // add ref_correction to ppm
    for (int i = 0; i < ppm.size(); i++)
    {
        ppm[i] += ref_correction;
    }

    if (dl < 1e-20) // if dl is not set, set dl to minimal peak height
    {
        dl = std::numeric_limits<double>::max();
        for (int i = 0; i < intensity.size(); i++) // intensity is peak height in the experimental peak list
        {
            if (intensity[i] < dl)
                dl = intensity[i];
        }
    }

    if (dl < noise_level * 5.5)
    {
        if (n_verbose_match_1d > 0)
            std::cout << "Detection_limit is set to " << dl << " from minimal peak height" << std::endl;
    }
    else
    {
        if (n_verbose_match_1d > 0)
            std::cout << "Detection_limit is set to " << noise_level * 5.5 << " from noise level" << std::endl;
        dl = noise_level * 5.5;
    }
    log_detection_limit = log(dl);

    // print infor and finish
    if (n_verbose_match_1d > 0)
    {
        std::cout << "loaded in " << ppm.size() << " peaks." << std::endl;
    }

    if (ppm.size() == 0)
        b_read = false;
    return b_read;
};

bool db_match_1d::clear_clashing_additional_cost()
{
    // allocate memory for the matching_compound_index
    matching_compound_index.clear();
    matching_compound_index.resize(ppm.size());

    // init clashing_additional_cost to all 0
    clashing_additional_cost.clear();
    clashing_additional_cost.resize(ppm.size(), 0);

    return true;
}

bool db_match_1d::print_matching_compound_index(std::ofstream &fout)
{
    for (int i = 0; i < matching_compound_index.size(); i++)
    {
        fout << ppm[i] << " ";
        for (int j = 0; j < matching_compound_index[i].size(); j++)
        {
            fout << matching_compound_index[i][j] << " ";
        }
        fout << std::endl;
    }
    return true;
}

// read in database peaks
bool db_match_1d::peak_reading_database(std::string fname)
{
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;

    int index = -1;
    int xpos_ppm = -1;
    int sigma = -1;
    int gamma = -1;
    int height = -1;
    int vol = -1;

    /**
     * @brief These two varibles are for special peaks.
     *
     */

    int group_id_pos = -1;

    std::ifstream fin(fname);

    // test if file is open
    if (!fin.is_open())
    {
        return false;
    }

    getline(fin, line);
    iss.str(line);
    while (iss >> p)
    {
        ps.push_back(p);
    }
    if (ps[0] != "VARS")
    {
        std::cout << "First word of first line must be VARS." << std::endl;
        return false;
    }
    ps.erase(ps.begin()); // remove first words (VARS)
    for (int i = 0; i < ps.size(); i++)
    {
        if (ps[i] == "X_PPM")
        {
            xpos_ppm = i;
        }
        else if (ps[i] == "SIGMAX")
        {
            sigma = i;
        }
        else if (ps[i] == "SIMGAX")
        {
            sigma = i;
        } // typo in pipe2, written by myself before.
        else if (ps[i] == "GAMMAX")
        {
            gamma = i;
        }
        else if (ps[i] == "HEIGHT")
        {
            height = i;
        }
        else if (ps[i] == "VOL")
        {
            vol = i;
        }

        else if (ps[i] == "GROUP_ID")
        {
            group_id_pos = i;
        }
    }

    if (xpos_ppm == -1)
    {
        std::cout << "One or more required varibles are missing." << std::endl;
        return false;
    }

    getline(fin, line); // read in second line
    int c = 0;
    while (getline(fin, line))
    {
        c++;
        iss.clear();
        iss.str(line);
        ps.clear();
        while (iss >> p)
        {
            ps.push_back(p);
        }

        if (ps.size() < 4)
            continue; // empty line??

        if (xpos_ppm != -1)
        {
            ppm2.push_back(stof(ps[xpos_ppm]));
        }

        if (height != -1)
        {
            amplitude2.push_back(stof(ps[height]));
        }

        if (vol != -1)
        {
            vol2.push_back(stof(ps[vol]));
        }

        if (sigma != -1)
        {
            float s = std::stof(ps[sigma]);
            sigmax2.push_back(s); // from FWHH to sigma, suppose GAussian shape
        }

        if (gamma != -1)
        {
            float s = std::stof(ps[gamma]);
            gammax2.push_back(s); // from FWHH to sigma, suppose GAussian shape
        }

        if (group_id_pos != -1 && group_id_pos < ps.size())
        {
            group_id.push_back(std::stoi(ps[group_id_pos]));
        }
        else
        {
            group_id.push_back(-1);
        }
    }

    return true;
};

/**
 * @brief read in experimental peaks from a file in pipe format
 */
bool db_match_1d::peak_reading_pipe(std::string fname)
{
    std::string line, p;
    std::vector<std::string> ps;
    std::stringstream iss;

    int index = -1;
    int xpos_ppm = -1;
    int sigma = -1;
    int gamma = -1;
    int height = -1;
    int vol = -1;
    int confidence = -1;

    int xpos = -1;
    int delta_height = -1;
    int integral = -1;
    int width = -1;
    int round = -1;
    int comment = -1;
    int background = -1;

    std::ifstream fin(fname);
    getline(fin, line);
    iss.str(line);
    while (iss >> p)
    {
        ps.push_back(p);
    }
    if (ps[0] != "VARS")
    {
        std::cout << "First word of first line must be VARS." << std::endl;
        return false;
    }
    ps.erase(ps.begin()); // remove first words (VARS)
    for (int i = 0; i < ps.size(); i++)
    {
        if (ps[i] == "X_PPM")
        {
            xpos_ppm = i;
        }
        else if (ps[i] == "SIGMAX")
        {
            sigma = i;
        }
        else if (ps[i] == "SIMGAX")
        {
            sigma = i;
        } // typo in my own pipe format previously!!
        else if (ps[i] == "GAMMAX")
        {
            gamma = i;
        }
        else if (ps[i] == "HEIGHT")
        {
            height = i;
        }
        else if (ps[i] == "VOL")
        {
            vol = i;
        }
        else if (ps[i] == "CONFIDENCE")
        {
            confidence = i;
        }
        else if (ps[i] == "X_AXIS")
        {
            xpos = i;
        }
        else if (ps[i] == "DHEIGHT")
        {
            delta_height = i;
        }
        else if (ps[i] == "INTEGRAL")
        {
            integral = i;
        }
        else if (ps[i] == "XW")
        {
            width = i;
        }
        else if (ps[i] == "NROUND")
        {
            round = i;
        }
        else if (ps[i] == "ASS")
        {
            comment = i;
        }
        else if (ps[i] == "BACKGROUND")
        {
            background = i;
        }
    }

    if (xpos_ppm == -1)
    {
        std::cout << "One or more required varibles are missing." << std::endl;
        return false;
    }

    getline(fin, line); // read in second line
    int c = 0;
    while (getline(fin, line))
    {
        c++;
        iss.clear();
        iss.str(line);
        ps.clear();
        while (iss >> p)
        {
            ps.push_back(p);
        }

        if (ps.size() < 4)
            continue; // empty line??

        if (xpos_ppm != -1)
        {
            ppm.push_back(stof(ps[xpos_ppm]));
        }

        if (height != -1)
        {
            intensity.push_back(stof(ps[height]));
        }

        if (vol != -1)
        {
            volume.push_back(stof(ps[vol]));
        }

        if (sigma != -1)
        {
            sigmax.push_back(std::stof(ps[sigma]));
        }

        if (gamma != -1)
        {
            gammax.push_back(std::stof(ps[gamma]));
        }

        if (xpos != -1)
        {
            pos.push_back(std::stof(ps[xpos]));
        }

        if (delta_height != -1)
        {
            delta_intensity.push_back(std::stof(ps[delta_height]));
        }

        if (integral != -1)
        {
            peak_integral.push_back(std::stof(ps[integral]));
        }

        if (width != -1)
        {
            peak_width.push_back(std::stof(ps[width]));
        }

        if (round != -1)
        {
            nround.push_back(std::stoi(ps[round]));
        }

        if (comment != -1)
        {
            user_comments.push_back(ps[comment]);
        }

        if (background != -1)
        {
            backgroud_flag.push_back(std::stoi(ps[background])); // 0 means not background, 1 means background
        }
        else
        {
            backgroud_flag.push_back(0); // default is not background
        }

        if (confidence != -1)
        {
            confident_level.push_back(std::stod(ps[confidence]));
        }
        else
        {
            confident_level.push_back(1.0); // default value. All peaks are highly confident.
        }
    }

    if (intensity.size() != volume.size())
    {
        std::cout << "intensity and volume must have the same size." << std::endl;
        return false;
    }

    // get effective_width=volume/height
    if (volume.size() > 0)
    {
        for (int i = 0; i < volume.size(); i++)
        {
            effective_width.push_back(volume[i] / intensity[i]);
        }
    }

    return true;
};

/**
 * @brief write peaks to a file in pipe format
 * but with assignment information (which peak match which compound)
 * for MatchType::MatchType_normal peaks only
 * One addtional column is added to the end of the file
 * The name is "ASSIGNMENT"
 * The format is %3d, which is the index of the matching compound (from 0)
 * -1 means unassigned peak (not matched to any compound)
 * Compound information is saved in result.txt and/or result.json
 * @param fname file name to write
 * @param int_matched An array of 0 and 1, same length as the number of compounds in the DB. 1 means matched, 0 means not matched
 * @return true always true
 */
bool db_match_1d::peak_writing(std::string fname, std::vector<int> int_matched)
{
    /**
     * @brief construct a vector of int with length of the number of compounds in the DB, called p_2_matched_compound
     * p_2_matched_compound[5]=3 means 5th db item is 3th in the matched compounds list
     * p_2_matched_compound[6]=-1 means 6th db item is not matched
     */
    std::vector<int> p_2_matched_compound;
    p_2_matched_compound.resize(int_matched.size(), -1);
    int current_index = 0;
    for (int i = 0; i < int_matched.size(); i++)
    {
        if (int_matched[i] == 1)
        {
            p_2_matched_compound[i] = current_index;
            current_index++;
        }
    }

    FILE *fp = fopen(fname.c_str(), "w");

    fprintf(fp, "VARS INDEX X_AXIS X_PPM XW HEIGHT DHEIGHT ASS INTEGRAL VOL SIMGAX GAMMAX CONFIDENCE NROUND ASSIGNMENT\n");
    fprintf(fp, "FORMAT %%5d %%9.4f %%8.4f %%7.3f %%+e %%+e %%s %%+e %%+e %%f %%f %%f %%4d %%3d\n");
    for (unsigned int i = 0; i < ppm.size(); i++)
    {
        /**
         * @brief assignment is an index of the compounds (in Alldatabase.list)
         * -1 means not assignment to any DB
         * if more than 1, we use the 1st one.
         */
        int assignment_index = -1;
        if (matching_compound_index[i].size() > 0)
        {
            assignment_index = matching_compound_index[i][0];
            /**
             * @brief because not all db items are matched (included in result.txt and/or result.json)
             * we need to convert the assignment_index from index to all DB items to index to only matched compounds
             */
            assignment_index = p_2_matched_compound[assignment_index];
        }

        fprintf(fp, "%5d %9.4f %8.4f %7.3f %+e %+e %s %+e %+e %f %f %f %4d %3d",
                i + 1, pos[i], ppm[i], peak_width[i], intensity[i], delta_intensity[i],
                user_comments[i].c_str(), peak_integral[i], volume[i],
                sigmax[i], gammax[i], confident_level[i], nround[i], assignment_index);

        fprintf(fp, "\n");
    }
    fclose(fp);
    return true;
}

/**
 * @brief Copy spectrum from base class to this class and apply ref_correction (for experimental spectrum)
 * Because we use different vaibles for base class (io, which handle read and write) and this db_match_1d class.
 * We also setup verbose level to minimal for peak fitter.
 */
bool db_match_1d::copy_and_ref_correction_spectrum(double t)
{
    reference_correction = t;

    stop = stop1 + t;
    begin = begin1 + t;
    step = step1;

    // copy spectrum  from base class to this class
    // spectrum is a static variable in this class
    spectrum = spect;

    /**
     * Get max of spectrum
     */
    max_value_of_exp_spectrum = 0.0;
    for (int i = 0; i < spectrum.size(); i++)
    {
        if (spectrum[i] > max_value_of_exp_spectrum)
        {
            max_value_of_exp_spectrum = spectrum[i];
        }
    }

    /**
     * We need to use 1D peak fitter to generate databse peak on-the-fly,
     * so we set up the parameters for 1D peak fitter here.
     */
    shared_data_1d::n_verbose = 0;
    shared_data_1d::b_doesy = false;

    return true;
}

/**
 * @brief This function is used to calculate the effective_width of each peak in the experimental spectrum
 */
double db_match_1d::get_median_experimental_peak_width()
{
    std::vector<double> effective_width_sorted = effective_width;
    std::sort(effective_width_sorted.begin(), effective_width_sorted.end());

    /**
     * @brief n_middle is the index of the middle element in the sorted vector
     */
    int n_middle = int((effective_width_sorted.size() - 0.01) / 2.0);

    /**
     * Convert to ppm
     */
    return effective_width_sorted[n_middle] * fabs(step);
}

/**
 * @brief This function is used to calculate the effective_width of each peak in the database
 */
double db_match_1d::get_median_database_peak_width()
{
    std::vector<double> effective_db_peak_width;
    for (int i = 0; i < vol2.size(); i++)
    {
        effective_db_peak_width.push_back(vol2[i] / amplitude2[i]);
    }
    std::sort(effective_db_peak_width.begin(), effective_db_peak_width.end());

    /**
     * @brief n_middle is the index of the middle element in the sorted vector
     * it is size()/2 for even size, and (size()-1)/2 for odd size, effectively.
     */
    int n_middle = int((effective_db_peak_width.size() - 0.01) / 2.0);

    /**
     * Convert to ppm
     */
    return effective_db_peak_width[n_middle] * fabs(step2);
}



/**
 * @brief This function is a combinatorial optimization solver to find the best assignment of all pms without clash
 * @param p_pms a vector of pattern_match_1d
 * @param indices_of_all_pms a vector of vector of int, indices_of_all_pms[0][0]=3 means the 0th assignment (best) of pms[0] is 3
 * @param costs_of_all_pms a vector of vector of double, costs_of_all_pms[0][0]=3.0 means the cost of the 3th assignment (best, defined in indices_of_all_pms ) of pms[0] is 3.0
 * @param best_assignment a vector of int, best_assignment={0,2,4} means the best assignment combination is: pms[0] is 0, pms[1] is 2, pms[2] is 4
 * @param best_cost a double, best_cost is the total cost of the best assignment combination
*/
bool db_match_1d::combinatory_optimization_solver(
    std::vector<pattern_match_1d *> const p_pms,
    std::vector<std::vector<int>> const indices_of_all_pms,
    std::vector<std::vector<double>> const costs_of_all_pms,
    std::vector<int> &best_assignment,
    double &best_cost)
{

    /**
     * Define varibles to run iterative algorithm to find the best assignment without clash
     * this can also be done using either recursive algorithm or iterative algorithm (selected here)
     * These varibles have SAME size. Each element is for one possible assignment of all pms.
     *
     * @param array_of_current_assignments_of_each_pms  a vector of current assignment of each pms
     * array_of_current_assignments_of_each_pms[3]={0,2,3} means the assignment (3) of pms[0] is 0, pms[1] is 2, pms[2] is 3
     * We then need indices_of_all_pms[0][0], indices_of_all_pms[1][2], indices_of_all_pms[2][3] to get the assignment within each pms
     * and costs_of_all_pms[0][0]+costs_of_all_pms[1][2]+costs_of_all_pms[2][3] to get the total cost of this assignment
     *
     * @param array_of_current_total_cost: total cost of each assignment
     * @param array_of_current_clash: 0 means no clash, 1 means clash for each assignment
     * @param array_of_current_clash_pms_location: array_of_current_clash_pms_location[3]={0,0,1,0,1} means the 3rd assignment has clash among p_pms 2 and 4
     */
    std::vector<std::vector<int>> array_of_current_assignments_of_each_pms;
    std::vector<double> array_of_current_total_cost;
    std::vector<int> array_of_current_clash; // 0 means no clash, 1 means clash, C++ doesn't have bool type for vectors
    std::vector<std::vector<int>> array_of_current_clash_pms_location;

    /**
     * Get max number of assignments of all pms
     */
    int max_number_of_assignments = 0;
    for (int j = 0; j < p_pms.size(); j++)
    {
        if (max_number_of_assignments < costs_of_all_pms[j].size())
        {
            max_number_of_assignments = costs_of_all_pms[j].size();
        }
    }

    /**
     * Loop through levels of assignments (level is defined as sum of current_assignments_of_each_pms)
     * The first level is 0, which means all pms are assigned to the 0th assignment (lowest cost) in each pms
     * The second level is 1, which means only one pms are assigned to the 1st assignment while others are assigned to the first assignment
     * The third level is 2, which means only two pms are assigned to the 2nd assignment while others are assigned to the first assignment
     *  or two are assigned to the 1st assignment while others are assigned to the 0th  assignment, etc.
     * level < max_number_of_assignments is actually unnecessary, because we will break the loop when we have no new kids, which will happen when reaching the last level
     */

    array_of_current_assignments_of_each_pms.push_back(std::vector<int>(p_pms.size(), 0)); // initialize the first level, all 0

    /**
     * previosu_end is the index of the last element of array_of_current_assignments_of_each_pms
     * at the previous level +1 (per C++ convention). It is 0 for the 0th level
     */
    int previosu_end = 0;
    double current_min_cost = std::numeric_limits<double>::max() / 100.0;
    int current_min_cost_index = 0;

    for (int level = 0; level < max_number_of_assignments; level++)
    {
        int begin = previosu_end;
        int end = array_of_current_assignments_of_each_pms.size();

        /**
         * Step 1, for each assignment in array_of_current_assignments_of_each_pms[begin:end], 
         * we calcuate the total cost of the assignment
         * and get the min_cost_assignment
         */
        for (int i = begin; i < end; i++)
        {
            /**
             * Get total cost of current_assignment_of_each_pms and clash of current_assignment_of_each_pms
             */
            double total_cost_of_current_assignment = 0.0;
            for (int j = 0; j < p_pms.size(); j++)
            {
                total_cost_of_current_assignment += costs_of_all_pms[j][array_of_current_assignments_of_each_pms[i][j]];
            }
            array_of_current_total_cost.push_back(total_cost_of_current_assignment);
        }

        /**
         * Step 2, for each assignment from array_of_current_assignments_of_each_pms[begin:end], remove it if its total cost is higher than current_min_cost
         */
        for (int i = end - 1; i >= begin; i--)
        {
            if (array_of_current_total_cost[i] > current_min_cost)
            {
                array_of_current_total_cost.erase(array_of_current_total_cost.begin() + i);
                array_of_current_assignments_of_each_pms.erase(array_of_current_assignments_of_each_pms.begin() + i);
                /**
                 * if we remove an assignment before current_min_cost_index, we need to update current_min_cost_index
                 */
                if (i <= current_min_cost_index)
                {
                    current_min_cost_index--;
                }
            }
        }
        end = array_of_current_assignments_of_each_pms.size();

        /**
         * Step 3, check duplicate assignments in array_of_current_assignments_of_each_pms[begin:end] and remove them
         * flag_of_same_assignment has the size of end-begin, each element is -1 to indicate no duplicate
         * flag_of_same_assignment[4]=2 means the 4th element of array_of_current_assignments_of_each_pms is the same as the 2nd element
         */
        std::vector<int> flag_of_same_assignment(end - begin, -1);
        for (int k1 = 0; k1 < end - begin; k1++)
        {
            if (flag_of_same_assignment[k1] >= 0)
            {
                continue;
            }
            for (int k2 = k1 + 1; k2 < end - begin; k2++)
            {
                if (flag_of_same_assignment[k2] >= 0)
                {
                    continue;
                }
                bool b_same = true;
                for (int j = 0; j < p_pms.size(); j++)
                {
                    if (array_of_current_assignments_of_each_pms[begin + k1][j] != array_of_current_assignments_of_each_pms[begin + k2][j])
                    {
                        b_same = false;
                        break;
                    }
                }
                if (b_same)
                {
                    flag_of_same_assignment[k2] = k1;
                }
            }
        }

        /**
         * Step 3.2, remove duplicate assignments in array_of_current_assignments_of_each_pms[begin:end]
         * Update end to the new size of array_of_current_assignments_of_each_pms after removal.
         */
        for (int k2 = end - begin - 1; k2 >= 0; k2--)
        {
            if (flag_of_same_assignment[k2] >= 0)
            {
                array_of_current_assignments_of_each_pms.erase(array_of_current_assignments_of_each_pms.begin() + begin + k2);
                array_of_current_total_cost.erase(array_of_current_total_cost.begin() + begin + k2);
                /**
                 * No need to update array_of_current_clash or array_of_current_clash_pms_location, because we only add them after this loop
                 */
            }
        }
        end = array_of_current_assignments_of_each_pms.size();

        /**
         * Get clash of current_assignment_of_each_pms.
         * First, get unsorted assignment of each pms
         */
        for (int i = begin; i < end; i++)
        {
            std::vector<int> current_assignment_of_each_pms_unsorted;
            for (int k = 0; k < p_pms.size(); k++)
            {
                current_assignment_of_each_pms_unsorted.push_back(indices_of_all_pms[k][array_of_current_assignments_of_each_pms[i][k]]);
            }
            bool b_clash = clash_detection(current_assignment_of_each_pms_unsorted, p_pms);
            array_of_current_clash.push_back(int(b_clash));
        }

      
        for (int i = begin; i < end; i++)
        {
            /**
             * Update current_min_cost_assignment if b_clash is false and total_cost_of_current_assignment is lower
             */
            if (array_of_current_clash[i] == 0 && array_of_current_total_cost[i] < current_min_cost)
            {
                current_min_cost = array_of_current_total_cost[i];
                current_min_cost_index = i;
            }
        }

        /**
         * Step 5. For each assignment from array_of_current_assignments_of_each_pms[begin:end], we append its kids to array_of_current_assignments_of_each_pms
         * only if the assignment has clash. For assignment without clash, we don't append its kids because they will have higher cost
         * Before that, we need to set previosu_end to the current size of array_of_current_assignments_of_each_pms for the next level
         */
        previosu_end = end;

        for (int i = begin; i < end; i++)
        {
            /**
             * Continue to next assignment if there is no clash
             */
            if (array_of_current_clash[i] == 0)
            {
                continue;
            }

            /**
             * current_assignment_of_each_pms is for programer's convenience, compiler will optimize it
             */
            std::vector<int> current_assignment_of_each_pms = array_of_current_assignments_of_each_pms[i];

            /**
             * Kids are just parent + 1 at each location.
             * For example, parent [0 1 1] has three kids: [1 1 1], [0 2 1], [0 1 2], subject to size at each location
             * however, if p_pms[j] does not part of clash group, we don't need to add kids
             * we won't add kid neither if we already reach the max number of assignments of p_pms[j]
             */
            for (int j = 0; j < p_pms.size(); j++)
            {
                /**
                 * skip if not part of clash group
                 */
                // if (array_of_current_clash_pms_location[i][j] == 0)
                // {
                //     continue;
                // }

                std::vector<int> kid_assignment_of_each_pms = current_assignment_of_each_pms;
                kid_assignment_of_each_pms[j] += 1;

                /**
                 * kid_assignment_of_each_pms[j]  must < indices_of_all_pms[j].size()
                 */
                if (kid_assignment_of_each_pms[j] >= indices_of_all_pms[j].size())
                {
                    continue;
                }

                /**
                 * add to array_of_current_assignments_of_each_pms.
                 * We will check for duplicate and remove them later (Step 1 and 2 for next level)
                 */
                array_of_current_assignments_of_each_pms.push_back(kid_assignment_of_each_pms);
            }
        }

        /**
         * if there is no new kid, then we can break the loop
         */
        if (previosu_end == array_of_current_assignments_of_each_pms.size())
        {
            break;
        }

        if(n_verbose_match_1d>0)
        {
            std::cout<<"finish loop "<<level<<std::endl;
        }

    } // end of loop through levels of assignments

    /**
     * Get the best assignment and its cost
     */
    best_assignment = array_of_current_assignments_of_each_pms[current_min_cost_index];

    /**
     * IMPORTANT: in case all possible assignment (within min_cost cutoff) have clash, best_assignment will be updated but best_cost will not be updated
     * This will server as a flag that this is a bad assignment
    */
    best_cost = current_min_cost;

    return true;
}