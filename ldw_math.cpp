#include <vector>
#include <deque>
#include <algorithm>

#include "ldw_math.h"

namespace ldw_math
{
    void sortArr(std::vector<double> &arr, std::vector<int> &ndx) 
    { 
        std::vector<std::pair<double, int> > vp; 
    
        for (int i = 0; i < arr.size(); ++i) { 
            vp.push_back(std::make_pair(arr[i], i)); 
        } 
    
        std::sort(vp.begin(), vp.end()); 
    
        for (int i = 0; i < vp.size(); i++)
        { 
            ndx.push_back(vp[i].second);
        } 
    };


    double calcualte_median(std::vector<double> scores)
    {
        size_t size = scores.size();

        if (size == 0)
        {
            return 0; // Undefined, really.
        }
        else
        {
            sort(scores.begin(), scores.end());
            if (size % 2 == 0)
            {
                return (scores[size / 2 - 1] + scores[size / 2]) / 2;
            }
            else
            {
                return scores[size / 2];
            }
        }
    };

    int calcualte_median_int(std::vector<int> scores)
    {
        size_t size = scores.size();

        if (size == 0)
        {
            return 0; // Undefined, really.
        }
        else
        {
            sort(scores.begin(), scores.end());
            if (size % 2 == 0)
            {
                return (scores[size / 2 - 1] + scores[size / 2]) / 2;
            }
            else
            {
                return scores[size / 2];
            }
        }
    };

    std::vector<std::deque<int> > bread_first(int *neighbor, int n, int n_min_size)
    {
        std::vector<std::deque<int> > clusters;
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

                #pragma omp parallel for
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

            if (work2.size() >= n_min_size)
            {
                clusters.push_back(work2);
            }
        }

        return clusters;
    };

    
    //linear regression, y = ax + b. r is correlation coefficient
    bool linreg(const std::vector<double> x, const std::vector<double> y, double &a, double &b, double &r)
    {
        double   sumx = 0.0;                      /* sum of x     */
        double   sumx2 = 0.0;                     /* sum of x**2  */
        double   sumxy = 0.0;                     /* sum of x * y */
        double   sumy = 0.0;                      /* sum of y     */
        double   sumy2 = 0.0;                     /* sum of y**2  */

        int n = x.size();

        for (int i=0;i<n;i++){ 
            sumx  += x[i];       
            sumx2 += x[i]*x[i];;  
            sumxy += x[i] * y[i];
            sumy  += y[i];      
            sumy2 += y[i]*y[i];; 
        } 

        double denom = (n * sumx2 - sumx * sumx);
        if (denom == 0)
        {
            // singular matrix. can't solve the problem.
            a = 0;
            b = 0;
            r = 0;
            return false;
        }

        a = (n * sumxy  -  sumx * sumy) / denom;
        b = (sumy * sumx2  -  sumx * sumxy) / denom;
        r = (sumxy - sumx * sumy / n) / std::sqrt((sumx2 - sumx*sumx/n) * (sumy2 - sumy*sumy/n));
        
        return true; 
    }

    bool random_sampling_consensus(std::vector<double> x, std::vector<double> y, int max_round, double error_cutoff, int init_number, int n_inline_cutoff,double &a_best,double &b_best,double &r_best)
    {
        int iterations = 0;
        double bestErr = 1e10;
        std::vector<double> x_maybeInliers,y_maybeInliers,x_alsoInliers,y_alsoInliers;
        double a,b; //linear regression parameters
        double r; //correlation coefficient


        r_best=0.0; //initialize, r is from 0 to 1 while 1 means best
        int max_size=0;

        while(iterations++ < max_round)
        {
            x_maybeInliers.clear();
            y_maybeInliers.clear();
            x_alsoInliers.clear();
            y_alsoInliers.clear();

            //random permutation of 0 to x.size()-1
            std::vector<int> permutation(x.size());
            for(int i=0;i<x.size();i++)
            {
                permutation[i] = i;
            }
            std::random_shuffle(permutation.begin(),permutation.end());

            //select first n points from x and y using permutation
            for(int i=0;i<init_number;i++)
            {
                x_maybeInliers.push_back(x[permutation[i]]);
                y_maybeInliers.push_back(y[permutation[i]]);
            }
            
            //fit linear model using x_maybeInliers and y_maybeInliers        
            linreg(x_maybeInliers,y_maybeInliers,a,b,r);
            
            //select from n to end points from x and y using permutation
            for(int i=init_number;i<x.size();i++)
            {
                if(std::fabs(y[permutation[i]] - a*x[permutation[i]] - b) < error_cutoff)
                {
                    x_alsoInliers.push_back(x[permutation[i]]);
                    y_alsoInliers.push_back(y[permutation[i]]);
                }
            }

            //if size of alsoInliers + maubeInliers is greater than d, then we have found a good model
            if(x_alsoInliers.size() + x_maybeInliers.size() > n_inline_cutoff)
            {
                //fit linear model using x_maybeInliers and y_maybeInliers    
                x_maybeInliers.insert(x_maybeInliers.end(),x_alsoInliers.begin(),x_alsoInliers.end());
                y_maybeInliers.insert(y_maybeInliers.end(),y_alsoInliers.begin(),y_alsoInliers.end());
                linreg(x_maybeInliers,y_maybeInliers,a,b,r);

                //if size of alsoInliers + maubeInliers is greater than d, then we have found a good model

                // if(std::fabs(r)>r_best)
                if(x_alsoInliers.size() + x_maybeInliers.size() > max_size)
                {
                    max_size = x_alsoInliers.size() + x_maybeInliers.size();
                    r_best = r;
                    a_best = a;
                    b_best = b;
                }
            }
        }
        return true;
    };
};