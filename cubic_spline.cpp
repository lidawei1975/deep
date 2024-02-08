#include "cubic_spline.h"

/**
 * @brief Construct a new cublic spline::cublic spline object
 */
cublic_spline::cublic_spline(){

};

/**
 * @brief Destroy the cublic spline::cublic spline object
 */
cublic_spline::~cublic_spline(){

};

/**
 * @brief calculate coefficients of cubic spline.
 * For simplicity, we require at least 15 points because inverse matrix is pre-calculated for at least 15 points
 * Error from approximation is about 0.001 for 15 points and approach 0.0001 for 30 or more points
 */
template <typename T>
bool cublic_spline::calculate_coefficients(const std::vector<T> &y)
{
    /**
     * we suppose x is 0,1,2,3,4,5,6,7,8,...,n
    */
    n=y.size()-1;

    if (n < 14)
    {
        std::cerr << "Error: cubic spline alrorithm implemented here requires at least 15 points." << std::endl;
        return false;
    }

    dd.resize(n+1);
    dd[0] = 0.0; // boundary condition, second derivative at first point is 0
    for (int i = 1; i < n; i++)
    {
        dd[i] = 3.0 * (y[i + 1] - 2.0 * y[i] + y[i - 1]);
    }
    dd[n] = 0.0; // boundary condition, second derivative at last point is 0

    /**
     * m(0)=a0*dd[0:7], m(1)=a1*dd[0:8], m(2)=a2*dd[0:9], m(3)=a3*dd[0:10],
     * m(4)=a4*dd[0:11], m(5)=a5*dd[0:12], m(6)=a6*dd[0:13]
     *
     * m(7)=a7*dd[0:14]
     * m(8)=a7*dd[1:15]
     * m(9)=a7*dd[2:16]
     * ...
     * m(n-6)=flip(a6)*dd[n-13:n]
     * m(n-5)=flip(a5)*dd[n-12:n]
     * ...
     * m(n)=flip(a0)*dd[n-7:n]
     * (2:6 means 2,3,4,5,6)
     */

    m.clear();
    m.resize(n+1);
    m[0] = a0[0] * dd[0] + a0[1] * dd[1] + a0[2] * dd[2] + a0[3] * dd[3] + a0[4] * dd[4] + a0[5] * dd[5] + a0[6] * dd[6] + a0[7] * dd[7];
    m[1] = a1[0] * dd[0] + a1[1] * dd[1] + a1[2] * dd[2] + a1[3] * dd[3] + a1[4] * dd[4] + a1[5] * dd[5] + a1[6] * dd[6] + a1[7] * dd[7] + a1[8] * dd[8];
    m[2] = a2[0] * dd[0] + a2[1] * dd[1] + a2[2] * dd[2] + a2[3] * dd[3] + a2[4] * dd[4] + a2[5] * dd[5] + a2[6] * dd[6] + a2[7] * dd[7] + a2[8] * dd[8] + a2[9] * dd[9];
    m[3] = a3[0] * dd[0] + a3[1] * dd[1] + a3[2] * dd[2] + a3[3] * dd[3] + a3[4] * dd[4] + a3[5] * dd[5] + a3[6] * dd[6] + a3[7] * dd[7] + a3[8] * dd[8] + a3[9] * dd[9] + a3[10] * dd[10];
    m[4] = a4[0] * dd[0] + a4[1] * dd[1] + a4[2] * dd[2] + a4[3] * dd[3] + a4[4] * dd[4] + a4[5] * dd[5] + a4[6] * dd[6] + a4[7] * dd[7] + a4[8] * dd[8] + a4[9] * dd[9] + a4[10] * dd[10] + a4[11] * dd[11];
    m[5] = a5[0] * dd[0] + a5[1] * dd[1] + a5[2] * dd[2] + a5[3] * dd[3] + a5[4] * dd[4] + a5[5] * dd[5] + a5[6] * dd[6] + a5[7] * dd[7] + a5[8] * dd[8] + a5[9] * dd[9] + a5[10] * dd[10] + a5[11] * dd[11] + a5[12] * dd[12];
    m[6] = a6[0] * dd[0] + a6[1] * dd[1] + a6[2] * dd[2] + a6[3] * dd[3] + a6[4] * dd[4] + a6[5] * dd[5] + a6[6] * dd[6] + a6[7] * dd[7] + a6[8] * dd[8] + a6[9] * dd[9] + a6[10] * dd[10] + a6[11] * dd[11] + a6[12] * dd[12] + a6[13] * dd[13];

    for (int i = 7; i <= n-7; i++)
    {
        m[i] = a7[0] * dd[i - 7] + a7[1] * dd[i - 6] + a7[2] * dd[i - 5] + a7[3] * dd[i - 4] + a7[4] * dd[i - 3] + a7[5] * dd[i - 2] + a7[6] * dd[i - 1] + a7[7] * dd[i] + a7[8] * dd[i + 1] + a7[9] * dd[i + 2] + a7[10] * dd[i + 3] + a7[11] * dd[i + 4] + a7[12] * dd[i + 5] + a7[13] * dd[i + 6] + a7[14] * dd[i + 7];
    }


    m[n] = a0[7] * dd[n-7] + a0[6] * dd[n-6] + a0[5] * dd[n-5] + a0[4] * dd[n-4] + a0[3] * dd[n-3] + a0[2] * dd[n-2] + a0[1] * dd[n-1] + a0[0] * dd[n];
    m[n-1] = a1[8] * dd[n-8] + a1[7] * dd[n-7] + a1[6] * dd[n-6] + a1[5] * dd[n-5] + a1[4] * dd[n-4] + a1[3] * dd[n-3] + a1[2] * dd[n-2] + a1[1] * dd[n-1] + a1[0] * dd[n];
    m[n-2] = a2[9] * dd[n-9] + a2[8] * dd[n-8] + a2[7] * dd[n-7] + a2[6] * dd[n-6] + a2[5] * dd[n-5] + a2[4] * dd[n-4] + a2[3] * dd[n-3] + a2[2] * dd[n-2] + a2[1] * dd[n-1] + a2[0] * dd[n];
    m[n-3] = a3[10] * dd[n-10] + a3[9] * dd[n-9] + a3[8] * dd[n-8] + a3[7] * dd[n-7] + a3[6] * dd[n-6] + a3[5] * dd[n-5] + a3[4] * dd[n-4] + a3[3] * dd[n-3] + a3[2] * dd[n-2] + a3[1] * dd[n-1] + a3[0] * dd[n];
    m[n-4] = a4[11] * dd[n-11] + a4[10] * dd[n-10] + a4[9] * dd[n-9] + a4[8] * dd[n-8] + a4[7] * dd[n-7] + a4[6] * dd[n-6] + a4[5] * dd[n-5] + a4[4] * dd[n-4] + a4[3] * dd[n-3] + a4[2] * dd[n-2] + a4[1] * dd[n-1] + a4[0] * dd[n];
    m[n-5] = a5[12] * dd[n-12] + a5[11] * dd[n-11] + a5[10] * dd[n-10] + a5[9] * dd[n-9] + a5[8] * dd[n-8] + a5[7] * dd[n-7] + a5[6] * dd[n-6] + a5[5] * dd[n-5] + a5[4] * dd[n-4] + a5[3] * dd[n-3] + a5[2] * dd[n-2] + a5[1] * dd[n-1] + a5[0] * dd[n];
    m[n-6] = a6[13] * dd[n-13] + a6[12] * dd[n-12] + a6[11] * dd[n-11] + a6[10] * dd[n-10] + a6[9] * dd[n-9] + a6[8] * dd[n-8] + a6[7] * dd[n-7] + a6[6] * dd[n-6] + a6[5] * dd[n-5] + a6[4] * dd[n-4] + a6[3] * dd[n-3] + a6[2] * dd[n-2] + a6[1] * dd[n-1] + a6[0] * dd[n];


    /**
     * Now we have m, we can calculate a,b,c,d
     */
    a.clear();
    b.clear();
    c.clear();
    d.clear();

    a.resize(n);
    b.resize(n);
    c.resize(n);
    d.resize(n);

    for (int i = 0; i < n; i++)
    {
        a[i] = y[i];
        c[i] = m[i] / 2.0;
        d[i] = (m[i + 1] - m[i]) / 6.0;
        b[i] = y[i + 1] - a[i] - c[i] - d[i];
    }

    return true;
}

/**
 * @brief calculate value of cubic spline at x, after coefficients are calculated
 */

double cublic_spline::calculate_value(const double x) const
{
    int N = m.size();
    int i = int(x);
    if (i < 0)
    {
        i = 0;
    }
    else if (i >= n)
    {
        i = n - 1;
    }
    double dx = x - i;
    return a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
}


/**
 * expose the template function with explicit instantiation
 */
template bool cublic_spline::calculate_coefficients(const std::vector<int> &y);
template bool cublic_spline::calculate_coefficients(const std::vector<float> &y);
template bool cublic_spline::calculate_coefficients(const std::vector<double> &y);