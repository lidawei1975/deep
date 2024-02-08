#include <vector>
#include <array>
#include <fstream>
#include <iostream>

#ifndef CUBIC_SPLINE_HEADER
#define CUBIC_SPLINE_HEADER

class cublic_spline
{
private:
    /**
     * @brief y is the input data, function value at x
     * For this class, x is always 0,1,2,3,4,5,6,7,8,...,n
     * where N is the size of y
     * y will be provieded by the user in function call calculate_coefficients, not stored in this class
     * dd vector is the divided difference of y
     * size of y and dd is always the same = n+1
     */
    std::vector<double> dd;
    int n;

    /**
     * @brief m is the coefficients (secondary derivatice at each point) of cubic spline
     * m=inv(matrix)*dd. size of m is also n+1
     * See below for matrix
     */
    std::vector<double> m;

    /**
     * y=a+b*x+c*x^2+dd*x^3, in each interval
     * size of a,b,c,dd is n
     */
    std::vector<double> a, b, c, d;

    /**
     * We need to do matrix inversion to get the coefficients
     * However, the matrix is always the same like this because x is on equal interval
     * 4 1 0 0 0 0 0
     * 1 4 1 0 0 0 0
     * 0 1 4 1 0 0 0
     * 0 0 1 4 1 0 0
     * 0 0 0 1 4 1 0
     * 0 0 0 0 1 4 1
     * 0 0 0 0 0 1 4
     *
     * So we can pre-calculate the inverse matrix (approximate) and store it in
     * a series of vectors
     * The farest we can go is 7 points along both directions
     */
    std::array<double, 8> a0 = {{0.5359, -0.1436, 0.0385, -0.0103, 0.0028, -0.0007, 0.0002, -0.0001}};
    std::array<double, 9> a1 = {{-0.1436, 0.5744, -0.1539, 0.0412, -0.0110, 0.0030, -0.0008, 0.0002, -0.0001}};
    std::array<double, 10> a2 = {{0.0385, -0.1539, 0.5771, -0.1546, 0.0414, -0.0111, 0.0030, -0.0008, 0.0002, -0.0001}};
    std::array<double, 11> a3 = {{-0.0103, 0.0412, -0.1546, 0.5773, -0.1547, 0.0415, -0.0111, 0.0030, -0.0008, 0.0002, -0.0001}};
    std::array<double, 12> a4 = {{0.0028, -0.0110, 0.0414, -0.1547, 0.5773, -0.1547, 0.0415, -0.0111, 0.0030, -0.0008, 0.0002, -0.0001}};
    std::array<double, 13> a5 = {{-0.0007, 0.0030, -0.0111, 0.0415, -0.1547, 0.5774, -0.1547, 0.0415, -0.0111, 0.0030, -0.0008, 0.0002, -0.0001}};
    std::array<double, 14> a6 = {{0.0002, -0.0008, 0.0030, -0.0111, 0.0415, -0.1547, 0.5774, -0.1547, 0.0415, -0.0111, 0.0030, -0.0008, 0.0002, -0.0001}};
    std::array<double, 15> a7 = {{-0.0001, 0.0002, -0.0008, 0.0030, -0.0111, 0.0415, -0.1547, 0.5774, -0.1547, 0.0415, -0.0111, 0.0030, -0.0008, 0.0002, -0.0001}};

public:
    cublic_spline();
    ~cublic_spline();

    /**
     * @brief calculate coefficients of cubic spline.
     * Calcuated coefficients are double
     * but input data can be int, float or double
    */
    template <typename T>
    bool calculate_coefficients(const std::vector<T> &y);

    /**
     * @brief calculate value of cubic spline at x, after coefficients are calculated
     */
    double calculate_value(const double x) const;
};



#endif