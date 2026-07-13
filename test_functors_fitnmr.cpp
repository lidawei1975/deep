#include <cmath>
#include <iostream>
#include <vector>

#include "cost_functors_fitnmr.h"

namespace
{
double max_abs_diff(const std::vector<double> &a, const std::vector<double> &b)
{
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

double max_abs_value(const std::vector<double> &a)
{
    double max_val = 0.0;
    for (size_t i = 0; i < a.size(); i++)
    {
        max_val = std::max(max_val, std::fabs(a[i]));
    }
    return max_val;
}

std::vector<double> evaluate_residuals(
    mycostfunction_fitnmr_doublet_singlet &cost_function,
    const int n_residual,
    double &a,
    double &x0,
    double &r2x,
    double &j,
    double &y0,
    double &r2y)
{
    const double *params[] = {&a, &x0, &r2x, &j, &y0, &r2y};
    std::vector<double> residuals(n_residual);

    cost_function.Evaluate(params, residuals.data(), NULL);
    return residuals;
}

double none_singlet_value_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq)
{
    return (std::exp(aq*r2)*r2 - r2*std::cos(aq*(omega - omega0)) + (omega - omega0)*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2)));
}

double none_singlet_d_omega0_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq)
{
    return (2*std::exp(aq*r2)*(omega - omega0)*r2 - (omega - omega0)*(2*r2 + aq*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(aq*(omega - omega0)) - (-std::pow(omega - omega0,2) + aq*std::pow(omega - omega0,2)*r2 + std::pow(r2,2) + aq*std::pow(r2,3))*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2));
}

double none_singlet_d_r2_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq)
{
    return (-(std::exp(aq*r2)*(omega - omega0 + r2)*(-omega + omega0 + r2)) + (-std::pow(omega - omega0,2) + aq*std::pow(omega - omega0,2)*r2 + std::pow(r2,2) + aq*std::pow(r2,3))*std::cos(aq*(omega - omega0)) - (omega - omega0)*(2*r2 + aq*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2));
}

double sp1_singlet_value_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    return (aq*(std::exp(aq*r2)*(end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(off*PI) + aq*(omega - omega0)*std::sin(aq*(omega - omega0))*(2*aq*(end - off)*PI*r2*std::cos(end*PI) + (-(std::pow(end - off,2)*std::pow(PI,2)) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(end*PI)) + std::cos(aq*(omega - omega0))*(-((end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(end*PI)) - aq*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(end*PI)) + aq*std::exp(aq*r2)*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(off*PI)))/(std::exp(aq*r2)*(std::pow(end - off,4)*std::pow(PI,4) - 2*std::pow(aq,2)*std::pow(end - off,2)*std::pow(PI,2)*(std::pow(omega - omega0,2) - std::pow(r2,2)) + std::pow(aq,4)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2)));
}

double sp1_singlet_d_omega0_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    return (std::pow(aq,2)*(-4*aq*(omega - omega0)*(std::pow(end - off,2)*std::pow(PI,2) - std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*(std::exp(aq*r2)*(end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(off*PI) + aq*(omega - omega0)*std::sin(aq*(omega - omega0))*(2*aq*(end - off)*PI*r2*std::cos(end*PI) + (-(std::pow(end - off,2)*std::pow(PI,2)) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(end*PI)) + std::cos(aq*(omega - omega0))*(-((end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(end*PI)) - aq*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(end*PI)) + aq*std::exp(aq*r2)*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(off*PI)) + (std::pow(end - off,4)*std::pow(PI,4) - 2*std::pow(aq,2)*std::pow(end - off,2)*std::pow(PI,2)*(std::pow(omega - omega0,2) - std::pow(r2,2)) + std::pow(aq,4)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2))*(std::sin(aq*(omega - omega0))*(-((end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + 2*aq*r2 + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(end*PI)) - (-(std::pow(end - off,2)*std::pow(PI,2)) + aq*std::pow(end - off,2)*std::pow(PI,2)*r2 + std::pow(aq,3)*r2*(std::pow(omega - omega0,2) + std::pow(r2,2)) + std::pow(aq,2)*(3*std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(end*PI)) + aq*(omega - omega0)*std::cos(aq*(omega - omega0))*(-2*(end - off)*PI*(1 + aq*r2)*std::cos(end*PI) + (std::pow(end - off,2)*std::pow(PI,2) + aq*(2*r2 - aq*(std::pow(omega - omega0,2) + std::pow(r2,2))))*std::sin(end*PI)) - 2*aq*std::exp(aq*r2)*(omega - omega0)*((-end + off)*PI*std::cos(off*PI) + aq*r2*std::sin(off*PI)))))/(std::exp(aq*r2)*std::pow(std::pow(end - off,4)*std::pow(PI,4) - 2*std::pow(aq,2)*std::pow(end - off,2)*std::pow(PI,2)*(std::pow(omega - omega0,2) - std::pow(r2,2)) + std::pow(aq,4)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2),2));
}

double sp1_singlet_d_r2_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    return (std::pow(aq,2)*(-4*aq*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*(std::exp(aq*r2)*(end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(off*PI) + aq*(omega - omega0)*std::sin(aq*(omega - omega0))*(2*aq*(end - off)*PI*r2*std::cos(end*PI) + (-(std::pow(end - off,2)*std::pow(PI,2)) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(end*PI)) + std::cos(aq*(omega - omega0))*(-((end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(end*PI)) - aq*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(end*PI)) + aq*std::exp(aq*r2)*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(off*PI)) - (std::pow(end - off,4)*std::pow(PI,4) - 2*std::pow(aq,2)*std::pow(end - off,2)*std::pow(PI,2)*(std::pow(omega - omega0,2) - std::pow(r2,2)) + std::pow(aq,4)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2))*(std::exp(aq*r2)*(end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(off*PI) + aq*(omega - omega0)*std::sin(aq*(omega - omega0))*(2*aq*(end - off)*PI*r2*std::cos(end*PI) + (-(std::pow(end - off,2)*std::pow(PI,2)) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(end*PI)) + std::cos(aq*(omega - omega0))*(-((end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(end*PI)) - aq*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(end*PI)) + aq*std::exp(aq*r2)*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(off*PI)) + (std::pow(end - off,4)*std::pow(PI,4) - 2*std::pow(aq,2)*std::pow(end - off,2)*std::pow(PI,2)*(std::pow(omega - omega0,2) - std::pow(r2,2)) + std::pow(aq,4)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2))*(2*aq*(omega - omega0)*std::sin(aq*(omega - omega0))*((end - off)*PI*std::cos(end*PI) + aq*r2*std::sin(end*PI)) + std::cos(aq*(omega - omega0))*(2*aq*(-end + off)*PI*r2*std::cos(end*PI) - (std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) + 3*std::pow(r2,2)))*std::sin(end*PI)) + std::exp(aq*r2)*((end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + 2*aq*r2 + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::cos(off*PI) + (std::pow(end - off,2)*std::pow(PI,2) + aq*std::pow(end - off,2)*std::pow(PI,2)*r2 + std::pow(aq,3)*r2*(std::pow(omega - omega0,2) + std::pow(r2,2)) + std::pow(aq,2)*(std::pow(omega - omega0,2) + 3*std::pow(r2,2)))*std::sin(off*PI)))))/(std::exp(aq*r2)*std::pow(std::pow(end - off,4)*std::pow(PI,4) - 2*std::pow(aq,2)*std::pow(end - off,2)*std::pow(PI,2)*(std::pow(omega - omega0,2) - std::pow(r2,2)) + std::pow(aq,4)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2),2));
}

double sp2_singlet_value_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    return ((2*r2)/(std::pow(omega - omega0,2) + std::pow(r2,2)) - (2*r2*std::cos(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2))) + std::pow(aq,2)*r2*(-(1/(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - 1/(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)))*std::cos(2*off*PI) + (std::pow(aq,2)*r2*std::cos(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (std::pow(aq,2)*r2*std::cos(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*omega*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2))) - (2*omega0*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2))) + (std::pow(aq,2)*omega*std::sin(2*off*PI))/(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (std::pow(aq,2)*omega0*std::sin(2*off*PI))/(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (2*aq*end*PI*std::sin(2*off*PI))/(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (2*aq*off*PI*std::sin(2*off*PI))/(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (std::pow(aq,2)*omega*std::sin(2*off*PI))/(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (std::pow(aq,2)*omega0*std::sin(2*off*PI))/(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (2*aq*end*PI*std::sin(2*off*PI))/(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (2*aq*off*PI*std::sin(2*off*PI))/(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (std::pow(aq,2)*omega*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (std::pow(aq,2)*omega0*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*aq*end*PI*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (2*aq*off*PI*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (std::pow(aq,2)*omega*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (std::pow(aq,2)*omega0*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (2*aq*end*PI*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*aq*off*PI*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))))/4.;
}

double sp2_singlet_d_omega0_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    return ((4*(omega - omega0)*r2)/std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2) + (4*(-omega + omega0)*r2*std::cos(aq*(omega - omega0)))/(std::exp(aq*r2)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2)) - (2*aq*omega*std::cos(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2))) + (2*aq*omega0*std::cos(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2))) + 2*std::pow(aq,3)*r2*((aq*(-omega + omega0) + 2*(-end + off)*PI)/std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (aq*(-omega + omega0) + 2*(end - off)*PI)/std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2))*std::cos(2*off*PI) + (2*std::pow(aq,3)*(aq*(omega - omega0) + 2*(-end + off)*PI)*r2*std::cos(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::pow(aq,3)*omega*std::cos(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (std::pow(aq,3)*omega0*std::cos(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (2*std::pow(aq,2)*end*PI*std::cos(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,2)*off*PI*std::cos(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,3)*(aq*(omega - omega0) + 2*(end - off)*PI)*r2*std::cos(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::pow(aq,3)*omega*std::cos(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (std::pow(aq,3)*omega0*std::cos(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,2)*end*PI*std::cos(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (2*std::pow(aq,2)*off*PI*std::cos(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (4*omega*(omega - omega0)*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2)) + (4*omega0*(-omega + omega0)*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2)) - (2*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2))) - (2*aq*r2*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2))) + (2*std::pow(aq,3)*omega*(aq*(omega - omega0) + 2*(end - off)*PI)*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (4*std::pow(aq,2)*end*PI*(aq*(omega - omega0) + 2*(end - off)*PI)*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (2*std::pow(aq,3)*omega0*(aq*(-omega + omega0) + 2*(-end + off)*PI)*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (4*std::pow(aq,2)*off*PI*(aq*(-omega + omega0) + 2*(-end + off)*PI)*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) - (std::pow(aq,2)*std::sin(2*off*PI))/(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (2*std::pow(aq,3)*omega*(aq*(-omega + omega0) + 2*(end - off)*PI)*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (4*std::pow(aq,2)*off*PI*(aq*(-omega + omega0) + 2*(end - off)*PI)*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (2*std::pow(aq,3)*omega0*(aq*(omega - omega0) + 2*(-end + off)*PI)*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (4*std::pow(aq,2)*end*PI*(aq*(omega - omega0) + 2*(-end + off)*PI)*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (std::pow(aq,2)*std::sin(2*off*PI))/(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (2*std::pow(aq,3)*omega*(aq*(-omega + omega0) + 2*(end - off)*PI)*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (4*std::pow(aq,2)*off*PI*(aq*(-omega + omega0) + 2*(end - off)*PI)*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (2*std::pow(aq,3)*omega0*(aq*(omega - omega0) + 2*(-end + off)*PI)*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (4*std::pow(aq,2)*end*PI*(aq*(omega - omega0) + 2*(-end + off)*PI)*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::pow(aq,2)*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (std::pow(aq,3)*r2*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,3)*omega0*(aq*(omega - omega0) + 2*(end - off)*PI)*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (4*std::pow(aq,2)*off*PI*(aq*(omega - omega0) + 2*(end - off)*PI)*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (2*std::pow(aq,3)*omega*(aq*(-omega + omega0) + 2*(-end + off)*PI)*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (4*std::pow(aq,2)*end*PI*(aq*(-omega + omega0) + 2*(-end + off)*PI)*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::pow(aq,2)*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (std::pow(aq,3)*r2*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))))/4.;
}

double sp2_singlet_d_r2_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    return ((2*std::pow(omega - omega0,2) - 2*std::pow(r2,2))/std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2) + (2*(-std::pow(omega - omega0,2) + aq*std::pow(omega - omega0,2)*r2 + std::pow(r2,2) + aq*std::pow(r2,3))*std::cos(aq*(omega - omega0)))/(std::exp(aq*r2)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2)) + std::pow(aq,2)*(-(1/(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - 1/(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + 2*std::pow(aq,2)*std::pow(r2,2)*(std::pow(std::pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2),-2) + std::pow(std::pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + std::pow(aq,2)*std::pow(r2,2),-2)))*std::cos(2*off*PI) - (2*std::pow(aq,4)*std::pow(r2,2)*std::cos(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::pow(aq,2)*std::cos(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (std::pow(aq,3)*r2*std::cos(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (2*std::pow(aq,4)*std::pow(r2,2)*std::cos(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::pow(aq,2)*std::cos(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (std::pow(aq,3)*r2*std::cos(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (4*omega*r2*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2)) + (4*omega0*r2*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2)) - (2*aq*omega*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2))) + (2*aq*omega0*std::sin(aq*(omega - omega0)))/(std::exp(aq*r2)*(std::pow(omega - omega0,2) + std::pow(r2,2))) - (2*std::pow(aq,4)*omega*r2*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (2*std::pow(aq,4)*omega0*r2*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) - (4*std::pow(aq,3)*end*PI*r2*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (4*std::pow(aq,3)*off*PI*r2*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (2*std::pow(aq,4)*omega*r2*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) - (2*std::pow(aq,4)*omega0*r2*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) - (4*std::pow(aq,3)*end*PI*r2*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (4*std::pow(aq,3)*off*PI*r2*std::sin(2*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (2*std::pow(aq,4)*omega*r2*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (2*std::pow(aq,4)*omega0*r2*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (4*std::pow(aq,3)*end*PI*r2*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (4*std::pow(aq,3)*off*PI*r2*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::pow(aq,3)*omega*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (std::pow(aq,3)*omega0*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (2*std::pow(aq,2)*end*PI*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,2)*off*PI*std::sin(aq*omega - aq*omega0 - 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,4)*omega*r2*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (2*std::pow(aq,4)*omega0*r2*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (4*std::pow(aq,3)*end*PI*r2*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (4*std::pow(aq,3)*off*PI*r2*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::pow(aq,3)*omega*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (std::pow(aq,3)*omega0*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,2)*end*PI*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (2*std::pow(aq,2)*off*PI*std::sin(aq*omega - aq*omega0 + 2*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))))/4.;
}

double sp3_singlet_value_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    return (aq*(-((aq*omega*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)))) + (aq*omega0*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*end*PI*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*off*PI*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*aq*r2*std::sin(off*PI))/(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (3*aq*r2*std::sin(off*PI))/(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (3*aq*r2*std::cos(aq*omega - aq*omega0 + end*PI - off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*aq*r2*std::cos(aq*omega - aq*omega0 - end*PI + off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (aq*r2*std::sin(3*off*PI))/(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (aq*r2*std::sin(3*off*PI))/(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (aq*r2*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (aq*r2*std::sin(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*aq*omega0*std::sin(off*PI)*std::sin(aq*(-omega + omega0) + (end - off)*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*end*PI*std::sin(off*PI)*std::sin(aq*(-omega + omega0) + (end - off)*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (aq*omega*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (aq*omega0*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*end*PI*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*off*PI*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (std::cos(3*off*PI)*((6*(end - off)*PI*(-9*std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) - std::pow(r2,2))))/(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + ((aq*(omega - omega0) + 3*(end - off)*PI)*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + aq*r2*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/std::exp(aq*r2)))/(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (3*aq*omega*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*aq*omega0*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*end*PI*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*off*PI*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*aq*omega*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*off*PI*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + 3*std::cos(off*PI)*(((aq*(-omega + omega0) + (-end + off)*PI)*std::cos(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (((aq*(omega - omega0) + (-end + off)*PI)*std::cos(aq*omega - aq*omega0 - end*PI + off*PI))/std::exp(aq*r2) + (2*(end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2))) - (aq*r2*((std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))*std::sin(aq*omega - aq*omega0 + end*PI - off*PI) - (std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))*std::sin(aq*omega - aq*omega0 - end*PI + off*PI)))/std::exp(aq*r2))/(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)))/(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)))))/8.;
}

double sp3_singlet_d_omega0_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    return (std::pow(aq,2)*((2*aq*omega*(aq*(-omega + omega0) + 3*(end - off)*PI)*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*off*PI*(aq*(-omega + omega0) + 3*(end - off)*PI)*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (2*aq*omega0*(aq*(omega - omega0) + 3*(-end + off)*PI)*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*end*PI*(aq*(omega - omega0) + 3*(-end + off)*PI)*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + std::cos(aq*(-omega + omega0) + 3*end*PI)/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (aq*r2*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (6*aq*(aq*(omega - omega0) + (end - off)*PI)*r2*std::sin(off*PI))/std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (6*aq*(aq*(omega - omega0) + (-end + off)*PI)*r2*std::sin(off*PI))/std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (6*aq*(aq*(-omega + omega0) + (-end + off)*PI)*r2*std::cos(aq*omega - aq*omega0 + end*PI - off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (3*aq*omega*std::cos(aq*omega - aq*omega0 + end*PI - off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*aq*omega0*std::cos(aq*omega - aq*omega0 + end*PI - off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*end*PI*std::cos(aq*omega - aq*omega0 + end*PI - off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*off*PI*std::cos(aq*omega - aq*omega0 + end*PI - off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (6*aq*(aq*(-omega + omega0) + (end - off)*PI)*r2*std::cos(aq*omega - aq*omega0 - end*PI + off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (3*aq*omega*std::cos(aq*omega - aq*omega0 - end*PI + off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*aq*omega0*std::cos(aq*omega - aq*omega0 - end*PI + off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*end*PI*std::cos(aq*omega - aq*omega0 - end*PI + off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*off*PI*std::cos(aq*omega - aq*omega0 - end*PI + off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*aq*(aq*(-omega + omega0) + 3*(-end + off)*PI)*r2*std::sin(3*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (2*aq*(aq*(-omega + omega0) + 3*(end - off)*PI)*r2*std::sin(3*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (2*aq*(aq*(omega - omega0) + 3*(end - off)*PI)*r2*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (aq*omega*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (aq*omega0*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*end*PI*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*off*PI*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (aq*omega0*std::sin(aq*omega - aq*omega0 - 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*end*PI*std::sin(aq*omega - aq*omega0 - 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*aq*(aq*(omega - omega0) + 3*(-end + off)*PI)*r2*std::sin(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (aq*omega*std::sin(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*off*PI*std::sin(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (6*aq*omega0*(aq*(omega - omega0) + (-end + off)*PI)*std::sin(off*PI)*std::sin(aq*(-omega + omega0) + (end - off)*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*end*PI*(aq*(omega - omega0) + (-end + off)*PI)*std::sin(off*PI)*std::sin(aq*(-omega + omega0) + (end - off)*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (3*std::sin(off*PI)*std::sin(aq*(-omega + omega0) + (end - off)*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*aq*omega0*(aq*(omega - omega0) + 3*(end - off)*PI)*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*off*PI*(aq*(omega - omega0) + 3*(end - off)*PI)*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (2*aq*omega*(aq*(-omega + omega0) + 3*(-end + off)*PI)*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*end*PI*(aq*(-omega + omega0) + 3*(-end + off)*PI)*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (aq*r2*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (std::cos(3*off*PI)*((12*std::exp(aq*r2)*(end - off)*PI*(3*(end - off)*PI*std::pow(aq*omega - aq*omega0 - 3*end*PI + 3*off*PI,2) + std::pow(aq,2)*(-2*aq*omega + 2*aq*omega0 + 3*end*PI - 3*off*PI)*std::pow(r2,2)))/std::pow(std::pow(aq*omega - aq*omega0 - 3*end*PI + 3*off*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) - (1 + aq*r2)*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + (aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*(aq*(omega - omega0) + 3*(end - off)*PI)*std::cos(3*off*PI)*((6*(end - off)*PI*(-9*std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) - std::pow(r2,2))))/(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + ((aq*(omega - omega0) + 3*(end - off)*PI)*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + aq*r2*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/std::exp(aq*r2)))/std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (6*aq*omega*(aq*(omega - omega0) + (end - off)*PI)*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*end*PI*(aq*(omega - omega0) + (end - off)*PI)*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*aq*omega0*(aq*(-omega + omega0) + (-end + off)*PI)*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*off*PI*(aq*(-omega + omega0) + (-end + off)*PI)*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (3*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*aq*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (6*aq*omega*(aq*(omega - omega0) + (-end + off)*PI)*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*off*PI*(aq*(omega - omega0) + (-end + off)*PI)*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (3*aq*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + 3*std::cos(off*PI)*((-2*std::pow(aq*(omega - omega0) + (end - off)*PI,2)*std::cos(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + std::cos(aq*omega - aq*omega0 + end*PI - off*PI)/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + ((aq*(-omega + omega0) + (-end + off)*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (4*std::exp(aq*r2)*(end - off)*PI*((end - off)*PI*std::pow(aq*omega - aq*omega0 + end*PI - off*PI,2) + std::pow(aq,2)*(2*aq*omega - 2*aq*omega0 + end*PI - off*PI)*std::pow(r2,2)) + aq*r2*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))*std::cos(aq*omega - aq*omega0 + end*PI - off*PI) - (1 + aq*r2)*std::pow(std::pow(aq*omega - aq*omega0 + end*PI - off*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)*std::cos(aq*omega - aq*omega0 - end*PI + off*PI) - 4*aq*(end - off)*PI*r2*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2)))*std::sin(aq*omega - aq*omega0 + end*PI - off*PI) + (aq*omega - aq*omega0 - end*PI + off*PI)*std::pow(std::pow(aq*omega - aq*omega0 + end*PI - off*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*(aq*(omega - omega0) + (-end + off)*PI)*(((aq*(omega - omega0) + (-end + off)*PI)*std::cos(aq*omega - aq*omega0 - end*PI + off*PI))/std::exp(aq*r2) + (2*(end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2))) - (aq*r2*((std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))*std::sin(aq*omega - aq*omega0 + end*PI - off*PI) - (std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))*std::sin(aq*omega - aq*omega0 - end*PI + off*PI)))/std::exp(aq*r2))/(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))))/std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2))))/8.;
}

double sp3_singlet_d_r2_direct(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    return (std::pow(aq,2)*((2*std::pow(aq,2)*omega*r2*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (2*std::pow(aq,2)*omega0*r2*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (6*aq*end*PI*r2*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*aq*off*PI*r2*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (aq*omega*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (aq*omega0*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*end*PI*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*off*PI*std::cos(aq*(-omega + omega0) + 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (6*std::pow(aq,2)*std::pow(r2,2)*std::sin(off*PI))/std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (3*std::sin(off*PI))/(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (6*std::pow(aq,2)*std::pow(r2,2)*std::sin(off*PI))/std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (3*std::sin(off*PI))/(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (6*std::pow(aq,2)*std::pow(r2,2)*std::cos(aq*omega - aq*omega0 + end*PI - off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (3*std::cos(aq*omega - aq*omega0 + end*PI - off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*aq*r2*std::cos(aq*omega - aq*omega0 + end*PI - off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (6*std::pow(aq,2)*std::pow(r2,2)*std::cos(aq*omega - aq*omega0 - end*PI + off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (3*std::cos(aq*omega - aq*omega0 - end*PI + off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*aq*r2*std::cos(aq*omega - aq*omega0 - end*PI + off*PI)*std::sin(off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,2)*std::pow(r2,2)*std::sin(3*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) - std::sin(3*off*PI)/(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + (2*std::pow(aq,2)*std::pow(r2,2)*std::sin(3*off*PI))/std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) - std::sin(3*off*PI)/(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (2*std::pow(aq,2)*std::pow(r2,2)*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (aq*r2*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*std::sin(3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,2)*std::pow(r2,2)*std::sin(aq*omega - aq*omega0 - 3*end*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (aq*r2*std::sin(aq*omega - aq*omega0 - 3*end*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + std::sin(aq*(-omega + omega0) + 3*end*PI)/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (2*std::pow(aq,2)*omega*r2*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (2*std::pow(aq,2)*omega0*r2*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*aq*end*PI*r2*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (6*aq*off*PI*r2*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (aq*omega*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (aq*omega0*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*end*PI*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*off*PI*std::sin(3*off*PI)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (2*aq*r2*std::cos(3*off*PI)*((6*(end - off)*PI*(-9*std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(std::pow(omega - omega0,2) - std::pow(r2,2))))/(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) + ((aq*(omega - omega0) + 3*(end - off)*PI)*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + aq*r2*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/std::exp(aq*r2)))/std::pow(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (std::cos(3*off*PI)*((24*std::pow(aq,2)*(end - off)*(omega - omega0)*PI*(aq*(-omega + omega0) + 3*(end - off)*PI)*r2)/std::pow(std::pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + ((aq*(-omega + omega0) + 3*(-end + off)*PI)*std::cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + (1 - aq*r2)*std::sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/std::exp(aq*r2)))/(std::pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2)) - (6*std::pow(aq,2)*omega*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*std::pow(aq,2)*omega0*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (6*aq*end*PI*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*aq*off*PI*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (3*aq*omega*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*aq*omega0*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*end*PI*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*off*PI*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (6*std::pow(aq,2)*omega*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*std::pow(aq,2)*omega0*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + (6*aq*end*PI*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (6*aq*off*PI*r2*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) - (3*aq*omega*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*aq*omega0*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + (3*end*PI*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (3*off*PI*std::sin(off*PI)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + 3*std::cos(off*PI)*((2*aq*(aq*(omega - omega0) + (end - off)*PI)*r2*std::cos(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2)) + ((aq*(omega - omega0) + (end - off)*PI)*std::cos(aq*omega - aq*omega0 + end*PI - off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) + ((8*std::pow(aq,2)*std::exp(aq*r2)*(end - off)*(omega - omega0)*PI*(aq*(omega - omega0) + (end - off)*PI)*r2)/std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (aq*(-omega + omega0) + (end - off)*PI)*std::cos(aq*omega - aq*omega0 - end*PI + off*PI) + ((-(std::pow(end - off,4)*std::pow(PI,4)) + aq*std::pow(end - off,4)*std::pow(PI,4)*r2 + 2*std::pow(aq,2)*std::pow(end - off,2)*std::pow(PI,2)*(std::pow(omega - omega0,2) - std::pow(r2,2)) - std::pow(aq,4)*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2) + std::pow(aq,5)*r2*std::pow(std::pow(omega - omega0,2) + std::pow(r2,2),2) + 2*std::pow(aq,3)*(end - off)*PI*r2*(-((end - off)*std::pow(omega - omega0,2)*PI) + 4*(-omega + omega0)*r2 + (end - off)*PI*std::pow(r2,2)))*std::sin(aq*omega - aq*omega0 + end*PI - off*PI))/std::pow(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2) + (1 - aq*r2)*std::sin(aq*omega - aq*omega0 - end*PI + off*PI))/(std::exp(aq*r2)*(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))) - (2*aq*r2*(((aq*(omega - omega0) + (-end + off)*PI)*std::cos(aq*omega - aq*omega0 - end*PI + off*PI))/std::exp(aq*r2) + (2*(end - off)*PI*(std::pow(end - off,2)*std::pow(PI,2) + std::pow(aq,2)*(-std::pow(omega - omega0,2) + std::pow(r2,2))) - (aq*r2*((std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))*std::sin(aq*omega - aq*omega0 + end*PI - off*PI) - (std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))*std::sin(aq*omega - aq*omega0 - end*PI + off*PI)))/std::exp(aq*r2))/(std::pow(aq*(omega - omega0) + (end - off)*PI,2) + std::pow(aq,2)*std::pow(r2,2))))/std::pow(std::pow(aq*(omega - omega0) + (-end + off)*PI,2) + std::pow(aq,2)*std::pow(r2,2),2))))/8.;
}

double singlet_value_direct(
    const double n,
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    if (n > 2.5)
        return sp3_singlet_value_direct(omega, omega0, r2, aq, off, end);
    if (n > 1.5)
        return sp2_singlet_value_direct(omega, omega0, r2, aq, off, end);
    if (n > 0.5)
        return sp1_singlet_value_direct(omega, omega0, r2, aq, off, end);
    return none_singlet_value_direct(omega, omega0, r2, aq);
}

double singlet_d_omega0_direct(
    const double n,
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    if (n > 2.5)
        return sp3_singlet_d_omega0_direct(omega, omega0, r2, aq, off, end);
    if (n > 1.5)
        return sp2_singlet_d_omega0_direct(omega, omega0, r2, aq, off, end);
    if (n > 0.5)
        return sp1_singlet_d_omega0_direct(omega, omega0, r2, aq, off, end);
    return none_singlet_d_omega0_direct(omega, omega0, r2, aq);
}

double singlet_d_r2_direct(
    const double n,
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end)
{
    if (n > 2.5)
        return sp3_singlet_d_r2_direct(omega, omega0, r2, aq, off, end);
    if (n > 1.5)
        return sp2_singlet_d_r2_direct(omega, omega0, r2, aq, off, end);
    if (n > 0.5)
        return sp1_singlet_d_r2_direct(omega, omega0, r2, aq, off, end);
    return none_singlet_d_r2_direct(omega, omega0, r2, aq);
}

void build_expected_doublet_singlet_outputs(
    std::vector<double> &residuals,
    std::vector<std::vector<double> > &jac_storage,
    const int xdim,
    const int ydim,
    const std::vector<double> &z,
    const double a,
    const double x0,
    const double r2x,
    const double j,
    const double y0,
    const double r2y,
    const double aqx,
    const double offx,
    const double endx,
    const double nx,
    const double aqy,
    const double offy,
    const double endy,
    const double ny)
{
    for (int i = 0; i < xdim; i++)
    {
        const double xl = singlet_value_direct(nx, i, x0 - 0.5 * j, r2x, aqx, offx, endx);
        const double xr = singlet_value_direct(nx, i, x0 + 0.5 * j, r2x, aqx, offx, endx);
        const double dxl = singlet_d_omega0_direct(nx, i, x0 - 0.5 * j, r2x, aqx, offx, endx);
        const double dxr = singlet_d_omega0_direct(nx, i, x0 + 0.5 * j, r2x, aqx, offx, endx);
        const double drl = singlet_d_r2_direct(nx, i, x0 - 0.5 * j, r2x, aqx, offx, endx);
        const double drr = singlet_d_r2_direct(nx, i, x0 + 0.5 * j, r2x, aqx, offx, endx);

        const double vx = 0.5 * (xl + xr);
        const double dx = 0.5 * (dxl + dxr);
        const double drx = 0.5 * (drl + drr);
        const double dj = 0.25 * (dxr - dxl);

        for (int j2 = 0; j2 < ydim; j2++)
        {
            const int ii = i * ydim + j2;
            const double vy = singlet_value_direct(ny, j2, y0, r2y, aqy, offy, endy);
            const double dy = singlet_d_omega0_direct(ny, j2, y0, r2y, aqy, offy, endy);
            const double dry = singlet_d_r2_direct(ny, j2, y0, r2y, aqy, offy, endy);

            residuals[ii] = a * vx * vy - z[ii];
            jac_storage[0][ii] = vx * vy;
            jac_storage[1][ii] = a * dx * vy;
            jac_storage[2][ii] = a * drx * vy;
            jac_storage[3][ii] = a * dj * vy;
            jac_storage[4][ii] = a * vx * dy;
            jac_storage[5][ii] = a * vx * dry;
        }
    }
}

void build_expected_singlet_singlet_outputs(
    std::vector<double> &residuals,
    std::vector<std::vector<double> > &jac_storage,
    const int xdim,
    const int ydim,
    const std::vector<double> &z,
    const double a,
    const double x0,
    const double r2x,
    const double y0,
    const double r2y,
    const double aqx,
    const double offx,
    const double endx,
    const double nx,
    const double aqy,
    const double offy,
    const double endy,
    const double ny)
{
    for (int i = 0; i < xdim; i++)
    {
        const double vx = singlet_value_direct(nx, i, x0, r2x, aqx, offx, endx);
        const double dx = singlet_d_omega0_direct(nx, i, x0, r2x, aqx, offx, endx);
        const double drx = singlet_d_r2_direct(nx, i, x0, r2x, aqx, offx, endx);

        for (int j2 = 0; j2 < ydim; j2++)
        {
            const int ii = i * ydim + j2;
            const double vy = singlet_value_direct(ny, j2, y0, r2y, aqy, offy, endy);
            const double dy = singlet_d_omega0_direct(ny, j2, y0, r2y, aqy, offy, endy);
            const double dry = singlet_d_r2_direct(ny, j2, y0, r2y, aqy, offy, endy);

            residuals[ii] = a * vx * vy - z[ii];
            jac_storage[0][ii] = vx * vy;
            jac_storage[1][ii] = a * dx * vy;
            jac_storage[2][ii] = a * drx * vy;
            jac_storage[3][ii] = a * vx * dy;
            jac_storage[4][ii] = a * vx * dry;
        }
    }
}

bool check_formula_jacobians(
    const std::string &label,
    const std::vector<double> &residuals,
    const std::vector<std::vector<double> > &jac_storage,
    const std::vector<double> &expected_residuals,
    const std::vector<std::vector<double> > &expected_jac,
    const char *const *param_names,
    const int n_param)
{
    bool ok = true;

    const double residual_formula_err = max_abs_diff(residuals, expected_residuals);
    std::cout << label
              << " residual vs direct-formula max error = "
              << residual_formula_err << std::endl;
    if (residual_formula_err > 1e-12)
        ok = false;

    for (int ip = 0; ip < n_param; ip++)
    {
        const double formula_err = max_abs_diff(jac_storage[ip], expected_jac[ip]);
        std::cout << label << " " << param_names[ip]
                  << " jacobian vs direct-formula max error = "
                  << formula_err << std::endl;
        if (formula_err > 1e-12)
            ok = false;
    }

    return ok;
}

template <typename evaluator_t>
bool check_fd_jacobians(
    const std::string &label,
    const std::vector<std::vector<double> > &jac_storage,
    const char *const *param_names,
    const int n_param,
    evaluator_t evaluate_for_fd)
{
    const double delta = 1e-7;
    bool ok = true;

    for (int ip = 0; ip < n_param; ip++)
    {
        const std::vector<double> residuals_plus = evaluate_for_fd(ip, delta);
        const std::vector<double> residuals_minus = evaluate_for_fd(ip, -delta);

        std::vector<double> fd(jac_storage[ip].size(), 0.0);
        std::vector<double> neg_analytic(jac_storage[ip].size(), 0.0);
        for (size_t k = 0; k < fd.size(); k++)
        {
            fd[k] = (residuals_plus[k] - residuals_minus[k]) / (2.0 * delta);
            neg_analytic[k] = -jac_storage[ip][k];
        }

        const double err = max_abs_diff(jac_storage[ip], fd);
        const double neg_err = max_abs_diff(neg_analytic, fd);
        std::cout << label << " " << param_names[ip]
                  << " max |analytic - finite-diff| = "
                  << err
                  << " ; max |-analytic - finite-diff| = "
                  << neg_err
                  << " ; max |analytic| = "
                  << max_abs_value(jac_storage[ip])
                  << " ; max |fd| = "
                  << max_abs_value(fd)
                  << std::endl;

        if (err > 1e-4)
            ok = false;
    }

    return ok;
}

int test_fitnmr_doublet_singlet()
{
    int return_code = 0;

    const int xdim = 5;
    const int ydim = 4;
    std::vector<double> z(xdim * ydim, 0.0);

    const double aqx = 0.12;
    const double offx = 0.5;
    const double endx = 1.0;
    const double aqy = 0.10;
    const double offy = 0.5;
    const double endy = 1.0;

    const double a = 1.3;
    const double x0 = 2.1;
    const double r2x = 0.35;
    const double j = 0.4;
    const double y0 = 1.4;
    const double r2y = 0.28;

    const double n_cases[16][2] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {2.0, 0.0},
        {3.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {2.0, 1.0},
        {3.0, 1.0},
        {0.0, 2.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 2.0},
        {0.0, 3.0},
        {1.0, 3.0},
        {2.0, 3.0},
        {3.0, 3.0}};

    for (int icase = 0; icase < 16; icase++)
    {
        const double nx = n_cases[icase][0];
        const double ny = n_cases[icase][1];
        const std::string label = "doublet case nx=" + std::to_string(nx) + ", ny=" + std::to_string(ny);

        mycostfunction_fitnmr_doublet_singlet cost_function(
            xdim, ydim, const_cast<double *>(z.data()),
            aqx, offx, endx, nx,
            aqy, offy, endy, ny);

#ifndef LMMIN
        cost_function.set_n_residuals(static_cast<int>(z.size()));
        for (int i = 0; i < 6; i++)
            cost_function.parameter_block_sizes()->push_back(1);
#endif

        double a_case = a;
        double x0_case = x0;
        double r2x_case = r2x;
        double j_case = j;
        double y0_case = y0;
        double r2y_case = r2y;

        const double *params[] = {&a_case, &x0_case, &r2x_case, &j_case, &y0_case, &r2y_case};
        std::vector<double> residuals(z.size(), 0.0);
        std::vector<std::vector<double> > jac_storage(6, std::vector<double>(z.size(), 0.0));
        double *jacobian[6];
        for (int i = 0; i < 6; i++)
            jacobian[i] = jac_storage[i].data();

        cost_function.Evaluate(params, residuals.data(), jacobian);

        std::vector<double> expected_residuals(z.size(), 0.0);
        std::vector<std::vector<double> > expected_jac(6, std::vector<double>(z.size(), 0.0));
        build_expected_doublet_singlet_outputs(
            expected_residuals, expected_jac,
            xdim, ydim, z,
            a_case, x0_case, r2x_case, j_case, y0_case, r2y_case,
            aqx, offx, endx, nx,
            aqy, offy, endy, ny);

        const char *param_names[6] = {"a", "x0", "r2x", "j", "y0", "r2y"};
        if (!check_formula_jacobians(label, residuals, jac_storage, expected_residuals, expected_jac, param_names, 6))
            return_code = 1;

        if (!check_fd_jacobians(label, jac_storage, param_names, 6,
                [&](const int ip, const double step) {
                    double a_fd = a_case;
                    double x0_fd = x0_case;
                    double r2x_fd = r2x_case;
                    double j_fd = j_case;
                    double y0_fd = y0_case;
                    double r2y_fd = r2y_case;

                    double *mutable_params[] = {&a_fd, &x0_fd, &r2x_fd, &j_fd, &y0_fd, &r2y_fd};
                    *mutable_params[ip] += step;

                    const double *eval_params[] = {&a_fd, &x0_fd, &r2x_fd, &j_fd, &y0_fd, &r2y_fd};
                    std::vector<double> fd_residuals(xdim * ydim, 0.0);
                    cost_function.Evaluate(eval_params, fd_residuals.data(), NULL);
                    return fd_residuals;
                }))
            return_code = 1;
    }

    std::cout << "Test fitnmr doublet-singlet done." << std::endl;
    return return_code;
}

int test_fitnmr_singlet_singlet()
{
    int return_code = 0;

    const int xdim = 5;
    const int ydim = 4;
    std::vector<double> z(xdim * ydim, 0.0);

    const double aqx = 0.12;
    const double offx = 0.5;
    const double endx = 1.0;
    const double aqy = 0.10;
    const double offy = 0.5;
    const double endy = 1.0;

    const double a = 1.3;
    const double x0 = 2.1;
    const double r2x = 0.35;
    const double y0 = 1.4;
    const double r2y = 0.28;

    const double n_cases[16][2] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {2.0, 0.0},
        {3.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0},
        {2.0, 1.0},
        {3.0, 1.0},
        {0.0, 2.0},
        {1.0, 2.0},
        {2.0, 2.0},
        {3.0, 2.0},
        {0.0, 3.0},
        {1.0, 3.0},
        {2.0, 3.0},
        {3.0, 3.0}};

    for (int icase = 0; icase < 16; icase++)
    {
        const double nx = n_cases[icase][0];
        const double ny = n_cases[icase][1];
        const std::string label = "singlet case nx=" + std::to_string(nx) + ", ny=" + std::to_string(ny);

        mycostfunction_fitnmr_singlet_singlet cost_function(
            xdim, ydim, const_cast<double *>(z.data()),
            aqx, offx, endx, nx,
            aqy, offy, endy, ny);

#ifndef LMMIN
        cost_function.set_n_residuals(static_cast<int>(z.size()));
        for (int i = 0; i < 5; i++)
            cost_function.parameter_block_sizes()->push_back(1);
#endif

        double a_case = a;
        double x0_case = x0;
        double r2x_case = r2x;
        double y0_case = y0;
        double r2y_case = r2y;

        const double *params[] = {&a_case, &x0_case, &r2x_case, &y0_case, &r2y_case};
        std::vector<double> residuals(z.size(), 0.0);
        std::vector<std::vector<double> > jac_storage(5, std::vector<double>(z.size(), 0.0));
        double *jacobian[5];
        for (int i = 0; i < 5; i++)
            jacobian[i] = jac_storage[i].data();

        cost_function.Evaluate(params, residuals.data(), jacobian);

        std::vector<double> expected_residuals(z.size(), 0.0);
        std::vector<std::vector<double> > expected_jac(5, std::vector<double>(z.size(), 0.0));
        build_expected_singlet_singlet_outputs(
            expected_residuals, expected_jac,
            xdim, ydim, z,
            a_case, x0_case, r2x_case, y0_case, r2y_case,
            aqx, offx, endx, nx,
            aqy, offy, endy, ny);

        const char *param_names[5] = {"a", "x0", "r2x", "y0", "r2y"};
        if (!check_formula_jacobians(label, residuals, jac_storage, expected_residuals, expected_jac, param_names, 5))
            return_code = 1;

        if (!check_fd_jacobians(label, jac_storage, param_names, 5,
                [&](const int ip, const double step) {
                    double a_fd = a_case;
                    double x0_fd = x0_case;
                    double r2x_fd = r2x_case;
                    double y0_fd = y0_case;
                    double r2y_fd = r2y_case;

                    double *mutable_params[] = {&a_fd, &x0_fd, &r2x_fd, &y0_fd, &r2y_fd};
                    *mutable_params[ip] += step;

                    const double *eval_params[] = {&a_fd, &x0_fd, &r2x_fd, &y0_fd, &r2y_fd};
                    std::vector<double> fd_residuals(xdim * ydim, 0.0);
                    cost_function.Evaluate(eval_params, fd_residuals.data(), NULL);
                    return fd_residuals;
                }))
            return_code = 1;
    }

    std::cout << "Test fitnmr singlet-singlet done." << std::endl;
    return return_code;
}
}

int main()
{
    {
        const double aqx = 0.12;
        const double offx = 0.5;
        const double endx = 1.0;
        const double omega_test = 2.0;
        const double omega0_test = 2.1;
        const double r2_test = 0.35;
        const double eps = 1e-7;

        const double fd_direct_omega0 =
            (sp1_singlet_value_direct(omega_test, omega0_test + eps, r2_test, aqx, offx, endx) -
             sp1_singlet_value_direct(omega_test, omega0_test - eps, r2_test, aqx, offx, endx)) / (2.0 * eps);

        const double fd_direct_r2 =
            (sp1_singlet_value_direct(omega_test, omega0_test, r2_test + eps, aqx, offx, endx) -
             sp1_singlet_value_direct(omega_test, omega0_test, r2_test - eps, aqx, offx, endx)) / (2.0 * eps);

        const double analytic_direct_omega0 =
            sp1_singlet_d_omega0_direct(omega_test, omega0_test, r2_test, aqx, offx, endx);
        const double analytic_direct_r2 =
            sp1_singlet_d_r2_direct(omega_test, omega0_test, r2_test, aqx, offx, endx);

        std::cout << "direct x-singlet d_omega0 analytic vs fd = "
                  << analytic_direct_omega0 << " vs " << fd_direct_omega0
                  << " ; abs err = " << std::fabs(analytic_direct_omega0 - fd_direct_omega0)
                  << std::endl;
        std::cout << "direct x-singlet d_r2 analytic vs fd = "
                  << analytic_direct_r2 << " vs " << fd_direct_r2
                  << " ; abs err = " << std::fabs(analytic_direct_r2 - fd_direct_r2)
                  << std::endl;
    }

    if (test_fitnmr_doublet_singlet() + test_fitnmr_singlet_singlet() != 0)
    {
        std::cout << "FitNMR review check failed." << std::endl;
        return 1;
    }

    std::cout << "FitNMR review check passed." << std::endl;
    return 0;
}
