#include <cmath>

#include "cost_functors_fitnmr.h"

#ifndef FITNMR_USE_SUBEXPRESSIONS
#define FITNMR_USE_SUBEXPRESSIONS 1
#endif

mycostfunction_fitnmr_base::~mycostfunction_fitnmr_base(){};

mycostfunction_fitnmr_base::mycostfunction_fitnmr_base(
    int xdim_, int ydim_, double *z_,
    double aqx_, double offx_, double endx_, double nx_,
    double aqy_, double offy_, double endy_, double ny_)
{
    xdim = xdim_;
    ydim = ydim_;
    n_datapoint = xdim * ydim;
    z = z_;

    nx = nx_;
    ny = ny_;
    aqx = aqx_;
    offx = offx_;
    endx = endx_;
    aqy = aqy_;
    offy = offy_;
    endy = endy_;
};

void mycostfunction_fitnmr_base::none_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    double *vv,
    double *r_omega0,
    double *r_r2) const
{
#if FITNMR_USE_SUBEXPRESSIONS
    const double dw = omega - omega0;
    const double dw2 = dw * dw;
    const double r22 = r2 * r2;
    const double den0 = dw2 + r22;
    const double e = std::exp(aq * r2);
    const double phase = aq * dw;
    const double s_phase = std::sin(phase);
    const double c_phase = std::cos(phase);
    const double common1 = 2.0 * r2 + aq * den0;
    const double common2 = -dw2 + aq * dw2 * r2 + r22 + aq * r22 * r2;

    *vv = (e * r2 - r2 * c_phase + dw * s_phase) / (e * den0);

    *r_omega0 = (2.0 * e * dw * r2 - dw * common1 * c_phase - common2 * s_phase) /
                (e * den0 * den0);

    *r_r2 = (-(e * (dw + r2) * (-dw + r2)) + common2 * c_phase - dw * common1 * s_phase) /
            (e * den0 * den0);
#else
    *vv = (exp(aq*r2)*r2 - r2*cos(aq*(omega - omega0)) + (omega - omega0)*sin(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2)));
    *r_omega0 = (2*exp(aq*r2)*(omega - omega0)*r2 - (omega - omega0)*(2*r2 + aq*(pow(omega - omega0,2) + pow(r2,2)))*cos(aq*(omega - omega0)) - (-pow(omega - omega0,2) + aq*pow(omega - omega0,2)*r2 + pow(r2,2) + aq*pow(r2,3))*sin(aq*(omega - omega0)))/(exp(aq*r2)*pow(pow(omega - omega0,2) + pow(r2,2),2));
    *r_r2 = (-(exp(aq*r2)*(omega - omega0 + r2)*(-omega + omega0 + r2)) + (-pow(omega - omega0,2) + aq*pow(omega - omega0,2)*r2 + pow(r2,2) + aq*pow(r2,3))*cos(aq*(omega - omega0)) - (omega - omega0)*(2*r2 + aq*(pow(omega - omega0,2) + pow(r2,2)))*sin(aq*(omega - omega0)))/(exp(aq*r2)*pow(pow(omega - omega0,2) + pow(r2,2),2));
#endif
}

void mycostfunction_fitnmr_base::none_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    double *vv) const
{
    double dummy_omega0, dummy_r2;
    none_helper(omega, omega0, r2, aq, vv, &dummy_omega0, &dummy_r2);
}

void mycostfunction_fitnmr_base::sp1_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end,
    double *vv,
    double *r_omega0,
    double *r_r2) const
{
#if FITNMR_USE_SUBEXPRESSIONS
    const double dw = omega - omega0;
    const double dw2 = dw * dw;

    const double c = end - off;
    const double cpi = c * PI;
    const double cpi2 = cpi * cpi;
    const double cpi4 = cpi2 * cpi2;

    const double aq2 = aq * aq;
    const double aq3 = aq2 * aq;
    const double aq4 = aq2 * aq2;

    const double g = r2;
    const double g2 = g * g;
    const double g3 = g2 * g;

    const double e = std::exp(aq * g);

    const double phase = aq * dw;
    const double s_phase = std::sin(phase);
    const double c_phase = std::cos(phase);

    const double s_off = std::sin(off * PI);
    const double c_off = std::cos(off * PI);
    const double s_end = std::sin(end * PI);
    const double c_end = std::cos(end * PI);

    const double sum_sq = dw2 + g2;
    const double mix_a = cpi2 + aq2 * (-dw2 + g2);
    const double mix_b = cpi2 + aq2 * sum_sq;
    const double den0 = cpi4 - 2.0 * aq2 * cpi2 * (dw2 - g2) + aq4 * sum_sq * sum_sq;

    const double core_n =
        e * cpi * mix_a * c_off
        + aq * dw * s_phase * (2.0 * aq * cpi * g * c_end + (-cpi2 + aq2 * sum_sq) * s_end)
        + c_phase * (-cpi * mix_a * c_end - aq * g * mix_b * s_end)
        + aq * e * g * mix_b * s_off;

    *vv = aq * core_n / (e * den0);

    const double d_omega0_inner =
        -4.0 * aq * dw * (cpi2 - aq2 * sum_sq) * core_n
        + den0 * (s_phase * (-(cpi * (cpi2 + 2.0 * aq * g + aq2 * (-dw2 + g2)) * c_end)
                             - (-cpi2 + aq * cpi2 * g + aq3 * g * sum_sq + aq2 * (3.0 * dw2 + g2)) * s_end)
                  + aq * dw * c_phase * (-2.0 * cpi * (1.0 + aq * g) * c_end
                                         + (cpi2 + aq * (2.0 * g - aq * sum_sq)) * s_end)
                  - 2.0 * aq * e * dw * (-cpi * c_off + aq * g * s_off));

    *r_omega0 = aq2 * d_omega0_inner / (e * den0 * den0);

    const double dr2_inner =
        2.0 * aq * dw * s_phase * (cpi * c_end + aq * g * s_end)
        + c_phase * (2.0 * aq * (-c) * PI * g * c_end - (cpi2 + aq2 * (dw2 + 3.0 * g2)) * s_end)
        + e * (cpi * (cpi2 + 2.0 * aq * g + aq2 * (-dw2 + g2)) * c_off
               + (cpi2 + aq * cpi2 * g + aq3 * g * sum_sq + aq2 * (dw2 + 3.0 * g2)) * s_off);

    *r_r2 = aq2 * (-4.0 * aq * g * mix_b * core_n - den0 * core_n + den0 * dr2_inner)
            / (e * den0 * den0);
#else
    *vv = (aq*(exp(aq*r2)*(end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(off*PI) + aq*(omega - omega0)*sin(aq*(omega - omega0))*(2*aq*(end - off)*PI*r2*cos(end*PI) + (-(pow(end - off,2)*pow(PI,2)) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(end*PI)) + cos(aq*(omega - omega0))*(-((end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(end*PI)) - aq*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(end*PI)) + aq*exp(aq*r2)*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(off*PI)))/(exp(aq*r2)*(pow(end - off,4)*pow(PI,4) - 2*pow(aq,2)*pow(end - off,2)*pow(PI,2)*(pow(omega - omega0,2) - pow(r2,2)) + pow(aq,4)*pow(pow(omega - omega0,2) + pow(r2,2),2)));
    *r_omega0 = (pow(aq,2)*(-4*aq*(omega - omega0)*(pow(end - off,2)*pow(PI,2) - pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*(exp(aq*r2)*(end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(off*PI) + aq*(omega - omega0)*sin(aq*(omega - omega0))*(2*aq*(end - off)*PI*r2*cos(end*PI) + (-(pow(end - off,2)*pow(PI,2)) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(end*PI)) + cos(aq*(omega - omega0))*(-((end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(end*PI)) - aq*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(end*PI)) + aq*exp(aq*r2)*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(off*PI)) + (pow(end - off,4)*pow(PI,4) - 2*pow(aq,2)*pow(end - off,2)*pow(PI,2)*(pow(omega - omega0,2) - pow(r2,2)) + pow(aq,4)*pow(pow(omega - omega0,2) + pow(r2,2),2))*(sin(aq*(omega - omega0))*(-((end - off)*PI*(pow(end - off,2)*pow(PI,2) + 2*aq*r2 + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(end*PI)) - (-(pow(end - off,2)*pow(PI,2)) + aq*pow(end - off,2)*pow(PI,2)*r2 + pow(aq,3)*r2*(pow(omega - omega0,2) + pow(r2,2)) + pow(aq,2)*(3*pow(omega - omega0,2) + pow(r2,2)))*sin(end*PI)) + aq*(omega - omega0)*cos(aq*(omega - omega0))*(-2*(end - off)*PI*(1 + aq*r2)*cos(end*PI) + (pow(end - off,2)*pow(PI,2) + aq*(2*r2 - aq*(pow(omega - omega0,2) + pow(r2,2))))*sin(end*PI)) - 2*aq*exp(aq*r2)*(omega - omega0)*((-end + off)*PI*cos(off*PI) + aq*r2*sin(off*PI)))))/(exp(aq*r2)*pow(pow(end - off,4)*pow(PI,4) - 2*pow(aq,2)*pow(end - off,2)*pow(PI,2)*(pow(omega - omega0,2) - pow(r2,2)) + pow(aq,4)*pow(pow(omega - omega0,2) + pow(r2,2),2),2));
    *r_r2 = (pow(aq,2)*(-4*aq*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*(exp(aq*r2)*(end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(off*PI) + aq*(omega - omega0)*sin(aq*(omega - omega0))*(2*aq*(end - off)*PI*r2*cos(end*PI) + (-(pow(end - off,2)*pow(PI,2)) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(end*PI)) + cos(aq*(omega - omega0))*(-((end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(end*PI)) - aq*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(end*PI)) + aq*exp(aq*r2)*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(off*PI)) - (pow(end - off,4)*pow(PI,4) - 2*pow(aq,2)*pow(end - off,2)*pow(PI,2)*(pow(omega - omega0,2) - pow(r2,2)) + pow(aq,4)*pow(pow(omega - omega0,2) + pow(r2,2),2))*(exp(aq*r2)*(end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(off*PI) + aq*(omega - omega0)*sin(aq*(omega - omega0))*(2*aq*(end - off)*PI*r2*cos(end*PI) + (-(pow(end - off,2)*pow(PI,2)) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(end*PI)) + cos(aq*(omega - omega0))*(-((end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(end*PI)) - aq*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(end*PI)) + aq*exp(aq*r2)*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + pow(r2,2)))*sin(off*PI)) + (pow(end - off,4)*pow(PI,4) - 2*pow(aq,2)*pow(end - off,2)*pow(PI,2)*(pow(omega - omega0,2) - pow(r2,2)) + pow(aq,4)*pow(pow(omega - omega0,2) + pow(r2,2),2))*(2*aq*(omega - omega0)*sin(aq*(omega - omega0))*((end - off)*PI*cos(end*PI) + aq*r2*sin(end*PI)) + cos(aq*(omega - omega0))*(2*aq*(-end + off)*PI*r2*cos(end*PI) - (pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) + 3*pow(r2,2)))*sin(end*PI)) + exp(aq*r2)*((end - off)*PI*(pow(end - off,2)*pow(PI,2) + 2*aq*r2 + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*cos(off*PI) + (pow(end - off,2)*pow(PI,2) + aq*pow(end - off,2)*pow(PI,2)*r2 + pow(aq,3)*r2*(pow(omega - omega0,2) + pow(r2,2)) + pow(aq,2)*(pow(omega - omega0,2) + 3*pow(r2,2)))*sin(off*PI)))))/(exp(aq*r2)*pow(pow(end - off,4)*pow(PI,4) - 2*pow(aq,2)*pow(end - off,2)*pow(PI,2)*(pow(omega - omega0,2) - pow(r2,2)) + pow(aq,4)*pow(pow(omega - omega0,2) + pow(r2,2),2),2));
#endif
}

void mycostfunction_fitnmr_base::sp1_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end,
    double *vv) const
{
    double dummy_omega0, dummy_r2;
    sp1_helper(omega, omega0, r2, aq, off, end, vv, &dummy_omega0, &dummy_r2);
}

void mycostfunction_fitnmr_base::sp2_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end,
    double *vv,
    double *r_omega0,
    double *r_r2) const
{
#if FITNMR_USE_SUBEXPRESSIONS
    const double dw = omega - omega0;
    const double dw2 = dw * dw;
    const double g = r2;
    const double g2 = g * g;
    const double g3 = g2 * g;

    const double aq2 = aq * aq;
    const double aq3 = aq2 * aq;
    const double aq4 = aq2 * aq2;

    const double e = std::exp(aq * g);
    const double inv_e = 1.0 / e;

    const double phase = aq * dw;
    const double c_phase = std::cos(phase);
    const double s_phase = std::sin(phase);

    const double two_off = 2.0 * off * PI;
    const double shift = 2.0 * (end - off) * PI;
    const double phase_m = phase - 2.0 * end * PI;
    const double phase_p = phase + 2.0 * end * PI;

    const double c2off = std::cos(two_off);
    const double s2off = std::sin(two_off);
    const double c_m = std::cos(phase_m);
    const double c_p = std::cos(phase_p);
    const double s_m = std::sin(phase_m);
    const double s_p = std::sin(phase_p);

    const double delta_m = phase - shift;
    const double delta_p = phase + shift;

    const double den0 = dw2 + g2;
    const double denm = delta_m * delta_m + aq2 * g2;
    const double denp = delta_p * delta_p + aq2 * g2;

    const double inv_den0 = 1.0 / den0;
    const double inv_den0_sq = inv_den0 * inv_den0;
    const double inv_denm = 1.0 / denm;
    const double inv_denp = 1.0 / denp;
    const double inv_denm_sq = inv_denm * inv_denm;
    const double inv_denp_sq = inv_denp * inv_denp;

    *vv = 0.25 * (2.0 * g * inv_den0
                  - 2.0 * g * c_phase * inv_e * inv_den0
                  + aq2 * g * (-inv_denp - inv_denm) * c2off
                  + aq2 * g * c_m * inv_e * inv_denm
                  + aq2 * g * c_p * inv_e * inv_denp
                  + 2.0 * dw * s_phase * inv_e * inv_den0
                  + aq * s2off * (delta_p * inv_denp - delta_m * inv_denm)
                  - aq * inv_e * (delta_m * s_m * inv_denm + delta_p * s_p * inv_denp));

    *r_omega0 = ((4.0 * dw * g) / (den0 * den0)
                 + (4.0 * (-dw) * g * c_phase) / (e * den0 * den0)
                 - (2.0 * aq * omega * c_phase) / (e * den0)
                 + (2.0 * aq * omega0 * c_phase) / (e * den0)
                 + 2.0 * aq3 * g * ((-delta_p) / (denp * denp) + (-delta_m) / (denm * denm)) * c2off
                 + (2.0 * aq3 * delta_m * g * c_m) / (e * denm * denm)
                 + (aq3 * omega * c_m) / (e * denm)
                 - (aq3 * omega0 * c_m) / (e * denm)
                 - (2.0 * aq2 * end * PI * c_m) / (e * denm)
                 + (2.0 * aq2 * off * PI * c_m) / (e * denm)
                 + (2.0 * aq3 * delta_p * g * c_p) / (e * denp * denp)
                 + (aq3 * omega * c_p) / (e * denp)
                 - (aq3 * omega0 * c_p) / (e * denp)
                 + (2.0 * aq2 * end * PI * c_p) / (e * denp)
                 - (2.0 * aq2 * off * PI * c_p) / (e * denp)
                 + (4.0 * omega * dw * s_phase) / (e * den0 * den0)
                 + (4.0 * omega0 * (-dw) * s_phase) / (e * den0 * den0)
                 - (2.0 * s_phase) / (e * den0)
                 - (2.0 * aq * g * s_phase) / (e * den0)
                 + (2.0 * aq3 * omega * delta_p * s2off) / (denp * denp)
                 + (4.0 * aq2 * end * PI * delta_p * s2off) / (denp * denp)
                 + (2.0 * aq3 * omega0 * (-delta_p) * s2off) / (denp * denp)
                 + (4.0 * aq2 * off * PI * (-delta_p) * s2off) / (denp * denp)
                 - (aq2 * s2off) / denp
                 + (2.0 * aq3 * omega * (-delta_m) * s2off) / (denm * denm)
                 + (4.0 * aq2 * off * PI * (-delta_m) * s2off) / (denm * denm)
                 + (2.0 * aq3 * omega0 * delta_m * s2off) / (denm * denm)
                 + (4.0 * aq2 * end * PI * delta_m * s2off) / (denm * denm)
                 + (aq2 * s2off) / denm
                 + (2.0 * aq3 * omega * (-delta_m) * s_m) / (e * denm * denm)
                 + (4.0 * aq2 * off * PI * (-delta_m) * s_m) / (e * denm * denm)
                 + (2.0 * aq3 * omega0 * delta_m * s_m) / (e * denm * denm)
                 + (4.0 * aq2 * end * PI * delta_m * s_m) / (e * denm * denm)
                 + (aq2 * s_m) / (e * denm)
                 + (aq3 * g * s_m) / (e * denm)
                 + (2.0 * aq3 * omega0 * delta_p * s_p) / (e * denp * denp)
                 + (4.0 * aq2 * off * PI * delta_p * s_p) / (e * denp * denp)
                 + (2.0 * aq3 * omega * (-delta_p) * s_p) / (e * denp * denp)
                 + (4.0 * aq2 * end * PI * (-delta_p) * s_p) / (e * denp * denp)
                 + (aq2 * s_p) / (e * denp)
                 + (aq3 * g * s_p) / (e * denp))
                / 4.0;

    *r_r2 = 0.25 * ((2.0 * dw2 - 2.0 * g2) * inv_den0_sq
                     + 2.0 * (-dw2 + aq * dw2 * g + g2 + aq * g3) * c_phase * inv_e * inv_den0_sq
                     + aq2 * (-inv_denp - inv_denm + 2.0 * aq2 * g2 * (inv_denp_sq + inv_denm_sq)) * c2off
                     - 2.0 * aq4 * g2 * c_m * inv_e * inv_denm_sq
                     + aq2 * c_m * inv_e * inv_denm
                     - aq3 * g * c_m * inv_e * inv_denm
                     - 2.0 * aq4 * g2 * c_p * inv_e * inv_denp_sq
                     + aq2 * c_p * inv_e * inv_denp
                     - aq3 * g * c_p * inv_e * inv_denp
                     - 4.0 * omega * g * s_phase * inv_e * inv_den0_sq
                     + 4.0 * omega0 * g * s_phase * inv_e * inv_den0_sq
                     - 2.0 * aq * omega * s_phase * inv_e * inv_den0
                     + 2.0 * aq * omega0 * s_phase * inv_e * inv_den0
                     - 2.0 * aq4 * omega * g * s2off * inv_denp_sq
                     + 2.0 * aq4 * omega0 * g * s2off * inv_denp_sq
                     - 4.0 * aq3 * end * PI * g * s2off * inv_denp_sq
                     + 4.0 * aq3 * off * PI * g * s2off * inv_denp_sq
                     + 2.0 * aq4 * omega * g * s2off * inv_denm_sq
                     - 2.0 * aq4 * omega0 * g * s2off * inv_denm_sq
                     - 4.0 * aq3 * end * PI * g * s2off * inv_denm_sq
                     + 4.0 * aq3 * off * PI * g * s2off * inv_denm_sq
                     + 2.0 * aq4 * omega * g * s_m * inv_e * inv_denm_sq
                     - 2.0 * aq4 * omega0 * g * s_m * inv_e * inv_denm_sq
                     - 4.0 * aq3 * end * PI * g * s_m * inv_e * inv_denm_sq
                     + 4.0 * aq3 * off * PI * g * s_m * inv_e * inv_denm_sq
                     + aq3 * omega * s_m * inv_e * inv_denm
                     - aq3 * omega0 * s_m * inv_e * inv_denm
                     - 2.0 * aq2 * end * PI * s_m * inv_e * inv_denm
                     + 2.0 * aq2 * off * PI * s_m * inv_e * inv_denm
                     + 2.0 * aq4 * omega * g * s_p * inv_e * inv_denp_sq
                     - 2.0 * aq4 * omega0 * g * s_p * inv_e * inv_denp_sq
                     + 4.0 * aq3 * end * PI * g * s_p * inv_e * inv_denp_sq
                     - 4.0 * aq3 * off * PI * g * s_p * inv_e * inv_denp_sq
                     + aq3 * omega * s_p * inv_e * inv_denp
                     - aq3 * omega0 * s_p * inv_e * inv_denp
                     + 2.0 * aq2 * end * PI * s_p * inv_e * inv_denp
                     - 2.0 * aq2 * off * PI * s_p * inv_e * inv_denp);
#else
    *vv = ((2*r2)/(pow(omega - omega0,2) + pow(r2,2)) - (2*r2*cos(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2))) + pow(aq,2)*r2*(-(1/(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2))) - 1/(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2)))*cos(2*off*PI) + (pow(aq,2)*r2*cos(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2))) + (pow(aq,2)*r2*cos(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2))) + (2*omega*sin(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2))) - (2*omega0*sin(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2))) + (pow(aq,2)*omega*sin(2*off*PI))/(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2)) - (pow(aq,2)*omega0*sin(2*off*PI))/(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2)) + (2*aq*end*PI*sin(2*off*PI))/(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2)) - (2*aq*off*PI*sin(2*off*PI))/(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2)) - (pow(aq,2)*omega*sin(2*off*PI))/(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2)) + (pow(aq,2)*omega0*sin(2*off*PI))/(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2)) + (2*aq*end*PI*sin(2*off*PI))/(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2)) - (2*aq*off*PI*sin(2*off*PI))/(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2)) - (pow(aq,2)*omega*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2))) + (pow(aq,2)*omega0*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2))) + (2*aq*end*PI*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2))) - (2*aq*off*PI*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2))) - (pow(aq,2)*omega*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2))) + (pow(aq,2)*omega0*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2))) - (2*aq*end*PI*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2))) + (2*aq*off*PI*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2))))/4.;
    *r_omega0 = ((4*(omega - omega0)*r2)/pow(pow(omega - omega0,2) + pow(r2,2),2) + (4*(-omega + omega0)*r2*cos(aq*(omega - omega0)))/(exp(aq*r2)*pow(pow(omega - omega0,2) + pow(r2,2),2)) - (2*aq*omega*cos(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2))) + (2*aq*omega0*cos(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2))) + 2*pow(aq,3)*r2*((aq*(-omega + omega0) + 2*(-end + off)*PI)/pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (aq*(-omega + omega0) + 2*(end - off)*PI)/pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2))*cos(2*off*PI) + (2*pow(aq,3)*(aq*(omega - omega0) + 2*(-end + off)*PI)*r2*cos(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (pow(aq,3)*omega*cos(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (pow(aq,3)*omega0*cos(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (2*pow(aq,2)*end*PI*cos(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,2)*off*PI*cos(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,3)*(aq*(omega - omega0) + 2*(end - off)*PI)*r2*cos(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (pow(aq,3)*omega*cos(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (pow(aq,3)*omega0*cos(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,2)*end*PI*cos(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (2*pow(aq,2)*off*PI*cos(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (4*omega*(omega - omega0)*sin(aq*(omega - omega0)))/(exp(aq*r2)*pow(pow(omega - omega0,2) + pow(r2,2),2)) + (4*omega0*(-omega + omega0)*sin(aq*(omega - omega0)))/(exp(aq*r2)*pow(pow(omega - omega0,2) + pow(r2,2),2)) - (2*sin(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2))) - (2*aq*r2*sin(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2))) + (2*pow(aq,3)*omega*(aq*(omega - omega0) + 2*(end - off)*PI)*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (4*pow(aq,2)*end*PI*(aq*(omega - omega0) + 2*(end - off)*PI)*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (2*pow(aq,3)*omega0*(aq*(-omega + omega0) + 2*(-end + off)*PI)*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (4*pow(aq,2)*off*PI*(aq*(-omega + omega0) + 2*(-end + off)*PI)*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) - (pow(aq,2)*sin(2*off*PI))/(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2)) + (2*pow(aq,3)*omega*(aq*(-omega + omega0) + 2*(end - off)*PI)*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (4*pow(aq,2)*off*PI*(aq*(-omega + omega0) + 2*(end - off)*PI)*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (2*pow(aq,3)*omega0*(aq*(omega - omega0) + 2*(-end + off)*PI)*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (4*pow(aq,2)*end*PI*(aq*(omega - omega0) + 2*(-end + off)*PI)*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (pow(aq,2)*sin(2*off*PI))/(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2)) + (2*pow(aq,3)*omega*(aq*(-omega + omega0) + 2*(end - off)*PI)*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (4*pow(aq,2)*off*PI*(aq*(-omega + omega0) + 2*(end - off)*PI)*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (2*pow(aq,3)*omega0*(aq*(omega - omega0) + 2*(-end + off)*PI)*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (4*pow(aq,2)*end*PI*(aq*(omega - omega0) + 2*(-end + off)*PI)*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (pow(aq,2)*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (pow(aq,3)*r2*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,3)*omega0*(aq*(omega - omega0) + 2*(end - off)*PI)*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (4*pow(aq,2)*off*PI*(aq*(omega - omega0) + 2*(end - off)*PI)*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (2*pow(aq,3)*omega*(aq*(-omega + omega0) + 2*(-end + off)*PI)*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (4*pow(aq,2)*end*PI*(aq*(-omega + omega0) + 2*(-end + off)*PI)*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (pow(aq,2)*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (pow(aq,3)*r2*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))))/4.;
    *r_r2 = ((2*pow(omega - omega0,2) - 2*pow(r2,2))/pow(pow(omega - omega0,2) + pow(r2,2),2) + (2*(-pow(omega - omega0,2) + aq*pow(omega - omega0,2)*r2 + pow(r2,2) + aq*pow(r2,3))*cos(aq*(omega - omega0)))/(exp(aq*r2)*pow(pow(omega - omega0,2) + pow(r2,2),2)) + pow(aq,2)*(-(1/(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2))) - 1/(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2)) + 2*pow(aq,2)*pow(r2,2)*(pow(pow(aq*omega - aq*omega0 + 2*end*PI - 2*off*PI,2) + pow(aq,2)*pow(r2,2),-2) + pow(pow(aq*omega - aq*omega0 - 2*end*PI + 2*off*PI,2) + pow(aq,2)*pow(r2,2),-2)))*cos(2*off*PI) - (2*pow(aq,4)*pow(r2,2)*cos(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (pow(aq,2)*cos(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (pow(aq,3)*r2*cos(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (2*pow(aq,4)*pow(r2,2)*cos(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (pow(aq,2)*cos(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (pow(aq,3)*r2*cos(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (4*omega*r2*sin(aq*(omega - omega0)))/(exp(aq*r2)*pow(pow(omega - omega0,2) + pow(r2,2),2)) + (4*omega0*r2*sin(aq*(omega - omega0)))/(exp(aq*r2)*pow(pow(omega - omega0,2) + pow(r2,2),2)) - (2*aq*omega*sin(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2))) + (2*aq*omega0*sin(aq*(omega - omega0)))/(exp(aq*r2)*(pow(omega - omega0,2) + pow(r2,2))) - (2*pow(aq,4)*omega*r2*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (2*pow(aq,4)*omega0*r2*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) - (4*pow(aq,3)*end*PI*r2*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (4*pow(aq,3)*off*PI*r2*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (2*pow(aq,4)*omega*r2*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) - (2*pow(aq,4)*omega0*r2*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) - (4*pow(aq,3)*end*PI*r2*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (4*pow(aq,3)*off*PI*r2*sin(2*off*PI))/pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (2*pow(aq,4)*omega*r2*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (2*pow(aq,4)*omega0*r2*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (4*pow(aq,3)*end*PI*r2*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (4*pow(aq,3)*off*PI*r2*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (pow(aq,3)*omega*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (pow(aq,3)*omega0*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (2*pow(aq,2)*end*PI*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,2)*off*PI*sin(aq*omega - aq*omega0 - 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,4)*omega*r2*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (2*pow(aq,4)*omega0*r2*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (4*pow(aq,3)*end*PI*r2*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (4*pow(aq,3)*off*PI*r2*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (pow(aq,3)*omega*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (pow(aq,3)*omega0*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,2)*end*PI*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (2*pow(aq,2)*off*PI*sin(aq*omega - aq*omega0 + 2*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 2*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))))/4.;
#endif
}

void mycostfunction_fitnmr_base::sp2_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end,
    double *vv) const
{
    double dummy_omega0, dummy_r2;
    sp2_helper(omega, omega0, r2, aq, off, end, vv, &dummy_omega0, &dummy_r2);
}

void mycostfunction_fitnmr_base::sp3_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end,
    double *vv,
    double *r_omega0,
    double *r_r2) const
{
#if FITNMR_USE_SUBEXPRESSIONS
    const double Pi = PI;
    const double g = r2;
    const double aq2 = aq * aq;
    const double aq3 = aq2 * aq;
    const double aq4 = aq2 * aq2;
    const double aq5 = aq4 * aq;
    const double dw = omega - omega0;
    const double dw2 = dw * dw;
    const double g2 = g * g;
    const double g3 = g2 * g;
    const double ed = end - off;
    const double ed2 = ed * ed;
    const double ed4 = ed2 * ed2;
    const double shift1 = ed * PI;
    const double shift3 = 3.0 * shift1;
    const double phase = aq * dw;
    const double phase1p = phase + shift1;
    const double phase1m = phase - shift1;
    const double phase3p = phase + shift3;
    const double phase3m = phase - shift3;
    const double phase1e = -phase + shift1;
    const double phase3e = -phase + 3.0 * end * PI;
    const double e = exp(aq * g);
    const double s_off = sin(off * PI);
    const double c_off = cos(off * PI);
    const double s_3off = sin(3.0 * off * PI);
    const double c_3off = cos(3.0 * off * PI);
    const double s_1p = sin(phase1p);
    const double c_1p = cos(phase1p);
    const double s_1m = sin(phase1m);
    const double c_1m = cos(phase1m);
    const double s_3p = sin(phase3p);
    const double c_3p = cos(phase3p);
    const double s_3m = sin(phase3m);
    const double c_3m = cos(phase3m);
    const double s_1e = sin(phase1e);
    const double c_1e = cos(phase1e);
    const double s_3e = sin(phase3e);
    const double c_3e = cos(phase3e);
    const double den1p = phase1p * phase1p + aq2 * g2;
    const double den1m = phase1m * phase1m + aq2 * g2;
    const double den3p = phase3p * phase3p + aq2 * g2;
    const double den3m = phase3m * phase3m + aq2 * g2;

    *vv = (aq * (-((aq*omega*c_3e)/(e*(den3m)))
                 + (aq*omega0*c_3e)/(e*(den3m))
                 + (3*end*PI*c_3e)/(e*(den3m))
                 - (3*off*PI*c_3e)/(e*(den3m))
                 + (3*aq*r2*s_off)/(den1p)
                 + (3*aq*r2*s_off)/(den1m)
                 - (3*aq*r2*c_1p*s_off)/(e*(den1p))
                 - (3*aq*r2*c_1m*s_off)/(e*(den1m))
                 - (aq*r2*s_3off)/(den3p)
                 - (aq*r2*s_3off)/(den3m)
                 + (aq*r2*c_3p*s_3off)/(e*(den3p))
                 + (aq*r2*s_3e)/(e*(den3m))
                 + (3*aq*omega0*s_off*s_1e)/(e*(den1m))
                 + (3*end*PI*s_off*s_1e)/(e*(den1m))
                 - (aq*omega*s_3off*s_3p)/(e*(den3p))
                 + (aq*omega0*s_3off*s_3p)/(e*(den3p))
                 - (3*end*PI*s_3off*s_3p)/(e*(den3p))
                 + (3*off*PI*s_3off*s_3p)/(e*(den3p))
                 + (c_3off*((6*(end - off)*PI*(-9*ed2*pow(PI,2) + aq2*(dw2 - g2)))/(den3m) + ((phase + 3*(end - off)*PI)*c_3p + aq*r2*s_3p)/e))/(den3p)
                 + (3*aq*omega*s_off*s_1p)/(e*(den1p))
                 - (3*aq*omega0*s_off*s_1p)/(e*(den1p))
                 + (3*end*PI*s_off*s_1p)/(e*(den1p))
                 - (3*off*PI*s_off*s_1p)/(e*(den1p))
                 + (3*aq*omega*s_off*s_1m)/(e*(den1m))
                 + (3*off*PI*s_off*s_1m)/(e*(den1m))
                 + 3*c_off*(((aq*(-omega + omega0) + (-end + off)*PI)*c_1p)/(e*(den1p)) + (((phase + (-end + off)*PI)*c_1m)/e + (2*(end - off)*PI*(ed2*pow(PI,2) + aq2*(-dw2 + g2)) - (aq*r2*((den1m)*s_1p - (den1p)*s_1m))/e)/(den1p))/(den1m))
                 )) / 8.0;

    *r_omega0 = (aq2 * ((2*aq*omega*(aq*(-omega + omega0) + 3*(end - off)*PI)*c_3e)/(e*pow(den3m,2))
                        + (6*off*PI*(aq*(-omega + omega0) + 3*(end - off)*PI)*c_3e)/(e*pow(den3m,2))
                        + (2*aq*omega0*(phase + 3*(-end + off)*PI)*c_3e)/(e*pow(den3m,2))
                        + (6*end*PI*(phase + 3*(-end + off)*PI)*c_3e)/(e*pow(den3m,2))
                        + c_3e/(e*(den3m))
                        + (aq*r2*c_3e)/(e*(den3m))
                        + (6*aq*(phase + (end - off)*PI)*r2*s_off)/pow(den1p,2)
                        + (6*aq*(phase + (-end + off)*PI)*r2*s_off)/pow(den1m,2)
                        + (6*aq*(aq*(-omega + omega0) + (-end + off)*PI)*r2*c_1p*s_off)/(e*pow(den1p,2))
                        - (3*aq*omega*c_1p*s_off)/(e*(den1p))
                        + (3*aq*omega0*c_1p*s_off)/(e*(den1p))
                        - (3*end*PI*c_1p*s_off)/(e*(den1p))
                        + (3*off*PI*c_1p*s_off)/(e*(den1p))
                        + (6*aq*(phase1e)*r2*c_1m*s_off)/(e*pow(den1m,2))
                        - (3*aq*omega*c_1m*s_off)/(e*(den1m))
                        + (3*aq*omega0*c_1m*s_off)/(e*(den1m))
                        + (3*end*PI*c_1m*s_off)/(e*(den1m))
                        - (3*off*PI*c_1m*s_off)/(e*(den1m))
                        + (2*aq*(aq*(-omega + omega0) + 3*(-end + off)*PI)*r2*s_3off)/pow(den3p,2)
                        + (2*aq*(aq*(-omega + omega0) + 3*(end - off)*PI)*r2*s_3off)/pow(den3m,2)
                        + (2*aq*(phase + 3*(end - off)*PI)*r2*c_3p*s_3off)/(e*pow(den3p,2))
                        + (aq*omega*c_3p*s_3off)/(e*(den3p))
                        - (aq*omega0*c_3p*s_3off)/(e*(den3p))
                        + (3*end*PI*c_3p*s_3off)/(e*(den3p))
                        - (3*off*PI*c_3p*s_3off)/(e*(den3p))
                        + (aq*omega0*sin(aq*omega - aq*omega0 - 3*end*PI))/(e*(den3m))
                        + (3*end*PI*sin(aq*omega - aq*omega0 - 3*end*PI))/(e*(den3m))
                        + (2*aq*(phase + 3*(-end + off)*PI)*r2*s_3e)/(e*pow(den3m,2))
                        + (aq*omega*s_3e)/(e*(den3m))
                        + (3*off*PI*s_3e)/(e*(den3m))
                        + (6*aq*omega0*(phase + (-end + off)*PI)*s_off*s_1e)/(e*pow(den1m,2))
                        + (6*end*PI*(phase + (-end + off)*PI)*s_off*s_1e)/(e*pow(den1m,2))
                        + (3*s_off*s_1e)/(e*(den1m))
                        + (2*aq*omega0*(phase + 3*(end - off)*PI)*s_3off*s_3p)/(e*pow(den3p,2))
                        + (6*off*PI*(phase + 3*(end - off)*PI)*s_3off*s_3p)/(e*pow(den3p,2))
                        + (2*aq*omega*(aq*(-omega + omega0) + 3*(-end + off)*PI)*s_3off*s_3p)/(e*pow(den3p,2))
                        + (6*end*PI*(aq*(-omega + omega0) + 3*(-end + off)*PI)*s_3off*s_3p)/(e*pow(den3p,2))
                        + (s_3off*s_3p)/(e*(den3p))
                        + (aq*r2*s_3off*s_3p)/(e*(den3p))
                        + (c_3off*((12*e*(end - off)*PI*(3*(end - off)*PI*pow(phase3m,2) + aq2*(-2*aq*omega + 2*aq*omega0 + 3*end*PI - 3*off*PI)*g2))/pow(den3p,2) - (1 + aq*r2)*c_3p + (phase3p)*s_3p))/(e*(den3p))
                        + (2*(phase + 3*(end - off)*PI)*c_3off*((6*(end - off)*PI*(-9*ed2*pow(PI,2) + aq2*(dw2 - g2)))/(den3m) + ((phase + 3*(end - off)*PI)*c_3p + aq*r2*s_3p)/e))/pow(den3p,2)
                        + (6*aq*omega*(phase + (end - off)*PI)*s_off*s_1p)/(e*pow(den1p,2))
                        + (6*end*PI*(phase + (end - off)*PI)*s_off*s_1p)/(e*pow(den1p,2))
                        + (6*aq*omega0*(aq*(-omega + omega0) + (-end + off)*PI)*s_off*s_1p)/(e*pow(den1p,2))
                        + (6*off*PI*(aq*(-omega + omega0) + (-end + off)*PI)*s_off*s_1p)/(e*pow(den1p,2))
                        - (3*s_off*s_1p)/(e*(den1p))
                        - (3*aq*r2*s_off*s_1p)/(e*(den1p))
                        + (6*aq*omega*(phase + (-end + off)*PI)*s_off*s_1m)/(e*pow(den1m,2))
                        + (6*off*PI*(phase + (-end + off)*PI)*s_off*s_1m)/(e*pow(den1m,2))
                        - (3*aq*r2*s_off*s_1m)/(e*(den1m))
                        + 3*c_off*((-2*pow(phase + (end - off)*PI,2)*c_1p)/(e*pow(den1p,2)) + c_1p/(e*(den1p)) + ((aq*(-omega + omega0) + (-end + off)*PI)*s_1p)/(e*(den1p)) + (4*e*(end - off)*PI*((end - off)*PI*pow(phase1p,2) + aq2*(2*aq*omega - 2*aq*omega0 + end*PI - off*PI)*g2) + aq*r2*(den1p)*(den1m)*c_1p - (1 + aq*r2)*pow(den1p,2)*c_1m - 4*aq*(end - off)*PI*r2*(ed2*pow(PI,2) + aq2*(-dw2 + g2))*s_1p + (phase1m)*pow(den1p,2)*s_1m)/(e*pow(den1p,2)*(den1m)) + (2*(phase + (-end + off)*PI)*(((phase + (-end + off)*PI)*c_1m)/e + (2*(end - off)*PI*(ed2*pow(PI,2) + aq2*(-dw2 + g2)) - (aq*r2*((den1m)*s_1p - (den1p)*s_1m))/e)/(den1p)))/pow(den1m,2))
                        )) / 8.0;

    *r_r2 = (aq2 * ((2*aq2*omega*r2*c_3e)/(e*pow(den3m,2))
                    - (2*aq2*omega0*r2*c_3e)/(e*pow(den3m,2))
                    - (6*aq*end*PI*r2*c_3e)/(e*pow(den3m,2))
                    + (6*aq*off*PI*r2*c_3e)/(e*pow(den3m,2))
                    + (aq*omega*c_3e)/(e*(den3m))
                    - (aq*omega0*c_3e)/(e*(den3m))
                    - (3*end*PI*c_3e)/(e*(den3m))
                    + (3*off*PI*c_3e)/(e*(den3m))
                    - (6*aq2*g2*s_off)/pow(den1p,2)
                    + (3*s_off)/(den1p)
                    - (6*aq2*g2*s_off)/pow(den1m,2)
                    + (3*s_off)/(den1m)
                    + (6*aq2*g2*c_1p*s_off)/(e*pow(den1p,2))
                    - (3*c_1p*s_off)/(e*(den1p))
                    + (3*aq*r2*c_1p*s_off)/(e*(den1p))
                    + (6*aq2*g2*c_1m*s_off)/(e*pow(den1m,2))
                    - (3*c_1m*s_off)/(e*(den1m))
                    + (3*aq*r2*c_1m*s_off)/(e*(den1m))
                    + (2*aq2*g2*s_3off)/pow(den3p,2)
                    - s_3off/(den3p)
                    + (2*aq2*g2*s_3off)/pow(den3m,2)
                    - s_3off/(den3m)
                    - (2*aq2*g2*c_3p*s_3off)/(e*pow(den3p,2))
                    + (c_3p*s_3off)/(e*(den3p))
                    - (aq*r2*c_3p*s_3off)/(e*(den3p))
                    + (2*aq2*g2*sin(aq*omega - aq*omega0 - 3*end*PI))/(e*pow(den3m,2))
                    + (aq*r2*sin(aq*omega - aq*omega0 - 3*end*PI))/(e*(den3m))
                    + s_3e/(e*(den3m))
                    + (2*aq2*omega*r2*s_3off*s_3p)/(e*pow(den3p,2))
                    - (2*aq2*omega0*r2*s_3off*s_3p)/(e*pow(den3p,2))
                    + (6*aq*end*PI*r2*s_3off*s_3p)/(e*pow(den3p,2))
                    - (6*aq*off*PI*r2*s_3off*s_3p)/(e*pow(den3p,2))
                    + (aq*omega*s_3off*s_3p)/(e*(den3p))
                    - (aq*omega0*s_3off*s_3p)/(e*(den3p))
                    + (3*end*PI*s_3off*s_3p)/(e*(den3p))
                    - (3*off*PI*s_3off*s_3p)/(e*(den3p))
                    - (2*aq*r2*c_3off*((6*(end - off)*PI*(-9*ed2*pow(PI,2) + aq2*(dw2 - g2)))/(den3m) + ((phase + 3*(end - off)*PI)*c_3p + aq*r2*s_3p)/e))/pow(den3p,2)
                    + (c_3off*((24*aq2*(end - off)*(omega - omega0)*PI*(aq*(-omega + omega0) + 3*(end - off)*PI)*r2)/pow(den3m,2) + ((aq*(-omega + omega0) + 3*(-end + off)*PI)*c_3p + (1 - aq*r2)*s_3p)/e))/(den3p)
                    - (6*aq2*omega*r2*s_off*s_1p)/(e*pow(den1p,2))
                    + (6*aq2*omega0*r2*s_off*s_1p)/(e*pow(den1p,2))
                    - (6*aq*end*PI*r2*s_off*s_1p)/(e*pow(den1p,2))
                    + (6*aq*off*PI*r2*s_off*s_1p)/(e*pow(den1p,2))
                    - (3*aq*omega*s_off*s_1p)/(e*(den1p))
                    + (3*aq*omega0*s_off*s_1p)/(e*(den1p))
                    - (3*end*PI*s_off*s_1p)/(e*(den1p))
                    + (3*off*PI*s_off*s_1p)/(e*(den1p))
                    - (6*aq2*omega*r2*s_off*s_1m)/(e*pow(den1m,2))
                    + (6*aq2*omega0*r2*s_off*s_1m)/(e*pow(den1m,2))
                    + (6*aq*end*PI*r2*s_off*s_1m)/(e*pow(den1m,2))
                    - (6*aq*off*PI*r2*s_off*s_1m)/(e*pow(den1m,2))
                    - (3*aq*omega*s_off*s_1m)/(e*(den1m))
                    + (3*aq*omega0*s_off*s_1m)/(e*(den1m))
                    + (3*end*PI*s_off*s_1m)/(e*(den1m))
                    - (3*off*PI*s_off*s_1m)/(e*(den1m))
                    + 3*c_off*((2*aq*(phase + (end - off)*PI)*r2*c_1p)/(e*pow(den1p,2)) + ((phase + (end - off)*PI)*c_1p)/(e*(den1p)) + ((8*aq2*e*(end - off)*(omega - omega0)*PI*(phase + (end - off)*PI)*r2)/pow(den1p,2) + (phase1e)*c_1m + ((-(ed4*pow(PI,4)) + aq*ed4*pow(PI,4)*r2 + 2*aq2*ed2*pow(PI,2)*(dw2 - g2) - aq4*pow(dw2 + g2,2) + aq5*r2*pow(dw2 + g2,2) + 2*aq3*(end - off)*PI*r2*(-((end - off)*dw2*PI) + 4*(-omega + omega0)*r2 + (end - off)*PI*g2))*s_1p)/pow(den1p,2) + (1 - aq*r2)*s_1m)/(e*(den1m)) - (2*aq*r2*(((phase + (-end + off)*PI)*c_1m)/e + (2*(end - off)*PI*(ed2*pow(PI,2) + aq2*(-dw2 + g2)) - (aq*r2*((den1m)*s_1p - (den1p)*s_1m))/e)/(den1p)))/pow(den1m,2))
                    )) / 8.0;
#else
    *vv = (aq*(-((aq*omega*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2)))) + (aq*omega0*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*end*PI*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*off*PI*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*aq*r2*sin(off*PI))/(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2)) + (3*aq*r2*sin(off*PI))/(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2)) - (3*aq*r2*cos(aq*omega - aq*omega0 + end*PI - off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*aq*r2*cos(aq*omega - aq*omega0 - end*PI + off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (aq*r2*sin(3*off*PI))/(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2)) - (aq*r2*sin(3*off*PI))/(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2)) + (aq*r2*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (aq*r2*sin(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*aq*omega0*sin(off*PI)*sin(aq*(-omega + omega0) + (end - off)*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*end*PI*sin(off*PI)*sin(aq*(-omega + omega0) + (end - off)*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (aq*omega*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (aq*omega0*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*end*PI*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*off*PI*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (cos(3*off*PI)*((6*(end - off)*PI*(-9*pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) - pow(r2,2))))/(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2)) + ((aq*(omega - omega0) + 3*(end - off)*PI)*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + aq*r2*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/exp(aq*r2)))/(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2)) + (3*aq*omega*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*aq*omega0*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*end*PI*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*off*PI*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*aq*omega*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*off*PI*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + 3*cos(off*PI)*(((aq*(-omega + omega0) + (-end + off)*PI)*cos(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (((aq*(omega - omega0) + (-end + off)*PI)*cos(aq*omega - aq*omega0 - end*PI + off*PI))/exp(aq*r2) + (2*(end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2))) - (aq*r2*((pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))*sin(aq*omega - aq*omega0 + end*PI - off*PI) - (pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))*sin(aq*omega - aq*omega0 - end*PI + off*PI)))/exp(aq*r2))/(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2)))/(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2)))))/8.;
    *r_omega0 = (pow(aq,2)*((2*aq*omega*(aq*(-omega + omega0) + 3*(end - off)*PI)*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*off*PI*(aq*(-omega + omega0) + 3*(end - off)*PI)*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (2*aq*omega0*(aq*(omega - omega0) + 3*(-end + off)*PI)*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*end*PI*(aq*(omega - omega0) + 3*(-end + off)*PI)*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + cos(aq*(-omega + omega0) + 3*end*PI)/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (aq*r2*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (6*aq*(aq*(omega - omega0) + (end - off)*PI)*r2*sin(off*PI))/pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (6*aq*(aq*(omega - omega0) + (-end + off)*PI)*r2*sin(off*PI))/pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (6*aq*(aq*(-omega + omega0) + (-end + off)*PI)*r2*cos(aq*omega - aq*omega0 + end*PI - off*PI)*sin(off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (3*aq*omega*cos(aq*omega - aq*omega0 + end*PI - off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*aq*omega0*cos(aq*omega - aq*omega0 + end*PI - off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*end*PI*cos(aq*omega - aq*omega0 + end*PI - off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*off*PI*cos(aq*omega - aq*omega0 + end*PI - off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (6*aq*(aq*(-omega + omega0) + (end - off)*PI)*r2*cos(aq*omega - aq*omega0 - end*PI + off*PI)*sin(off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (3*aq*omega*cos(aq*omega - aq*omega0 - end*PI + off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*aq*omega0*cos(aq*omega - aq*omega0 - end*PI + off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*end*PI*cos(aq*omega - aq*omega0 - end*PI + off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*off*PI*cos(aq*omega - aq*omega0 - end*PI + off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*aq*(aq*(-omega + omega0) + 3*(-end + off)*PI)*r2*sin(3*off*PI))/pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (2*aq*(aq*(-omega + omega0) + 3*(end - off)*PI)*r2*sin(3*off*PI))/pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (2*aq*(aq*(omega - omega0) + 3*(end - off)*PI)*r2*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (aq*omega*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (aq*omega0*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*end*PI*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*off*PI*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (aq*omega0*sin(aq*omega - aq*omega0 - 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*end*PI*sin(aq*omega - aq*omega0 - 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*aq*(aq*(omega - omega0) + 3*(-end + off)*PI)*r2*sin(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (aq*omega*sin(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*off*PI*sin(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (6*aq*omega0*(aq*(omega - omega0) + (-end + off)*PI)*sin(off*PI)*sin(aq*(-omega + omega0) + (end - off)*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*end*PI*(aq*(omega - omega0) + (-end + off)*PI)*sin(off*PI)*sin(aq*(-omega + omega0) + (end - off)*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (3*sin(off*PI)*sin(aq*(-omega + omega0) + (end - off)*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*aq*omega0*(aq*(omega - omega0) + 3*(end - off)*PI)*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*off*PI*(aq*(omega - omega0) + 3*(end - off)*PI)*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (2*aq*omega*(aq*(-omega + omega0) + 3*(-end + off)*PI)*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*end*PI*(aq*(-omega + omega0) + 3*(-end + off)*PI)*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (aq*r2*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (cos(3*off*PI)*((12*exp(aq*r2)*(end - off)*PI*(3*(end - off)*PI*pow(aq*omega - aq*omega0 - 3*end*PI + 3*off*PI,2) + pow(aq,2)*(-2*aq*omega + 2*aq*omega0 + 3*end*PI - 3*off*PI)*pow(r2,2)))/pow(pow(aq*omega - aq*omega0 - 3*end*PI + 3*off*PI,2) + pow(aq,2)*pow(r2,2),2) - (1 + aq*r2)*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + (aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*(aq*(omega - omega0) + 3*(end - off)*PI)*cos(3*off*PI)*((6*(end - off)*PI*(-9*pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) - pow(r2,2))))/(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2)) + ((aq*(omega - omega0) + 3*(end - off)*PI)*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + aq*r2*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/exp(aq*r2)))/pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (6*aq*omega*(aq*(omega - omega0) + (end - off)*PI)*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*end*PI*(aq*(omega - omega0) + (end - off)*PI)*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*aq*omega0*(aq*(-omega + omega0) + (-end + off)*PI)*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*off*PI*(aq*(-omega + omega0) + (-end + off)*PI)*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (3*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*aq*r2*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (6*aq*omega*(aq*(omega - omega0) + (-end + off)*PI)*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*off*PI*(aq*(omega - omega0) + (-end + off)*PI)*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (3*aq*r2*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + 3*cos(off*PI)*((-2*pow(aq*(omega - omega0) + (end - off)*PI,2)*cos(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + cos(aq*omega - aq*omega0 + end*PI - off*PI)/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + ((aq*(-omega + omega0) + (-end + off)*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (4*exp(aq*r2)*(end - off)*PI*((end - off)*PI*pow(aq*omega - aq*omega0 + end*PI - off*PI,2) + pow(aq,2)*(2*aq*omega - 2*aq*omega0 + end*PI - off*PI)*pow(r2,2)) + aq*r2*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))*cos(aq*omega - aq*omega0 + end*PI - off*PI) - (1 + aq*r2)*pow(pow(aq*omega - aq*omega0 + end*PI - off*PI,2) + pow(aq,2)*pow(r2,2),2)*cos(aq*omega - aq*omega0 - end*PI + off*PI) - 4*aq*(end - off)*PI*r2*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2)))*sin(aq*omega - aq*omega0 + end*PI - off*PI) + (aq*omega - aq*omega0 - end*PI + off*PI)*pow(pow(aq*omega - aq*omega0 + end*PI - off*PI,2) + pow(aq,2)*pow(r2,2),2)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*(aq*(omega - omega0) + (-end + off)*PI)*(((aq*(omega - omega0) + (-end + off)*PI)*cos(aq*omega - aq*omega0 - end*PI + off*PI))/exp(aq*r2) + (2*(end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2))) - (aq*r2*((pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))*sin(aq*omega - aq*omega0 + end*PI - off*PI) - (pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))*sin(aq*omega - aq*omega0 - end*PI + off*PI)))/exp(aq*r2))/(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))))/pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2))))/8.;
    *r_r2 = (pow(aq,2)*((2*pow(aq,2)*omega*r2*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (2*pow(aq,2)*omega0*r2*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (6*aq*end*PI*r2*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*aq*off*PI*r2*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (aq*omega*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (aq*omega0*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*end*PI*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*off*PI*cos(aq*(-omega + omega0) + 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (6*pow(aq,2)*pow(r2,2)*sin(off*PI))/pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (3*sin(off*PI))/(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2)) - (6*pow(aq,2)*pow(r2,2)*sin(off*PI))/pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (3*sin(off*PI))/(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2)) + (6*pow(aq,2)*pow(r2,2)*cos(aq*omega - aq*omega0 + end*PI - off*PI)*sin(off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (3*cos(aq*omega - aq*omega0 + end*PI - off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*aq*r2*cos(aq*omega - aq*omega0 + end*PI - off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (6*pow(aq,2)*pow(r2,2)*cos(aq*omega - aq*omega0 - end*PI + off*PI)*sin(off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (3*cos(aq*omega - aq*omega0 - end*PI + off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*aq*r2*cos(aq*omega - aq*omega0 - end*PI + off*PI)*sin(off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,2)*pow(r2,2)*sin(3*off*PI))/pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) - sin(3*off*PI)/(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2)) + (2*pow(aq,2)*pow(r2,2)*sin(3*off*PI))/pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) - sin(3*off*PI)/(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2)) - (2*pow(aq,2)*pow(r2,2)*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (aq*r2*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI)*sin(3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,2)*pow(r2,2)*sin(aq*omega - aq*omega0 - 3*end*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (aq*r2*sin(aq*omega - aq*omega0 - 3*end*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + sin(aq*(-omega + omega0) + 3*end*PI)/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (2*pow(aq,2)*omega*r2*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (2*pow(aq,2)*omega0*r2*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*aq*end*PI*r2*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (6*aq*off*PI*r2*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (aq*omega*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (aq*omega0*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*end*PI*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*off*PI*sin(3*off*PI)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (2*aq*r2*cos(3*off*PI)*((6*(end - off)*PI*(-9*pow(end - off,2)*pow(PI,2) + pow(aq,2)*(pow(omega - omega0,2) - pow(r2,2))))/(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2)) + ((aq*(omega - omega0) + 3*(end - off)*PI)*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + aq*r2*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/exp(aq*r2)))/pow(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (cos(3*off*PI)*((24*pow(aq,2)*(end - off)*(omega - omega0)*PI*(aq*(-omega + omega0) + 3*(end - off)*PI)*r2)/pow(pow(aq*(omega - omega0) + 3*(-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2) + ((aq*(-omega + omega0) + 3*(-end + off)*PI)*cos(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI) + (1 - aq*r2)*sin(aq*omega - aq*omega0 + 3*end*PI - 3*off*PI))/exp(aq*r2)))/(pow(aq*(omega - omega0) + 3*(end - off)*PI,2) + pow(aq,2)*pow(r2,2)) - (6*pow(aq,2)*omega*r2*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*pow(aq,2)*omega0*r2*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (6*aq*end*PI*r2*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*aq*off*PI*r2*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (3*aq*omega*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*aq*omega0*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*end*PI*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*off*PI*sin(off*PI)*sin(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) - (6*pow(aq,2)*omega*r2*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*pow(aq,2)*omega0*r2*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + (6*aq*end*PI*r2*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (6*aq*off*PI*r2*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2)) - (3*aq*omega*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*aq*omega0*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + (3*end*PI*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (3*off*PI*sin(off*PI)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) + 3*cos(off*PI)*((2*aq*(aq*(omega - omega0) + (end - off)*PI)*r2*cos(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2)) + ((aq*(omega - omega0) + (end - off)*PI)*cos(aq*omega - aq*omega0 + end*PI - off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))) + ((8*pow(aq,2)*exp(aq*r2)*(end - off)*(omega - omega0)*PI*(aq*(omega - omega0) + (end - off)*PI)*r2)/pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (aq*(-omega + omega0) + (end - off)*PI)*cos(aq*omega - aq*omega0 - end*PI + off*PI) + ((-(pow(end - off,4)*pow(PI,4)) + aq*pow(end - off,4)*pow(PI,4)*r2 + 2*pow(aq,2)*pow(end - off,2)*pow(PI,2)*(pow(omega - omega0,2) - pow(r2,2)) - pow(aq,4)*pow(pow(omega - omega0,2) + pow(r2,2),2) + pow(aq,5)*r2*pow(pow(omega - omega0,2) + pow(r2,2),2) + 2*pow(aq,3)*(end - off)*PI*r2*(-((end - off)*pow(omega - omega0,2)*PI) + 4*(-omega + omega0)*r2 + (end - off)*PI*pow(r2,2)))*sin(aq*omega - aq*omega0 + end*PI - off*PI))/pow(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2),2) + (1 - aq*r2)*sin(aq*omega - aq*omega0 - end*PI + off*PI))/(exp(aq*r2)*(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))) - (2*aq*r2*(((aq*(omega - omega0) + (-end + off)*PI)*cos(aq*omega - aq*omega0 - end*PI + off*PI))/exp(aq*r2) + (2*(end - off)*PI*(pow(end - off,2)*pow(PI,2) + pow(aq,2)*(-pow(omega - omega0,2) + pow(r2,2))) - (aq*r2*((pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2))*sin(aq*omega - aq*omega0 + end*PI - off*PI) - (pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))*sin(aq*omega - aq*omega0 - end*PI + off*PI)))/exp(aq*r2))/(pow(aq*(omega - omega0) + (end - off)*PI,2) + pow(aq,2)*pow(r2,2))))/pow(pow(aq*(omega - omega0) + (-end + off)*PI,2) + pow(aq,2)*pow(r2,2),2))))/8.;
#endif
}

void mycostfunction_fitnmr_base::sp3_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double aq,
    const double off,
    const double end,
    double *vv) const
{
    double dummy_omega0, dummy_r2;
    sp3_helper(omega, omega0, r2, aq, off, end, vv, &dummy_omega0, &dummy_r2);
}

void mycostfunction_fitnmr_base::doublet_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double j,
    const double aq,
    const double off,
    const double end,
    double *vv,
    double *r_omega0,
    double *r_r2,
    double *r_j) const
{
    double v_left, dx_left, dr_left;
    double v_right, dx_right, dr_right;

    singlet_helper(omega, omega0 - 0.5 * j, r2, nx, aq, off, end, &v_left, &dx_left, &dr_left);
    singlet_helper(omega, omega0 + 0.5 * j, r2, nx, aq, off, end, &v_right, &dx_right, &dr_right);

    *vv = 0.5 * (v_left + v_right);
    *r_omega0 = 0.5 * (dx_left + dx_right);
    *r_r2 = 0.5 * (dr_left + dr_right);
    *r_j = 0.25 * (dx_right - dx_left);
}

void mycostfunction_fitnmr_base::doublet_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double j,
    const double aq,
    const double off,
    const double end,
    double *vv) const
{
    double dummy_omega0, dummy_r2, dummy_j;
    doublet_helper(omega, omega0, r2, j, aq, off, end, vv, &dummy_omega0, &dummy_r2, &dummy_j);
}

void mycostfunction_fitnmr_base::singlet_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double n,
    const double aq,
    const double off,
    const double end,
    double *vv,
    double *r_omega0,
    double *r_r2) const
{
    if (n > 2.5)
        sp3_helper(omega, omega0, r2, aq, off, end, vv, r_omega0, r_r2);
    else if (n > 1.5)
        sp2_helper(omega, omega0, r2, aq, off, end, vv, r_omega0, r_r2);
    else if (n > 0.5)
        sp1_helper(omega, omega0, r2, aq, off, end, vv, r_omega0, r_r2);
    else
        none_helper(omega, omega0, r2, aq, vv, r_omega0, r_r2);
}

void mycostfunction_fitnmr_base::singlet_helper(
    const double omega,
    const double omega0,
    const double r2,
    const double n,
    const double aq,
    const double off,
    const double end,
    double *vv) const
{
    if (n > 2.5)
        sp3_helper(omega, omega0, r2, aq, off, end, vv);
    else if (n > 1.5)
        sp2_helper(omega, omega0, r2, aq, off, end, vv);
    else if (n > 0.5)
        sp1_helper(omega, omega0, r2, aq, off, end, vv);
    else
        none_helper(omega, omega0, r2, aq, vv);
}

mycostfunction_fitnmr_singlet_singlet::~mycostfunction_fitnmr_singlet_singlet(){};

mycostfunction_fitnmr_singlet_singlet::mycostfunction_fitnmr_singlet_singlet(
    int xdim_, int ydim_, double *z_,
    double aqx_, double offx_, double endx_, double nx_,
    double aqy_, double offy_, double endy_, double ny_)
    : mycostfunction_fitnmr_base(xdim_, ydim_, z_, aqx_, offx_, endx_, nx_, aqy_, offy_, endy_, ny_)
{
}

mycostfunction_fitnmr_doublet_singlet::~mycostfunction_fitnmr_doublet_singlet(){};

mycostfunction_fitnmr_doublet_singlet::mycostfunction_fitnmr_doublet_singlet(
    int xdim_, int ydim_, double *z_,
    double aqx_, double offx_, double endx_, double nx_,
    double aqy_, double offy_, double endy_, double ny_)
    : mycostfunction_fitnmr_base(xdim_, ydim_, z_, aqx_, offx_, endx_, nx_, aqy_, offy_, endy_, ny_)
{
}

bool mycostfunction_fitnmr_singlet_singlet::Evaluate(
    double const *const *xx, double *residual, double **jaco) const
{
    const double a = xx[0][0];
    const double x0 = xx[1][0];
    const double r2x = std::fabs(xx[2][0]);
    const double y0 = xx[3][0];
    const double r2y = std::fabs(xx[4][0]);

    double vvx, r_x, r_r2x;
    double vvy, r_y, r_r2y;

    if (jaco != NULL)
    {
        for (int i = 0; i < xdim; i++)
        {
            singlet_helper(i, x0, r2x, nx, aqx, offx, endx, &vvx, &r_x, &r_r2x);

            for (int j2 = 0; j2 < ydim; j2++)
            {
                const int ii = i * ydim + j2;

                singlet_helper(j2, y0, r2y, ny, aqy, offy, endy, &vvy, &r_y, &r_r2y);

                residual[ii] = a * vvx * vvy - z[ii];

                jaco[0][ii] = vvx * vvy;
                jaco[1][ii] = a * r_x * vvy;
                jaco[2][ii] = a * r_r2x * vvy;
                jaco[3][ii] = a * vvx * r_y;
                jaco[4][ii] = a * vvx * r_r2y;
            }
        }
    }
    else
    {
        for (int i = 0; i < xdim; i++)
        {
            singlet_helper(i, x0, r2x, nx, aqx, offx, endx, &vvx);

            for (int j2 = 0; j2 < ydim; j2++)
            {
                const int ii = i * ydim + j2;
                singlet_helper(j2, y0, r2y, ny, aqy, offy, endy, &vvy);
                residual[ii] = a * vvx * vvy - z[ii];
            }
        }
    }

    return true;
}

bool mycostfunction_fitnmr_doublet_singlet::Evaluate(
    double const *const *xx, double *residual, double **jaco) const
{
    const double a = xx[0][0];
    const double x0 = xx[1][0];
    const double r2x = std::fabs(xx[2][0]);
    const double j = std::fabs(xx[3][0]);
    const double y0 = xx[4][0];
    const double r2y = std::fabs(xx[5][0]);

    double vvx, r_x, r_r2x, r_j;
    double vvy, r_y, r_r2y;

    if (jaco != NULL)
    {
        for (int i = 0; i < xdim; i++)
        {
            doublet_helper(i, x0, r2x, j, aqx, offx, endx, &vvx, &r_x, &r_r2x, &r_j);

            for (int j2 = 0; j2 < ydim; j2++)
            {
                const int ii = i * ydim + j2;

                singlet_helper(j2, y0, r2y, ny, aqy, offy, endy, &vvy, &r_y, &r_r2y);

                residual[ii] = a * vvx * vvy - z[ii];

                jaco[0][ii] = vvx * vvy;       // a
                jaco[1][ii] = a * r_x * vvy;   // x0
                jaco[2][ii] = a * r_r2x * vvy; // r2x
                jaco[3][ii] = a * r_j * vvy;   // j
                jaco[4][ii] = a * vvx * r_y;   // y0
                jaco[5][ii] = a * vvx * r_r2y; // r2y
            }
        }
    }
    else
    {
        for (int i = 0; i < xdim; i++)
        {
            doublet_helper(i, x0, r2x, j, aqx, offx, endx, &vvx);

            for (int j2 = 0; j2 < ydim; j2++)
            {
                const int ii = i * ydim + j2;

                singlet_helper(j2, y0, r2y, ny, aqy, offy, endy, &vvy);
                residual[ii] = a * vvx * vvy - z[ii];
            }
        }
    }

    return true;
}
