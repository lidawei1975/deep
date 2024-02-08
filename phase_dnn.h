//phase_dnn header

#include <vector>
#include "dnn_base.h"

#ifndef PHASE_DNN_H
#define PHASE_DNN_H


class phase_dnn
{
    public:
        phase_dnn();
        ~phase_dnn();

    protected:

    /**
     * @brief Define validation_dnn here
     * The order or layers are same as order defination here
     */
    conv1d validation_dnn_c1, validation_dnn_c2, validation_dnn_c3;
    pool1d validation_pool1;
    conv1d validation_dnn_c4;
    pool1d validation_pool2;
    dense validation_dnn_d1, validation_dnn_d2, validation_dnn_d3;

    /**
     * @brief Define phasing_dnn here
     * The order of layers are same as order defination here
     */
    conv1d phase_dnn_c1, phase_dnn_c2, phase_dnn_c3;
    pool1d phase_pool1;
    conv1d phase_dnn_c4;
    pool1d phase_pool2;
    dense phase_dnn_d1, phase_dnn_d2, phase_dnn_d3;

    bool has_phase_error(std::vector<float> spectrum_part) const;
    bool is_valid(std::vector<float> spectrum_part) const;


    private:
};

#endif // PHASE_DNN_H