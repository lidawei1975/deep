#include "phase_dnn.h"
#include "phase_data.h"

phase_dnn::phase_dnn()
{
    /**
     * @brief initialize validation_dnn here
     * First one always have 1 input channel (from 1D spectrum)
     */
    int n = 0;
    validation_dnn_c1.set_size(21, 1, 10);                // nkernel, ninput, nfilter
    n += validation_dnn_c1.read(validation_dnn_data + n); // 220

    validation_dnn_c2.set_size(11, 10, 6);
    n += validation_dnn_c2.read(validation_dnn_data + n); // 666

    validation_dnn_c3.set_size(11, 6, 4);
    n += validation_dnn_c3.read(validation_dnn_data + n); // 268

    validation_pool1.set_size(4, 3); // ninput,npool

    validation_dnn_c4.set_size(11, 4, 4);
    n += validation_dnn_c4.read(validation_dnn_data + n); // 180

    validation_pool2.set_size(4, 3); // ninput,npool

    /**
     * There is a flatten layer here. Which is not implemented explicitly
     * We need to reshape the output from previous layer to a 1D array
     */

    validation_dnn_d1.set_size(2000, 10); // ninput,noutput
    validation_dnn_d1.set_act(activation_function::relu);
    n += validation_dnn_d1.read(validation_dnn_data + n); // 20010

    validation_dnn_d2.set_size(10, 5);
    validation_dnn_d2.set_act(activation_function::relu);
    n += validation_dnn_d2.read(validation_dnn_data + n); // 55

    validation_dnn_d3.set_size(5, 2);
    validation_dnn_d3.set_act(activation_function::softmax);
    n += validation_dnn_d3.read(validation_dnn_data + n); // 12

    /**
     * @brief initialize phase_dnn here
     * The two DNNs have identical structure (with different weights and biases)
     * First one always have 1 input channel (from 1D spectrum)
     */
    n = 0;
    phase_dnn_c1.set_size(21, 1, 10);           // nkernel, ninput, nfilter
    n += phase_dnn_c1.read(phase_dnn_data + n); // 220

    phase_dnn_c2.set_size(11, 10, 6);
    n += phase_dnn_c2.read(phase_dnn_data + n); // 666

    phase_dnn_c3.set_size(11, 6, 4);
    n += phase_dnn_c3.read(phase_dnn_data + n); // 268

    phase_pool1.set_size(4, 3); // ninput,npool

    phase_dnn_c4.set_size(11, 4, 4);
    n += phase_dnn_c4.read(phase_dnn_data + n); // 180

    phase_pool2.set_size(4, 3); // ninput,npool

    /**
     * There is a flatten layer here. Which is not implemented explicitly
     * We need to reshape the output from previous layer to a 1D array
     */

    phase_dnn_d1.set_size(2000, 10); // ninput,noutput
    phase_dnn_d1.set_act(activation_function::relu);
    n += phase_dnn_d1.read(phase_dnn_data + n); // 20010

    phase_dnn_d2.set_size(10, 5);
    phase_dnn_d2.set_act(activation_function::relu);
    n += phase_dnn_d2.read(phase_dnn_data + n); // 55

    phase_dnn_d3.set_size(5, 2);
    phase_dnn_d3.set_act(activation_function::softmax);
    n += phase_dnn_d3.read(phase_dnn_data + n); // 12

};

phase_dnn::~phase_dnn()
{
};


/**
 * @brief Apply DNN to check spectral data of 500 pixels on the left and right side of the peak
 *
 * @param y: input spectrum to be checked
 * @return true: if the phase error can be detected (input spectrum is valid)
 * @return false: if the phase error can not be detected (input spectrum is invalid)
 */
bool phase_dnn::is_valid(std::vector<float> y) const
{

    bool b_phase_error = true;
    /**
     * @brief y[0] should have the largest value
     * Any peak after y[0] should < 0.7*y[0], if not, return false (cannot determine whether phase is correct or not)
     */
    for (int i = 4; i < y.size(); i++)
    {
        // if y[i] is a peak (local maximum) and y[i] > 0.4*y[0], return false
        if (y[i] > y[i - 1] && y[i] > y[i + 1] && y[i] > 0.4 * y[0])
        {
            b_phase_error = false;
            break;
        }
    }

    if (b_phase_error == false) // no need to check phase
    {
        return false;
    }

    y.resize(500); // only check 500 pixels for DNN

    /**
     * @brief  apply validation DNN to check if the phase can be determined or not
     * false means cannot determine the phase
     * See spectrum_phasing_1d constructor for details of the DNN
     */
    std::vector<float> t1, t2, t3, t4, t5, t6, t7, t8, output;

    // y is 1*500*1

    validation_dnn_c1.predict(500, y, t1);  // 1*500*10
    validation_dnn_c2.predict(500, t1, t2); // 1*500*6
    validation_dnn_c3.predict(500, t2, t3); // 1*500*4
    validation_pool1.predict(500, t3, t4);  // 1*500*4
    validation_dnn_c4.predict(500, t4, t5); // 1*500*4
    validation_pool2.predict(500, t5, t6);  // 1*500*4

    // flattern t6 from 1*500*4 to 1*2000 implicitly without any operation

    validation_dnn_d1.predict(1, t6, t7);     // 1*10
    validation_dnn_d2.predict(1, t7, t8);     // 1*5
    validation_dnn_d3.predict(1, t8, output); // size of t9 is 2

    if (output[1] > output[0]) // valid
    {
        return true;
    }
    else
    {
        return false;
    }
};

/**
 * @brief Apply DNN to check spectral data of 500 pixels on the left and right side of the peak
 *
 * @param y input spectrum to be checked
 * @return true if phase is incorrect
 * @return false if phase is correct
 */

bool phase_dnn::has_phase_error(std::vector<float> y) const
{
    /**
     * @brief apply phase DNN to determine the phase is correct or not
     * false means phase is correct
     */
    std::vector<float> t1, t2, t3, t4, t5, t6, t7, t8, output;

    // y is 1*500*1

    phase_dnn_c1.predict(500, y, t1);  // 1*500*10
    phase_dnn_c2.predict(500, t1, t2); // 1*500*6
    phase_dnn_c3.predict(500, t2, t3); // 1*500*4
    phase_pool1.predict(500, t3, t4);  // 1*500*4
    phase_dnn_c4.predict(500, t4, t5); // 1*500*4
    phase_pool2.predict(500, t5, t6);  // 1*500*4

    // flattern t6 from 1*500*4 to 1*2000 implicitly without any operation

    phase_dnn_d1.predict(1, t6, t7);     // 1*10
    phase_dnn_d2.predict(1, t7, t8);     // 1*5
    phase_dnn_d3.predict(1, t8, output); // size of t9 is 2

    if (output[1] > output[0]) // has phase error
    {
        return true;
    }
    else
    {
        return false;
    }
};