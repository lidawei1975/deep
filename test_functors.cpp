#include <vector>
#include <valarray>
#include <iostream>



#include "cost_functors.h"

int test_voigt_lorentz()
{
    int return_code = 0; // 0 means success
    /**
     * Test mycostfunction_voigt_lorentz
     * zdata is NOT used in the test of Jacobian implementation
     */
    double zdata[25];

    for(int i = 0; i < 25; i++)
    {
        zdata[i] = 0.0;
    }

    mycostfunction_voigt_lorentz x(5,5,zdata);

    double **params = new double*[6];
    for (int i = 0; i < 6; i++)
    {
        params[i] = new double[1];
    }

    double *residuals = new double[25];
    double *residuals2 = new double[25];


    double **jacobian = new double*[6];

    for (int i = 0; i < 6; i++)
    {
        jacobian[i] = new double[25];
    }

    double delta = 1e-8; // finite difference step size to test Jacobian against numerical derivative

    params[0][0] = 1.0; //a
    params[1][0] = 3.1; //x0
    params[2][0] = 2.2; //sigma x
    params[3][0] = 1.3; //gamma x
    params[4][0] = 2.4; //y0
    params[5][0] = 2.5; //gamma y

    for(int i=0;i<6;i++)
    {
        params[i][0] += delta;
        x.Evaluate(params, residuals2, jacobian);
        params[i][0] -= delta; // reset the value
        x.Evaluate(params, residuals, jacobian);

        /**
         * Test Jacobian implementation
         * (residuals2-residuals)/delta should be close to the jacobian[i]
         */
        for(int j=0;j<25;j++)
        {
            if (std::abs((residuals2[j]-residuals[j])/delta - jacobian[i][j]) > 1e-5)
            {
                std::cout << "Jacobian test failed for parameter " << i << " at index " << j << std::endl;
                std::cout << "Jacobian: " << jacobian[i][j] << " Numerical derivative: " << (residuals2[j]-residuals[j])/delta << std::endl;
                return_code = 1;
            }
        }
    }
    
    for (int i = 0; i < 6; i++)
    {
        delete[] params[i];
    }
    delete[] params;

    delete[] residuals;
    delete[] residuals2;

    for (int i = 0; i < 6; i++)
    {
        delete[] jacobian[i];
    }
    delete[] jacobian;

    std::cout<<"Test voigt-lorentz done."<<std::endl;
    return return_code;
}


int test_voigt()
{
    int return_code = 0; // 0 means success
    /**
     * Test mycostfunction_voigt_lorentz
     * zdata is NOT used in the test of Jacobian implementation
     */
    double zdata[25];

    for(int i = 0; i < 25; i++)
    {
        zdata[i] = 0.0;
    }

    mycostfunction_voigt x(5,5,zdata);

    double **params = new double*[7];
    for (int i = 0; i < 7; i++)
    {
        params[i] = new double[1];
    }

    double *residuals = new double[25];
    double *residuals2 = new double[25];


    double **jacobian = new double*[7];

    for (int i = 0; i < 7; i++)
    {
        jacobian[i] = new double[25];
    }

    double delta = 1e-8; // finite difference step size to test Jacobian against numerical derivative

    params[0][0] = 1.0; //a
    params[1][0] = 3.1; //x0
    params[2][0] = 2.2; //sigma x
    params[3][0] = 1.3; //gamma x
    params[4][0] = 2.4; //y0
    params[5][0] = 2.5; //sigma y
    params[6][0] = 2.6; //gamma y

    for(int i=0;i<7;i++)
    {
        params[i][0] += delta;
        x.Evaluate(params, residuals2, jacobian);
        params[i][0] -= delta; // reset the value
        x.Evaluate(params, residuals, jacobian);

        /**
         * Test Jacobian implementation
         * (residuals2-residuals)/delta should be close to the jacobian[i]
         */
        for(int j=0;j<25;j++)
        {
            if (std::abs((residuals2[j]-residuals[j])/delta - jacobian[i][j]) > 1e-5)
            {
                std::cout << "Jacobian test failed for parameter " << i << " at index " << j << std::endl;
                std::cout << "Jacobian: " << jacobian[i][j] << " Numerical derivative: " << (residuals2[j]-residuals[j])/delta << std::endl;
                return_code = 1;
            }
        }
    }
    
    for (int i = 0; i < 6; i++)
    {
        delete[] params[i];
    }
    delete[] params;

    delete[] residuals;
    delete[] residuals2;

    for (int i = 0; i < 6; i++)
    {
        delete[] jacobian[i];
    }
    delete[] jacobian;

    std::cout<<"Test voigt done."<<std::endl;
    return return_code;
}

int test_lorentz_1d()
{
    int return_code = 0; // 0 means success
    /**
     * Test mycostfunction_voigt_lorentz
     * zdata is NOT used in the test of Jacobian implementation
     */
    double zdata[5];

    for(int i = 0; i < 5; i++)
    {
        zdata[i] = 0.0;
    }

    mycostfunction_lorentz1d x(5,zdata);

    double **params = new double*[3];
    for (int i = 0; i < 3; i++)
    {
        params[i] = new double[1];
    }

    double *residuals = new double[5];
    double *residuals2 = new double[5];

    double **jacobian = new double*[3];
    for(int i=0;i<3;i++)
    {
        jacobian[i] = new double[5];
    }

    double delta = 1e-8; // finite difference step size to test Jacobian against numerical derivative

    params[0][0] = 1.0; //a
    params[1][0] = 3.1; //x0
    params[2][0] = 2.2; //sigma x

    for(int i=0;i<3;i++)
    {
        params[i][0] += delta;
        x.Evaluate(params, residuals2, jacobian);
        params[i][0] -= delta; // reset the value
        x.Evaluate(params, residuals, jacobian);

        /**
         * Test Jacobian implementation
         * (residuals2-residuals)/delta should be close to the jacobian[i]
         */
        for(int j=0;j<5;j++)
        {
            if (std::abs((residuals2[j]-residuals[j])/delta - jacobian[i][j]) > 1e-5)
            {
                std::cout << "Jacobian test failed for parameter " << i << " at index " << j << std::endl;
                std::cout << "Jacobian: " << jacobian[i][j] << " Numerical derivative: " << (residuals2[j]-residuals[j])/delta << std::endl;
                return_code = 1;
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        delete[] params[i];
    }
    delete[] params;

    delete[] residuals;
    delete[] residuals2;

    for (int i = 0; i < 3; i++)
    {
        delete[] jacobian[i];
    }
    delete[] jacobian;

    std::cout<<"Test lorentz 1d done."<<std::endl;
    return return_code;
}

int test_voigt1d_approx()
{
    int return_code = 0; // 0 means success
    /**
     * Test mycostfunction_voigt_lorentz
     * zdata is NOT used in the test of Jacobian implementation
     */
    double zdata[15];

    for(int i = 0; i < 15; i++)
    {
        zdata[i] = 0.0;
    }

    mycostfunction_voigt1d_approx x(15,zdata);

    double **params = new double*[4];
    for (int i = 0; i < 4; i++)
    {
        params[i] = new double[1];
    }

    double *residuals = new double[15];
    double *residuals2 = new double[15];

    double **jacobian = new double*[4];
    for(int i=0;i<4;i++)
    {
        jacobian[i] = new double[15];
    }

    double delta = 1e-8; // finite difference step size to test Jacobian against numerical derivative

    params[0][0] = 1.3; //a
    params[1][0] = 3.1; //x0
    params[2][0] = 2.2; //sigma x
    params[3][0] = 1.3; //gamma x

    for(int i=0;i<4;i++)
    {
        params[i][0] += delta;
        x.Evaluate(params, residuals2, jacobian);
        params[i][0] -= delta; // reset the value
        x.Evaluate(params, residuals, jacobian);

        /**
         * Test Jacobian implementation
         * (residuals2-residuals)/delta should be close to the jacobian[i]
         */
        for(int j=0;j<15;j++)
        {
            if (std::abs((residuals2[j]-residuals[j])/delta - jacobian[i][j]) > 1e-5)
            {
                std::cout << "Jacobian test failed for parameter " << i << " at index " << j << std::endl;
                std::cout << "Jacobian: " << jacobian[i][j] << " Numerical derivative: " << (residuals2[j]-residuals[j])/delta << std::endl;
                return_code = 1;
            }
        }
    }

    for (int i=0; i<4; i++)
    {
        delete[] params[i];
    }
    delete[] params;

    delete[] residuals;
    delete[] residuals2;

    for (int i=0; i<4; i++)
    {
        delete[] jacobian[i];
    }
    delete[] jacobian;

    std::cout<<"Test voigt1d_approx done."<<std::endl;
    return return_code;
}

int test_voigt_1d_3peaks()
{
    int return_code = 0; // 0 means success
    /**
     * Test mycostfunction_3voigt1d
     * zdata is NOT used in the test of Jacobian implementation
     */
    double zdata[15];
    for(int i = 0; i < 15; i++)
    {
        zdata[i] = 0.0;
    }
    mycostfunction_3voigt1d x(15,zdata);
    double **params = new double*[3];
    for (int i = 0; i < 3; i++)
    {
        params[i] = new double[4];
    }

    double *residuals = new double[15];
    double *residuals2 = new double[15];
    double **jacobian = new double*[3];
    for(int i=0;i<3;i++)
    {
        jacobian[i] = new double[15*4];
    }

    double delta = 1e-8; // finite difference step size to test Jacobian against numerical derivative
    params[0][0] = 1.3; //a
    params[0][1] = 3.1; //x0
    params[0][2] = 2.2; //sigma x
    params[0][3] = 1.3; //gamma x
    params[1][0] = 0.9; //a
    params[1][1] = 0.4; //x0, relative position to peak 1
    params[1][2] = 0.2; //sigma x, relative sigma to peak 1
    params[1][3] = -0.3; //gamma x, relative gamma to peak 1
    params[2][0] = 0.8; //a
    params[2][1] = -0.5; //x0, relative position to peak 1
    params[2][2] = -0.3; //sigma x, relative sigma to peak 1
    params[2][3] = 0.4; //gamma x, relative gamma to peak 1

    for(int i=0;i<3;i++)
    {
        for(int j=0;j<4;j++)
        {
            params[i][j] += delta;
            x.Evaluate(params, residuals2, jacobian);
            params[i][j] -= delta; // reset the value
            x.Evaluate(params, residuals, jacobian);

            /**
             * Test Jacobian implementation
             * (residuals2-residuals)/delta should be close to the jacobian[i]
             */
            for(int k=0;k<15;k++)
            {
                if (std::abs((residuals2[k]-residuals[k])/delta - jacobian[i][k*4+j]) > 1e-5)
                {
                    std::cout << "Jacobian test failed for parameter " << i << " at index " << k << std::endl;
                    std::cout << "Jacobian: " << jacobian[i][k*4+j] << " Numerical derivative: " << (residuals2[k]-residuals[k])/delta << std::endl;
                    return_code = 1;
                }
            }
        }
    }

    for (int i = 0; i < 3; i++)
    {
        delete[] params[i];
    }
    delete[] params;
    delete[] residuals;
    delete[] residuals2;

    for (int i = 0; i < 3; i++)
    {
        delete[] jacobian[i];
    }
    delete[] jacobian;
    std::cout<<"Test 3 peaks voigt1d done."<<std::endl;
    return return_code;
}


int main()
{
    /**
     * Run the tests, return 0 if all tests pass, 1 if one fail, 2 if 2 fail, etc.
    */
    return test_voigt_lorentz()+test_voigt()+test_lorentz_1d()+test_voigt1d_approx()+test_voigt_1d_3peaks();
}