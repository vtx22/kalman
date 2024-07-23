#ifndef _V_KALMAN_HPP
#define _V_KALMAN_HPP

#include "matrix/matrix.hpp"

#include <cstdint>

template <uint8_t STATES, uint8_t MEASURE_STATES>
class Kalman
{
public:
    Kalman() {}

    void initialize(const Matrix<STATES, STATES> &covariance, const Matrix<MEASURE_STATES, STATES> &H, const Matrix<STATES, 1> &x0)
    {
        _P_n = covariance;
        _H = H;
        _x_n = x0;
    }

    Matrix<STATES, 1> &predict(const Matrix<STATES, STATES> &A, const Matrix<STATES, STATES> &system_noise, const Matrix<STATES, 1> Bu)
    {
        _x_n = A * _x_n + Bu;
        _y_n = _H * _x_n;
        _P_n = A * _P_n * A.transposed() + system_noise;
        return _x_n;
    }

    Matrix<STATES, 1> &predict(const Matrix<STATES, STATES> &A, const Matrix<STATES, STATES> &system_noise)
    {
        return predict(A, system_noise, Matrix<STATES, 1>(ZEROES));
    }

    Matrix<STATES, 1> &correct(const Matrix<MEASURE_STATES, MEASURE_STATES> &measure_noise, const Matrix<MEASURE_STATES, 1> &measurements)
    {
        _K_n = _P_n * _H.transposed() * (_H * _P_n * _H.transposed() + measure_noise).inverse();
        _x_n += _K_n * (measurements - _y_n);
        _P_n = (Matrix<STATES, STATES>(IDENTITY) - _K_n * _H) * _P_n;

        return _x_n;
    }

    Matrix<STATES, 1> &get_x() const
    {
        return _x_n;
    }

private:
    Matrix<STATES, STATES> _P_n;
    Matrix<STATES, 1> _x_n;
    Matrix<MEASURE_STATES, STATES> _H;
    Matrix<MEASURE_STATES, 1> _y_n;
    Matrix<STATES, MEASURE_STATES> _K_n;
};

#endif // _V_KALMAN_HPP