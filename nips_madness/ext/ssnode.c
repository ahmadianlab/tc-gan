#include <math.h>

double dot(int dim, double x[restrict dim], double y[restrict dim]) {
  double result = 0;
#pragma omp simd reduction(+:result)
  for (int i = 0; i < dim; ++i){
    result += x[i] * y[i];
  }
  return result;
}

double rate_to_volt(double rate, double k, double n) {
  return pow(rate / k, 1 / n);
}

inline
double io_alin(double v, double r0, double r1, double v0, double k, double n) {
  /* r1 is not used; it's just to match calling convention with io_atanh */
  if (v <= 0) {
    return 0;
  } else if (v <= v0) {
    return k * pow(v, n);
  } else {
    return r0 + k * (r0/v0) * n * (v - v0);
  }
}

inline
double io_atanh(double v, double r0, double r1, double v0, double k, double n) {
  if (v <= 0) {
    return 0;
  } else if (v <= v0) {
    return k * pow(v, n);
  } else {
    return r0 + (r1 - r0) * tanh(n * r0 / (r1 - r0) * (v - v0) / v0);
  }
}

#define ODE_STEP(io_fun, dt_) \
  r1[i] = r0[i] + \
    (- r0[i] + io_fun(dot(dim, W + dim * i, r0) + ext[i], \
                      rate_soft_bound, rate_hard_bound, v0, k, n)) * dt_

int solve_dynamics_asym_linear(
        /* Model parameters: */
        int N, double *W, double *ext, double k, double n,
        double *r0, double *r1,
        double tau_E, double tau_I,
        /* Solver parameters: */
        double dt, int max_iter, double atol,
        double rate_soft_bound, double rate_hard_bound) {
  int dim = 2 * N;
  double *r_tmp;
  double dt_E = dt / tau_E;
  double dt_I = dt / tau_I;
  double v0 = rate_to_volt(rate_soft_bound, k, n);
  int converged;
  for (int step = 0; step < max_iter; ++step){
#pragma omp parallel for schedule(static,1)
    for (int i = 0; i < N; ++i){
      ODE_STEP(io_alin, dt_E);
    }
#pragma omp parallel for schedule(static,1)
    for (int i = N; i < dim; ++i){
      ODE_STEP(io_alin, dt_I);
    }

    converged = 1;
    for (int i = 0; i < dim; ++i){
      if (fabs(r1[i] - r0[i]) >= atol) {
        converged = 0;
        break;
      }
    }

    if (converged) {
      for (int i = 0; i < dim; ++i){
        r0[i] = r1[i];
      }
      return 0;
    }
    r_tmp = r0;
    r0 = r1;
    r1 = r_tmp;
  }
  return 1;
}

int solve_dynamics_asym_tanh(
        /* Model parameters: */
        int N, double *W, double *ext, double k, double n,
        double *r0, double *r1,
        double tau_E, double tau_I,
        /* Solver parameters: */
        double dt, int max_iter, double atol,
        double rate_soft_bound, double rate_hard_bound) {
  int dim = 2 * N;
  double *r_tmp;
  double dt_E = dt / tau_E;
  double dt_I = dt / tau_I;
  double v0 = rate_to_volt(rate_soft_bound, k, n);
  int converged;
  for (int step = 0; step < max_iter; ++step){
#pragma omp parallel for schedule(static,1)
    for (int i = 0; i < N; ++i){
      ODE_STEP(io_atanh, dt_E);
    }
#pragma omp parallel for schedule(static,1)
    for (int i = N; i < dim; ++i){
      ODE_STEP(io_atanh, dt_I);
    }

    converged = 1;
    for (int i = 0; i < dim; ++i){
      if (fabs(r1[i] - r0[i]) >= atol) {
        converged = 0;
        break;
      }
    }

    if (converged) {
      for (int i = 0; i < dim; ++i){
        r0[i] = r1[i];
      }
      return 0;
    }
    r_tmp = r0;
    r0 = r1;
    r1 = r_tmp;
  }
  return 1;
}
