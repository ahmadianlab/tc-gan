#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

typedef struct {
  /* Model parameters: */
  int N;
  double *W, *ext, k, n, tau_E, tau_I;
  double rate_soft_bound, rate_hard_bound, v0;
} SSNParam;

double dot(int dim,
           const double x[restrict dim],
           const double y[restrict dim]) {
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
double io_pow(double v, double r0, double r1, double v0, double k, double n) {
  /* unused args for just to match calling convention with io_atanh */
  if (v <= 0) {
    return 0;
  } else {
    return k * pow(v, n);
  }
}

inline
double io_alin(double v, double r0, double r1, double v0, double k, double n) {
  /* r1 is not used; it's just to match calling convention with io_atanh */
  if (v <= 0) {
    return 0;
  } else if (v <= v0) {
    return k * pow(v, n);
  } else {
    return r0 + pow(v0, n-1) * n * (v - v0);
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

#define dYdT(io_fun, tau) \
  (- y[i] + io_fun(dot(dim, ssn->W + dim * i, y) + ssn->ext[i], \
                   ssn->rate_soft_bound, ssn->rate_hard_bound,  \
                   ssn->v0, ssn->k, ssn->n)) / tau

int dydt_asym_tanh(double t, const double y[], double dydt[], void *params) {
  const SSNParam *ssn = params;
  int dim = ssn->N * 2;
  for (int i = 0; i < ssn->N; ++i){
    dydt[i] = dYdT(io_atanh, ssn->tau_E);
  }
  for (int i = ssn->N; i < dim; ++i){
    dydt[i] = dYdT(io_atanh, ssn->tau_I);
  }
  return GSL_SUCCESS;
}

int dydt_asym_linear(double t, const double y[], double dydt[], void *params) {
  const SSNParam *ssn = params;
  int dim = ssn->N * 2;
  for (int i = 0; i < ssn->N; ++i){
    dydt[i] = dYdT(io_alin, ssn->tau_E);
  }
  for (int i = ssn->N; i < dim; ++i){
    dydt[i] = dYdT(io_alin, ssn->tau_I);
  }
  return GSL_SUCCESS;
}

#define COMMON_ARG_DEF \
        /* Model parameters: */ \
        int N, double *W, double *ext, double k, double n, \
        double *r0, double *r1, \
        double tau_E, double tau_I, \
        /* Solver parameters: */ \
        double dt, int max_iter, double atol, \
        double rate_soft_bound, double rate_hard_bound

int solve_dynamics_gsl(
        COMMON_ARG_DEF,
        int (* dydt) (double t, const double y[], double dydt[], void * params)) {
  SSNParam ssn;
  gsl_odeiv2_system sys = {dydt, NULL, 2 * N, &ssn};
  gsl_odeiv2_driver *driver =
    gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_msadams,
                                  dt, 1e-6, 0.0);
  int dim = 2 * N;
  int converged;
  int error_code = 1;

  ssn.N = N;
  ssn.W = W;
  ssn.ext = ext;
  ssn.k = k;
  ssn.n = n;
  ssn.tau_E = tau_E;
  ssn.tau_I = tau_I;
  ssn.rate_soft_bound = rate_soft_bound;
  ssn.rate_hard_bound = rate_hard_bound;
  ssn.v0 = rate_to_volt(rate_soft_bound, k, n);

  for (int i = 0; i < dim; ++i){
    r1[i] = r0[i];
  }

  for (int step = 0; step < max_iter; ++step){
    for (int i = 0; i < dim; ++i){
      r0[i] = r1[i];
    }

    double t = 0;
    int status = gsl_odeiv2_driver_apply(driver, &t, 0.1, r1);
    if (status != GSL_SUCCESS) {
      error_code = 1000 + status;
      goto end;
    }

    converged = 1;
    for (int i = 0; i < dim; ++i){
      if (fabs(r1[i] - r0[i]) >= atol) {
        converged = 0;
        break;
      }
    }

    if (converged) {
      error_code = 0;
      goto end;
    }
  }
 end:
  gsl_odeiv2_driver_free(driver);
  return error_code;
}

int solve_dynamics_asym_linear_gsl(COMMON_ARG_DEF) {
  return solve_dynamics_gsl(N, W, ext, k, n, r0, r1, tau_E, tau_I,
                            dt, max_iter, atol,
                            rate_soft_bound, rate_hard_bound,
                            dydt_asym_linear);
}

int solve_dynamics_asym_tanh_gsl(COMMON_ARG_DEF) {
  return solve_dynamics_gsl(N, W, ext, k, n, r0, r1, tau_E, tau_I,
                            dt, max_iter, atol,
                            rate_soft_bound, rate_hard_bound,
                            dydt_asym_tanh);
}

#define ODE_STEP(io_fun, dt_) \
  r1[i] = r0[i] + \
    (- r0[i] + io_fun(dot(dim, W + dim * i, r0) + ext[i], \
                      rate_soft_bound, rate_hard_bound, v0, k, n)) * dt_

int solve_dynamics_asym_power_euler(COMMON_ARG_DEF) {
  int dim = 2 * N;
  double *r_tmp;
  double dt_E = dt / tau_E;
  double dt_I = dt / tau_I;
  double v0 = rate_to_volt(rate_soft_bound, k, n);

  for (int step = 0; step < max_iter; ++step){
    for (int i = 0; i < N; ++i){
      ODE_STEP(io_pow, dt_E);
    }
    for (int i = N; i < dim; ++i){
      ODE_STEP(io_pow, dt_I);
    }

    int converged = 1;
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

    for (int i = 0; i < dim; ++i){
      if (r1[i] >= rate_hard_bound) {
        return 2;
      }
    }

    r_tmp = r0;
    r0 = r1;
    r1 = r_tmp;
  }
  return 1;
}

int solve_dynamics_asym_linear_euler(COMMON_ARG_DEF) {
  int dim = 2 * N;
  double *r_tmp;
  double dt_E = dt / tau_E;
  double dt_I = dt / tau_I;
  double v0 = rate_to_volt(rate_soft_bound, k, n);

  for (int step = 0; step < max_iter; ++step){
    for (int i = 0; i < N; ++i){
      ODE_STEP(io_alin, dt_E);
    }
    for (int i = N; i < dim; ++i){
      ODE_STEP(io_alin, dt_I);
    }

    int converged = 1;
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

    for (int i = 0; i < dim; ++i){
      if (r1[i] >= rate_hard_bound) {
        return 2;
      }
    }

    r_tmp = r0;
    r0 = r1;
    r1 = r_tmp;
  }
  return 1;
}

int solve_dynamics_asym_tanh_euler(COMMON_ARG_DEF) {
  int dim = 2 * N;
  double *r_tmp;
  double dt_E = dt / tau_E;
  double dt_I = dt / tau_I;
  double v0 = rate_to_volt(rate_soft_bound, k, n);

  for (int step = 0; step < max_iter; ++step){
    for (int i = 0; i < N; ++i){
      ODE_STEP(io_atanh, dt_E);
    }
    for (int i = N; i < dim; ++i){
      ODE_STEP(io_atanh, dt_I);
    }

    int converged = 1;
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
