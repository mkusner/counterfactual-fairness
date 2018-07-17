
data {
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of covariates
  matrix[N, K]   a; // sensitive variables
  real           ugpa[N]; // UGPA
  int            lsat[N]; // LSAT
  real           zfya[N]; // ZFYA
  
}

transformed data {
  
 vector[K] zero_K;
 vector[K] one_K;
 
 zero_K = rep_vector(0,K);
 one_K = rep_vector(1,K);

}

parameters {

  vector[N] u;

  real ugpa0;
  real eta_u_ugpa;
  real lsat0;
  real eta_u_lsat;
  real eta_u_zfya;
  
  vector[K] eta_a_ugpa;
  vector[K] eta_a_lsat;
  vector[K] eta_a_zfya;
  
  
  real<lower=0> sigma_g_Sq;
}

transformed parameters  {
 // Population standard deviation (a positive real number)
 real<lower=0> sigma_g;
 // Standard deviation (derived from variance)
 sigma_g = sqrt(sigma_g_Sq);
}

model {
  
  // don't have data about this
  u ~ normal(0, 1);
  
  ugpa0      ~ normal(0, 1);
  eta_u_ugpa ~ normal(0, 1);
  lsat0     ~ normal(0, 1);
  eta_u_lsat ~ normal(0, 1);
  eta_u_zfya ~ normal(0, 1);

  eta_a_ugpa ~ normal(zero_K, one_K);
  eta_a_lsat ~ normal(zero_K, one_K);
  eta_a_zfya ~ normal(zero_K, one_K);

  sigma_g_Sq ~ inv_gamma(1, 1);

  // have data about these
  ugpa ~ normal(ugpa0 + eta_u_ugpa * u + a * eta_a_ugpa, sigma_g);
  lsat ~ poisson(exp(lsat0 + eta_u_lsat * u + a * eta_a_lsat));
  zfya ~ normal(eta_u_zfya * u + a * eta_a_zfya, 1);

}
