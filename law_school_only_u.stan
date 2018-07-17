
data {
  int<lower = 0> N; // number of observations
  int<lower = 0> K; // number of covariates
  matrix[N, K]   a; // sensitive variables
  real           ugpa[N]; // UGPA
  int            lsat[N]; // LSAT
  //real           zfya[N]; // ZFYA
  //int<lower = 0> pass[N]; // PASS
  real           ugpa0;
  real           eta_u_ugpa;
  vector[K]      eta_a_ugpa;
  real           lsat0;
  real           eta_u_lsat;
  vector[K]      eta_a_lsat;
  //real           eta_u_zfya;
  //vector[K]      eta_a_zfya;
  //real           pass0;
  //real           eta_u_pass;
  //vector[K]      eta_a_pass;
  real           sigma_g;
  
 
}


parameters {

  vector[N] u;

}


model {
  
  u ~ normal(0, 1);

  // have data about these
  ugpa ~ normal(ugpa0 + eta_u_ugpa * u + a * eta_a_ugpa, sigma_g); 
  lsat ~ poisson(exp(lsat0 + eta_u_lsat * u + a * eta_a_lsat)); 
  //zfya ~ normal(eta_u_zfya * u + a * eta_a_zfya,1);
  //pass ~ bernoulli_logit(pass0 + eta_u_pass * u + a * eta_a_pass);
  

}
