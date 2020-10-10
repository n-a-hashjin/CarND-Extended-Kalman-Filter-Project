#include "kalman_filter.h"
#include <iostream>
#include <math.h>

#define PI_ 3.1415926

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in, 
                        Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in, Eigen::MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

// Update KF
void KalmanFilter::Update(const VectorXd &z) {
  
  VectorXd z_pred = H_ * x_;
  
  VectorXd y = z - z_pred;
  MeasurementUpdate(y);
}

// Update EKF
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  
  float rho = sqrt(x_(0)*x_(0)+x_(1)*x_(1));
  float theta = atan2(x_(1),x_(0));
  float rho_dot = (rho < 0.0001) ? 0.5 : (x_(0)*x_(2) + x_(1)*x_(3)) / rho;
  
  VectorXd z_pred(3);
  z_pred << rho, theta, rho_dot;
  
  VectorXd y = z - z_pred;
  for (; y(1) < -PI_; y(1) += 2*PI_) {}
  for (; y(1) > PI_;  y(1) -= 2*PI_) {}
  
  MeasurementUpdate(y);
}

void KalmanFilter::MeasurementUpdate(const Eigen::VectorXd &y) {
  
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}