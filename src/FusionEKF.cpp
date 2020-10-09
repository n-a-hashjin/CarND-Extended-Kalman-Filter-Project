#include "FusionEKF.h"
#include <iostream>
#include "Eigen/Dense"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;
using std::vector;

// Constructor
FusionEKF::FusionEKF()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  
  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;
  
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
  
  VectorXd x_in = VectorXd::Zero(4);
  MatrixXd P_in = MatrixXd::Zero(4,4);
  MatrixXd F_in = MatrixXd::Identity(4,4);
  MatrixXd Q_in = MatrixXd::Zero(4,4);
  ekf_.Init(x_in, P_in, F_in, H_laser_, R_laser_, Q_in);
}

// Destructor
FusionEKF::~FusionEKF() {}

// Run the whole Kalman filter
void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  //Initialization
  if (!is_initialized_)
  {
    
    cout << "EKF: " << endl;
    
    ekf_.x_ << 1, 1, 1, 1;
    
    ekf_.P_ << 1, 0,   0,   0,
               0, 1,   0,   0,
               0, 0, 100,   0,
               0, 0,   0, 100;
    
    ekf_.F_ << 1, 0, 0.05, 0,
               0, 1, 0, 0.05,
               0, 0, 1, 0,
               0, 0, 0, 1;
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
      
      float rho = measurement_pack.raw_measurements_(0);
      float theta = measurement_pack.raw_measurements_(1);
      float rho_dot = measurement_pack.raw_measurements_(2);
      
      float px = rho * cos(theta);
      float py = rho * sin(theta);
      float vx = rho_dot * cos(theta);
      float vy = rho_dot * sin(theta);
      
      ekf_.x_ << px, py, vx, vy;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      
      float px = measurement_pack.raw_measurements_(0);
      float py = measurement_pack.raw_measurements_(1);
      ekf_.x_ << px, py, 0, 0;
    }
    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }
  
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;
  ekf_.F_(0,2) = dt;
  ekf_.F_(1,3) = dt;
  
  float a = dt*dt;
  float b = a*dt/2;
  float c = b*dt/2;

  float noise_ax = 9;
  float noise_ay = 9;
  
  ekf_.Q_ << c*noise_ax, 0, b*noise_ax, 0,
             0, c*noise_ay, 0, b*noise_ay,
             b*noise_ax, 0, a*noise_ax, 0,
             0, b*noise_ay, 0, a*noise_ay;
  
  
  // Prediction
  ekf_.Predict();
  
  // Update
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
    
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);

  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}