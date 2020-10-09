#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  
  if (estimations.size() == 0){
      cout << "::Error:: Estimations size is zero!" << endl;
  }
  else if (estimations.size() != ground_truth.size()){
      cout << "::Error:: estimation missmatches ground_truth!" << endl;
  }
  else{
      for (unsigned int i=0; i < estimations.size(); ++i) {
        VectorXd residual = estimations[i]-ground_truth[i];
        residual = residual.array()*residual.array();
        rmse += residual;
      }
  }
  // mean of residual
  rmse = rmse/estimations.size();
  // root mean square
  rmse = rmse.array().sqrt();
  //cout << "rmse:  " << rmse << endl;
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  
  MatrixXd Hj(3,4);
  
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  // pre-compute a set of terms to avoid repeated calculation
  float a = px*px+py*py;
  float b = sqrt(a);
  float c = (a*b);
  
  // check division by zero
  if (fabs(a) < 0.0001) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }
  
  // compute the Jacobian matrix
  Hj << (px/b), (py/b), 0, 0,
       -(py/a), (px/a), 0, 0,
        py*(vx*py - vy*px)/c, px*(px*vy - py*vx)/c, px/b, py/b;
  
  return Hj;
}
