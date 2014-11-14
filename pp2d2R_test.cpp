/* Trials with CHOMP.
 *
 * Copyright (C) 2014 Roland Philippsen. All rights reserved.
 *
 * BSD license:
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of
 *    contributors to this software may be used to endorse or promote
 *    products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR THE CONTRIBUTORS TO THIS SOFTWARE BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
   \file pp2d.cpp

   Interactive trials with CHOMP for point vehicles moving
   holonomously in the plane.  There is a fixed start and goal
   configuration, and you can drag a circular obstacle around to see
   how the CHOMP algorithm reacts to that.  Some of the computations
   involve guesswork, for instance how best to compute velocities, so
   a simple first-order scheme has been used.  This appears to produce
   some unwanted drift of waypoints from the start configuration to
   the end configuration.  Parameters could also be tuned a bit
   better.  Other than that, it works pretty nicely.
*/

//////////////////////////////////////////////////
/* 21.10.14 pb
    Test environment for implementation of multiple robot case.
   22.10.14 pb
    Try to implement a second obstacle
   23.10.14 pb
    This is the updated and improved version of pp2d_2obs_sepobj. In contrast to these,
    here the second obstacle is integrated to the gradient of the obstacle objective of
    the first obstacle. Thus it is similar to the version presented in the paper. It behaves
    identical to the other version with seperable obstacle objectives. But it is less
    computational demanding and thus I affect this version.
    (pp2d_2obs_oneobj = two obstacles, one obstacle objective)
   25.10.14 pb
    It should be possible to move the start and end points as well. This is implemented here.
   27.10.14 pb
    Finally implemented the possibility to move the start and end points. Aside that the circle
    surrounds the point one iteration step later, everything works quite nicely. Points are now
    highlighted with corresponding color.
    (pp2d2R_ movpoi = two Robots, two obstacles, MOVable start and end POInts)
*/
//////////////////////////////////////////////////

#include "gfx.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdlib.h>
#include <sys/time.h>
#include <err.h>

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::Isometry3d Transform;

using namespace std;
//

//////////////////////////////////////////////////
// trajectory etc

Vector xi;			// the trajectory (q_1, q_2, ...q_n)
static size_t const nq (2*20);	// number of q stacked into xi
static size_t const cdim (2);	// dimension of config space
static size_t const xidim (nq * cdim); // dimension of trajectory, xidim = nq * cdim
static double const dt (1.0);	       // time step
static double const eta (100.0);    // >= 1, regularization factor for gradient descent
static double const lambda (1.0);   // weight of smoothness objective
static double const mu (0.4);       // weight of interference objective  -> 20.10.14: better performance (less oscillation) with mu < 1

//////////////////////////////////////////////////
// gradient descent etc

Matrix AA;			// metric
Vector bb;			// acceleration bias for start and end config
Matrix Ainv;        // inverse of AA
Matrix ZZ;          // zero matrix for "B" and "C" blocks in a big matrix
Matrix AAR;         // stacked up matrix for multiple robots case
Vector bbR;         // stacked up vector for multiple robots case
Matrix AARinv;      // inverse of AAR

//////////////////////////////////////////////////
// gui stuff

enum { PAUSE, STEP, RUN } state;

struct handle_s {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  handle_s (double radius, double red, double green, double blue, double alpha)
    : point_(2),
      radius_(radius),
      red_(red),
      green_(green),
      blue_(blue),
      alpha_(alpha)
  {
  }

  Vector point_;
  double radius_, red_, green_, blue_, alpha_;
};

static handle_s repulsor (1.5, 0.0, 0.0, 1.0, 0.2);         // initialize first obstacle
static handle_s repulsor2 (1.5, 0.0, 0.0, 1.0, 0.2);        // initialize second obstacle

static handle_s startp1 (1.0, 0.0, 0.0, 1.0, 0.2);        // initialize starting point of first robot
static handle_s startp2 (1.0, 0.0, 0.0, 1.0, 0.2);        // initialize starting point of second robot
static handle_s endp1 (1.0, 0.0, 0.0, 1.0, 0.2);        // initialize end point of first robot
static handle_s endp2 (1.0, 0.0, 0.0, 1.0, 0.2);        // initialize end point of second robot

static handle_s * handle[] = { &repulsor, 0 };
static handle_s * handle2[] = { &repulsor2, 0 };

static handle_s * handle_sp1[] = { &startp1, 0 };       // here for every start/end point a handle is introduced
static handle_s * handle_sp2[] = { &startp2, 0 };
static handle_s * handle_ep1[] = { &endp1, 0 };
static handle_s * handle_ep2[] = { &endp2, 0 };

static handle_s * grabbed (0);
static Vector grab_offset (3);


//////////////////////////////////////////////////
// robot (one per waypoint)

class Robot
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Robot ()
    : position_ (Vector::Zero(2))
  {
  }


  void update (Vector const & position)
  {
    if (position.size() != 2) {
      errx (EXIT_FAILURE, "Robot::update(): position has %zu DOF (but needs 2)",
        (size_t) position.size());
    }
    position_ = position;
  }


  void draw () const                // settings for trajectories drawing
  {
    // translucent disk for base
    gfx::set_pen (1.0, 0.7, 0.7, 0.7, 0.5);
    gfx::fill_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);

    // thick circle outline for base
    gfx::set_pen (3.0, 0.2, 0.2, 0.2, 1.0);
    gfx::draw_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);
  }


  void draw_sp () const            // function for drawing start points
  {
    // translucent disk for base
    gfx::set_pen (1.0, 0.7, 0.2, 0.2, 0.5);
    gfx::fill_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);

    // thick circle outline for base
    gfx::set_pen (3.0, 0.2, 0.2, 0.2, 1.0);
    gfx::draw_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);
  }


  void draw_ep () const            // function for drawing end points
  {
    // translucent disk for base
    gfx::set_pen (1.0, 0.2, 0.7, 0.2, 0.5);
    gfx::fill_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);

    // thick circle outline for base
    gfx::set_pen (3.0, 0.2, 0.2, 0.2, 1.0);
    gfx::draw_arc (position_[0], position_[1], radius_, 0.0, 2.0 * M_PI);
  }

  static double const radius_;

  Vector position_;
};

double const Robot::radius_ (0.5);

Robot rstart;
Robot rstart2;      // start point of second robot
Robot rend;
Robot rend2;        // end point of second robot
vector <Robot> robots;


//////////////////////////////////////////////////

static void update_robots ()
{
  rstart.update (startp1.point_);
  rstart2.update (startp2.point_);
  rend.update (endp1.point_);
  rend2.update (endp2.point_);                      //// der letzte Punkt muss das qi des zweiten roboters sein..
  if (nq != robots.size()) {
    robots.resize (nq);
  }
  for (size_t ii (0); ii < nq; ++ii) {
    robots[ii].update (xi.block (ii * cdim, 0, cdim, 1));
  }
}


static void init_chomp ()
{
  repulsor.point_ << 4.0, 0.0;
  repulsor2.point_ << -4.0, 0.0;        // initial position of second obstacle

  startp1.point_ << -5.0, -5.0;         // initialize start and end points of both robots
  endp1.point_ << 7.0, 7.0;
  startp2.point_ << 5.0, -5.0;
  endp2.point_ << -7.0, 7.0;

  xi = Vector::Zero (xidim);
  for (size_t ii(0); ii < nq; ++ii) {
      if (ii < nq/2) {
          xi.block (ii * cdim, 0, cdim, 1) = startp1.point_;
      }
      else {
          xi.block (ii * cdim, 0, cdim, 1) = startp2.point_;
      }
  }

  // cout << "qs1\n" << qs1
  //      << "\nxi\n" << xi
  //      << "\nqe1\n" << qe1 << "\n\n";

  AA = Matrix::Zero (xidim/2, xidim/2);
  for (size_t ii(0); ii < nq/2; ++ii) {
    AA.block (cdim * ii, cdim * ii, cdim , cdim) = 2.0 * Matrix::Identity (cdim, cdim);
    if (ii > 0) {
      AA.block (cdim * (ii-1), cdim * ii, cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
      AA.block (cdim * ii, cdim * (ii-1), cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
    }
  }

  ZZ = Matrix::Zero (xidim/2, xidim/2);
  AAR = Matrix::Zero (xidim, xidim);
  AAR << AA, ZZ, ZZ, AA;                    //// this creates the matrix AA for two Robots

  AAR /= dt * dt * (nq + 1);

  // not needed anyhow
  // double cc (double (startp1.point_.transpose() * startp1.point_) + double (endp1.point_.transpose() * endp1.point_));
  // cc /= dt * dt * (nq + 1);
  // double ccR (double (startp1.point_.transpose() * startp1.point_) + double (endp1.point_.transpose() * endp1.point_) + double (startp2.point_.transpose() * startp2.point_) + double (endp2.point_.transpose() * endp2.point_));
  // ccR /= dt * dt * (nq + 1);

  Ainv = AA.inverse();
  AARinv = AAR.inverse();

  // cout << "AA\n" << AA
  //      << "\nAinv\n" << Ainv
  //      << "\nbb\n" << bb << "\n\n";
}


static void cb_step ()
{
  state = STEP;
}


static void cb_run ()
{
  if (RUN == state) {
    state = PAUSE;
  }
  else {
    state = RUN;
  }
}


static void cb_jumble ()
{
  for (size_t ii (0); ii < xidim; ++ii) {
    xi[ii] = double (rand()) / (0.1 * numeric_limits<int>::max()) - 5.0;
  }
  update_robots();
}


static void cb_idle ()
{
  if (PAUSE == state) {
    return;
  }
  if (STEP == state) {
    state = PAUSE;
  }

  // initialize Vector b which has to be updated every time as long as it is possible to move start and end points
  bbR = Vector::Zero (xidim);
  bbR.block (0,                 0, cdim, 1) = startp1.point_;
  bbR.block (xidim/2 - cdim,    0, cdim, 1) = endp1.point_;
  bbR.block (xidim/2,           0, cdim, 1) = startp2.point_;
  bbR.block (xidim - cdim,      0, cdim, 1) = endp2.point_;
  bbR /= - dt * dt * (nq + 1);

  //////////////////////////////////////////////////
  // beginning of "the" CHOMP iteration

  //// calculation of gradient of smoothness objective
  Vector nabla_smooth (AAR * xi + bbR);      //// changed from AA to AAR
  Vector const & xidd (nabla_smooth); // indeed, it is the same in this formulation...

  //// calculation of gradient of obstacle objective
  Vector nabla_obs (Vector::Zero (xidim));
  Vector nabla_int (Vector::Zero (xidim));
  Vector nabla_int_half (Vector::Zero (xidim/2));
  Matrix const JJ (Matrix::Identity (2, 2)); // a little silly here, as noted down.
  for (size_t iq (0); iq < nq; ++iq) {
    Vector const qq (xi.block (iq * cdim, 0, cdim, 1));
    Vector qd;
    if (iq == 0) {
        qd = (xi.block ((iq+1) * cdim, 0, cdim, 1) - startp1.point_) / (2*dt);
    }
    else if (iq == nq/2 - 1) {                   //// difference of end point of FIRST robot to second last point
        qd = (endp1.point_ - xi.block ((iq-1) * cdim, 0, cdim, 1)) / (2*dt);
    }
    else if (iq == nq/2) {
        qd = (xi.block ((iq+1) * cdim, 0, cdim, 1) - startp2.point_) / (2*dt);
    }
    else if (iq == nq - 1) {
        qd = (endp2.point_ - xi.block ((iq-1) * cdim, 0, cdim, 1)) / (2*dt);
    }
    else {
        qd = (xi.block ((iq+1) * cdim, 0, cdim, 1) - xi.block ((iq-1) * cdim, 0, cdim, 1)) / (2*dt);
    }

    // In this case, C and W are the same, Jacobian is identity.  We
    // still write more or less the full-fledged CHOMP expressions
    // (but we only use one body point) to make subsequent extension
    // easier.
    //
    Vector const & xx (qq);
    Vector const & xd (qd);
    double const vel (xd.norm());
    if (vel < 1.0e-3) {	// avoid div by zero further down
      continue;
    }
    Vector const xdn (xd / vel);
    Vector const xdd (JJ * xidd.block (iq * cdim, 0, cdim , 1));
    Matrix const prj (Matrix::Identity (2, 2) - xdn * xdn.transpose()); // hardcoded planar case
    Vector const kappa (prj * xdd / pow (vel, 2.0));

    // Distance between corresponding trajectory points of the two robots
    if (iq < nq/2) {
      Vector dd2(2);
      dd2 = xi.block (iq * cdim, 0, cdim, 1) - xi.block (((iq * cdim) + 40), 0, cdim, 1);  // calculates the difference between every q_i and its correspondent q_n+2+i of the second robot
      double const ddnorm (dd2.norm());   // stack the distances between the two robot trajectory points in a vector
      static double const maxdistR (3.0);
      // I can not use this if statement here, because otherwise the rest (i.e. nabla_obs) won't be calculated
      if ((ddnorm >= maxdistR) || (ddnorm < 1e-9)) {
        goto NablaObjective;
      }

      // Calculation of cost function and gradient of interference objective nabla_int
      static double const gainR (1.0); // hardcoded param
      double const costR (gainR * maxdistR * pow (1.0 - ddnorm / maxdistR, 3.0) / 3.0);
      dd2 *= - gainR * pow (1.0 - ddnorm / maxdistR, 2.0) / ddnorm;

      nabla_int_half.block (iq * cdim, 0, cdim, 1) += JJ.transpose() * vel * (prj * dd2 - costR * kappa);
      nabla_int << nabla_int_half, - nabla_int_half;
    }

    // Calculation of the cost function for nabla_obs
    NablaObjective:
    double mindist;
    Vector delta (Vector::Zero (cdim));
    Vector delta1 (xx - repulsor.point_);   // first obstacle - robots distance calculations
    Vector delta2 (xx - repulsor2.point_);  // second obstacle - robots distance calculations
    double const dist1 (delta1.norm());
    double const dist2 (delta2.norm());
    if (dist1 <= dist2) {                   // find minimum distance obstacle - robot
       mindist = dist1;                     // for multiple obstacle case, search minimum from all obstacles
       delta = delta1;
    }
    else {
       mindist = dist2;
       delta = delta2;
    }
    static double const maxdist (4.0); // hardcoded param
    static double const gain (2.0); // hardcoded param
    if ((mindist <= maxdist) && (mindist > 1e-9)) {
      double const cost1 (gain * maxdist * pow (1.0 - mindist / maxdist, 3.0) / 3.0); // hardcoded param
      delta *= - gain * pow (1.0 - mindist / maxdist, 2.0) / mindist; // hardcoded param

      nabla_obs.block (iq * cdim, 0, cdim, 1) += JJ.transpose() * vel * (prj * delta - cost1 * kappa);
    }
  }

  //// applying the update rule
  Vector dxi (AARinv * (nabla_obs + lambda * nabla_smooth + mu * nabla_int));
  xi -= dxi / eta;

  // end of "the" CHOMP iteration
  //////////////////////////////////////////////////

  update_robots ();
}


static void cb_draw ()
{
  //////////////////////////////////////////////////
  // set bounds

  Vector bmin(2);
  Vector bmax(2);

  Vector qsex(4);       // if adapted to multiple robot case, replace it with number of robots multiplied by 2
  Vector qsey(4);
  Vector qsetot(8);     // sum of qsex and qsey
  qsetot << startp1.point_, startp2.point_, endp1.point_, endp2.point_;

  for (size_t jj (0); jj < 4; ++jj) {
      qsex[jj] = qsetot[jj*cdim];
      qsey[jj] = qsetot[1 + jj*cdim];
  }

  bmin[0] = qsex.minCoeff(); bmin[1] = qsey.minCoeff();
  bmax[0] = qsex.maxCoeff(); bmax[1] = qsey.maxCoeff();

  for (size_t ii (0); ii < 2; ++ii) {
    for (size_t jj (0); jj < nq; ++jj) {
      if (xi[ii + cdim * jj] < bmin[ii]) {
    bmin[ii] = xi[ii + cdim * jj];
      }
      if (xi[ii + cdim * jj] > bmax[ii]) {
    bmax[ii] = xi[ii + cdim * jj];
      }
    }
  }

  gfx::set_view (bmin[0] - 2.0, bmin[1] - 2.0, bmax[0] + 2.0, bmax[1] + 2.0);

  //////////////////////////////////////////////////
  // robots

  rstart.draw_sp();
  rstart2.draw_sp();
  for (size_t ii (0); ii < robots.size(); ++ii) {
    robots[ii].draw();
  }
  rend.draw_ep();
  rend2.draw_ep();

  //////////////////////////////////////////////////
  // trj

  gfx::set_pen (1.0, 0.2, 0.2, 0.2, 1.0);
  gfx::draw_line (startp1.point_[0], startp1.point_[1], xi[0], xi[1]);
  gfx::draw_line (startp2.point_[0], startp2.point_[1], xi[xidim/2], xi[xidim/2 + 1]);
  for (size_t ii (1); ii < nq/2; ++ii) {
    gfx::draw_line (xi[(ii-1) * cdim], xi[(ii-1) * cdim + 1], xi[ii * cdim], xi[ii * cdim + 1]);
  }
  for (size_t ii (nq/2 + 1); ii < nq; ++ii) {
    gfx::draw_line (xi[(ii-1) * cdim], xi[(ii-1) * cdim + 1], xi[ii * cdim], xi[ii * cdim + 1]);
  }
  gfx::draw_line (xi[(nq/2 - 1) * cdim], xi[(nq/2 - 1) * cdim + 1], endp1.point_[0], endp1.point_[1]);
  gfx::draw_line (xi[(nq-1) * cdim], xi[(nq-1) * cdim + 1], endp2.point_[0], endp2.point_[1]);

  gfx::set_pen (5.0, 0.8, 0.2, 0.2, 1.0);
  gfx::draw_point (startp1.point_[0], startp1.point_[1]);
  gfx::draw_point (startp2.point_[0], startp2.point_[1]);
  gfx::set_pen (5.0, 0.5, 0.5, 0.5, 1.0);
  for (size_t ii (0); ii < nq/2; ++ii) {
    gfx::set_pen (5.0, 0.75 - 0.6*sin(ii*0.4), 0.24 + 0.6*sin(ii*0.7), 0.16 + 0.8*sin(ii*0.2), 1.0);    // get a rainbow trajectory
    //gfx::set_pen (5.0, 0.8 - 0.6*ii*0.05, 0.2 + 0.6*ii*0.05, 0.2, 1.0);                           // draws the trajectory from red to green
    gfx::draw_point (xi[ii * cdim], xi[ii * cdim + 1]);
    gfx::draw_point (xi[(ii * cdim) + 40], xi[(ii * cdim) + 41]);
  }
  gfx::set_pen (5.0, 0.2, 0.8, 0.2, 1.0);
  gfx::draw_point (endp1.point_[0], endp1.point_[1]);
  gfx::draw_point (endp2.point_[0], endp2.point_[1]);

  //////////////////////////////////////////////////
  // handles

  for (handle_s ** hh (handle); *hh != 0; ++hh) {
    gfx::set_pen (1.0, (*hh)->red_, (*hh)->green_, (*hh)->blue_, (*hh)->alpha_);
    gfx::fill_arc ((*hh)->point_[0], (*hh)->point_[1], (*hh)->radius_, 0.0, 2.0 * M_PI);
  }
  for (handle_s ** hh (handle2); *hh != 0; ++hh) {                                          // draw second obstacle
    gfx::set_pen (1.0, (*hh)->red_, (*hh)->green_, (*hh)->blue_, (*hh)->alpha_);
    gfx::fill_arc ((*hh)->point_[0], (*hh)->point_[1], (*hh)->radius_, 0.0, 2.0 * M_PI);
  }
}


static void cb_mouse (double px, double py, int flags)
{
  if (flags & gfx::MOUSE_PRESS) {
    for (handle_s ** hh (handle); *hh != 0; ++hh) {
      Vector offset ((*hh)->point_);
      offset[0] -= px;
      offset[1] -= py;
      if (offset.norm() <= (*hh)->radius_) {
        grab_offset = offset;
        grabbed = *hh;
        break;
      }
    }
    for (handle_s ** hh (handle2); *hh != 0; ++hh) {
      Vector offset ((*hh)->point_);
      offset[0] -= px;
      offset[1] -= py;
      if (offset.norm() <= (*hh)->radius_) {
        grab_offset = offset;
        grabbed = *hh;
        break;
      }
    }
    for (handle_s ** hh (handle_sp1); *hh != 0; ++hh) {
      Vector offset ((*hh)->point_);
      offset[0] -= px;
      offset[1] -= py;
      if (offset.norm() <= (*hh)->radius_) {
        grab_offset = offset;
        grabbed = *hh;
        break;
      }
    }
    for (handle_s ** hh (handle_ep1); *hh != 0; ++hh) {
      Vector offset ((*hh)->point_);
      offset[0] -= px;
      offset[1] -= py;
      if (offset.norm() <= (*hh)->radius_) {
        grab_offset = offset;
        grabbed = *hh;
        break;
      }
    }
    for (handle_s ** hh (handle_sp2); *hh != 0; ++hh) {
      Vector offset ((*hh)->point_);
      offset[0] -= px;
      offset[1] -= py;
      if (offset.norm() <= (*hh)->radius_) {
        grab_offset = offset;
        grabbed = *hh;
        break;
      }
    }
    for (handle_s ** hh (handle_ep2); *hh != 0; ++hh) {
      Vector offset ((*hh)->point_);
      offset[0] -= px;
      offset[1] -= py;
      if (offset.norm() <= (*hh)->radius_) {
        grab_offset = offset;
        grabbed = *hh;
        break;
      }
    }
  }
  else if (flags & gfx::MOUSE_DRAG) {
    if (0 != grabbed) {
      grabbed->point_[0] = px;
      grabbed->point_[1] = py;
      grabbed->point_ += grab_offset;
    }
  }
  else if (flags & gfx::MOUSE_RELEASE) {
    grabbed = 0;
  }
//cout << endl << "startp1.point_" << endl << startp1.point_ << endl;
}


int main()
{
  struct timeval tt;
  gettimeofday (&tt, NULL);
  srand (tt.tv_usec);

  init_chomp();
  update_robots();
  state = PAUSE;

  gfx::add_button ("jumble", cb_jumble);
  gfx::add_button ("step", cb_step);
  gfx::add_button ("run", cb_run);
  gfx::main ("chomp", cb_idle, cb_draw, cb_mouse);
}
