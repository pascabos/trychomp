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


/*
  28.10.14 pb
   Here I present the curvature vector graphically in the GUI. You can see
   how it grows with sharper curves.
  04.11.14 pb
   This implementation of the curvature objective is derivated on the extra papers
   glued into my workbook. See the date of 03./04.11.14. It doesn't work properly,
   so it wasn't further improved.
  24.11.14 pb
   Inside the CHOMP iteration I implement the curvature objective. This is the second
   time I try it the way taking the partial derivatives of the discretized waypoints
   in x- and y-direction. The first try on the extra sheet from 17.11.14 is replaced
   with the velocity formulation of Roland.
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


//////////////////////////////////////////////////
// trajectory etc

Vector xi;			// the trajectory (q_1, q_2, ...q_n)
Vector qs;			// the start config a.k.a. q_0
Vector qe;			// the end config a.k.a. q_(n+1)
Vector qfix;        // zero velocity and acceleration points for the start and endpoint
Vector curvature;   // curvature vector a.k.a kappa
static size_t const nq (20);	// number of q stacked into xi
static size_t const cdim (2);	// dimension of config space
static size_t const xidim (nq * cdim); // dimension of trajectory, xidim = nq * cdim
static double const dt (1.0);	       // time step
static double const eta (100.0); // >= 1, regularization factor for gradient descent
static double const lambda (1.0); // weight of smoothness objective
static double const omega (1.0); // weight of smoothness objective

//////////////////////////////////////////////////
// gradient descent etc

Matrix AA;			// metric
Vector bb;			// acceleration bias for start and end config
Matrix Ainv;			// inverse of AA

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

static handle_s repulsor (1.5, 0.0, 0.0, 1.0, 0.2);

static handle_s * handle[] = { &repulsor, 0 };
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


  void draw () const
  {
    // translucent disk for base
    gfx::set_pen (1.0, 0.7, 0.7, 0.7, 0.5);
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
Robot rend;
vector <Robot> robots;


//////////////////////////////////////////////////

static void update_robots ()
{
  rstart.update (qs);
  rend.update (qe);
  if (nq != robots.size()) {
    robots.resize (nq);
  }
  for (size_t ii (0); ii < nq; ++ii) {
    robots[ii].update (xi.block (ii * cdim, 0, cdim, 1));
  }
}


static void init_chomp ()
{
  qs.resize (cdim);
  qs << -5.0, -5.0;
  qe.resize (cdim);
  qe << 7.0, 7.0;
  xi = Vector::Zero (xidim);
  curvature = Vector::Zero (xidim);
  qfix.resize (cdim);
  qfix << 0.0, 0.0;

  repulsor.point_ << 3.0, 0.0;

  // cout << "qs\n" << qs
  //      << "\nxi\n" << xi
  //      << "\nqe\n" << qe << "\n\n";

  AA = Matrix::Zero (xidim, xidim);
  for (size_t ii(0); ii < nq; ++ii) {
    AA.block (cdim * ii, cdim * ii, cdim , cdim) = 2.0 * Matrix::Identity (cdim, cdim);
    if (ii > 0) {
      AA.block (cdim * (ii-1), cdim * ii, cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
      AA.block (cdim * ii, cdim * (ii-1), cdim , cdim) = -1.0 * Matrix::Identity (cdim, cdim);
    }
  }
  AA /= dt * dt * (nq + 1);

  bb = Vector::Zero (xidim);
  bb.block (0,            0, cdim, 1) = qs;
  bb.block (xidim - cdim, 0, cdim, 1) = qe;
  bb /= - dt * dt * (nq + 1);

  // not needed anyhow
  // double cc (double (qs.transpose() * qs) + double (qe.transpose() * qe));
  // cc /= dt * dt * (nq + 1);

  Ainv = AA.inverse();

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

  //////////////////////////////////////////////////
  // beginning of "the" CHOMP iteration

  Vector nabla_smooth (AA * xi + bb);
  Vector const & xidd (nabla_smooth); // indeed, it is the same in this formulation...

  Vector nabla_obs (Vector::Zero (xidim));
  curvature = Vector::Zero (xidim);         // important to set it zero here! if done in init_chomp(), values could be saved where no curvature is anymore
  Vector velocity (Vector::Zero (xidim));
  Vector kappa_l (Vector::Zero (nq));
  for (size_t iq (0); iq < nq; ++iq) {

    // Position
    Vector const qq (xi.block (iq * cdim, 0, cdim, 1));

    // Velocity
    Vector qd;
    if (iq == 0) {
        qd = (xi.block ((iq+1) * cdim, 0, cdim, 1) - qs) / (2*dt);
    }
    else if (iq == nq - 1) {
      qd = (qe - xi.block ((iq-1) * cdim, 0, cdim, 1)) / (2*dt);
    }
    else {
      qd = (xi.block ((iq+1) * cdim, 0, cdim, 1) - xi.block ((iq-1) * cdim, 0, cdim, 1)) / (2*dt);
    }
    velocity.block (iq * cdim, 0, cdim, 1) = qd;

    // In this case, C and W are the same, Jacobian is identity.  We
    // still write more or less the full-fledged CHOMP expressions
    // (but we only use one body point) to make subsequent extension
    // easier.
    //
    Vector const & xx (qq);
    Vector const & xd (qd);
    Matrix const JJ (Matrix::Identity (2, 2)); // a little silly here, as noted above.
    double const vel (xd.norm());
    if (vel < 1.0e-3) {	// avoid div by zero further down
      continue;
    }
    Vector const xdn (xd / vel);
    Vector const xdd (JJ * xidd.block (iq * cdim, 0, cdim , 1));
    Matrix const prj (Matrix::Identity (2, 2) - xdn * xdn.transpose()); // hardcoded planar case
    Vector const kappa (prj * xdd / pow (vel, 2.0));

    // obstacle objective calculations
    Vector delta (xx - repulsor.point_);
    double const dist (delta.norm());
    static double const maxdist (4.0); // hardcoded param
    static double const gain (10.0); // hardcoded param
    if ((dist <= maxdist) && (dist > 1e-9)) {
      double const cost (gain * maxdist * pow (1.0 - dist / maxdist, 3.0) / 3.0); // hardcoded param
      delta *= - gain * pow (1.0 - dist / maxdist, 2.0) / dist; // hardcoded param
      nabla_obs.block (iq * cdim, 0, cdim, 1) += JJ.transpose() * vel * (prj * delta - cost * kappa);
    }
    curvature.block (iq * cdim, 0, cdim, 1) = kappa;
    kappa_l(iq) = kappa.norm();
  }

  // Acceleration
  Vector acceleration (Vector::Zero (xidim));
//  for (size_t iq (0); iq < nq; ++iq) {
//    Vector qdd (2);
//    if (iq == 0) {
//      qdd = (velocity.block ((iq+1) * cdim, 0, cdim, 1) - qfix) / (2*dt);
//    }
//    else if (iq == nq - 1) {
//      qdd = (qfix - velocity.block ((iq-1) * cdim, 0, cdim, 1)) / (2*dt);
//    }
//    else {
//      qdd = (velocity.block ((iq+1) * cdim, 0, cdim, 1) - velocity.block ((iq-1) * cdim, 0, cdim, 1)) / (2*dt);
//    }
//    acceleration.block (iq * cdim, 0, cdim, 1) = qdd;
//  }
    acceleration = xidd;

  // Jerk
  Vector jerk (Vector::Zero (xidim));
  for (size_t iq (0); iq < nq; ++iq) {
    Vector qddd (2);
    if (iq == 0) {
      qddd = (acceleration.block ((iq+1) * cdim, 0, cdim, 1) - qfix) / (2*dt);
    }
    else if (iq == nq - 1) {
      qddd = (qfix - acceleration.block ((iq-1) * cdim, 0, cdim, 1)) / (2*dt);
    }
    else {
      qddd = (acceleration.block ((iq+1) * cdim, 0, cdim, 1) - acceleration.block ((iq-1) * cdim, 0, cdim, 1)) / (2*dt);
    }
    jerk.block (iq * cdim, 0, cdim, 1) = qddd;
  }

  // calculation of nabla_curv
  Vector nabla_curvature (Vector::Zero (xidim));
  for (size_t iq (0); iq < nq; ++iq) {
    Vector kappa (2);
    kappa = curvature.block (iq * cdim, 0, cdim, 1);
    double const length (kappa.norm());
    static double const maxlength (0.01);

    // cost function
    double cost (0.0);
    if (length >= maxlength) {
      cost = (1/(2*maxlength)) * pow (length - maxlength, 2.0);
    }
    else {
      cost = 0.0;
    }
      // define the gradient cost function
    double nabla_cost = 0.0;
    if (length >= maxlength) {
      nabla_cost = (1/maxlength) * (length-maxlength);
    }
    else {
      nabla_cost = 0.0;
    }

    // gradient of kappa
    double const vel ((velocity.block(iq * cdim, 0, cdim, 1)).norm());
    Matrix const prj (Matrix::Identity (2, 2) - velocity.block (iq * cdim, 0, cdim, 1) * (velocity.block(iq * cdim, 0, cdim, 1)).transpose());
    Matrix const accvel (acceleration.block (iq * cdim, 0, cdim, 1) * (velocity.block(iq * cdim, 0, cdim, 1)).transpose() + velocity.block(iq * cdim, 0, cdim, 1) * (acceleration.block(iq * cdim, 0, cdim, 1)).transpose());
    Vector const firstp (accvel * acceleration.block(iq * cdim, 0, cdim, 1) / pow (vel, 3));
    Vector const secondp (prj * jerk.block(iq * cdim, 0, cdim, 1) / vel);
    Vector const nabla_kappa (firstp + secondp);
    nabla_curvature.block (iq * cdim, 0, cdim, 1) += (nabla_cost * kappa + cost * nabla_kappa);
  }

  //////////////////////////////////////////
  // Try of curvature with discretized partial derivatives
  Vector const qq (xi);
  Vector nabla_curv (Vector::Zero (xidim));
  Vector kk (Vector::Zero (nq));
  for (size_t ii (0); ii < nq; ++ii) {
    double x_m1;        // x_m1 is the entry x_(i-1) in xi
    double y_m1;        // y_m1 is the entry y_(i-1) in xi
    double x_i;
    double y_i;
    double x_p1;        // x_p1 is the entry x_(i+1) in xi
    double y_p1;        // y_p1 is the entry y_(i+1) in xi

    if (ii == 0) {
      x_m1 = qs[0];
      y_m1 = qs[1];
      x_i = qq[((ii) * cdim)];
      y_i = qq[((ii) * cdim) + 1];
      x_p1 = qq[((ii + 1) * cdim)];
      y_p1 = qq[((ii + 1) * cdim) + 1];
    }

    else if (ii == nq - 1) {
      x_m1 = qq[((ii - 1) * cdim)];
      y_m1 = qq[((ii - 1) * cdim) + 1];
      x_i = qq[((ii) * cdim)];
      y_i = qq[((ii) * cdim) + 1];
      x_p1 = qe[0];
      y_p1 = qe[1];
    }

    else {
      x_m1 = qq[((ii - 1) * cdim)];
      y_m1 = qq[((ii - 1) * cdim) + 1];
      x_i = qq[((ii) * cdim)];
      y_i = qq[((ii) * cdim) + 1];
      x_p1 = qq[((ii + 1) * cdim)];
      y_p1 = qq[((ii + 1) * cdim) + 1];
    }

    double const NN = x_p1*(y_m1-y_i) + x_i*(y_p1-y_m1) + x_m1*(y_i-y_m1);
    double const DD = pow((x_m1 - x_p1), 2) + pow((y_m1 - y_p1), 2);
    kk(ii) = 8 * NN / pow(DD, (3/2));           // this is kappa

    double const dx_m1 = ((8 * (y_i-y_p1)) / pow(DD, (3/2))) - ((24 * (x_m1-x_p1) * NN) / pow(DD, (5/2)));
    double const dx_i = ((8 * (y_p1-y_m1)) / pow(DD, (3/2)));
    double const dx_p1 = ((8 * (y_m1-y_i)) / pow(DD, (3/2))) - ((24 * (x_p1-x_m1) * NN) / pow(DD, (5/2)));
    double const dy_m1 = ((8 * (x_p1-x_i)) / pow(DD, (3/2))) - ((24 * (y_m1-y_p1) * NN) / pow(DD, (5/2)));
    double const dy_i = ((8 * (x_m1-x_p1)) / pow(DD, (3/2)));
    double const dy_p1 = ((8 * (x_i-x_m1)) / pow(DD, (3/2))) - ((24 * (y_m1-y_p1) * NN) / pow(DD, (5/2)));

    Vector nabla_k (2);
    nabla_k << dx_m1 + dx_i + dx_p1, dy_m1 + dy_i + dy_p1;

    nabla_curv.block (ii * cdim, 0, cdim, 1) += 2 * kk(ii) * nabla_k;

    //cout << endl << "kk" << endl << kk << endl;
  }
  //cout << endl << "nabla_curv" << endl << nabla_curv << endl;

  Vector dxi (Ainv * (nabla_obs + lambda * nabla_smooth + omega * nabla_curv));
  xi -= dxi / eta;

  // computation of k_i for a 2D planar case
  Vector k_i (Vector::Zero (nq));
  for (size_t iq (0); iq < nq; ++iq) {
      double const velx = velocity.block(iq * cdim, 0, 1, 1).norm();
      double const vely = velocity.block((iq * cdim) + 1, 0, 1, 1).norm();
      double const accx = acceleration.block((iq * cdim), 0, 1, 1).norm();
      double const accy = acceleration.block((iq * cdim) + 1, 0, 1, 1).norm();

      double numerator = velx * accy - vely * accx;
      double denumerator = pow( pow(velx, 2) + pow(vely, 2), (3/2));
      double const kk = numerator / denumerator;
      k_i(iq) = kk;
  }

  //cout << endl << "k_i" << endl << k_i << endl;
  //cout << endl << "velocity" << endl << velocity << endl;
  cout << endl << "nabla_curvature" << endl << nabla_curvature << endl;

  curvature *= 150;         // for visualization the curvature vector is multiplied by a factor
  //cout << endl << "curvature" << endl << curvature << endl;
  //cout << endl << "jerk" << endl << jerk << endl;
  //cout << endl << "kappa_l" << endl << kappa_l << endl;

  // end of "the" CHOMP iteration
  //////////////////////////////////////////////////

  update_robots ();
}


static void cb_draw ()
{
  //////////////////////////////////////////////////
  // set bounds

  Vector bmin (qs);
  Vector bmax (qs);
  for (size_t ii (0); ii < 2; ++ii) {
    if (qe[ii] < bmin[ii]) {
      bmin[ii] = qe[ii];
    }
    if (qe[ii] > bmax[ii]) {
      bmax[ii] = qe[ii];
    }
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

  rstart.draw();
  for (size_t ii (0); ii < robots.size(); ++ii) {
    robots[ii].draw();
  }
  rend.draw();

  //////////////////////////////////////////////////
  // trj

  gfx::set_pen (1.0, 0.2, 0.2, 0.2, 1.0);
  gfx::draw_line (qs[0], qs[1], xi[0], xi[1]);
  for (size_t ii (1); ii < nq; ++ii) {
    gfx::draw_line (xi[(ii-1) * cdim], xi[(ii-1) * cdim + 1], xi[ii * cdim], xi[ii * cdim + 1]);
  }
  gfx::draw_line (xi[(nq-1) * cdim], xi[(nq-1) * cdim + 1], qe[0], qe[1]);

  gfx::set_pen (1.0, 0.1, 0.4, 0.5, 1.0);
  for (size_t ii (0); ii < nq; ++ii) {          // drawing the curvature vector in x direction
      gfx::draw_line (xi[(ii) * cdim], xi[(ii*cdim) + 1], xi[(ii) * cdim] + curvature[ii * cdim], xi[(ii*cdim) + 1]);
  }
  gfx::set_pen (1.0, 0.3, 0.6, 0.3, 1.0);
  for (size_t ii (0); ii < nq; ++ii) {          // drawing the curvature vector in y direction
      gfx::draw_line (xi[(ii) * cdim], xi[(ii)*cdim + 1], xi[(ii) * cdim], xi[(ii*cdim) + 1] + curvature[(ii*cdim) + 1]);
  }
  gfx::set_pen (2.5, 0.6, 0.2, 0.2, 1.0);
  for (size_t ii (0); ii < nq; ++ii) {          // drawing the curvature vector in y direction
      gfx::draw_line (xi[(ii) * cdim], xi[(ii)*cdim + 1], xi[(ii) * cdim] + curvature[ii * cdim], xi[(ii*cdim) + 1] + curvature[(ii*cdim) + 1]);
  }

  gfx::set_pen (5.0, 0.8, 0.2, 0.2, 1.0);
  gfx::draw_point (qs[0], qs[1]);
  gfx::set_pen (5.0, 0.5, 0.5, 0.5, 1.0);
  for (size_t ii (0); ii < nq; ++ii) {
    gfx::draw_point (xi[ii * cdim], xi[ii * cdim + 1]);
  }
  gfx::set_pen (5.0, 0.2, 0.8, 0.2, 1.0);
  gfx::draw_point (qe[0], qe[1]);

  //////////////////////////////////////////////////
  // handles

  for (handle_s ** hh (handle); *hh != 0; ++hh) {
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
