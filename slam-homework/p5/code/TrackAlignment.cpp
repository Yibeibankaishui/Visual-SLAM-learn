#include <iostream>
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <sophus/se3.hpp>
#include <fstream>
#include <iomanip>
#include <pangolin/pangolin.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>
#include <unistd.h>


using namespace std;
using namespace Sophus;
// using namespace cv;

string track_file = "../compare.txt";

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef vector<cv::Point3f> PointType;

void pose_estimation_3d3d(
  const PointType &pts1,
  const PointType &pts2,
  cv::Mat &R, cv::Mat &t
);

// void read_file(string in_file, PointType &pts1, PointType &pts2);


// void transform(PointType &ptbefore, PointType &ptafter, const cv::Mat &R, const cv::Mat &t);

// TODO
// 读取txt，绘制轨迹图
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

int ReadTrajectory(const string &path, TrajectoryType &groundtruth, TrajectoryType &estimated);

// TrajectoryType --> PointType
void TrajectoryToPoint(const TrajectoryType &Trajectory, PointType &Points);
// TODO
// 进行变换
void TrajectoryTransform(const TrajectoryType &Trajectory, TrajectoryType &TrajectoryNew, const cv::Mat &R, const cv::Mat &t);

int main(int argc, char **argv)
{
    PointType ptg;    // groundtruth
    PointType pte;    // estimate
    TrajectoryType groundtruth;
    TrajectoryType estimated;
    // 读取文本得到点集，TrajectoryType
    if(ReadTrajectory(track_file, groundtruth, estimated)==0)
      return 0;
    // read_file(track_file, ptg, pte);
    // 画图，转化前
    // DrawTrajectory(groundtruth,estimated);
    // cv::waitKey(0);
    // 得到PointType的点集数据 TrajectoryType  ->  PointType
    // SE3 -> cv::Point3f
    TrajectoryToPoint(groundtruth, ptg);
    TrajectoryToPoint(estimated, pte);
    // cv::waitKey(0);
    cv::Mat R, t;
    // 计算变换
    pose_estimation_3d3d(ptg, pte, R, t);

    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;
    // cv::waitKey(0);
    // 坐标变换，转化到同一个坐标系中 TrajectoryType
    TrajectoryType estimated_new;
    TrajectoryTransform(estimated, estimated_new, R, t);
    // 画图，转化后
    DrawTrajectory(groundtruth, estimated_new);
    cv::waitKey(0);


    return 0;
}


void TrajectoryTransform(const TrajectoryType &Trajectory, TrajectoryType &TrajectoryNew, const cv::Mat &R, const cv::Mat &t)
{
  // cv::Mat --> Eigen::Matrix
  Eigen::MatrixXd R_(3, 3);
  Eigen::MatrixXd t_(3, 1);
  cv::cv2eigen(R, R_);
  cv::cv2eigen(t, t_);
  Sophus::SE3d T(R_, t_);
  for (int i = 0; i < Trajectory.size(); i++)
  {
    auto pt = Trajectory[i];
    auto pt_new = T * pt;
    TrajectoryNew.push_back(pt_new);
  }
}


void TrajectoryToPoint(const TrajectoryType &Trajectory, PointType &Points)
{
  for (int i = 0; i < Trajectory.size(); i++)
  {
    auto pt = Trajectory[i];
    cv::Point3f point(pt.translation()[0], pt.translation()[1], pt.translation()[2]);
    Points.push_back(point);
  }

}

// void read_file(string in_file, PointType &ptg, PointType &pte)
// {
//     // 读取文件
//     ifstream infile(in_file); 
//     // infile_2d.open(p2d_file, ios::in);
//     double time_e, tx_e, ty_e, tz_e, q1_e, q2_e, q3_e, q4_e, time_g, tx_g, ty_g, tz_g, q1_g, q2_g, q3_g, q4_g;
//     while (infile >> time_e >> tx_e >> ty_e >> tz_e >> q1_e >> q2_e >> q3_e >> q4_e >> time_g >> tx_g >> ty_g >> tz_g >> q1_g >> q2_g >> q3_g >> q4_g) 
//     {
//         cv::Point3f pt_e(tx_e, ty_e, tz_e);
//         cv::Point3f pt_g(tx_g, ty_g, tz_g);
        
//         ptg.push_back(pt_g);
//         pte.push_back(pt_e);
//     }
//     infile.close();
// }


void pose_estimation_3d3d(const PointType &pts1,
                          const PointType &pts2,
                          cv::Mat &R, cv::Mat &t) {
  cv::Point3f p1, p2;     // center of mass
  int N = pts1.size();
  for (int i = 0; i < N; i++) {
    p1 += pts1[i];
    p2 += pts2[i];
  }
  p1 = cv::Point3f(cv::Vec3f(p1) / N);
  p2 = cv::Point3f(cv::Vec3f(p2) / N);
  PointType q1(N), q2(N); // remove the center
  for (int i = 0; i < N; i++) {
    q1[i] = pts1[i] - p1;
    q2[i] = pts2[i] - p2;
  }

  // compute q1*q2^T
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  for (int i = 0; i < N; i++) {
    W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
  }
  cout << "W=" << W << endl;

  // SVD on W
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  cout << "U=" << U << endl;
  cout << "V=" << V << endl;

  Eigen::Matrix3d R_ = U * (V.transpose());
  if (R_.determinant() < 0) {
    R_ = -R_;
  }
  Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

  // convert to cv::Mat
  R = (cv::Mat_<double>(3, 3) <<
    R_(0, 0), R_(0, 1), R_(0, 2),
    R_(1, 0), R_(1, 1), R_(1, 2),
    R_(2, 0), R_(2, 1), R_(2, 2)
  );
  t = (cv::Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}


int ReadTrajectory(const string &path, TrajectoryType &groundtruth, TrajectoryType &estimated) {
  ifstream fin(path);
  TrajectoryType trajectory;
  if (!fin) {
    cerr << path << " not found." << endl;
    return 0;
  }
  
  while (!fin.eof()) {
    double time_e, tx_e, ty_e, tz_e, qx_e, qy_e, qz_e, qw_e, time_g, tx_g, ty_g, tz_g, qx_g, qy_g, qz_g, qw_g;
    fin >> time_e >> tx_e >> ty_e >> tz_e >> qx_e >> qy_e >> qz_e >> qw_e >> time_g >> tx_g >> ty_g >> tz_g >> qx_g >> qy_g >> qz_g >> qw_g;
    Sophus::SE3d pg(Eigen::Quaterniond(qw_g, qx_g, qy_g, qz_g), Eigen::Vector3d(tx_g, ty_g, tz_g));
    Sophus::SE3d pe(Eigen::Quaterniond(qw_e, qx_e, qy_e, qz_e), Eigen::Vector3d(tx_e, ty_e, tz_e));
    groundtruth.push_back(pg);
    estimated.push_back(pe);

  }
  fin.close();
  return 1;
}


void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));


  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glLineWidth(2);
    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
      glBegin(GL_LINES);
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }

}

