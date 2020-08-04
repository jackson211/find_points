/*
The image.png is made up of a regular points grid with three special points.
Please write a C/C++ function that finds all points in the calibration image.

Your solution will be evaluated on efficiency, coding style, readability, and
correctness. In addition to correct results and time taken, optimal code is a
significant factor in determining your grade.

Assume that this is for the PC platform.
*/

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

struct Point {
  int x;
  int y;
};

void remove_repeated(std::vector<Point> &data) {
  data.erase(std::unique(data.begin(), data.end(),
                         [](const Point &lhs, const Point &rhs) {
                           return lhs.x == rhs.x && lhs.y == rhs.y;
                         }),
             data.end());
}
void sort_points(std::vector<Point> &points, bool sort_on_x = true) {
  sort_on_x ? std::sort(points.begin(), points.end(),
                        [](const Point &lhs, const Point &rhs) {
                          return lhs.x < rhs.x;
                        })
            : std::sort(points.begin(), points.end(),
                        [](const Point &lhs, const Point &rhs) {
                          return lhs.y < rhs.y;
                        });
}

double distance(const Point &p1, const Point &p2) {
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

void find_points(std::vector<Point> &regular_points,
                 std::vector<Point> &special_points) {

  sort_points(regular_points);
  Point p0 = regular_points[0];
  Point p1 = regular_points.back();
  std::cout << "p0: " << p0.x << " " << p0.y << std::endl;
  std::cout << "p1: " << p1.x << " " << p1.y << std::endl;

  sort_points(regular_points, false);
  Point p2 = regular_points[0];
  Point p3 = regular_points.back();
  std::cout << "p2: " << p2.x << " " << p2.y << std::endl;
  std::cout << "p3: " << p3.x << " " << p3.y << std::endl;

  // Coordinates Rotation
  double a = p3.y - p0.y;
  double b = p3.x - p0.x;
  double h = distance(p3, p0);
  double sin_theta = a / h;
  double cos_theta = b / h;

  std::cout << "a: " << a << "b: " << b << "h: " << h
            << " sin_theta: " << sin_theta << " cos_theta: " << cos_theta
            << std::endl;

  std::vector<Point> normalized_points;
  for (auto &p : regular_points) {
    int x = p.x;
    int y = p.y;
    x = x * cos_theta + y * sin_theta;
    y = -x * sin_theta + y * cos_theta;
    p = Point{x, y};
    normalized_points.push_back(p);
  }

  sort_points(normalized_points);
  int min_x = normalized_points[0].x;
  int min_y = -p1.x * sin_theta + p1.y * cos_theta;
  cv::Mat img(800, 1000, CV_8UC3, cv::Scalar(255, 255, 255));
  for (int i = 0; i < normalized_points.size(); i++) {
    Point p = normalized_points[i];
    int norm_x = p.x - min_x;
    int norm_y = p.y - min_y;
    normalized_points[i] = Point{norm_x, norm_y};
    cv::circle(img, cv::Point(norm_x, norm_y), 3, cv::Scalar(0, 0, 255), -1, 8,
               0);
  }

  bool first_visit = true;
  int last_interval = 0;
  bool half_interval = false;
  for (int i = 1; i < normalized_points.size() - 1; i++) {
    int interval = normalized_points[i].x - normalized_points[i - 1].x;
    if (interval > 10) {
      // Setting up last_interval for the first time
      if (first_visit) {
        last_interval = interval;
        first_visit = false;
      } else {
        // Unusual interval change
        if ((last_interval - interval) > 10) {
          special_points.push_back(normalized_points[i]);
          half_interval = true;
        } else {
          half_interval = false;
        }
        last_interval = interval;
      }
    } else if (half_interval) {
      if ((last_interval - interval) < -10) {
        half_interval = false;
      }
      special_points.push_back(normalized_points[i]);
    }
  }

  for (int i = 0; i < special_points.size(); i++) {
    int px = special_points[i].x;
    int py = special_points[i].y;
    cv::circle(img, cv::Point(px, py), 5, cv::Scalar(0, 0, 0), -1, 8, 0);
  }

  // Show in a window
  cv::namedWindow("Normalized Points", cv::WINDOW_AUTOSIZE);
  cv::imshow("Normalized Points", img);
  cv::waitKey(0);
}

void find_contours(cv::Mat &image_gray, std::vector<Point> &points) {
  cv::Mat canny_output;
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  int thresh = 50;
  // Detect edges using canny
  Canny(image_gray, canny_output, thresh, thresh * 2, 3);
  // Find contours
  findContours(canny_output, contours, hierarchy, cv::RETR_TREE,
               cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

  // Draw contours
  cv::RNG rng(12345);
  cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
  for (int i = 0; i < contours.size(); i++) {
    double area = cv::contourArea(contours[i]);
    // Filter out small circles
    if (area < 100)
      continue;

    // Find moment of the circle
    cv::Moments m = cv::moments(contours[i]);
    int x = int(m.m10 / m.m00);
    int y = int(m.m01 / m.m00);

    if (x < 0 || y < 0)
      std::cerr << "Error: Negative value " << x << " " << y << std::endl;

    Point p{x, y};
    points.push_back(p);

    cv::drawContours(drawing, contours, i, cv::Scalar(255, 0, 255), 2, 8,
                     hierarchy, 0, cv::Point());
    cv::circle(drawing, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1, 8, 0);
  }

  remove_repeated(points);
  sort_points(points);
  cv::Point p0(points[0].x, points[0].y);
  cv::Point p1(points.back().x, points.back().y);

  sort_points(points, false);
  cv::Point p2(points[0].x, points[0].y);
  cv::Point p3(points.back().x, points.back().y);

  cv::circle(drawing, p0, 10, cv::Scalar(255, 255, 0), -1, 8, 0);
  cv::circle(drawing, p1, 10, cv::Scalar(255, 255, 0), -1, 8, 0);
  cv::circle(drawing, p2, 10, cv::Scalar(255, 255, 0), -1, 8, 0);
  cv::circle(drawing, p3, 10, cv::Scalar(255, 255, 0), -1, 8, 0);

  // Show in a window
  cv::namedWindow("Contours", cv::WINDOW_AUTOSIZE);
  cv::imshow("Contours", drawing);
  cv::waitKey(0);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("usage: DisplayImage.out <Image_Path>\n");
    return -1;
  }
  cv::Mat image = imread(argv[1], cv::IMREAD_COLOR);

  if (!image.data) {
    printf("No image data \n");
    return -1;
  }
  std::cout << "Rows: " << image.rows << "Columns: " << image.cols << std::endl;

  cv::Mat gray, out;
  cvtColor(image, gray, cv::COLOR_BGR2GRAY);
  medianBlur(gray, gray, 5);

  std::vector<Point> points, special_points;
  find_contours(gray, points);
  find_points(points, special_points);
}
