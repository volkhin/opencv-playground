#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <vector>

using namespace cv;
using std::vector;

const int MIN_CONTOUR_AREA = 300;

bool compareFrames(Mat& image1, Mat& image2, vector<Rect>& areas) {
  Mat out;
  Mat image1gray, image2gray;
  cvtColor(image1, image1gray, CV_BGR2GRAY, 1);
  cvtColor(image2, image2gray, CV_BGR2GRAY, 1);

  Mat delta = abs(image1gray - image2gray);
  Mat blurred;
  imshow("1 delta", delta);
  GaussianBlur(delta, delta, Size(21, 21), 0);
  imshow("2 delta blurred", delta);
  delta = delta > 50;
  imshow("3 delta > threshhold", delta);
  Mat res;
  dilate(delta, res, Mat::ones(5, 5, CV_8U));

  cvtColor(res, out, CV_GRAY2BGR);

  areas.clear();
  vector< vector<Point> > contours;
  findContours(delta, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
  for (int i = 0; i < contours.size(); i++) {
    if (contourArea(contours[i]) < MIN_CONTOUR_AREA) {
      continue;
    }
    Rect r = boundingRect(contours[i]);
    areas.push_back(r);
  }
  imshow("delta", out);

  return areas.size() > 0;
}

void showPairOfImages(const Mat& image1, const Mat& image2, Mat& result) {
  Size size1 = image1.size(), size2 = image2.size();
  int height = std::max(size1.height, size2.height);
  result = Mat(height, size1.width + size2.width, CV_8UC3);
  Mat left(result, Rect(0, 0, size1.width, size1.height));
  image1.copyTo(left);
  Mat right(result, Rect(size1.width, 0, size2.width, size2.height));
  image2.copyTo(right);
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " <filename>" << std::endl;
    return -1;
  }

  VideoCapture video(argv[1]);
  if (!video.isOpened()) {
    std::cerr << "can't open video file " << argv[1] << std::endl;
    return -1;
  }

  int fps = video.get(CV_CAP_PROP_FPS);
  int width = video.get(CV_CAP_PROP_FRAME_WIDTH);
  int height = video.get(CV_CAP_PROP_FRAME_HEIGHT);
  std::cerr << "fps: " << fps << std::endl;
  std::cerr << "width: " << width << std::endl;
  std::cerr << "height: " << height << std::endl;
  std::cerr << "number of frames: " << video.get(CV_CAP_PROP_FRAME_COUNT) << std::endl;

  Mat previousFrame;
  video >> previousFrame;

  namedWindow("current", WINDOW_AUTOSIZE);
  moveWindow("current", 0, 0);
  namedWindow("delta", WINDOW_AUTOSIZE);
  moveWindow("delta", 0, 800);

  vector<Rect> areas;

  for (int i = 0; i < 1000; i++) {
    Mat currentFrame;
    for (int j = 0; j < 10; j++) {
      video.grab();
    }
    if (!video.retrieve(currentFrame)) {
      std::cerr << "no more frames" << std::endl;
      break;
    }

    Mat out = currentFrame.clone();
    if (compareFrames(previousFrame, currentFrame, areas)) {
      std::cerr << "showing " << i << " frame" << std::endl;

      for (int j = 0; j < areas.size(); j++) {
        rectangle(out, areas[j].tl(), areas[j].br(), Scalar(0, 255, 255), 2);
      }

      imshow("current", out);
      waitKey(0);
    }

    const double previousCoeff = 10.0;
    previousFrame = (previousCoeff * previousFrame + currentFrame) / (previousCoeff + 1.0);
    imshow("previousFrame", previousFrame);
  }

  return 0;
}
