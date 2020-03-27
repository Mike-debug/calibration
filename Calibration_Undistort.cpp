#include<bits/stdc++.h>
#include<opencv.hpp>
#include<calib3d/calib3d_c.h>
#include<imgproc/types_c.h>
#include<io.h>
#include<direct.h>
using namespace std;
using namespace cv;

/*
获取某个目录下制定文件类型的文件
将数组FileNames中原有元素清空
将所有符合要求的文件放置到FileNames数组中
*/
void GetFile(string& Path, string& Type, vector<string>& FileNames) {
	struct _finddata_t FileInfo;//查询到的文件信息
	intptr_t lfile = 0;//查询到的long类型文件句柄
	string sfile_r;//搜索文件时用到的正则表达式
	lfile = _findfirst(sfile_r.assign(Path).append("\\*").append(Type).c_str(), &FileInfo);
	if (lfile != -1L) {//查找存在符合要求的文件
		do {
			if (FileInfo.attrib != _A_SUBDIR) {//如果不是文件夹
				FileNames.push_back(sfile_r.assign(Path).append("\\").append(FileInfo.name));
			}
		} while (_findnext(lfile, &FileInfo) == 0);
	}
	else {//未查到符合要求的文件, 提示后退出
		_findclose(lfile);
		cout << "No such type files!" << endl;
		exit(0);
	}
	return;
}

/*
根据一组图片标定相机
要求输入图片的角点信息一致、像素尺寸一致
*/
void MyCalibration(
	vector<string>& FilesName,//文件名
	Size corner_size,//角点矩阵尺寸
	Size square_size,//角点矩形尺寸
	Mat& cameraMatrix,
	Mat& distCoeffs,
	vector<Mat>& rvecsMat,
	vector<Mat>& tvecsMat,
	Size& SizeImage
)
{
	ofstream fout("标定结果.txt");
	
	cout << "==================提取角点=============================================" << endl;
	int image_count = 0;
	Size image_size;//图像大小
	
	vector<Point2f> image_corner;
	vector<vector<Point2f> > image_corners;
	
	for (int i = 0; i < FilesName.size(); ++i) {
		++image_count;
		cout << "提取第 " << i + 1 << " 幅图像角点" << endl;
		cout << FilesName.at(i) << endl;
		Mat imageInput = imread(FilesName.at(i));//读取第i+1幅图像
		
		if (image_count == 1) {//获取图像像素矩阵尺寸
			image_size.width = imageInput.cols;
			image_size.height = imageInput.rows;
			SizeImage = Size(imageInput.cols, imageInput.rows);
		}

		bool ok = findChessboardCorners(imageInput, corner_size, image_corner, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE);
		string cornersfind("cornersfind");
		if (!ok) {
			cout << "  第 " << i+1 << " 张照片提取角点失败，请删除后，重新标定！" << endl; //找不到角点
			imshow("失败照片", imageInput);
			waitKey(1000);
			destroyWindow("失败照片");
		}
		else {
			Mat View_gray;
			//cout << "  第 " << i + 1 << " 张照片是 " << imageInput.channels() << " 位图" << endl;
			cvtColor(imageInput, View_gray, CV_RGB2GRAY);//如果输入是RGB图像，则输出为灰度图
			/*亚像素精细化*/
			cornerSubPix(View_gray, image_corner, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 20, 0.01));
			image_corners.push_back(image_corner);
			/*在图像上显示角点的位置*/
			drawChessboardCorners(View_gray, corner_size, image_corner, true);
			//显示找到的角点
			cout << "  保存显示找到的角点" << endl;
			
			//保存找到角点后的图片
			string filename(cornersfind);
			stringstream ss;
			string s;
			ss << i + 1;
			ss >> s;
			filename.append("\\"+cornersfind);
			filename.append(s);
			filename.append(".jpg");
			imwrite(filename.c_str(), View_gray);
			ss.clear();
			s.clear();
		}
	}
	cout << "==================提取角点完成=========================================" << endl << endl << endl;


	/*三维棋盘信息*/
	vector<vector<Point3f> > object_points_seq;
	for (int t = 0; t < image_count; ++t) {
		vector<Point3f> object_points;
		for (int i = 0; i < corner_size.height; ++i) {
			for (int j = 0; j < corner_size.width; ++j) {
				Point3f realpoint;
				/*假设标定板放在世界坐标系中z=0的平面上*/
				realpoint.x = i * square_size.width;
				realpoint.y = j * square_size.height;
				realpoint.z = 0;
				object_points.push_back(realpoint);
			}
		}
		object_points_seq.push_back(object_points);
	}
	
	/*运行标定函数*/
	cout << "==================相机标定=============================================" << endl;
	
	double err_first = calibrateCamera(
		object_points_seq, 
		image_corners, 
		image_size, 
		cameraMatrix, 
		distCoeffs, 
		rvecsMat, 
		tvecsMat,
		CV_CALIB_FIX_K3
	);
	
	fout << "重投影误差1：" << err_first << "像素" << endl << endl;
	cout << "==================标定完成=============================================" << endl << endl << endl;

	cout << "==================评价标定结果==========================================" << endl;
	double total_err = 0.0;//所有图像的平均误差的总和
	double err = 0.0;//每幅图像的平均误差
	double totalErr = 0.0;
	double totalPoints = 0.0;
	vector<Point2f> image_points_pro;//保存重新计算得到的投影点
	for (int i = 0; i < image_count; ++i) {
		projectPoints(
			object_points_seq.at(i), 
			rvecsMat.at(i), 
			tvecsMat.at(i), 
			cameraMatrix, 
			distCoeffs, 
			image_points_pro
		);
		err = norm(Mat(image_corners.at(i)), Mat(image_points_pro), NORM_L2);
		totalErr += err * err;
		totalPoints += object_points_seq.at(i).size();
		err /= object_points_seq.at(i).size();
		fout << "第" << i + 1 << "幅图像的平均误差：" << err << "像素" << endl;
		total_err += err;
	}
	fout << "重投影误差2：" << sqrt(totalErr / totalPoints) << "像素" << endl << endl;
	fout << "重投影误差3：" << total_err / image_count << "像素" << endl << endl;
	cout << "==================评价标定结果完成======================================" << endl << endl << endl;


	cout << "==================保存标定结果==========================================" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));/*保存每幅图像的旋转矩阵*/
	fout << "相机内参数矩阵：" << endl;
	fout << cameraMatrix << endl << endl;
	fout << "畸变系数：" << endl;
	fout << distCoeffs << endl << endl << endl;
	for (int i = 0; i < image_count; ++i) {
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << rvecsMat.at(i) << endl;
		/* 将旋转向量转换为相对应的旋转矩阵 */
		Rodrigues(rvecsMat.at(i), rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << tvecsMat.at(i) << endl << endl;
	}
	cout << "==================保存标定结果完成======================================" << endl << endl << endl;
	fout.close();
	return;
}

//根据标定相机的结果校正图片
void MyUndistort(
	vector<string>& FilesName, 
	Size image_size, 
	Mat& cameraMatrix, 
	Mat& distCoeffs
)
{
	Mat mapx = Mat(image_size, CV_32FC1);//X坐标重映射参数
	Mat mapy = Mat(image_size, CV_32FC1);//Y坐标重映射参数
	Mat R = Mat::eye(3, 3, CV_32F);

	cout << "==================保存矫正图像==========================================" << endl;
	string imageFileName;                  //校正后图像的保存路径
	stringstream ss;
	string s;
	string filename("undistorted");
	for (int i = 0; i < FilesName.size(); ++i) {
		Mat imageSource = imread(FilesName[i]);
		Mat newimage = imageSource.clone();
		
		//方法一：使用initUndistortRectifyMap和remap两个函数配合实现
		/*initUndistortRectifyMap(cameraMatrix,distCoeffs,R, Mat(),image_size,CV_32FC1,mapx,mapy);
		remap(imageSource,newimage,mapx, mapy, INTER_LINEAR);*/

		//方法二：不需要转换矩阵的方式，使用undistort函数实现
		undistort(imageSource, newimage, cameraMatrix, distCoeffs);

		

		ss << i + 1;
		ss >> s;
		filename.append("\\" + filename);
		filename.append(s);
		filename.append(".jpg");
		cout << filename << endl;
		imwrite(filename.c_str(), newimage);
		ss.clear();
		s.clear();
		filename.clear();
		filename.assign("undistorted");
	}

	cout << "==================保存结束==============================================" << endl << endl << endl;
	return;
}

int main(void)
{
	/*=====================文件获取========================================================*/
	string Image_Directory("exp1");
	string Image_Type(".jpg");
	vector<string> ImageNames;
	//获取文件
	GetFile(Image_Directory, Image_Type, ImageNames);

	/*=====================相机标定========================================================*/
	Size image_Size;
	Size board_size = Size(9, 6);//标定板上每行、列的角点数 
	Size square_size = Size(30, 30);//实际测量得到的标定板上每个棋盘格的物理尺寸，单位mm
	Mat cameraMatrix = Mat(3, 3, CV_32FC1, Scalar::all(0));//摄像机内参数矩阵
	Mat distCoeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));//摄像机的5个畸变系数：k1,k2,p1,p2,k3
	vector<Mat> rvecsMat;//存放所有图像的旋转向量，每一副图像的旋转向量为一个mat
	vector<Mat> tvecsMat;// 存放所有图像的平移向量，每一副图像的平移向量为一个mat
	MyCalibration(
		ImageNames, 
		board_size, 
		square_size, 
		cameraMatrix, 
		distCoeffs, 
		rvecsMat, 
		tvecsMat,
		image_Size
	);
	
	/*=====================图像校正========================================================*/
	MyUndistort(ImageNames, image_Size, cameraMatrix, distCoeffs);
	return 0;
}