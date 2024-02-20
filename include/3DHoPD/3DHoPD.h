// some of source code for LRF estimation is taken from PCL's SHOT descriptor. Thanks to the developers!
#include <iostream>
#include <string>
#include <bitset>
#include <vector>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <valarray>
// #include <sys/time.h>
#include <ctime>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/features/normal_3d_omp.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
// #include <pcl/filters/filter.h>
// #include <pcl/filters/filter.hpp>
 

// #include <pcl/io/pcd_io.h>
// #include <pcl/point_types.h>
// #include <pcl/PointIndices.h>
// #include <pcl/sample_consensus/ransac.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/registration/correspondence_rejection_sample_consensus.h>
// #include <pcl/registration/correspondence_estimation.h>
// #include <pcl/visualization/histogram_visualizer.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/visualization/cloud_viewer.h>
// #include <pcl/features/normal_3d_omp.h>
// #include <boost/thread/thread.hpp>
// #include <pcl/common/common_headers.h>
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/console/parse.h>
// #include <pcl/point_cloud.h>
// #include <pcl/kdtree/kdtree_flann.h>
// #include <pcl/kdtree/impl/kdtree_flann.hpp>

// #include <pcl/surface/convex_hull.h>
// #include <pcl/filters/uniform_sampling.h>

using namespace std;
using namespace Eigen;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

typedef pcl::PointXYZ PointType;

class patch_descriptor
{
public:
    std::vector<float> vector;
};

class histogram_descriptor
{
public:
    std::vector<float> vector;
};

class threeDHoPD
{
public:
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::PointCloud<pcl::PointXYZ> cloud_keypoints;

    pcl::PointCloud<pcl::Normal> cloud_normals;
   // pcl::PointCloud<pcl::Normal>::Ptr cloud_normals ( pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<pcl::PrincipalCurvatures> principal_curvatures;

    std::vector<patch_descriptor> cloud_patch_descriptors;
    std::vector<patch_descriptor> cloud_LRF_descriptors;
    std::vector<patch_descriptor> cloud_distance_histogram_descriptors;

    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    std::vector<int> cloud_keypoints_indices;
    std::vector<int> patch_descriptor_indices;

    int K0 = 5;
    int K1 = 5; // 50
    float a1, b1, c1, a2, b2, c2, a3, b3, c3, a4, b4, c4;
    float n1;
    float r1;
    float direction1;
    float path1;
    float n2;
    float r2;
    float direction2;
    float path2;
    int count_nan = 0;  int count_nan1=0;
    void detect_uniform_keypoints_on_cloud(float keypoint_radius)
    {
        // pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        // voxel_grid.setLeafSize(keypoint_radius, keypoint_radius, keypoint_radius);
        // voxel_grid.setInputCloud(cloud.makeShared());
        // voxel_grid.filter(cloud_keypoints);
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        std::vector<int> indices_a;
        pcl::removeNaNFromPointCloud(*cloud.makeShared(), *cloud.makeShared(), indices_a);

        ne.setInputCloud(cloud.makeShared());
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());
        ne.setSearchMethod(tree_n);
      //  ne.setRadiusSearch(0.035); /// normal estimation 0.015 //0.02
       ne.setKSearch(10);
        ne.compute(cloud_normals);
      

        pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
        voxel_grid.setLeafSize(keypoint_radius, keypoint_radius, keypoint_radius);
        voxel_grid.setInputCloud(cloud.makeShared());
        voxel_grid.filter(cloud_keypoints);
    }

    void JUST_REFERENCE_FRAME_descriptors(float patch_radius)
    {
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_un1;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_un2;
        kdtree_un1.setInputCloud(cloud.makeShared());
        kdtree_un2.setInputCloud(cloud.makeShared());

        for (int i = 0; i < cloud_keypoints.size(); i++)
        {
            pcl::PointXYZ currentPoint = cloud_keypoints[i];

            std::vector<int> nn_indices;
            std::vector<float> nn_sqr_distances;

            if (kdtree.radiusSearch(currentPoint, patch_radius, nn_indices, nn_sqr_distances) > 0)
            {
                pcl::PointCloud<pcl::PointXYZ> raw_patch;
                pcl::copyPointCloud(cloud, nn_indices, raw_patch);

                // cout << "size: "<< raw_patch.size() << endl;

                Eigen::Vector4f mean_raw_patch;
                pcl::compute3DCentroid(raw_patch, mean_raw_patch);

                for (int j = 0; j < raw_patch.size(); j++)
                {
                    raw_patch.points[j].x = raw_patch.points[j].x - mean_raw_patch[0];
                    raw_patch.points[j].y = raw_patch.points[j].y - mean_raw_patch[1];
                    raw_patch.points[j].z = raw_patch.points[j].z - mean_raw_patch[2];
                }
                // Important to remove the min_raw_patch from currentPoint
                currentPoint.x = currentPoint.x - mean_raw_patch[0];
                currentPoint.y = currentPoint.y - mean_raw_patch[1];
                currentPoint.z = currentPoint.z - mean_raw_patch[2];

                Eigen::Matrix3f LRF;
                get_local_rf(currentPoint, patch_radius, raw_patch, LRF);

                Eigen::Vector3f point_here = LRF * currentPoint.getVector3fMap();

                patch_descriptor pt_LRF;

                // cout << transformedPoint.x << endl;
                pt_LRF.vector.push_back(point_here[0]);
                pt_LRF.vector.push_back(point_here[1]);
                pt_LRF.vector.push_back(point_here[2]);

                cloud_LRF_descriptors.push_back(pt_LRF);

                patch_descriptor_indices.push_back(i);

                Eigen::Matrix4f TRX = Eigen::Matrix4f::Identity();
                TRX(0, 0) = LRF(0, 0);
                TRX(0, 1) = LRF(0, 1);
                TRX(0, 2) = LRF(0, 2);
                TRX(1, 0) = LRF(1, 0);
                TRX(1, 1) = LRF(1, 1);
                TRX(1, 2) = LRF(1, 2);
                TRX(2, 0) = LRF(2, 0);
                TRX(2, 1) = LRF(2, 1);
                TRX(2, 2) = LRF(2, 2);

                transformPointCloud(raw_patch, raw_patch, TRX);

                //////////////////////////////////////////////////////////////////////////////////////////

  patch_descriptor hist_xyz;   patch_descriptor hist_xyz_n;
 float min_x, min_y, min_z, max_x, max_y, max_z; min_x = min_y = min_z = max_x = max_y = max_z = 0;
                for (int j = 0; j < raw_patch.size(); j++)
                {
                    if (min_x > raw_patch.points[j].x)
                        min_x = raw_patch.points[j].x;
                    if (min_y > raw_patch.points[j].y)
                        min_y = raw_patch.points[j].y;
                    if (min_z > raw_patch.points[j].z)
                        min_z = raw_patch.points[j].z;
                    if (max_x < raw_patch.points[j].x)
                        max_x = raw_patch.points[j].x;
                    if (max_y < raw_patch.points[j].y)
                        max_y = raw_patch.points[j].y;
                    if (max_z < raw_patch.points[j].z)
                        max_z = raw_patch.points[j].z;
                }

                std::vector<float> hist_x(5,0); std::vector<float> hist_y(5,0); std::vector<float> hist_z(5,0);
                for (int j = 0; j < raw_patch.size(); j++)
                {
                    //cout << "Index : " <<  std::floor( 9.0*(float)(std::abs(min_x) + raw_patch.points[j].x)/(float)(max_x-min_x)) << endl;
                    hist_x[std::floor( 4.0*(float)(std::abs(min_x) + raw_patch.points[j].x)/(float)(max_x-min_x)) ]++;
                    hist_y[std::floor( 4.0*(float)(std::abs(min_y) + raw_patch.points[j].y)/(float)(max_y-min_y)) ]++;
                    hist_z[std::floor( 4.0*(float)(std::abs(min_z) + raw_patch.points[j].z)/(float)(max_z-min_z)) ]++;
                }

                float acc_norm_x, acc_norm_y, acc_norm_z; acc_norm_x = acc_norm_y = acc_norm_z = 0;
                for (int j = 0; j < 5; j++)
                {
                    acc_norm_x += hist_x[j] * hist_x[j];
                    acc_norm_y += hist_y[j] * hist_y[j];
                    acc_norm_z += hist_z[j] * hist_z[j];
                }
                acc_norm_x = sqrt (acc_norm_x);
                acc_norm_y = sqrt (acc_norm_y);
                acc_norm_z = sqrt (acc_norm_z);

                for (int j = 0; j < 5; j++)
                {
                    hist_x[j] /= static_cast<float> (acc_norm_x);
                    hist_y[j] /= static_cast<float> (acc_norm_y);
                    hist_z[j] /= static_cast<float> (acc_norm_z);
                }

              //  patch_descriptor hist_xyz;
                for (int p = 0; p < 5; p++)
                {
                    hist_xyz.vector.push_back( hist_x[p]);
                    hist_xyz.vector.push_back(hist_y[p]);
                     hist_xyz.vector.push_back(hist_z[p]);

                }
               



                // New Descriptor

               
             		//////////////////////////////////////////////////////////////////////////////////////////////////////

                     std::vector<int> pointIdxNKNSearch1(K0);
                std::vector<float> pointNKNSquaredDistance1(K0);
                //	kdtree_un.setInputCloud(cloud);
                pcl::PointXYZ searchPoint_un1;
                searchPoint_un1.x = cloud_keypoints.points[i].x;
                searchPoint_un1.y = cloud_keypoints.points[i].y;
                searchPoint_un1.z = cloud_keypoints.points[i].z;
              
                pcl::PointXYZ currentPoint_un;
                
               
                 float charge1=0; double charge2=0; double patht=0; double mag=0;


                              std::vector<float> sum_path_full(9,0); int a=1;
            if (kdtree_un1.nearestKSearch(searchPoint_un1, K0, pointIdxNKNSearch1, pointNKNSquaredDistance1) > 0)
				{
					for (int i1 = 1; i1 <  pointIdxNKNSearch1.size(); i1++)
					{ 
                        
                 
                 a1 = pow((cloud.points[pointIdxNKNSearch1[0]].x - cloud.points[pointIdxNKNSearch1[i1]].x), 2);
					b1 = pow((cloud.points[pointIdxNKNSearch1[0]].y - cloud.points[pointIdxNKNSearch1[i1]].y ), 2);
					c1 = pow((cloud.points[pointIdxNKNSearch1[0]].z - cloud.points[pointIdxNKNSearch1[i1]].z), 2);

                   float path1 = (sqrt(a1 + b1 +c1)  );

				    charge2 = cloud.points[pointIdxNKNSearch1[i1]].x * cloud.points[pointIdxNKNSearch1[0]].x + cloud.points[pointIdxNKNSearch1[i1]].y *cloud.points[pointIdxNKNSearch1[0]].y
                   +cloud.points[pointIdxNKNSearch1[i1]].z * cloud.points[pointIdxNKNSearch1[0]].z  ;
									
					
                    	Eigen::Vector3f p ;//= cloud_normals.points[pointIdxNKNSearch1[0]].getNormalVector4fMap();
                        p[0] = cloud_normals.points[pointIdxNKNSearch1[0]].normal_x;
                        p[1] = cloud_normals.points[pointIdxNKNSearch1[0]].normal_y;
                        p[2] = cloud_normals.points[pointIdxNKNSearch1[0]].normal_z;
						Eigen::Vector3f q ;//= cloud_normals.points[pointIdxNKNSearch1[i1]].getNormalVector4fMap();
                        q[0] = cloud_normals.points[pointIdxNKNSearch1[i1]].normal_x;
                        q[1] = cloud_normals.points[pointIdxNKNSearch1[i1]].normal_y;
                        q[2] = cloud_normals.points[pointIdxNKNSearch1[i1]].normal_z;
                        Eigen::Vector3f pq ;  Eigen::Vector3f pqm ;

                        // cout<<"pppppp"<<p[0];
                        
                       if (isnan(p[0]) == 1  || isnan(p[1]) == 1  || isnan(p[2]) == 1    )
							{ 
                                p[0] = 0; p[1] =0; p[2] = 0; p[3] = 0;
                                cout<<"pppppp"<<p;



                            }
                            else if (isinf(p[0]) == 1  || isinf(p[1]) == 1  || isinf(p[2]) == 1    )
							{ 
                                p[0] = 0; p[1] =0; p[2] = 0; p[3] = 0;
                                cout<<"pppppp11111"<<q;


                            }


                           else if (isnan(q[0]) == 1  || isnan(q[1]) == 1  || isnan(q[2]) == 1     )
							{ 
                                q[0] = 0; q[1] = 0; q[2] = 0; q[3] = 0;
                               cout<<"qqqqq"<<q;
                            }
                           else if (isinf(q[0]) == 1  || isinf(q[1]) == 1  || isinf(q[2]) == 1   )
							{ 
                                q[0] = 0; q[1] =0; q[2] = 0; q[3] = 0;
                                cout<<"qqqqqq111"<<q;


                            }
                            
					                   
					//else
                    {
                        patht =log (  ( (q-p).dot((q-p)) ) + 0.00001);
                         if (isnan(patht) == 1)
							{ cout<<patht<<endl;
							patht=0;
                             cout<<"pathttt";
						    }
                     //   patht = (  ( (q-p).dot((q-p)) ) );
                        pqm = p-q;
                      mag = log(sqrt(pqm[0] *pqm[0] + pqm[1]*pqm[1] + pqm[2]*pqm[2] ) )   ;
                      if (isnan(mag) == 1)
							{ cout<<mag<<endl;
							mag=0;
                             cout<<"mag";
						    }
                       //  mag =  (sqrt(pqm[0] *pqm[0] + pqm[1]*pqm[1] + pqm[2]*pqm[2] + pqm[3]*pqm[3]) )   ;

                        {
                        
                      pq=   (     ( (charge2)*  ((q-p)/(mag+0.000001))   ) );// /( abs(patht)+0.000001 ) )  ;
                       sum_path_full[i1]=  log (  (sqrt(pq[0] *pq[0] + pq[1]*pq[1] + pq[2]*pq[2]   ) )  +0.00001 ) ;
                      // pq=   log(charge2+0.00001) ;
                        // sum_path_full[i1]=   (  (sqrt(pq[0] *pq[0] + pq[1]*pq[1] + pq[2]*pq[2] +pq[3]*pq[3]  ) )  ) ;

                         
                        }
                        
                     if (isinf(sum_path_full[i1]) == 1)
							{
                                cout<<sum_path_full[i1]<<endl;
							sum_path_full[i1]=0;
                            cout<<"a"<<sum_path_full[i1]<<endl;
                            count_nan1++;
                            
						    }
						    if (isnan(sum_path_full[i1]) == 1)
							{ cout<<sum_path_full[i1]<<endl;
							sum_path_full[i1]=0;
                             cout<<"a"<<sum_path_full[i1]<<endl;
                             count_nan1++;
						    }
                            
                           // cout<<sum_path_full[i1]<<endl;
        //  hist_xyz.vector.push_back(1)	;
	//	hist_xyz.vector.push_back(log(1/(charge2*charge2))+0.00001 )	;
        hist_xyz.vector.push_back(log((charge2*1))+0.00001 )	;
    // hist_xyz.vector.push_back(sum_path_full[i1])	;
                    }
		
                    }
                }
            /**/
            
			pcl::PointXYZ searchPoint_un2;

						searchPoint_un2.x = point_here[0];
						searchPoint_un2.y = point_here[1];
						searchPoint_un2.z = point_here[2];
						
						std::vector<int> pointIdxNKNSearch2(K1);
						std::vector<float> pointNKNSquaredDistance2(K1);
						
					//	cout << raw_patch.size()<< endl;
					if (kdtree_un2.nearestKSearch(searchPoint_un2, K1, pointIdxNKNSearch2, pointNKNSquaredDistance2) > 0)
						{
						for (int i2 = 0; i2 < pointIdxNKNSearch2.size()-1; i2++)
							{
                                //  cout<< raw_patch.points[i2].x <<endl;
                                  //cout<< raw_patch.points[i2].y<<endl;
                                   //cout<< raw_patch.points[i2].z<<endl;
                                    //cout<< "aaa"<<endl;
                      //          charge1 = raw_patch.points[i2].x * raw_patch.points[0].x + raw_patch.points[i2].y * raw_patch.points[0].y
                 //  +raw_patch.points[i2].z * raw_patch.points[0].z  ;
                   
							   charge1 = raw_patch.points[i2].x * point_here[0] + raw_patch.points[i2].y * point_here[1]
                  +raw_patch.points[i2].z * point_here[2] ;
			  hist_xyz.vector.push_back(log(charge1+0.00001));
			//////////////////////////////////////////////////////////////////////////////////////////////////////
			
			
			
			
			
			
			
					       }
				    }
                    






                 cloud_distance_histogram_descriptors.push_back(hist_xyz);
          
               

                /////////////////////////////////////////////////////////////////////////////////////////

            }

        }
 //cout<<"count_nan = "<<count_nan<<endl;
  //cout<<"count_nan1 = "<<count_nan1<<endl;
    }
                

    void get_local_rf(pcl::PointXYZ current_point, float lrf_radius, pcl::PointCloud<pcl::PointXYZ> &cloud_here, Eigen::Matrix3f &rf)
    {

        int current_point_idx;

        pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
        kdtree.setInputCloud(cloud_here.makeShared());

        std::vector<int> n_indices;
        std::vector<float> n_sqr_distances;

        if (kdtree.radiusSearch(current_point, lrf_radius, n_indices, n_sqr_distances) > 0)
        {
            current_point_idx = n_indices[0];
        }

        const Eigen::Vector4f &central_point = (cloud_here)[current_point_idx].getVector4fMap();

        pcl::PointXYZ searchPoint;
        searchPoint = cloud_here[current_point_idx];

        Eigen::Matrix<double, Eigen::Dynamic, 4> vij(n_indices.size(), 4);

        Eigen::Matrix3d cov_m = Eigen::Matrix3d::Zero();

        double distance = 0.0;
        double sum = 0.0;

        int valid_nn_points = 0;

        for (size_t i_idx = 0; i_idx < n_indices.size(); ++i_idx)
        {
            Eigen::Vector4f pt = cloud_here.points[n_indices[i_idx]].getVector4fMap();
            if (pt.head<3>() == central_point.head<3>())
                continue;

            // Difference between current point and origin
            vij.row(valid_nn_points).matrix() = (pt - central_point).cast<double>();
            vij(valid_nn_points, 3) = 0;

            distance = lrf_radius - sqrt(n_sqr_distances[i_idx]);

            // Multiply vij * vij'
            cov_m += distance * (vij.row(valid_nn_points).head<3>().transpose() * vij.row(valid_nn_points).head<3>());

            sum += distance;
            valid_nn_points++;
        }

        if (valid_nn_points < 5)
        {
            // PCL_ERROR ("[pcl::%s::getLocalRF] Warning! Neighborhood has less than 5 vertexes. Aborting Local RF computation of feature point (%lf, %lf, %lf)\n", "SHOTLocalReferenceFrameEstimation", central_point[0], central_point[1], central_point[2]);
            rf.setConstant(std::numeric_limits<float>::quiet_NaN());

            // cout <<"\n\n\n\ Something CRAZY is Happening dude!!! \n\n\n"<< endl;
        }

        cov_m /= sum;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov_m);

        const double &e1c = solver.eigenvalues()[0];
        const double &e2c = solver.eigenvalues()[1];
        const double &e3c = solver.eigenvalues()[2];

        if (!pcl_isfinite(e1c) || !pcl_isfinite(e2c) || !pcl_isfinite(e3c))
        {
            // PCL_ERROR ("[pcl::%s::getLocalRF] Warning! Eigenvectors are NaN. Aborting Local RF computation of feature point (%lf, %lf, %lf)\n", "SHOTLocalReferenceFrameEstimation", central_point[0], central_point[1], central_point[2]);
            rf.setConstant(std::numeric_limits<float>::quiet_NaN());

            // cout <<"\n\n\n\ Something CRAZY is Happening dude!!! \n\n\n"<< endl;
        }

        // Disambiguation
        Eigen::Vector4d v1 = Eigen::Vector4d::Zero();
        Eigen::Vector4d v3 = Eigen::Vector4d::Zero();
        v1.head<3>().matrix() = solver.eigenvectors().col(2);
        v3.head<3>().matrix() = solver.eigenvectors().col(0);

        int plusNormal = 0, plusTangentDirection1 = 0;
        for (int ne = 0; ne < valid_nn_points; ne++)
        {
            double dp = vij.row(ne).dot(v1);
            if (dp >= 0)
                plusTangentDirection1++;

            dp = vij.row(ne).dot(v3);
            if (dp >= 0)
                plusNormal++;
        }

        // TANGENT
        plusTangentDirection1 = 2 * plusTangentDirection1 - valid_nn_points;
        if (plusTangentDirection1 == 0)
        {
            int points = 5; // std::min(valid_nn_points*2/2+1, 11);
            int medianIndex = valid_nn_points / 2;

            for (int i = -points / 2; i <= points / 2; i++)
                if (vij.row(medianIndex - i).dot(v1) > 0)
                    plusTangentDirection1++;

            if (plusTangentDirection1 < points / 2 + 1)
                v1 *= -1;
        }
        else if (plusTangentDirection1 < 0)
            v1 *= -1;

        // Normal
        plusNormal = 2 * plusNormal - valid_nn_points;
        if (plusNormal == 0)
        {
            int points = 5; // std::min(valid_nn_points*2/2+1, 11);
            int medianIndex = valid_nn_points / 2;

            for (int i = -points / 2; i <= points / 2; i++)
                if (vij.row(medianIndex - i).dot(v3) > 0)
                    plusNormal++;

            if (plusNormal < points / 2 + 1)
                v3 *= -1;
        }
        else if (plusNormal < 0)
            v3 *= -1;

        rf.row(0).matrix() = v1.head<3>().cast<float>();
        rf.row(2).matrix() = v3.head<3>().cast<float>();
        rf.row(1).matrix() = rf.row(2).cross(rf.row(0));
    }

    // end of class
};
