#include <3DHoPD/3DHoPD.h>

vector<vector<double>> splitString(string str) {

    std::stringstream ss(str);

    vector<string> words;
    string word;

    // Read each word separated by spaces and store them in the vector
    while (ss >> word) {
        string temp = word;
        words.push_back(temp);
    }
    vector<vector<double>> res(4,vector<double>(4,0));

    int i = 0;

    for(int a = 0 ; a < 4 ; a++){
        for(int b = 0 ; b < 4 ; b++){
            double temp = stod(words[i++]);
            res[a][b] = temp;
        }
    }


    // cout<<"String array :-\n";
    // // Print the split words
    // for (const std::string& w : words) {
    //     cout << w << endl;
    // }

    return res;
}

double frobeniusNorm(vector<vector<double>> matrix) {
    double norm = 0.0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            norm += matrix[i][j] * matrix[i][j];
        }
    }
    return sqrt(norm);
}

double compareMatrices(vector<vector<double>> matrix1, vector<vector<double>> matrix2) {
    // Calculate the Frobenius norm of the difference between the matrices
    vector<vector<double>> difference(4,vector<double>(4,0));

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            difference[i][j] = matrix1[i][j] - matrix2[i][j];
        }
    }

    double norm = frobeniusNorm(difference);
    return norm;
}

class Mario {
    public:
        string name = "Mario";
        vector<string> objects = {".\\objects\\mario\\mario000.ply.pcd",".\\objects\\mario\\mario001.ply.pcd",".\\objects\\mario\\mario002.ply.pcd",".\\objects\\mario\\mario003.ply.pcd",".\\objects\\mario\\mario004.ply.pcd",".\\objects\\mario\\mario005.ply.pcd",".\\objects\\mario\\mario006.ply.pcd",".\\objects\\mario\\mario007.ply.pcd",".\\objects\\mario\\mario008.ply.pcd",".\\objects\\mario\\mario009.ply.pcd",".\\objects\\mario\\mario010.ply.pcd",".\\objects\\mario\\mario011.ply.pcd",".\\objects\\mario\\mario013.ply.pcd"};
        vector<string> scenes = {".\\scenes\\scene001.ply.pcd",".\\scenes\\scene003.ply.pcd",".\\scenes\\scene006.ply.pcd",".\\scenes\\scene010.ply.pcd",".\\scenes\\scene014.ply.pcd",".\\scenes\\scene019.ply.pcd",".\\scenes\\scene031.ply.pcd",".\\scenes\\scene032.ply.pcd",".\\scenes\\scene036.ply.pcd",".\\scenes\\sceneTuning.ply.pcd"};
        vector<string> check = {".\\check\\scene001_mario.txt",".\\check\\scene003_mario.txt",".\\check\\scene006_mario.txt",".\\check\\scene010_mario.txt",".\\check\\scene014_mario.txt",".\\check\\scene019_mario.txt",".\\check\\scene031_mario.txt",".\\check\\scene032_mario.txt",".\\check\\scene036_mario.txt",".\\check\\sceneTuning_mario.txt"};
};

class  Doll {
    public:
        string name = "Doll";
        vector<string> objects = {".\\objects\\doll\\Doll005.ply.pcd",".\\objects\\doll\\Doll006.ply.pcd",".\\objects\\doll\\Doll007.ply.pcd",".\\objects\\doll\\Doll008.ply.pcd",".\\objects\\doll\\Doll009.ply.pcd",".\\objects\\doll\\Doll010.ply.pcd",".\\objects\\doll\\Doll011.ply.pcd",".\\objects\\doll\\Doll012.ply.pcd",".\\objects\\doll\\Doll013.ply.pcd",".\\objects\\doll\\Doll014.ply.pcd",".\\objects\\doll\\Doll015.ply.pcd",".\\objects\\doll\\Doll016.ply.pcd",".\\objects\\doll\\Doll017.ply.pcd",".\\objects\\doll\\Doll018.ply.pcd",".\\objects\\doll\\Doll019.ply.pcd"};
        vector<string> scenes = {".\\scenes\\scene006.ply.pcd",".\\scenes\\scene010.ply.pcd",".\\scenes\\scene013.ply.pcd",".\\scenes\\scene022.ply.pcd",".\\scenes\\scene030.ply.pcd",".\\scenes\\scene039.ply.pcd",".\\scenes\\sceneTuning.ply.pcd"};
        vector<string> check = {".\\check\\scene006_doll.txt",".\\check\\scene010_doll.txt",".\\check\\scene013_doll.txt",".\\check\\scene022_doll.txt",".\\check\\scene030_doll.txt",".\\check\\scene039_doll.txt",".\\check\\sceneTuning_doll.txt"};
};

class Duck {
    public:
        string name = "Duck";
        vector<string> objects = {".\\objects\\duck\\duck000.ply.pcd",".\\objects\\duck\\duck001.ply.pcd",".\\objects\\duck\\duck002.ply.pcd",".\\objects\\duck\\duck003.ply.pcd",".\\objects\\duck\\duck004.ply.pcd",".\\objects\\duck\\duck005.ply.pcd",".\\objects\\duck\\duck006.ply.pcd",".\\objects\\duck\\duck007.ply.pcd",".\\objects\\duck\\duck008.ply.pcd",".\\objects\\duck\\duck009.ply.pcd",".\\objects\\duck\\duck010.ply.pcd",".\\objects\\duck\\duck011.ply.pcd",".\\objects\\duck\\duck012.ply.pcd",".\\objects\\duck\\duck013.ply.pcd",".\\objects\\duck\\duck017.ply.pcd",".\\objects\\duck\\duck018.ply.pcd"};
        vector<string> scenes = {".\\scenes\\scene011.ply.pcd",".\\scenes\\scene013.ply.pcd",".\\scenes\\scene014.ply.pcd"};
        vector<string> check = {".\\check\\scene011_duck.txt",".\\check\\scene013_duck.txt",".\\check\\scene014_duck.txt"};
};

class Rabbit {
    public:
        string name = "Rabbit";
        vector<string> objects = {".\\objects\\peterRabbit\\PeterRabbit000.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit001.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit002.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit003.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit004.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit005.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit006.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit007.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit008.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit009.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit010.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit011.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit012.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit013.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit014.ply.pcd",".\\objects\\peterRabbit\\PeterRabbit015.ply.pcd"};
        vector<string> scenes = {".\\scenes\\scene001.ply.pcd",".\\scenes\\scene003.ply.pcd",".\\scenes\\scene011.ply.pcd",".\\scenes\\scene018.ply.pcd",".\\scenes\\scene019.ply.pcd",".\\scenes\\scene022.ply.pcd",".\\scenes\\scene032.ply.pcd",".\\scenes\\scene036.ply.pcd",".\\scenes\\scene039.ply.pcd",".\\scenes\\sceneTuning.ply.pcd"};
        vector<string> check = {".\\check\\scene001_rabbit.txt",".\\check\\scene003_rabbit.txt",".\\check\\scene011_rabbit.txt",".\\check\\scene018_rabbit.txt",".\\check\\scene019_rabbit.txt",".\\check\\scene022_rabbit.txt",".\\check\\scene032_rabbit.txt",".\\check\\scene036_rabbit.txt",".\\check\\scene039_rabbit.txt",".\\check\\sceneTuning_rabbit.txt"};
};
class Squirrel {
    public:
        string name = "Squirrel";
        vector<string> objects = {".\\objects\\squirrel\\Squirrel000.ply.pcd",".\\objects\\squirrel\\Squirrel001.ply.pcd",".\\objects\\squirrel\\Squirrel002.ply.pcd",".\\objects\\squirrel\\Squirrel003.ply.pcd",".\\objects\\squirrel\\Squirrel004.ply.pcd",".\\objects\\squirrel\\Squirrel005.ply.pcd",".\\objects\\squirrel\\Squirrel006.ply.pcd",".\\objects\\squirrel\\Squirrel007.ply.pcd",".\\objects\\squirrel\\Squirrel008.ply.pcd",".\\objects\\squirrel\\Squirrel009.ply.pcd",".\\objects\\squirrel\\Squirrel010.ply.pcd",".\\objects\\squirrel\\Squirrel011.ply.pcd",".\\objects\\squirrel\\Squirrel012.ply.pcd",".\\objects\\squirrel\\Squirrel013.ply.pcd",".\\objects\\squirrel\\Squirrel015.ply.pcd"};
        vector<string> scenes = {".\\scenes\\scene001.ply.pcd",".\\scenes\\scene003.ply.pcd",".\\scenes\\scene006.ply.pcd",".\\scenes\\scene013.ply.pcd",".\\scenes\\scene014.ply.pcd",".\\scenes\\scene030.ply.pcd",".\\scenes\\scene031.ply.pcd",".\\scenes\\scene032.ply.pcd",".\\scenes\\scene036.ply.pcd"};
        vector<string> check = {".\\check\\scene001_squirrel.txt",".\\check\\scene003_squirrel.txt",".\\check\\scene006_squirrel.txt",".\\check\\scene013_squirrel.txt",".\\check\\scene014_squirrel.txt",".\\check\\scene030_squirrel.txt",".\\check\\scene031_squirrel.txt",".\\check\\scene032_squirrel.txt",".\\check\\scene036_squirrel.txt"};
};
class Frog {
    public:
        string name = "Frog";
        vector<string> objects = {".\\objects\\frog\\Frog000.ply.pcd",".\\objects\\frog\\Frog001.ply.pcd",".\\objects\\frog\\Frog002.ply.pcd",".\\objects\\frog\\Frog003.ply.pcd",".\\objects\\frog\\Frog004.ply.pcd",".\\objects\\frog\\Frog005.ply.pcd",".\\objects\\frog\\Frog006.ply.pcd",".\\objects\\frog\\Frog007.ply.pcd",".\\objects\\frog\\Frog008.ply.pcd",".\\objects\\frog\\Frog009.ply.pcd",".\\objects\\frog\\Frog010.ply.pcd",".\\objects\\frog\\Frog011.ply.pcd",".\\objects\\frog\\Frog012.ply.pcd",".\\objects\\frog\\Frog013.ply.pcd",".\\objects\\frog\\Frog014.ply.pcd",".\\objects\\frog\\Frog015.ply.pcd",".\\objects\\frog\\Frog016.ply.pcd",".\\objects\\frog\\Frog017.ply.pcd",".\\objects\\frog\\Frog018.ply.pcd",".\\objects\\frog\\Frog019.ply.pcd"};
        vector<string> scenes = {".\\scenes\\scene018.ply.pcd",".\\scenes\\scene019.ply.pcd",".\\scenes\\scene022.ply.pcd",".\\scenes\\scene030.ply.pcd",".\\scenes\\scene031.ply.pcd",".\\scenes\\scene032.ply.pcd",".\\scenes\\scene039.ply.pcd"};
        vector<string> check = {".\\check\\scene018_frog.txt",".\\check\\scene019_frog.txt",".\\check\\scene022_frog.txt",".\\check\\scene030_frog.txt",".\\check\\scene031_frog.txt",".\\check\\scene032_frog.txt",".\\check\\scene039_frog.txt"};
};

int main(int argc,char** argv)
{

    // Mario obj;
    // Duck obj;
    // Doll obj;
    // Frog obj;
    // Squirrel obj;
    Rabbit obj;

    int scenesCount = obj.scenes.size();
    int objectCount = obj.objects.size();

    cout<<"\n\nObject = "<<obj.name<<endl;
    cout<<"Total scenes = "<<scenesCount<<endl;
    cout<<"Total objects = "<<objectCount<<endl;

    for(double ds = 0.06; ds > 0 ; ds -= 0.01){
        
        double accSum = 0;
        double timeSum = 0;

        for(int j = 0 ; j < objectCount ; j++){

            int match = 0;
            double tempTime = 0;

            // cout<<"\n\nObject "<<j<<endl;

            for(int i = 0 ; i < scenesCount ; i++){

                // cout<<"\nScene "<<i<<endl;

                pcl::PointCloud<pcl::PointXYZ> cloud2, cloud1;

                // pcl::io::loadPCDFile<pcl::PointXYZ>(scene, cloud2);          // Scene 1
                pcl::io::loadPCDFile<pcl::PointXYZ>(obj.scenes[i], cloud2);          // Scene 1

                //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/PeterRabbit001_0.pcd", cloud1);  // Model 1 for Scene 1
                //pcl::io::loadPCDFile<pcl::PointXYZ>("../sample_data/Doll018_0.pcd", cloud1);         // Model 2 for Scene 1
                // pcl::io::loadPCDFile<pcl::PointXYZ>(obj, cloud1);        // Model 3 for Scene 1
                pcl::io::loadPCDFile<pcl::PointXYZ>(obj.objects[j], cloud1);        // Model 3 for Scene 1

                threeDHoPD RP1, RP2;


                // Using Simple Uniform Keypoint Detection, instead ISS keypoints can also be used!

                RP1.cloud = cloud1; // Model
                RP1.detect_uniform_keypoints_on_cloud(0.01);
                // cout << "Keypoints on Model: " << RP1.cloud_keypoints.size() << endl;

                RP2.cloud = cloud2; // Scene
                RP2.detect_uniform_keypoints_on_cloud(0.02);
                // cout << "Keypoints on Scene: " << RP2.cloud_keypoints.size() << endl;



                clock_t start1, end1;
                double cpu_time_used1;
                start1 = clock();

                // setup
                RP1.kdtree.setInputCloud(RP1.cloud.makeShared());// 
                RP2.kdtree.setInputCloud(RP2.cloud.makeShared());// 

                RP1.JUST_REFERENCE_FRAME_descriptors(ds); // this is where descriptors are being built
                RP2.JUST_REFERENCE_FRAME_descriptors(ds);

                end1 = clock();
                cpu_time_used1 = ((double) (end1 - start1)) / CLOCKS_PER_SEC;
                // cout <<  "Time taken for Feature Descriptor Extraction: " << (double)cpu_time_used1 << "\n";

                // tTime += (double)cpu_time_used1;

                // time
                tempTime += (double)cpu_time_used1;

                pcl::Correspondences corrs;

                clock_t start_shot2, end_shot2;
                double cpu_time_used_shot2;
                start_shot2 = clock();

                // KD Tree of scene
                pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_LRF;
                pcl::PointCloud<pcl::PointXYZ> pcd_LRF;
                for (int i = 0; i < RP2.cloud_LRF_descriptors.size(); i++)
                {
                    pcl::PointXYZ point;
                    point.x = RP2.cloud_LRF_descriptors[i].vector[0];
                    point.y = RP2.cloud_LRF_descriptors[i].vector[1];
                    point.z = RP2.cloud_LRF_descriptors[i].vector[2];

                    pcd_LRF.push_back(point);
                }



                // KD Tree of Model
                kdtree_LRF.setInputCloud(pcd_LRF.makeShared());
                for (int i = 0; i < RP1.cloud_LRF_descriptors.size(); i++)
                {
                    pcl::PointXYZ searchPoint;
                    searchPoint.x = RP1.cloud_LRF_descriptors[i].vector[0];
                    searchPoint.y = RP1.cloud_LRF_descriptors[i].vector[1];
                    searchPoint.z = RP1.cloud_LRF_descriptors[i].vector[2];

                    std::vector<int> nn_indices;
                    std::vector<float> nn_sqr_distances;

                    std::vector<double> angles_vector;

                float threshold_to_remove_false_positives = 0.075; // Changed, init = 0.0075
                    if (kdtree_LRF.radiusSearch(searchPoint,threshold_to_remove_false_positives, nn_indices,nn_sqr_distances) > 0)// Based on this threshold, false positives are removed!
                    {
                        for (int j = 0; j < nn_indices.size(); j++)
                        {
                            {
                                Eigen::VectorXf vec1, vec2;
                                vec1.resize(23); vec2.resize(23); // desc size = 50 new


                                for (int k = 0; k < 23; k++)// then nearest neighbour based matching with HoPD
                                {
                                    vec1[k] = RP1.cloud_distance_histogram_descriptors[i].vector[k];
                                    vec2[k] = RP2.cloud_distance_histogram_descriptors[nn_indices[j]].vector[k];
                                //   cout<<vec1[k]<<endl;

                                }
                                //cout<<"aaaaa"<<endl;

                                double dist = (vec1-vec2).norm();
                                angles_vector.push_back(dist);

                            }
                        }



                        std::vector<double>::iterator result;
                        result = std::min_element(angles_vector.begin(), angles_vector.end());

                        int min_element_index = std::distance(angles_vector.begin(), result);

                        pcl::Correspondence corr;
                        corr.index_query = RP1.patch_descriptor_indices[i];
                        corr.index_match = RP2.patch_descriptor_indices[nn_indices[min_element_index]];

                        corrs.push_back(corr);

                    }
                }


                end_shot2 = clock();
                cpu_time_used_shot2 = ((double) (end_shot2 - start_shot2)) / CLOCKS_PER_SEC;


                // cout << "No. of Reciprocal Correspondences : " << corrs.size() << endl;


                // RANSAC based false matches removal
                pcl::CorrespondencesConstPtr corrs_const_ptr = boost::make_shared< pcl::Correspondences >(corrs);

                pcl::Correspondences corr_shot;
                pcl::registration::CorrespondenceRejectorSampleConsensus< pcl::PointXYZ > Ransac_based_Rejection_shot;
                Ransac_based_Rejection_shot.setInputSource(RP1.cloud_keypoints.makeShared());
                Ransac_based_Rejection_shot.setInputTarget(RP2.cloud_keypoints.makeShared());
                Ransac_based_Rejection_shot.setInlierThreshold(0.035); // Changed, init = 0.02
                Ransac_based_Rejection_shot.setInputCorrespondences(corrs_const_ptr);
                Ransac_based_Rejection_shot.getCorrespondences(corr_shot);


                // Printing the transformation matrix
                Eigen::Matrix<float,4,4,0,4,4> tpMatrix = Ransac_based_Rejection_shot.getBestTransformation();

                // cout << "Transformation Matrix : \n" << endl;
                // cout << tpMatrix << endl;

                // cout<<"Type :-\n";
                // cout<< typeid(Ransac_based_Rejection_shot.getBestTransformation()).name()<<endl;

                std::stringstream matrixStream;

                matrixStream << tpMatrix;

                // Convert the matrix stream to a string
                std::string matrixStr = matrixStream.str();

                // Print the string representation of the matrix
                // std::cout << "Matrix as string:\n" << matrixStr << std::endl;

                // Reading data from .xf file
                std::ifstream inputFile(obj.check[i]);

                if (!inputFile.is_open()) {
                    std::cerr << "Failed to open the file." << std::endl;
                    return 1;
                }
                
                std::stringstream contentStream;
                
                contentStream << inputFile.rdbuf();
                
                inputFile.close();
                
                std::string content = contentStream.str();
                // std::cout << "File Content:\n" << content << std::endl;

                // formatting truth matrix
                vector<vector<double>> givenTransMatrix = splitString(content);

                // cout<<"Truth matrix :-"<<endl;

                // for(int a = 0 ; a < 4 ; a++){
                //     for(int b = 0 ; b < 4 ; b++){
                //         cout<<givenTransMatrix[a][b]<<" ";
                //     }
                //     cout<<endl;
                // }

                // formatting predicted matrix
                vector<vector<double>> predictedTransMatrix = splitString(matrixStr);

                // cout<<"predicted matrix :-"<<endl;

                // for(int a = 0 ; a < 4 ; a++){
                //     for(int b = 0 ; b < 4 ; b++){
                //         cout<<predictedTransMatrix[a][b]<<" ";
                //     }
                //     cout<<endl;
                // }

                // cout << "True correspondences after RANSAC : " << corr_shot.size() << endl;

                double diff = compareMatrices(givenTransMatrix,predictedTransMatrix);

                // cout<<"difference = "<<diff<<endl;

                if(diff <= 2.0){
                    match++;
                }

                // Visualization
                // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
                // viewer->setBackgroundColor (255, 255, 255);



                // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color1(RP1.cloud_keypoints.makeShared(), 255, 0, 0);
                // viewer->addPointCloud<pcl::PointXYZ> (RP1.cloud_keypoints.makeShared(), single_color1, "sample cloud1");
                // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");

                // viewer->initCameraParameters ();

                // Eigen::Matrix4f t;
                // t<<1,0,0,0.4,
                //         0,1,0,0,
                //         0,0,1,0,
                //         0,0,0,1;


                // pcl::transformPointCloud(RP2.cloud_keypoints,RP2.cloud_keypoints,t);


                // pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color2(RP2.cloud_keypoints.makeShared(), 0, 0, 255);
                // viewer->addPointCloud<pcl::PointXYZ> (RP2.cloud_keypoints.makeShared(), single_color2, "sample cloud2");
                // viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud2");


                // viewer->addCorrespondences<pcl::PointXYZ>(RP1.cloud_keypoints.makeShared(), RP2.cloud_keypoints.makeShared(), corr_shot, "correspondences");


                // while (!viewer->wasStopped ())
                // {
                //     viewer->spinOnce (100);
                //     boost::this_thread::sleep (boost::posix_time::microseconds (100000));
                // }

                // cout<<"DONE !!"<<endl;
            }

            double currAcc = (1.0*match/scenesCount)*100;
            double currTime = 1.0*tempTime/scenesCount;
            
            // cout<<"\nCurr Obj Acc = "<<currAcc<<" %\n";
            // cout<<"Curr Obj Time = "<<currTime<<" sec\n";

            accSum += currAcc;
            timeSum += currTime;
        }


        double acc = (accSum/objectCount);
        double timeTaken = (timeSum/objectCount);

        cout<<"---------------------------------------------------"<<endl;
        cout<<"d size = "<<ds<<endl;
        cout<<"\nFinal Accuracy = "<<acc<<" %\n";
        cout<<"Final Time = "<<timeTaken<<" sec"<<endl;
        cout<<"-------------------------------------------------------"<<endl;
    }

    return 0;
}
