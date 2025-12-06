#include <ros/ros.h>
#include <unitree_legged_msgs/HighCmd.h>
#include <unitree_legged_msgs/HighState.h>
#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include "convert.h"
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32.h>

using namespace UNITREE_LEGGED_SDK;
int command = 5;

void torqueCallback(const std_msgs::Int32::ConstPtr& msg) {
    command = msg->data;
}



int main(int argc, char **argv)
{
    ros::init(argc, argv, "example_walk_without_lcm");

    std::cout << "WARNING: Control level is set to HIGH-level." << std::endl
              << "Make sure the robot is standing on the ground." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    ros::NodeHandle nh;

    ros::Rate loop_rate(500);

    long motiontime = 0;
    long motiontime1 = 0;
    long motiontime2 = 0;
    long motiontime3 = 0;
    long motiontime4 = 0;
    long motiontime5 = 0;
    long motiontime6 = 0;
    long motiontime7 = 0;
    long motiontime8 = 0;

    unitree_legged_msgs::HighCmd high_cmd_ros;

    ros::Publisher pub = nh.advertise<unitree_legged_msgs::HighCmd>("high_cmd", 1000);
    ros::Subscriber torque_sub = nh.subscribe("/command", 10, torqueCallback);
    while (ros::ok())
    {

        

        high_cmd_ros.head[0] = 0xFE;
        high_cmd_ros.head[1] = 0xEF;
        high_cmd_ros.levelFlag = HIGHLEVEL;
        high_cmd_ros.mode = 0;
        high_cmd_ros.gaitType = 0;
        high_cmd_ros.speedLevel = 0;
        high_cmd_ros.footRaiseHeight = 0;
        high_cmd_ros.bodyHeight = 0;
        high_cmd_ros.euler[0] = 0;
        high_cmd_ros.euler[1] = 0;
        high_cmd_ros.euler[2] = 0;
        high_cmd_ros.velocity[0] = 0.0f;
        high_cmd_ros.velocity[1] = 0.0f;
        high_cmd_ros.yawSpeed = 0.0f;
        high_cmd_ros.reserve = 0;
        
        
        
	if (command==7)
	{
	motiontime += 2;
	
        motiontime1 = 0;
        motiontime2 = 0;
        motiontime3 = 0;
        motiontime4 = 0;
        motiontime5 = 0;
        motiontime6 = 0;
        motiontime7 = 0;
        motiontime8 = 0;
        
        if (motiontime > 0 && motiontime < 1000)
        {
            high_cmd_ros.mode = 1;
            high_cmd_ros.euler[0] = -0.3;
        }
        if (motiontime > 1000 && motiontime < 2000)
        {
            high_cmd_ros.mode = 1;
            high_cmd_ros.euler[0] = 0.3;
        }
        if (motiontime > 2000 && motiontime < 3000)
        {
            high_cmd_ros.mode = 1;
            high_cmd_ros.euler[1] = -0.2;
        }
        if (motiontime > 3000 && motiontime < 4000)
        {
            high_cmd_ros.mode = 1;
            high_cmd_ros.euler[1] = 0.2;
        }
        if (motiontime > 4000 && motiontime < 5000)
        {
            high_cmd_ros.mode = 1;
            high_cmd_ros.euler[2] = -0.2;
        }
        if (motiontime > 5000 && motiontime < 6000)
        {
            high_cmd_ros.mode = 1;
            high_cmd_ros.euler[2] = 0.2;
        }
        if (motiontime > 13000 && motiontime < 14000)
        {
            high_cmd_ros.mode = 1;
        }
        if (motiontime > 14000 && motiontime < 15000)
        {
            high_cmd_ros.mode = 11;
        }
        }
        
        
        
        if (command==0)
        {
        motiontime2 += 2;
        motiontime1 = 0;
        motiontime = 0;
        motiontime3 = 0;
        motiontime4 = 0;
        motiontime5 = 0;
        motiontime6 = 0;
        motiontime7 = 0;
        motiontime8 = 0;
        
        if (motiontime2 > 0 && motiontime2 < 4000)
        {
            high_cmd_ros.mode = 2;
            high_cmd_ros.gaitType = 2;
            high_cmd_ros.velocity[0] = 0.4f; // -1  ~ +1
            
            // printf("walk\n");
        }else{
        high_cmd_ros.mode = 1;
        }
        }
        
        
        
        
        
        if (command==1)
        {
        motiontime1 += 2;
        motiontime = 0;
        motiontime2 = 0;
        motiontime3 = 0;
        motiontime4 = 0;
        motiontime5 = 0;
        motiontime6 = 0;
        motiontime7 = 0;
        motiontime8 = 0;
        if (motiontime1 > 0 && motiontime1 < 4000)
        {
            high_cmd_ros.mode = 2;
            high_cmd_ros.gaitType = 2;
            high_cmd_ros.velocity[0] = -0.4f; // -1  ~ +1
            
            // printf("walk\n");
        }else{
        high_cmd_ros.mode = 1;
        }
        }
        
        
        
        if (command==2)
        {
        motiontime3 += 2;
        motiontime1 = 0;
        motiontime2 = 0;
        motiontime = 0;
        motiontime4 = 0;
        motiontime5 = 0;
        motiontime6 = 0;
        motiontime7 = 0;
        motiontime8 = 0;
        
        if (motiontime3 > 0 && motiontime3 < 4000)
        {
            high_cmd_ros.mode = 2;
            high_cmd_ros.gaitType = 2;
            high_cmd_ros.velocity[1] = 0.4f; // -1  ~ +1
        
            // printf("walk\n");
        }else{
        high_cmd_ros.mode = 1;
        }
        }
        
        
        
        
        
        if (command==3)
        {
        motiontime4 += 2;
        motiontime1 = 0;
        motiontime2 = 0;
        motiontime3 = 0;
        motiontime = 0;
        motiontime5 = 0;
        motiontime6 = 0;
        motiontime7 = 0;
        motiontime8 = 0;
        
        if (motiontime4 > 0 && motiontime4 < 4000)
        {
            high_cmd_ros.mode = 2;
            high_cmd_ros.gaitType = 2;
            high_cmd_ros.velocity[1] = -0.4f; // -1  ~ +1
           
            // printf("walk\n");
        }else{
        high_cmd_ros.mode = 1;
        }
        }
        
        
        
        if (command==4)
        {
        motiontime6 += 2;
        motiontime1 = 0;
        motiontime2 = 0;
        motiontime3 = 0;
        motiontime4 = 0;
        motiontime5 = 0;
        motiontime = 0;
        motiontime7 = 0;
        motiontime8 = 0;
        if (motiontime6 > 0 && motiontime6 < 1000)
        {
            high_cmd_ros.mode = 5;
        }
        if (motiontime6 > 1000 && motiontime6 < 2000)
        {
            high_cmd_ros.mode = 7;
        }
        }
        
        
        
        
        
        if (command==8)
        {
        motiontime7 = 0;
        motiontime1 = 0;
        motiontime2 = 0;
        motiontime3 = 0;
        motiontime4 = 0;
        motiontime5 = 0;
        motiontime6 = 0;
        motiontime = 0;
        motiontime8 = 0;
        high_cmd_ros.mode = 0;
        }
        
        
        
        
        if (command==5)
        {
        motiontime7 += 2;
        motiontime1 = 0;
        motiontime2 = 0;
        motiontime3 = 0;
        motiontime4 = 0;
        motiontime5 = 0;
        motiontime6 = 0;
        motiontime = 0;
        motiontime8 = 0;
        
        
            high_cmd_ros.mode = 6;
            
        
        }
        
        
        if (command==6)
        {
        motiontime8 += 2;
        motiontime1 = 0;
        motiontime2 = 0;
        motiontime3 = 0;
        motiontime4 = 0;
        motiontime5 = 0;
        motiontime6 = 0;
        motiontime7 = 0;
        motiontime = 0;
        
        if (motiontime8 > 0 && motiontime8 < 1000)
        {
            high_cmd_ros.mode = 2;
            high_cmd_ros.gaitType = 2;
            high_cmd_ros.yawSpeed = 2;
          
            // printf("walk\n");
        }else{
        high_cmd_ros.mode = 1;
        }
        }
        
        
        
        
        
        
        
       

        pub.publish(high_cmd_ros);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
