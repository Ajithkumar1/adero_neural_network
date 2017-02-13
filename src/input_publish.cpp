#include <stdio.h>
#include <stdlib.h>

#include "ros/ros.h"

#include "adero_neural_network/Input.h"
#include <vector>

using namespace std;

int main(int argc, char **argv)
{


	ros::init(argc, argv, "arrayPublisher");

	ros::NodeHandle n;

	ros::Publisher pub = n.advertise<adero_neural_network::Input>("training_pattern", 100);

	while (ros::ok())
	{
		adero_neural_network::Input inputs;
		//Clear array
		inputs.input.clear();
		inputs.target.clear();
		//for loop, pushing data in the size of the array

		inputs.number = 4;

		static const float arr[] = {0,1,0,1};
		vector<float> vec (arr, arr + sizeof(arr) / sizeof(arr[0]) );

		static const float arr2[] = {0.8,0.9,0.7,0.85};
		vector<float> vec2 (arr2, arr2 + sizeof(arr2) / sizeof(arr2[0]) );

		inputs.input = vec;
		inputs.target = vec2;

		//Publish array
		pub.publish(inputs);
		//Let the world know
		ROS_INFO("Inputs publish");
		//Do this.
		ros::spinOnce();
		//Added a delay so not to spam
		sleep(20);
	}

}