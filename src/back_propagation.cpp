#include "ros/ros.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Float32.h"
#include "std_msgs/String.h"
#include "adero_neural_network/Input.h"
#include "adero_neural_network/ActualInput.h"
#include <iostream>

//made a seperate class for weights and I/O number specifications
class Weight
{
public:
	static float *weight_lower_;
	static float *weight_upper_;
	static int noc;
	static int nic;
};



float *Weight::weight_lower_;
float *Weight::weight_upper_;
int Weight::noc;
int Weight::nic;

//binary sigmoid activation function
float sigmoid(float x)
{
	float f;
	f=(1/(1+ exp(-x)));
	return f;
}

//derivative of binary sigmoid
float sigmoidDerivative(float x)
{
	float f;
	f=sigmoid(x)*(1-sigmoid(x));
	return f;
}

// for  performing training operation
// please have a look at "Fundamentals Of Neural Networks" by Laurene_Fausett. Thsi code exactly follows his conventions
// We didnt use the zeroth index value of most of the arrays. We are planning to use them for setting unique flags which
// can be used while working on complex ai system 
void arrayCallback(const adero_neural_network::Input::ConstPtr& in)
{

	Weight weight_object;
	int ni=in->number,no=in->number;
	int test[no+1];
	float x[ni+1];
	float t[no+1];
	float v[ni+1][ni];
	float w[ni+1][no];
	float z_in[ni+1];
	float z[ni+1];
	float summation = 0;
	float y_in[no];
	float y[no];
	float del_k[no+1];
	float del_w[ni][no];
	float alpha = 0.25;
	float del_inj[ni+1];
	float del_j[ni];
	float del_vij[ni][ni];
	float error_range;
	int check=1;


	// cout<<"No. of inputs";
	// cin>>ni;

	x[0]=1;
	z[0]=1;
	for(int i=1;i<=ni;i++)
	{
		x[i]=in->input[i-1];
	}

	//float x[4] = {0,1,1,0};

	// cout<<"No. of targets";
	// cin>>no;

	for(int i=1;i<=no;i++)
	{
		t[i]=in->target[i-1];
	}

	//float t[2]={0,0.8};


	for(int i=0;i<=ni;i++)
	{
		for(int j=1;j<=ni;j++)
		{
			v[i][j]=0;
			//cout<<"weight_v"<<i<<j<<"="<<v[i][j]<<endl;
		}
	}

	for(int j=0;j<=ni;j++)
	{
		for(int k=1;k<=no;k++)
		{
			w[j][k]=0;
			//cout<<"weight_w"<<j<<k<<"="<<w[j][k]<<endl;
		}
	}

	do{
		check = 1;
		//cout<<"do entered"<<endl;

		z[0]=1;
		z_in[0]=0;//check here

		for(int j=1;j<=ni;j++)
		{
			for(int i=1;i<=ni;i++)
			{
				summation += (x[i]*v[i][j]);//summation to zero
			}
			z_in[j]=v[0][j]+summation;
			z[j]=sigmoid(z_in[j]);
		}

		//upper layer starts



		summation = 0;
		for(int k=1;k<=no;k++)
		{
			for(int j=1;j<=ni;j++)
			{
				summation += (z[j]*w[j][k]);
			}
			y_in[k]=w[0][k]+summation;
			y[k]=sigmoid(y_in[k]);
		}
		//upper layer ends
		
		//test starts
		for(int p=1;p<=no;p++)
		{

			ROS_INFO("Target : %f",t[p]);
		}
		for(int p=1;p<=no;p++)
		{

			ROS_INFO("output : %f",y[p]);

		}

		for(int k=1;k<=no;k++)
		{
			if(t[k]!=y[k])
			{
				if(t[k]>y[k])error_range=t[k]-y[k];
				else error_range=y[k]-t[k];
				if(error_range>0.01)
				{
					test[k] = 0;
					ROS_INFO(" Not Equal");
					//cout<<"Not equal"<<endl;
				}
				else
				{
					ROS_INFO("Equal");
					//cout<<"Equal"<<endl;
					test[k] = 1;
				}
			}
			else{
				//cout<<"Equal"<<endl;
				ROS_INFO("Equal");
				test[k] = 1;
			}
		}
		//test ends



		for(int k=1; k<=no;k++)
		{
			del_k[k]=(t[k]-y[k])*(sigmoidDerivative(y_in[k]));
		}


		for(int k=1;k<=no;k++)
		{
			for(int j=1;j<=ni;j++)
			{
				del_w[j][k]=alpha*del_k[k]*z[j];
			}
		}

		for(int k=1;k<=no;k++)
		{
			del_w[0][k]=alpha*del_k[k];
		}



		summation =0;
		for(int j=1;j<=ni;j++)
		{
			for(int k=1;k<=no;k++)
			{
				summation +=del_k[k]*w[j][k];
			}
			del_inj[j]=summation;
		}


		for(int j=1;j<=ni;j++)
		{
			del_j[j]=del_inj[j]*sigmoidDerivative(z_in[j]);
		}


		for(int i=1;i<=ni;i++)
		{
			for(int j=1;j<=ni;j++)
			{
				del_vij[i][j]=alpha*del_j[j]*x[i];
			}
		}

		for(int j=1;j<=ni;j++)
		{
			del_vij[0][j]=alpha*del_j[j];
		}

		for(int j=0;j<=ni;j++)
		{
			for(int k=1;k<=no;k++)
			{
				w[j][k]=w[j][k]+del_w[j][k];
		//		cout <<"w_weight "<<j<<k<<","<<w[j][k]<<endl;
			}
		}

		for(int i=0;i<=ni;i++)
		{
			for(int j=1;j<=ni;j++)
			{
				v[i][j]=v[i][j]+del_vij[i][j];
		//		cout <<"v_weight "<<i<<j<<","<<v[i][j]<<endl;

			}
		}
		//cout<<"Test:"<<test<<endl;
		for(int i=1;i<=no;i++)
		{
			check *=test[i];
		}
		ROS_INFO("Check : %d", check);
	}while(check == 0);
	//cout<<"Trained";
	//cout<<w[0][0];

	//std_msgs::String status = "success";

	std_msgs::String result;
	result.data = (check)?"success":"Failure";
	ROS_INFO("%s",result.data.c_str());




	weight_object.weight_upper_ = new float[(ni+1)*(no)];

	for(int j=0;j<=ni;j++)
	{
		for(int k=1;k<=no;k++)
		{
			weight_object.weight_upper_[j*ni+k]=w[j][k];
		//		cout <<"v_weight "<<i<<j<<","<<v[i][j]<<endl;

		}
	}

	// for(int j=0;j<=ni;j++)
	// {
	// 	for(int k=1;k<=no;k++)
	// 	{
	// 		cout <<"v_weight "<<j<<k<<","<<weight_upper_[j*ni+k]<<endl;

	// 	}
	// }

    weight_object.nic=ni;
    weight_object.noc=no;

	weight_object.weight_lower_ = new float[(ni+1)*(ni)];

	for(int i=0;i<=ni;i++)
	{
		for(int j=1;j<=ni;j++)
		{
			weight_object.weight_lower_[i*ni+j]=v[i][j];
		//		cout <<"v_weight "<<i<<j<<","<<v[i][j]<<endl;

		}
	}


}

// after the network is trained. The weights are used for working with actual input sets
void inputCallback(const adero_neural_network::ActualInput::ConstPtr& in)
{
	Weight w_obj;
	int ni=w_obj.nic,no=w_obj.noc;
	int test[no+1];
	float x[ni+1];
	float t[no+1];
	float v[ni+1][ni];
	float w[ni+1][no];
	float z_in[ni+1];
	float z[ni+1];
	float summation = 0;
	float y_in[no];
	float y[no];

	x[0]=1;
	z[0]=1;
	for(int i=1;i<=ni;i++)
	{
		x[i]=in->data[i-1];
	}

	for(int j=0;j<=ni;j++)
	{
		for(int k=1;k<=no;k++)
		{
			w[j][k]= w_obj.weight_upper_[j*ni+k];
		//		cout <<"w_weight "<<i<<j<<","<<v[i][j]<<endl;
		}
	}

	for(int i=0;i<=ni;i++)
	{
		for(int j=1;j<=ni;j++)
		{
			v[i][j]=w_obj.weight_lower_[i*ni+j];
		//		cout <<"v_weight "<<i<<j<<","<<v[i][j]<<endl;

		}
	}


		//after training
	z[0]=1;
	z_in[0]=0;//check here

	for(int j=1;j<=ni;j++)
	{
		for(int i=1;i<=ni;i++)
		{
			summation += (x[i]*v[i][j]);//summation to zero
		}
		z_in[j]=v[0][j]+summation;
		z[j]=sigmoid(z_in[j]);
	}
		//upper layer starts


	summation = 0;
	for(int k=1;k<=no;k++)
	{
		for(int j=1;j<=ni;j++)
		{
			summation += (z[j]*w[j][k]);
		}
		y_in[k]=w[0][k]+summation;
		y[k]=sigmoid(y_in[k]);
	}
	//upper layer ends
	//test starts
	for(int p=1;p<=no;p++)
	{
	    ROS_INFO("Actual output %d = %f",p,y[p]);
	}
}


int main(int argc, char **argv)
{
	ros::init(argc, argv,"demo_topic_publisher");
	ros::NodeHandle node_obj;
	ros::Publisher number_publisher = node_obj.advertise<std_msgs::String>("/numbers",10);

	ros::NodeHandle subscrive_input;
	ros::Subscriber sub3 = subscrive_input.subscribe("training_pattern", 100, arrayCallback);

	ros::NodeHandle subscrive_actual_input;
	ros::Subscriber sub_actual = subscrive_actual_input.subscribe("actual_input", 100, inputCallback);

	ros::Rate loop_rate(10);
	
    ros::spin();
    return 0;
}
