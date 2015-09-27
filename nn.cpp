// Implementation of feedforward neural networks with backpropagation.
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#define RECT 0
#define TANH 1


typedef double (*activation_function)(double*,double*,int); // typedef double (*)() activation_function;


class neuron{
	int indegree; // the number of neurons from the lower layer connected to this neuron.
	int* neuron_idcs; // Indices of neurons in the previous (lower) layer this neuron requires access to.
	double* weights; // Weights for the computation of the neuron signal - including the bias as the last entry.
	activation_function activation; // Activation function for the neuron.
	char type[4]; // Short string descriptor of the activation function.
	double value = 0; // Starting neuron signal.
	private:
		void fire (double*);
		void def_activation (void);

};

void neuron::fire(double* input){
	if (!activation){
		value = activation(weights, input, indegree);
	}
	else{
		value = 0;
	}
	
}

void neuron::def_activation(){
	if (type == "rect"){
		activation = activation_type(RECT);
	}
	else if (type == "tanh"){
		activation = activation_type(TANH);
	}
}


class layer{
};

class n_network{
};


double n_rect(double* weights, double* input, int length){
	double sum = 0;
	for (int i = 0; i < length; i++){
		sum += weights[i] * input[i];
	}
	sum += weights[length];

	if (sum >1){
		return 1.0;
	}
	else if (sum<0){
		return 0;
	}
	else{
		return sum;
	}
}

double n_tanh(double* weights, double* input, int length){
	double sum = 0;
	for (int i = 0; i < length; i++){
		sum += weights[i] * input[i];
	}
	sum += weights[length];

	return 0.5*tanh(sum)+1;
}


activation_function activation_type(int type){
	switch (type){
		case RECT:
			return (activation_function) n_rect;
		case TANH:
			return (activation_function) n_tanh;
		default:
			throw -1;
	}
}







int main(int argc, char** argv){
	double test_weights[3] = {1.0,2.0,0.0};
	double test_input1[2] = { -2.0/3.0, 2.0/3.0 };
	double test_input2[2] = {1.0,0.0};
	printf("This is a test: ");
	getchar();
	printf("%f, %f\n", n_rect(test_weights, test_input1, 2), n_rect(test_weights, test_input2, 2));
	getchar();
	activation_function thefunction = activation_type(RECT);
	printf("This is the doom: %f\n", thefunction(test_weights, test_input1, 2) );
	getchar();

	return 0;
}