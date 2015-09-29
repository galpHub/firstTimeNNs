// Implementation of feedforward neural networks with backpropagation.
#include <iostream>


#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>
#include <vector>


// Defining neuron types for simplicity.
#define RECT 0
#define TANH 1

// A function to generate ('real-valued') random numbers between 0 and 1
double random_gen(){
	return static_cast <double> (rand()) / static_cast <double>(RAND_MAX);
}

// Ease of referencing a function pointer for neurons to get their activation functions from the same place.
typedef double (*activation_function)(double*,double*,int);

// The neuron class. Implements a neuron by specifying the indices of the neurons in a previous layer from
// which to take its inputs, what the weights corresponding to those inputs (with an additional bias term)
// are, the type of neuron and the corresponding activation function in case type is specified. The input
// layer doesn't have an activation function.
class neuron{
	public:
		int indegree = 0; // the number of neurons from the lower layer connected to this neuron.
		int* neuron_idcs = NULL; // Indices of neurons in the previous (lower) layer this neuron requires access to.
		double* weights = NULL; // Weights for the computation of the neuron signal - including the bias as the last entry.
		activation_function activation_f; // Activation function for the neuron.
		std::string type; // Short string descriptor of the activation function.
		double value = 0; // Starting neuron signal.
		void fire(double*); // Updates the variable 'value' by evaluating the activation function on an input array.

		// Constructors
		neuron() = default;
		neuron(int* input_neurons, int num_inputs, std::string act_type);

	private:
		void def_activation(void); // 

};


// Initializes the neuron by indicating which neurons it's connected to, how many incomming connections it has
// and what type of neuron is used. It also initializes the weights to uniform random numbers between 0 and 1.
neuron::neuron(int* input_neurons, int num_inputs, std::string act_type){
	indegree = num_inputs;
	type = act_type;
	std::vector<double> weights(num_inputs,0.0);
	std::vector<int> neuron_idcs(num_inputs, 0);
	neuron::def_activation();
	for (int i = 0; i < num_inputs + 1; i++){
		weights[i] = random_gen();
	}
}


// Implementation of activation functions for neurons.
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

	return 0.5*tanh(sum) + 1;
}

// A get-er function for activation functions 
activation_function activation_type(int type){
	switch (type){
	case RECT:
		return (activation_function)n_rect;
	case TANH:
		return (activation_function)n_tanh;
	default:
		throw - 1;
	}
}

// Updates the value of the "values" variable in the neuron by evaluating the activation function.
void neuron::fire(double* input){
	if (!activation_f){
		value = activation_f(weights, input, indegree);
	}
	else{
		value = 0;
	}
	
}

// Initializes the activation function of a neuron based on the value of "type", either rectilinear (RECT) or hyberbolic tangent
// (TANH) currently implemented.
void neuron::def_activation(){
	if (type == "rect"){
		activation_f = activation_type(RECT);
	}
	else if (type == "tanh"){
		activation_f = activation_type(TANH);
	}
}




class layer{
};

class n_network{
};










int main(int argc, char** argv){

	double test_weights[3] = {1.0,2.0,0.0};
	double test_input1[2] = { -2.0/3.0, 2.0/3.0 };
	double test_input2[2] = {1.0,0.0};
	printf("This is a test of the activation functions' output: \n");
	printf("Direct output of the rect functions in two sample cases: \n");
	printf("%f, %f\n\n", n_rect(test_weights, test_input1, 2), n_rect(test_weights, test_input2, 2));
	getchar();
	activation_function function1 = activation_type(RECT);
	activation_function function2 = activation_type(RECT);
	printf("This is the outputs of the rect activation function called via activation_type: %f, %f\n",
		function1(test_weights, test_input1, 2), function2(test_weights, test_input2, 2));
	getchar();

	return 0;
}