// Implementation of feedforward neural networks with backpropagation.
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>
#include <vector>

// Defining neuron types for simplicity.
#define RECT 0
#define TANH 1

// A function to generate (hopefully 'real-valued') random numbers between 0 and 1
double random_gen(){
	return static_cast <double> (rand()) / static_cast <double>(RAND_MAX);
}

// Ease of referencing a function pointer for neurons to get their activation functions from the same place.
typedef double (*activation_function)(double*,double*,int);


class neuron{
	public:
		int indegree = 0; // the number of neurons from the lower layer connected to this neuron.
		int* neuron_idcs = NULL; // Indices of neurons in the previous (lower) layer this neuron requires access to.
		double* weights = NULL; // Weights for the computation of the neuron signal - including the bias as the last entry.
		activation_function activation; // Activation function for the neuron.
		std::string type; // Short string descriptor of the activation function.
		double value = 0; // Starting neuron signal.

	public:
		neuron() = default;
		neuron(int* input_neurons, int num_inputs, std::string act_type);
		void fire(double*); // Updates the variable 'value' by evaluating the activation function on an input array.
	
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