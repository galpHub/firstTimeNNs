// Implementation of feedforward neural networks with backpropagation.
#include <iostream>
#define RECT 0
#define TANH 1



typedef double (*activation_function)(double*,double*,int); // typedef double (*)() activation_function;


class neuron{
	int indegree;
	int* neuron_idcs; // Indices of neurons in the previous (lower) layer this neuron requires access to.
	double* weights; // Weights for the computation of the neuron signal.
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


class layer{
};

class n_network{
};


double rect(double* weights, double* input, int length){
	double sum = 0;
	for (int i = 0; i < length; i++){
		sum += weights[i] * input[i];
	}

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


activation_function activation_type(int type){
	switch (type){
		case RECT:
			return (activation_function) rect;
		//case TANH:
		//	return (activation_function) tanh;
		default:
			throw -1;
	}
}







int main(int argc, char** argv){
	double test_weights[2] = {1.0,2.0};
	double test_input1[2] = { -2.0/3.0, 2.0/3.0 };
	double test_input2[2] = {1.0,0.0};
	printf("This is a test: ");
	getchar();
	printf("%f, %f\n", rect(test_weights, test_input1, 2), rect(test_weights, test_input2, 2));
	getchar();
	activation_function thefunction = activation_type(0);
	printf("This is the doom: %f\n", thefunction(test_weights, test_input1, 2) );
	getchar();

	return 0;
}