// Implementation of simple feedforward neural networks with backpropagation.
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


// The neuron class. Implements a neuron by specifying the indices of the neurons in a previous layer from
// which to take its inputs, what the weights corresponding to those inputs (with an additional bias term)
// are, the type of neuron and the corresponding activation function in case type is specified. The input
// layer doesn't have an activation function.
class neuron{
	public:
		double value = 0; // Starting neuron signal.
		std::vector<double> weights; // Weights for the computation of the neuron signal - including the bias as the last entry.

		// Constructors
		neuron() = default;
		neuron(std::vector<int> input_neurons, std::string act_type);

		// Member functions
		double n_rect( std::vector<double> input);
		double n_tanh(std::vector<double> input); 

		int get_indegree() { return indegree; }
		std::string get_type() { return type; }

		std::vector<double> gradWrtWeights(std::vector<double>);

		// Updates the variable 'value' by evaluating the activation function on an input array.
		void fire(std::vector<double>); 
	private:
		int indegree = 0; // the number of neurons from the lower layer connected to this neuron.
		std::vector<int> neuron_idcs; // Indices of neurons in the previous (lower) layer this neuron requires access to.
		std::string type; // Short string descriptor of the activation function.

};

// Computes the gradient of the neuron's output (i.e. value) with respect to the weights.
std::vector<double> neuron::gradWrtWeights(std::vector<double> inputs){

	std::vector<double> grad(indegree + 1,0);
	double chainRuleFactor;
	if (type == "rect"){
		if (value != 0){
			for (int i = 0; i < indegree; i++){
				grad[i] = inputs[neuron_idcs[i]];
			}
			grad[indegree] = 1.0;
		}
		return grad;
	}
	else if (type == "tanh"){
		chainRuleFactor = 1 - pow(value,2);
		for (int i = 0; i < indegree; i++){
			grad[i] = chainRuleFactor*inputs[neuron_idcs[i]];
		}
		grad[indegree] = chainRuleFactor;
		return grad;
	}
	else{
		std::runtime_error(std::string("Unrecognized Neuron type!") );
	}
	return grad;
}

// Initializes the neuron by indicating which neurons it's connected to, how many incomming connections it has
// and what type of neuron is used. It also initializes the weights to uniform random numbers between 0 and 1.
neuron::neuron(std::vector<int> input_neurons, std::string act_type){
	int num_inputs = input_neurons.size();
	indegree = num_inputs;
	type = act_type;

	weights.reserve(num_inputs + 1);
	neuron_idcs.reserve(num_inputs);

	for (int i = 0; i < num_inputs; i++){
		neuron_idcs.push_back(input_neurons[i]);
		weights.push_back(random_gen());
	}
	weights.push_back(random_gen());
}


// Implementation of activation functions for neurons. Input is assumed to be a vector
// corresponding to the collection of neuron firing signals from the previous layer.
double neuron::n_rect(std::vector<double> input){
	double sum = 0;
	for (int i = 0; i < indegree; i++){
		sum += weights[i] * input[neuron_idcs[i]];
	}
	sum += weights[indegree];

	if (sum<0){
		return 0;
	}
	else{
		return sum;
	}
}

double neuron::n_tanh(std::vector<double> input){
	double sum = 0;
	for (int i = 0; i < indegree; i++){
		sum += weights[i] * input[neuron_idcs[i]];
	}
	sum += weights[indegree];

	return 0.5*(tanh(sum) + 1);
}


// Updates the value of the "values" variable in the neuron by evaluating the activation function.
// Neurons in the input layer have their values unmodified by the fire function.
void neuron::fire(std::vector<double> input){
	if (type == "rect"){
		value = n_rect(input);
	}
	else if (type == "tanh"){
		value = n_tanh(input);
	}

}




// Neuron layers are just a mess of disconnected neurons until the connections
// are specified.
class layer{
	public:
		layer(int neuronas = NULL, layer* prev = NULL) {
			numOfNeurons = neuronas;
			prevLayer = prev;
			if (prev != NULL){
				noIncommingConnections = false;
			}
			neuron neuron_t;
			neuronVector.reserve(numOfNeurons);
			neuronSignals.reserve(numOfNeurons);
			for (int i = 0; i < numOfNeurons; i++){
				neuronVector.push_back(neuron_t);
				neuronSignals.push_back(0.0); // neuronVector[i].value == 0.0
			}
		}

		// Indicates where to connect the neurons of current layer
		// and initializes the neuron types. Does not check if 'prev == NULL'.
		// Overloaded to include a single type of neuron per layer of varying
		// types within a layer.
		void layerWiring(std::vector<std::vector<int>>, std::vector<std::string>);
		void layer::layerWiring(std::vector<std::vector<int>>, std::string);

		// Initializes the input layer by setting neuronSignals = inputs
		void layerLoadInput(std::vector<double> inputs)
		{
			if (noIncommingConnections){
				neuronSignals = inputs;
			}
			else{
				throw std::runtime_error(std::string(
					"Incomming connections to this layer! Cannot initialize as an input layer."));
			}
		}

		// Fires neurons in the current layer by drawing on the neuronSignals of
		// the previous layer and updating the neuronSignals of the current layer.
		void layerFire()
		{
			if (!noIncommingConnections && prevLayer != NULL){
				std::vector<double> inputs = prevLayer->neuronSignals;
				for (int i = 0; i < numOfNeurons; i++){
					neuronVector[i].fire(inputs);
				}
				neuronSignals = getLayerVals();
			}
			else{
				throw std::runtime_error(std::string(
					"Unitialized connections or no previous layer! Cannot fire neurons.")
					+std::string(" Check if 'this' is the input layer"));
			}
		}

		// Get the current signals of the neurons in this layer.
		std::vector<double> getLayerVals(){
			if (numOfNeurons != 0){
				std::vector < double > neuronVals(numOfNeurons);
				for (int i = 0; i < numOfNeurons; i++){
					neuronVals[i]= neuronVector[i].value;
				}
			}
			else{
				throw std::runtime_error(std::string(
					"This layer has no neurons!"));
			}
		}

		// Command for use in implementing dropout.
		void supressRandomNeurons(){
			double coinflip = 0.0;
			for (int i = 0; i < numOfNeurons; i++){
				coinflip = random_gen();
				if (coinflip < 0.5){
					neuronSignals[i] = 0;
				}
			}
		}

		int getNeuronNum() { return numOfNeurons; };

	private:

		int numOfNeurons;
		std::vector<neuron> neuronVector;
		std::vector<double> neuronSignals;
		layer* prevLayer = NULL;
		bool noIncommingConnections = true;

};

void layer::layerWiring(std::vector<std::vector<int>> neuronConnections,
	std::vector<std::string> neuronTypes){
	if (neuronConnections.size()!= numOfNeurons || neuronTypes.size() != numOfNeurons ||
		neuronConnections.size() !=neuronTypes.size()){
		std::runtime_error(std::string(
			"Mismatch between number of neurons and attempted number of Neuron initializations!"));
	}
	else{
		
		for (int i = 0; i < (int)neuronConnections.size(); i++){
			neuron neura(neuronConnections[i], neuronTypes[i]);
			neuronVector[i] = neura;
		}
		noIncommingConnections = false;
	}
}

void layer::layerWiring(std::vector<std::vector<int>> neuronConnections, std::string neuronType){
	if ((int)neuronConnections.size() != numOfNeurons){
		std::runtime_error(std::string("Mismatch between number of neurons and")
			+std::string(" attempted number of Neuron initializations!"));
	}
	else{

		for (int i = 0; i < (int)neuronConnections.size(); i++){
			neuron neura(neuronConnections[i], neuronType);
			neuronVector[i] = neura;
		}
		noIncommingConnections = false;
	}
}





class n_network{
};










int main(int argc, char** argv){
	
	return 0;
}