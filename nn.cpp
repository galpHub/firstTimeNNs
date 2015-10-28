// Implementation of simple feedforward neural networks with backpropagation.
#include <iostream>
#include <fstream>
#include <sstream>

#define _USE_MATH_DEFINES
#include <math.h>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <utility>
#include <string>

#define CANNOT_LINK_LAYERS std::string("Cannot link these layers! Wrong number of neurons or wiring specs!")
#define UNRECOGNIZED_NEURON_TYPE std::string("Unrecognized Neuron type!")
#define CANNOT_LOAD_NONINPUT_LAYER std::string("Incomming connections to this layer! Cannot initialize as an input layer.")
#define CANNOT_FIRE_INPUT_LAYER std::string("Unitialized connections or no previous layer! Cannot fire neurons. Check if 'this' is the input layer")
#define NO_NEURONS_EXIST std::string("This layer has no neurons!")
#define MISMATCHED_NEURONS_INITIALIZATION std::string("Mismatch between number of neurons and attempted number of Neuron initializations!")
#define NULL_LAYER_PTR std::string("Cannot be compatible with null layer!")
#define MISSING_CONNECTIONS std::string("NN either has a single layer or Layers may be incompatibly connected!")
#define NO_BACKPROP_W_1LAYER std::string("Cannot backpropagate with 1 layer!")

// Defining neuron types for simplicity.
#define RECT 0
#define TANH 1

std::vector<double> operator+(std::vector<double> a, std::vector<double> b){
	std::vector<double> c;
	int aSize, bSize;
	aSize = a.size();
	bSize = b.size();
	if (aSize == bSize){
		c.reserve(aSize);
		for (int i = 0; i < aSize; i++){
			c.push_back(a[i] + b[i]);
		}
		return c;
	}
	else{
		throw std::runtime_error(std::string("Vectors of different sizes cannot be added"));
	}
	
}
std::vector<double> operator-(std::vector<double> a, std::vector<double> b){
	std::vector<double> c;
	int aSize, bSize;
	aSize = a.size();
	bSize = b.size();
	if (aSize == bSize){
		c.reserve(aSize);
		for (int i = 0; i < aSize; i++){
			c.push_back(a[i] - b[i]);
		}
		return c;
	}
	else{
		throw std::runtime_error(std::string("Vectors of different sizes cannot be subtracted"));
	}

}
std::vector<double> operator*(double a, std::vector<double> b){
	std::vector<double> c;
	int bSize = b.size();
	c.reserve(bSize);
	for (int i = 0; i < bSize; i++){
		c.push_back(a * b[i]);
	}
	return c;
}
double sum(std::vector<double> a){
	double out = 0;
	for (int i = 0; i < (int)a.size(); i++){
		out += a[i];
	}
	return out;
}


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

		std::vector<int> getNeuronIdcs(){ return neuron_idcs; }
		int get_indegree() { return indegree; }
		std::string get_type() { return type; }

		std::vector<double> gradWrtWeights(std::vector<double>);
		std::vector<double> gradWrtInputs();

		// Updates the variable 'value' by evaluating the activation function on an input array.
		void fire(std::vector<double>);
		void suppress() { isSuppressed = true; }
		void restore(){ isSuppressed = false; }
		bool isSuppressedNeuron(){ return isSuppressed; }

	private:
		int indegree = 0; // the number of neurons from the lower layer connected to this neuron.
		std::vector<int> neuron_idcs; // Indices of neurons in the previous (lower) layer this neuron requires access to.
		std::string type; // Short string descriptor of the activation function.
		bool isSuppressed = false;
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
		throw std::runtime_error( UNRECOGNIZED_NEURON_TYPE );
	}
	return grad;
}
std::vector<double> neuron::gradWrtInputs(){

	std::vector<double> grad(indegree, 0);
	double chainRuleFactor;
	if (type == "rect"){
		if (value != 0){
			for (int i = 0; i < indegree; i++){
				grad[i] = weights[i];
			}
		}
		return grad;
	}
	else if (type == "tanh"){
		chainRuleFactor = 1 - pow(value, 2);
		for (int i = 0; i < indegree; i++){
			grad[i] = chainRuleFactor*weights[i];
		}
		return grad;
	}
	else{
		throw std::runtime_error(UNRECOGNIZED_NEURON_TYPE);
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

	if (!isSuppressed){

		if (type == "rect"){
			value = n_rect(input);
		}
		else if (type == "tanh"){
			value = n_tanh(input);
		}

	}

}




// Neuron layers are just a mess of disconnected neurons until the connections
// are specified.
class layer{
	public:
		layer(int neuronas = NULL, layer* prev = NULL) {
			numOfNeurons = neuronas;
			prevLayer = prev;
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
		void layerWiring(std::vector<std::vector<int>>, std::string);

		// Checks if the layer below from which the current layer is going to take inputs
		// is compatible in terms of memory accesses, i.e. neurons.
		bool checkCompatibleLayer(layer *layerBelow) {

			int neuronsBelow;

			if (layerBelow == NULL){
				throw std::invalid_argument(NULL_LAYER_PTR);
			}
			neuronsBelow = layerBelow->getNeuronNum();

			if (noIncommingConnections && neuronsBelow>0){
				return true;
			}
			else if (neuronsBelow ==0){
				return false;
			}

			int maxIdcs;
			std::vector<int> wirings;
			for (int i = 0; i < getNeuronNum(); i++){
				wirings = neuronVector[i].getNeuronIdcs();
				maxIdcs = *std::max_element(wirings.begin(), wirings.end());
				if (neuronsBelow - 1 < maxIdcs){
					return false;
				}
			}
			return true;
		}
		bool checkCompatibleLayer() {

			layer *layerBelow = prevLayer;
			int neuronsBelow;

			if (layerBelow == NULL){
				throw std::invalid_argument(NULL_LAYER_PTR);
			}
			neuronsBelow = layerBelow->getNeuronNum();

			if (noIncommingConnections && neuronsBelow > 0){
				return true;
			}
			else if (neuronsBelow == 0){
				return false;
			}


			int maxIdcs;
			std::vector<int> wirings;
			for (int i = 0; i < getNeuronNum(); i++){
				wirings = neuronVector[i].getNeuronIdcs();
				maxIdcs = *std::max_element(wirings.begin(), wirings.end());
				if (neuronsBelow - 1 < maxIdcs){
					return false;
				}
			}
			return true;
		}
		bool isConnected(){
			return numOfNeurons!= 0 && checkCompatibleLayer() && noIncommingConnections==false && prevLayer != NULL;
		}
		bool isPrevLayerNull(){ return prevLayer == NULL; }

		// Initializes the input layer by setting neuronSignals = inputs
		void layerLoadInput(std::vector<double> inputs)
		{
			if (noIncommingConnections){
				neuronSignals = inputs;
			}
			else{
				throw std::runtime_error(CANNOT_LOAD_NONINPUT_LAYER);
			}
		}

		// Fires neurons in the current layer by drawing on the neuronSignals of
		// the previous layer and updating the neuronSignals of the current layer.
		void layerFire()
		{
			if (!noIncommingConnections && prevLayer != NULL){
				std::vector<double> inputs = prevLayer->getNeuronSignals();
				for (int i = 0; i < numOfNeurons; i++){
					neuronVector[i].fire(inputs);
				}
				updateLayerVals();
			}
			else{
				throw std::runtime_error(CANNOT_FIRE_INPUT_LAYER);
			}
		}

		// Get the current signals of the neurons in this layer.
		void updateLayerVals(){
			if (numOfNeurons != 0){
				for (int i = 0; i < numOfNeurons; i++){
					neuronSignals[i]= neuronVector[i].value;
				}
			}
			else{
				throw std::runtime_error(NO_NEURONS_EXIST);
			}
		}

		// Command for use in implementing dropout.
		void suppressRandomNeurons(double probOfSuppress = 0.5){
			double coinflip = 0.0;
			for (int i = 0; i < numOfNeurons; i++){
				coinflip = random_gen();
				if (coinflip < probOfSuppress){
					neuronVector[i].suppress();
				}
				else{
					neuronVector[i].restore();
				}
			}
		}
		void restoreAllNeurons(){
			for (int i = 0; i < numOfNeurons; i++){
				neuronVector[i].restore();
			}
		}



		std::vector<bool> getNonSuppressedNeurons(){
			std::vector<bool> notSuppressed(numOfNeurons,1);
			for (int i = 0; i < numOfNeurons; i++){
				notSuppressed[i] = !neuronVector[i].isSuppressedNeuron();
			}
			return notSuppressed;
		}

		int getNeuronNum() { return numOfNeurons; };
		std::vector<double> getNeuronSignals(){ return neuronSignals; };
		std::vector<neuron> getNeuronVector(){ return neuronVector; };

		void updateWeights(std::vector<double> dE_dval, double learnRate){
			bool canUpdate;
			for (int i = 0; i < numOfNeurons; i++){
				canUpdate = !neuronVector[i].isSuppressedNeuron();
				if (canUpdate){
					std::vector<double> delta = (-learnRate*dE_dval[i])*neuronVector[i].gradWrtWeights(prevLayer->getNeuronSignals());
					neuronVector[i].weights = neuronVector[i].weights + delta;
				}
			}
		}
		layer* getPrevLayer() { return prevLayer; }

	private:
		int numOfNeurons;
		std::vector<neuron> neuronVector;
		std::vector<double> neuronSignals;

		// Indicates from which layer the neurons in the current take their input.
		layer* prevLayer = NULL;

		// Determines whether incomming neuron-neuron connections have been specified
		bool noIncommingConnections = true; 

};

void layer::layerWiring(std::vector<std::vector<int>> neuronConnections,
	std::vector<std::string> neuronTypes){
	if (neuronConnections.size()!= numOfNeurons || neuronTypes.size() != numOfNeurons ||
		neuronConnections.size() !=neuronTypes.size()){
		throw std::runtime_error(MISMATCHED_NEURONS_INITIALIZATION);
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
		throw std::runtime_error(MISMATCHED_NEURONS_INITIALIZATION);
	}
	else{

		for (int i = 0; i < (int)neuronConnections.size(); i++){
			neuron neura(neuronConnections[i], neuronType);
			neuronVector[i] = neura;
		}
		noIncommingConnections = false;
	}
}





std::vector<double> leastSquares(std::vector<double> expected, std::vector<double> observed){
	int expSize = expected.size();
	int obsSize = observed.size();
	if (expSize != obsSize){
		throw std::runtime_error("Mismatched dimensions in Least Squares gradient.");
	}

	return expected - observed;
}





class n_network{
	public:
		n_network() = default;

		void fireNetworkToTrain(std::vector<double> inputs){
			int numOfLayers = networkLayers.size();

			if (canFireNetwork()){
				networkLayers[0].layerLoadInput(inputs);
				for (int i = 1; i < numOfLayers-1; i++){
					networkLayers[i].suppressRandomNeurons();
					networkLayers[i].layerFire();
				}
				networkLayers[numOfLayers - 1].layerFire();

				

			}
			else{
				throw std::runtime_error(MISSING_CONNECTIONS);
			}
		}

		std::vector<double> fireNetwork(std::vector<double> inputs){
			int numOfLayers = networkLayers.size();

			if (canFireNetwork()){
				networkLayers[0].layerLoadInput(inputs);
				for (int i = 1; i < numOfLayers - 1; i++){
					networkLayers[i].restoreAllNeurons();
					networkLayers[i].layerFire();
				}
				networkLayers[numOfLayers - 1].layerFire();

				return networkLayers[numOfLayers - 1].getNeuronSignals();
			}
			else{
				throw std::runtime_error(MISSING_CONNECTIONS);
			}
		}
		

		std::vector < layer > networkLayers;

		void backPropagation(std::vector<double> inputs, std::vector<double>outputs,
								double learnRate);

		void setLayers(std::vector<int> neuronsPerLayer, 
			std::vector<std::vector<std::vector<int>>> interLayerConnections, 
			std::vector<std::string> neuronTypes){

			int numOfLayers = neuronsPerLayer.size();
			networkLayers = std::vector<layer>(numOfLayers);

			networkLayers[0] = layer(neuronsPerLayer[0]);
			for (int i = 1; i < numOfLayers; i++){
				networkLayers[i] = layer(neuronsPerLayer[i], &networkLayers[i-1]);
				networkLayers[i].layerWiring(interLayerConnections[i - 1], neuronTypes[i - 1]);
			}

		}


	private:
		bool canFireNetwork(){
			int numOfLayers = networkLayers.size();
			if (numOfLayers < 2){
				return false;
			}
			for (int i = 1; i < numOfLayers; i++){
				if (!networkLayers[i].isConnected()){
					return false;
				}
			}
			return true;
		}
		std::vector<double>(*dE_doutput)(std::vector<double>,
			std::vector<double>) = leastSquares;

		void trainNetwork(std::vector<std::vector<double>>, std::vector<std::vector<double>>);

		std::vector<double> errorProp(layer nthLayer,std::vector<double> dE, layer layerBelow){

			
			int numOfNeurons = nthLayer.getNeuronNum();
			int numOfNeuronsBelow = layerBelow.getNeuronNum();

			int indegree;
			std::vector < double > dEnextLayer(numOfNeuronsBelow);
			std::vector<neuron> nthLayerNeurons = nthLayer.getNeuronVector();
			std::vector<double> gradInInputs;
			bool neuronIsActive;

			std::vector<int> neuronIdcs;
			double neuronValue;

			if (numOfNeurons != 0 && numOfNeuronsBelow != 0){
				for (int i = 0; i < numOfNeurons; i++){
					neuronIsActive = !nthLayerNeurons[i].isSuppressedNeuron();

					if (neuronIsActive){
						indegree = nthLayerNeurons[i].get_indegree();
						neuronIdcs = nthLayerNeurons[i].getNeuronIdcs();
						neuronValue = nthLayerNeurons[i].value;
						gradInInputs = nthLayerNeurons[i].gradWrtInputs();

						for (int j = 0; j < indegree; j++){
							dEnextLayer[neuronIdcs[j]] += neuronValue*gradInInputs[j];
						}

					}
	
				}
				return dEnextLayer;
			}
			else{
				throw std::runtime_error(NO_NEURONS_EXIST);
			}
		}

};


void n_network::backPropagation(std::vector<double> inputs,std::vector<double>outputs, double learnRate){
	int numOfLayers = networkLayers.size();
	if (numOfLayers < 2){
		throw std::runtime_error(NO_BACKPROP_W_1LAYER);
	}

	std::vector<std::vector<double>> dE_dvalues(numOfLayers);
	fireNetworkToTrain(inputs);


	dE_dvalues[numOfLayers - 1] = dE_doutput(outputs,
									networkLayers[numOfLayers-1].getNeuronSignals());
	
	// Backpropagation allows us to update dE_dvalues in the n-1-th layer
	// by using the values of dE_dvalues in the n-th layer. Of course, the input
	// layer isnt relevant because we can't change the input.
	for (int j = numOfLayers - 1; j > 1; j--){
		dE_dvalues[j - 1] = errorProp(networkLayers[j], dE_dvalues[j], networkLayers[j - 1]);
		networkLayers[j].updateWeights(dE_dvalues[j], learnRate);
	}

	networkLayers[1].updateWeights(dE_dvalues[1], learnRate);





}

void n_network::trainNetwork(std::vector<std::vector<double>>vectorOfInputs,
	std::vector<std::vector<double>> vectorOfOutputs){


	// Simple learning rate formula.
	int numOfInputs = vectorOfInputs.size();
	int numOfOutputs = vectorOfOutputs.size();
	if (numOfInputs == numOfOutputs){
		for (int i = 0; i < numOfInputs;i++){
			backPropagation(vectorOfInputs[i], vectorOfOutputs[i],0.5/(i+1));
		}
	}

}

// Ancilliary functions for reading and writting from or to a file.
std::vector<int> stoiVector(std::vector<std::string> myString){
	int n = myString.size();
	std::vector<int> intVector(n);
	for (int i = 0; i < n; i++)
	{
		intVector.push_back(std::stoi( myString[i] ));
	}
	return intVector;
}

std::vector<std::string> split(const std::string myString, char delim)
{
	std::stringstream S(myString);
	std::string token;
	std::vector<std::string> elem;
	while (std::getline(S, token, delim)){
		elem.push_back(token);
	}
	return elem;
}




n_network nnFromFile(char* filename)
{
	// FILE FORMAT:
	// A line with contents: "-L 4" indicates the start of a new layer with 4 neurons.
	// The four lines immediately following it each contain a sequence of space separated
	// integers. The i-th line contains the connection information of the i-th neuron in the
	// current layer. The integers are merely the indices of the neurons in the previous
	// layer to which it is connected. The input layer needs no such lines because it has
	// no previous layer to connect to. Thus the first two lines of the file should read 
	// something like:
	//
	// -L 100
	// -L 2
	// 10 77 52
	// 33 42 16
	// -L 1
	// #EOF
	//
	// The first line indicates that there are 100 neurons, with no inputs (it's the input 
	// layer), and the next layer only contains two neurons, each with three incomming connections.
	// The first of these neurons is connected to the 10th, 77th and 52nd inputs. Similarly for
	// the second neuron.
	// 

	std::ifstream inputFile;
	n_network brain;
	try
	{
		inputFile.open(filename);
	}
	catch (int e)
	{
		printf("An execption occurred. Exception Number: %i", e);
		throw e;
	}
	if (inputFile.is_open()){
		std::string line;
		std::vector < int > parameters;
		int currentLayer = -1;
		int numOfNeurons = 0;
		std::vector<int> neuronsPerLayer;
		std::vector < std::vector<std::vector<int>> > interLayerConnections;
		std::vector<std::string> neuronTypes;

 

		while (std::getline(inputFile, line))
		{
			std::vector<std::string> splitStr = split(line,(char)" ");
			if (splitStr[0] == "-L")
			{ // Start of newLayer
				currentLayer += 1;
				numOfNeurons = std::stoi(splitStr[1]);
				neuronsPerLayer.push_back(numOfNeurons);
				if (currentLayer > 0){
					interLayerConnections.push_back(std::vector<std::vector<int>>());
					neuronTypes.push_back(splitStr[1]);
				}
				
			}
			else if (currentLayer > 0)
			{
				interLayerConnections[currentLayer].push_back(stoiVector(splitStr));
			}


		}
		
		brain.setLayers(neuronsPerLayer,interLayerConnections,neuronTypes);
		return brain;


	}
	else
	{
		std::runtime_error(std::string ("COULD NOT OPEN FILE"));
	}

}




int main(int argc, char** argv){

	//
	// Usage of this code should proceed as:
	// nn.exe -i InputFile.txt -tr TrainSamples.txt -te TestSamples.txt -w Weights.txt
	//
	
	n_network brain = nnFromFile(argv[1]);

	getchar();
	return 0;
}