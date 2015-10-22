#include "stdafx.h"
#include "CppUnitTest.h"
#include "../nn.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

neuron simpleNeuron(std::string name)
{
	std::vector<int> input_neurons(2,0);
	input_neurons[1] = 1;
	int num_inputs = 2;

	neuron jimmy(input_neurons, name);

	for (int i = 0; i < num_inputs + 1; i++){
		jimmy.weights[i] = 1.0 / 4.0;
	}
	return jimmy;
}


namespace UnitTest1
{		
	TEST_CLASS(NeuronTests)
	{
	public:

		TEST_METHOD(Neuron_DefaultConstructor_SUCCESS)
		{
			neuron jimmy;
		}

		TEST_METHOD(Neuron_NonDefaultConstructor_SUCCESS)
		{
			std::vector<int> input_neurons(2, 0);
			input_neurons[1] = 1;
			int num_inputs = 2;
			int numOf = 0;
			std::vector<double> weights(3, 0.0);

			neuron jimmy( input_neurons, "rect");
			
			if (jimmy.weights == weights){
				Assert::Fail();
			}
			
		}

		TEST_METHOD(Neuron_rectFireCorrectOutput_SUCCESS)
		{
			std::vector<double> test_input(2,0);
			std::vector<double> test_output(2, 0);
			std::vector<double> expected_output(2, 0);



			neuron jimmy = simpleNeuron("rect");
			for (int j = 0; j < jimmy.get_indegree(); j++){
				test_input[j] = 1.0;
				jimmy.fire(test_input);
				test_output[j] = jimmy.value;
				expected_output[j] = 1.0*j / 4.0 + 2 * (1.0 / 4.0);

				if (abs(expected_output[j] - test_output[j]) > 1e-16){
					Assert::Fail();
				}
			}
			
		}

		TEST_METHOD(Neuron_tanhFireCorrectOutput_SUCCESS)
		{
			std::vector<double> test_input(2, 0);
			std::vector<double> test_output(2, 0);
			std::vector<double> expected_output(2, 0);

			neuron jimmy = simpleNeuron("tanh");
			for (int j = 0; j < jimmy.get_indegree(); j++){
				test_input[j] = 1.0;

				jimmy.fire(test_input);

				test_output[j] = jimmy.value;

				expected_output[j] = 0.5*(tanh(1.0*j / 4.0 + 2 * (1.0 / 4.0)) + 1);

				if (abs(expected_output[j] - test_output[j]) > 1e-16){
					Assert::Fail();
				}
			}
		}

		TEST_METHOD(Neuron_defaultConstructorNeuronNotFiring_SUCCESS)
		{
			neuron jimmy;
			std::vector<double> inputs(50, 1);
			jimmy.fire(inputs);
			Assert::AreEqual(jimmy.value, 0.0);
		}

		TEST_METHOD(Neuron_independenceOfNeuronsInVectorOfNeurons_SUCCESS)
		{
			std::vector<double> test_input(2, 0);
			neuron jimmy = simpleNeuron("rect");
			std::vector<neuron> james(4, jimmy);

			// Some arbitrary values between 0 and 1 to test the firing
			// of the neuron.
			double input_values[4] = { .6,.1,0.7,.3 };


			for (int i = 0; i < 4; i++){
				test_input[0] = test_input[1] = input_values[i];
				james[i].fire(test_input);
			}

			for (int i = 0; i < 4; i++){
				for (int j = 0; j < 4; j++){
					if (james[i].value == james[j].value && i!=j){
						Assert::Fail();
						break;
					}
				}

			}

		}


		TEST_METHOD(Neuron_initVectorOfNeuronsWithReferenceNeuron_SUCCESS)
		{
			std::vector<double> test_input(2, 0);
			neuron jimmy = simpleNeuron("rect");
			std::vector<neuron> james(4, jimmy);

			// Some arbitrary values between 0 and 1 to test the firing
			// of the neuron.
			double input_values[4] = { .6, .1, 0.7, .3 };

			for (int i = 0; i < 4; i++){
				test_input[0] = test_input[1] = input_values[i];
				james[i].fire(test_input);
				jimmy.fire(test_input);
				if (james[i].value != jimmy.value){
					Assert::Fail();
					break;
				}
			}

		}
		
		TEST_METHOD(Neuron_computeRectGradientWrtWeights_SUCCESS){
			neuron jimmy = simpleNeuron("rect");
			std::vector < double > inputToGrad(jimmy.get_indegree(), 0);

			inputToGrad[0] = 37; inputToGrad[1] = 23;
			jimmy.fire(inputToGrad);

			std::vector<double> observed_grad = jimmy.gradWrtWeights(inputToGrad);
			std::vector<double> expected_grad(jimmy.get_indegree()+1);
			expected_grad[0] = inputToGrad[0]; 
			expected_grad[1] = inputToGrad[1];
			expected_grad[2] = 1.0;
			
			for (int j = 0; j < jimmy.get_indegree()+1; j++){
				if (abs(observed_grad[j]- expected_grad[j]) > 1e-16){
					Assert::Fail();
				}
			}
		}

		TEST_METHOD(Neuron_computeTanhGradientWrtWeights_SUCCESS){
			neuron jimmy = simpleNeuron("tanh");
			std::vector < double > inputToGrad(jimmy.get_indegree(), 0);

			inputToGrad[0] = 1.0/4; inputToGrad[1] = 2.0/4;
			jimmy.fire(inputToGrad);
			double chainRuleFactor = 1 - pow(jimmy.value, 2);

			std::vector<double> observed_grad = jimmy.gradWrtWeights(inputToGrad);
			std::vector<double> expected_grad(jimmy.get_indegree() + 1);
			expected_grad[0] = chainRuleFactor*inputToGrad[0];
			expected_grad[1] = chainRuleFactor*inputToGrad[1];
			expected_grad[2] = chainRuleFactor;

			for (int j = 0; j < jimmy.get_indegree() + 1; j++){
				if (abs(observed_grad[j] - expected_grad[j]) > 1e-16){
					Assert::Fail();
				}
			}
		}

		TEST_METHOD(Neuron_computeRectGradientWrtInputs_SUCCESS){
			neuron jimmy = simpleNeuron("rect");
			std::vector < double > inputToGrad(jimmy.get_indegree(), 0);

			inputToGrad[0] = 37; inputToGrad[1] = 23;
			jimmy.fire(inputToGrad);
			std::vector<double> expected_grad = (1.0)*jimmy.weights;
			

			std::vector<double> observed_grad = jimmy.gradWrtInputs();

			for (int j = 0; j < jimmy.get_indegree() ; j++){
				if (abs(observed_grad[j] - expected_grad[j]) > 1e-16){
					Assert::Fail();
				}
			}
		}

		TEST_METHOD(Neuron_computeTanhGradientWrtInputs_SUCCESS){
			neuron jimmy = simpleNeuron("tanh");
			std::vector < double > inputToGrad(jimmy.get_indegree(), 0);

			inputToGrad[0] = 37; inputToGrad[1] = 23;
			jimmy.fire(inputToGrad);

			std::vector<double> expected_grad = (1-pow(jimmy.value,2))*jimmy.weights;

			std::vector<double> observed_grad = jimmy.gradWrtInputs();

			for (int j = 0; j < jimmy.get_indegree(); j++){
				if (abs(observed_grad[j] - expected_grad[j]) > 1e-16){
					Assert::Fail();
				}
			}
		}
	};

	TEST_CLASS(LayerTests){
		public:
			TEST_METHOD(Layer_fireInputLayer_SUCCESS){

				auto func = [] {
					layer testLayer(10);
					testLayer.layerFire(); 
				};

				Assert::ExpectException<std::runtime_error,void>(func);

			}

			TEST_METHOD(Layer_fireRectLayer_SUCCESS){
				layer InputLayer(9);
				layer *link = &InputLayer;
				layer OutputLayer(3, link);
				std::vector<double> inputs(9);
				std::vector<neuron> testNeurons(3);
				std::vector<double> observed(3);
				
				std::vector < std::vector<int>> wires(3);
				for (int i = 0; i < 3; i++){
					wires[i].reserve(3);
					for (int j = 0; j < 3; j++){
						wires[i].push_back(3 * j + i);
					}
				}

				OutputLayer.layerWiring(wires, "rect");

				if (OutputLayer.checkCompatibleLayer()){
					// These inputs are all 0.
					for (int k = 0; k < 9; k++){
						inputs = std::vector<double>(9, 0);
						inputs[k] = 1;
						InputLayer.layerLoadInput(inputs);
						// The OutputLayer should update all their values to the bias
						// value of the weight vectors of each neuron.
						testNeurons = OutputLayer.getNeuronVector();
						OutputLayer.layerFire();
						observed = OutputLayer.getNeuronSignals();
						for (int i = 0; i < 3; i++){
							testNeurons[i].fire(link->getNeuronSignals());
							if (abs(observed[i] - testNeurons[i].value) >1e-16){
								Assert::Fail();
							}
						}
					}

				}
				else{
					Assert::Fail();
				}
			}
			TEST_METHOD(Layer_fireRectLayerCorrectOutput_SUCCESS){
				layer InputLayer(9);
				layer OutputLayer(3, &InputLayer);
				std::vector<neuron> testNeurons(3);
				std::vector<double> observed(3);
				int idx,idx2;

				std::vector < std::vector<int>> wires(3);
				for (int i = 0; i < 3; i++){
					wires[i].reserve(3);
					for (int j = 0; j < 3; j++){
						wires[i].push_back(j + 3*i);
					}
				}

				OutputLayer.layerWiring(wires, "rect");

				if (OutputLayer.checkCompatibleLayer()){
					// These inputs are all 0.
					std::vector<double> testInput(9, 0);
					std::vector<double> allWeights(12, 0);
					std::vector<neuron> testNeurons(3);
					std::vector<double> expectedOutput(3);
					testNeurons = OutputLayer.getNeuronVector();

					for (int i = 0; i < 3; i++){
						for (int j = 0; j < 4; j++){
							allWeights[j + 4 * i] = testNeurons[i].weights[j];
						}
					}

					for (int j = 0; j < 9; j++){
						testInput = std::vector<double>(9, 0);
						testInput[j] = 1;
						InputLayer.layerLoadInput(testInput);
						OutputLayer.layerFire();
						expectedOutput = std::vector<double>(3, 0);
						int j2 = j % 3;
						int j1 = j / 3;
						expectedOutput[j1] = allWeights[j2+4*j1];
		

						for (int i = 0; i < 3; i++){
							// The operations on expectedOutput in the loop just
							// add the bias terms to each output neuron.
							expectedOutput[i] += allWeights[3 + 4 * i];
							if (abs(OutputLayer.getNeuronSignals()[i] - expectedOutput[i]) >1e-16){
								Assert::Fail();
							}
						}
					}

				

				}
				else{
					Assert::Fail();
				}
			}

			TEST_METHOD(Layer_fireTanhLayer_SUCCESS){
				layer InputLayer(9);
				layer *link = &InputLayer;
				layer OutputLayer(3, link);
				std::vector<double> inputs(9);
				std::vector<neuron> testNeurons(3);
				std::vector<double> observed(3);

				std::vector < std::vector<int>> wires(3);
				for (int i = 0; i < 3; i++){
					wires[i].reserve(3);
					for (int j = 0; j < 3; j++){
						wires[i].push_back(3 * j + i);
					}
				}

				OutputLayer.layerWiring(wires, "tanh");

				if (OutputLayer.checkCompatibleLayer()){
					// These inputs are all 0.
					for (int k = 0; k < 9; k++){
						inputs = std::vector<double>(9, 0);
						inputs[k] = 1;
						InputLayer.layerLoadInput(inputs);
						// The OutputLayer should update all their values to the bias
						// value of the weight vectors of each neuron.
						testNeurons = OutputLayer.getNeuronVector();
						OutputLayer.layerFire();
						observed = OutputLayer.getNeuronSignals();
						for (int i = 0; i < 3; i++){
							testNeurons[i].fire(InputLayer.getNeuronSignals());
							if (abs(observed[i] - testNeurons[i].value) >1e-16){
								Assert::Fail();
							}
						}
					}

				}
				else{
					Assert::Fail();
				}
			}

			TEST_METHOD(Layer_loadInputLayer_SUCCESS){
				layer testLayer(10);
				std::vector<double> inputs(10, 1);

				auto func = [testLayer](std::vector<double> input)mutable {
					testLayer.layerLoadInput(input);
					return testLayer.getNeuronSignals();
				};
				Assert::IsTrue(func(inputs) == inputs);
			}

			TEST_METHOD(Layer_invalidConnectionOfLayers_1_SUCCESS){
				layer InputLayer(10);
				layer *link = &InputLayer;
				layer OutputLayer(3, link);

				// Incorporate incompatible wirings accesing the 11-th neuron of
				// the input layer
				std::vector < std::vector<int>> wires(3);
				for (int i = 0; i < 3; i++){
					wires[i].reserve(3);
					for (int j = 0; j < 3; j++){
						wires[i].push_back(3*j+i+2);
					}
				}
				OutputLayer.layerWiring(wires, "rect");
				
				Assert::IsFalse(OutputLayer.checkCompatibleLayer());

			}

			TEST_METHOD(Layer_invalidConnectionOfLayers_2_SUCCESS){
				layer InputLayer;
				layer *link = &InputLayer;
				layer OutputLayer(3, link);

				// No neurons in the input layer to connect to.
				Assert::IsFalse(OutputLayer.checkCompatibleLayer());

			}

			TEST_METHOD(Layer_invalidConnectionOfLayers_3_SUCCESS){
				layer InputLayer(10);
				layer OutputLayer(3);
				layer *link = &InputLayer;

				// Check that the output layer could in principle be appropiately
				// connected to the input neuron layer.
				Assert::IsTrue(OutputLayer.checkCompatibleLayer(link));
			}

			TEST_METHOD(Layer_OutputIsImproperlyConnected_SUCCESS){
				layer InputLayer(10);
				layer *link = &InputLayer;
				layer OutputLayer(3, link);

				// Incorporate incompatible wirings accesing the 11-th neuron of
				// the input layer
				std::vector < std::vector<int>> wires(3);
				for (int i = 0; i < 3; i++){
					wires[i].reserve(3);
					for (int j = 0; j < 3; j++){
						wires[i].push_back(3 * j + i + 2);
					}
				}
				OutputLayer.layerWiring(wires, "rect");

				// The output layer "isn't connected" because it's trying to access an
				// 11-th neuron in the input layer.
				Assert::IsFalse(OutputLayer.isConnected());
			}

			TEST_METHOD(Layer_InputIsProperlyConnected_EXCEPTION){
				layer InputLayer(10);
				layer *link = &InputLayer;
				layer OutputLayer(3, link);

				// Incorporate incompatible wirings accesing the 11-th neuron of
				// the input layer
				std::vector < std::vector<int>> wires(3);
				for (int i = 0; i < 3; i++){
					wires[i].reserve(3);
					for (int j = 0; j < 3; j++){
						wires[i].push_back(3 * j + i + 2);
					}
				}
				OutputLayer.layerWiring(wires, "rect");

				auto func = [InputLayer]() mutable {InputLayer.isConnected(); };

				Assert::ExpectException<std::invalid_argument>(func);
				
			}

			TEST_METHOD(Layer_OutputIsProperlyConnected_SUCCESS){
				layer InputLayer(10);
				layer *link = &InputLayer;
				layer OutputLayer(3, link);

				// Incorporate incompatible wirings accesing the 11-th neuron of
				// the input layer
				std::vector < std::vector<int>> wires(3);
				for (int i = 0; i < 3; i++){
					wires[i].reserve(3);
					for (int j = 0; j < 3; j++){
						wires[i].push_back(3 * j + i);
					}
				}
				OutputLayer.layerWiring(wires, "rect");

				// The output layer "isn't connected" because it's trying to access an
				// 11-th neuron in the input layer.
				Assert::IsTrue(OutputLayer.isConnected());
			}
	};

	TEST_CLASS(NeuralNetworkTests){
		TEST_METHOD(NeuralNetwork_CreateTwoLayers_SUCCESS){

			n_network brain0;
			brain0.networkLayers = std::vector<layer>(2);

			std::vector<double> testInput(10);
			std::vector<double> testOutput(1);

			std::vector<int> neuronIdcsLayer0(10);
			double outputVal = 0;

			for (int i = 0; i < 10; i++){
				neuronIdcsLayer0[i] = i;
			}
			std::vector<std::vector<int>> connections(1);
			connections[0] = neuronIdcsLayer0;

			brain0.networkLayers[0] = layer(10);
			brain0.networkLayers[1] = layer(1, &brain0.networkLayers[0]);
			brain0.networkLayers[1].layerWiring(connections, "rect");

			neuron outputNeuron = brain0.networkLayers[1].getNeuronVector()[0];


			testOutput[0] = outputNeuron.weights[10];
			for (int i = 0; i < 10; i++){
				testInput[i] = i;
				testOutput[0] += i*outputNeuron.weights[i];
				brain0.fireNetwork(testInput);
				outputVal = brain0.networkLayers[1].getNeuronSignals()[0];
	

				if ( abs(testOutput[0]  - outputVal)>1e-14 ){
					Assert::Fail();
				}
			}



		}

		TEST_METHOD(NeuralNetwork_CreateTwoLayers2_SUCCESS){

			n_network brain0;
			brain0.networkLayers = std::vector<layer>(2);

			std::vector<double> testInput(10);
			std::vector<double> testOutput(1);

			std::vector<int> neuronIdcsLayer0(10);
			double outputVal = 0;

			for (int i = 0; i < 10; i++){
				neuronIdcsLayer0[i] = i;
			}
			std::vector<std::vector<int>> connections(1);
			connections[0] = neuronIdcsLayer0;

			std::vector<int> neuronsPerLayer(2);
			std::vector<std::vector<std::vector<int>>> nnConnections(1,connections);
			std::vector<std::string> neuronTypes(1,"rect");

			neuronsPerLayer[0]=10;
			neuronsPerLayer[1]=1;
			brain0.setLayers(neuronsPerLayer, nnConnections, neuronTypes);

			neuron outputNeuron = brain0.networkLayers[1].getNeuronVector()[0];


			testOutput[0] = outputNeuron.weights[10];
			for (int i = 0; i < 10; i++){
				testInput[i] = i;
				testOutput[0] += i*outputNeuron.weights[i];
				brain0.fireNetwork(testInput);
				outputVal = brain0.networkLayers[1].getNeuronSignals()[0];


				if (abs(testOutput[0] - outputVal)>1e-14){
					Assert::Fail();
				}
			}


		}

	};

}