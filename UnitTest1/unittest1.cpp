#include "stdafx.h"
#include "CppUnitTest.h"
#include "../nn.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

neuron simpleNeuron(std::string name)
{
	std::vector<int> input_neurons(2,0);
	input_neurons[1] = 1;
	int num_inputs = 2;

	neuron jimmy(input_neurons, num_inputs, name);

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

			neuron jimmy( input_neurons, num_inputs, "rect");
			
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
			double test_value;
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
			double test_value;
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

	};
}