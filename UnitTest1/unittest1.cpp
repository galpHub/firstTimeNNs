#include "stdafx.h"
#include "CppUnitTest.h"
#include "../nn.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

neuron simpleNeuron(std::string name)
{
	int input_neurons[2] = { 0, 1 };
	int num_inputs = 2;

	neuron jimmy(input_neurons, num_inputs, name);

	for (int i = 0; i < num_inputs + 1; i++){
		jimmy.weights[i] = 1.0 / 4.0;
	}
	return jimmy;
}


namespace UnitTest1
{		
	TEST_CLASS(UnitTest1)
	{
	public:

		TEST_METHOD(Neuron_Default_SUCCESS)
		{
			neuron jimmy;
		}

		TEST_METHOD(Neuron_Initialize_SUCCESS)
		{
			int input_neurons[2] = {0,1};
			int num_inputs = 2;
			int numOf = 0;
			std::vector<double> weights(3, 0.0);

			neuron jimmy( input_neurons, num_inputs, "rect");
			
			if (jimmy.weights == weights){
				Assert::Fail();
			}
			
		}

		TEST_METHOD(Neuron_rectFire_SUCCESS)
		{
			double test_input[2] = { 0, 0 };
			double test_output[2] = { 0, 0 };
			double expected_output[2] = { 0, 0 };

			neuron jimmy = simpleNeuron("rect");
			for (int j = 0; j < jimmy.indegree; j++){
				test_input[j] = 1.0;
				jimmy.fire(test_input);
				test_output[j] = jimmy.value;
				expected_output[j] = 1.0*j / 4.0 + 2 * (1.0 / 4.0);

				if (abs(expected_output[j] - test_output[j]) > 1e-16){
					Assert::Fail();
				}
			}
			
		}

		TEST_METHOD(Neuron_tanhFire_SUCCESS)
		{
			double test_input[2] = { 0, 0 };
			double test_output[2] = { 0, 0 };
			double expected_output[2] = { 0, 0 };

			neuron jimmy = simpleNeuron("tanh");
			for (int j = 0; j < jimmy.indegree; j++){
				test_input[j] = 1.0;

				jimmy.fire(test_input);

				test_output[j] = jimmy.value;

				expected_output[j] = 0.5*(tanh(1.0*j / 4.0 + 2 * (1.0 / 4.0)) + 1);

				if (abs(expected_output[j] - test_output[j]) > 1e-16){
					Assert::Fail();
				}
			}
		}

		TEST_METHOD(Neuron_nullFire_SUCCESS)
		{
			neuron jimmy;
			Assert::AreEqual(jimmy.value, 0.0);
		}

		TEST_METHOD(NeuronVector_init_access_SUCCESS)
		{
			double test_input[2];
			double test_value;
			neuron jimmy = simpleNeuron("rect");
			std::vector<neuron> james(4, jimmy);


			for (int i = 0; i < 4; i++){
				test_input[0] = test_input[1] = random_gen();
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