#include "Transformer.h"

int main() {
	Seed();
	
	/*uint64_t TOKEN_DIMENTIONS = 16;
	uint64_t QUERY_DIMENTIONS = 8;
	uint64_t LINEAR_FEED_DIMENTIONS = 32;
	uint64_t NUM_HEADS = 4;
	Transformer transformer(TOKEN_DIMENTIONS, QUERY_DIMENTIONS, LINEAR_FEED_DIMENTIONS, NUM_HEADS);
	transformer.exportParameters("TransformerParameters.txt");*/
	/*float* input = new float[TOKEN_DIMENTIONS];
	float* output = new float[TOKEN_DIMENTIONS];
	for (int i = 0; i < TOKEN_DIMENTIONS; i++) {
		input[i] = normalRand();
	}
	transformer.run(input, output);
	for (int i = 0; i < TOKEN_DIMENTIONS; i++) {
		cout << output[i] << endl;
	}
	delete[] input;
	delete[] output;*/
	
	float* randList = new float[10];
	for (int i = 0; i < 10; i++) {
		randList[i] = normalRand();
		cout << randList[i] << endl;
	}
	ofstream file("TransformerParameters.txt", ios::binary);
	file.write((char*)randList, sizeof(float) * 10);
	file.close();
	cout << endl;
	
	float* randList2 = new float[10];
	ifstream file2("TransformerParameters.txt", ios::binary);
	file2.read((char*)randList2, sizeof(float) * 10);
	for (int i = 0; i < 10; i++) {
		cout << randList2[i] << endl;
	}
	
	return 0;
}