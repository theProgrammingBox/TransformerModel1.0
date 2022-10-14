#include "Transformer.h"

Transformer::Transformer(uint64_t TOKEN_DIMENTIONS, uint64_t QUERY_DIMENTIONS, uint64_t LINEAR_FEED_DIMENTIONS, uint64_t NUM_HEADS) {
	this->TOKEN_DIMENTIONS = TOKEN_DIMENTIONS;
	this->QUERY_DIMENTIONS = QUERY_DIMENTIONS;
	this->LINEAR_FEED_DIMENTIONS = LINEAR_FEED_DIMENTIONS;
	this->NUM_HEADS = NUM_HEADS;
	this->numRuns = 0;
	queryWeights = new float[TOKEN_DIMENTIONS * QUERY_DIMENTIONS * NUM_HEADS];
	keyWeights = new float[TOKEN_DIMENTIONS * QUERY_DIMENTIONS * NUM_HEADS];
	valueWeights = new float[TOKEN_DIMENTIONS * QUERY_DIMENTIONS * NUM_HEADS];
	concatWeights = new float[QUERY_DIMENTIONS * NUM_HEADS * TOKEN_DIMENTIONS];
	ffWeights1 = new float[TOKEN_DIMENTIONS * LINEAR_FEED_DIMENTIONS];
	ffWeights2 = new float[LINEAR_FEED_DIMENTIONS * TOKEN_DIMENTIONS];
}

Transformer::~Transformer() {
	delete[] queryWeights;
	delete[] keyWeights;
	delete[] valueWeights;
	delete[] concatWeights;
	delete[] ffWeights1;
	delete[] ffWeights2;
	for (int i = 0; i < numRuns; i++) {
		delete[] querysList[i];
		delete[] keysList[i];
		delete[] valuesList[i];
		delete[] ScoresList[i];
		delete[] output1[i];
		delete[] output2[i];
		delete[] output3[i];
		delete[] output4[i];
	}
}