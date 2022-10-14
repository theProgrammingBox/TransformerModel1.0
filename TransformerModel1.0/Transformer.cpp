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
		for (int j = 0; j < numRuns; j++) {
			delete[] scoresList[i][j];
		}
		delete[] headValues[i];
		delete[] concatenatedOutput[i];
		delete[] output3[i];
		delete[] output4[i];
	}
}

void Transformer::run(float* input) {
	AllocateMemory();
	GenerateQueryKeyValue(input);
	MyQueryDotAllKeys();
	Softmax();
	GetHeadValues();
	ConcatenateHeads();
}

void Transformer::AllocateMemory() {
	querysList.push_back(new float[QUERY_DIMENTIONS * NUM_HEADS]);
	keysList.push_back(new float[QUERY_DIMENTIONS * NUM_HEADS]);
	valuesList.push_back(new float[QUERY_DIMENTIONS * NUM_HEADS]);
	scoresList.push_back(vector<float*>());
	for (int i = 0; i <= numRuns; i++) {
		scoresList[numRuns].push_back(new float[NUM_HEADS]);
	}
	headValues.push_back(new float[QUERY_DIMENTIONS * NUM_HEADS]);
	concatenatedOutput.push_back(new float[TOKEN_DIMENTIONS]);
	output3.push_back(new float[LINEAR_FEED_DIMENTIONS]);
	output4.push_back(new float[TOKEN_DIMENTIONS]);
}

void Transformer::GenerateQueryKeyValue(float* input) {
	for (int i = 0; i < NUM_HEADS; i++) {
		for (int j = 0; j < QUERY_DIMENTIONS; j++) {
			querysList[numRuns][i * QUERY_DIMENTIONS + j] = 0;
			keysList[numRuns][i * QUERY_DIMENTIONS + j] = 0;
			valuesList[numRuns][i * QUERY_DIMENTIONS + j] = 0;
			for (int k = 0; k < TOKEN_DIMENTIONS; k++) {
				querysList[numRuns][i * QUERY_DIMENTIONS + j] += input[k] * queryWeights[i * QUERY_DIMENTIONS * TOKEN_DIMENTIONS + j * TOKEN_DIMENTIONS + k];
				keysList[numRuns][i * QUERY_DIMENTIONS + j] += input[k] * keyWeights[i * QUERY_DIMENTIONS * TOKEN_DIMENTIONS + j * TOKEN_DIMENTIONS + k];
				valuesList[numRuns][i * QUERY_DIMENTIONS + j] += input[k] * valueWeights[i * QUERY_DIMENTIONS * TOKEN_DIMENTIONS + j * TOKEN_DIMENTIONS + k];
			}
		}
	}
}

void Transformer::MyQueryDotAllKeys() {
	for (int i = 0; i < numRuns; i++) {
		for (int j = 0; j < NUM_HEADS; j++) {
			for (int k = 0; k < QUERY_DIMENTIONS; k++) {
				scoresList[numRuns][i][j] += querysList[numRuns][j * QUERY_DIMENTIONS + k] * keysList[i][j * QUERY_DIMENTIONS + k];
			}
			scoresList[numRuns][i][j] /= sqrt(QUERY_DIMENTIONS);
		}
	}
}

void Transformer::Softmax() {
	for (int i = 0; i < NUM_HEADS; i++) {
		float sum = 0;
		for (int j = 0; j < numRuns; j++) {
			sum += exp(scoresList[numRuns][j][i]);
		}
		for (int j = 0; j < numRuns; j++) {
			scoresList[numRuns][j][i] = exp(scoresList[numRuns][j][i]) / sum;
		}
	}
}

void Transformer::GetHeadValues() {
	for (int i = 0; i < NUM_HEADS; i++) {
		for (int j = 0; j < QUERY_DIMENTIONS; j++) {
			headValues[numRuns][i * QUERY_DIMENTIONS + j] = 0;
			for (int k = 0; k < numRuns; k++) {
				headValues[numRuns][i * QUERY_DIMENTIONS + j] += scoresList[numRuns][k][i] * valuesList[k][i * QUERY_DIMENTIONS + j];
			}
		}
	}
}

void Transformer::ConcatenateHeads() {
	for (int i = 0; i < TOKEN_DIMENTIONS; i++) {
		concatenatedOutput[numRuns][i] = 0;
		for (int j = 0; j < QUERY_DIMENTIONS * NUM_HEADS; j++) {
			concatenatedOutput[numRuns][i] += headValues[numRuns][j] * concatWeights[j * TOKEN_DIMENTIONS + i];
		}
	}
}