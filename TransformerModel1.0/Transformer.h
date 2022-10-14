#pragma once
#include "Header.h"

/*
thanks to 
https://jalammar.github.io/illustrated-gpt2/
https://jalammar.github.io/illustrated-transformer/
build for single input at a time with possibly infinite length
*/

class Transformer {
public:
	Transformer(uint64_t TOKEN_DIMENTIONS, uint64_t QUERY_DIMENTIONS, uint64_t LINEAR_FEED_DIMENTIONS, uint64_t NUM_HEADS);
	~Transformer();
	void run(float* input, float* output);			// input is a pointer to an array of TOKEN_DIMENTIONS floats
	void exportParameters(string fileName);
	void importParameters(string fileName);

private:
	uint64_t TOKEN_DIMENTIONS;			// number of dimentions in each input token
	uint64_t QUERY_DIMENTIONS;			// number of dimentions in each query, key, and value
	uint64_t LINEAR_FEED_DIMENTIONS;	// number of dimentions in the linear feed forward layer
	uint64_t NUM_HEADS;					// number of heads in multi-head attention
	uint64_t numRuns;
	float* queryWeights;				// [TOKEN_DIMENTIONS * QUERY_DIMENTIONS * NUM_HEADS]
	float* keyWeights;					// [TOKEN_DIMENTIONS * QUERY_DIMENTIONS * NUM_HEADS]
	float* valueWeights;				// [TOKEN_DIMENTIONS * QUERY_DIMENTIONS * NUM_HEADS]
	float* concatWeights;				// [QUERY_DIMENTIONS * NUM_HEADS * TOKEN_DIMENTIONS]
	float* hiddenWeights;				// [TOKEN_DIMENTIONS * LINEAR_FEED_DIMENTIONS]
	float* outputWeights;				// [LINEAR_FEED_DIMENTIONS * TOKEN_DIMENTIONS]
	vector<float*> querysList;			// List of current and past querys, [QUERY_DIMENTIONS * NUM_HEADS], number of runs elements
	vector<float*> keysList;			// List of current and past keys, [QUERY_DIMENTIONS * NUM_HEADS], number of runs elements
	vector<float*> valuesList;			// List of current and past values, [QUERY_DIMENTIONS * NUM_HEADS], number of runs elements
	vector<vector<float*>> scoresList;	// List of (list of scores for each key given the current query, [NUM_HEADS], number of runs elements), number of runs elements
	vector<float*> headValues;			// List of the sum of all values multiplied by their respective scores, [QUERY_DIMENTIONS * NUM_HEADS], number of runs elements
	vector<float*> concatenatedOutput;	// List of the output of the concatination linear feed forward layer, [TOKEN_DIMENTIONS], number of runs elements
	vector<float*> hiddenLayer;			// List of the output of the first linear feed forward layer, [LINEAR_FEED_DIMENTIONS], number of runs elements
	vector<float*> outputLayer;			// List of the output of the final linear feed forward layer, [TOKEN_DIMENTIONS], number of runs elements

	void RandomizeWeights();
	void AllocateMemory();
	void PositionalEncoding(float* input);
	void GenerateQueryKeyValue(float* input);
	void MyQueryDotAllKeys();
	void Softmax();
	void GetHeadValues();
	void ConcatenateHeads(float* input);
	void LayerNorm();
	void LinearHiddenFeedForward();
	void LinearOutputFeedForward(float* input, float* output);
};