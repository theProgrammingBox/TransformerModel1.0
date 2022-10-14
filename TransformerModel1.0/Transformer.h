#pragma once
#include "Header.h"

class Transformer {
public:
	uint64_t TOKEN_DIMENTIONS;			// number of dimentions in each input token
	uint64_t QUERY_DIMENTIONS;			// number of dimentions in each query, key, and value
	uint64_t LINEAR_FEED_DIMENTIONS;	// number of dimentions in the linear feed forward layer
	uint64_t NUM_HEADS;					// number of heads in multi-head attention
	uint64_t numRuns;
	float* queryWeights;				// [TOKEN_DIMENTIONS * QUERY_DIMENTIONS * NUM_HEADS]
	float* keyWeights;					// [TOKEN_DIMENTIONS * QUERY_DIMENTIONS * NUM_HEADS]
	float* valueWeights;				// [TOKEN_DIMENTIONS * QUERY_DIMENTIONS * NUM_HEADS]
	float* concatWeights;					// [QUERY_DIMENTIONS * NUM_HEADS * TOKEN_DIMENTIONS]
	float* ffWeights1;					// [TOKEN_DIMENTIONS * LINEAR_FEED_DIMENTIONS]
	float* ffWeights2;					// [LINEAR_FEED_DIMENTIONS * TOKEN_DIMENTIONS]
	vector<float*> querysList;			// List of current and past querys, [QUERY_DIMENTIONS * NUM_HEADS], number of runs elements
	vector<float*> keysList;			// List of current and past keys, [QUERY_DIMENTIONS * NUM_HEADS], number of runs elements
	vector<float*> valuesList;			// List of current and past values, [QUERY_DIMENTIONS * NUM_HEADS], number of runs elements
	vector<float*> ScoresList;			// List of scores for each key given the current query, [NUM_HEADS], number of runs elements
	vector<float*> output1;				// List of the sum of all values multiplied by their respective scores, [QUERY_DIMENTIONS * NUM_HEADS], number of runs elements
	vector<float*> output2;				// List of the output of the concatination linear feed forward layer, [TOKEN_DIMENTIONS], number of runs elements
	vector<float*> output3;				// List of the output of the first linear feed forward layer, [LINEAR_FEED_DIMENTIONS], number of runs elements
	vector<float*> output4;				// List of the output of the final linear feed forward layer, [TOKEN_DIMENTIONS], number of runs elements

	Transformer(uint64_t TOKEN_DIMENTIONS, uint64_t QUERY_DIMENTIONS, uint64_t LINEAR_FEED_DIMENTIONS, uint64_t NUM_HEADS);
	~Transformer();
};