#include "Transformer.h"

int main() {
	Transformer transformer;
	transformer.TOKEN_DIMENTIONS = 3;
	transformer.QUERY_DIMENTIONS = 2;
	transformer.LINEAR_FEED_DIMENTIONS = 4;
	transformer.NUM_HEADS = 2;
	transformer.queryWeights = new float[transformer.TOKEN_DIMENTIONS * transformer.QUERY_DIMENTIONS * transformer.NUM_HEADS];
	transformer.keyWeights = new float[transformer.TOKEN_DIMENTIONS * transformer.QUERY_DIMENTIONS * transformer.NUM_HEADS];
	transformer.valueWeights = new float[transformer.TOKEN_DIMENTIONS * transformer.QUERY_DIMENTIONS * transformer.NUM_HEADS];
	transformer.weights1 = new float[transformer.QUERY_DIMENTIONS * transformer.NUM_HEADS * transformer.TOKEN_DIMENTIONS];
	transformer.weights2 = new float[transformer.TOKEN_DIMENTIONS * transformer.LINEAR_FEED_DIMENTIONS];
	transformer.weights3 = new float[transformer.LINEAR_FEED_DIMENTIONS * transformer.TOKEN_DIMENTIONS];
	transformer.querysList.push_back(new float[transformer.QUERY_DIMENTIONS * transformer.NUM_HEADS]);
	transformer.keysList.push_back(new float[transformer.QUERY_DIMENTIONS * transformer.NUM_HEADS]);
	transformer.valuesList.push_back(new float[transformer.QUERY_DIMENTIONS * transformer.NUM_HEADS]);
	transformer.ScoresList.push_back(new float[transformer.NUM_HEADS]);
	transformer.output1.push_back(new float[transformer.QUERY_DIMENTIONS * transformer.NUM_HEADS]);
	transformer.output2.push_back(new float[transformer.TOKEN_DIMENTIONS]);
	transformer.output3.push_back(new float[transformer.LINEAR_FEED_DIMENTIONS]);
	transformer.output4.push_back(new float[transformer.TOKEN_DIMENTIONS]);
	
	return 0;
}