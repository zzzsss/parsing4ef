#ifndef _MODEL_MODEL
#define _MODEL_MODEL

#include <vector>
#include <string>
using std::vector;
using std::string;
using std::pair;

// basic data type
using REAL = float;

// The interface to the models
// currently assuming simple feedforward model (simple input)
// -- NOPE -- using Input = pair<vector<int>*, vector<int>*>;		// !! the second index is to the sentence, not embeddings
struct Input{
	vector<int>* first{nullptr};    // first is the index of words
	vector<int>* second{nullptr};   // second is the index of other features, like distance and labels
	int which{0};                   // which is the dependency category, like left or right
	Input(vector<int>* f, vector<int>* s, int w): first(f), second(s), which(w){}
};
using Output = vector<REAL>*;
const int NONEXIST_INDEX = -100000;		// for specifying index for the token in a sentence

// the abstract class
// -- simplify this model, no caches
class ModelZ{
public:
	virtual ~ModelZ(){}
	virtual void new_sentence(vector<vector<int>*>) = 0;	// word/pos's index to embedding, create Graph
	virtual void end_sentence() = 0;
	virtual vector<Output> forward(const vector<Input>& x) = 0;
	virtual void backward(const vector<Input>& in, const vector<int>&index, const vector<REAL>&grad) = 0;
	virtual void update(const REAL lr) = 0;
	virtual void write(const string& file) = 0;
	virtual void report_and_reset() = 0;
	virtual void set_training(bool t) = 0;	// mainly for dropout
	static ModelZ* read_init(const string& file);
	static ModelZ* newone_init(const string& mss);
};

//#define USE_MODEL_DYNET
#define USE_MODEL_DYNET2 1
// #define USE_MODEL_DUMMY
//#define EIGEN_USE_MKL_ALL


#endif
