#ifndef _MODEL_MODEL
#define _MODEL_MODEL

#include <vector>
#include <string>
using std::vector;
using std::string;

// basic data type
using REAL = float;

// The interface to the models
// currently assuming simple feedforward model (simple input)
using Input = vector<int>*;
using Output = vector<REAL>*;

// the abstract class
// -- simplify this model, no caches
class ModelZ{
public:
	virtual ~ModelZ(){}
	virtual vector<Output> forward(const vector<Input>& x) = 0;
	virtual void backward(const vector<Input>& in, const vector<int>&index, const vector<REAL>&grad) = 0;
	virtual void update(const REAL lr) = 0;
	virtual void write(const string& file) = 0;
	virtual void report_and_reset() = 0;
	static ModelZ* read_init(const string& file);
	static ModelZ* newone_init(const string& mss);
};

#define USE_MODEL_DYNET
#define EIGEN_USE_MKL_ALL


#ifdef USE_MODEL_DYNET
#endif // USE_MODEL_DYNET


#endif
