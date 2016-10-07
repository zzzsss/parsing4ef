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
class Input{
public:
};

// vectored output
class Output{
public:
	virtual vector<REAL> get_vec() = 0;
};

class Model{
public:
	virtual ~Model(){}
	virtual vector<Input*> make_input(const vector<vector<int>>& x) = 0;		// this is the current one
	virtual vector<Output*> forward(const vector<Input*>& in) = 0;
	virtual void backward(const vector<Output*>& out, const vector<int>&index, const vector<REAL>&grad) = 0;
	virtual void update(const REAL lr) = 0;
	virtual void write(const string& file) = 0;
	static Model* read_init(const string& file);
	static Model* newone_init(int outd);
};

#endif
