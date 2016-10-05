#ifndef _MODEL_MODELDUMMY
#define _MODEL_MODELDUMMY

#include "Model.h"

// just a dummy testing stub

class InputDummy: public Input{};
class OutputDummy: public Output{
private:
	int num;
public:
	OutputDummy(int n): num(n){}
	vector<REAL> get_vec() override{
		vector<REAL> ret;
		for(int i = 0; i < num; i++)
			ret.push_back(1.0f / rand());
		return ret;
	}
};

class ModelDummy: public Model{
private:
	int out_dim;
public:
	~ModelDummy(){}
	ModelDummy(int n): out_dim(n){}
	vector<Input*> make_input(const vector<vector<int>>& x) override{
		vector<Input*> ret;
		for(auto& t : x)
			ret.emplace_back(new InputDummy{});
		return ret;
	}
	vector<Output*> forward(const vector<Input*>& in) override{
		vector<Output*> ret;
		for(auto* t : in)
			ret.emplace_back(new OutputDummy{out_dim});
		return ret;
	}
	void backward(const vector<Output*>& out, const vector<int>&index, const vector<REAL>&grad) override{}
	void update(const REAL lr) override{}
};

#endif