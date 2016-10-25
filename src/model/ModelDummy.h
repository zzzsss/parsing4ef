#ifndef _MODEL_MODELDUMMY
#define _MODEL_MODELDUMMY

#include "Model.h"
#include <fstream>
#include <climits>
#include "../tools/DpTools.h"

// just a dummy testing stub

class ModelDummy: public ModelZ{
private:
	int num_forw{0};
	int num_back{0};
	int out_dim;
public:
	~ModelDummy(){}
	ModelDummy(int n): out_dim(n){}
	vector<Output> forward(const vector<Input>& x) override{
		vector<Output> ret;
		for(unsigned i = 0; i < x.size(); i++){
			Output one = new vector<REAL>(out_dim);
			for(int j = 0; j < out_dim; j++)
				(*one)[j] = REAL(rand()) / INT_MAX;
			ret.emplace_back(one);
		}
		num_forw += x.size();
		return ret;
	}
	void backward(const vector<Input>& in, const vector<int>&index, const vector<REAL>&grad) override{
		num_back += in.size();
	}
	void update(const REAL lr) override{}
	void write(const string& file){
		std::ofstream fout;
		fout.open(file);
		fout << out_dim;
		fout.close();
	}
	void report_and_reset() override{
		Logger::get_output() << "- this time f/b:" << num_forw << "/" << num_back << endl;
		num_forw = num_back = 0;
	}
	void clear(){}
	void new_sentence(vector<vector<int>*>){}
	void end_sentence(){}
};

#endif