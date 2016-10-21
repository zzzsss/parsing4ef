#ifndef _MODEL_MODELDYNET
#define _MODEL_MODELDYNET

#include "Model.h"
#include "Spec.h"
#include "../tools/DpTools.h"
#ifdef USE_MODEL_DYNET

#include "dynet/dynet.h"
#include "dynet/training.h"

using dynet::expr::Expression;
using dynet::ComputationGraph;
using dynet::Parameter;
using dynet::LookupParameter;
using dynet::Model;

class ModelDynet: public ModelZ{
private:
	int num_forw{0};
	int num_back{0};
	// init
	void create_model();	// init with sp
	ModelDynet(Spec* ss): sp(ss){ create_model(); }
	//
	Spec* sp{nullptr};
	// init
	typename dynet::Model* mach{nullptr};
	typename dynet::Trainer* trainer{nullptr};
	// parameters
	vector<LookupParameter> param_lookups;	// sp->embed
	vector<Parameter> param_w;				//sp->layers
	vector<Parameter> param_b;
	//
	Expression TMP_forward(ComputationGraph& cg, const vector<Input>& x);
public:
	~ModelDynet(){ 
		delete sp; 
		delete trainer;
		delete mach;
	}
	static ModelZ* read_init(const string& file);
	static ModelZ* newone_init(const string& mss);
	vector<Output> forward(const vector<Input>& x) override;
	void backward(const vector<Input>& in, const vector<int>&index, const vector<REAL>&grad) override;
	void update(const REAL lr) override{
		trainer->update(lr);
	}
	void write(const string& file) override;
	void report_and_reset() override{
		Logger::get_output() << "- this time f/b:" << num_forw << "/" << num_back << endl;
		num_forw = num_back = 0;
	};
	void new_graph() override{}
};


#endif
#endif
