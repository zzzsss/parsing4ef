#include "ModelDynet.h"
#include <fstream>
#include <cmath>

#ifdef USE_MODEL_DYNET
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "dynet/nodes.h"
#include "dynet/expr.h"

using dynet::ParameterInitUniform;
using dynet::SimpleSGDTrainer;
using dynet::MomentumSGDTrainer;
using dynet::AdagradTrainer;
void ModelDynet::create_model()
{
	// 0. init
	int argc = 7;
	string tmp_wd_one = dp_num2str(sp->weight_decay);	// this bug is really ...
	const char* argv[] = {"", "--dynet_mem", sp->memory.c_str(), "--dynet-l2", tmp_wd_one.c_str(), "--dynet-seed", "12345"};
	char** argv2 = const_cast<char**>(&argv[0]);
	dynet::initialize(argc, argv2);
	// 1. mach
	mach = new typename dynet::Model();
	for(unsigned i = 0; i < sp->embed_outd.size(); i++){
		auto pi = ParameterInitUniform{sp->layer_initb[0]};
		auto one_param = mach->add_lookup_parameters(sp->embed_ind[i], {sp->embed_outd[i]});
		pi.initialize_params(one_param.get()->all_values);
		param_lookups.push_back(one_param);
	}
	for(unsigned i = 1; i < sp->layer_size.size(); i++){	// start with one
		REAL fanio = std::sqrt((REAL)(sp->layer_size[i]+sp->layer_size[i - 1]));
		param_w.push_back(mach->add_parameters({sp->layer_size[i], sp->layer_size[i-1]}, sp->layer_initw[i] / fanio));
		param_b.push_back(mach->add_parameters({sp->layer_size[i]}, sp->layer_initb[i]));
	}
	// 2. trainer
	switch(sp->update_mode){
	case SGD:
	{
		if(sp->momemtum <= 0)
			trainer = new SimpleSGDTrainer(mach, 1);
		else
			trainer = new MomentumSGDTrainer(mach, 1, sp->momemtum);
		break;
	}
	case ADAGRAD:
	{
		trainer = new AdagradTrainer(mach, 1);
		break;
	}
	default:
		Logger::Error("Unkown update mode.");
	}
}

const string SPEC_SUFFIX = ".spec";
ModelZ* ModelDynet::read_init(const string& file)
{
	// 1. read file.spec
	string file_spec = file + SPEC_SUFFIX;
	ifstream fin;
	fin.open(file_spec);
	Spec* sp = Spec::read(fin);
	fin.close();
	// 2. init
	ModelDynet* ret = new ModelDynet(sp);
	// 3. read init
	fin.open(file);
	boost::archive::text_iarchive ia(fin);
	ia >> ret->mach;
	for(auto& x : ret->param_lookups)
		ia >> x;
	for(auto& x : ret->param_w)
		ia >> x; 
	for(auto& x : ret->param_b)
		ia >> x;
	fin.close();
	return ret;
}
ModelZ* ModelDynet::newone_init(const string& mss)
{
	// 1. spec
	Spec* sp = new Spec{mss};
	// 2. init
	ModelDynet* ret = new ModelDynet(sp);
	return ret;
}
void ModelDynet::write(const string& file)
{
	// 1. write file.spec
	string file_spec = file + SPEC_SUFFIX;
	ofstream fout;
	fout.open(file_spec);
	sp->write(fout);
	fout.close();
	// 2. write model
	fout.open(file);
	boost::archive::text_oarchive oa(fout);
	oa << mach;
	for(auto& x : param_lookups)
		oa << x;
	for(auto& x : param_w)
		oa << x;
	for(auto& x : param_b)
		oa << x;
	fout.close();
	return;
}

using namespace dynet::expr;
using dynet::Dim;
using dynet::Reshape;
using dynet::VariableIndex;
/* ----------------------------------------------------- */
// special one to fix the unbatched reshape problem
// y = reshape(x_1, --> to)
struct SpecialReshape: public Reshape {
	explicit SpecialReshape(const std::initializer_list<VariableIndex>& a, const Dim& to):Reshape(a, to){}
	bool supports_multibatch() const override { return true; }
};
Expression sreshape(const Expression& x, const Dim& d) { return Expression(x.pg, x.pg->add_function<SpecialReshape>({x.i}, d)); }
/* ----------------------------------------------------- */

Expression ModelDynet::TMP_forward(ComputationGraph& cg, const vector<Input>& x)
{
	vector<Expression> weights;
	vector<Expression> biases;
	for(unsigned i = 0; i < param_w.size(); i++)
		weights.emplace_back(parameter(cg, param_w[i]));
	for(unsigned i = 0; i < param_b.size(); i++)
		biases.emplace_back(parameter(cg, param_b[i]));
	// 1. collect all the lookups
	vector<Expression> them;
	{
		// temp check
		unsigned su = 0;
		for(auto k : sp->embed_num)
			su += k;
		if(su != x[0]->size())
			Logger::Error("Unmatched token size for the input.");
	}
	for(auto* one: x){
		unsigned i = 0;
		unsigned next = 0;
		int which = 0;
		for(auto k: sp->embed_num){
			next += k;
			while(i < next){
				them.emplace_back(lookup(cg, param_lookups[which], (*one)[i]));
				i++;
			}
			which++;
		}
	}
	// 2. concate and reshape
	auto c0 = concatenate(them);
	// -- auto r_c0 = cg.get_value(c0);
	unsigned batch_size = x.size();
	// -- this is the problem (change batch) !! fix with self defined one !!
	auto h0 = sreshape(c0, Dim({(unsigned)sp->layer_size[0]}, batch_size));
	// -- 
	// -- auto r_h0 = cg.get_value(h0);
	// 3. forward next
	for(unsigned i = 0; i < param_w.size(); i++){
		h0 = weights[i] * h0 + biases[i];
		// currently no dropout
		switch(sp->layer_act[i + 1]){
		case LINEAR:
			break;
		case TANH:
			h0 = tanh(h0);
			break;
		default:
			Logger::Error("Unknown activation type.");
			break;
		}
	}
	return h0;
}

// forward, backward
vector<Output> ModelDynet::forward(const vector<Input>& x)
{
	if(x.size() <= 0)
		return vector<Output>{};
	ComputationGraph cg;
	auto results = TMP_forward(cg, x);
	auto pointer = cg.forward(results).v;
	int outdim = sp->layer_size.back();
	vector<Output> ret;
	for(unsigned i = 0; i < x.size(); i++){	// copy them
		Output one = new vector<REAL>(pointer, pointer+outdim);
		ret.push_back(one);
		pointer += outdim;
	}
	num_forw += x.size();
	return ret;
}

void ModelDynet::backward(const vector<Input>& in, const vector<int>&index, const vector<REAL>&grad)
{
	if(in.size() <= 0)
		return;
	ComputationGraph cg;
	auto results = TMP_forward(cg, in);
	// prepare gradients
	unsigned batch_size = in.size();
	unsigned outdim = sp->layer_size.back();
	vector<REAL>* the_grad = new vector<REAL>(batch_size*outdim, 0);
	for(unsigned i = 0; i < batch_size; i++)
		(*the_grad)[i*outdim + index[i]] = grad[i];
	auto expr_grad = input(cg, Dim({outdim}, batch_size), the_grad);
	auto dotproduct = dot_product(results, expr_grad);
	auto loss = sum_batches(dotproduct);
	cg.forward(loss);
	cg.backward(loss);
	delete the_grad;
	num_back += in.size();
}

#endif // USE_MODEL_DYNET
