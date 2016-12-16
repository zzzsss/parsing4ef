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
	param_w = vector<vector<Parameter>>(sp->param_num);
	param_b = vector<vector<Parameter>>(sp->param_num);
	for(unsigned i = 1; i < sp->layer_size.size(); i++){	// start with one
		REAL fanio = std::sqrt((REAL)(sp->layer_size[i] + sp->layer_size[i - 1]));
		for(int j = 0; j < sp->param_num; j++){
			param_w[j].push_back(mach->add_parameters({sp->layer_size[i], sp->layer_size[i - 1]}, sp->layer_initw[i] / fanio));
			param_b[j].push_back(mach->add_parameters({sp->layer_size[i]}, sp->layer_initb[i]));
		}
	}
	// -- 1.5 blstm --
	if(sp->blstm_size > 0){
		unsigned lstm_input_dim = 0;
		for(int i = 0; i < sp->blstm_tillembed; i++)
			lstm_input_dim += sp->embed_outd[i];
		// default initialization for them ...
		lstm_forward = new LSTMBuilder(sp->blstm_layer, lstm_input_dim, sp->blstm_size/2, mach);	// half of final output
		lstm_backward = new LSTMBuilder(sp->blstm_layer, lstm_input_dim, sp->blstm_size/2, mach);	// another half
		param_lstm_nope = mach->add_parameters({sp->blstm_size}, sp->layer_initb[0]);				// special NOPE node
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
	dynet::load_dynet_model(file, ret->mach);
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
	dynet::save_dynet_model(file, mach);
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

// fix the memory issue of cg-revert (!! this might be fixed in the newer version of dynet !!)
namespace{
	unsigned before_size = 0;
	void TMP_cg_checkpoint(ComputationGraph* cg){
		before_size = cg->nodes.size();
		cg->checkpoint();
	}
	void TMP_cg_revert(ComputationGraph* cg){
		for(unsigned i = before_size; i < cg->nodes.size(); i++)
			delete cg->nodes[i];
		cg->revert();
	}
}
/* ----------------------------------------------------- */

Expression ModelDynet::TMP_forward(const vector<Input>& x, int which)
{
	// first for which param
	const vector<Parameter>& param_wi = param_w[which];
	const vector<Parameter>& param_bi = param_b[which];
	vector<Expression> weights;
	vector<Expression> biases;
	for(unsigned i = 0; i < param_wi.size(); i++)
		weights.emplace_back(parameter(*cg, param_wi[i]));
	for(unsigned i = 0; i < param_bi.size(); i++)
		biases.emplace_back(parameter(*cg, param_bi[i]));
	// 1. collect all the lookups
	vector<Expression> them;
	{
		// temp check
		unsigned su = 0;
		for(auto k : sp->embed_num)
			su += k;
		if(su != x[0].first->size())
			Logger::Error("Unmatched token size for the input.");
	}
	for(auto& one_pair: x){
		// blstm ??
		if(sp->blstm_size > 0){
			// ---- special one ----
			for(auto one_token : *(one_pair.second)){	
				if(one_token == NONEXIST_INDEX)
					them.emplace_back(lstm_repr_spe[REPR_NOPE]);
				else if(one_token < 0)
					them.emplace_back(lstm_repr_spe[REPR_START]);
				else if(one_token >= (int)lstm_repr.size())
					them.emplace_back(lstm_repr_spe[REPR_END]);
				else
					them.emplace_back(lstm_repr[one_token]);
			}
			// ---- special one ----
		}
		// other embeddings
		const int GROUPS_BLSTM_INPUT = sp->blstm_tillembed;	// words or word+pos
		unsigned i = 0;
		unsigned next = 0;
		int which = 0;
		for(auto k: sp->embed_num){
			next += k;
			while(i < next){	// this is a potential bug, but always set blstm_remainembed==1 when not using lstm.
				if(sp->blstm_remainembed || which >= GROUPS_BLSTM_INPUT)	// skip them ??
					them.emplace_back(lookup(*cg, param_lookups[which], (*(one_pair.first))[i]));
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
	for(unsigned i = 0; i < param_wi.size(); i++){
		REAL this_drop = sp->layer_drop[i];
		if(this_drop > 0 && is_training){	// not for the last layer
			h0 = dropout(h0, this_drop);
		}
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

// return vector of vector of indexes
vector<vector<int>> ModelDynet::TMP_select_input(const vector<Input>& x)
{
	vector<vector<int>> ret(sp->param_num);	
	for(unsigned i = 0; i < x.size(); i++)
		ret[(x[i].which % sp->param_num)].push_back(i);	// here use %
	return ret;
}

// forward, backward
vector<Output> ModelDynet::forward(const vector<Input>& x)
{	
	if(x.size() <= 0)
		return vector<Output>{};
	vector<Output> ret(x.size(), nullptr);
	auto input_indexes = TMP_select_input(x);
	for(unsigned ni = 0; ni < input_indexes.size(); ni++){
		// prepare this input
		vector<Input> this_x;
		if(input_indexes[ni].empty())
			continue;
		for(int j : input_indexes[ni])
			this_x.emplace_back(x[j]);
		// real forward
		TMP_cg_checkpoint(cg);	// lstm info
		auto results = TMP_forward(this_x, ni);
		auto pointer = cg->forward(results).v;
		int outdim = sp->layer_size.back();
		for(int j : input_indexes[ni]){	// copy them
			Output one = new vector<REAL>(pointer, pointer + outdim);
			ret[j] = one;
			pointer += outdim;
		}
		TMP_cg_revert(cg);		// lstm info
	}
	num_forw += x.size();
	return ret;
}

void ModelDynet::backward(const vector<Input>& in, const vector<int>&index, const vector<REAL>&grad)
{
	if(in.size() <= 0)
		return;
	auto input_indexes = TMP_select_input(in);
	for(unsigned ni = 0; ni < input_indexes.size(); ni++){
		// prepare this input
		vector<Input> this_in;
		if(input_indexes[ni].empty())
			continue;
		for(int j : input_indexes[ni])
			this_in.emplace_back(in[j]);
		// real forward & backward
		TMP_cg_checkpoint(cg);	// lstm info
		auto results = TMP_forward(this_in, ni);
		// prepare gradients
		unsigned batch_size = this_in.size();
		unsigned outdim = sp->layer_size.back();
		vector<REAL>* the_grad = new vector<REAL>(batch_size*outdim, 0);
		for(unsigned i = 0; i < batch_size; i++){
			int LOCAL_index = input_indexes[ni][i];
			(*the_grad)[i*outdim + index[LOCAL_index]] = grad[LOCAL_index];
		}
		auto expr_grad = input(*cg, Dim({outdim}, batch_size), the_grad);
		auto dotproduct = dot_product(results, expr_grad);
		auto loss = sum_batches(dotproduct);
		cg->forward(loss);
		cg->backward(loss);
		delete the_grad;
		TMP_cg_revert(cg);		// lstm info
	}
	num_back += in.size();	
	return;
}

// build lstm repr
#include "../components/FeatureManager.h"	// depends on the Magic numbers
#include <algorithm>
#include "dynet/exec.h"

inline Expression TMP_concat(vector<Expression> x, int num)
{
	if(num == 1)
		return x[0];
	else
		return concatenate(x);
}

void ModelDynet::new_sentence(vector<vector<int>*> x)
{
	cg = new ComputationGraph();
	cg->ee->invalidate();		// on some machine, the no-init of ee could be a bug ...
	if(sp->blstm_size <= 0){
		zeroes(*cg, Dim{1});	// add a dummy node, maybe for kindof dynet's bug
		return;
	}
	// dropout setting
	if(sp->blstm_drop > 0){
		if(is_training){
			lstm_forward->set_dropout(sp->blstm_drop);
			lstm_backward->set_dropout(sp->blstm_drop);
		}
		else{	// in fact, testing should multply sth, but
			lstm_forward->disable_dropout();
			lstm_backward->disable_dropout();
		}
	}
	//
	Expression embed_start = TMP_concat(vector<Expression>{
		lookup(*cg, param_lookups[0], FeatureManager::settle_word(WORD_START)),		// WORD
		lookup(*cg, param_lookups[1], FeatureManager::settle_word(POS_START))		// POS
	}, sp->blstm_tillembed);
	Expression embed_end = TMP_concat(vector<Expression>{
		lookup(*cg, param_lookups[0], FeatureManager::settle_word(WORD_END)),		// WORD
		lookup(*cg, param_lookups[1], FeatureManager::settle_word(POS_END))		// POS
	}, sp->blstm_tillembed);
	//
	vector<Expression> expr_l2r;
	vector<Expression> expr_r2l;
	int length = x[0]->size();
	// forward lstm
	lstm_forward->new_graph(*cg);
	lstm_forward->start_new_sequence();
	expr_l2r.emplace_back(lstm_forward->add_input(embed_start));
	for(int i = 0; i < length; i++){
		Expression embed_cur = TMP_concat(vector<Expression>{
			lookup(*cg, param_lookups[0], FeatureManager::settle_word(x[0]->at(i))),		// WORD
				lookup(*cg, param_lookups[1], FeatureManager::settle_word(x[1]->at(i)))		// POS
		}, sp->blstm_tillembed);
		expr_l2r.emplace_back(lstm_forward->add_input(embed_cur));
	}
	expr_l2r.emplace_back(lstm_forward->add_input(embed_end));
	// backward lstm
	lstm_backward->new_graph(*cg);
	lstm_backward->start_new_sequence();
	expr_r2l.emplace_back(lstm_backward->add_input(embed_end));
	for(int i = length-1; i >= 0; i--){
		Expression embed_cur = TMP_concat(vector<Expression>{
			lookup(*cg, param_lookups[0], FeatureManager::settle_word(x[0]->at(i))),		// WORD
				lookup(*cg, param_lookups[1], FeatureManager::settle_word(x[1]->at(i)))		// POS
		}, sp->blstm_tillembed);
		expr_r2l.emplace_back(lstm_backward->add_input(embed_cur));
	}
	expr_r2l.emplace_back(lstm_backward->add_input(embed_start));
	std::reverse(expr_r2l.begin(), expr_r2l.end());
	// combine them
	lstm_repr.resize(length);
	for(int i = 0; i < length; i++)
		lstm_repr[i] = concatenate({expr_l2r[i+1], expr_r2l[i+1]});		// remember i+1
	lstm_repr_spe.resize(REPR_MAX);
	lstm_repr_spe[REPR_START] = concatenate({expr_l2r.front(), expr_r2l.front()});
	lstm_repr_spe[REPR_END] = concatenate({expr_l2r.back(), expr_r2l.back()});
	lstm_repr_spe[REPR_NOPE] = parameter(*cg, param_lstm_nope);
	zeroes(*cg, Dim{1});	// add a dummy node, maybe for kindof dynet's bug
}

void ModelDynet::end_sentence()
{
	delete cg;
	cg = nullptr;
	lstm_repr.clear();
	lstm_repr_spe.clear();
}

#endif // USE_MODEL_DYNET
