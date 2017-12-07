#include "ModelDynet2.h"
#include <fstream>
#include <cmath>

#ifdef USE_MODEL_DYNET2
//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>
#include "dynet/nodes.h"
#include "dynet/expr.h"

using dynet::SimpleSGDTrainer;
using dynet::MomentumSGDTrainer;
using dynet::AdagradTrainer;
using dynet::AdamTrainer;

void ModelDynet2::create_model()
{
  // 0. init
  int argc = 7;
  string tmp_wd_one = dp_num2str(sp->weight_decay);	// this bug is really ...
  const char* argv[] = {"", "--dynet_mem", sp->memory.c_str(), "--dynet-l2", tmp_wd_one.c_str(), "--dynet-seed", "12345"};
  char** argv2 = const_cast<char**>(&argv[0]);
  dynet::initialize(argc, argv2);
  // 1. mach
  mach = new typename dynet::ParameterCollection();
  for(unsigned i = 0; i < sp->embed_outd.size(); i++){
    //-- use default ones: ParameterInitGlorot
    auto one_param = mach->add_lookup_parameters(sp->embed_ind[i], {sp->embed_outd[i]});
    param_lookups.push_back(one_param);
  }
  param_w = vector<vector<Parameter>>(sp->param_num);
  param_b = vector<vector<Parameter>>(sp->param_num);
  for(unsigned i = 1; i < sp->layer_size.size(); i++){
    //-- use default ones: ParameterInitGlorot and 0. for b
    for(int j = 0; j < sp->param_num; j++){
      param_w[j].push_back(mach->add_parameters({sp->layer_size[i], sp->layer_size[i-1]}));
      param_b[j].push_back(mach->add_parameters({sp->layer_size[i]}, 0.f));
    }
  }
  // -- 1.5 blstm --
  if(sp->blstm_size > 0){
    unsigned lstm_input_dim = 0;
    for(int i = 0; i < sp->blstm_tillembed; i++)
      lstm_input_dim += sp->embed_outd[i];
    //-- use default ones: maybe ParameterInitGlorot
    forward_lstms.push_back(new LSTMBuilder(1, lstm_input_dim, sp->blstm_size / 2, *mach));
    backward_lstms.push_back(new LSTMBuilder(1, lstm_input_dim, sp->blstm_size / 2, *mach));
    for(unsigned i = 1; i < sp->blstm_layer; i++){
      forward_lstms.push_back(new LSTMBuilder(1, sp->blstm_size, sp->blstm_size / 2, *mach));
      backward_lstms.push_back(new LSTMBuilder(1, sp->blstm_size, sp->blstm_size / 2, *mach));
    }
    param_lstm_nope = mach->add_parameters({sp->get_blstm_outsize()}, sp->layer_initb[0]);				// special NOPE node
    if(sp->blstm_fsize > 0){
      param_fmlp_w = mach->add_parameters({sp->blstm_fsize, sp->blstm_size});
      param_fmlp_b = mach->add_parameters({sp->blstm_fsize}, 0.f);
    }
  }
  // 2. trainer (currently using its own default lrate, schedule it later)
  switch(sp->update_mode){
  case SGD:
  {
    if(sp->momemtum <= 0)
      trainer = new SimpleSGDTrainer(*mach);
    else
      trainer = new MomentumSGDTrainer(*mach, 0.01, sp->momemtum);
    break;
  }
  case ADAGRAD:
  {
    trainer = new AdagradTrainer(*mach);
    break;
  }
  case ADAM:
  {
    trainer = new AdamTrainer(*mach);
    break;
  }
  default:
    Logger::Error("Unkown update mode.");
  }
}

const string SPEC_SUFFIX = ".spec";
ModelZ* ModelDynet2::read_init(const string& file)
{
  // 1. read file.spec
  string file_spec = file + SPEC_SUFFIX;
  ifstream fin;
  fin.open(file_spec);
  Spec* sp = Spec::read(fin);
  fin.close();
  // 2. init
  ModelDynet2* ret = new ModelDynet2(sp);
  // 3. read init
  dynet::load_dynet_model(file, ret->mach);
  return ret;
}
ModelZ* ModelDynet2::newone_init(const string& mss)
{
  // 1. spec
  Spec* sp = new Spec{mss};
  // 2. init
  ModelDynet2* ret = new ModelDynet2(sp);
  return ret;
}
void ModelDynet2::write(const string& file)
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

using dynet::Dim;
using dynet::Reshape;
using dynet::VariableIndex;

Expression ModelDynet2::TMP_forward(const vector<Input>& x, int which)
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
  // 1. collect all the lookups (TODO(warn): special weaving)
  vector<Expression> them;
  //vector<Expression> them_divs;
  //vector<Expression> them_final;   // final one idx if odd
  {
    // temp check
    unsigned su0=0, su1=0;
    for(unsigned i = 0; i < sp->embed_num.size(); i++){
      if(i < (unsigned)sp->blstm_tillembed)
        su0 += sp->embed_num[i];
      else
        su1 += sp->embed_num[i];
    }
    su0 /= 2;
    // checking only the first one assuming that the rest are the same
    if(su1 != x[0].first->size() || su0 != x[0].second->size())
      Logger::Error("Unmatched token size for the input.");
  }
  for(auto& one_pair : x){
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
    // other embeddings (only non-word ones)
    // !! sp->blstm_remainembed is always 0
    //const int GROUPS_BLSTM_INPUT = sp->blstm_tillembed;	// words or word+pos
    unsigned i = 0;
    unsigned next = 0;
    for(unsigned embed_which = sp->blstm_tillembed; embed_which < sp->embed_num.size(); embed_which++){
      auto k = sp->embed_num[embed_which];
      next += k;
      while(i < next){
        them.emplace_back(lookup(*cg, param_lookups[embed_which], (*(one_pair.first))[i]));
        i++;
      }
    }
  }
  //for(auto one : them)
  //  cout << one.dim() << "\t";
  // 2. concate and reshape
  auto c0 = concatenate(them);
  // -- auto r_c0 = cg.get_value(c0);
  unsigned batch_size = x.size();
  // -- this is the problem (change batch) !! fix with self defined one !!
  auto h0 = dynet::reshape(c0, Dim({(unsigned)sp->layer_size[0]}, batch_size));
  // -- 
  // -- auto r_h0 = cg.get_value(h0);
  // 3. forward next
  for(unsigned i = 0; i < param_wi.size(); i++){
    REAL this_drop = sp->layer_drop[i];
    if(this_drop > 0 && is_training){	// not for the last layer
      h0 = dynet::dropout(h0, this_drop);
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
vector<vector<int>> ModelDynet2::TMP_select_input(const vector<Input>& x)
{
  vector<vector<int>> ret(sp->param_num);
  for(unsigned i = 0; i < x.size(); i++)
    ret[(x[i].which % sp->param_num)].push_back(i);	// here use %
  return ret;
}

// forward, backward
vector<Output> ModelDynet2::forward(const vector<Input>& x)
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
    cg->checkpoint();	// lstm info
    auto results = TMP_forward(this_x, ni);
    auto pointer = cg->incremental_forward(results).v;
    int outdim = sp->layer_size.back();
    for(int j : input_indexes[ni]){	// copy them
      Output one = new vector<REAL>(pointer, pointer + outdim);
      ret[j] = one;
      pointer += outdim;
    }
    cg->revert();		// lstm info
  }
  num_forw += x.size();
  return ret;
}

void ModelDynet2::backward(const vector<Input>& in, const vector<int>&index, const vector<REAL>&grad)
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
    cg->checkpoint();	// lstm info
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
    cg->incremental_forward(loss);
    cg->backward(loss);
    delete the_grad;
    cg->revert();		// lstm info
  }
  num_back += in.size();
  return;
}

// build lstm repr
#include "../components/FeatureManager.h"	// depends on the Magic numbers
#include <algorithm>
#include "dynet/exec.h"

void ModelDynet2::new_sentence(vector<vector<int>*> x)
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
      for(auto* f : forward_lstms)
        f->set_dropout(sp->blstm_drop);
      for(auto* b : forward_lstms)
        b->set_dropout(sp->blstm_drop);
    }
    else{
      for(auto* f: forward_lstms)
        f->disable_dropout();
      for(auto* b : forward_lstms)
        b->disable_dropout();
    }
  }
  // calculate layer-by-layer
  // - the first layer with embeddings as inputs
  vector<Expression> sequence;
  int length = x[0]->size();
  Expression embed_start = dynet::concatenate({
    lookup(*cg, param_lookups[0], FeatureManager::settle_word(WORD_START)),
      lookup(*cg, param_lookups[1], FeatureManager::settle_word(POS_START))});
  sequence.push_back(embed_start);
  for(int i = 0; i < length; i++){
    Expression embed_cur = dynet::concatenate({
      lookup(*cg, param_lookups[0], FeatureManager::settle_word(x[0]->at(i))),		// WORD
        lookup(*cg, param_lookups[1], FeatureManager::settle_word(x[1]->at(i)))		// POS
    });
    sequence.push_back(embed_cur);
  }
  Expression embed_end = dynet::concatenate({
    lookup(*cg, param_lookups[0], FeatureManager::settle_word(WORD_END)),
      lookup(*cg, param_lookups[1], FeatureManager::settle_word(POS_END))});
  sequence.push_back(embed_end);
  // looping for the layers and get the final sequence
  for(unsigned li = 0; li < sp->blstm_layer; li++){
    vector<Expression> expr_l2r;
    vector<Expression> expr_r2l;
    auto lstm_forward = forward_lstms[li];
    auto lstm_backward = backward_lstms[li];
    lstm_forward->new_graph(*cg);
    lstm_forward->start_new_sequence();
    lstm_backward->new_graph(*cg);
    lstm_backward->start_new_sequence();
    for(int i = 0; i < length + 1; i++){
      expr_l2r.emplace_back(lstm_forward->add_input(sequence[i]));
      expr_r2l.emplace_back(lstm_backward->add_input(sequence[length-i]));
    }
    std::reverse(expr_r2l.begin(), expr_r2l.end());
    // combine
    sequence.clear();
    for(int i = 0; i < length + 1; i++)
      sequence.push_back(dynet::concatenate({expr_l2r[i], expr_r2l[i]}));
  }
  // final layer if any
  if(sp->blstm_fsize > 0){
    Expression expr_fmlp_w = parameter(*cg, param_fmlp_w);
    Expression expr_fmlp_b = parameter(*cg, param_fmlp_b);
    for(int i = 0; i < length + 1; i++){
      auto lstm_outv = sequence[i];
      // the embedding drop as dropout for these layers
      if(is_training && sp->layer_drop[0] > 0)
        lstm_outv = dynet::dropout(lstm_outv, sp->layer_drop[0]);
      auto TMP_expr = dynet::affine_transform({expr_fmlp_b, expr_fmlp_w, lstm_outv});
      TMP_expr = tanh(TMP_expr);
      sequence[i] = TMP_expr;
    }
  }
  // final repr
  lstm_repr.resize(length);
  for(int i = 0; i < length; i++)
    lstm_repr[i] = sequence[i+1];		// remember i+1
  lstm_repr_spe.resize(REPR_MAX);
  lstm_repr_spe[REPR_START] = sequence.front();
  lstm_repr_spe[REPR_END] = sequence.back();
  lstm_repr_spe[REPR_NOPE] = parameter(*cg, param_lstm_nope);
  zeroes(*cg, Dim{1});	// add a dummy node, maybe for kindof dynet's bug
}

void ModelDynet2::end_sentence()
{
  delete cg;
  cg = nullptr;
  lstm_repr.clear();
  lstm_repr_spe.clear();
}

#endif // USE_MODEL_DYNET
