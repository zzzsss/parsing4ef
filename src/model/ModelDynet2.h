#ifndef _MODEL_MODELDYNET2
#define _MODEL_MODELDYNET2

// dynet version is "commit bd2cb457612c0c0f3d5f1e82d5188e53660bcc7c" Sep. 5th, 2017
#include "Model.h"
#include "Spec.h"
#include "../tools/DpTools.h"
#include "../ef/DpDictionary.h"
#ifdef USE_MODEL_DYNET2

#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/lstm.h"

using dynet::Expression;
using dynet::ComputationGraph;
using dynet::Parameter;
using dynet::LookupParameter;
using dynet::ParameterCollection;
using dynet::LSTMBuilder;

class ModelDynet2: public ModelZ{
private:
  int num_forw{0};
  int num_back{0};
  bool is_training{false};	// mainly for dropout
                            // init
  void create_model();	// init with sp
  ModelDynet2(Spec* ss): sp(ss){ create_model(); }
  //
  Spec* sp{nullptr};
  // init
  typename dynet::ParameterCollection* mach{nullptr};
  typename dynet::Trainer* trainer{nullptr};
  // parameters
  vector<LookupParameter> param_lookups;	// sp->embed
  vector<vector<Parameter>> param_w;				//sp->layers
  vector<vector<Parameter>> param_b;
  // for remembering the states, the expressions from lstm are build one per sentence
  vector<LSTMBuilder*> forward_lstms;
  vector<LSTMBuilder*> backward_lstms;
  Parameter param_fmlp_w;
  Parameter param_fmlp_b;
  //
  ComputationGraph* cg{nullptr};
  vector<Expression> lstm_repr;	// [0, size-of-sentence)
  enum LSTM_SPECIAL_REPR{ REPR_START, REPR_END, REPR_NOPE, REPR_MAX };
  vector<Expression> lstm_repr_spe;	// special representations
  Parameter param_lstm_nope;			// REPR_NOPE (learnable)
                                  //
  Expression TMP_forward(const vector<Input>& x, int which);		// compute graph
  vector<vector<int>> TMP_select_input(const vector<Input>& x);	// return vectors of indexes
public:
  ~ModelDynet2(){
    delete sp;
    delete trainer;
    delete mach;
  }
  static ModelZ* read_init(const string& file);
  static ModelZ* newone_init(const string& mss);
  vector<Output> forward(const vector<Input>& x) override;
  void backward(const vector<Input>& in, const vector<int>&index, const vector<REAL>&grad) override;
  void update(const REAL lr) override{
    // check whether change lrate
    const REAL prev_lr = trainer->learning_rate;
    if(lr != prev_lr)
      trainer->restart(lr);
    trainer->update();
  }
  void write(const string& file) override;
  void report_and_reset() override{
    Logger::get_output() << "- this time f/b:" << num_forw << "/" << num_back << endl;
    num_forw = num_back = 0;
  };
  void new_sentence(vector<vector<int>*>) override;
  void end_sentence() override;
  void set_training(bool t) override{ is_training = t; }
  // special
  void init_embed(string CONF_embed_WL, string CONF_embed_EM, string CONF_embed_f, REAL CONF_embed_ISCALE, DpDictionary* dict);
};


#endif

#endif