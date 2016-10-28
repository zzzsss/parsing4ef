#ifndef _MODEL_SPEC
#define _MODEL_SPEC

#include <string>
#include <vector>
#include <iostream>
using namespace std;
using REAL = float;	// same as Model.h

enum LAYER_ACT{ LINEAR, TANH };
enum TR_UPDATE_MODE{ SGD, ADAGRAD };

// the specifications for the model
// -- currently only support single digit, can overwrite, but must be sequential (h1 must follow h0)
struct Spec{
private:
	Spec() = default;
public:
	// layers: h*-s<size>-d<drop>-a<act>-i<init>
	vector<unsigned> layer_size;			// size
	vector<unsigned> layer_act;			// activation
	vector<REAL> layer_drop;	// dropout
	vector<REAL> layer_initw;	// [-init, init] for the weight and bias below the layer
	vector<REAL> layer_initb;	// bias and also for wv
	// embeds: e*-o<outd>-i<ind>-n<num>
	vector<unsigned> embed_outd;	// dimension of embedding
	vector<unsigned> embed_ind;	// vocab's size
	vector<unsigned> embed_num;	// how many embed in one instance
	// others(updates): o-<name>-value;
	int update_mode{SGD};
	REAL momemtum{0.6f};
	REAL weight_decay{1e-8f};
	string memory{"1024"};
	unsigned blstm_size{0};	// forward + backward: all size
	unsigned blstm_layer{1};
	int blstm_remainembed{1};	//remain original embed for words and pos?
	int blstm_tillembed{2};		//till which embed is the blstm's input (default 2: word and pos)
	REAL blstm_drop{0.f};
	//
	Spec(const string& mss);	// plus default mss
	void write(ostream& fout);
	static Spec* read(istream& fin);
};

#endif // !_MODEL_SPEC

