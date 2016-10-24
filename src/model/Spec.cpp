#include "Spec.h"
#include "../tools/DpTools.h"

// Simple feed-forward nn with embedding as the first layer(h0)

namespace{
	const string DEFAULT_MSS = "h0-s0|h1-s256-a1|h2-s128-a1|h3-s0";	// one embedding layer plus three hidden layer 
	const int DEFAULT_ACT = LINEAR;
	const REAL DEFAULT_DROP = 0.0f;
	const REAL DEFAULT_INITW = 1.0f;
	const REAL DEFAULT_INITB = 0.1f;
}

Spec::Spec(const string& mss)
{
	// start
	string one = DEFAULT_MSS + '|' + mss;
	auto them = dp_split(one, '|');
	for(auto& s : them){
		if(s.empty())
			continue;
		auto fields = dp_split(s, '-');
		switch(fields[0][0]){
		case 'h':	// hidden layers
		{
			unsigned which = dp_str2num<unsigned>(s.substr(1));
			if(which == layer_size.size()){
				// append a new one
				layer_size.push_back(0);
				layer_act.push_back(DEFAULT_ACT);
				layer_drop.push_back(DEFAULT_DROP);
				layer_initw.push_back(DEFAULT_INITW);
				layer_initb.push_back(DEFAULT_INITB);
			}
			else if(which > layer_size.size())
				Logger::Error(string("mss ERROR, oversize ") + s);
			// modify those
			for(unsigned i = 1; i < fields.size(); i++){
				switch(fields[i][0]){
				case 's': layer_size[which] = dp_str2num<unsigned>(fields[i].substr(1)); break;
				case 'a': layer_act[which] = dp_str2num<unsigned>(fields[i].substr(1)); break;
				case 'd': layer_drop[which] = dp_str2num<REAL>(fields[i].substr(1)); break;
				case 'w': layer_initw[which] = dp_str2num<REAL>(fields[i].substr(1)); break;
				case 'b': layer_initb[which] = dp_str2num<REAL>(fields[i].substr(1)); break;
				default: Logger::Error(string("mss ERROR, unkown field ") + s);
				}
			}
			break;
		}
		case 'e':	// embeds
		{
			unsigned which = dp_str2num<unsigned>(s.substr(1));
			if(which == embed_outd.size()){
				// append a new one
				embed_outd.push_back(0);
				embed_ind.push_back(0);
				embed_num.push_back(0);
			}
			else if(which > embed_outd.size())
				Logger::Error(string("mss ERROR, oversize ") + s);
			// modify those
			for(unsigned i = 1; i < fields.size(); i++){
				switch(fields[i][0]){
				case 'o': embed_outd[which] = dp_str2num<unsigned>(fields[i].substr(1)); break;
				case 'i': embed_ind[which] = dp_str2num<unsigned>(fields[i].substr(1)); break;
				case 'n': embed_num[which] = dp_str2num<unsigned>(fields[i].substr(1)); break;
				default: Logger::Error(string("mss ERROR, unkown field") + s);
				}
			}
			break;
		}
		case 'o':	//others
		{
			if(fields[1] == "momemtum")
				momemtum = dp_str2num<REAL>(fields[2]);
			else if(fields[1] == "memory")
				memory = fields[2];
			else if(fields[1] == "update_mode")
				update_mode = dp_str2num<int>(fields[2]);
			else if(fields[1] == "blstm_size")
				blstm_size = dp_str2num<unsigned>(fields[2]);
			else if(fields[1] == "blstm_layer")
				blstm_layer = dp_str2num<unsigned>(fields[2]);
			else if(fields[1] == "blstm_remainembed")
				blstm_remainembed = dp_str2num<int>(fields[2]);
			else
				Logger::Error(string("mss ERROR, unkown field") + s);
			break;
		}
		default: Logger::Error(string("mss ERROR, unkown field") + s);
		}
	}
	// check and force modify them -- simple mode
	int h0 = 0;
	for(unsigned i = 0; i < embed_outd.size(); i++){
		h0 += embed_outd[i] * embed_num[i];
	}
	h0 += blstm_size*embed_num[0];	// number of tokens
	if(!blstm_remainembed){	// do not include WORD and POS embeddings
		h0 -= embed_outd[0] * embed_num[0];
		h0 -= embed_outd[1] * embed_num[1];
	}
	layer_size[0] = h0;
	layer_act[0] = LINEAR;
	layer_act.back() = LINEAR;
	// report
	write(Logger::get_output());
}

void Spec::write(ostream& fout)
{
	fout << layer_size.size() << '\n';
	for(unsigned i = 0; i < layer_size.size(); i++)
		fout << layer_size[i] << ' ' << layer_act[i] << ' ' << layer_drop[i] << ' ' << layer_initw[i] << ' ' << layer_initb[i] << '\n';
	fout << embed_outd.size() << '\n';
	for(unsigned i = 0; i < embed_outd.size(); i++)
		fout << embed_outd[i] << ' ' << embed_ind[i] << ' ' << embed_num[i] << '\n';
	fout << update_mode << ' ' << momemtum << ' ' << weight_decay << ' ' << memory << '\n';
}

Spec* Spec::read(istream& fin)
{
	Spec* one = new Spec();
	int lsize = 0;
	fin >> lsize;
	one->layer_size = vector<unsigned>(lsize);
	one->layer_act = vector<unsigned>(lsize);
	one->layer_drop = vector<REAL>(lsize);
	one->layer_initw = vector<REAL>(lsize);
	one->layer_initb = vector<REAL>(lsize);
	for(int i = 0; i < lsize; i++)
		fin >> one->layer_size[i] >> one->layer_act[i] >> one->layer_drop[i] >> one->layer_initw[i] >> one->layer_initb[i];
	int esize = 0;
	fin >> esize;
	one->embed_outd = vector<unsigned>(esize);
	one->embed_ind = vector<unsigned>(esize);
	one->embed_num = vector<unsigned>(esize);
	for(int i = 0; i < esize; i++)
		fin >> one->embed_outd[i] >> one->embed_ind[i] >> one->embed_num[i];
	fin >> one->update_mode >> one->momemtum >> one->weight_decay >> one->memory;
	one->write(Logger::get_output());	// report
	return one;
}
