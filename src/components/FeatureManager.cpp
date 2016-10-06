#include "FeatureManager.h"
#include "State.h"
#include "../tools/DpTools.h"

namespace{
	// default options
	inline string TMP_get_fss(const string& s){
		string base_std = "m-5|mn1-3|mf1-3|mn2-3|mt1-3|mt2-1|mt3-1|h-5|hn1-3|hf1-3|hn2-3|ht1-3|ht2-1|ht3-1";
		string base_eager = base_std + "|hp1|hp1n2|hp2";
		string base_distance = "d-m-h";
		string base_label = "l-mn1|l-hn1";
		if(s == "efstd")
			return base_std + base_distance + base_label;
		else if(s == "sfeager")
			return base_eager + base_distance + base_label;
		else
			return s;
	}
}

// read from option
FeatureManager::FeatureManager(const string& fss, DpDictionary* d, int ef_mode): dictionary(d)
{
	// lookup the shortcuts
	string x = TMP_get_fss(fss);
	// analyse them -- this procedure does not care about efficiency
	auto them = dp_split(fss, '|');
	vector<string> fss_ds;
	vector<string> fss_la;
	for(auto& one : them){
		if(one[0] == 'd')
			fss_ds.emplace_back(move(one));
		else if(one[0] == 'l')
			fss_la.emplace_back(move(one));
		else{
			auto fields = dp_split(one, '-');
			int one_span = dp_str2int(fields[1]);
			string& one_name = fields[0];
			//if(index.find(one_name) != index.end())	// may repeat and overwrite
			//	throw runtime_error("Repeated fss.");
			index[one_name] = spans.size();
			spans.push_back(one_span);
		}
	}
	for(auto& one : fss_ds){
		auto fields = dp_split(one, '-');
		string& l = fields[1];
		string& r = fields[2];
		int li = index[l];	// throw if not found
		int ri = index[r];
		distance_pairs.emplace_back(make_pair(li, ri));
	}
	for(auto& one : fss_la){
		auto fields = dp_split(one, '-');
		string& l = fields[1];
		int li = index[l];	// throw if not found
		labels.emplace_back(li);
	}
	// check mode -- skip
	// TODO: as a warning
	// report
	cout << "#nodes:" << num_nodes() << "#nodes-all:" << num_nodes_all() 
		<< ", #distances:" << num_distances() << ", #labels:" << num_labels() << endl;
}

// this one is repeated many times, thus it is not efficient enough? 
//TODO: optimize?
Feature* FeatureManager::make_feature(State* s, int m, int h)
{
	vector<int> nodes(num_nodes(), -10000);
	vector<int> distances(num_distances(), -10000);
	vector<int> tlabels(num_labels(), -10000);
	// firstly the nodes: for simplicity
	for(decltype(index)::const_iterator x = index.begin(); x != index.end(); x++){
		// 1. the basic
		const string& name = x->first;
		const int place = x->second;
		int one = m;
		int other = h;
		if(name[0] == 'h'){
			one = h;
			other = m;
		}
		int reverse = ((other>one)?1:-1);
		// 2. locate the node
		int base = one;
		string::const_iterator iter = name.begin() + 1;	// the second one
		while(iter != name.end() && base >= 0){
			int num = *(iter + 1) - '0';
			switch(*iter){
			case 'c':	//near child
				base = s->travel_down(base, reverse, num);
				break;
			case 'd':	//near child backward
				base = s->travel_downmost(base, reverse);
				base = s->travel_up(base, num-1);
				break;
			case 'p':	//parent
				base = s->travel_up(base, num);
				break;
			case 't':	//top nodes
				base = s->travel_upmost(base);
				base = s->travel_lr(base, -1*reverse*num);
				break;
			case 'l':	//absolute-left child
				base = s->travel_down(base, -1 * num, 1);
				break;
			case 'r':	//absolute-rigth child
				base = s->travel_down(base, num, 1);
				break;
			case 'n':	//near 'other' node
				base = s->travel_down(base, reverse*num, 1);
				break;
			case 'f':	//far from 'other' node
				base = s->travel_down(base, -1*reverse*num, 1);
				break;
			default:
				throw runtime_error("Unkown fss.");
			}
			iter += 2;
		}
		//3. got it
		if(base < 0) // this based on that NOPE_YET<0
			nodes[place] = NON_EXIST;
		else
			nodes[place] = base;
	}
	// then labels
	for(auto x: tlabels){
		int one = nodes[x];
		if(one == NON_EXIST)
			tlabels.push_back(NON_NOPELABEL);	// this should be different from State::NOPE_YET
		else
			tlabels.push_back(s->get_rel_label(one));
	}
	// ok, record and return that -- use move
	auto ret = new Feature(move(nodes), move(tlabels));
	records.push_back(ret);
	return ret;
}

vector<int> FeatureManager::feature_expand(Feature* ff, DP_PTR sent)
{
	vector<int> ret;
	const vector<int>& vn = ff->getn();
	const vector<int>& vl = ff->getl();
	// word
	for(unsigned i = 0; i < vn.size(); i++){
		int cur = vn[i];
		int range = spans[i];
		for(int j = cur - range / 2; j <= cur + range / 2; j++)	// inside the window
			ret.push_back(settle_word(sent->get_index_w(j)));
	}
	// pos
	for(unsigned i = 0; i < vn.size(); i++){
		int cur = vn[i];
		int range = spans[i];
		for(int j = cur - range / 2; j <= cur + range / 2; j++)	// inside the window
			ret.push_back(settle_word(sent->get_index_p(j)));
	}
	// distance -- calculate now
	for(unsigned i = 0; i < distance_pairs.size(); i++){
		auto x = distance_pairs[i];
		int one = vn[x.first];
		int two = vn[x.second];
		if(one == NON_EXIST || two == NON_EXIST)
			ret.push_back(settle_distance(NON_DIS));
		else
			ret.push_back(settle_distance(one-two));
	}
	// relation
	for(auto i: vl){
		ret.push_back(settle_label(i));
	}
	return ret;
}
