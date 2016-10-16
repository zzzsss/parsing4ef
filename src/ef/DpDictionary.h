#ifndef _EF_DPDICTIONARY
#define _EF_DPDICTIONARY

#include "DpSentence.h"
#include "DpOptions.h"
#include <unordered_map>
using namespace std;

// the dictionary for the parser
// -- main functions: build_map, index_dps, put_rels, read&write, <ENUMs>
class DpDictionary{
private:
	// mapping and reverse map
	unordered_map<string, int> map_word;
	unordered_map<string, int> map_pos;
	unordered_map<string, int> map_rel;
	vector<string> list_word;
	vector<string> list_pos;
	vector<string> list_rel;
	//
	DpDictionary() = default;
	void build_map(DPS_PTR, const DpOptions&);		// this is the initialization of the maps
public:
	// main methods
	void index_dps(DPS_PTR);		// complete some of the rest of the DpSentences
	void put_rels(DPS_PTR);			// put back rel names for sentences
	//init and io
	static DpDictionary* read_init(const string& file);
	static DpDictionary* newone_init(DPS_PTR dps, const DpOptions& op);
	void write(string file);
	//number -- should return the size of maps
	unsigned num_word(){ return map_word.size(); }
	unsigned num_pos(){ return map_pos.size(); }
	unsigned num_rel(){ return map_rel.size(); }
	//helpers
	void clear(){
		for(auto& x : vector<decltype(map_word)>{map_word, map_pos, map_rel})
			x.clear();
		for(auto& x : vector<decltype(list_word)>{list_word, list_pos, list_rel})
			x.clear();
	}
	bool empty(){	// this check should be enough
		return num_word() == 0;
	}
};

#endif
