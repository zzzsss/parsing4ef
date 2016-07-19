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
public:
	// special reserved tokens indexes -- !! This has to coordinate with TEMP_SPE_*
	enum{ WORD_START = 0, WORD_END, WORD_UNK };
	enum{ POS_START = 0, POS_END, POS_UNK };
	// main methods
	DpDictionary()=default;
	void build_map(DPS_PTR, DpOptions&);		// this is the initialization of the maps
	void index_dps(DPS_PTR);		// complete some of the rest of the DpSentences
	void put_rels(DPS_PTR);			// put back rel names for sentences
	//io
	DpDictionary(string file);
	void write(string file);
	//number -- should return the size of maps
	unsigned num_word(){ return map_word.size(); }
	unsigned num_pos(){ return map_pos.size(); }
	unsigned num_rel(){ return map_rel.size(); }
};

#endif
