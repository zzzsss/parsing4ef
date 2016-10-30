#ifndef _EF_DPSENTENCE
#define _EF_DPSENTENCE

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <memory>
using namespace std;

// this is the base one, multiple partial ones share this one instance
// SPECIFIED: <ROOT> as the index 0 ...

/*
	The building steps for DpSentence: 
		{ read_corpus -> [build_map] -> DpDictionary::index_dps -> [train/predict] -> DpDictionary::put_rels -> write_corpus }
	1. read_corpus: basic ones and gold ones
	2. index_dps: index_*
	3. predict: predict_heads + index_predict_rels
	4. put_rels: predict_rels
*/

// special reserved tokens indexes -- !! This has to coordinate with TEMP_SPE_*
enum SPEC_TOKENW{ WORD_START = 0, WORD_END, WORD_UNK };
enum SPEC_TOKENP{ POS_START = 0, POS_END, POS_UNK };

class DpSentence {
public:		// all public: all init and assign are default
	//basic ones --- for simplicity: "_" means nope
	vector<string> forms;
	vector<string> words_norm;		//normalized words
	vector<string> postags;

	//gold ones
	vector<int> heads;
	/* --- no use currently ---
	vector<vector<int>> childs;		//for convenience
	vector<vector<int>> lchilds;		//near to far
	vector<vector<int>> rchilds;		//near to far
	*/
	vector<int> spans;	// the size for the subtree rooted at this one
	vector<string> rels;

	//predict ones
	vector<int> predict_heads;
	vector<string> predict_rels;	//this is the last built one

	//some indexes --- after building the dictionary
	vector<int> index_forms;
	vector<int> index_postags;
	vector<int> index_rels;

	//predict one
	vector<int> index_predict_rels;

public:	
	//some routines
	int size() { return forms.size(); }
	// init or read/write files
	void read_one(const vector<string>& them);	//add one-line to (forms, words_norm, postags, heads, rels)
	void finish_one();							//finish adding, currently only need build (the-childs)
	void write_this(ostream& fout);
	// only these method can see golds
	bool is_correct(int m, int h, int r){ return heads[m] == h &&  index_rels[m] == r; }
	bool is_correct(int m, int h){ return heads[m] == h; }
	int get_head(int m){ return heads[m]; }
	int get_rel(int m){ return index_rels[m]; }
	int get_span(int m){ return spans[m]; }
	int get_pred_head(int m){ return predict_heads[m]; }
	int get_pred_rel(int m){ return index_predict_rels[m]; }
	void assign(vector<int>& h, vector<int>& r);
	inline int get_index_w(int i);
	inline int get_index_p(int i);
};

inline int DpSentence::get_index_w(int i){
	if(i < 0)
		return WORD_START;
	else if(i >= size())
		return WORD_END;
	else
		return index_forms[i];
}
inline int DpSentence::get_index_p(int i){
	if(i < 0)
		return POS_START;
	else if(i >= size())
		return POS_END;
	else
		return index_postags[i];
}

// read and write them all -- again point to vector of pointers
using DP_PTR = DpSentence*;
using DPS_PTR = vector<DP_PTR>*;
extern DPS_PTR read_corpus(string file);
extern void write_corpus(DPS_PTR instances, string file);
inline void free_corpus(DPS_PTR f){
	for(auto* p : *f)
		delete p;
	delete f;
}

#endif
