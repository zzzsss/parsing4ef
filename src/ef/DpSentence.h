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

class DpSentence {
public:		// all public: all init and assign are default
	//basic ones --- for simplicity: "_" means nope
	vector<string> forms;
	vector<string> words_norm;		//normalized words
	vector<string> postags;

	//gold ones
	vector<int> heads;
	vector<vector<int>> childs;		//for convenience
	vector<vector<int>> lchilds;		//near to far
	vector<vector<int>> rchilds;		//near to far
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
	void assign(vector<int>& h, vector<int>& r);
	int get_index_w(int i);
	int get_index_p(int i);
};

// read and write them all -- again point to vector of pointers
using DP_PTR = DpSentence*;
using DPS_PTR = vector<DP_PTR>*;
extern DPS_PTR read_corpus(string file);
extern void write_corpus(DPS_PTR instances, string file);

#endif
