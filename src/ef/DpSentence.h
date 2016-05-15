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
class DpSentence {
public:		// all public: all init and assign are default
	//basic ones --- for simplicity: "_" means nope
	vector<string> forms;
	vector<string> postags;

	//gold ones
	vector<int> heads;
	vector<vector<int>> childs;		//for convenience
	vector<vector<int>> lchilds;		//near to far
	vector<vector<int>> rchilds;		//near to far
	vector<string> rels;

	//predict ones
	vector<int> predict_heads;
	vector<string> predict_rels;

	//some indexes --- after building the dictionary
	vector<int> index_forms;
	vector<int> index_postags;
	vector<int> index_rels;
	vector<int> index_predict_rels;

public:	
	//some routines
	int size() { return forms.size(); }
	// init or read/write files
	void read_one(const vector<unique_ptr<string>>& them);	//add one-line to (forms, postags, heads, rels)
	void finish_one();							//finish adding, currently only need build (the-childs)
	void write_this(ostream& fout);
};

// read and write them all
using DP_PTR = unique_ptr<DpSentence>;
using DPS_PTR = unique_ptr<vector<DP_PTR>>;
extern DPS_PTR read_corpus(string file);
extern void write_corpus(DPS_PTR& instances, string file);

#endif
