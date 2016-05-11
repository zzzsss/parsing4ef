#ifndef _TOOLS_DPSENTENCE
#define _TOOLS_DPSENTENCE

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
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
	void read_one(const vector<string*>& them);	//add one-line to (forms, postags, heads, rels)
	void finish_one();							//finish adding, currently only need build (the-childs)
	void write_this(ostream& fout);
};

// read and write them all
extern vector<DpSentence*>* read_corpus(string file);
extern void write_corpus(vector<DpSentence*>* instances, string file);

// evaluate
extern double dp_evaluate(string act_file, string pred_file, bool labeled = true);

#endif
