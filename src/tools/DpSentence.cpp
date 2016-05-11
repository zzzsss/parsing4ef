#include "DpSentence.h"
#include <sstream>
#include <stdexcept>

// TMP helper methods
inline static int TMP_to_int(const string& x)
{
	stringstream tmp_str(x);
	int y = 0;
	tmp_str >> y;
	if(y==0 && x[0]!='0')
		throw exception("Int-Error: transfer to int.");
	return y;
}

// some routines for DpSentence
void DpSentence::read_one(const vector<string*>& them)
{
	//In fact, add one line and in CoNLL 08 format
	//1,index; 2,word; 3,pos; 4,head; 5,rel
	if(size() == 0){
		//append the root node
		forms.emplace_back("<root>");
		postags.emplace_back("<root-pos>");
		heads.emplace_back(-1);
		rels.emplace_back("<root-rel>");
	}
	if(TMP_to_int(*them[0]) != size())
		throw exception("Format-Error: wrong field[0].");
	forms.emplace_back(*them[1]);
	postags.emplace_back(*them[4]);
	heads.emplace_back(TMP_to_int(*them[8]));
	rels.emplace_back(*them[9]);
}

void DpSentence::finish_one()
{
	// currently only build child
	int len = size();
	childs = vector<vector<int>>(len);
	lchilds = vector<vector<int>>(len);
	rchilds = vector<vector<int>>(len);
	for(int i = 1; i < len; i++){
		int h = heads[i];
		childs[h].emplace_back(i);
		lchilds[h].emplace_back(i);
		rchilds[h].insert(rchilds[h].begin(), i);
	}
}

void DpSentence::write_this(ostream & fout)
{
	//CoNLL 08 format: len-1 lines plus an empty line
	//Print out predicted values
	int len = size();
	for(int i = 1; i < len; i++){
		fout << i << "\t" << forms[i] << "\t_\t_\t" << postags[i] << "\t_\t_\t_\t" 
			<< predict_heads[i] << "\t" << predict_rels[i] << "\n";
	}
	fout << "\n";
}

// read and write them all
vector<DpSentence*>* read_corpus(string file)
{
	ifstream fin;
	fin.open(file);
	vector<DpSentence*>* dps = new vector<DpSentence*>();
	//read them in
	string cur_line;
	DpSentence* one = new DpSentence();
	int all_tokens = 0;
	while(getline(fin, cur_line)){
		stringstream tmp_str(cur_line);
		vector<string*> tmp_fields;
		//split the fields for empty ones
		string* tmp_one = new string();
		while(tmp_str >> *tmp_one){
			tmp_fields.emplace_back(tmp_one);
		}
		delete tmp_one;
		//sentence
		if(tmp_fields.size() > 0){
			one->read_one(tmp_fields);
			for(auto ptr : tmp_fields)
				delete ptr;
		}
		else if(one->size() > 0){
			// finish one
			one->finish_one();
			dps->emplace_back(one);
			all_tokens += one->size() - 1;
			one = new DpSentence();		//allocate new one
		}
	}
	//finish the last one maybe
	if(one->size() > 0){
		// finish one
		one->finish_one();
		dps->emplace_back(one);
		all_tokens += one->size() - 1;
	}
	else
		delete one;
	fin.close();
	cout << "- Read file " << file << ": sentence " << dps->size() << "; tokens " << all_tokens << endl;
	return dps;
}

void write_corpus(vector<DpSentence*>* instances, string file)
{
	ofstream fout;
	fout.open(file);
	for(auto ins_ptr : *instances)
		ins_ptr->write_this(fout);
	fout.close();
}


