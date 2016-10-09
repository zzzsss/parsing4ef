#include "DpSentence.h"
#include "../tools/DpTools.h"
#include "DpDictionary.h"
#include <regex>

namespace{
	// special treatment for words
	const regex TMP_renum{"[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+|[0-9]+\\\\/[0-9]+"};
	const vector<pair<regex, string>> TEMP_RE_MATCH
		= {{TMP_renum, "<num>"}};
}

// some routines for DpSentence
void DpSentence::read_one(const vector<string>& them)
{
	//In fact, add one line and in CoNLL 08 format
	//1,index; 2,word; 3,pos; 4,head; 5,rel
	if(size() == 0){
		//append the root node
		forms.emplace_back("<root>");
		words_norm.emplace_back("<root>");
		postags.emplace_back("<root-pos>");
		heads.emplace_back(-1);
		rels.emplace_back("<root-rel>");
	}
	if(dp_str2int(them[0]) != size())
		Logger::Error("Format-Error: wrong field[0].");
	forms.emplace_back(them[1]);
	// -- norm with re
	string temp_norm = them[1];
	for(auto& p : TEMP_RE_MATCH){
		if(regex_match(temp_norm, p.first)){
			temp_norm = p.second;
			break;
		}
	}
	words_norm.emplace_back(temp_norm);
	// -- norm with re
	postags.emplace_back(them[4]);
	heads.emplace_back(dp_str2int(them[8]));
	rels.emplace_back(them[9]);
}

void DpSentence::finish_one()
{
	// currently only build child
	/* --- no use currently ---
	int len = size();
	childs = vector<vector<int>>(len);
	lchilds = vector<vector<int>>(len);
	rchilds = vector<vector<int>>(len);
	for(int i = 1; i < len; i++){
		int h = heads[i];
		childs[h].emplace_back(i);
		if(i < h)
			lchilds[h].emplace_back(i);
		else
			rchilds[h].insert(rchilds[h].begin(), i);
	}
	*/
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

// read and write them all -- no checking for file error
DPS_PTR read_corpus(string file)
{
	Recorder TMP_recorder{string{"read file "}+file};
	ifstream fin;
	fin.open(file);
	DPS_PTR dps{new vector<DP_PTR>()};
	//read them in
	string cur_line;
	DP_PTR one{new DpSentence()};
	int all_tokens = 0;
	while(getline(fin, cur_line)){
		stringstream tmp_str(cur_line);
		vector<string> tmp_fields;
		//split the fields for empty ones
		string tmp_one;
		while(tmp_str >> tmp_one)
			tmp_fields.emplace_back(tmp_one);
		//sentence
		if(tmp_fields.size() > 0)
			one->read_one(tmp_fields);
		else if(one->size() > 0){
			// finish one
			one->finish_one();
			all_tokens += one->size() - 1;
			dps->emplace_back(one);
			one = new DpSentence();		//allocate new one
		}
	}
	//finish the last one maybe
	if(one->size() > 0){
		// finish one
		one->finish_one();
		all_tokens += one->size() - 1;
		dps->emplace_back(one);
	}
	else
		delete one;
	fin.close();
	Logger::get_output() << "-- Read file " << file << ": sentence " << dps->size() << "; tokens " << all_tokens << endl;
	return dps;
}

void write_corpus(DPS_PTR instances, string file)
{
	Recorder TMP_recorder{string{"write file "}+file};
	ofstream fout;
	fout.open(file);
	for(auto& ins_ptr : *instances)
		ins_ptr->write_this(fout);
	fout.close();
}

// other interfaces
void DpSentence::assign(vector<int>& h, vector<int>& r){
	if(h.size() != heads.size() || r.size() != index_rels.size())
		Logger::Error("DpSentence assign not match.");
	predict_heads = h;
	index_predict_rels = r;
}
