#include "DpDictionary.h"
#include "../tools/DpTools.h"
#include <fstream>
#include <algorithm>

namespace{
	// !! This has to coordinate with the enums SPEC_TOKEN*
	const vector<string> TEMP_SPE_WORD = {"<w-s>", "<w-e>", "<w-unk>"};
	const vector<string> TEMP_SPE_POS = {"<p-s>", "<p-e>", "<p-unk>"};

	// lookup in a map with default
	int TEMP_lookup(unordered_map<string, int>& m, string& s, int de)
	{
		auto z = m.find(s);
		if(z == m.end()){
			if(de < 0)	//throw an runtime
				Logger::Error("Lookup Error.");
			else
				return de;
		}
		else
			return z->second;
		return 0;
	}
}

void DpDictionary::build_map(DPS_PTR corpus, const DpOptions& conf)
{
	// build all the maps
	Recorder TMP_recorder{"build_map"};
	// 0. prepare the default ones
	for(string x : TEMP_SPE_WORD)
		map_word.insert({x, map_word.size()});
	for(string x : TEMP_SPE_POS)
		map_pos.insert({x, map_pos.size()});
	// 1. add pos and rel, no processing
	for(DP_PTR one : *corpus){
		// pos and rel
		for(string x: one->postags)
			if(map_pos.find(x) == map_pos.end())
				map_pos.insert({x, map_pos.size()});
		for(unsigned i = 1; i < one->rels.size(); i++)	// need special treatment
			if(map_rel.find(one->rels[i]) == map_rel.end())
				map_rel.insert({one->rels[i], map_rel.size()});
	}
	// 2. special treatment for words
	// -- first one scan for all words
	unordered_map<string, int> map_freq{};
	for(DP_PTR one : *corpus){
		for(string x : one->words_norm){
			auto z = map_freq.find(x);
			if(z == map_freq.end())
				map_freq.insert({x, 1});
			else
				map_freq.at(x)++;
		}
	}
	// -- report before
	Logger::get_output() << "-- Build_map before: word/pos/rel are " << (num_word()+map_freq.size()) << "/" << num_pos() << "/" << num_rel() << endl;
	// 3. cut the low frequency ones
	Logger::get_output() << "-- Cut words less than " << conf.dict_remove << " times." << endl;
	if(conf.dict_reorder){
		vector<pair<string, int>> temp_flist;
		for(auto iter : map_freq)
			temp_flist.push_back({iter.first, iter.second});
		using TEMP_type = pair<string, int>;
		sort(temp_flist.begin(), temp_flist.end(), [](TEMP_type a, TEMP_type b){ return a.second > b.second; });
		for(auto iter : temp_flist){
			if(iter.second >= conf.dict_remove)
				map_word.insert({iter.first, map_word.size()});
			else
				break;
		}
	}
	else{
		for(auto iter : map_freq)
			if(iter.second >= conf.dict_remove)
				map_word.insert({iter.first, map_word.size()});
	}
	// -- report after
	Logger::get_output() << "-- Build_map after: word/pos/rel are " << num_word() << "/" << num_pos() << "/" << num_rel() << endl;
	// 4. put them into the list
	list_word.resize(num_word());
	list_pos.resize(num_pos());
	list_rel.resize(num_rel());
	for(auto i : map_word)
		list_word[i.second] = i.first;
	for(auto i : map_pos)
		list_pos[i.second] = i.first;
	for(auto i : map_rel)
		list_rel[i.second] = i.first;
	return;
}

void DpDictionary::index_dps(DPS_PTR corpus)
{
	// three vectors to be indexed
	for(DP_PTR one : *corpus){
		for(auto& x : one->words_norm)
			one->index_forms.emplace_back(TEMP_lookup(map_word, x, WORD_UNK));
		for(auto& x : one->postags)
			one->index_postags.emplace_back(TEMP_lookup(map_pos, x, POS_UNK));
		one->index_rels.emplace_back(0);	// not important
		for(unsigned i = 1; i < one->rels.size(); i++)	// need special treatment
			one->index_rels.emplace_back(TEMP_lookup(map_rel, one->rels[i], -1));
	}
}

void DpDictionary::put_rels(DPS_PTR corpus)
{
	// final step: predicted rel index -> rel strings
	for(DP_PTR one : *corpus){
		one->predict_rels.clear();	// bug for dev corpus (multiple assign).
		one->predict_rels.emplace_back("");
		for(unsigned i = 1; i < one->index_predict_rels.size(); i++){
			int x = one->index_predict_rels[i];
			one->predict_rels.emplace_back(list_rel.at(x));
		}
	}
}

// -- IO: read and write
// --- format: #words ... #pos ... #rel ...
DpDictionary* DpDictionary::read_init(const string& file)
{
	Recorder TMP_recorder{string{"Read maps "}+file};
	// clear possibly
	ifstream fin;
	fin.open(file);
	if(!fin){
		return nullptr;
	}
	DpDictionary* ret = new DpDictionary{};
	vector<pair<vector<string>*, unordered_map<string, int>*>> temp_maps{
		{&ret->list_word, &ret->map_word}, {&ret->list_pos, &ret->map_pos}, {&ret->list_rel, &ret->map_rel}};
	for(auto x : temp_maps){
		auto the_list = x.first;
		auto the_map = x.second;
		int number = 0;
		fin >> number;
		the_list->resize(number);	// this is somewhat dangerous, but ...
		for(int i = 0; i < number; i++){
			int tmp_index;
			string tmp_str;
			fin >> tmp_str >> tmp_index;
			the_list->at(tmp_index) = tmp_str;
			the_map->insert({tmp_str, tmp_index});
		}
		if(the_map->size() != the_list->size())
			Logger::Error("Map read error.");
	}
	fin.close();
	return ret;
}

DpDictionary* DpDictionary::newone_init(DPS_PTR dps, const DpOptions& op)
{
	DpDictionary* ret = new DpDictionary{};
	ret->build_map(dps, op);
	return ret;
}

void DpDictionary::write(string file)
{
	Recorder TMP_recorder{string{"Write maps "}+file};
	ofstream fout;
	fout.open(file);
	vector<pair<vector<string>*, unordered_map<string, int>*>> temp_maps{{&list_word, &map_word},{&list_pos, &map_pos},{&list_rel, &map_rel}};
	for(auto x : temp_maps){
		auto the_list = x.first;
		auto the_map = x.second;
		fout << the_list->size() << endl;
		for(auto& s : *the_list)
			fout << s << " " << the_map->at(s) << endl;
	}
	fout.close();
}

