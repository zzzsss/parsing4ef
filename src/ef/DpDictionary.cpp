#include "DpDictionary.h"
#include "../tools/DpTools.h"
#include <regex>

namespace{
	// !! This has to coordinate with DpDictionary's enums
	const vector<string> TEMP_SPE_WORD = {"<w-s>", "<w-e>", "w-unk"};
	const vector<string> TEMP_SPE_POS = {"<p-s>", "<p-e>", "p-unk"};
	// special treatment for words
	const regex TMP_renum{"[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+|[0-9]+\\\\/[0-9]+"};
	const vector<pair<regex, string>> TEMP_RE_MATCH
		= {{TMP_renum, "<num>"}};
}

void DpDictionary::build_map(DPS_PTR corpus, DpOptions& conf)
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
		for(int i = 1; i < one->rels.size(); i++)	// need special treatment
			if(map_rel.find(one->rels[i]) == map_rel.end())
				map_rel.insert({one->rels[i], map_rel.size()});
	}
	// 2. special treatment for words
	// -- first one scan for all words
	unordered_map<string, int> map_freq{};
	for(DP_PTR one : *corpus){
		for(string x : one->forms){
			for(auto p : TEMP_RE_MATCH){
				if(regex_match(x, p.first)){
					x = p.second;
					break;
				}
			}
			auto z = map_freq.find(x);
			if(z == map_freq.end())
				map_freq.insert({x, 1});
			else
				map_freq.at(x)++;
		}
	}
	// -- report before
	cout << "-- Build_map before: word/pos/rel are " << (num_word()+map_freq.size()) << "/" << num_pos() << "/" << num_rel() << endl;
	// 3. cut the low frequency ones
	cout << "-- Cut words less than " << conf.dict_remove << " times." << endl;
	for(auto iter: map_freq)
		if(iter.second >= conf.dict_remove)
			map_word.insert({iter.first, map_word.size()});
	// -- report after
	cout << "-- Build_map after: word/pos/rel are " << num_word() << "/" << num_pos() << "/" << num_rel() << endl;
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
