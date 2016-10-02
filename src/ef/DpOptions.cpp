#include "DpOptions.h"
#include <fstream>

namespace{
	// get description of modes, also checking (these will throw if wrong mode)
	unordered_map<string, vector<string>> TMP_description = {
		{"ef", vector<string>{"standard", "eager"}},
		{"loss", vector<string>{"perceptron", "a-crf", "special"}},
		{"update", vector<string>{"till-end", "max-violation", "early-update", "restart"}},
		{"updatediv", vector<string>{"one", "current_len", "sentence_len"}},
		{"recomb", vector<string>{"nope", "strict", "spine", "spine2", "topc", "topc2", "top"}}
	};
	string TMP_get_desc(string one, int mode){
		return TMP_description.at(one).at(mode);
	}
}

// Read options from cmd, first read from files, then cmd arguments
//	"-" means laters are all cmd args.
DpOptions::DpOptions(int argc, char** argv)
{
	vector<pair<string, string>> ps;
	init(ps);
	check_and_report();
}

void DpOptions::init(vector<pair<string, string>>& ps)
{

}
