#include "DpOptions.h"
#include "../tools/DpTools.h"
#include <fstream>
#include <sstream>

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

#define DATA_LINE_LEN 10000
#define TMP_assign_key(s) if(key==#s) stringstream(value)>>s
#define TMP_print(pr, var, end) pr << #var << ":" << var << end

// Read options from cmd, first read from files, then cmd arguments
//	"--" means laters are all cmd args.
DpOptions::DpOptions(int argc, char** argv)
{
	vector<pair<string, string>> ps;
	if(argc < 2){
		Logger::Error("cmd error, it should be: command <file> [--] <key:value> ... ");
	}
	string conf{argv[1]};
	if(conf != "--"){
		ifstream fin;
		char line[DATA_LINE_LEN];
		fin.open(conf);
		if(!fin){
			fin.close();
			Logger::Error(string("Conf-file error: ")+conf);
		}
		while(!fin.eof()){
			string buf;
			fin >> buf;
			if(buf[0] == '#') { fin.getline(line, DATA_LINE_LEN); continue; } // skip comments
			auto them = dp_split(buf, ':', 1);
			if(them.size()==2)
				ps.emplace_back(std::make_pair(them[0], them[1]));
		}
		fin.close();
	}
	for(int i = 2; i < argc; i++){
		string one{argv[i]};
		auto them = dp_split(one, ':', 1);
		ps.emplace_back(std::make_pair(them[0], them[1]));
	}
	init(ps);
	check_and_report();
}

void DpOptions::init(vector<pair<string, string>>& ps)
{
	for(unsigned i = 0; i < ps.size(); i++){
		string key = ps[i].first;
		string value = ps[i].second;
		TMP_assign_key(iftrain);
		else TMP_assign_key(iftest);
		else TMP_assign_key(ifevaluate);
		else TMP_assign_key(file_train);
		else TMP_assign_key(file_dev);
		else TMP_assign_key(file_test);
		else TMP_assign_key(file_output_dev);
		else TMP_assign_key(file_output_test);
		else TMP_assign_key(file_dict);
		else TMP_assign_key(file_tdict);
		else TMP_assign_key(file_model);
		else TMP_assign_key(file_tmodel);
		else TMP_assign_key(dict_remove);
		else TMP_assign_key(dict_reorder);
		else TMP_assign_key(ef_mode);
		else if(key == "fss") fss = fss + '|' + value;	// special one append
		else TMP_assign_key(margin);
		else TMP_assign_key(update_mode);
		else TMP_assign_key(updatediv_mode);
		else TMP_assign_key(loss_mode);
		else TMP_assign_key(beam_flabel);
		else TMP_assign_key(beam_div);
		else TMP_assign_key(beam_all);
		else TMP_assign_key(recomb_mode);
		else TMP_assign_key(gold_inum);
		else if(key == "mss") mss = mss + '|' + value;	// special one append
		else TMP_assign_key(dim_w);
		else TMP_assign_key(dim_p);
		else TMP_assign_key(dim_d);
		else TMP_assign_key(dim_l);
		else TMP_assign_key(tr_lrate);
		else TMP_assign_key(tr_lrmul);
		else TMP_assign_key(tr_iters);
		else TMP_assign_key(tr_cut);
		else TMP_assign_key(tr_cut_times);
		else TMP_assign_key(tr_cut_iters);
		else TMP_assign_key(tr_sample);
		else TMP_assign_key(tr_minibatch);
		else Logger::Error(string("Unknown key") + key + ":" + value);
	}
}

void DpOptions::check_and_report()
{
	using ERROR_REPORTER = decltype(&Logger::Error);
	ostream& printer = Logger::get_output();
	ERROR_REPORTER errorer = &Logger::Error;
	printer << "-- Now check and report confs ..." << endl;
	// 0. procedures
	printer << "-0: train/test/eval:" << iftrain << iftest << ifevaluate << endl;
	// 1. files
	printer << "-1: files:";
	TMP_print(printer, file_train, ";");
	TMP_print(printer, file_dev, ";");
	TMP_print(printer, file_test, ";");
	TMP_print(printer, file_output_dev, ";");
	TMP_print(printer, file_output_test, ";");
	TMP_print(printer, file_dict, ";");
	TMP_print(printer, file_tdict, ";");
	TMP_print(printer, file_model, ";");
	TMP_print(printer, file_tmodel, endl);
	if(iftrain){
		if(file_train.empty())	errorer("No train-file.");
		if(file_dev.empty())	errorer("No dev-file.");
	}
	if(iftest){
		if(file_test.empty())	errorer("No test-file.");
	}
	// 2. dictionary
	printer << "-2: dictionary:" << dict_remove << "/" << dict_reorder << endl;
	// 3. transition & features
	printer << "-3: transition&feature:" << TMP_get_desc("ef", ef_mode) << "/" << fss << endl;
	// 4. searching
	printer << "-4.1: searching-scheme:" << margin << "/" << TMP_get_desc("update", update_mode) << "/"
		<< TMP_get_desc("updatediv", updatediv_mode) << "/" << TMP_get_desc("loss", loss_mode) << endl;
	printer << "-4.2: searching-beam:" << beam_flabel << "/" << beam_div << "/" << beam_all << "/"
		<< TMP_get_desc("recomb", recomb_mode) << endl;
	printer << "-4.3: searching-insert:" << gold_inum << endl;
	if(gold_inum <= 0 && loss_mode != LOSS_IMRANK)
		errorer("For current loss, inum should >0.");
	// 5. model
	printer << "-5: model:" << mss << endl;
	// 6. training
	// ... skip
	printer << "------------" << endl;
}