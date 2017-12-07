#include "DpOptions.h"
#include "../tools/DpTools.h"
#include <fstream>
#include <sstream>

namespace{
	// get description of modes, also checking (these will throw if wrong mode)
	unordered_map<string, vector<string>> TMP_description = {
		{"ef", vector<string>{"standard", "eager"}},
		{"loss", vector<string>{"perceptron", "a-crf", "reorder"}},
		{"update", vector<string>{"till-end", "max-violation", "early-update", "restart", "till-end-WithMultiUpdates"}},
		{"updatediv", vector<string>{"one", "current_len", "sentence_len"}},
		{"recomb", vector<string>{"nope", "strict", "spine", "spine2", "topc", "topc2", "top", "topc2+spanend"}}
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
		else if(key == "fss"){
			string x = fss.empty() ? "" : "|";
			fss = fss + x + value;	// special one append
		}
		else TMP_assign_key(margin);
		else TMP_assign_key(mloss_struct);
		else TMP_assign_key(mloss_labels);
		else TMP_assign_key(mloss_future);
		else TMP_assign_key(mloss_span);
		else TMP_assign_key(update_mode);
		else TMP_assign_key(updatediv_mode);
		else TMP_assign_key(loss_mode);
		else TMP_assign_key(rloss_exp);
		else TMP_assign_key(rloss_alpha);
		else TMP_assign_key(rloss_confine);
		else TMP_assign_key(beam_flabel);
		else if(key == "beam_div"){		// bind them when setting beam_div
			stringstream(value) >> beam_div;
			beam_flabel = beam_div;
		}
		else TMP_assign_key(beam_all);
		else TMP_assign_key(recomb_mode);
		else TMP_assign_key(recomb_divL);
		else TMP_assign_key(recomb_divU);
		else TMP_assign_key(gold_inum);
		else TMP_assign_key(drop_is_drop);
		else TMP_assign_key(drop_random);
		else if(key == "mss"){
			string x = mss.empty() ? "" : "|";
			mss = mss + x + value;	// special one append
		}
		else TMP_assign_key(dim_w);
		else TMP_assign_key(dim_p);
		else TMP_assign_key(dim_d);
		else TMP_assign_key(dim_l);
		else TMP_assign_key(embed_wl);
		else TMP_assign_key(embed_em);
		else TMP_assign_key(embed_file);
		else TMP_assign_key(embed_scale);
		else TMP_assign_key(tr_lrate);
		else TMP_assign_key(tr_lrate_lbound);
		else TMP_assign_key(tr_iters);
		else TMP_assign_key(tr_cut);
		else TMP_assign_key(tr_cut_times);
		else TMP_assign_key(tr_cut_iters);
    else TMP_assign_key(tr_nocut_iters);
		else TMP_assign_key(tr_cut_sthres);
		else TMP_assign_key(tr_sample);
		else TMP_assign_key(tr_minibatch);
    else TMP_assign_key(tr_report_freq);
		else if(key == "changes")
			iter_changes.push_back(value);
		else Logger::Error(string("Unknown key ") + key + ":" + value);
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
	printer << "-4.1: searching-scheme:(margin/lstrcut/llabels/lfuture/span)" 
		<< margin << "/" << mloss_struct << "/" << mloss_labels << "/" << mloss_future << "/" << mloss_span << "/"
		<< TMP_get_desc("update", update_mode) << "/" << TMP_get_desc("updatediv", updatediv_mode) << "/" << TMP_get_desc("loss", loss_mode) << endl;
	printer << "-4.1.1: rloss-mode:" << rloss_exp << "/" << rloss_alpha << endl;
	printer << "-4.2: searching-beam:" << beam_flabel << "/" << beam_div << "/" << beam_all << "/"
		<< TMP_get_desc("recomb", recomb_mode) << endl;
	printer << "-4.3: searching-insert/drop:" << gold_inum << "/" << drop_is_drop << endl;
	if(gold_inum <= 0 && loss_mode != LOSS_IMRANK)
		errorer("For current loss, inum should >0.");
	if(drop_is_drop > 0){	// check
		switch(update_mode){
		case UPDATE_EU: Logger::Warn("drop_is_drop will not affect EU."); break;
		case UPDATE_RESTART: errorer("drop_is_drop should be off for UPDATE_RESTART."); break;
		default: break;
		}
	}
	// 5. model
	printer << "-5: model:" << mss << endl;
	// 6. training
	printer << "-6: training: " << "lr/iter/cut-iter/mbatch " << tr_lrate << "/" << tr_iters 
		<< "/" << tr_cut_iters << "/" << tr_minibatch << "/" << endl;
	// 7. changes --- check them
	printer << "-7: changes: ";
	for(auto& s: iter_changes){
		auto them = dp_split(s, ':', 1);
		if(them.size() != 2)	
			errorer(string("Illegal change: ") + s);
		if(dp_str2num<int>(them[0]) >= tr_iters)
			errorer(string("Illegal change iter: ") + them[0]);
		printer << them[0] << "-c-" << them[1] << '\t';
	}
	// ... skip
	printer << "------------" << endl;
}

// this only has effect for the options's own parameters, not for fss,mss,tr_*,
// -- but, why change those
void DpOptions::change_self(int iter)
{
	vector<pair<string, string>> ps;
	for(auto& s : iter_changes){
		auto them = dp_split(s, ':', 1);
		int it = dp_str2num<int>(them[0]);
		if(iter == it){	// hit
			auto those = dp_split(them[1], '|');
			for(auto& x : those){
				auto one_split = dp_split(x, ':', 1);
				ps.emplace_back(std::make_pair(one_split[0], one_split[1]));
			}
		}
	}
	for(auto& x : ps)
		Logger::get_output() << "-- Change options at iter " << iter << ": " << x.first << "=>" << x.second << endl;
	init(ps);
}
