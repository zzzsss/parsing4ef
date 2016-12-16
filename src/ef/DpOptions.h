#ifndef _EF_DPOPTIONS
#define _EF_DPOPTIONS

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
using namespace std;

// option codes
// t0. the transition-system type
enum EF_MODE{ EF_STD, EF_EAGER };
// t1: obejct function for the training: perceptron, approximate CRF, implicit-ranking
enum LOSS_MODE{ LOSS_PERCEPTRON, LOSS_ACRF, LOSS_IMRANK };
// t2.1: the update strategy: till-the-end, max-violation, early-update, restart with one gold
enum UPDATE_MODE{ UPDATE_END, UPDATE_MAXV, UPDATE_EU, UPDATE_RESTART, UPDATE_END_WMU };
// t2.2: the divisor when updating (this influences lr): 1, current_len, sentence_len
enum UPDATEDIV_MODE{ UPDATEDIV_ONE, UPDATEDIV_CUR, UPDATEDIV_ALL };
// t3: recombination mode
enum RECOMB_MODE{ RECOMB_NOPE, RECOMB_STRICT, RECOMB_SPINE, RECOMB_SPINE2, RECOMB_TOPC, RECOMB_TOPC2, RECOMB_TOP, RECOMB_TOPC2_SPAN };

/*
	Explanation for the options:
	1. Separated by <BLANK>: different options
	2. Separated by ':": key:value
	3. fss: '|','-'; mss: '|','-'. (Thankfully we don't have negative-valued options)
*/
class DpOptions{	//TODO: the options
private:
	// for initialization, a vector of "key value" pairs
	void init(vector<pair<string, string>>&);
	void check_and_report();	// after init
public:
	// 0. procedures
	int iftrain{1};		// whether training; [default 1]
	int iftest{1};		// whether testing; [default 1]
	int ifevaluate{0};	// whether evaluating; [default 0]
	// 1. files
	// 1.0. corpus files, easy to see what it means
	string file_train;
	string file_dev;
	string file_test;
	string file_output_dev{"dev-output.txt"};		// temp outputs for dev
	string file_output_test{"test-output.txt"};		// output file for test
	// 1.1. model files
	string file_dict{"dictionary.txt"};	// to build for training or to load for testing
	string file_tdict{""};				// to load for training (train-dict)
	string file_model{"model.mach"};	// to build and save best for training and to load for testing
	string file_model_curr_suffix{".curr"};		// "" if nope
	string file_tmodel{""};				// to load for training (train-init-model)
	//2. about dictionary
	int dict_remove{2};		//remove words appears < this times; [default 2 (remove 1 time)]
	int dict_reorder{1};	//whether re-order the words according to frequency; [default True]
	//3. about transition & features
	int ef_mode{EF_STD};		//ef_std or ef_eager or ... (State)
	string fss{""};				//feature specifier: see FeatureManager for details (may repeat it and can overwrite)
	//4. about searching
	//4.1 training schemes
	double margin{1.0};		// margin for the scores
	int mloss_struct{1};
	int mloss_labels{1};
	int mloss_future{1};
	int mloss_span{0};		// span of a word / sentence length * loss_span
	int update_mode{UPDATE_END};		// update strategies (Agenda)
	int updatediv_mode{UPDATEDIV_CUR};	// what is the divisor for update
	int loss_mode{LOSS_PERCEPTRON};		// object when update (Agenda)
	//4.1.1 about rloss --- exp(x^\alpha) [default: exp(x)]
	int rloss_exp{1};
	double rloss_alpha{1.0};
	//4.2 beam sizes && recombination option
	unsigned beam_flabel{4};		// first filter for labels, which controls diversity on one beam
	unsigned beam_div{4};		// main diversity beam, controls diversity for same structure
	unsigned beam_all{8};		// final beam-size
	int recomb_mode{RECOMB_STRICT};		// recombination mode: 0: no recombination, 1: all-spine, 2: top+outside-child, 3: top
	int recomb_div{RECOMB_STRICT};		// unlabeled second beam recombination mode for structure diversity
	//4.3 when gold falls out of beam (notice when update, we always select the best ones)
	unsigned gold_inum{1};		// how many golds to insert when golds fall out of beam (could be less)
	int drop_is_drop{0};		// force drop when training
	float drop_random{0};		// random drop for exploration when training
	//5. about model
	string mss{""};			//model specifier: see model/Spec.h for details
	int dim_w{50};			// dimensions of embedding for word,pos,distance,label
	int dim_p{30};
	int dim_d{30};
	int dim_l{30};
	string embed_wl{""};	// embed word-list
	string embed_em{""};	// embed embed-file
	string embed_file{""};		// redundant option, wl_ef
	float embed_scale{1.0f};	// scale value
	//6. about training
	double tr_lrate{0.04};		// initial learning rate
	int tr_iters{12};			// training iterations
	double tr_cut{0.5};			// cutting rate for lr
	int tr_cut_times{0};		// at least cut this times (so real iters maybe more than iter)
	int tr_cut_iters{3};		// force cut if no cutting for how many iters
	double tr_cut_sthres{-1};	// score threshold, default -1 means never
	double tr_sample{0.95};		// sample rate, skip sentence randomly by (1-tr_sample)
	int tr_minibatch{2};		// number of sentences before one update
	//7. changes
	// string of changes (change at the beginning of iter): FORMAT: [change:<num>:c1|c2|...]
	vector<string> iter_changes{};	
	// Initialization
	DpOptions(int argc, char** argv);
	// change
	void change_self(int iter);
};

#endif
