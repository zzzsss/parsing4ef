#ifndef _EF_DPOPTIONS
#define _EF_DPOPTIONS

#include <unordered_map>
#include <vector>
#include <string>
using namespace std;

class DpOptions{	//TODO: the options
private:
	// for initialization, a vector of "key value" pairs
	void init(vector<pair<string, string>>&);
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
	string file_output_dev;		// temp outputs for dev
	string file_output_test;	// output file for test
	// 1.1. model files
	string file_dict{"dictionary.txt"};	// to build for training or to load for testing
	string file_tdict{""};	// to load for training (train-dict)
	
	//1. about dictionary
	int dict_remove{2};		//remove words appears < this times; [default 2 (remove 1 time)]
	int dict_reorder{1};	//whether re-order the words according to frequency; [default True]
	//2. about features
	int ef_mode{0};		//ef_std or ef_eager or ...
	string fss{""};			//feature specifier: see FeatureManager for details (may repeat it and can overwrite)

	// Initialization
	DpOptions(int argc, char** argv);
};

#endif
