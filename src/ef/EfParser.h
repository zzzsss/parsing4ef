#ifndef _EF_EFPARSER
#define _EF_EFPARSER

#include "DpOptions.h"
#include "DpSentence.h"
#include "DpDictionary.h"
#include "../components/FeatureManager.h"
#include "../model/Model.h"
#include "EfTRHelper.h"

// the class of main easy-first parser
class EfParser{
private:
	// 0. options
	DpOptions options;
	// 1. corpus ...
	// 2. essentials
	DpDictionary* dict{nullptr};
	FeatureManager* fm{nullptr};
	Model* model{nullptr};
	// sub-rountines
	void do_train(DPS_PTR train, EfTRHelper* h);
	double do_dev_test(DPS_PTR test, DPS_PTR gold, string f_out, string f_gold);	// return acc if dev
public:
	// all the confs are in the options, simple building
	EfParser(int argc, char** argv);
	~EfParser(){
		delete dict;
		delete fm;
		delete model;
	}
	void run(){		// train & test
		if(options.iftrain)
			train();
		if(options.iftest)
			test();
		if(options.ifevaluate)
			evaluate();
	}
	// run them
	void train();
	void test();
	void evaluate();
};

#endif
