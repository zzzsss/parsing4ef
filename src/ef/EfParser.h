#ifndef _EF_EFPARSER
#define _EF_EFPARSER

#include "DpOptions.h"
#include "DpSentence.h"
#include "DpDictionary.h"
#include "../components/FeatureManager.h"
#include "../model/Model.h"

// the class of main easy-first parser
class EfParser{
private:
	// 0. options
	DpOptions options;
	// 1. corpus
	DPS_PTR corpus_train{nullptr};
	DPS_PTR corpus_dev{nullptr};
	DPS_PTR corpus_test{nullptr};
	// 2. essentials
	DpDictionary dict;
	FeatureManager* fm{nullptr};
	Model* model{nullptr};
	// sub-rountines

public:
	// all the confs are in the options, simple building
	EfParser(int argc, char** argv): options(argc, argv){}
	~EfParser(){
		delete corpus_train;
		delete corpus_dev;
		delete corpus_test;
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
