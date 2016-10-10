#include "EfParser.h"
#include "../tools/DpTools.h"
#include "../components/Searcher.h"
#include <cstdlib>
#include <sstream>
#include <climits>

// sub-rountines of protected methods

// MAIN public methods
void EfParser::train()
{
	// 1. prepare train and dev corpus (read corpus could be NULLPTR)
	auto corpus_train = read_corpus(options.file_train);
	auto corpus_dev = read_corpus(options.file_dev);
	// 2. init from outside or build dictionary && index corpus
	if(options.file_tdict != "")
		dict = DpDictionary::read_init(options.file_tdict);
	if(dict == nullptr)
		dict = DpDictionary::newone_init(corpus_train, options);
	dict->index_dps(corpus_train);
	dict->index_dps(corpus_dev);
	dict->write(options.file_dict);
	// 3. build FeatureManager and Model
	fm = new FeatureManager{options.fss, dict, options.ef_mode};
	if(options.file_tmodel != "")
		model = Model::read_init(options.file_tmodel);
	if(model == nullptr)
		model = Model::newone_init(dict->num_rel());	// todo: specifying model
	// 4. main training
	EfTRHelper helper{&options};
	while(helper.keepon()){
		ACCRECORDER_RESET("training");
		Recorder TMP_recorder{string("one-iter")};
		do_train(corpus_train, &helper);
		model->report_and_reset();
		double cur_s = do_dev_test(corpus_dev, corpus_dev, options.file_output_dev, options.file_dev);
		if(helper.end_iter_save(cur_s))
			model->write(options.file_model);
		model->report_and_reset();
		ACCRECORDER_REPORT();
	}
}

void EfParser::test()
{
	// 1. prepare test corpus
	auto corpus_test = read_corpus(options.file_test);
	// 2. read dictionary if no training before
	if(dict == nullptr)
		dict = DpDictionary::read_init(options.file_dict);
	dict->index_dps(corpus_test);
	// 3. build FeatureManager and read Model
	if(fm == nullptr)
		fm = new FeatureManager{options.fss, dict, options.ef_mode};
	if(model == nullptr)
		model = Model::read_init(options.file_model);	// read from best model
	// 4. testing
	ACCRECORDER_RESET("testing");
	do_dev_test(corpus_test, nullptr, options.file_output_test, options.file_test);
	ACCRECORDER_REPORT();
}

void EfParser::evaluate()
{
	dp_evaluate(options.file_output_test, options.file_test);
}

// init
EfParser::EfParser(int argc, char** argv): options(argc, argv)
{
	srand(12345);	// fix this
}

double EfParser::do_dev_test(DPS_PTR test, DPS_PTR gold, string f_out, string f_gold)
{
	int token_num = 0;	//token number
	int token_correct = 0;
	Recorder::report_time("testing");
	Searcher se{&options, false, model, fm};
	// add the results to test in place
	for(unsigned int i = 0; i<test->size(); i++){
		auto* t = (*test)[i];
		int length = t->size();
		token_num += length - 1;
		se.ef_search(t);	// perform the ef-parsing
		if(gold){
			for(int i2 = 1; i2 < length; i2++){	//ignore root
				if(t->get_head(i2) == gold->at(i)->get_pred_head(i2))	// unlabeled
					token_correct++;
			}
		}
	}
	se.report_stat(Logger::get_output());
	double rate = (token_correct + 0.0) / token_num;
	{
		stringstream tmpss;
		string tmps;
		tmpss << "finish-testing[" << token_correct << "/" << token_num << "/" << rate << "]";
		tmpss >> tmps;
		Recorder::report_time(tmps);
	}
	dict->put_rels(test);
	write_corpus(test, f_out);
	dp_evaluate(f_gold, f_out);
	return rate;
}

void EfParser::do_train(DPS_PTR train, EfTRHelper* h)
{
	int acc_sent_num = 0;	// for update
	Recorder::report_time("training iter " + dp_num2str(h->get_iter()) + " with " + dp_num2str(h->get_lrate()));
	Searcher se{&options, true, model, fm};
	for(unsigned int i = 0; i < train->size(); i++){
		if(((rand()+0.0)/INT_MAX) > options.tr_sample)	// skip
			continue;
		acc_sent_num++;
		auto* t = (*train)[i];
		se.ef_search(t);
		if(acc_sent_num >= options.tr_minibatch){
			model->update(h->get_lrate());
			acc_sent_num = 0;
		}
	}
	se.report_stat(Logger::get_output());
	return;
}
