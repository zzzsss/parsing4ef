#include "EfParser.h"
#include "../tools/DpTools.h"
#include "../components/Searcher.h"
#include <cstdlib>
#include <sstream>
#include <climits>
#include "../model/ModelDynet.h"

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
		model = ModelZ::read_init(options.file_tmodel);
	if(model == nullptr){
		stringstream tempss;
		tempss << "|e0-o" << options.dim_w << "-i" << fm->get_wind() << "-n" << fm->num_nodes_all()
			<< "|e1-o" << options.dim_p << "-i" << fm->get_pind() << "-n" << fm->num_nodes_all()
			<< "|e2-o" << options.dim_d << "-i" << fm->get_dind() << "-n" << fm->num_distances()
			<< "|e3-o" << options.dim_l << "-i" << fm->get_lind() << "-n" << fm->num_labels()
			<< "|hz-s" << dict->num_rel() << "|";
		string mss_embed;
		tempss >> mss_embed;
		model = ModelZ::newone_init(options.mss+mss_embed);
		if(ModelDynet* dy = dynamic_cast<ModelDynet*>(model))	// init from pre-trained embeddings
			dy->init_embed(options.embed_wl, options.embed_em, options.embed_file, options.embed_scale, dict);
	}
	// 4. main training
	EfTRHelper helper{&options};
	while(helper.keepon()){
		ACCRECORDER_RESET("training");
		Recorder TMP_recorder{string("one-iter")};
		options.change_self(helper.get_iter());
		do_train(corpus_train, &helper);
		model->report_and_reset();
		Searcher::report_and_reset_all();
		if(!options.file_model_curr_suffix.empty())
			model->write(options.file_model + options.file_model_curr_suffix);
		double cur_s = do_dev_test(corpus_dev, corpus_dev, options.file_output_dev, options.file_dev);
		if(helper.end_iter_save(cur_s))
			model->write(options.file_model);
		model->report_and_reset();
		Searcher::report_and_reset_all();
		ACCRECORDER_REPORT();
	}
	helper.report();
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
	delete model;
	model = nullptr;
	model = ModelZ::read_init(options.file_model);	// read from best model (memory-leak, but nevermind)
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
	srand(12345);	// fix this to a constant
	for(int i = 0; i < argc; i++)
		Logger::get_output() << argv[i] << '\t';
	Logger::get_output() << endl;
	// do some inits
	Searcher::init_all(&options);
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
	Logger::get_output() << endl;
	Recorder::report_time("training iter " + dp_num2str(h->get_iter()) + " with " + dp_num2str(h->get_lrate()));
	Searcher se{&options, true, model, fm};
	for(unsigned int i = 0; i < train->size(); i++){
		if(((rand()+0.0)/INT_MAX) > options.tr_sample)	// skip
			continue;
		acc_sent_num++;
		auto* t = (*train)[i];
		se.ef_search(t);
		if(acc_sent_num >= options.tr_minibatch){
			model->update(h->get_lrate()/options.tr_minibatch);		// divide it by minibatch
			acc_sent_num = 0;
		}
	}
	se.report_stat(Logger::get_output());
	return;
}
