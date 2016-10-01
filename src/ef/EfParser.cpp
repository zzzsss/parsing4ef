#include "EfParser.h"
#include "../tools/DpTools.h"

// sub-rountines of protected methods

// MAIN public methods
void EfParser::train()
{
	// 1. prepare train and dev corpus (read corpus could be NULLPTR)
	corpus_train = read_corpus(options.file_train);
	corpus_dev = read_corpus(options.file_dev);
	// 2. init from outside or build dictionary && index corpus
	if(options.file_tdict != "")
		dict.read(options.file_tdict);
	if(dict.empty())
		dict.build_map(corpus_train, options);
	dict.index_dps(corpus_train);
	dict.index_dps(corpus_dev);
	dict.write(options.file_dict);
	// 3. build Scorer
	// 4. prepare others
	//TODO
	// 5. main training

}

void EfParser::test()
{
	// 1. prepare test corpus
	corpus_test = read_corpus(options.file_test);
	// 2. read dictionary if no training before
	if(dict.empty())
		dict.read(options.file_dict);
	dict.index_dps(corpus_test);
	// 3. read Scorer
	// 4. prepare others
	//TODO
	// 5. testing
}

void EfParser::evaluate()
{
	dp_evaluate(options.file_output_test, options.file_test);
}
