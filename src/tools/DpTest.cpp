#include "DpTools.h"
#include "../ef/DpDictionary.h"

//#define TEST_EVALUATOR
#ifdef TEST_EVALUATOR
int main(int argc, char** argv){
	//dp_evaluate(string{argv[1]}, string{argv[2]});
	dp_evaluate("Debug-test\\test.right", "Debug-test\\output.txt");
}
#endif // TEST_EVALUATOR

//#define TEST_DICTIONARY
#ifdef TEST_DICTIONARY
// ./a.out <train-file> <dictionary-file>
int main(int argc, char** argv){
	DPS_PTR train = read_corpus(argv[1]);
	DpDictionary d{};
	DpOptions x;
	d.build_map(train, x);
	d.index_dps(train);
	d.write(string(argv[2]));
	DpDictionary d2{};
	d2.read(string(argv[2]));
	cout << d2.num_word() << "-" << d2.num_pos() << "-" << d2.num_rel() << endl;
}
#endif // TEST_DICTIONARY
