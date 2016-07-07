#include "DpTools.h"
#include "../ef/DpDictionary.h"

//#define TEST_EVALUATOR
#ifdef TEST_EVALUATOR
int main(int argc, char** argv){
	//dp_evaluate(string{argv[1]}, string{argv[2]});
	dp_evaluate("Debug-test\\test.right", "Debug-test\\output.txt");
}
#endif // TEST_EVALUATOR

#define TEST_DICTIONARY
#ifdef TEST_DICTIONARY
int main(int argc, char** argv){
	DPS_PTR train = read_corpus(argv[1]);
	DpDictionary d{};
	DpOptions x;
	d.build_map(train, x);
}
#endif // TEST_DICTIONARY
