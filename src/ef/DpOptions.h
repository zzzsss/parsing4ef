#ifndef _EF_DPOPTIONS
#define _EF_DPOPTIONS

#include <unordered_map>
using namespace std;

class DpOptions{
public:
	//1. about dictionary
	int dict_remove{2};	//remove words appears < this times; [default 2 (remove 1 time)]
};

#endif
