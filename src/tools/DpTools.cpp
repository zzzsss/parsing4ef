#include "DpTools.h"

#ifdef _VERBOSE_ACCRECORDER
AccRecorder global_recorder{};
#endif

// split string with deliminator of one char
vector<string> dp_split(const string &s, char x, int cut_time)
{
	vector<string> ret;
	if(s.empty())	// first check
		return ret;
	string before;
	int times = 0;
	for(char one : s){
		if(one == x && times != cut_time){	// again NEGATIVE trick
			ret.emplace_back(before);
			times++;
			before = "";
		}
		else
			before += one;
	}
	ret.emplace_back(before);
	return ret;
}
