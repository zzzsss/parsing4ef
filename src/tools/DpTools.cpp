#include "DpTools.h"

#ifdef _VERBOSE_ACCRECORDER
AccRecorder global_recorder{};
#endif

// split string with deliminator of one char
vector<string> dp_split(const string &s, char x, int cut_time)
{
	vector<string> ret;
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

// string to int
int dp_str2int(const string& x)
{
	stringstream tmp_str(x);
	int y = 0;
	tmp_str >> y;
	if(y == 0 && x[0] != '0')
		Logger::Error("Int-Error: transfer to int.");
	return y;
}