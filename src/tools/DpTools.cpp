#include "DpTools.h"

#ifdef _VERBOSE_ACCRECORDER
AccRecorder global_recorder{};
#endif

// split string with deliminator of one char
vector<string> dp_split(const string &s, char x)
{
	vector<string> ret;
	string before;
	for(char one : s){
		if(one == x){
			ret.emplace_back(before);
			before = "";
		}
		else
			before += one;
	}
	ret.emplace_back(before);
	return ret;
}

// string to int
inline int dp_str2int(const string& x)
{
	stringstream tmp_str(x);
	int y = 0;
	tmp_str >> y;
	if(y == 0 && x[0] != '0')
		throw runtime_error("Int-Error: transfer to int.");
	return y;
}