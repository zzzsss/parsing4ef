#include "DpOptions.h"
#include <fstream>

// Read options from cmd, first read from files, then cmd arguments
//	"-" means laters are all cmd args.
DpOptions::DpOptions(int argc, char** argv)
{
	vector<pair<string, string>> ps;
	init(ps);
}

void DpOptions::init(vector<pair<string, string>>& ps)
{

}
