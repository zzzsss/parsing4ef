#include "Model.h"
#include "ModelDummy.h"
#include <fstream>
using namespace std;

// todo: specifying model
Model* Model::read_init(const string& file)
{
	ifstream fin;
	fin.open(file);
	int outd = 0;
	fin >> outd;
	fin.close();
	return new ModelDummy{outd};
}

Model* Model::newone_init(int outd)
{
	return new ModelDummy{outd};
}
