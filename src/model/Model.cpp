#include "Model.h"
#include "ModelDummy.h"
#include "ModelDynet.h"
#include <fstream>
using namespace std;

// todo: specifying model
ModelZ* ModelZ::read_init(const string& file)
{
#ifdef USE_MODEL_DYNET
	return ModelDynet::read_init(file);
#endif // USE_MODEL_DYNET
}

ModelZ* ModelZ::newone_init(const string& mss)
{
#ifdef USE_MODEL_DYNET
	return ModelDynet::newone_init(mss);
#endif // USE_MODEL_DYNET
}
