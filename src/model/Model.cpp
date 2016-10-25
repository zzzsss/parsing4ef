#include "Model.h"
#include "ModelDummy.h"
#include "ModelDynet.h"
#include <fstream>
using namespace std;

// todo: specifying model
ModelZ* ModelZ::read_init(const string& file)
{
#ifdef USE_MODEL_DUMMY
	return new ModelDummy{12};	// just testing
#elif defined USE_MODEL_DYNET
	return ModelDynet::read_init(file);
#else
	return nullptr;
#endif // USE_MODEL_DYNET
}

ModelZ* ModelZ::newone_init(const string& mss)
{
#ifdef USE_MODEL_DUMMY
	return new ModelDummy{12};
#elif defined USE_MODEL_DYNET
	return ModelDynet::newone_init(mss);
#else
	return nullptr;
#endif // USE_MODEL_DYNET
}
