#include "Model.h"
#include "ModelDummy.h"
#include "ModelDynet.h"
#include "ModelDynet2.h"
#include <fstream>
using namespace std;

// todo: specifying model
ModelZ* ModelZ::read_init(const string& file)
{
#ifdef USE_MODEL_DUMMY
	return new ModelDummy{12};	// just testing
#elif defined USE_MODEL_DYNET
	return ModelDynet::read_init(file);
#elif defined USE_MODEL_DYNET2
  return ModelDynet2::read_init(file);
#else
	return nullptr;
#endif
}

ModelZ* ModelZ::newone_init(const string& mss)
{
#ifdef USE_MODEL_DUMMY
	return new ModelDummy{12};
#elif defined USE_MODEL_DYNET
	return ModelDynet::newone_init(mss);
#elif defined USE_MODEL_DYNET2
  return ModelDynet2::newone_init(mss);
#else
	return nullptr;
#endif
}
