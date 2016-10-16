#include "ef/DpOptions.h"
#include "ef/EfParser.h"

// The main procedure
// -- no error handling, let it throw and let's see that in gdb
int main(int argc, char** argv)
{
	// simple as it looks ...
	EfParser efp{argc, argv};
	efp.run();
	return 0;
}
