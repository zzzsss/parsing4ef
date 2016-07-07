#ifndef _TOOLS_DPTOOLS
#define _TOOLS_DPTOOLS

#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
using namespace std;
using namespace std::chrono;

// evaluate
extern double dp_evaluate(string act_file, string pred_file, bool labeled = true);

//Recorder: record time (RAII style)
class Recorder{
private:
	string description;
	time_point<steady_clock> start;
public:
	Recorder(string x): description{x}{ 
		start = steady_clock::now();
		cout << "- Start " << description << "." << endl; 
	}
	~Recorder(){
		auto end = steady_clock::now();
		cout << "- End " << description << ", which took " << duration_cast<milliseconds>(end-start).count() << " milliseconds." << endl;
	}
	static void report_time(){
		std::time_t now;
		std::time(&now);
		std::cout << "- Current time: " << ctime(&now) << '\n';
	}
};

#endif
