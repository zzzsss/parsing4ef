#ifndef _TOOLS_DPTOOLS
#define _TOOLS_DPTOOLS

#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <iomanip>
using namespace std;
using namespace std::chrono;
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <sstream>

// OS
#ifdef _WIN32
#define DP_USING_WINDOWS
#else
#define DP_USING_LINUX	//currently assume that
#endif

// Detailed Recording
#define _VERBOSE_ACCRECORDER

// evaluate
extern double dp_evaluate(string act_file, string pred_file, bool labeled = true);

// helpers
extern vector<string> dp_split(const string &s, char x);
extern int dp_str2int(const string& x);

//Recorder (Temp): record time (RAII style)
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
#ifdef DP_USING_WINDOWS
		char TMP_buf[64];
		ctime_s(TMP_buf, sizeof(TMP_buf), &now);
		cout << "- Current time: " << TMP_buf << '\n';
#else
		cout << "- Current time: " << ctime(&now) << '\n';
#endif
	}
};

//Accumulated Recorder: record time for several lines
class AccRecorder{
private:	// recordings must be in all these maps
	string description;
	unordered_map<string, bool> started;
	unordered_map<string, time_point<steady_clock>> start_points;
	unordered_map<string, long long> times;
public:
	void reset(const string& des){
		description = des;
		started.clear();
		start_points.clear();
		times.clear();
	}
	void start(const string& one){
		auto z = started.find(one);
		if(z == started.end()){
			// add one for recording
			started.insert({one, true});
			start_points.insert({one, steady_clock::now()});
			times.insert({one, 0});
		}
		else{
			// restart
			start_points.insert({one, steady_clock::now()});
		}
	}
	void end(const string& one){
		auto z = started.find(one);
		if(z == started.end() || !z->second)
			throw runtime_error(string("AccRecorder have not started ")+one+".");
		else{
			// accumulate
			auto cur = steady_clock::now();
			started.insert({one, false});
			times[one] += duration_cast<milliseconds>(cur - start_points[one]).count();
		}
	}
	void report(const string& one){
		auto z = times.find(one);
		if(z == times.end())
			throw runtime_error(string("AccRecorder have not started ") + one + ".");
		else
			cout << "Record" << description << ": Acc time for " << one << ":" << z->second << " milliseconds." << endl;
	}
	void report_all(){
		for(auto& x : started)
			report(x.first);
	}
};

class AccRecorderWarpper{
	// RAII style wrapper for AccRecorder
	AccRecorder* rec;
	string des;
public:
	AccRecorderWarpper(AccRecorder* r, string d): rec(r), des(d){ rec->start(d); }
	~AccRecorderWarpper(){ rec->end(des); }
};

#ifdef _VERBOSE_ACCRECORDER 
extern AccRecorder global_recorder; 
#define ACCRECORDER_ONCE(x) AccRecorderWarpper TMP_rec_all{&global_recorder, x}
#define ACCRECORDER_RESET(x) global_recorder.reset(x)
#else
#define ACCRECORDER_ONCE(x)
#define ACCRECORDER_RESET(x)
#endif

#endif
