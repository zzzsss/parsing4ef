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

// Detailed Recording (for both recorders)
//#define _VERBOSE_ACCRECORDER

// evaluate
extern double dp_evaluate(string act_file, string pred_file, bool labeled = true);

// Logger
class Logger{
public:
	enum LOG_LEVEL{};
	static ostream& get_output(int l = 0){ return cout; }
	static void Error(const string &x){
		cerr << "Fatal error: " << x << endl;
		throw runtime_error(x);
	}
	static void Warn(const string &x){
		cerr << "Warning: " << x << endl;
	}
};

// helpers
extern vector<string> dp_split(const string &s, char x, int cut_time=-1);	// -1 means always

template<class T>
T dp_str2num(const string& x){
	stringstream tmp_str(x);
	T y = 0;
	tmp_str >> y;
	if(y == 0 && x[0] != '0')
		Logger::Error("Int-Error: transfer to num.");
	return y;
}

template<class T>
string dp_num2str(const T& x){
	stringstream tmp_str;
	tmp_str << x;
	string ss;
	tmp_str >> ss;
	return ss;
}

//Recorder (Temp): record time (RAII style)
class Recorder{
private:
	string description;
	time_point<steady_clock> start;
public:
	Recorder(string x): description{x}{
#ifdef _VERBOSE_ACCRECORDER
		start = steady_clock::now();
		Logger::get_output() << "- Start " << description << "." << endl;
#endif
	}
	~Recorder(){
#ifdef _VERBOSE_ACCRECORDER
		auto end = steady_clock::now();
		Logger::get_output() << "- End " << description << ", which took " << duration_cast<milliseconds>(end-start).count() << " milliseconds." << endl;
#endif
	}
	static void report_time(const string& s){
		std::time_t now;
		std::time(&now);
		Logger::get_output() << "- For " << s << " at";
#ifdef DP_USING_WINDOWS
		char TMP_buf[64];
		ctime_s(TMP_buf, sizeof(TMP_buf), &now);
		Logger::get_output() << " current time: " << TMP_buf << std::flush;
#else
		Logger::get_output() << " current time: " << ctime(&now) << std::flush;
#endif
	}
};

//Accumulated Recorder: record time for several lines
class AccRecorder{
private:	// recordings must be in all these maps
	using TIME_GRAIN = microseconds;
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
			if(started[one])
				Logger::Error(string("AccRecorder have not ended ") + one + ".");
			// restart
			started[one] = true;
			start_points[one] = steady_clock::now();
		}
	}
	void end(const string& one){
		auto z = started.find(one);
		if(z == started.end() || !z->second)
			Logger::Error(string("AccRecorder have not started ")+one+".");
		else{
			// accumulate
			auto cur = steady_clock::now();
			started[one] = false;
			times[one] += duration_cast<TIME_GRAIN>(cur - start_points[one]).count();
		}
	}
	void report(const string& one){
		auto z = times.find(one);
		if(z == times.end())
			Logger::Error(string("AccRecorder have not started ") + one + ".");
		else
			Logger::get_output() << "Record" << description << ": Acc time for " << one << ":" << z->second << "." << endl;
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
#define ACCRECORDER_REPORT() global_recorder.report_all()
#else
#define ACCRECORDER_ONCE(x)
#define ACCRECORDER_RESET(x)
#define ACCRECORDER_REPORT()
#endif

template<class T>
void CHECK_EQUAL(const T& a, const T& b)
{
	if(a != b)
		Logger::Error("Equality check failed.");
}

#endif
