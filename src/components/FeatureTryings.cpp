#include <string>
#include <unordered_map>
using namespace std;

unordered_map<string, string> TMP_TRYING_FSS = {
	// firstly efstd
	{"z0", "m-1|mn1-1|mn2-1|mf1-1|mf2-1|mt1-1|mt2-1|mt3-1|h-1|hn1-1|hn2-1|hf1-1|hf2-1|ht1-1|ht2-1|ht3-1|d-m-h|d-mn1-m|d-mf1-m|d-hn1-h|d-hf1-h|l-mn1|l-hn1|l-mf1|l-hf1|"},
	{"z1", "m-5|mn1-3|mn2-3|mf1-3|mf2-3|mt1-3|mt2-1|mt3-1|h-5|hn1-3|hn2-3|hf1-3|hf2-3|ht1-3|ht2-1|ht3-1|d-m-h|d-mn1-m|d-mf1-m|d-hn1-h|d-hf1-h|l-mn1|l-hn1|l-mf1|l-hf1|"},
	{"z2", "m-7|mn1-5|mn2-5|mf1-5|mf2-5|mt1-5|mt2-1|mt3-1|h-7|hn1-5|hn2-5|hf1-5|hf2-5|ht1-5|ht2-1|ht3-1|d-m-h|d-mn1-m|d-mf1-m|d-hn1-h|d-hf1-h|l-mn1|l-hn1|l-mf1|l-hf1|"},
	// how about efeager
	{"y0", "m-1|mn1-1|mn2-1|mf1-1|mf2-1|mt1-1|mt2-1|mt3-1|h-1|hn1-1|hn2-1|hf1-1|hf2-1|ht1-1|ht2-1|ht3-1|d-m-h|d-mn1-m|d-mf1-m|d-hn1-h|d-hf1-h|l-mn1|l-hn1|l-mf1|l-hf1|"
	"|hp1-1|hp1n2-1|hp2-1|d-h-hp1|d-hp1-hp2|l-h|l-hp1|hc2-1|d-hc2-hn1|l-hc2"},
	{"y1", "m-5|mn1-3|mn2-3|mf1-3|mf2-3|mt1-3|mt2-1|mt3-1|h-5|hn1-3|hn2-3|hf1-3|hf2-3|ht1-3|ht2-1|ht3-1|d-m-h|d-mn1-m|d-mf1-m|d-hn1-h|d-hf1-h|l-mn1|l-hn1|l-mf1|l-hf1|"
	"|hp1-3|hp1n2-1|hp2-1|d-h-hp1|d-hp1-hp2|l-h|l-hp1|hc2-1|d-hc2-hn1|l-hc2"},
	{"y2", "m-7|mn1-5|mn2-5|mf1-5|mf2-5|mt1-5|mt2-1|mt3-1|h-7|hn1-5|hn2-5|hf1-5|hf2-5|ht1-5|ht2-1|ht3-1|d-m-h|d-mn1-m|d-mf1-m|d-hn1-h|d-hf1-h|l-mn1|l-hn1|l-mf1|l-hf1|"
	"|hp1-5|hp1n2-1|hp2-1|d-h-hp1|d-hp1-hp2|l-h|l-hp1|hc2-1|d-hc2-hn1|l-hc2"},
	// how about the new features
	{"y3", "m-1|mn1-1|mn2-1|mf1-1|mf2-1|mt1-1|mt2-1|mt3-1|h-1|hn1-1|hn2-1|hf1-1|hf2-1|ht1-1|ht2-1|ht3-1|d-m-h|d-mn1-m|d-mf1-m|d-hn1-h|d-hf1-h|l-mn1|l-hn1|l-mf1|l-hf1|"
	"|hs1-1|hs2-1|hs3-1|hc2-1|hc3-1|d-hc2-hn1|d-h-hs1|l-h|l-hs1|l-hs2|l-hs3"},

	// ok, next explore them (a for efstd, b for efeager)
	// 1. minimal
	{"a1", "m|h|d-m-h"},
	{"b1", "m|h|hp1|d-m-h|d-h-hp1|l-h"},
	// 2. min + tops(or spines-line)
	{"a2", "m|h|mt1|mt2|mt3|mt4|ht1|ht2|ht3|ht4|d-m-h"},
	{"b2", "m|h|hp1|mt1|mt2|mt3|mt4|ht1|ht2|ht3|ht4|d-m-h|d-h-hp1|l-h"},
	{"b2s", "m|h|hp1|mt1|mt2|mt3|mt4|hs1|hs2|hs3|hs4|d-m-h|d-h-hp1|l-h"},
	// 3. min + top + childs
	{"a3nf", "m|h|mt1|mt2|mt3|mt4|ht1|ht2|ht3|ht4|d-m-h|" 
			"mn1|mn2|mf1|mf2|hn1|hn2|hf1|hf2|d-mn1-m|d-mf1-m|d-hn1-h|d-hf1-h|l-mn1|l-mf1|l-hn1|l-hf1"},
	{"a3lr", "m|h|mt1|mt2|mt3|mt4|ht1|ht2|ht3|ht4|d-m-h|"
			"ml1|ml2|mr1|mr2|hl1|hl2|hr1|hr2|d-ml1-m|d-mr1-m|d-hl1-h|d-hr1-h|l-ml1|l-mr1|l-hl1|l-hr1"},
	{"a3nflr", "m|h|mt1|mt2|mt3|mt4|ht1|ht2|ht3|ht4|d-m-h|" 
			"mn1|mn2|mf1|mf2|hn1|hn2|hf1|hf2|d-mn1-m|d-mf1-m|d-hn1-h|d-hf1-h|l-mn1|l-mf1|l-hn1|l-hf1|"
			"ml1|ml2|mr1|mr2|hl1|hl2|hr1|hr2|d-ml1-m|d-mr1-m|d-hl1-h|d-hr1-h|l-ml1|l-mr1|l-hl1|l-hr1|"},
	// 4. min + top + childs-1 + span-end
	{"a4", "m|h|mt1|mt2|mt3|mt4|ht1|ht2|ht3|ht4|d-m-h|"
			"ml1|mr1|hl1|hr1|d-ml1-m|d-mr1-m|d-hl1-h|d-hr1-h|l-ml1|l-mr1|l-hl1|l-hr1|"
			"ma0|mb0|ha0|hb0"},
	// 5. a3lr + span-end
	{"a5", "m|h|mt1|mt2|mt3|mt4|ht1|ht2|ht3|ht4|d-m-h|"
			"ml1|ml2|mr1|mr2|hl1|hl2|hr1|hr2|d-ml1-m|d-mr1-m|d-hl1-h|d-hr1-h|l-ml1|l-mr1|l-hl1|l-hr1|"
			"ma0|mb0|ha0|hb0"},
};
