#ifndef _EF_COMP_FEATURE
#define _EF_COMP_FEATURE

#include<vector>
#include<string>
using std::vector;
using std::string;

// We aim to minimize the size of Feature, thus this one is coupled with FeatureManager
class Feature{
private:
	//friend class Feature_hasher;
	//friend bool operator ==(const Feature& x, const Feature& y);
	vector<int> nodes;
	// no-need here: vector<int> distances;
	vector<int> labels;
	string ident{""};
public:
	Feature(vector<int>&& n, vector<int>&& l): nodes(n), labels(l){
		// no need for distance
		for(auto i: nodes)
			ident += i;		// char should be enough for sentence's length 
		for(auto i : labels)
			ident += i;
	}
	const vector<int>& getn(){ return nodes; }
	const vector<int>& getl(){ return labels; }
	const string& get_ident() const { return ident; }
};

/*
inline bool operator ==(const Feature& x, const Feature& y){
	return x.nodes == y.nodes && x.labels == y.labels;
}

class Feature_hasher{
public:
	size_t operator()(const Feature& f) const
	{
		return std::hash<string>{}(f.ident);
	}
};
*/

#endif