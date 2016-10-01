#ifndef _EF_COMP_FEATURE
#define _EF_COMP_FEATURE

// We aim to minimize the size of Feature.
class Feature{
protected:
	vector<int> nodes;
	vector<int> distances;
	vector<int> labels;
public:
	Feature(vector<int>&& n, vector<int>&& d, vector<int>&& l): nodes(n), distances(d), labels(l){}
	Feature(Feature&&) = default;
	Feature(const Feature&) = default;
	~Feature() = default;
};

#endif