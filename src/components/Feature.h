#ifndef _EF_COMP_FEATURE
#define _EF_COMP_FEATURE

// We aim to minimize the size of Feature.
class Feature{
private:
	friend class FeatureManager;
	friend class Feature_hasher;
	vector<int> nodes;
	vector<int> distances;
	vector<int> labels;
	string ident{""};
	Feature(vector<int>&& n, vector<int>&& d, vector<int>&& l): nodes(n), distances(d), labels(l){
		// no need for distance
		for(auto i: nodes)
			ident += i;		// char should be enough for sentence's length 
		for(auto i : labels)
			ident += i;
	}
	bool operator ==(const Feature& x){
		return x.nodes == nodes && x.distances == distances && x.labels == labels;
	}
};

struct Feature_hasher{
	size_t operator()(const Feature& f)
	{
		return std::hash<string>{}(f.ident);
	}
};

#endif