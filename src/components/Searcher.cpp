#include "Searcher.h"
#include "State.h"
#include "Agenda.h"
#include "Scorer.h"
#include "../tools/DpTools.h"

/*
* ef_search: the searching process itself **
* --> <what effect> train: only back-propagate error; test: store best to one
* --> <managed by> State*: Agenda, StateTemp: local(stack), Feature*: FeatureManager, Score*: Scorer
* ----> (Those * need pointers mainly because we have to store them for future use.)
*/
/*
What is the process for this search?
- Agenda.init(start)
- while(not stopping):
1. collect all the StateTemp for the States in the beam (create StateTemp and Feature)
2. score all the Feature with the scorer (possibly from Scorer's cache), obtain Scores
3. the agenda rank all the StateTemp and make several decisions, return empty beam if ending
- possibly assign and clean up
*/
/*
Who needs to know what options?
1. testing or training: Agenda
2. std or eager/ features: State (using FeatureManager)
4. beam and recombinition: Agenda
5. gold-drop and criteria: Agenda
*/
void Searcher::ef_search(DP_PTR one)
{
	// helpers
	ACCRECORDER_ONCE("Search-all");
	Scorer the_scorer{model};
	FeatureManager& the_featurer = *fm;
	Agenda the_agenda{is_training, options};
	// the empty one for the start
	State* start = State::make_empty(one, options->ef_mode);
	vector<State*> beam = the_agenda.init(start);
	// start the searching
	num_sent++;
	num_token += one->size() - 1;
	while(!beam.empty()){
		num_steps++;
		// expand -- first collect all candidates 
		// -- (for simplicity, reconstruct Features)
		vector<StateTemp> candidates{};
		{
			ACCRECORDER_ONCE("Search-expand");
			for(State* s : beam){
				vector<StateTemp> ones = s->expand();
				candidates.insert(candidates.end(), ones.begin(), ones.end());
			}
		}
		// construct the features
		{
			ACCRECORDER_ONCE("Search-feature");
			the_featurer.feature_them(candidates);
		}
		// score the Features withe Scorer 
		// -- rely on the cache of Scorer for avoiding dup-calculations
		{
			ACCRECORDER_ONCE("Search-score");
			the_scorer.score_them(candidates, the_featurer);
		}
		// get the new beam for the next round
		{
			ACCRECORDER_ONCE("Search-rank");
			beam = the_agenda.rank_them(candidates, the_scorer);
		}
	}
	// assign and clear
	if(!is_training){
		State* end = the_agenda.get_best();
		end->assignto(one);
	}
	// possibly clean up
	{
		ACCRECORDER_ONCE("Search-clean");
		the_featurer.clear();
	}
	return;
}

void Searcher::report_and_reset_all()
{
	// static methods, for reporting stat
	Scorer::report_and_reset();
	Agenda::report_and_reset();
}