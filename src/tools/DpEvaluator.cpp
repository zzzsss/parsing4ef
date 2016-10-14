#include "../ef/DpSentence.h"
#include "DpTools.h"
#include <set>
#include <cstdio>
#include <stdexcept>

double dp_evaluate(string act_file, string pred_file, bool labeled)
{
	set<string> punctSet = set<string>();
	punctSet.insert("''");
	punctSet.insert("``");
	punctSet.insert(".");
	punctSet.insert(":");
	punctSet.insert(",");
	punctSet.insert("PU");	//for CTB

	DPS_PTR gold = read_corpus(act_file);
	DPS_PTR pred = read_corpus(pred_file);
	if(gold->size() != pred->size())
		Logger::Error("Not matching of #sent.");

	int total = 0;
	int total_root = 0;
	int total_non_root = 0;
	int corr = 0;
	int corr_root = 0;
	int corr_non_root = 0;
	int corrL = 0;
	int corrL_root = 0;
	int corrL_non_root = 0;
	int numsent = 0;
	int corrsent = 0;
	int corrsentL = 0;

	int totalNoPunc = 0;
	int totalNoPunc_root = 0;
	int totalNoPunc_non_root = 0;
	int corrNoPunc = 0;
	int corrNoPunc_root = 0;
	int corrNoPunc_non_root = 0;
	int corrLNoPunc = 0;
	int corrLNoPunc_root = 0;
	int corrLNoPunc_non_root = 0;
	int corrsentNoPunc = 0;
	int corrsentLNoPunc = 0;

	for(unsigned int i = 0; i < gold->size(); i++) {
		DP_PTR& goldInstance = gold->at(i);
		DP_PTR& predInstance = pred->at(i);
		int instanceLength = goldInstance->size();
		if(instanceLength != predInstance->size())
			Logger::Error("Not matching of sent length.");

		vector<int>* goldHeads = &goldInstance->heads;
		vector<string>* goldLabels = &goldInstance->rels;
		vector<int>* predHeads = &predInstance->heads;		//because after reading, the predict ones goes there
		vector<string>* predLabels = &predInstance->rels;
		vector<string>* pos = &goldInstance->postags;

		bool whole = true;
		bool wholeL = true;
		bool wholeNP = true;
		bool wholeLNP = true;
		for(int i = 1; i < instanceLength; i++){
			if((*goldHeads)[i] == 0){
				total_root++;
			}
			else{
				total_non_root++;
			}
			if((*predHeads)[i] == (*goldHeads)[i]){
				corr++;
				if((*goldHeads)[i] == 0){
					corr_root++;
				}
				else{
					corr_non_root++;
				}
				if(labeled){
					if((*predLabels)[i] == (*goldLabels)[i]){
						corrL++;
						if((*goldHeads)[i] == 0){
							corrL_root++;
						}
						else{
							corrL_non_root++;
						}
					}
					else{
						wholeL = false;
					}
				}
			}
			else{
				whole = false;
				wholeL = false;
			}

			if(punctSet.count((*pos)[i]) <= 0){
				totalNoPunc++;
				if((*goldHeads)[i] == 0){
					totalNoPunc_root++;
				}
				else{
					totalNoPunc_non_root++;
				}
				if((*predHeads)[i] == (*goldHeads)[i]){
					corrNoPunc++;
					if((*goldHeads)[i] == 0){
						corrNoPunc_root++;
					}
					else{
						corrNoPunc_non_root++;
					}
					if(labeled){
						if((*predLabels)[i] == (*goldLabels)[i]){
							corrLNoPunc++;
							if((*goldHeads)[i] == 0){
								corrLNoPunc_root++;
							}
							else{
								corrLNoPunc_non_root++;
							}
						}
						else{
							wholeLNP = false;
						}
					}
				}
				else{
					wholeNP = false;
					wholeLNP = false;
				}
			}
		}
		total += instanceLength - 1;
		if(whole){
			corrsent++;
		}
		if(wholeL){
			corrsentL++;
		}
		if(wholeNP){
			corrsentNoPunc++;
		}
		if(wholeLNP){
			corrsentLNoPunc++;
		}
		numsent++;
	}

	printf("Tokens: %d\n", total);
	printf("Correct: %d\n", corr);
	printf("Unlabeled Accuracy: %.2lf%%\n", ((double)corr) * 100 / total);
	printf("Unlabeled Complete Correct: %.2lf%%\n", ((double)corrsent) * 100 / numsent);
	if(labeled){
		printf("Labeled Accuracy: %.2lf%%\n", ((double)corrL) * 100 / total);
		printf("Labeled Complete Correct: %.2lf%%\n", ((double)corrsentL) * 100 / numsent);
	}

	printf("\n");

	printf("Tokens Root: %d\n", total_root);
	printf("Correct Root: %d\n", corr_root);
	printf("Unlabeled Accuracy Root: %.2lf%%\n", ((double)corr_root) * 100 / total_root);
	if(labeled){
		printf("Labeled Accuracy Root: %.2lf%%\n", ((double)corrL_root) * 100 / total_root);
	}

	printf("\n");
	/*
	printf("Tokens Non Root: %d\n", total_non_root);
	printf("Correct Non Root: %d\n", corr_non_root);
	printf("Unlabeled Accuracy Non Root: %.2lf%%\n", ((double)corr_non_root) * 100 / total_non_root);
	if(labeled){
		printf("Labeled Accuracy Non Root: %.2lf%%\n", ((double)corrL_non_root) * 100 / total_non_root);
	}
	
	printf("\n");
	*/
	printf("Tokens No Punc: %d\n", totalNoPunc);
	printf("Correct No Punc: %d\n", corrNoPunc);
	printf("Unlabeled Accuracy No Punc: %.2lf%%\n", ((double)corrNoPunc) * 100 / totalNoPunc);
	printf("Unlabeled Complete Correct No Punc: %.2lf%%\n", ((double)corrsentNoPunc) * 100 / numsent);
	if(labeled){
		printf("Labeled Accuracy No Punc: %.2lf%%\n", ((double)corrLNoPunc) * 100 / totalNoPunc);
		printf("Labeled Complete Correct No Punc: %.2lf%%\n", ((double)corrsentLNoPunc) * 100 / numsent);
	}
	/*
	printf("\n");

	printf("Tokens No Punc Root: %d\n", totalNoPunc_root);
	printf("Correct No Punc Root: %d\n", corrNoPunc_root);
	printf("Unlabeled Accuracy No Punc Root: %.2lf%%\n", ((double)corrNoPunc_root) * 100 / totalNoPunc_root);
	if(labeled){
		printf("Labeled Accuracy No Punc Root: %.2lf%%\n", ((double)corrLNoPunc_root) * 100 / totalNoPunc_root);
	}

	printf("\n");

	printf("Tokens No Punc Non Root: %d\n", totalNoPunc_non_root);
	printf("Correct No Punc Non Root: %d\n", corrNoPunc_non_root);
	printf("Unlabeled Accuracy No Punc Non Root: %.2lf%%\n", ((double)corrNoPunc_non_root) * 100 / totalNoPunc_non_root);
	if(labeled){
		printf("Labeled Accuracy No Punc Non Root: %.2lf%%\n", ((double)corrLNoPunc_non_root) * 100 / totalNoPunc_non_root);
	}
	*/
	free_corpus(gold);
	free_corpus(pred);
	return ((double)corr) / total;
}


