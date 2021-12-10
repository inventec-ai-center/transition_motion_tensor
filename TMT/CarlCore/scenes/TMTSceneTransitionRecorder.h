#pragma once

#include "scenes/TMTSceneTemplateController.h"
#include "util/tqdm.h"
#include <map>
#include <tuple>

class cTMTSceneTransitionRecorder : virtual public cTMTSceneTemplateController
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	cTMTSceneTransitionRecorder();
	virtual ~cTMTSceneTransitionRecorder();

	virtual void ParseArgs(const std::shared_ptr<cArgParser>& parser);
	virtual void Init();
	virtual std::string GetName() const;

	virtual bool CheckValidEpisode() const;
	virtual bool NeedNewAction(int agent_id) const;
	virtual void SetAction(int agent_id, const Eigen::VectorXd& action);

protected:

	typedef struct
	{
		int srcMotion;
		int dstMotion;
		double srcPhase;
		double dstPhase;
		double transPhase;
		tVector srcVelocity;
		tVector dstVelocity;
		int transition_index;
		double transition_phase;
		bool isFallen;
		bool isValid;
		std::vector<double> losses_energy;
		std::vector<double> speed_rewards;
		std::vector<double> heading_rewards;
		std::vector<double> height_rewards;
		std::vector<tVector> positions;
		std::vector<std::vector<bool>> foot_contacts;
	} TransitionSample;

	int mExpIndex;
	int mUpdateCounter;
	int mNumCollectTransitionSamples;
	int mUpdateRecordCharStateCount;
	double mPreTransitionTime;
	double mPostTransitionTime;
	bool mHasSwitched;
	bool mHasVelocityExploded;
	bool mHasFallen;
	bool mNeedUpdateAction;
	std::string mOutputFilename;
	std::vector<TransitionSample> mTransitionSamples;
	tqdm mProgressBar;

	virtual void InitTransitionSample(int exp_id);
	virtual void BuildCollectTransitionTrajectories();
	virtual void GenerateRandTransitionSample(TransitionSample &trans_sample) const;
	virtual void WriteCollectTransitionTrajectories(std::string output_filename, int start_idx, int end_idx);
	virtual bool CheckTransitionSuccess() const;
	virtual bool CheckCharacterFallen(const std::shared_ptr<cSimCharacter> sim_char) const;

	virtual void UpdateCharacters(double timestep);
	virtual void PostUpdate(double timestep);
	virtual void ResetCharacters();
	virtual void OnEvaluationEnd();
	virtual void SyncPose(int kin_index, double motion_time);
};
