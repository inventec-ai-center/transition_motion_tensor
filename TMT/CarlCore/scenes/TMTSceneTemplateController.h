#pragma once

#include "scenes/CarlRLSceneSimChar.h"
#include "anim/CarlKinCharacter.h"
#include "util/MotionTrajectory.h"

class cTMTSceneTemplateController : virtual public cCarlRLSceneSimChar
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	enum eMotionType
	{
		eMotionTypeCanter,
		eMotionTypeJump,
		eMotionTypePace,
		eMotionTypeTrot,
		eMotionTypeGeneric,
		eMotionTypeMax
	};

	enum eControlMode
	{
		ePhaseCtrl,
		eHighLevelCtrl
	};

	cTMTSceneTemplateController();
	virtual ~cTMTSceneTemplateController();

	virtual void ParseArgs(const std::shared_ptr<cArgParser>& parser);
	virtual void Init();

	virtual const eMotionType GetMotionType() const;
	virtual const std::shared_ptr<cCarlKinCharacter>& GetKinChar() const;
	virtual const std::shared_ptr<cCarlKinCharacter>& GetKinChar(const int char_idx) const;
	virtual void EnableRandRotReset(bool enable);
	virtual bool EnabledRandRotReset() const;

	virtual bool SwitchMotion(int agent_id, int target_motion, double target_phase);

	virtual void SyncKinCharRoot();
	virtual void SyncKinCharNewCycle(const cSimCharacter& sim_char, cCarlKinCharacter& out_kin_char) const;
	virtual void SetTargetVelocity(tVector vel);
	virtual tVector GetTargetVelocity() const;
	virtual tVector GetTargetPosition() const;
	virtual Eigen::VectorXd GetKinCharState() const;
	virtual tVector GetKinVcom() const;
	virtual double GetPhase() const;
	virtual double GetLocalPhase() const;
	virtual int GetMotion() const;
	virtual void SetTargetHeading(double heading);
	virtual double GetTargetHeading() const;
	virtual eControlMode GetControlMode() const;
	virtual std::string GetName() const;

	virtual void RecordPhase(int agent_id, double& out_phase) const;
	virtual void RecordGoal(int agent_id, Eigen::VectorXd& out_goal) const;
	virtual void RecordGoalTarget(int agent_id, Eigen::VectorXd& out_goal) const;
	virtual void RecordMotionLabel(int agent_id, int& out_motion_label) const;

	virtual int GetGoalSize(int agent_id) const;
	virtual void BuildGoalOffsetScale(int agent_id, Eigen::VectorXd& out_offset, Eigen::VectorXd& out_scale) const;
	virtual void BuildGoalNormGroups(int agent_id, Eigen::VectorXi& out_groups) const;
	virtual void ResolveCharGroundIntersect();

	virtual int GetNumMotions() const;
	virtual double CalcReward(int agent_id) const;
	virtual void CtrlHeading(int agent_id, double delta);

	virtual double GetLossEnergyConsumption() const;
	virtual double GetLossContactForce() const;
	virtual double GetRewardControl() const;
	virtual double GetRewardSpeed() const;
	virtual double GetRewardHeading() const;
	virtual double GetRewardHeight() const;

	virtual bool CheckContactCycleStart() const;
	virtual std::vector<bool> GetFootContacts() const;
	virtual std::vector<int> GetFootContactJoints() const;
	virtual int GetContactIndex() const;
	virtual double GetSimCharPhase() const;

protected:

	std::vector<std::string> mMotionFiles;
	std::vector<std::string> mMotionNames;
	std::vector<std::string> mTrajectoryFiles;
	std::vector<std::shared_ptr<cCarlKinCharacter>> mKinChars;
	std::vector<cMotionTrajectory> mMotionTrajectories;
	std::vector<double> mGoalVector;
	std::vector<int> mFootContactIndices;
	std::deque<double> mContactCycleStartTimes;

	bool mEnableRandRotReset;
	bool mEnableRandRotation;
	bool mEnableRootRotFail;
	bool mSyncCharRootPos;
	bool mSyncCharRootRot;
	int mUpdateVelocityCount;
	int mUpdateTargetHeadingCount;
	int mUpdateContactCycleStartTimesCount;
	int mKinCharIndex;
	int mPrevContactIndex;
	double mNextKinCharPhase;
	double mNextEndTime;

	double mTargetHeading;
	tVector mTargetVelocity;
	tVector mTargetPosition;
	tVector mStartPosition;

	bool mSyncCharRootInUpdate;
	double mKinCharPhase;
	std::vector<eMotionType> mKinCharMotionTypes;
	eControlMode mCtrlMode;

	virtual bool BuildCharacters();
	virtual void BuildKinChars();
	virtual bool BuildKinCharacter(int id, std::shared_ptr<cCarlKinCharacter>& out_char) const;
	virtual void BuildKinCharsVcomTable(std::string output_file);
	virtual void BuildMotionTrajectories();
	virtual bool BuildController(const cCtrlBuilder::tCtrlParams& ctrl_params, std::shared_ptr<cCharController>& out_ctrl);
	virtual void InitFootContacts();

	virtual void UpdateTimers(double timestep);
	virtual void UpdateCharacters(double timestep);
	virtual void UpdateKinChar(double timestep);
	virtual void UpdateTargetVelocity(double timestep);
	virtual void UpdateNextRandKinChar(bool rand_phase);

	virtual double GetKinTime() const;
	virtual void SyncCharacters();
	virtual bool EnableSyncChar() const;
	virtual void ResetCharacters();
	virtual void ResetKinChar();
	virtual void ResetKinChar(int motion_idx, double phase);

	virtual bool HasFallen(const cSimCharacter& sim_char) const;
	virtual bool CheckRootRotFail(const cSimCharacter& sim_char, tVector target_velocity) const;

	virtual double CalcRewardVelocity(const cSimCharacter& sim_char, const tVector target_velocity) const;
	virtual double CalcRewardEnergyConsumption(const cSimCharacter& sim_char) const;
	virtual double CalcRewardContactForce(const cSimCharacter& sim_char) const;
	virtual double CalcRewardSpeed(const cSimCharacter& sim_char, const tVector target_velocity) const;
	virtual double CalcRewardHeading(const cSimCharacter& sim_char, const tVector target_velocity) const;
	virtual double CalcRewardHeight(const cSimCharacter& sim_char, const tVector target_position) const;
};
