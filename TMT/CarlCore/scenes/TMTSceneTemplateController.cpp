#include "TMTSceneTemplateController.h"
#include <math.h>
#include "sim/RBDUtil.h"
#include "sim/CtController.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include "sim/CharController.h"
#include "util/MathUtilExtend.h"
#include "anim/CarlKinTree.h"

const double gControlTimestep = 1.0 / 30.0;
const int gMaxContactCycleBufferSize = 3;


double cTMTSceneTemplateController::CalcRewardVelocity(const cSimCharacter& sim_char, const tVector target_velocity) const
{
	tVector current_position = sim_char.CalcCOM();
	tVector current_velocity = sim_char.CalcCOMVel();

	double speed_reward, rotation_reward, position_reward, height_reward;
	speed_reward = rotation_reward = position_reward = height_reward = 0;

	// Speed Reward
	double speed_error = pow(current_velocity.norm() - target_velocity.norm(), 2);
	speed_reward = exp(-0.8 * speed_error);

	// Rotation Reward
	tVector v0 = tVector(current_velocity[0], 0, current_velocity[2], 0).normalized();
	tVector v1 = tVector(target_velocity[0], 0, target_velocity[2], 0).normalized();
	double rotation_error = v0.dot(v1);
	rotation_reward = exp(8 * (rotation_error - 1));

	// Position Reward
	double position_error = pow(current_position[2], 2);
	// if (mKinCharMotionTypes[mKinCharIndex] == eMotionTypeJump)
	// {
	// 	double height_error = abs(current_position[1] - mTargetPosition[1]);
	// 	position_reward = 0.5 * exp(-1 * position_error) + 0.5 * exp(-5 * height_error);
	// }
	// else
	// {
		position_reward = exp(-1 * position_error);
	// }

	// Height Reward
	double height_error = pow(target_velocity[1] - current_velocity[1], 2);
	height_reward = exp(-3 * height_error);

	// double reward = 0.3 * speed_reward + 0.3 * position_reward + 0.4 * height_reward;
	// double reward = 0.5 * speed_reward + 0.5 * rotation_reward;
	double reward = 0.4 * speed_reward + 0.4 * position_reward + 0.2 * height_reward;
	// double reward = speed_reward * position_reward * height_reward;

	// printf("speed_reward=%f, position_reward=%f, height_reward=%f\n", speed_reward, position_reward, height_reward);
	return reward;
}

double cTMTSceneTemplateController::CalcRewardEnergyConsumption(const cSimCharacter& sim_char) const
{
	const cDeepMimicCharController *sim_ctrl = dynamic_cast<cDeepMimicCharController*>(sim_char.GetController().get());
	Eigen::VectorXd tau = sim_ctrl->GetTau();
	int num_joints = sim_char.GetNumJoints();

	double total_energy_cost = 0;
	for (int i = 0; i < num_joints; i++)
	{
		Eigen::VectorXd tau = sim_char.GetJoint(i).GetTau();
		total_energy_cost += tau.norm();
	}

	return total_energy_cost;
}

double cTMTSceneTemplateController::CalcRewardContactForce(const cSimCharacter& sim_char) const
{
	double sum_contact_force = 0;

	for (int i = 0; i < mFootContactIndices.size(); ++i)
	{
		tEigenArr<cContactManager::tContactPt> contact_pts = sim_char.GetContactPts(mFootContactIndices[i]);
		if (contact_pts.size() > 0)
		{
			double avg_joint_contact_force = 0;
			for (int j = 0; j < contact_pts.size(); ++j)
			{
				avg_joint_contact_force += contact_pts[j].mForce.norm();
			}
			avg_joint_contact_force /= contact_pts.size();
			sum_contact_force += avg_joint_contact_force;
		}
	}

	return sum_contact_force;
}

double cTMTSceneTemplateController::CalcRewardSpeed(const cSimCharacter& sim_char, const tVector target_velocity) const
{
	tVector current_velocity = sim_char.CalcCOMVel();

	double speed_error = pow(current_velocity.norm() - target_velocity.norm(), 2);
	double speed_reward = exp(-0.8 * speed_error);

	return speed_reward;
}

double cTMTSceneTemplateController::CalcRewardHeading(const cSimCharacter& sim_char, const tVector target_velocity) const
{
	tVector current_velocity = sim_char.CalcCOMVel();

	tVector v0 = tVector(current_velocity[0], 0, current_velocity[2], 0).normalized();
	tVector v1 = tVector(target_velocity[0], 0, target_velocity[2], 0).normalized();
	double heading_error = v0.dot(v1);
	double heading_reward = exp(8 * (heading_error - 1));

	return heading_reward;
}

double cTMTSceneTemplateController::CalcRewardHeight(const cSimCharacter& sim_char, const tVector target_position) const
{
	tVector current_position = sim_char.CalcCOM();

	double height_error = abs(current_position[1] - target_position[1]);
	double height_reward = exp(-5 * height_error);

	return height_reward;
}

cTMTSceneTemplateController::cTMTSceneTemplateController()
{
	mEnableRandRotReset = false;
	mEnableRootRotFail = false;
	mEnableRandRotation = false;
	mSyncCharRootRot = false;
	mSyncCharRootPos = false;
	mSyncCharRootInUpdate = false;
	mUpdateVelocityCount = 0;
	mUpdateTargetHeadingCount = 0;
	mUpdateContactCycleStartTimesCount = 0;
	mKinCharIndex = 0;
	mKinCharPhase = 0;
	mTargetHeading = 0;
	mPrevContactIndex = -1;
	mTargetPosition = tVector::Zero();
	mStartPosition = tVector::Zero();
	mMotionNames.clear();
	mMotionFiles.clear();
}

cTMTSceneTemplateController::~cTMTSceneTemplateController()
{
}

void cTMTSceneTemplateController::ParseArgs(const std::shared_ptr<cArgParser>& parser)
{
	cCarlRLSceneSimChar::ParseArgs(parser);

	parser->ParseStrings("motion_files", mMotionFiles);
	parser->ParseBool("enable_root_rot_fail", mEnableRootRotFail);
	parser->ParseBool("enable_rand_rot_reset", mEnableRandRotReset);
	parser->ParseBool("sync_char_root_pos", mSyncCharRootPos);
	parser->ParseBool("sync_char_root_rot", mSyncCharRootRot);
	parser->ParseStrings("motion_names", mMotionNames);
	parser->ParseStrings("trajectory_files", mTrajectoryFiles);

	std::string ctrl_mode = "";
	parser->ParseString("control_mode", ctrl_mode);
	if (ctrl_mode == "phase") {
		mCtrlMode = eControlMode::ePhaseCtrl;
	}
	else if (ctrl_mode == "high_level") {
		mCtrlMode = eControlMode::eHighLevelCtrl;
	}
	else {
		printf("Unrecognized control mode \"%s\"\n", ctrl_mode.c_str());
		exit(1);
	}

	if (mMotionNames.size() != mMotionFiles.size())
	{
		printf("[ERROR] mMotionNames.size() != mMotionFiles.size()\n");
		exit(1);
	}
}

void cTMTSceneTemplateController::Init()
{
	mKinChars.clear();
	mKinChars.resize(mMotionFiles.size());
	for (int i = 0; i < mKinChars.size(); ++i)
	{
		mKinChars[i].reset();
	}
	BuildKinChars();
	cCarlRLSceneSimChar::Init();
	BuildMotionTrajectories();
	InitFootContacts();

	UpdateTargetVelocity(0);
	mStartPosition = mTargetPosition = GetCharacter()->CalcCOM();
}

const cTMTSceneTemplateController::eMotionType cTMTSceneTemplateController::GetMotionType() const
{
	return mKinCharMotionTypes[mKinCharIndex];
}

const std::shared_ptr<cCarlKinCharacter>& cTMTSceneTemplateController::GetKinChar() const
{
	return GetKinChar(mKinCharIndex);
}

const std::shared_ptr<cCarlKinCharacter>& cTMTSceneTemplateController::GetKinChar(const int char_idx) const
{
	assert(char_idx >= 0 && char_idx < mKinChars.size());
	return mKinChars[char_idx];
}

void cTMTSceneTemplateController::EnableRandRotReset(bool enable)
{
	mEnableRandRotReset = enable;
}

bool cTMTSceneTemplateController::EnabledRandRotReset() const
{
	bool enable = mEnableRandRotReset;
	return enable;
}

void cTMTSceneTemplateController::SetTargetVelocity(tVector vel)
{
	mTargetVelocity = vel;
}

tVector cTMTSceneTemplateController::GetTargetVelocity() const
{
	return mTargetVelocity;
}

tVector cTMTSceneTemplateController::GetTargetPosition() const
{
	return mTargetPosition;
}

Eigen::VectorXd cTMTSceneTemplateController::GetKinCharState() const
{
	const auto& kin_char = GetKinChar();
	const Eigen::VectorXd& pose = kin_char->GetPose();
	const Eigen::VectorXd& vel = kin_char->GetVel();

	Eigen::VectorXd kin_state(pose.size() + vel.size());
	kin_state << pose, vel;
	return kin_state;
}

tVector cTMTSceneTemplateController::GetKinVcom() const
{
	tVector vel = mMotionTrajectories[mKinCharIndex].Sample(mKinCharPhase);
	return vel;
}

double cTMTSceneTemplateController::CalcReward(int agent_id) const
{
	const cSimCharacter* sim_char = GetAgentChar(agent_id);
	bool fallen = HasFallen(*sim_char);

	double r = 0;
	int max_id = 0;
	if (!fallen)
	{
		r = CalcRewardVelocity(*sim_char, mTargetVelocity);
		CalcRewardEnergyConsumption(*sim_char);
	}

	return r;
}

double cTMTSceneTemplateController::GetPhase() const
{
	return mKinCharPhase;
}

double cTMTSceneTemplateController::GetLocalPhase() const
{
	return mMotionTrajectories[mKinCharIndex].CalcLocalPhase(mKinCharPhase);
}

int cTMTSceneTemplateController::GetMotion() const
{
	return mKinCharIndex;
}

void cTMTSceneTemplateController::RecordPhase(int agent_id, double& out_phase) const
{
	out_phase = GetSimCharPhase();
}

void cTMTSceneTemplateController::RecordGoal(int agent_id, Eigen::VectorXd& out_goal) const
{
	int goal_size = GetGoalSize(agent_id);
	out_goal = Eigen::VectorXd(goal_size);
	out_goal.setZero();

	if (mCtrlMode == eControlMode::eHighLevelCtrl)
	{
		for (int i = 0; i < goal_size; ++i)
		{
			out_goal(i) = mGoalVector[i];
		}
	}
}

void cTMTSceneTemplateController::RecordGoalTarget(int agent_id, Eigen::VectorXd& out_goal) const
{
	out_goal = Eigen::VectorXd(1);
	out_goal.setZero();
}

void cTMTSceneTemplateController::RecordMotionLabel(int agent_id, int& out_motion_label) const
{
	out_motion_label = mKinCharIndex;
}

int cTMTSceneTemplateController::GetGoalSize(int agent_id) const
{
	if (mCtrlMode == eControlMode::eHighLevelCtrl) {
		return 4;
	}
	else {
		return 0;
	}
}

void cTMTSceneTemplateController::BuildGoalOffsetScale(int agent_id, Eigen::VectorXd& out_offset, Eigen::VectorXd& out_scale) const
{
	int goal_size = GetGoalSize(agent_id);
	out_offset = Eigen::VectorXd::Zero(goal_size);
	out_scale = Eigen::VectorXd::Ones(goal_size);
}

void cTMTSceneTemplateController::BuildGoalNormGroups(int agent_id, Eigen::VectorXi& out_groups) const
{
	int goal_size = GetGoalSize(agent_id);
	out_groups = cCharController::gNormGroupSingle * Eigen::VectorXi::Ones(goal_size);
}

void cTMTSceneTemplateController::ResolveCharGroundIntersect()
{
	cSceneSimChar::ResolveCharGroundIntersect();
	mStartPosition = mTargetPosition = GetCharacter()->CalcCOM();
}

std::string cTMTSceneTemplateController::GetName() const
{
	return "template_controller";
}

void cTMTSceneTemplateController::CtrlHeading(int agent_id, double delta)
{
	double target_heading = cMathUtilExtend::ClampEuler(GetTargetHeading() + delta);
	SetTargetHeading(target_heading);
	UpdateTargetVelocity(0);
}

bool cTMTSceneTemplateController::SwitchMotion(int agent_id, int target_motion, double target_phase)
{
	// printf("[SwitchMotion] target_motion=%d, target_phase=%f\n", target_motion, target_phase);
	mKinCharIndex = target_motion;
	mKinCharPhase = target_phase;

	if (mCtrlMode == eControlMode::eHighLevelCtrl)
	{
		// Use the current heading as the target for pace-turn motion (Disable this while collecting transitions)
		tVector v_com_norm = GetCharacter()->CalcCOMVel().normalized();
		double theta_curr = atan2(-v_com_norm[2], v_com_norm[0]);
		mTargetHeading = theta_curr;

		// Force update the goal vector after switching motion
		UpdateTargetVelocity(0);
	}
	else if (mCtrlMode == eControlMode::ePhaseCtrl)
	{
		ResetKinChar(mKinCharIndex, mKinCharPhase);
		SyncKinCharRoot();

		const std::shared_ptr<cSimCharacter> sim_char = GetCharacter();
		const std::shared_ptr<cCarlKinCharacter> kin_char = GetKinChar();

		const auto& ctrl = sim_char->GetController();
		auto ct_ctrl = dynamic_cast<cCtController*>(ctrl.get());
		if (ct_ctrl != nullptr)
		{
			double kin_time = GetKinTime();
			ct_ctrl->SetInitTime(kin_time);

			double cylcle_dur = kin_char->GetMotionDuration();
			ct_ctrl->SetCyclePeriod(cylcle_dur);
		}
	}

	mContactCycleStartTimes.clear();
	return true;
}

int cTMTSceneTemplateController::GetNumMotions() const
{
	return mTrajectoryFiles.size();
}

void cTMTSceneTemplateController::SetTargetHeading(double heading)
{
	mTargetHeading = heading;
}

double cTMTSceneTemplateController::GetTargetHeading() const
{
	return mTargetHeading;
}

cTMTSceneTemplateController::eControlMode cTMTSceneTemplateController::GetControlMode() const
{
	return mCtrlMode;
}

double cTMTSceneTemplateController::GetLossEnergyConsumption() const
{
	return CalcRewardEnergyConsumption(*GetCharacter());
}

double cTMTSceneTemplateController::GetLossContactForce() const
{
	return CalcRewardContactForce(*GetCharacter());
}

double cTMTSceneTemplateController::GetRewardControl() const
{
	return CalcRewardVelocity(*GetCharacter(), mTargetVelocity);
}

double cTMTSceneTemplateController::GetRewardSpeed() const
{
	return CalcRewardSpeed(*GetCharacter(), mTargetVelocity);
}

double cTMTSceneTemplateController::GetRewardHeading() const
{
	return CalcRewardHeading(*GetCharacter(), mTargetVelocity);
}

double cTMTSceneTemplateController::GetRewardHeight() const
{
	return CalcRewardHeight(*GetCharacter(), mTargetPosition);
}

bool cTMTSceneTemplateController::BuildCharacters()
{
	bool succ = cCarlRLSceneSimChar::BuildCharacters();
	if (EnableSyncChar())
	{
		SyncCharacters();
	}
	return succ;
}

void cTMTSceneTemplateController::BuildMotionTrajectories()
{
	mMotionTrajectories.resize(mTrajectoryFiles.size());
	for (int i = 0; i < mTrajectoryFiles.size(); ++i)
	{
		bool succ = mMotionTrajectories[i].Load(mTrajectoryFiles[i]);
		if (!succ)
		{
			printf("Open trajectory file %s failed. Trying to generate it\n", mTrajectoryFiles[i].c_str());
			BuildKinCharsVcomTable(mTrajectoryFiles[i]);
			mMotionTrajectories[i].Load(mTrajectoryFiles[i]);
		}
		printf("Successfully open trajectory file %s (%.6f seconds)\n", mTrajectoryFiles[i].c_str(), mMotionTrajectories[i].GetDuration());
	}
}

bool cTMTSceneTemplateController::BuildController(const cCtrlBuilder::tCtrlParams& ctrl_params, std::shared_ptr<cCharController>& out_ctrl)
{
	bool succ = cSceneSimChar::BuildController(ctrl_params, out_ctrl);
	if (succ)
	{
		auto ct_ctrl = dynamic_cast<cCtController*>(out_ctrl.get());
		if (ct_ctrl != nullptr)
		{
			const auto& kin_char = GetKinChar();
			double cycle_dur = kin_char->GetMotionDuration();
			ct_ctrl->SetCyclePeriod(cycle_dur);
		}
	}
	return succ;
}

void cTMTSceneTemplateController::InitFootContacts()
{
	mFootContactIndices.clear();

	const std::shared_ptr<cCarlKinCharacter> kin_char = GetKinChar();
	const Eigen::MatrixXd& joint_mat = kin_char->GetJointMat();
	for (int i = 0; i < kin_char->GetNumJoints(); ++i)
	{
		if (cCarlKinTree::IsFootContact(joint_mat, i))
		{
			mFootContactIndices.push_back(i);
		}
	}
}

void cTMTSceneTemplateController::BuildKinChars()
{
	for (int i = 0; i < mKinChars.size(); ++i)
	{
		bool succ = BuildKinCharacter(i, mKinChars[i]);
		if (!succ)
		{
			printf("Failed to build kin character\n");
			assert(false);
		}
		printf("mKinChars[%d]: %f secs\n", i, mKinChars[i]->GetMotionDuration());
	}

	mKinCharMotionTypes.resize(mKinChars.size());
	for (int i = 0; i < mKinChars.size(); ++i)
	{
		if (mMotionNames[i] == "Canter" || mMotionNames[i] == "canter")
		{
			mKinCharMotionTypes[i] = eMotionType::eMotionTypeCanter;
		}
		else if (mMotionNames[i] == "Jump" || mMotionNames[i] == "jump")
		{
			mKinCharMotionTypes[i] = eMotionType::eMotionTypeJump;
		}
		else if (mMotionNames[i] == "Pace" || mMotionNames[i] == "pace")
		{
			mKinCharMotionTypes[i] = eMotionType::eMotionTypePace;
		}
		else if (mMotionNames[i] == "Trot" || mMotionNames[i] == "trot")
		{
			mKinCharMotionTypes[i] = eMotionType::eMotionTypeTrot;
		}
		else
		{
			mKinCharMotionTypes[i] = eMotionType::eMotionTypeGeneric;
			// printf("Invalid Motion Type Found\n");
			// assert(false);
		}
	}
}

bool cTMTSceneTemplateController::BuildKinCharacter(int id, std::shared_ptr<cCarlKinCharacter>& out_char) const
{
	auto kin_char = std::shared_ptr<cCarlKinCharacter>(new cCarlKinCharacter());
	const cSimCharacter::tParams& sim_char_params = mCharParams[0];
	cCarlKinCharacter::tParams kin_char_params;

	kin_char_params.mID = id;
	kin_char_params.mCharFile = sim_char_params.mCharFile;
	kin_char_params.mOrigin = sim_char_params.mInitPos;
	kin_char_params.mLoadDrawShapes = false;
	kin_char_params.mMotionFile = mMotionFiles[id];

	bool succ = kin_char->Init(kin_char_params);
	if (succ)
	{
		out_char = kin_char;
	}
	return succ;
}

void cTMTSceneTemplateController::BuildKinCharsVcomTable(std::string output_file)
{
	const double velocity_timestep = 1.0 / 600;

	const auto& sim_char = GetCharacter();
	const Eigen::MatrixXd& joint_mat = sim_char->GetJointMat();
	const Eigen::MatrixXd& body_def_mat = sim_char->GetBodyDefs();

	for (int i = 0; i < mKinChars.size(); ++i)
	{
		const auto& kin_char = GetKinChar(i);

		double motion_duration = mKinChars[i]->GetMotionDuration();
		int num_samples = floor(motion_duration / velocity_timestep);

		kin_char->Reset();
		kin_char->SetOriginRot(tQuaternion::Identity());

		printf("Motion File: %s\n", mMotionFiles[i].c_str());

		std::ofstream fout(output_file);
		if (fout.is_open())
		{
			fout << "-1\n"; // skip first line
			for (int j = 0; j < num_samples; ++j)
			{
				double sample_time = j * velocity_timestep;
				kin_char->SetTime(sample_time);
				kin_char->Pose(sample_time);

				tVector com_world, com_vel_world;
				cRBDUtil::CalcCoM(joint_mat,
								body_def_mat,
								kin_char->GetPose(),
								kin_char->GetVel(),
								com_world,
								com_vel_world);

				fout << velocity_timestep << ' ' << com_vel_world[0] << ' ' << com_vel_world[1] << ' ' << com_vel_world[2] << '\n';
			}
		}
		fout.close();
	}
}

void cTMTSceneTemplateController::UpdateCharacters(double timestep)
{
	mUpdateContactCycleStartTimesCount++;
	if (mUpdateContactCycleStartTimesCount % 10 == 0)
	{
		mUpdateContactCycleStartTimesCount = 0;
		if (CheckContactCycleStart())
		{
			if (mContactCycleStartTimes.size() >= gMaxContactCycleBufferSize)
			{
				mContactCycleStartTimes.pop_front();
			}
			mContactCycleStartTimes.push_back(mTimer.GetTime());
		}
		mPrevContactIndex = GetContactIndex();
	}

	UpdateTargetVelocity(timestep);
	UpdateKinChar(timestep);

	mUpdateTargetHeadingCount++;
	if (mEnableRandRotation && mUpdateTargetHeadingCount >= 300) // 2 FPS (0.5 sec)
	{
		mUpdateTargetHeadingCount = 0;
		mTargetHeading = cMathUtilExtend::ClampEuler(mTargetHeading + cMathUtil::RandDouble(-0.15, 0.15));
		// if (cMathUtil::FlipCoin(0.1))
		// {
		// 	mTargetHeading = cMathUtil::RandDouble(-M_PI, M_PI);
		// }
	}

	cCarlRLSceneSimChar::UpdateCharacters(timestep);
}

void cTMTSceneTemplateController::UpdateKinChar(double timestep)
{
	auto kin_char = GetKinChar();
	kin_char->Update(timestep);
	double curr_time = kin_char->GetTime();

	if (curr_time >= mNextEndTime)
	{
		UpdateNextRandKinChar(false);

		kin_char = GetKinChar();
		kin_char->SetTime(mNextKinCharPhase);
		kin_char->Pose(mNextKinCharPhase);

		const auto& sim_char = GetCharacter();
		SyncKinCharNewCycle(*sim_char, *kin_char);
	}

	if (mSyncCharRootInUpdate)
	{
		SyncKinCharRoot();
	}
}

void cTMTSceneTemplateController::SyncKinCharRoot()
{
	const auto& sim_char = GetCharacter();
	tVector sim_root_pos = sim_char->GetRootPos();
	double sim_heading = sim_char->CalcHeading();

	const auto& kin_char = GetKinChar();
	double kin_heading = kin_char->CalcHeading();

	tQuaternion drot = tQuaternion::Identity();
	if (mSyncCharRootRot)
	{
		drot = cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), sim_heading - kin_heading);
	}

	kin_char->RotateRoot(drot);
#if 0
	kin_char->SetRootPos(sim_root_pos);
#else
	tVector kin_root_pos = kin_char->GetRootPos();
	kin_char->SetRootPos(tVector(sim_root_pos[0], kin_root_pos[1], sim_root_pos[2], 0));
#endif
}

void cTMTSceneTemplateController::SyncKinCharNewCycle(const cSimCharacter& sim_char, cCarlKinCharacter& out_kin_char) const
{
	if (mSyncCharRootRot)
	{
		double sim_heading = sim_char.CalcHeading();
		double kin_heading = out_kin_char.CalcHeading();
		tQuaternion drot = cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), sim_heading - kin_heading);
		out_kin_char.RotateRoot(drot);
	}

	if (mSyncCharRootPos)
	{
		tVector sim_root_pos = sim_char.GetRootPos();
		tVector kin_root_pos = out_kin_char.GetRootPos();
		kin_root_pos[0] = sim_root_pos[0];
		kin_root_pos[2] = sim_root_pos[2];

		tVector origin = out_kin_char.GetOriginPos();
		double dh = kin_root_pos[1] - origin[1];
		double ground_h = mGround->SampleHeight(kin_root_pos);
		kin_root_pos[1] = ground_h + dh;

		out_kin_char.SetRootPos(kin_root_pos);
	}
}

void cTMTSceneTemplateController::UpdateTargetVelocity(double timestep)
{
	// Update target velocity and position
	mTargetVelocity = GetKinVcom();
	if (GetMotionType() == eMotionTypePace)
	{
		tQuaternion qrot = cMathUtil::AxisAngleToQuaternion(tVector(0, 1, 0, 0), mTargetHeading);
		mTargetVelocity = cMathUtil::QuatRotVec(qrot, mTargetVelocity);
	}
	mTargetPosition += mTargetVelocity * timestep;

	// Update goal vector
	if (mCtrlMode == eControlMode::eHighLevelCtrl)
	{
		mUpdateVelocityCount += 1; // 600 fps
		if (timestep == 0 || mUpdateVelocityCount % 20 == 0) // 30 fps
		{
			mUpdateVelocityCount = 0;
			mGoalVector.resize(GetGoalSize(0));

			mGoalVector[0] = mTargetVelocity[0];
			mGoalVector[1] = mTargetVelocity[1];
			mGoalVector[2] = mTargetVelocity[2];

			if (GetMotionType() == eMotionTypePace)
			{
				tVector v_com_norm = GetCharacter()->CalcCOMVel().normalized();
				double theta_curr = atan2(-v_com_norm[2], v_com_norm[0]);
				double theta_delta = cMathUtilExtend::ClampEuler(mTargetHeading - theta_curr);

				mGoalVector[mGoalVector.size() - 1] = theta_delta * 5;
			}
			else
			{
				double target_height = mTargetPosition[1];
				mGoalVector[mGoalVector.size() - 1] = target_height;
			}
		}
	}
}

void cTMTSceneTemplateController::UpdateNextRandKinChar(bool rand_phase)
{
	if (!rand_phase)
	{
		mNextEndTime = GetKinChar()->GetMotionDuration();
		mNextKinCharPhase = 0;
	}
	else
	{
		mNextEndTime = GetKinChar()->GetMotionDuration();
		mNextKinCharPhase = cMathUtil::RandDouble(0, mNextEndTime);
	}
}

void cTMTSceneTemplateController::UpdateTimers(double timestep)
{
	cCarlRLSceneSimChar::UpdateTimers(timestep);

	// Update phase
	double dt_phase = mMotionTrajectories[mKinCharIndex].GetPhaseStep(timestep);
	mKinCharPhase += dt_phase;
	if (mKinCharPhase >= 1)
	{
		mKinCharPhase -= 1;
		mTargetPosition[1] = mStartPosition[1];
	}
}

double cTMTSceneTemplateController::GetKinTime() const
{
	const auto& kin_char = GetKinChar();
	return kin_char->GetTime();
}

void cTMTSceneTemplateController::SyncCharacters()
{
	const auto& kin_char = GetKinChar();
	const Eigen::VectorXd& pose = kin_char->GetPose();
	const Eigen::VectorXd& vel = kin_char->GetVel();

	const auto& sim_char = GetCharacter();
	sim_char->SetPose(pose);
	sim_char->SetVel(vel);

	const auto& ctrl = sim_char->GetController();
	auto ct_ctrl = dynamic_cast<cCtController*>(ctrl.get());
	if (ct_ctrl != nullptr)
	{
		double kin_time = GetKinTime();
		double cycle_dur = kin_char->GetMotionDuration();
		ct_ctrl->SetInitTime(kin_time);
		ct_ctrl->SetCyclePeriod(cycle_dur);
	}
}

bool cTMTSceneTemplateController::EnableSyncChar() const
{
	const auto& kin_char = GetKinChar();
	return kin_char->HasMotion();
}

void cTMTSceneTemplateController::ResetCharacters()
{
	cCarlRLSceneSimChar::ResetCharacters();

	mUpdateVelocityCount = 0;
	mUpdateContactCycleStartTimesCount = 0;
	mTargetHeading = 0;
	mPrevContactIndex = -1;
	mContactCycleStartTimes.clear();

	if (mEnableRandRotation)
	{
		mTargetHeading = cMathUtil::RandDouble(-M_PI * 0.2, M_PI * 0.2); // +- 36 degree
	}

	ResetKinChar();
	UpdateTargetVelocity(0);

	if (EnableSyncChar())
	{
		SyncCharacters();
	}
}

void cTMTSceneTemplateController::ResetKinChar()
{
	mKinCharIndex = cMathUtil::RandInt(0, mMotionTrajectories.size());
	mKinCharPhase = cMathUtil::RandDouble(0, 1);

	ResetKinChar(mKinCharIndex, mKinCharPhase);
}

void cTMTSceneTemplateController::ResetKinChar(int motion_idx, double phase)
{
	mKinCharIndex = motion_idx;
	mKinCharPhase = phase;

	const cSimCharacter::tParams& char_params = mCharParams[0];
	const auto& kin_char = GetKinChar();
	double kinCharTime = mKinCharPhase * kin_char->GetMotionDuration();

	kin_char->Reset();
	kin_char->SetOriginRot(tQuaternion::Identity());
	kin_char->SetOriginPos(char_params.mInitPos); // reset origin
	kin_char->SetTime(kinCharTime);
	kin_char->Pose(kinCharTime);

	tVector root_pos = kin_char->GetRootPos();
	kin_char->SetRootPos(tVector(0, root_pos[1], 0, 0));

	if (EnabledRandRotReset())
	{
		double rand_theta = mRand.RandDouble(-M_PI, M_PI);
		mTargetHeading = rand_theta;
		kin_char->RotateOrigin(cMathUtil::EulerToQuaternion(tVector(0, rand_theta, 0, 0)));
	}

	mNextEndTime = kin_char->GetMotionDuration();
}

bool cTMTSceneTemplateController::HasFallen(const cSimCharacter& sim_char) const
{
	bool fallen = cCarlRLSceneSimChar::HasFallen(sim_char);
	if (mEnableRootRotFail)
	{
		fallen |= CheckRootRotFail(sim_char, mTargetVelocity);
	}

	return fallen;
}

bool cTMTSceneTemplateController::CheckRootRotFail(const cSimCharacter& sim_char, tVector target_velocity) const
{
	tVector v_com = sim_char.CalcCOMVel();
	tVector v_sim = tVector(v_com[0], 0, v_com[2], 0).normalized();
	tVector v_kin = tVector(target_velocity[0], 0, target_velocity[2], 0).normalized();
	double v_proj = v_sim.dot(v_kin);

	return v_proj < 0;
}

bool cTMTSceneTemplateController::CheckContactCycleStart() const
{
	if (mPrevContactIndex < 0)
		return false;

	bool is_cycle_start = false;
	int contact_index = GetContactIndex();

	if (mKinCharMotionTypes[mKinCharIndex] == eMotionTypePace)
	{
		is_cycle_start = (contact_index == 5) && (contact_index != mPrevContactIndex);
	}
	else if (mKinCharMotionTypes[mKinCharIndex] == eMotionTypeTrot)
	{
		is_cycle_start = (contact_index == 6) && (contact_index != mPrevContactIndex);
	}
	else if (mKinCharMotionTypes[mKinCharIndex] == eMotionTypeCanter)
	{
		is_cycle_start = (contact_index == 1) && (contact_index != mPrevContactIndex);
	}
	return is_cycle_start;
}

std::vector<bool> cTMTSceneTemplateController::GetFootContacts() const
{
	const std::shared_ptr<cSimCharacter> sim_char = GetCharacter();
	const std::shared_ptr<cCarlKinCharacter> kin_char = GetKinChar();

	const Eigen::MatrixXd& joint_mat = kin_char->GetJointMat();
	const Eigen::MatrixXd& pose = sim_char->GetPose();

	std::vector<bool> contacts(mFootContactIndices.size());
	for (int i = 0; i < mFootContactIndices.size(); ++i)
	{
		int joint_id = mFootContactIndices[i];
		double thres = cCarlKinTree::GetContactThreshold(joint_mat, joint_id);
		tVector offset = cCarlKinTree::GetContactOffset(joint_mat, joint_id);
		tMatrix m = cCarlKinTree::JointWorldTrans(joint_mat, pose, joint_id);
		tVector pos_contact = m * offset;
		bool contact = pos_contact[1] < thres;
		contacts[i] = contact;
	}

	return contacts;
}

std::vector<int> cTMTSceneTemplateController::GetFootContactJoints() const
{
	return mFootContactIndices;
}

int cTMTSceneTemplateController::GetContactIndex() const
{
	int contact_index = 0;
	std::vector<bool> foot_contacts = GetFootContacts();
	for (int i = 0; i < foot_contacts.size(); ++i)
	{
		if (foot_contacts[i])
		{
			contact_index += pow(2, i);
		}
	}
	return contact_index;
}

double cTMTSceneTemplateController::GetSimCharPhase() const
{
	double phase = -1;
	if (mKinCharMotionTypes[mKinCharIndex] == eMotionTypeCanter ||
		mKinCharMotionTypes[mKinCharIndex] == eMotionTypePace   ||
		mKinCharMotionTypes[mKinCharIndex] == eMotionTypeTrot)
	{
		if (mContactCycleStartTimes.size() > 1)
		{
			double max_cycle_length = -INT_MAX;
			for (int i = 1; i < mContactCycleStartTimes.size(); ++i)
			{
				double cycle_length = mContactCycleStartTimes[i] - mContactCycleStartTimes[i - 1];
				max_cycle_length = std::fmax(cycle_length, max_cycle_length);
			}
			phase = (mTimer.GetTime() - mContactCycleStartTimes.back()) / max_cycle_length;
		}
	}
	else
	{
		phase = mKinCharPhase;
	}
	return phase;
}
