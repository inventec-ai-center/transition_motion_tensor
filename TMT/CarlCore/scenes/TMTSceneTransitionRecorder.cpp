#include "TMTSceneTransitionRecorder.h"
#include <math.h>
#include <iterator>
#include <algorithm>
#include "sim/RBDUtil.h"
#include "sim/CtController.h"
#include "util/FileUtil.h"
#include "util/JsonUtil.h"
#include "sim/CharController.h"
#include "render/DrawUtil.h"
#include "util/MathUtilExtend.h"

const int gWriteIntermediateIter = 100;
const double gPhaseTolerance = 0.01;
const int gMaxContactCycleBufferSize = 3;


cTMTSceneTransitionRecorder::cTMTSceneTransitionRecorder() : cTMTSceneTemplateController()
{
	mExpIndex = 0;
	mUpdateCounter = 0;
	mUpdateRecordCharStateCount = 0;
	mHasSwitched = true;
	mEnableRandRotReset = false;
	mEnableRootRotFail = false;
	mEnableRandRotation = false;
	mSyncCharRootRot = false;
	mHasVelocityExploded = false;
	mHasFallen = false;
	mNeedUpdateAction = false;
}

cTMTSceneTransitionRecorder::~cTMTSceneTransitionRecorder()
{
}

void cTMTSceneTransitionRecorder::ParseArgs(const std::shared_ptr<cArgParser>& parser)
{
	cTMTSceneTemplateController::ParseArgs(parser);

	parser->ParseString("eval_output_filename", mOutputFilename);
	parser->ParseInt("num_collect_transition_samples", mNumCollectTransitionSamples);
	parser->ParseDouble("pre_transition_time", mPreTransitionTime);
	parser->ParseDouble("post_transition_time", mPostTransitionTime);
}

void cTMTSceneTransitionRecorder::Init()
{
	cTMTSceneTemplateController::Init();

	BuildCollectTransitionTrajectories();
	InitTransitionSample(mExpIndex);
	printf("Number of transition samples: %d\n", mTransitionSamples.size());

	UpdateTargetVelocity(0);
}

std::string cTMTSceneTransitionRecorder::GetName() const
{
	return "Transition Recorder";
}

bool cTMTSceneTransitionRecorder::CheckValidEpisode() const
{
	return true;
}

bool cTMTSceneTransitionRecorder::NeedNewAction(int agent_id) const
{
	return mNeedUpdateAction || cTMTSceneTemplateController::NeedNewAction(agent_id);
}

void cTMTSceneTransitionRecorder::SetAction(int agent_id, const Eigen::VectorXd& action)
{
	mNeedUpdateAction = false;
	cTMTSceneTemplateController::SetAction(agent_id, action);
}

void cTMTSceneTransitionRecorder::UpdateCharacters(double timestep)
{
	mUpdateContactCycleStartTimesCount++; // 600 FPS
	if (mUpdateContactCycleStartTimesCount % 10 == 0) // 60 FPS
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

	// Switch motion
	if (!mHasSwitched && mTimer.GetTime() >= mPreTransitionTime)
	{
		double phase_sim = GetSimCharPhase();
		if (abs(phase_sim - mTransitionSamples[mExpIndex].transPhase) < gPhaseTolerance)
		{
			mNeedUpdateAction = true;
			mHasSwitched = true;
			mTransitionSamples[mExpIndex].transition_index = mTransitionSamples[mExpIndex].losses_energy.size();
			mTransitionSamples[mExpIndex].transition_phase = phase_sim;

			SwitchMotion(0, mTransitionSamples[mExpIndex].dstMotion, mTransitionSamples[mExpIndex].dstPhase);
		}
	}

	if (!mHasVelocityExploded)
	{
		mHasVelocityExploded = GetCharacter()->HasVelExploded();
	}

	if (!mHasFallen)
	{
		mHasFallen = CheckCharacterFallen(GetCharacter());
	}

	UpdateTargetVelocity(timestep);
	UpdateKinChar(timestep);

	cCarlRLSceneSimChar::UpdateCharacters(timestep);
}

void cTMTSceneTransitionRecorder::PostUpdate(double timestep)
{
	cTMTSceneTemplateController::PostUpdate(timestep);

	mUpdateCounter++; // 600 FPS
	if (mUpdateCounter % 20 == 0) // 30 FPS
	{
		mUpdateCounter = 0;

		const auto& sim_char = GetCharacter();
		double loss_energy = GetLossEnergyConsumption();
		double speed_reward = GetRewardSpeed();
		double heading_reward = GetRewardHeading();
		double height_reward = GetRewardHeight();
		tVector position = sim_char->CalcCOM();
		std::vector<bool> foot_contacts = GetFootContacts();

		mTransitionSamples[mExpIndex].losses_energy.push_back(loss_energy);
		mTransitionSamples[mExpIndex].foot_contacts.push_back(foot_contacts);
		mTransitionSamples[mExpIndex].speed_rewards.push_back(speed_reward);
		mTransitionSamples[mExpIndex].heading_rewards.push_back(heading_reward);
		mTransitionSamples[mExpIndex].height_rewards.push_back(height_reward);
		mTransitionSamples[mExpIndex].positions.push_back(position);
	}

	if (mTimer.GetTime() >= mPreTransitionTime + mPostTransitionTime)
	{
		OnEvaluationEnd();
	}
}

void cTMTSceneTransitionRecorder::OnEvaluationEnd()
{
	mTransitionSamples[mExpIndex].isFallen = mHasFallen;
	mTransitionSamples[mExpIndex].isValid = !mHasVelocityExploded && mHasSwitched;
	// printf("[OnEvaluationEnd] isFallen=%s\n", mTransitionSamples[mExpIndex].isFallen ? "true" : "false");

	// Update experiment index and progress bar
	mExpIndex += 1;
	mProgressBar.progress(mExpIndex, mTransitionSamples.size());

	// Write intermediate resutls
	if (gWriteIntermediateIter > 0 &&
		mExpIndex > 0 &&
		((mExpIndex % gWriteIntermediateIter == 0) || (mExpIndex == mNumCollectTransitionSamples)))
	{
		int count, start_exp, end_exp;
		if (mExpIndex % gWriteIntermediateIter == 0)
		{
			count = mExpIndex / gWriteIntermediateIter;
			start_exp = (count - 1) * gWriteIntermediateIter;
			end_exp = count * gWriteIntermediateIter;
		}
		else
		{
			count = mExpIndex / gWriteIntermediateIter + 1;
			start_exp = (count - 1) * gWriteIntermediateIter;
			end_exp = mNumCollectTransitionSamples;
		}

		std::string replace_from = ".json";
		std::string replace_to = "_part" + std::to_string(count) + ".json";
		size_t start_pos = mOutputFilename.find(replace_from);
		if(start_pos != std::string::npos)
		{
			std::string output_filename = mOutputFilename;
			output_filename.replace(start_pos, replace_from.length(), replace_to);
			WriteCollectTransitionTrajectories(output_filename, start_exp, end_exp);
			printf("Write intermediate reusults to file %s (%d, %d)\n", output_filename.c_str(), start_exp, end_exp);

			if (end_exp >= mTransitionSamples.size())
			{
				exit(0);
			}
		}
		else
		{
			printf("Failed to create intermediate file from %s\n", mOutputFilename.c_str());
		}
	}
	ResetScene();
}

void cTMTSceneTransitionRecorder::ResetCharacters()
{
	cCarlRLSceneSimChar::ResetCharacters();

	mUpdateCounter = 0;
	mUpdateVelocityCount = 0;
	mUpdateRecordCharStateCount = 0;
	mUpdateContactCycleStartTimesCount = 0;

	InitTransitionSample(mExpIndex);
	UpdateTargetVelocity(0);
}

void cTMTSceneTransitionRecorder::InitTransitionSample(int exp_id)
{
	mNeedUpdateAction = false;
	mHasSwitched = false;
	mHasVelocityExploded = false;
	mHasFallen = false;
	mKinCharIndex = mTransitionSamples[exp_id].srcMotion;
	mKinCharPhase = mTransitionSamples[exp_id].srcPhase;

	double motion_time = mKinCharPhase * GetKinChar(mKinCharIndex)->GetMotionDuration();
	SyncPose(mKinCharIndex, motion_time);

	mPrevContactIndex = -1;
	mContactCycleStartTimes.clear();
}

void cTMTSceneTransitionRecorder::BuildCollectTransitionTrajectories()
{
	mTransitionSamples.clear();
	for (int i = 0; i < mNumCollectTransitionSamples; ++i)
	{
		TransitionSample trans_sample;
		GenerateRandTransitionSample(trans_sample);
		mTransitionSamples.push_back(trans_sample);
	}
}

void cTMTSceneTransitionRecorder::GenerateRandTransitionSample(TransitionSample &trans_sample) const
{
	int src_motion = cMathUtil::RandInt(0, mMotionTrajectories.size());
	int dst_motion = (src_motion + cMathUtil::RandInt(1, mMotionTrajectories.size())) % mMotionTrajectories.size();

	trans_sample.srcMotion = src_motion;
	trans_sample.dstMotion = dst_motion;
	trans_sample.srcPhase = cMathUtil::RandDouble(0, 1);
	trans_sample.dstPhase = cMathUtil::RandDouble(0, 1);
	trans_sample.transPhase = cMathUtil::RandDouble(0, 1);
	trans_sample.srcVelocity = mMotionTrajectories[src_motion].Sample(trans_sample.srcPhase);
	trans_sample.dstVelocity = mMotionTrajectories[dst_motion].Sample(trans_sample.dstPhase);
	trans_sample.transition_index = -1;
}

void cTMTSceneTransitionRecorder::WriteCollectTransitionTrajectories(std::string output_filename, int start_idx, int end_idx)
{
	std::string json = "[\n";
	for (int i = start_idx; i < end_idx; ++i)
	{
		TransitionSample sample = mTransitionSamples[i];

		std::string loss_energy_json = "[";
		std::string foot_contacts_json = "[";
		std::string speed_reward_json = "[";
		std::string heading_reward_json = "[";
		std::string height_reward_json = "[";
		std::string position_json = "[";
		std::string src_motion_json = "\"" + mMotionNames[sample.srcMotion] + "\"";
		std::string dst_motion_json = "\"" + mMotionNames[sample.dstMotion] + "\"";
		std::string src_phase_json = std::to_string(sample.transition_phase);
		std::string dst_phase_json = std::to_string(sample.dstPhase);
		std::string transition_index_json = std::to_string(sample.transition_index);
		std::string is_fallen_json = sample.isFallen ? "true" : "false";
		std::string is_valid_json = sample.isValid ? "true" : "false";

		if (sample.isValid)
		{
			int num_steps = sample.positions.size();
			for (int j = 0; j < num_steps; ++j)
			{
				loss_energy_json += std::to_string(sample.losses_energy[j]);
				speed_reward_json += std::to_string(sample.speed_rewards[j]);
				heading_reward_json += std::to_string(sample.heading_rewards[j]);
				height_reward_json += std::to_string(sample.height_rewards[j]);
				position_json += "[" + cJsonUtil::BuildVectorString(sample.positions[j]) + "]";

				std::vector<bool> foot_contacts = sample.foot_contacts[j];
				std::string foot_contact_str = "\"";
				for (int k = 0; k < foot_contacts.size(); ++k)
				{
					foot_contact_str += foot_contacts[k] ? "1" : "0";
				}
				foot_contact_str += "\"";
				foot_contacts_json += foot_contact_str;

				if (j != num_steps - 1)
				{
					loss_energy_json += ", ";
					foot_contacts_json += ", ";
					speed_reward_json += ", ";
					heading_reward_json += ", ";
					height_reward_json += ", ";
					position_json += ", ";
				}
			}
		}

		loss_energy_json += "]";
		foot_contacts_json += "]";
		speed_reward_json += "]";
		heading_reward_json += "]";
		height_reward_json += "]";
		position_json += "]";

		std::string exp_json = "{\n";
		exp_json += "\"src_motion\": " + src_motion_json + ",\n";
		exp_json += "\"src_phase\": " + src_phase_json + ",\n";
		exp_json += "\"dst_motion\": " + dst_motion_json + ",\n";
		exp_json += "\"dst_phase\": " + dst_phase_json + ",\n";
		exp_json += "\"is_fallen\": " + is_fallen_json + ",\n";
		exp_json += "\"is_valid\": " + is_valid_json + ",\n";
		exp_json += "\"loss_energy\": " + loss_energy_json + ",\n";
		exp_json += "\"speed_reward\": " + speed_reward_json + ",\n";
		exp_json += "\"heading_reward\": " + heading_reward_json + ",\n";
		exp_json += "\"height_reward\": " + height_reward_json + ",\n";
		exp_json += "\"position\": " + position_json + ",\n";
		exp_json += "\"foot_contacts\": " + foot_contacts_json + ",\n";
		exp_json += "\"transition_index\": " + transition_index_json + "\n";
		exp_json += "}";
		exp_json += (i == end_idx - 1) ? "\n" : ",\n";

		json += exp_json;
	}
	json += "]";

	FILE* f = cFileUtil::OpenFile(output_filename, "w");
	if (f != nullptr)
	{
		fprintf(f, "%s", json.c_str());
		cFileUtil::CloseFile(f);
	}

	// Release memory
	for (int i = start_idx; i < end_idx; ++i)
	{
		mTransitionSamples[i].losses_energy.clear();
		mTransitionSamples[i].speed_rewards.clear();
		mTransitionSamples[i].heading_rewards.clear();
		mTransitionSamples[i].height_rewards.clear();
		mTransitionSamples[i].positions.clear();
		mTransitionSamples[i].foot_contacts.clear();
	}
}

bool cTMTSceneTransitionRecorder::CheckTransitionSuccess() const
{
	const auto& sim_char = GetCharacter();
	bool fallen = CheckCharacterFallen(sim_char);

	return !fallen & !mHasVelocityExploded;
}

bool cTMTSceneTransitionRecorder::CheckCharacterFallen(const std::shared_ptr<cSimCharacter>sim_char) const
{
	int num_parts = sim_char->GetNumBodyParts();
	for (int b = 0; b < num_parts; ++b)
	{
		if (sim_char->IsValidBodyPart(b) && sim_char->EnableBodyPartFallContact(b))
		{
			const auto& curr_part = sim_char->GetBodyPart(b);
			bool has_contact = curr_part->IsInContact();
			if (has_contact)
			{
				return true;
			}
		}
	}
	return false;
}

void cTMTSceneTransitionRecorder::SyncPose(int kin_index, double motion_time)
{
	const cSimCharacter::tParams& char_params = mCharParams[0];
	const auto& kin_char = GetKinChar(kin_index);
	const auto& sim_char = GetCharacter();

	kin_char->Reset();
	kin_char->SetOriginRot(tQuaternion::Identity());
	kin_char->SetOriginPos(char_params.mInitPos); // reset origin
	kin_char->SetTime(motion_time);
	kin_char->Pose(motion_time);

	const Eigen::VectorXd& pose = kin_char->GetPose();
	const Eigen::VectorXd& vel = kin_char->GetVel();

	sim_char->SetPose(pose);
	sim_char->SetVel(vel);

	const auto& ctrl = sim_char->GetController();
	auto ct_ctrl = dynamic_cast<cCtController*>(ctrl.get());
	if (ct_ctrl != nullptr)
	{
		double kin_time = kin_char->GetTime();
		ct_ctrl->SetInitTime(kin_time);

		double cylcle_dur = kin_char->GetMotionDuration();
		ct_ctrl->SetCyclePeriod(cylcle_dur);
	}

	tVector root_pos = kin_char->GetRootPos();
	kin_char->SetRootPos(tVector(0, root_pos[1], 0, 0));
}
