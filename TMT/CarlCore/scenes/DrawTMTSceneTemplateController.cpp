#include "DrawTMTSceneTemplateController.h"
#include <math.h>
#include "render/DrawCharacterExtend.h"
#include "render/DrawUtilExtend.h"
#include "sim/DeepMimicCharController.h"
#include "scenes/SceneSimChar.h"
#include "util/MathUtilExtend.h"
#include "anim/CarlKinTree.h"

extern int g_ScreenWidth;
extern int g_ScreenHeight;

const tVector gCurrentVelocityArrowColor = tVector(1, 0, 0, 0.5);
const tVector gTargetVelocityArrowColor = tVector(0, 1, 0, 0.5);
const double gMoveSpeed = 0.1;
const double gRotateSpeed = 0.2;
const int PhaseLogBufferSize = 3000;
const int FootContactsLogBufferSize = 1000;


cDrawTMTSceneTemplateController::cDrawTMTSceneTemplateController()
{
	mEnableDrawPhaseInfo = true;
	mEnableDrawContactsInfo = false;
	mPhaseLog.Reserve(PhaseLogBufferSize);
}

cDrawTMTSceneTemplateController::~cDrawTMTSceneTemplateController()
{
}

void cDrawTMTSceneTemplateController::Init()
{
	cDrawSceneSimChar::Init();
	cRLScene::Init();

	ResetScene();
	InitFootContacts();
}

void cDrawTMTSceneTemplateController::Clear()
{
	cDrawSceneSimChar::Clear();
	cDrawCarlRLScene::Clear();

	mPhaseLog.Clear();
}

bool cDrawTMTSceneTemplateController::IsEpisodeEnd() const
{
	return cDrawCarlRLScene::IsEpisodeEnd();
}

bool cDrawTMTSceneTemplateController::CheckValidEpisode() const
{
	return cDrawCarlRLScene::CheckValidEpisode();
}

void cDrawTMTSceneTemplateController::Update(double time_elapsed)
{
	cDrawSceneSimChar::Update(time_elapsed);

	std::shared_ptr<cTMTSceneTemplateController> scene = std::dynamic_pointer_cast<cTMTSceneTemplateController>(GetScene());
	const std::shared_ptr<cSimCharacter> character = scene->GetCharacter();

	if (mEnableDrawContactsInfo)
	{
		UpdateFootContacts();
	}

	if (mEnableDrawPhaseInfo)
	{
		std::vector<double> phase_info;
		phase_info.push_back(scene->GetPhase());
		phase_info.push_back(scene->GetLocalPhase());
		phase_info.push_back(scene->GetMotion());
		phase_info.push_back(scene->GetSimCharPhase());

		mPhaseLog.Add(phase_info);
	}
}

void cDrawTMTSceneTemplateController::CtrlHeading(double delta)
{
	cTMTSceneTemplateController* scene = dynamic_cast<cTMTSceneTemplateController*>(mScene.get());
	if (scene->GetControlMode() == cTMTSceneTemplateController::eControlMode::eHighLevelCtrl)
	{
		double target_heading = cMathUtilExtend::ClampEuler(scene->GetTargetHeading() + delta);
		scene->SetTargetHeading(target_heading);
	}
}

void cDrawTMTSceneTemplateController::Keyboard(unsigned char key, double device_x, double device_y)
{
	cTMTSceneTemplateController* scene = dynamic_cast<cTMTSceneTemplateController*>(mScene.get());

	switch (key)
	{
	case 'a':
		CtrlHeading(gRotateSpeed);
		break;
	case 'd':
		CtrlHeading(-gRotateSpeed);
		break;
	case 'x':
		SpawnProjectile();
		break;
	default:
		break;
	}
}

std::string cDrawTMTSceneTemplateController::GetName() const
{
	return cDrawCarlRLScene::GetName();
}

void cDrawTMTSceneTemplateController::ResetScene()
{
	cDrawSceneSimChar::ResetScene();

	mPhaseLog.Clear();
}

tVector cDrawTMTSceneTemplateController::GetCamTrackPos() const
{
	const auto& character = mScene->GetCharacter();
	tVector track_pos = character->CalcCOM();

	return track_pos;
}

cRLScene* cDrawTMTSceneTemplateController::GetRLScene() const
{
	return dynamic_cast<cRLScene*>(mScene.get());
}

cCarlRLScene* cDrawTMTSceneTemplateController::GetCarlRLScene() const
{
	return dynamic_cast<cCarlRLScene*>(mScene.get());
}

const std::shared_ptr<cCarlKinCharacter>& cDrawTMTSceneTemplateController::GetKinChar() const
{
	const cTMTSceneTemplateController* scene = dynamic_cast<const cTMTSceneTemplateController*>(mScene.get());
	return scene->GetKinChar();
}

void cDrawTMTSceneTemplateController::DrawInfo() const
{
	DrawInfoText();

	if (mEnableDrawPhaseInfo)
	{
		DrawPhaseInfo();
	}

	if (mEnableDrawContactsInfo)
	{
		DrawFootContactsInfo();
	}
}

void cDrawTMTSceneTemplateController::DrawInfoText() const
{
	cDrawUtilExtend::BeginDrawString();
	{
		const std::shared_ptr<cTMTSceneTemplateController> vel_scene = std::dynamic_pointer_cast<cTMTSceneTemplateController>(mScene);

		cTMTSceneTemplateController::eMotionType motion_type = vel_scene->GetMotionType();
		std::string motion_type_str = "";
		if (motion_type == cTMTSceneTemplateController::eMotionType::eMotionTypeCanter)
		{
			motion_type_str = "Canter";
		}
		else if (motion_type == cTMTSceneTemplateController::eMotionType::eMotionTypeJump)
		{
			motion_type_str = "Jump";
		}
		else if (motion_type == cTMTSceneTemplateController::eMotionType::eMotionTypeTrot)
		{
			motion_type_str = "Trot";
		}
		else if (motion_type == cTMTSceneTemplateController::eMotionType::eMotionTypePace)
		{
			motion_type_str = "Pace";
		}
		else
		{
			motion_type_str = "Generic";
		}

		tVector v_com = vel_scene->GetCharacter()->CalcCOMVel();
		const std::shared_ptr<cSimCharacter> sim_char = mScene->GetCharacter();
		tVector world_pos = sim_char->CalcCOM();
		world_pos[1] += 0.5;
		float speed = sim_char->CalcCOMVel().norm();
		float heading = 0;
		float com_height = world_pos[1];
		double reward = vel_scene->CalcReward(0);

		char str[512];
		float screen_y = g_ScreenHeight * 0.95;
		float screen_x = g_ScreenHeight * 0.02;

		double phase = vel_scene->GetPhase();
		double local_phase = vel_scene->GetLocalPhase();
		double phase_sim = vel_scene->GetSimCharPhase();

		sprintf(str, "Time: %.1fs\nMotion: %s\nPhase: %f\nHeight: %f\nSpeed: %f\nReward: %f\n",
				mTimer.GetTime(), motion_type_str.c_str(), phase_sim, com_height, speed, reward);

		cDrawUtilExtend::DrawString(screen_x, screen_y, str);
	}
	cDrawUtilExtend::EndDrawString();
}

void cDrawTMTSceneTemplateController::DrawPhaseInfo() const
{
	const auto& character = mScene->GetCharacter();

	double min_val = 0;
	double max_val = 1;

	int num_val = static_cast<int>(mPhaseLog.GetSize());
	double aspect = mCamera.GetAspectRatio();

	const double h = 0.3;
	const double w = 1.94;

	tVector origin = tVector::Zero();
	origin[0] = -0.97;
	origin[1] = -0.98;
	origin[2] = -1;

	int capacity = static_cast<int>(mPhaseLog.GetCapacity());

	cDrawUtil::SetLineWidth(1);
	cDrawUtil::SetColor(tVector(1, 1, 1, 0.5));
	cDrawUtil::DrawRect(origin + 0.5 * tVector(w, h, 0, 0), tVector(w, h, 0, 0));
	cDrawUtil::SetColor(tVector(0, 0, 0, 1));
	cDrawUtil::DrawRect(origin + 0.5 * tVector(w, h, 0, 0), tVector(w, h, 0, 0), cDrawUtil::eDrawWireSimple);

	cDrawUtil::SetLineWidth(2);
	cDrawUtil::SetPointSize(6);

	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	if (num_val > 0)
	{
		double prev_val = mPhaseLog[0][3];
		for (int i = 1; i < num_val; ++i)
		{
			double curr_val = mPhaseLog[i][3];
			int motion_label = int(mPhaseLog[i][2]);

			// bool is_contact_start = mPhaseLog[i][4] > 0.5;
			bool is_contact_start = false;

			tVector a = tVector::Zero();
			tVector b = tVector::Zero();

			a[0] = w * (i - 1.0) / (capacity - 1.0);
			b[0] = w * (i) / (capacity - 1.0);

			a[1] = h * cMathUtil::Clamp((prev_val - min_val) / (max_val - min_val), 0.0, 1.0);
			b[1] = h * cMathUtil::Clamp((curr_val - min_val) / (max_val - min_val), 0.0, 1.0);

			a += origin;
			b += origin;

			if (motion_label == 0)
			{
				cDrawUtil::SetColor(tVector(1, 0, 0, 1));
			}
			else if (motion_label == 1)
			{
				cDrawUtil::SetColor(tVector(0.5, 0.5, 0, 1));
			}
			else if (motion_label == 2)
			{
				cDrawUtil::SetColor(tVector(0, 0, 1, 1));
			}
			else if (motion_label == 3)
			{
				cDrawUtil::SetColor(tVector(1, 0, 1, 1));
			}

			cDrawUtil::DrawLine(a, b);
			if (is_contact_start)
			{
				cDrawUtil::SetColor(tVector(0, 0, 0, 1));
				cDrawUtil::DrawPoint(b);
			}
			prev_val = curr_val;
		}
	}
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_LINE_SMOOTH);
}

void cDrawTMTSceneTemplateController::DrawMisc() const
{
	const cTMTSceneTemplateController* scene = dynamic_cast<const cTMTSceneTemplateController*>(mScene.get());

	if (mEnableTrace)
	{
		DrawTrace();
	}

	if (mEnableDrawContactsInfo)
	{
		DrawFootContactPoints();
	}

	if (scene->GetControlMode() == cTMTSceneTemplateController::eControlMode::eHighLevelCtrl)
	{
		DrawHeading();
	}

	cDrawSceneSimChar::DrawMisc();
}

void cDrawTMTSceneTemplateController::BuildScene(std::shared_ptr<cSceneSimChar>& out_scene) const
{
	out_scene = std::shared_ptr<cTMTSceneTemplateController>(new cTMTSceneTemplateController());
}

void cDrawTMTSceneTemplateController::DrawCharacters() const
{
	cDrawSceneSimChar::DrawCharacters();

	std::shared_ptr<cTMTSceneTemplateController> scene = std::dynamic_pointer_cast<cTMTSceneTemplateController>(GetScene());
	if (scene->GetControlMode() == cTMTSceneTemplateController::eControlMode::ePhaseCtrl)
	{
		DrawKinCharacter();
	}
}

void cDrawTMTSceneTemplateController::DrawKinCharacter() const
{
	const auto& kin_char = GetKinChar();
	const double gLinkWidth = 0.025f;
	const tVector gLineColor = tVector(0, 0, 0, 1);

	if (kin_char->HasDrawShapes())
	{
		cDrawCharacter::Draw(*kin_char, 0, tVector(1.25, 0.75, 0.75, 1), gLineColor);
		cDrawCharacterExtend::DrawBones(*kin_char, 0.025f, tVector(1, 0.75, 0.25, 0.75), tVector(1, 0, 0, 1));
	}
	else
	{
		cDrawCharacterExtend::DrawBones(*kin_char, gLinkWidth, tVector(0, 0, 0, 1), tVector(1, 0, 0, 1));
	}
}

void cDrawTMTSceneTemplateController::DrawHeading() const
{
	std::shared_ptr<cTMTSceneTemplateController> scene = std::dynamic_pointer_cast<cTMTSceneTemplateController>(GetScene());
	const std::shared_ptr<cSimCharacter> character = scene->GetCharacter();
	const double arrow_size = 0.2;
	tVector p_com = character->CalcCOM();
	tVector v_com = character->CalcCOMVel();
	tVector target_velocity = scene->GetTargetVelocity();
	tVector root_pos = character->GetRootPos();
	double target_heading = scene->GetTargetHeading();
	tVector start, end;

	// Draw target velocity
	start = p_com;
	end = start + target_velocity.normalized() * sqrt(target_velocity.norm());
	cDrawUtil::SetColor(gTargetVelocityArrowColor);
	cDrawUtil::DrawArrow2D(start, end, arrow_size);

	// Draw center-of-mass velocity
	start = p_com;
	end = start + v_com.normalized() * sqrt(v_com.norm());
	cDrawUtil::SetColor(gCurrentVelocityArrowColor);
	cDrawUtil::DrawArrow2D(start, end, arrow_size);
}

void cDrawTMTSceneTemplateController::InitFootContacts()
{
	mSimFootContactStates.clear();

	cTMTSceneTemplateController* scene = dynamic_cast<cTMTSceneTemplateController*>(mScene.get());

	std::vector<bool> sim_contacts = scene->GetFootContacts();
	for (int i = 0; i < sim_contacts.size(); ++i)
	{
		mSimFootContactStates.push_back(std::deque<bool>());
	}
}

void cDrawTMTSceneTemplateController::UpdateFootContacts()
{
	cTMTSceneTemplateController* scene = dynamic_cast<cTMTSceneTemplateController*>(mScene.get());
	const std::shared_ptr<cCarlKinCharacter> kin_char = scene->GetKinChar();

	std::vector<bool> sim_contacts = scene->GetFootContacts();
	for (int i = 0; i < sim_contacts.size(); ++i)
	{
		mSimFootContactStates[i].push_back(sim_contacts[i]);

		if (mSimFootContactStates[i].size() > FootContactsLogBufferSize)
		{
			mSimFootContactStates[i].pop_front();
		}
	}
}

void cDrawTMTSceneTemplateController::DrawFootContactPoints() const
{
	cTMTSceneTemplateController* scene = dynamic_cast<cTMTSceneTemplateController*>(mScene.get());
	const std::shared_ptr<cSimCharacter> sim_char = scene->GetCharacter();
	const std::shared_ptr<cCarlKinCharacter> kin_char = scene->GetKinChar();
	const std::vector<int> footContactIndices = scene->GetFootContactJoints();

	const Eigen::MatrixXd& joint_mat = kin_char->GetJointMat();
	const Eigen::MatrixXd& pose = sim_char->GetPose();

	glDisable(GL_DEPTH_TEST);

	std::vector<bool> contacts(footContactIndices.size());
	for (int i = 0; i < footContactIndices.size(); ++i)
	{
		int joint_id = footContactIndices[i];
		double thres = cCarlKinTree::GetContactThreshold(joint_mat, joint_id);
		tVector offset = cCarlKinTree::GetContactOffset(joint_mat, joint_id);
		tMatrix m = cCarlKinTree::JointWorldTrans(joint_mat, pose, joint_id);
		tVector pos_contact = m * offset;
		bool contact = pos_contact[1] < thres;

		cDrawUtil::PushMatrixView();
		if (contact)
		{
			cDrawUtil::SetColor(tVector(0, 1, 0, 0.5));
		}
		else
		{
			cDrawUtil::SetColor(tVector(1, 0, 0, 0.5));
		}
		cDrawUtil::Translate(pos_contact);
		cDrawUtil::DrawSphere(0.05f);
		cDrawUtil::PopMatrixView();
	}

	glEnable(GL_DEPTH_TEST);
}

void cDrawTMTSceneTemplateController::DrawFootContactsInfo() const
{
	int num_val = static_cast<int>(mSimFootContactStates[0].size());
	double aspect = mCamera.GetAspectRatio();

#if 0
	const double h = 0.4;
	const double w = 16.0 / 9 * h / aspect;
	tVector origin = tVector::Zero();
	origin[0] = 1 - w * 1.05;
	origin[1] = 0.95 - h * 1.05;
	origin[2] = -1;
#else
	const double h = 0.3;
	const double w = 1.94;
	tVector origin = tVector::Zero();
	origin[0] = -0.97;
	origin[1] = -0.95;
	origin[2] = -1;
#endif

	int capacity = FootContactsLogBufferSize;

	cDrawUtil::SetLineWidth(1);
	cDrawUtil::SetColor(tVector(1, 1, 1, 0.5));
	cDrawUtil::DrawRect(origin + 0.5 * tVector(w, h, 0, 0), tVector(w, h, 0, 0));

	cDrawUtil::SetLineWidth(1);
	cDrawUtil::SetPointSize(2);
	cDrawUtil::SetColor(tVector(1, 0, 0, 0.75));

	const double h_step = h / (double)mSimFootContactStates.size();

	if (num_val > 0)
	{
		for (int i = 0; i < mSimFootContactStates.size(); ++i)
		{
			cDrawUtil::SetColor(tVector(0, 1, 1, 0.5f));

			double prev_val = mSimFootContactStates[i][0] ? 1 : 0;

			int rect_start_j = -1;
			if (prev_val > 0)
			{
				rect_start_j = 0;
			}

			for (int j = 1; j < num_val; ++j)
			{
				double curr_val = mSimFootContactStates[i][j] ? 1 : 0;

				if (rect_start_j >= 0 && ((curr_val != prev_val) || (j == num_val - 1)))
				{
					tVector rect_origin = tVector::Zero();
					tVector rect_size = tVector::Zero();

					rect_size[0] = w * (j - rect_start_j) / (capacity - 1.0);
					rect_size[1] = h_step;

					rect_origin[0] = w * rect_start_j / (capacity - 1.0) + rect_size[0] * 0.5;
					rect_origin[1] = h_step * i + h_step - rect_size[1] * 0.5;
					rect_origin += origin;

					cDrawUtil::DrawRect(rect_origin, rect_size);

					rect_start_j = -1;
				}
				else if (rect_start_j < 0 && curr_val != prev_val)
				{
					rect_start_j = j;
				}

				prev_val = curr_val;
			}

			if (i > 0)
			{
				cDrawUtil::SetColor(tVector(0, 0, 0, 0.5));
				cDrawUtil::DrawLine(origin + tVector(0, h_step * i, 0, 0), origin + tVector(w, h_step * i, 0, 0));
			}
		}
	}

	cDrawUtil::SetLineWidth(1);
	cDrawUtil::SetColor(tVector(0, 0, 0, 1));
	cDrawUtil::DrawRect(origin + 0.5 * tVector(w, h, 0, 0), tVector(w, h, 0, 0), cDrawUtil::eDrawWireSimple);
}
