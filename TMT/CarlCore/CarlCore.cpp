#include "CarlCore.h"
#include "scenes/CarlSceneKinChar.h"
#include "scenes/CarlSceneTaskImitate.h"
#include "scenes/CarlSceneTaskSpeed.h"
#include "scenes/CarlSceneTaskHeading.h"
#include "scenes/TMTSceneTemplateController.h"
#include "scenes/TMTSceneTransitionRecorder.h"
#include "scenes/DrawCarlSceneKinChar.h"
#include "scenes/DrawCarlSceneTaskImitate.h"
#include "scenes/DrawCarlSceneTaskSpeed.h"
#include "scenes/DrawCarlSceneTaskHeading.h"
#include "scenes/DrawTMTSceneTemplateController.h"
#include "scenes/DrawTMTSceneTransitionRecorder.h"
#include "scenes/SceneBuilder.h"
#include "render/DrawUtilExtend.h"


cCarlCore::cCarlCore(bool enable_draw) : cDeepMimicCore(enable_draw)
{

}

cCarlCore::~cCarlCore()
{

}

void cCarlCore::Init()
{
	if (EnableDraw())
	{
		cDrawUtil::InitDrawUtil();
		cDrawUtilExtend::InitDrawUtilExtend();
		InitFrameBuffer();
	}
	SetupScene();
}

void cCarlCore::Reshape(int w, int h)
{
	cDeepMimicCore::Reshape(w, h);
	cDrawUtilExtend::Reshape(w, h);
}

bool cCarlCore::IsCarlRLScene() const
{
	const auto& carl_rl_scene = GetCarlRLScene();
	return carl_rl_scene != nullptr;
}

double cCarlCore::RecordPhase(int agent_id) const
{
	const auto& carl_rl_scene = GetCarlRLScene();
	if (carl_rl_scene != nullptr)
	{
		double phase = 0;
		carl_rl_scene->RecordPhase(agent_id, phase);

		return phase;
	}
	return 0;
}

int cCarlCore::RecordMotionLabel(int agent_id) const
{
	const auto& carl_rl_scene = GetCarlRLScene();
	if (carl_rl_scene != nullptr)
	{
		int motion_label = 0;
		carl_rl_scene->RecordMotionLabel(agent_id, motion_label);

		return motion_label;
	}
	return 0;
}

std::vector<double> cCarlCore::RecordGoalTarget(int agent_id) const
{
	const auto& carl_rl_scene = GetCarlRLScene();
	if (carl_rl_scene != nullptr)
	{
		Eigen::VectorXd goal;
		carl_rl_scene->RecordGoalTarget(agent_id, goal);

		std::vector<double> out_goal;
		ConvertVector(goal, out_goal);
		return out_goal;
	}
	return std::vector<double>(0);
}

void cCarlCore::LogGatingWeights(int agent_id, const std::vector<double>& weights)
{

}

void cCarlCore::LogPrimitivesMeanStd(int agent_id, int num_primitives, const std::vector<double>& means, const std::vector<double>& stds)
{

}

bool cCarlCore::SwitchMotion(int agent_id, int target_motion, double target_phase)
{
	const auto& carl_rl_scene = GetCarlRLScene();
	if (carl_rl_scene != nullptr)
	{
		return carl_rl_scene->SwitchMotion(agent_id, target_motion, target_phase);
	}
	return false;
}

void cCarlCore::SetupScene()
{
	ClearScene();

	std::string scene_name = "";
	mArgParser->ParseString("scene", scene_name);
	if (scene_name == "")
	{
		printf("No scene specified\n");
		assert(false);
	}

	mScene = nullptr;
	mRLScene = nullptr;
	mCarlRLScene = nullptr;
	if (EnableDraw())
	{
		if (scene_name == "kin_char")
		{
			mScene = std::shared_ptr<cDrawCarlSceneKinChar>(new cDrawCarlSceneKinChar());
		}
		else if (scene_name == "task_imitate")
		{
			mScene = std::shared_ptr<cDrawCarlSceneTaskImitate>(new cDrawCarlSceneTaskImitate());
		}
		else if (scene_name == "task_speed")
		{
			mScene = std::shared_ptr<cDrawCarlSceneTaskSpeed>(new cDrawCarlSceneTaskSpeed());
		}
		else if (scene_name == "task_heading")
		{
			mScene = std::shared_ptr<cDrawCarlSceneTaskHeading>(new cDrawCarlSceneTaskHeading());
		}
		else if (scene_name == "template_controller")
		{
			mScene = std::shared_ptr<cDrawTMTSceneTemplateController>(new cDrawTMTSceneTemplateController());
		}
		else if (scene_name == "transition_recorder")
		{
			mScene = std::shared_ptr<cDrawTMTSceneTransitionRecorder>(new cDrawTMTSceneTransitionRecorder());
		}
		else
		{
			cSceneBuilder::BuildDrawScene(scene_name, mScene);
		}
	}
	else
	{
		if (scene_name == "kin_char")
		{
			mScene = std::shared_ptr<cCarlSceneKinChar>(new cCarlSceneKinChar());
		}
		else if (scene_name == "task_imitate")
		{
			mScene = std::shared_ptr<cCarlSceneTaskImitate>(new cCarlSceneTaskImitate());
		}
		else if (scene_name == "task_speed")
		{
			mScene = std::shared_ptr<cCarlSceneTaskSpeed>(new cCarlSceneTaskSpeed());
		}
		else if (scene_name == "task_heading")
		{
			mScene = std::shared_ptr<cCarlSceneTaskHeading>(new cCarlSceneTaskHeading());
		}
		else if (scene_name == "template_controller")
		{
			mScene = std::shared_ptr<cTMTSceneTemplateController>(new cTMTSceneTemplateController());
		}
		else if (scene_name == "transition_recorder")
		{
			mScene = std::shared_ptr<cTMTSceneTransitionRecorder>(new cTMTSceneTransitionRecorder());
		}
		else
		{
			cSceneBuilder::BuildScene(scene_name, mScene);
		}
	}

	if (mScene != nullptr)
	{
		mRLScene = std::dynamic_pointer_cast<cRLScene>(mScene);
		mCarlRLScene = std::dynamic_pointer_cast<cCarlRLScene>(mScene);
		mScene->ParseArgs(mArgParser);
		mScene->Init();
		printf("Loaded scene: %s\n", mScene->GetName().c_str());
	}
}

const std::shared_ptr<cCarlRLScene>& cCarlCore::GetCarlRLScene() const
{
	return mCarlRLScene;
}
