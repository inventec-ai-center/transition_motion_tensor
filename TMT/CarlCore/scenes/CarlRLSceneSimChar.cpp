#include "CarlRLSceneSimChar.h"

void cCarlRLSceneSimChar::RecordPhase(int agent_id, double& out_phase) const
{
    out_phase = 0;
}

void cCarlRLSceneSimChar::RecordMotionLabel(int agent_id, int& out_motion_label) const
{
    out_motion_label = 0;
}

void cCarlRLSceneSimChar::RecordGoalTarget(int agent_id, Eigen::VectorXd& out_goal) const
{

}

void cCarlRLSceneSimChar::LogGatingWeights(int agent_id, const std::vector<double>& weights)
{

}

void cCarlRLSceneSimChar::LogPrimitivesMeanStd(int agent_id, int num_primitives, const std::vector<double>& means, const std::vector<double>& stds)
{

}

double cCarlRLSceneSimChar::CalcReward(int agent_id) const
{
    return 0;
}

bool cCarlRLSceneSimChar::SwitchMotion(int agent_id, int target_motion, double target_phase)
{
    return false;
}
