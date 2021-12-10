#pragma once

#include "scenes/RLScene.h"

class cCarlRLScene : virtual public cRLScene
{
public:

    virtual void RecordPhase(int agent_id, double& out_phase) const = 0;
    virtual void RecordMotionLabel(int agent_id, int& out_motion_label) const = 0;
    virtual void RecordGoalTarget(int agent_id, Eigen::VectorXd& out_goal) const = 0;
    virtual void LogGatingWeights(int agent_id, const std::vector<double>& weights);
	virtual void LogPrimitivesMeanStd(int agent_id, int num_primitives, const std::vector<double>& means, const std::vector<double>& stds);
    virtual bool SwitchMotion(int agent_id, int target_motion, double target_phase) = 0;

protected:

};
