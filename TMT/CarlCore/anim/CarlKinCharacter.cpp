#include "CarlKinCharacter.h"
#include "CarlKinTree.h"


bool cCarlKinCharacter::LoadSkeleton(const Json::Value& root)
{
	return cCarlKinTree::Load(root, mJointMat);
}

int cCarlKinCharacter::GetStateSize() const
{
	int pose_size = GetNumDof() - 5; // remove root.pos.x, root.pos.z, root.rot.x, root.rot.y and root.rot.z
	int vel_size = GetNumDof();
	return pose_size + vel_size;
}

void cCarlKinCharacter::RecordState(double time, Eigen::VectorXd& out_state) const
{
	int state_size = GetStateSize();
	out_state = std::numeric_limits<double>::quiet_NaN() * Eigen::VectorXd::Ones(state_size);
	Eigen::VectorXd pose;
	Eigen::VectorXd vel;
	CalcPose(time, pose);
	CalcVel(time, vel);
	int pose_offset = 0;
	int pose_size = GetNumDof() - 5; // remove root.pos.x, root.pos.z, root.rot.x, root.rot.y and root.rot.z
	int vel_offset = pose_offset + pose_size;
	int vel_size = GetNumDof();

	out_state[0] = pose[1]; // root.pos.y
	out_state.segment(1, pose_size - 1) = pose.segment(6, pose.rows() - 6);
	out_state.segment(vel_offset, vel_size) = vel;
}
