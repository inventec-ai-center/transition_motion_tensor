#pragma once

#include "util/MathUtil.h"

class cMotionTrajectory
{
public:

	cMotionTrajectory();
	virtual ~cMotionTrajectory();

	virtual bool Load(std::string filename);
	virtual tVector Sample(double phase) const;
	virtual double GetDuration() const;
	virtual double GetPhaseStep(double timestep) const;
	virtual double CalcLocalPhase(double phase) const;

protected:

	std::vector<double> mTimes;
	std::vector<double> mCumulativeTimes;
	std::vector<tVector> mVelocities;
	std::vector<int> mLocalCyclePeaks;
};