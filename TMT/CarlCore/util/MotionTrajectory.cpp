#include "MotionTrajectory.h"
#include "util/FileUtil.h"

cMotionTrajectory::cMotionTrajectory()
{
	mTimes.clear();
	mCumulativeTimes.clear();
	mVelocities.clear();
	mLocalCyclePeaks.clear();
}

cMotionTrajectory::~cMotionTrajectory()
{
}

bool cMotionTrajectory::Load(std::string filename)
{
	std::ifstream fin(filename);
	if (fin.is_open())
	{
		// first line stores indices of cycle peaks
		std::string line;
		std::getline(fin, line);

		std::istringstream in(line);
		std::string token;
		mLocalCyclePeaks.clear();
		while (std::getline(in, token, ' '))
		{
			int val = std::stoi(token);
			// printf("%s %d\n", token.c_str(), val);
			if (val >= 0)
			{
				mLocalCyclePeaks.push_back(val);
			}
		}

		double t = 0;
		tVector v = tVector::Zero();
		while (fin >> t >> v[0] >> v[1] >> v[2])
		{
			mTimes.push_back(t);
			mVelocities.push_back(v);
			// printf("%.3f, %.3f, %.3f, %.3f\n", t, v[0], v[1], v[2]);
		}
		fin.close();

		mCumulativeTimes.resize(mTimes.size(), 0);
		for (int i = 0; i < mTimes.size(); ++i)
		{
			if (i > 0)
			{
				mCumulativeTimes[i] = mCumulativeTimes[i-1] + mTimes[i];
			}
			else
			{
				mCumulativeTimes[i] = mTimes[i];
			}
		}

		return true;
	}
	else
	{
		return false;
	}
}

tVector cMotionTrajectory::Sample(double phase) const
{
	tVector vel;
	int idx_lower = std::floor(phase * (mVelocities.size() - 1));
	int idx_upper = std::ceil(phase * (mVelocities.size() - 1));
	if (idx_lower != idx_upper)
	{
		double t = phase * (mVelocities.size() - 1) - idx_lower;
		vel = mVelocities[idx_lower] * (1 - t) + mVelocities[idx_upper] * t;
	}
	else
	{
		vel = mVelocities[idx_lower];
	}
	return vel;
}

double cMotionTrajectory::GetDuration() const
{
	return mCumulativeTimes.back();
}

double cMotionTrajectory::GetPhaseStep(double timestep) const
{
	double phase_step = timestep / GetDuration();
	return phase_step;
}

double cMotionTrajectory::CalcLocalPhase(double phase) const
{
	double local_phase = 0;
	if (mLocalCyclePeaks.size() > 1)
	{
		double pos = phase * mVelocities.size();

		double cycle_length = 0;
		for (int i = 1; i < mLocalCyclePeaks.size() - 1; ++i)
		{
			cycle_length += mLocalCyclePeaks[i + 1] - mLocalCyclePeaks[i];
		}
		cycle_length /= (mLocalCyclePeaks.size() - 2);

		std::vector<double> distances;
		std::vector<int> indices;
		for (int i = 0; i < mLocalCyclePeaks.size(); ++i)
		{
			double dist = pos - mLocalCyclePeaks[i];
			distances.push_back(dist);
			if (dist > 0)
			{
				indices.push_back(i);
			}
		}

		if (indices.size() == 0)
		{
			local_phase = (cycle_length - (mLocalCyclePeaks[0] - pos)) / cycle_length;
		}
		else if (indices.size() == mLocalCyclePeaks.size())
		{
			local_phase = (pos - mLocalCyclePeaks.back()) / cycle_length;
		}
		else
		{
			int idx = indices.back();
			local_phase = distances[idx] / (mLocalCyclePeaks[idx+1] - mLocalCyclePeaks[idx]);
		}
	}
	else
	{
		local_phase = phase;
	}
	return local_phase;
}
