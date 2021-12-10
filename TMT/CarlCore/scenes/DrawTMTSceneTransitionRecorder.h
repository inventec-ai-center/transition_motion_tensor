#pragma once

#include "scenes/DrawTMTSceneTemplateController.h"

class cDrawTMTSceneTransitionRecorder : virtual public cDrawTMTSceneTemplateController
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	cDrawTMTSceneTransitionRecorder();
	virtual ~cDrawTMTSceneTransitionRecorder();

	virtual std::string GetName() const;

protected:

	virtual void BuildScene(std::shared_ptr<cSceneSimChar>& out_scene) const;

};
