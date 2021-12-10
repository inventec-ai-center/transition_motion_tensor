#include "DrawTMTSceneTransitionRecorder.h"
#include "TMTSceneTransitionRecorder.h"

cDrawTMTSceneTransitionRecorder::cDrawTMTSceneTransitionRecorder() : cDrawTMTSceneTemplateController()
{
}

cDrawTMTSceneTransitionRecorder::~cDrawTMTSceneTransitionRecorder()
{
}

std::string cDrawTMTSceneTransitionRecorder::GetName() const
{
	return "Transition Recorder";
}

void cDrawTMTSceneTransitionRecorder::BuildScene(std::shared_ptr<cSceneSimChar>& out_scene) const
{
	out_scene = std::shared_ptr<cTMTSceneTransitionRecorder>(new cTMTSceneTransitionRecorder());
}
