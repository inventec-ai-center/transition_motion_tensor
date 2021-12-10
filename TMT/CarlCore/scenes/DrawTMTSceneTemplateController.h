#pragma once

#include "scenes/DrawCarlRLScene.h"
#include "scenes/DrawSceneSimChar.h"
#include "scenes/TMTSceneTemplateController.h"
#include "util/CircularBuffer.h"

class cDrawTMTSceneTemplateController : virtual public cDrawCarlRLScene, virtual public cDrawSceneSimChar
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	cDrawTMTSceneTemplateController();
	virtual ~cDrawTMTSceneTemplateController();

	virtual void Init();
	virtual void Clear();
	virtual bool IsEpisodeEnd() const;
	virtual bool CheckValidEpisode() const;

	virtual void Update(double time_elapsed);
	virtual void Keyboard(unsigned char key, double device_x, double device_y);

	virtual std::string GetName() const;

protected:

	bool mEnableDrawPhaseInfo;
	cCircularBuffer<std::vector<double>> mPhaseLog;

	bool mEnableDrawContactsInfo;
	std::vector<std::deque<bool>> mSimFootContactStates;

	virtual cRLScene* GetRLScene() const;
	virtual cCarlRLScene* GetCarlRLScene() const;
	virtual const std::shared_ptr<cCarlKinCharacter>& GetKinChar() const;

	virtual void CtrlHeading(double delta);  // Horizontal Angle (x, z)

	virtual void DrawInfo() const;
	virtual void DrawInfoText() const;
	virtual void DrawPhaseInfo() const;
	virtual void DrawFootContactsInfo() const;
	virtual void DrawMisc() const;
	virtual void BuildScene(std::shared_ptr<cSceneSimChar>& out_scene) const;
	virtual void DrawCharacters() const;
	virtual void DrawKinCharacter() const;
	virtual void DrawHeading() const;
	virtual void DrawFootContactPoints() const;

	virtual void ResetScene();
	virtual tVector GetCamTrackPos() const;

	virtual void InitFootContacts();
	virtual void UpdateFootContacts();
};
