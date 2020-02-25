@echo off
set img=%~1
if "%img%" == "" (
	set img=bottleRightCamLeft.jpg
)
python click_and_crop.py --image "%img%" --smoothen 1
resukt.jpg