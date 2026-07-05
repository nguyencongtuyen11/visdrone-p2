@echo off
title RL-SAHI Training Monitor
echo ============================================================
echo   RL-SAHI TRAINING MONITOR  (dong cua so nay khong dung train)
echo   Log: runs\dqn\train_run.log
echo ============================================================
echo.
powershell -NoProfile -Command "Get-Content 'C:\Users\LO PC\Downloads\do-an-moi\Test\runs\dqn\train_run.log' -Wait -Tail 80"
