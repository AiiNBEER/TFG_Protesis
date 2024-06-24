@echo off

echo Running first Python script...
python raw_sql_creation.py

echo Finished running first Python script.
echo Running second Python script...
python window_sql_creator.py

echo Finished running second Python script.
pause
