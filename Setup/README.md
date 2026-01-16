# Main Workflow

1. Choose any competition (Slug) from `good_kaggle_comps.json` file for e.g. `playground-series-s3e14`.

2. Download [Competition.csv](https://www.kaggle.com/datasets/kaggle/meta-kaggle?select=Competitions.csv) file and fetch competition Overview from it for that chosen slug. [see attached `screenshot_1.png`]

3. Run `solution_sh_creation.py` from `kaggle_solution_sh` folder to get the top solution write-ups.
    - For e.g. 
    ```
    python3 kaggle_solution_sh/solution_sh_creation.py playground-series-s3e14 --return-solutions-only
    ```
    - This will create a file like, `output_solution_playground-series-s3e14.md` - this file will contain all top Kaggle solutions, which we will use in our prompt in Cursor.

4. **Create the train/test dataset**: Download the data from the Kaggle competition and put `train.csv` in `data` folder and `test.csv` and `test_ground_truth.csv` in the tests folder.
    - Since, there will be no groundtruth available for test data, might need to create `test.csv` out of original `train.csv` - for this, use simple scripts like `test_file_creation_from_train_split.py` (attached)
    - If you're using any other dataset, do this step accordingly.

5. In Cursor, use the following prompt with proper inputs:
    - Change the `FOLDER_NAME` to your respective task folder name.
    - Add the competition overview from the step 2 inplace of `COMPETITION_OVERVIEW_FROM_STEP_2` below in the prompt.
    - Add the solution writeup `.md` file (from step 3) inplace of `OUTPUT_SOLUTION_FILE.md` below in the prompt.
    - **Make sure all the @ mentioned files in the below prompt are properly tagged in Cursor - only then content of those files will be accessible to the model.**
    - Feel free to update the prompt, if required:

The Cursor Prompt:
```
We need to create a new task like @tasks/taxi_fare_predictor_v1 in @tasks folder in `FOLDER_NAME` folder.

There are 4 main files.
1. @tasks/taxi_fare_predictor_v1/task.yaml - It will have all the relevant info and where the data is stored, metric, etc.
2. @tasks/taxi_fare_predictor_v1/solution.sh - it will have a good ML solution. It will have model training and testing code. No need to do K-fold validation here. Simply train on full data, after feature engineering and hyper parameter tuning (if applicable), etc.
3. @tasks/taxi_fare_predictor_v1/grader.py - It will measure @tasks/taxi_fare_predictor_v1/solution.sh 's output and will assign a score.
4 And @tasks/taxi_fare_predictor_v1/Dockerfile - the docker file.

The above are sample files as well. Like these files, you need to create files in the `FOLDER_NAME` folder.
I have added train and test data `FOLDER_NAME` folder in `data` and `tests` folder respectively.

Now we need to create the above 4 files for this following machine learning task:
COMPETITION_OVERVIEW_FROM_STEP_2

And the top solution posts for the above task can be found in @kaggle_solution_sh/OUTPUT_SOLUTION_FILE.md file. Choose the threashold value for `grader.py`  based on the scores achieved by these top solution write ups.

See @TASK_REVIEWING_SKILL.md to know more about each of the above four files, file structure and things to keep in mind, while creating those files.

So, based on this, can you create `task.yaml`, `grader.py`, `solution.sh` and `Dockerfile` for the `FOLDER_NAME` task in the respective folder?
```

And then Cursor will create all the necessary file, and then simply run the solution in local once (using the following command), if everything is okay, push the task to Apex.
```
apex-arena test-solution <FOLDER_NAME>
```

**You might need to tweak the threashold in `grader.py` to increase / decrease the pass / fail threasholds.**
Basically, your `solution.sh` should pass the `grader.py` pass / fail criterias.

6. And Push the task to Apex using (below is spec ID for Lunara):
```
apex-arena tasks push tasks/<FOLDER_NAME> --spec f154b153-b82c-4b69-956d-dd4c16ef5da9
```

7. And then, run 3-5 hosted evaluations / rollouts using `starburst-plus`, `nova-ultra` and `biggie` models on Apex UI.

8. After you run these evals / rollouts on Apex, you will see how models are performing. Sometimes, model will easily get 1 score, meaning the current thereshold is very easy for the models to pass.
    - So, in those cases, you need to update the `grader.py` threashold to make the task
    more difficult to pass. 
    - And hence, you also need to update `solution.sh` to meet those new difficult passing criterias. **Here, your machine learning / Kaggle skills will get used, to increase the performance of your solution.sh code.**
    - **Hack**: You can also see model's approach to get the higer score in evaluations, and simply use that approach in your `solution.sh`!!

9. After 1-2 retries, you will make the task difficult enough to get the eval / rollout scores in the range of 0 to 0.5 (ideally, 0.1 to 0.4)

10. And you're done with the task!!