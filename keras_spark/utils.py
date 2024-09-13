from threading import Thread
import time
import hashlib
import json

class SparkJobPool:
    #class to faciliate batchwise processing of a larger dataset
    #tsh-> a tsh of 0.5 determines that a new batch is started whenever the preceding job has reached 50% completion
    #this should help to control  cluster utilization and avoid OOM erros with large models - moreover
    #intermediate output is stored and so processing doesn't need to start from scratch after a crash
    def __init__(self, sc, job_func, tasks, tsh=0.5):
        self.sc = sc
        self.job_func = job_func
        self.tasks = tasks
        self.tsh = float(tsh)
        self.running_jobs = []
        self.has_spawned = {}
        self.update_interval = 3

    def monitor_job(self, job_id):

        get_status = lambda job_id: self.sc.statusTracker().getJobInfo(job_id).status

        if job_id in self.sc.statusTracker().getActiveJobsIds():
            stage_ids = list(self.sc.statusTracker().getJobInfo(job_id).stageIds)
            total_tasks = 0
            completed_tasks = 0

            for stage_id in stage_ids:
                stage_info = self.sc.statusTracker().getStageInfo(stage_id)
                if stage_info:
                    total_tasks += stage_info.numTasks
                    completed_tasks += stage_info.numCompletedTasks

            return (completed_tasks / total_tasks)
        else:

            if get_status(job_id) != "SUCCEEDED":
                raise Exception(f"job with id {job_id} has failed")
            else:
                return 1.0

    def start_job(self, param):
        get_active = lambda: set(self.sc.statusTracker().getActiveJobsIds())
        currently_active = get_active()
        Thread(target=self.job_func, args=(param,)).start()
        new_jobs = set()
        while len(new_jobs) == 0:
            new_jobs = get_active() - currently_active
            time.sleep(0.5)
        return list(new_jobs)[0]

    def manage_jobs(self):
        if self.tasks:
            new_task = self.tasks.pop()
            self.running_jobs.append(self.start_job(new_task))

        while self.tasks:
            all_progresses = {}

            for job_id in self.running_jobs:

                job_progress = self.monitor_job(job_id)
                all_progresses[job_id] = job_progress

                if job_progress > self.tsh and not self.has_spawned.get(job_id, False):
                    if self.tasks:
                        new_task = self.tasks.pop()
                        self.running_jobs.append(self.start_job(new_task))
                        self.has_spawned[job_id] = True

                if job_progress == 1.0 and self.has_spawned.get(job_id, False):
                    self.running_jobs.remove(job_id)

            time.sleep(self.update_interval)
            print("Remaining tasks:", self.tasks)
            print("Running jobs:", str(all_progresses))


def hash_keras_model(model):
    """
    Creates an MD5 hash for a given Keras model based on its architecture (or config) and weights.

    Parameters:
    model (tf.keras.Model): The Keras model to hash.

    Returns:
    str: A hexadecimal MD5 hash of the model.
    """
    hash_md5 = hashlib.md5()

    # Try to hash model architecture via to_json()
    try:
        model_json = model.to_json()
        hash_md5.update(model_json.encode('utf-8'))
    except Exception as e:
        print(f"model.to_json() failed: {e}. Using model configuration instead.")

        # Fall back to model.get_config() if to_json() fails
        model_config = json.dumps(model.get_config())
        hash_md5.update(model_config.encode('utf-8'))

    # Hash model weights
    weights = model.get_weights()

    # Convert the weights into a byte string
    weights_json = json.dumps([w.tolist() for w in weights])
    hash_md5.update(weights_json.encode('utf-8'))

    return hash_md5.hexdigest()

