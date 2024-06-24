import airflow
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import timedelta
from docker.types import Mount
import os
from airflow.decorators import task


# Define path to data
t2_command = "python3 /opt/airflow/dags/move_3d/app/0011_04_train_model.py"
t3_command = "python3 /opt/airflow/dags/move_3d/app/0011_05_test_model.py"
t4_command = "python3 /opt/airflow/dags/move_3d/app/0011_06_test_cvimwrite.py"

default_args = {
    'start_date': airflow.utils.dates.days_ago(0),
    'retries': 1,
    'retry_delay': timedelta(minutes=1)
}

dag = DAG(
    'move_3d',
    default_args=default_args,
    description='move_3d_dag',
    schedule_interval=None,
    dagrun_timeout=timedelta(minutes=20)
)

t2 = BashOperator(
    task_id='train',
    bash_command=t2_command,
    dag=dag,
)

t3 = BashOperator(
    task_id='test',
    bash_command=t3_command,
    dag=dag,
)
t4 = BashOperator(
    task_id='cvimwrite',
    bash_command=t4_command,
    dag=dag,
)
t2 >> t3 >> t4