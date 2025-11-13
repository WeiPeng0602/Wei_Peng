import pm4py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

event_log = pm4py.read_xes("D:\桌面\BPI Challenge 2017_1_all\BPI Challenge 2017.xes")

#How does a case typically look like?
event_log.info()

#How do long cases look like? How do short ones look like?
case_event_count = event_log.groupby('case:concept:name')['concept:name'].count().reset_index(name='event_count')
long_cases = case_event_count[case_event_count['event_count'] > case_event_count['event_count'].median()]['case:concept:name'].tolist()
short_cases = case_event_count[case_event_count['event_count'] <= case_event_count['event_count'].median()]['case:concept:name'].tolist()
print(f"\nlong_case: {long_cases[0]}")
print(f"short_case: {short_cases[0]}")

#Number of cases
num_cases = event_log['case:concept:name'].nunique()
print(f"\nNumber_of_cases: {num_cases}")

#Number of process variants
process_variants = pm4py.get_variants(event_log)
num_variants = len(process_variants)
print(f"\nNumber_of_process_variants: {num_variants}")

#Number of events / activities / resources
num_events = len(event_log)
print(f"\nNumber_of_events: {num_events}")
num_activities = event_log['concept:name'].nunique()
print(f"Number_of_activities: {num_activities}")
num_resources = event_log['org:resource'].nunique()
print(f"Number_of_resources: {num_resources}")

#Distributions of attributes
case_accepted_dist = event_log.groupby('Accepted')['case:concept:name'].nunique().reset_index(name='case_count')
resource_dist = event_log.groupby('org:resource')['case:concept:name'].nunique().reset_index(name='case_count')

print("\nDistributions_of_attributes: ")
print(case_accepted_dist)

#Average cycle times of cases
from pm4py.statistics.traces.generic.log import case_statistics

case_durations = case_statistics.get_all_case_durations(event_log, parameters={case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
case_ids = event_log['case:concept:name'].unique().tolist()
if len(case_ids) == len(case_durations):
    case_durations_df = pd.DataFrame(
        list(zip(case_ids, case_durations)),
        columns=['case_id', 'duration']
    )
    avg_case_duration = case_durations_df['duration'].mean() / (24 * 3600)
    std_case_duration = case_durations_df['duration'].std() / (24 * 3600)
    print(f"\nAverage_cycle_times_of_cases: {avg_case_duration:.2f} days(std. dev.: {std_case_duration:.2f} days)")
else:
    print("\nWarning: Mismatch between case IDs and durations (case cycle time calculation skipped)")

#Average cycle times of activities
start_events = event_log[event_log['lifecycle:transition'] == 'start'][['case:concept:name', 'concept:name', 'time:timestamp']]
complete_events = event_log[event_log['lifecycle:transition'] == 'complete'][['case:concept:name', 'concept:name', 'time:timestamp']]
start_events = start_events.rename(columns={'time:timestamp': 'start_time'})
complete_events = complete_events.rename(columns={'time:timestamp': 'complete_time'})

activity_times = pd.merge(
    start_events, 
    complete_events, 
    on=['case:concept:name', 'concept:name'],
    how='inner'
)
activity_times['duration'] = (activity_times['complete_time'] - activity_times['start_time']).dt.total_seconds()
avg_activity_duration = activity_times.groupby('concept:name')['duration'].mean().reset_index(name='avg_duration')
avg_activity_duration['avg_duration'] = avg_activity_duration['avg_duration'] / 3600
print("\nAverage cycle times of activities (top 5 longest): ")
print(avg_activity_duration.sort_values(by='avg_duration', ascending=False).head(5))

case_event_count = event_log.groupby('case:concept:name')['concept:name'].count().reset_index(name='event_count')
median_count = case_event_count['event_count'].median()

#Case Length Distribution
case_event_count = event_log.groupby('case:concept:name')['concept:name'].count().reset_index(name='event_count')
plt.hist(case_event_count['event_count'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Number of Events per Case')
plt.ylabel('Number of Cases')
plt.title('Case Length Distribution')
plt.grid(True, alpha=0.3)
plt.show()

#Case Acceptance Distribution
case_accepted_dist = event_log.groupby('Accepted')['case:concept:name'].nunique().reset_index(name='case_count')
plt.pie(case_accepted_dist['case_count'], labels=case_accepted_dist['Accepted'], autopct='%1.1f%%', 
        colors=['lightcoral', 'lightgreen'])
plt.title('Case Acceptance Distribution')
plt.show()

#Resource Workload Distribution
resource_dist = event_log.groupby('org:resource')['case:concept:name'].nunique().reset_index(name='case_count')
resource_dist = resource_dist.sort_values('case_count', ascending=False).head(20)
plt.barh(resource_dist['org:resource'], resource_dist['case_count'], color='lightsteelblue')
plt.xlabel('Number of Cases Handled')
plt.ylabel('Resources')
plt.title('Resource Workload Distribution (Top 20)')
plt.gca().invert_yaxis()
plt.show()

#Activity Frequency Distribution
activity_freq = event_log['concept:name'].value_counts().head(15)
plt.barh(activity_freq.index, activity_freq.values, color='wheat')
plt.xlabel('Frequency')
plt.ylabel('Activity Names')
plt.title('Most Frequent Activities (Top 15)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

#Process tree
process_model = pm4py.discover_process_tree_inductive(event_log)
bpmn_model = pm4py.convert_to_bpmn(process_model)
pm4py.view_bpmn(bpmn_model)