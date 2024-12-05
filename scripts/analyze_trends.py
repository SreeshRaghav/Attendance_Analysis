import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the attendance log
LOG_PATH = 'logs/attendance.csv'
OUTPUT_FOLDER = 'outputs/'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Load attendance data
attendance = pd.read_csv(LOG_PATH)
attendance['Time'] = pd.to_datetime(attendance['Time'])

# Add a date column for grouping
attendance['Date'] = attendance['Time'].dt.date

# 1. Daily Attendance Counts
daily_counts = attendance.groupby(['Date', 'Name']).size().reset_index(name='Count')
daily_counts.to_csv(os.path.join(OUTPUT_FOLDER, 'daily_attendance.csv'), index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=daily_counts, x='Date', y='Count', hue='Name')
plt.title('Daily Attendance Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'daily_attendance_count.png'))
plt.close()

# 2. Absentee Analysis
all_dates = attendance['Date'].unique()
all_names = attendance['Name'].unique()

import itertools
expected_attendance = pd.DataFrame(list(itertools.product(all_dates, all_names)), columns=['Date', 'Name'])
merged = expected_attendance.merge(daily_counts, on=['Date', 'Name'], how='left')
merged['Absent'] = merged['Count'].isna()
absentees = merged[merged['Absent']]
absentees.to_csv(os.path.join(OUTPUT_FOLDER, 'absentees.csv'), index=False)

pivot = merged.pivot_table(index='Name', columns='Date', values='Absent', aggfunc='sum', fill_value=0)
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, cmap='Reds', cbar=True, linewidths=0.5)
plt.title('Absentee Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'absentee_heatmap.png'))
plt.close()

# 3. Average Check-In Times
attendance['Hour'] = attendance['Time'].dt.hour
average_check_in = attendance.groupby('Name')['Hour'].mean().reset_index(name='Average Check-In Hour')
average_check_in.to_csv(os.path.join(OUTPUT_FOLDER, 'average_check_in.csv'), index=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=average_check_in, x='Name', y='Average Check-In Hour')
plt.title('Average Check-In Times')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'average_check_in_times.png'))
plt.close()

# 4. Attendance Rate
attendance_rate = merged.groupby('Name')['Absent'].apply(lambda x: 1 - x.mean()).reset_index(name='Attendance Rate')
attendance_rate['Attendance Rate'] *= 100
attendance_rate.to_csv(os.path.join(OUTPUT_FOLDER, 'attendance_rate.csv'), index=False)

plt.figure(figsize=(8, 8))
plt.pie(attendance_rate['Attendance Rate'], labels=attendance_rate['Name'], autopct='%1.1f%%', startangle=140)
plt.title('Overall Attendance Rate')
plt.savefig(os.path.join(OUTPUT_FOLDER, 'attendance_rate_pie.png'))
plt.close()

print(f"Analysis complete! Reports saved in '{OUTPUT_FOLDER}'.")
