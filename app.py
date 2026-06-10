import os
import pandas as pd
from datetime import datetime
import streamlit as st

# Get the current date
date = datetime.now().strftime("%d-%m-%Y")

# Attendance folder setup
attendance_folder = "Attendance"
attendance_file = os.path.join(attendance_folder, f"Attendance_{date}.csv")

# Ensure the Attendance folder exists
os.makedirs(attendance_folder, exist_ok=True)

# Check if the file exists
if not os.path.isfile(attendance_file):
    # If the file doesn't exist, create it with column headers
    df = pd.DataFrame(columns=['NAME', 'TIME'])
    df.to_csv(attendance_file, index=False)
    st.write(f"Created new attendance file: {attendance_file}")
else:
    # If the file exists, load the data
    df = pd.read_csv(attendance_file)
    if df.empty:
        st.write("The attendance file is empty.")
    else:
        st.write(f"Loaded existing attendance file: {attendance_file}")

# Show the contents of the DataFrame
st.write(df)
